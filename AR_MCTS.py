'''
Source: https://github.com/brilee/python_uct/blob/master/naive_impl.py
'''
import random
import math
import app
import pandas as pd
import numpy as np

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"

class UCTNode():
  def __init__(self, state, parent=None, prior=0):
    self.state = state
    self.is_expanded = False
    self.parent = parent  # Optional[UCTNode]
    self.children = {}  # Dict[move, UCTNode]
    self.prior = prior  # float
    self.total_value = 0  # float
    self.number_visits = 0  # int

  def Q(self):  # returns float
    # print("self.total_value: ", self.total_value, "self.number_visits: ", self.number_visits)
    return self.total_value / (1 + self.number_visits)

  def U(self):  # returns float
    # print("self.parent.number_visits: ", self.parent.number_visits, "self.prior: ", self.prior, "self.number_visits: ", self.number_visits)
    return (math.sqrt(self.parent.number_visits)
        * self.prior / (1 + self.number_visits))

  def best_child(self):
    # for child in self.children.values():
      # print("child: ", child.state, "child.Q(): ", child.Q(), "child.U(): ", child.U(), "child.Q() + child.U(): ", (child.Q() + child.U()))
      # print("=====================================")
    return max(self.children.values(),
               key=lambda node: node.Q() + node.U())

  def select_leaf(self):
    current = self
    while current.is_expanded:
      current = current.best_child()
    return current

  def expand(self, child_priors):
    self.is_expanded = True
    for move, row in child_priors.iterrows():
      self.add_child(move, row['avg_score'], row['mutated_sequence'])

  def add_child(self, move, prior, child_seq):
    self.children[move] = UCTNode(
        child_seq, parent=self, prior=prior)

  def backup(self, value_estimate: float):
    current = self
    while current.parent is not None:
      current.number_visits += 1
      current.total_value += value_estimate
      current = current.parent
    # print("========END OF ITERATION========")

def UCT_search(state, max_length, tokenizer, Tmodel, AA_vocab=AA_vocab, extension_factor=1, past_key_values=None, filter='hpf', intermediate_sampling_threshold=96, batch=20, model_type='Tranception'):
  root = UCTNode(state)
  for _ in range(max_length):
    leaf = root.select_leaf()
    child_priors, value_estimate, past_key_values = Evaluate(leaf.state, tokenizer, AA_vocab, Tmodel, extension_factor, past_key_values=past_key_values, filter=filter, IST=intermediate_sampling_threshold, batch=batch, model_type=model_type)
    leaf.expand(child_priors)
    leaf.backup(value_estimate)
    output = max(root.children.items(), key=lambda item: item[1].number_visits)
  return output[1].state, past_key_values

def Evaluate(seq, tokenizer, AA_vocab, Tmodel, extension_factor=1, past_key_values=None, filter='hpf', IST=96, batch=20, model_type='Tranception'):
    df_seq = pd.DataFrame.from_dict({'mutated_sequence': [seq]})
    # print(f"Tmodel: {Tmodel}")
    results, _, past_key_values = app.score_multi_mutations(sequence=None, extra_mutants=df_seq, mutation_range_start=None, mutation_range_end=None, scoring_mirror=False, batch_size_inference=batch, max_number_positions_per_heatmap=50, num_workers=8, AA_vocab=AA_vocab, tokenizer=tokenizer, AR_mode=True, Tranception_model=Tmodel, past_key_values=past_key_values, model_type=model_type)

    # get top n mutants
    results = results.sort_values(by=['avg_score'], ascending=False, ignore_index=True).head(100)

    extension = app.extend_sequence_by_n(seq, extension_factor, AA_vocab, output_sequence=True)

    if filter == 'hpf':
      # print("Filtering MCTS with HPF")
      trimmed = app.trim_DMS(DMS_data=extension, sampled_mutants=results, mutation_rounds=0)
      IST = min(IST, len(trimmed)) # Required
      extension = trimmed.sample(n=IST)
    
    # extension = app.extend_sequence_by_n(seq, extension_factor, AA_vocab, output_sequence=True)
    prior, _, past_key_values = app.score_multi_mutations(sequence=None, extra_mutants=extension, mutation_range_start=None, mutation_range_end=None, scoring_mirror=False, batch_size_inference=batch, max_number_positions_per_heatmap=50, num_workers=8, AA_vocab=AA_vocab, tokenizer=tokenizer, AR_mode=True, Tranception_model=Tmodel, past_key_values=past_key_values, model_type=model_type)
    
    child_priors = prior
    value_estimate = float(results['avg_score'].values[0])
    
    return child_priors, value_estimate, past_key_values

