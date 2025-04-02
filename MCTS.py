'''
Source: https://github.com/brilee/python_uct/blob/master/naive_impl.py
'''
import random
import math
import app
import pandas as pd
import numpy as np
import app
from sampling import top_k_sampling, temperature_sampler

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
final_sampler = temperature_sampler(1.0)

class UCTNode():
  def __init__(self, state, parent=None, prior=0, move=None):
    self.state = state
    self.move = move
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
      self.add_child(move, row['avg_score'], row['mutated_sequence'], row['mutant'])

  def add_child(self, move, prior, child_seq, mutation):
    self.children[move] = UCTNode(
        child_seq, parent=self, prior=prior, move=mutation)

  def backup(self, value_estimate: float):
    current = self
    while current.parent is not None:
      current.number_visits += 1
      current.total_value += value_estimate
      current = current.parent
    print("========END OF ITERATION========")

def UCT_search(state, max_length, extra, tokenizer, Tmodel, AA_vocab=AA_vocab, past_key_values=None, filter='hpf', ev_model=None, intermediate_sampling_threshold=96, model_type='Tranception'):
  root = UCTNode(state)
  for _ in range(max_length):
    leaf = root.select_leaf()
    child_priors, value_estimate, past_key_values = Evaluate(leaf.state, extra, tokenizer, AA_vocab, max_length, Tmodel, past_key_values=past_key_values, filter=filter, ev_model=ev_model, IST=intermediate_sampling_threshold, model_type=model_type)
    leaf.expand(child_priors)
    leaf.backup(value_estimate)
    output = max(root.children.items(), key=lambda item: item[1].number_visits)
  return output[1].move, past_key_values

def Evaluate(seq, extra, tokenizer, AA_vocab, max_length, Tmodel, past_key_values=None, filter='hpf', ev_model=None, IST=96, model_type='Tranception'):
    # df_seq = pd.DataFrame.from_dict({'mutated_sequence': [seq]})
    score_heatmap, suggested_mutation, results, _, past_key_values = app.score_and_create_matrix_all_singles(seq, Tmodel, None, None, scoring_mirror=False, batch_size_inference=20, max_number_positions_per_heatmap=50, num_workers=8, AA_vocab=AA_vocab, tokenizer=tokenizer, with_heatmap=False, past_key_values=past_key_values, model_type=model_type)
    mutation_count = 1
    
    # Get top k mutations
    results = results.sort_values(by=['avg_score'], ascending=False, ignore_index=True).head(max_length)
    mutation_count += 1

    # Extend and filter the results
    assert filter in ['hpf', 'qff', 'ams'], "Filter must be one of 'hpf', 'qff', or 'ams'"
    if filter == 'hpf':
      print("Filtering MCTS with HPF")
      extension = app.apply_gen_1extra(results)
      trimmed = app.trim_DMS(DMS_data=extension, sampled_mutants=results, mutation_rounds=mutation_count)
      extension = trimmed.sample(n=IST)

    if filter == 'qff':
      print("Filtering MCTS with QFF")
      extension = app.apply_gen_1extra(results)
      assert ev_model is not None, "ev_model must be provided for QFF filter"
      extension = app.predict_evmutation(DMS=extension, top_n=IST, ev_model=ev_model)

    if filter == 'ams':
      print("Filtering MCTS with AMS")
      assert ev_model is not None, "ev_model must be provided for AMS filter"
      att_mutations = app.get_attention_mutants(DMS=results, Tranception_model=Tmodel, focus='highest', top_n=5) #top_n is the number of attention positions to focus on
      extension = app.predict_evmutation(DMS=att_mutations, top_n=IST, ev_model=ev_model)


    prior, _, past_key_values = app.score_multi_mutations(sequence=None, Tranception_model=Tmodel, extra_mutants=extension, mutation_range_start=None, mutation_range_end=None, scoring_mirror=False, batch_size_inference=20, max_number_positions_per_heatmap=50, num_workers=8, AA_vocab=AA_vocab, tokenizer=tokenizer, AR_mode=True, past_key_values=past_key_values, model_type=model_type)
    
    child_priors = prior
    value_estimate = float(results['avg_score'].values[0])
    
    return child_priors, value_estimate, past_key_values

