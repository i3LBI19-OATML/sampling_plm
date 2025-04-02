import torch
from torch.distributions import Categorical
import pandas as pd
import math
import app
from decimal import Decimal

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"

class temperature_sampler:
  def __init__(self, temperature: float = 1.0):
    self.temperature = temperature
  def __call__(self, logits: torch.Tensor):
    dist = Categorical(logits=logits / self.temperature)
    return dist.sample()

# Modified version of sampling for DataFrame containing probabilities

# Top-k sampling
def top_k_sampling(scores: pd.DataFrame, k: int, sampler = temperature_sampler(temperature=1.0), multi=False):
  if multi:
    scores = scores.sort_values(by=['avg_score'], ascending=False)
    scores = scores.reset_index(drop=True)
    scores = scores.iloc[:k]
    return scores
  raw_score = torch.tensor(scores['avg_score'].values, device='cuda:0')
  raw_score = torch.nan_to_num(raw_score, float("-inf"))
  zeros = raw_score.new_ones(raw_score.shape) * float('-inf')
  values, indices = torch.topk(raw_score, k=k, dim=-1)
  zeros.scatter_(-1, indices, values)
  
  sampled_score = sampler(zeros).item()
  return scores['mutant'][sampled_score]

# Typical sampling
def typical_sampling(scores: pd.DataFrame, mass: float = 0.9, sampler = temperature_sampler(temperature=1.0), multi=False):
  raw_score = torch.tensor(scores['avg_score'].values)
  raw_score = torch.nan_to_num(raw_score, float("-inf"))
  # calculate entropy
  normalized = torch.nn.functional.log_softmax(raw_score, dim=-1)
  p = torch.exp(normalized)
  ent = -(normalized * p).nansum(-1, keepdim=True)

  # shift and sort
  shifted_scores = torch.abs((-normalized) - ent)
  sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
  sorted_logits = raw_score.gather(-1, sorted_indices)
  cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

  # Remove tokens with cumulative mass above the threshold
  last_ind = (cumulative_probs < mass).sum(dim=-1)
  last_ind[last_ind < 0] = 0
  sorted_indices_to_remove = sorted_scores > sorted_scores.gather(-1, last_ind.view(-1))
  indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)

  raw_score = raw_score.masked_fill(indices_to_remove, float("-inf"))
  sampled_score = sampler(raw_score).item()
  # return res
  if multi:
    p_list = []
    for index, value in enumerate(raw_score.tolist()): 
      if value != float("-inf"):
        # print(value) 
        p_list.append(index)
    # print(p_list)
    return scores.iloc[p_list]
  else:
    return scores['mutant'][sampled_score]

# Top-p sampling
def top_p_sampling(scores: pd.DataFrame, p: float, sampler = temperature_sampler(temperature=1.0), multi=False):
  raw_score = torch.tensor(scores['avg_score'].values)
  raw_score = torch.nan_to_num(raw_score, float("-inf"))
  
  sorted_logits, sorted_indices = torch.sort(raw_score, dim=-1, descending=True)
  cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

  nucleus = cumulative_probs > p
 # Shift the indices to the right to keep also the first token above the threshold
  nucleus[..., 1:] = nucleus[..., :-1].clone()
  nucleus[..., 0] = 0
  indices_to_remove = nucleus.scatter(-1, sorted_indices, nucleus)
  raw_score = raw_score.masked_fill(indices_to_remove, float("-inf"))
  sampled_score = sampler(raw_score).item()

  # return res
  if multi:
    p_list = []
    for index, value in enumerate(raw_score.tolist()): 
      if value != float("-inf"):
        # print(value) 
        p_list.append(index)
    # print(p_list)
    return scores.iloc[p_list]
  else:
    return scores['mutant'][sampled_score]

# Mirostat Helper Functions
def estimate_s(prob):
  result = 0
  num = 0
  den = 0
  n = len(prob) if len(prob) < 100 else 100
  for i in range(0, n-1):
    try:
      b = prob[i]/prob[i+1]
    except ZeroDivisionError:
      b = 0
    t = (i+2)/(i+1)
    num += math.log(b if b>0 else 1)*math.log(t if t>0 else 1)
    den += math.log(t if t>0 else 1)**2
  return num/den


def compute_k(n,s,tau):
  eps = s-1
  k = (Decimal(eps) * Decimal(Decimal(2) ** Decimal(tau))) / (Decimal(1) - Decimal(n) ** Decimal(-eps))
  k = Decimal((Decimal(k) ** (Decimal(1) / Decimal(s))))
  return round(k)

# Mirostat Sampling
def mirostat_sampling(scores: pd.DataFrame, tau:float = 3.0, sampler = temperature_sampler(temperature=1.0), vocab=AA_vocab, multi=False):
  max_surprise = 2*tau
  n = len(vocab)

  raw_score = torch.tensor(scores['avg_score'].values)
  raw_score = torch.nan_to_num(raw_score, float("-inf"))

  sorted_logits, sorted_indices = torch.sort(raw_score, descending=True)
  listed_prob = sorted_logits.tolist()

  # Estimate s
  s = estimate_s(listed_prob)
  # Compute k
  k = compute_k(n,s,max_surprise)+1

  sorted_logits = sorted_logits[0:k]
  sorted_indices = sorted_indices[0:k]
  scores = scores.iloc[0:k, :]
  prob_topk = sorted_logits
  sampled_score = sampler(prob_topk).item()

  if multi:
    return scores
    # return scores['mutant']
  else:
    sampled_score = sampler(prob_topk).item()
    return scores['mutant'][sampled_score]

# Random Sampling
def random_sampling(scores: pd.DataFrame, sampler = temperature_sampler(temperature=1.0), multi=False):
  raw_score = torch.tensor(scores['avg_score'].values, device='cuda:0')
  raw_score = torch.nan_to_num(raw_score, float("-inf"))
  sampled_score = sampler(raw_score).item()
  
  if multi:
    return scores
  else:
    return scores['mutant'][sampled_score]


def beam_search(scores: pd.DataFrame, beam_width: int, max_length:int, tokenizer, Tmodel, score_mirror=False, batch=20, max_pos=50, sampler=temperature_sampler(temperature=1.0), multi=False, past_key_values=None, filter='hpf', ev_model=None, IST=96, model_type='Tranception'):
  length = 1
  while length < max_length:
    # Get top k mutations
    assert beam_width <= len(scores), "Beam width must be less than or equal to the number of mutations ({}).".format(len(scores))
    scores = top_k_sampling(scores, k=beam_width, sampler=sampler, multi=True)
    length += 1

    # Extend and filter the results
    assert filter in ['hpf', 'qff', 'ams'], "Filter must be one of 'hpf', 'qff', or 'ams'"
    if filter == 'hpf':
      levels = app.apply_gen_1extra(scores)
      print("Filtering MCTS with HPF")
      trimmed = app.trim_DMS(DMS_data=levels, sampled_mutants=scores, mutation_rounds=length)
      levels = trimmed.sample(n=IST)

    if filter == 'qff':
      levels = app.apply_gen_1extra(scores)
      print("Filtering MCTS with QFF")
      assert ev_model is not None, "ev_model must be provided for QFF filter"
      levels = app.predict_evmutation(DMS=levels, top_n=IST, ev_model=ev_model)

    if filter == 'ams':
      print("Filtering MCTS with AMS")
      assert ev_model is not None, "ev_model must be provided for AMS filter"
      att_mutations = app.get_attention_mutants(DMS=scores, Tranception_model=Tmodel, focus='highest', top_n=5, model_type=model_type) #top_n is the number of attention positions to focus on
      levels = app.predict_evmutation(DMS=att_mutations, top_n=IST, ev_model=ev_model)

    # Score each mutation
    scores, _, past_key_values = app.score_multi_mutations(sequence=None, extra_mutants=levels, Tranception_model=Tmodel, scoring_mirror=score_mirror, batch_size_inference=batch, max_number_positions_per_heatmap=max_pos, num_workers=8, AA_vocab=AA_vocab, tokenizer=tokenizer, AR_mode=True, past_key_values=past_key_values, model_type=model_type)
  if length == max_length:
    scores = top_k_sampling(scores, k=1, sampler=sampler, multi=True)
    if multi:
      return scores, past_key_values
    else:
      return scores['mutant'][0], past_key_values