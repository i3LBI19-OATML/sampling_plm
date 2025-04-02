"""
For AR_generator.py

Identical to sampling.py, but the final output is the mutated sequence instead of mutation.
Addition of ARbeam_search function
"""

import torch
from torch.distributions import Categorical
import pandas as pd
import math
import app
from decimal import Decimal
from statistics import mean
import os

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"

class ARtemperature_sampler:
  def __init__(self, temperature: float = 1.0):
    self.temperature = temperature
  def __call__(self, logits: torch.Tensor):
    dist = Categorical(logits=logits / self.temperature)
    return dist.sample()

# Modified version of sampling for DataFrame containing probabilities

# Top-k sampling
def ARtop_k_sampling(scores: pd.DataFrame, k: int, sampler = ARtemperature_sampler(temperature=1.0), multi=False):
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
  return scores['mutated_sequence'][sampled_score]

# Typical sampling
def ARtypical_sampling(scores: pd.DataFrame, mass: float = 0.9, sampler = ARtemperature_sampler(temperature=1.0), multi=False):
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
    return scores['mutated_sequence'][sampled_score]

# Top-p sampling
def ARtop_p_sampling(scores: pd.DataFrame, p: float, sampler = ARtemperature_sampler(temperature=1.0), multi=False):
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
    return scores['mutated_sequence'][sampled_score]

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
    except TypeError:
      b = mean(prob[i])/mean(prob[i+1])
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
def ARmirostat_sampling(scores: pd.DataFrame, tau:float = 3.0, sampler = ARtemperature_sampler(temperature=1.0), vocab=AA_vocab, multi=False):
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
    return scores['mutated_sequence'][sampled_score]

# Random Sampling
def ARrandom_sampling(scores: pd.DataFrame, sampler = ARtemperature_sampler(temperature=1.0), multi=False):
  raw_score = torch.tensor(scores['avg_score'].values, device='cuda:0')
  raw_score = torch.nan_to_num(raw_score, float("-inf"))
  sampled_score = sampler(raw_score).item()
  
  if multi:
    return scores
  else:
    return scores['mutated_sequence'][sampled_score]

def ARbeam_search(scores: pd.DataFrame, beam_width: int, max_length:int, tokenizer, Tmodel, score_mirror=False, batch=20, max_pos=50, sampler=ARtemperature_sampler(temperature=1.0), multi=False, past_key_values=None, extension_factor=1, filter='hpf', IST=96, verbose=0, model_type='Tranception'):
  length = 1
  while length < max_length:
    # Get top k mutations
    beam_width = min(beam_width, len(scores))
    assert beam_width <= len(scores), "Beam width must be less than or equal to the number of mutations ({}).".format(len(scores))
    scores = ARtop_k_sampling(scores, k=beam_width, sampler=sampler, multi=True)
    # Extend each mutation by one
    levels = pd.DataFrame(columns=['mutated_sequence']) # Initialize DataFrame
    for i, row in scores.iterrows():
      extension = app.extend_sequence_by_n(row['mutated_sequence'], extension_factor, AA_vocab, output_sequence=True)
      levels = pd.concat([levels, extension], ignore_index=True)
    
    # assert filter in ['hpf', 'qff'], "Filter must be one of 'hpf' or 'qff'"
    if filter == 'hpf':
      print("Filtering Beam Search with HPF") if verbose == 1 else None
      trimmed = app.trim_DMS(DMS_data=levels, sampled_mutants=scores, mutation_rounds=0)
      levels = trimmed.sample(n=IST)

    # Score each mutation
    scores, _, past_key_values = app.score_multi_mutations(
      sequence=None, extra_mutants=levels, scoring_mirror=score_mirror, batch_size_inference=batch, 
      max_number_positions_per_heatmap=max_pos, num_workers=2, AA_vocab=AA_vocab, tokenizer=tokenizer, 
      AR_mode=True, Tranception_model=Tmodel, past_key_values=past_key_values, verbose=verbose,
      model_type=model_type)
    length += 1
  if length == max_length:
    scores = ARtop_k_sampling(scores, k=1, sampler=sampler, multi=True)
    if multi:
      return scores, past_key_values
    else:
      return scores['mutated_sequence'][0], past_key_values

      
