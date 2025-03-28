import torch
import torch.nn as nn
import transformers
from transformers import PreTrainedTokenizerFast
import tranception
import datasets
from tranception import config, model_pytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow import keras
import itertools
import re
import tqdm
import time
from scoring_metrics.util import identify_mutation, extract_mutations
from EVmutation.model import CouplingsModel
from EVmutation.tools import predict_mutation_table
from sampling import top_k_sampling
from RITA import compute_fitness

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tranception/utils/tokenizers/Basic_tokenizer"),
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )

def create_all_single_mutants(sequence, AA_vocab=AA_vocab, mutation_range_start=None, mutation_range_end=None):
    sequence_list = list(sequence)
    if mutation_range_start is None: mutation_range_start = 1
    if mutation_range_end is None: mutation_range_end = len(sequence)
    positions = range(mutation_range_start-1, mutation_range_end)
    aas = AA_vocab
    combos = itertools.product(positions, aas)
    all_single_mutants = [{'mutant': f"{sequence_list[i]}{i+1}{aa}", 'mutated_sequence': ''.join([aa if j == i else sequence_list[j] for j in range(len(sequence_list))])} for i, aa in combos if sequence_list[i] != aa]
    return pd.DataFrame(all_single_mutants)

def extend_sequence_by_n(sequence, n: int, reference_vocab, output_sequence=True):
  permut = ["".join(i) for i in itertools.permutations(reference_vocab, n)]
  extend_sequence = [sequence + ext for ext in permut]
  if output_sequence:
    df_es = pd.DataFrame.from_dict({"mutated_sequence": extend_sequence})
    return df_es
  else:
    df_ext = pd.DataFrame.from_dict({"extension": permut})
    return df_ext

def trim_DMS(DMS_data:pd.DataFrame, sampled_mutants:pd.DataFrame, mutation_rounds:int):
  if mutation_rounds == 0:
    # get sequences in DMS that contains a substring of the sampled mutants
    trimmed_variants = DMS_data[DMS_data["mutated_sequence"].str.contains('|'.join(sampled_mutants['mutated_sequence']))].reset_index(drop=True)
    trimmed_variants = trimmed_variants.drop_duplicates(subset=['mutated_sequence']).reset_index(drop=True)
    return trimmed_variants[['mutated_sequence']]
  else:
    for mutation in range(2):
      if mutation == 0:
        DMS_data[f'past_mutation'] = DMS_data["mutant"].map(lambda x: ":".join(x.split(":", mutation_rounds-1)[:mutation_rounds-1]))
      else:
        DMS_data[f'current_mutation'] = DMS_data["mutant"].map(lambda x: ":".join(x.split(":", mutation_rounds-1)[mutation_rounds-1:]))
    trimmed_variants = DMS_data[DMS_data[f'past_mutation'].isin(sampled_mutants['mutant'])].reset_index(drop=True)
    # print(f'Trimmed DMS: {len(trimmed_variants)}')
    trimmed_variants = trimmed_variants.drop_duplicates(subset=['mutant']).reset_index(drop=True)
    return trimmed_variants[['mutant','mutated_sequence']]

def create_scoring_matrix_visual(scores,sequence,image_index=0,mutation_range_start=None,mutation_range_end=None,AA_vocab=AA_vocab,annotate=True,fontsize=20):
  filtered_scores=scores.copy()
  filtered_scores=filtered_scores[filtered_scores.position.isin(range(mutation_range_start,mutation_range_end+1))]
  piv=filtered_scores.pivot(index='position',columns='target_AA',values='avg_score').round(4)
  mutation_range_len = mutation_range_end - mutation_range_start + 1
  fig, ax = plt.subplots(figsize=(50,mutation_range_len))
  scores_dict = {}
  valid_mutant_set=set(filtered_scores.mutant)  
  ax.tick_params(bottom=True, top=True, left=True, right=True)
  ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True)
  if annotate:
    for position in range(mutation_range_start,mutation_range_end+1):
      for target_AA in list(AA_vocab):
        mutant = sequence[position-1]+str(position)+target_AA
        if mutant in valid_mutant_set:
          scores_dict[mutant]= float(filtered_scores.loc[filtered_scores.mutant==mutant,'avg_score'])
        else:
          scores_dict[mutant]=0.0
    labels = (np.asarray(["{} \n {:.4f}".format(symb,value) for symb, value in scores_dict.items() ])).reshape(mutation_range_len,len(AA_vocab))
    heat = sns.heatmap(piv,annot=labels,fmt="",cmap='RdYlGn',linewidths=0.30,ax=ax,vmin=np.percentile(scores.avg_score,2),vmax=np.percentile(scores.avg_score,98),\
                cbar_kws={'label': 'Log likelihood ratio (mutant / starting sequence)'},annot_kws={"size": fontsize})
  else:
    heat = sns.heatmap(piv,cmap='RdYlGn',linewidths=0.30,ax=ax,vmin=np.percentile(scores.avg_score,2),vmax=np.percentile(scores.avg_score,98),\
                cbar_kws={'label': 'Log likelihood ratio (mutant / starting sequence)'},annot_kws={"size": fontsize})
  heat.figure.axes[-1].yaxis.label.set_size(fontsize=int(fontsize*1.5))
  heat.figure.axes[-1].yaxis.set_ticklabels(heat.figure.axes[-1].yaxis.get_ticklabels(), fontsize=fontsize)
  heat.set_title("Higher predicted scores (green) imply higher protein fitness",fontsize=fontsize*2, pad=40)
  heat.set_ylabel("Sequence position", fontsize = fontsize*2)
  heat.set_xlabel("Amino Acid mutation", fontsize = fontsize*2)
  yticklabels = [str(pos)+' ('+sequence[pos-1]+')' for pos in range(mutation_range_start,mutation_range_end+1)]
  heat.set_yticklabels(yticklabels)
  heat.set_xticklabels(heat.get_xmajorticklabels(), fontsize = fontsize)
  heat.set_yticklabels(heat.get_ymajorticklabels(), fontsize = fontsize, rotation=0)
  plt.tight_layout()

  # Save output
  save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'scoring_matrix/')
  os.mkdir(save_path) if not os.path.exists(save_path) else None
  image_path = os.path.join(save_path, 'fitness_scoring_substitution_matrix_{}.png'.format(image_index))
  plt.savefig(image_path,dpi=100)
  # plt.show()

  return image_path

def suggest_mutations(scores, multi=False):
  intro_message = "The following mutations may be sensible options to improve fitness: \n\n"
  #Best mutants
  top_mutants=list(scores.sort_values(by=['avg_score'],ascending=False).head(5).mutant)
  top_mutants_fitness=list(scores.sort_values(by=['avg_score'],ascending=False).head(5).avg_score)
  top_mutants_recos = [top_mutant+" ("+str(round(top_mutant_fitness,4))+")" for (top_mutant,top_mutant_fitness) in zip(top_mutants,top_mutants_fitness)]
  # sorted_mutant_df = pd.DataFrame(list(zip(top_mutants, top_mutants_fitness)), columns =['top_mutants', 'top_mutants_score'])
  mutant_recos = "The single mutants with highest predicted fitness are (positive scores indicate fitness increase Vs starting sequence, negative scores indicate fitness decrease):\n {} \n\n".format(", ".join(top_mutants_recos))
  if not multi:
    #Best positions
    positive_scores = scores[scores.avg_score > 0]
    positive_scores_position_avg = positive_scores.groupby(['position']).mean(numeric_only=True)
    top_positions=list(positive_scores_position_avg.sort_values(by=['avg_score'],ascending=False).head(5).index.astype(str))
    position_recos = "The positions with the highest average fitness increase are (only positions with at least one fitness increase are considered):\n {} \n\n".format(", ".join(top_positions))
    return print(intro_message+mutant_recos+position_recos)
  else:
    return print(intro_message+mutant_recos)

def check_valid_mutant(sequence,mutant,AA_vocab=AA_vocab, multi_mutant=False):
  valid = True
  # print("Original Sequence: ", sequence)
  # print("Mutation: ", mutant)
  if multi_mutant:
    mutant_record = {}
    mutants = mutant.split(':')
    for index, mutant in enumerate(mutants):
      try:
        from_AA, position, to_AA = mutant[0], int(mutant[1:-1]), mutant[-1]
      except:
        AssertionError(f"Invalid mutant {mutant}")
      mutant_record[index] = mutant
      tmp_list = []
      tmp_mutant_record = mutant_record.copy()
      tmp_mutant_record.popitem()
      for key, value in tmp_mutant_record.items():
        if int(value[1:-1]) == position:
          tmp_list.append(key)
      last_index = max(tmp_list) if len(tmp_list) > 0 else None

      # check if range is valid and mutation is in AA_vocab
      assert int(mutant_record[index][1:-1])>=1 and int(mutant_record[index][1:-1])<=len(sequence),f"position {int(mutant_record[index][1:-1])} is out of range"
      assert mutant_record[index][-1] in AA_vocab, f"to_AA {mutant_record[index][-1]} is not in AA_vocab"
      # check if from_AA is consistent with previous mutant
      if last_index is not None:
        assert mutant_record[last_index][-1] == from_AA, f"To_AA ({mutant_record[last_index][-1]} in {last_index+1}) and From_AA ({from_AA} in {index+1}) are not consistent"
      else:
      # elif sequence[int(mutant_record[index][1:-1])-1]!=from_AA:
        # check if from_AA is consistent with sequence
        assert sequence[int(mutant_record[index][1:-1])-1]==from_AA, f"from_AA {from_AA} at {position} is not consistent with AA in sequence {sequence[int(mutant_record[index][1:-1])-1]} at position {int(mutant_record[index][1:-1])}"

  else:
    try:
      from_AA, position, to_AA = mutant[0], int(mutant[1:-1]), mutant[-1]
    except:
      valid = False
    assert sequence[position-1]==from_AA, f"from_AA {from_AA} is not consistent with sequence AA {sequence[position-1]}"
    assert position>=1 or position<=len(sequence), f"position {position} is out of range"
    assert to_AA in AA_vocab, f"to_AA {to_AA} is not in AA_vocab"
  return valid

def get_mutated_protein(sequence,mutant):
  mutated_sequence = list(sequence)
  multi_mutant=True if len(mutant.split(':'))>1 else False
  if multi_mutant:
    # print("multi mutant detected")
    assert check_valid_mutant(sequence,mutant, multi_mutant=True), "The mutant is not valid"
    for m in mutant.split(':'):
      from_AA, position, to_AA = m[0], int(m[1:-1]), m[-1]
      mutated_sequence[position-1]=to_AA
    return ''.join(mutated_sequence)
  else:
    # print("single mutant detected")
    assert check_valid_mutant(sequence,mutant), "The mutant is not valid"
    from_AA, position, to_AA = mutant[0], int(mutant[1:-1]), mutant[-1]
    mutated_sequence[position-1]=to_AA
  return ''.join(mutated_sequence)

def score_and_create_matrix_all_singles(sequence, Tranception_model, mutation_range_start=None,mutation_range_end=None,scoring_mirror=False,batch_size_inference=20,max_number_positions_per_heatmap=50,num_workers=0,AA_vocab=AA_vocab, tokenizer=tokenizer, with_heatmap=True, past_key_values=None, model_type='Tranception'):
  if mutation_range_start is None: mutation_range_start=1
  if mutation_range_end is None: mutation_range_end=len(sequence)
  assert len(sequence) > 0, "no sequence entered"
  assert mutation_range_start <= mutation_range_end, "mutation range is invalid"
  model = Tranception_model
  if torch.cuda.is_available():
    model.cuda()
    print("Inference will take place on GPU")
  else:
    print("Inference will take place on CPU")
  model.config.tokenizer = tokenizer
  all_single_mutants = create_all_single_mutants(sequence,AA_vocab,mutation_range_start,mutation_range_end)
  # print("Single variants generated")
  if model_type == 'Tranception':
    scores, past_key_values = model.score_mutants(DMS_data=all_single_mutants, 
                                      target_seq=sequence, 
                                      scoring_mirror=scoring_mirror, 
                                      batch_size_inference=batch_size_inference,  
                                      num_workers=num_workers, 
                                      indel_mode=False,
                                      past_key_values=past_key_values
                                      )
    print("Single scores computed")
    scores = pd.merge(scores,all_single_mutants,on="mutated_sequence",how="left")
  elif model_type == 'RITA':
    model_scores = compute_fitness.calc_fitness(model=model, prots=np.array(all_single_mutants['mutated_sequence']), tokenizer=tokenizer)
    scores = pd.concat([all_single_mutants, pd.DataFrame(model_scores, columns=['avg_score'])], axis=1)
    past_key_values = None
  else:
    raise ValueError('Invalid model type')  
  scores["position"]=scores["mutant"].map(lambda x: int(x[1:-1]))
  scores["target_AA"] = scores["mutant"].map(lambda x: x[-1])
  score_heatmaps = []
  if with_heatmap:
    mutation_range = mutation_range_end - mutation_range_start + 1
    number_heatmaps = int((mutation_range - 1) / max_number_positions_per_heatmap) + 1
    image_index = 0
    window_start = mutation_range_start
    window_end = min(mutation_range_end,mutation_range_start+max_number_positions_per_heatmap-1)
    for image_index in range(number_heatmaps):
      score_heatmaps.append(create_scoring_matrix_visual(scores,sequence,image_index,window_start,window_end,AA_vocab))
      window_start += max_number_positions_per_heatmap
      window_end = min(mutation_range_end,window_start+max_number_positions_per_heatmap-1)
  # return score_heatmaps, suggest_mutations(scores), scores, all_single_mutants, past_key_values
  return score_heatmaps, None, scores, all_single_mutants, past_key_values

def score_multi_mutations(sequence:str, extra_mutants:pd.DataFrame, Tranception_model, mutation_range_start=None,mutation_range_end=None,scoring_mirror=False,batch_size_inference=20,max_number_positions_per_heatmap=50,num_workers=0,AA_vocab=AA_vocab, tokenizer=tokenizer, AR_mode=False, past_key_values=None, verbose=0, model_type='Tranception'):
  if sequence is not None:
    if mutation_range_start is None: mutation_range_start=1
    if mutation_range_end is None: mutation_range_end=len(sequence)
  # assert len(sequence) > 0, "no sequence entered"
    assert mutation_range_start <= mutation_range_end, "mutation range is invalid"
  model = Tranception_model
  # print(f'model: {model}')
  model.config.tokenizer = tokenizer
  if torch.cuda.is_available():
    model.cuda()
    print("Inference will take place on GPU") if verbose == 1 else None
  else:
    print("Inference will take place on CPU") if verbose == 1 else None
  if model_type == 'Tranception':
    scores, past_key_values = model.score_mutants(DMS_data=extra_mutants, 
                                      target_seq=sequence, 
                                      scoring_mirror=scoring_mirror, 
                                      batch_size_inference=batch_size_inference,  
                                      num_workers=num_workers, 
                                      indel_mode=False,
                                      past_key_values=past_key_values,
                                      verbose=verbose
                                      )
    print("Scoring done") if verbose == 1 else None
    scores = pd.merge(scores,extra_mutants,on="mutated_sequence",how="left")
  elif model_type == 'RITA':
    model_scores = compute_fitness.calc_fitness(model=model, prots=np.array(extra_mutants['mutated_sequence']), tokenizer=tokenizer)
    scores = pd.concat([extra_mutants, pd.DataFrame(model_scores, columns=['avg_score'])], axis=1)
    past_key_values = None
  else:
    raise ValueError('Invalid model type')
  
  if AR_mode:
    return scores, extra_mutants, past_key_values
  else:
    # return suggest_mutations(scores, multi=True), scores, extra_mutants, past_key_values
    return None, scores, extra_mutants, past_key_values

def extract_sequence(example):
  label, taxon, sequence = example
  return sequence

def clear_inputs(protein_sequence_input,mutation_range_start,mutation_range_end):
  protein_sequence_input = ""
  mutation_range_start = None
  mutation_range_end = None
  return protein_sequence_input,mutation_range_start,mutation_range_end, extra_mutants

def load_savedmodel(model_path):
  model = keras.models.load_model(model_path)
  return model

def predict_evmutation(DMS, top_n, ev_model, return_evscore=False):
  # Load Model
  # c = CouplingsModel(model_params)
  c = ev_model
  start_predict = time.time()
  # print("===Predicting EVmutation===")
  DMS['mutant'] = DMS['mutant'].str.replace(':', ',')
  # print(f'ev predict table: {DMS}')
  DMS = predict_mutation_table(c, DMS, output_column="EVmutation")
  DMS = DMS.sort_values(by = 'EVmutation', ascending = False, ignore_index = True)
  # print(f'ev result table: {DMS}')
  # print("===Predicting EVmutation Done===")
  # print(f"Evmutation prediction time: {time.time() - start_predict}")
  DMS['mutant'] = DMS['mutant'].str.replace(',', ':')
  if return_evscore:
    return DMS[['mutated_sequence', 'mutant', 'EVmutation']].head(top_n)
  else:
    return DMS[['mutated_sequence', 'mutant']].head(top_n)

def get_all_possible_mutations_at_pos(sequence: str, position: int, or_mutant=None, return_dict=False, AA_vocab=AA_vocab):
    assert position >= 0 and position < len(sequence), "Invalid position"
    mutations = []
    for aa in AA_vocab:
        if aa != sequence[position]:
            new_sequence = sequence[:position] + aa + sequence[position+1:]
            if or_mutant:
              mutant_code = f"{or_mutant}:{sequence[position]}{position+1}{aa}"
            else:
              mutant_code = f"{sequence[position]}{position+1}{aa}"
            mutations.append({"mutated_sequence": new_sequence, "mutant": mutant_code})
    if return_dict:
      return mutations
    else:
      return pd.DataFrame(mutations)

def get_attention_mutants(DMS, AMSmodel, focus='highest', top_n = 5, AA_vocab=AA_vocab, tokenizer=tokenizer, model_type='Tranception'):
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  new_mutations = []
  for idx, row in tqdm.tqdm(DMS.iterrows(), total=len(DMS), desc=f'Getting attention mutants'):
    sequence = row['mutated_sequence']
    mutant = row['mutant']

    # Get attention scores
    inputs = torch.tensor([tokenizer.encode(sequence)]).to("cuda")
    if model_type == 'Tranception':
      attention_weights = AMSmodel(input_ids=inputs, return_dict=True, output_attentions=True).attentions
      attention_scores = attention_weights[-1][0].mean(dim=(0, 1))[1:-1].tolist()
    elif model_type == 'RITA':
      # TODO: Implement RITA attention score
      attention_weights = AMSmodel(input_ids=inputs).hidden_states
      attention_scores = attention_weights[-1][0].mean(dim=(0))[1:-1].tolist()
    else: 
      raise ValueError('Invalid model type')
    # print(f'as: {attention_scores}')
    

    if focus == 'highest':
      ind = np.argpartition(attention_scores, -top_n)[-top_n:]
    elif focus == 'lowest':
      ind = np.argpartition(attention_scores, top_n)[:top_n]
    else:
      raise ValueError('Invalid focus value')
    
    # Mutate sequence at positions with highest attention scores
    for pos in ind:
      dict_pos = get_all_possible_mutations_at_pos(sequence, pos, or_mutant=mutant, return_dict=True)
      new_mutations.extend(dict_pos)
  
  new_mutations = pd.DataFrame(new_mutations)
  new_mutations = new_mutations.drop_duplicates(subset=['mutant']).reset_index(drop=True)
  return new_mutations[['mutant','mutated_sequence']]

  
def split_mask(s):
  # Split string using regular expression pattern
  parts = re.split(r'(\?)', s)
  # Replace [MASK] with space
  # parts = [part.replace('?', ' ') for part in parts]
  # Remove empty parts
  parts = [part for part in parts if part]
  return parts

def replacer(s:str, position:list, replacement:str = '?'):
  s_list = list(s)
  for pos in position:
    s_list[pos] = replacement
  s = ''.join(s_list)
  return s
    
def process_prompt_protxlnet(s):
  # s = s.replace('[MASK]', '?')
  s = " ".join(s)
  # s = s.replace('?', '[MASK]')
  s = re.sub(r"[UZOB]", "<unk>", s)
  return s

def stratified_filtering(DMS, threshold, column_name='EVmutation'):
  try:
    DMS['strata'] = pd.qcut(DMS[column_name], q=4, labels=['very low', 'low', 'high', 'very high'])
  except IndexError:
    print(f'Indexing Error: {column_name} may not be scored properly')
    return DMS[['mutant', 'mutated_sequence']].reset_index(drop=True).sample(n=threshold)
  if threshold >= 4:
    threshold = threshold // 4
  elif threshold < 4:
    threshold = 1
  filtered = DMS.groupby('strata', group_keys=False).apply(lambda x: x.sample(min(len(x), threshold)))
  return filtered[['mutant', 'mutated_sequence']].reset_index(drop=True)
############################################################################################################
def list_of_dicts_to_df(lst):
    return pd.DataFrame(lst)

def generate_1extra_mutation(row, AA_vocab=AA_vocab, mutation_range_start=None, mutation_range_end=None):
    seq = row["mutated_sequence"]
    if mutation_range_start is None: mutation_range_start=1
    if mutation_range_end is None: mutation_range_end=len(seq)
    new_variants = []
    for i in range(mutation_range_start-1, mutation_range_end):
        for aa in AA_vocab:
            if aa != seq[i]:
                new_variant = {
                    "mutated_sequence": seq[:i] + aa + seq[i+1:],
                    "mutant": row["mutant"] + f":{seq[i]}{i+1}{aa}"
                }
                new_variants.append(new_variant)
    return new_variants

def apply_gen_1extra(DMS):
  # print(f'Creating 1 extra mutation')
  data = DMS.apply(generate_1extra_mutation, axis=1)
  # Apply function to each element of the Series
  df = data.apply(list_of_dicts_to_df)
  df

  # Concatenate resulting DataFrames into a single DataFrame
  df = pd.concat(df.to_list(), ignore_index=True)
  return df