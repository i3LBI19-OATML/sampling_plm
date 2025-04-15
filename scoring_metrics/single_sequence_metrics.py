# Single sequence metrics
# ESM-1v, ESM-1v-mask6, CARP-640m-logp, Repeat-1, Repeat-2, Repeat-3, Repeat-4

# CARP_640m_logp = True
# ESM_1v = True
# ESM_1v_mask6 = True
# repeat_1 = True
# repeat_2 = True
# repeat_3 = True
# repeat_4 = True

from .util import add_metric, identify_mutation, add_sequence
import tempfile
import subprocess
import pandas as pd
from glob import glob
import torch
from pgen.utils import parse_fasta
import os
from Bio.SeqIO.FastaIO import SimpleFastaParser
import numpy as np

from transformers import PreTrainedTokenizerFast
import tranception
from tranception import model_pytorch

#CARP
def CARP_640m_logp(target_seqs_file, results, device): 
  with tempfile.TemporaryDirectory() as output_dir:
    try:
      proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp/extract.py"), "carp_640M", target_seqs_file, output_dir + "/", "--repr_layers", "logits", "--include", "logp", "--device", device], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
      print(e.stderr.decode('utf-8'))
      print(e.stdout.decode('utf-8'))
      raise e
    df = pd.read_table(output_dir + '/carp_640M_logp.tsv')
    df = df.rename(columns={'name': 'id', 'logp': 'carp640m_logp'},)
    for _, row in df.iterrows():
      add_metric(results, row["id"], "CARP-640m", row["carp640m_logp"])
    del df

# ESM1v (ProteinGym Version)
def ESM_1v(target_files, results, device, orig_seq): #TODO: allow other devices?
  if device=='cuda:0':
    torch.cuda.empty_cache()
  if not os.path.isfile(os.path.expanduser("~/esm1v_t33_650M_UR90S_1.pt")):
    subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt', '-P', os.path.expanduser("~/")], check=True)
  with tempfile.TemporaryDirectory() as output_dir:
    outfile = output_dir + "/esm_results.csv"
    try:
      # proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "protein_gibbs_sampler/src/pgen/likelihood_esm.py"), "-i", targets_fasta, "-o", outfile, "--model", "esm1v", "--masking_off", "--score_name", "score", "--device", "gpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
      
      # ProteinGym Version
      seq_name, seq = parse_fasta(target_files,return_names=True, clean="unalign")
      # ref_name, ref = parse_fasta(reference_seqs_file,return_names=True, clean="unalign")

      df_target = pd.DataFrame({"id":seq_name, "sequence":seq})
      # identify mutations
      df_target = df_target[df_target['sequence'].apply(lambda x: len(x) == len(orig_seq))]
      df_target['mutant'] = df_target['sequence'].apply(lambda x: identify_mutation(orig_seq, x, sep=":"))
      # remove nan
      df_target = df_target[df_target['mutant'] != np.nan]
      df_perfect_target = df_target[df_target['mutant'] == np.nan]

      if df_target.shape[0] > 0:
        with tempfile.TemporaryDirectory() as temp_dir:
          df_target.to_csv(os.path.join(temp_dir, "target.csv"), index=False)
          df_target = os.path.join(temp_dir, "target.csv")

          # print(f'ESM1v:')
          proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "ProteinGym/proteingym/baselines/esm/compute_fitness.py"), "--sequence", orig_seq, "--model_type", "ESM1v", "--dms-input", os.path.join(temp_dir, "target.csv"), "--dms-output", outfile, "--mutation-col", "mutant", "--model-location", os.path.expanduser("~/esm1v_t33_650M_UR90S_1.pt"), "--overwrite-prior-scores"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      else:
        outfile = None
    
    except subprocess.CalledProcessError as e:
      print(e.stderr.decode('utf-8'))
      print(e.stdout.decode('utf-8'))
      raise e

    # print(proc.stdout)
    # print(proc.stderr)
    df = pd.read_csv(outfile) if outfile else pd.DataFrame()
    if df_perfect_target.shape[0] > 0:
      df_perfect_target['Ensemble_ESM1v'] = 1
      df = pd.concat([df, df_perfect_target])
      df = df.drop_duplicates(subset=['id'])
      print(f'ESM-MSA result.shape: {df.shape} (should be 100)')

    # print(f'ESM1v.columns: {df.columns}')
    # print(f'P-Gym ESM-1v results: {df[["id", "Ensemble_ESM1v"]].head()}')
    for i, row in df.iterrows():
      add_metric(results, row["id"], "ESM-1v_PGym", row["Ensemble_ESM1v"])
    del df

# ESM1v (unmasked) - Sean R Johnson version
def ESM_1v_unmask(targets_fasta, results, device, return_pred=False): #TODO: allow other devices?
  if device=='cuda:0':
    torch.cuda.empty_cache()
  pred_arr = []
  with tempfile.TemporaryDirectory() as output_dir:
    outfile = output_dir + "/esm_results.tsv"
    try:
      proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "protein_gibbs_sampler/src/pgen/likelihood_esm.py"), "-i", targets_fasta, "-o", outfile, "--model", "esm1v", "--masking_off", "--score_name", "score", "--device", "gpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
      print(e.stderr.decode('utf-8'))
      print(e.stdout.decode('utf-8'))
      raise e
    df = pd.read_table(outfile)
    for i, row in df.iterrows():
      add_metric(results, row["id"], "ESM-1v", row["score"])
      if return_pred:
        p = row['score']
        pred_arr.append(p)
    del df
    if return_pred:
      return pred_arr

# ProGen2
def Progen2(target_files, results, device): #TODO: allow other devices?
  if device=='cuda:0':
    torch.cuda.empty_cache()
  if not os.path.isdir(os.path.expanduser("~/progen2-base")):
    subprocess.run(['wget', 'https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-base.tar.gz', '-P', os.path.expanduser("~/")], check=True)
    os.mkdir(os.path.expanduser("~/progen2-base"), exist_ok=True)
    subprocess.run(['tar', '-xzf', os.path.expanduser("~/progen2-base.tar.gz"), '-C', os.path.expanduser("~/progen2-base")], check=True)
    subprocess.run(['rm', os.path.expanduser("~/progen2-base.tar.gz")], check=True)

  with tempfile.TemporaryDirectory() as output_dir:
    outfile = output_dir + "/progen2_results.csv"
    try:      
      # ProteinGym Version
      seq_name, seq = parse_fasta(target_files,return_names=True, clean="unalign")
      # ref_name, ref = parse_fasta(reference_seqs_file,return_names=True, clean="unalign")

      df_target = pd.DataFrame({"id":seq_name, "mutated_sequence":seq})

      with tempfile.TemporaryDirectory() as temp_dir:
        df_target.to_csv(os.path.join(temp_dir, "target.csv"), index=False)
        df_target = os.path.join(temp_dir, "target.csv")

        proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "ProteinGym/proteingym/baselines/progen2/compute_fitness.py"), "--DMS_data_folder", os.path.join(temp_dir, "target.csv"), "--output_scores_folder", outfile, "--Progen2_model_name_or_path", os.path.expanduser("~/progen2-base"), "--indel_mode"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    except subprocess.CalledProcessError as e:
      print(e.stderr.decode('utf-8'))
      print(e.stdout.decode('utf-8'))
      raise e

    # print(proc.stdout)
    # print(proc.stderr)
    df = pd.read_csv(outfile)

    for i, row in df.iterrows():
      add_metric(results, row["id"], "Progen2", row["Progen2_score"])
    del df

# ESM1v mask 6
def ESM_1v_mask6(target_files, results, device): #TODO: allow other devices?
  if device=='cuda:0':
    torch.cuda.empty_cache()
  for targets_fasta in target_files:
    with tempfile.TemporaryDirectory() as output_dir:
      outfile = output_dir + "/esm_results.tsv"
      proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "protein_gibbs_sampler/src/pgen/likelihood_esm.py"), "-i", targets_fasta, "-o", outfile, "--model", "esm1v", "--mask_distance", "6", "--score_name", "score", "--device", "gpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
      # print(proc.stdout)
      # print(proc.stderr)
      df = pd.read_table(outfile)
      for i, row in df.iterrows():
        add_metric(results, row["id"], "ESM-1v mask6", row["score"])
      del df

# repeat
def find_longest_repeat(seq, k):
  longest = [1] * len(seq)
  pattern = [None] * len(seq)
  
  seq_len = len(seq)
  for i in range(seq_len):
    if i + k <= seq_len:
      pattern[i] = seq[i:i+k]
    if i - k >= 0:
      if pattern[i-k] == pattern[i]:
        longest[i] = longest[i-k] + 1
  return -1 * max(longest)

def Repeat(target_files, repeat_score, results):
    repeat_ks = list()
    if repeat_score['repeat_1']:
        repeat_ks.append(1)
    if repeat_score['repeat_2']:
        repeat_ks.append(2)
    if repeat_score['repeat_3']:
        repeat_ks.append(3)
    if repeat_score['repeat_4']:
        repeat_ks.append(4)

    for k in repeat_ks:
        for targets_fasta in target_files:
            for name, seq in zip(*parse_fasta(targets_fasta, return_names=True, clean="unalign")):
                score = find_longest_repeat(seq, k)
                add_metric(results, name, f"longest_repeat_{k}", score)

# Tranception
def Tranception(target_files, orig_seq, results, device, model_type="Large", local_model=os.path.expanduser("~/Tranception_Large"), past_key_values=None):
  if device=='cuda:0':
    torch.cuda.empty_cache()
  for targets_fasta in target_files:
    with open(targets_fasta) as fasta_file:  # Will close handle cleanly
      identifiers = []
      seqeunce = []
      for title, sequence in SimpleFastaParser(fasta_file):
          identifiers.append(title.split(None, 1)[0])  # First word is ID
          seqeunce.append(sequence)
      targets = pd.DataFrame({"id":identifiers,"mutated_sequence":seqeunce})
    print("Tranception scores computing...")
    try:
      model = model_pytorch.TranceptionLMHeadModel.from_pretrained(local_model, local_files_only=True)
      print("Tranception model loaded from local file")
    except:
      print("Downloading Tranception model...")
      if model_type=="Small":
        model = model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Small")
      elif model_type=="Medium":
        model = model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Medium")
      elif model_type=="Large":
        model = model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Large")
    if torch.cuda.is_available():
      model.cuda()
      print("Inference will take place on GPU")
    else:
      print("Inference will take place on CPU")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tokenizers/Basic_tokenizer"),
                                                    unk_token="[UNK]",
                                                    sep_token="[SEP]",
                                                    pad_token="[PAD]",
                                                    cls_token="[CLS]",
                                                    mask_token="[MASK]"
                                                )
    model.config.tokenizer = tokenizer
    scores, past_key_values = model.score_mutants(DMS_data=targets, 
                                    target_seq=orig_seq, # need template seq
                                    scoring_mirror=False, 
                                    batch_size_inference=20,  
                                    num_workers=0, 
                                    indel_mode=True,
                                    past_key_values=past_key_values,
                                    )
    print("Tranception scores computed")
    scores = pd.merge(scores,targets,on="mutated_sequence",how="left")
    for i, row in scores.iterrows():
        add_metric(results, row["id"], "Tranception", row["avg_score"])
    del scores
    return past_key_values

def add_sequence_to_result(target_files, results):
  seq_name, seq = parse_fasta(target_files,return_names=True, clean="unalign")
  df_target = pd.DataFrame({"id":seq_name, "sequence":seq})

  for i, row in df_target.iterrows():
    add_sequence(results, row["id"], row["sequence"])