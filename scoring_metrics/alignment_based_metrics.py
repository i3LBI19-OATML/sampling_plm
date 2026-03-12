# Alignment-based metrics
# ESM-MSA, Identity to closest reference, Subtitution matix (BLOSUM62 or PFASUM15) score mean of mutated positions

# ESM_MSA = True
# Substitution_matrix_score_mean_of_mutated_positions = True
# Identity_to_closest_reference = True

from .util import add_metric, identify_mutation, extract_mutations
import subprocess
import tempfile
import pandas as pd
import numpy as np
from pgen.utils import parse_fasta
from scipy.spatial.distance import pdist
import os
from EVmutation.model import CouplingsModel
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.Emboss.Applications import NeedleCommandline


# ESM-MSA
def ESM_MSA(target_seqs_file, reference_seqs_file, results, orig_seq, msa_weights):
  print("Scoring with ESM-MSA")
  # rand_id = np.random.randint(100000,100000)
  # msa_result_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"tmp/esm_msa_cache/esm_msa_{rand_id}.csv")
  # os.makedirs(os.path.dirname(os.path.realpath(msa_result_path))) if not os.path.exists(os.path.dirname(os.path.realpath(msa_result_path))) else None
  # check if model is downloaded
  if not os.path.isfile(os.path.expanduser("~/esm_msa1b_t12_100M_UR50S.pt")):
    subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt', '-P', os.path.expanduser("~/")], check=True)

  with tempfile.TemporaryDirectory() as output_dir:
    outfile = os.path.join(output_dir, "esm_results.csv")
    try:
      # proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "protein_gibbs_sampler/src/pgen/likelihood_esm_msa.py"), "-i", target_seqs_file, "-o", outfile, "--reference_msa", reference_seqs_file, "--subset_strategy", "top_hits", "--alignment_size", "384", "--count_gaps", "--mask_distance", "6", "--device", "gpu"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # stdout=subprocess.PIPE, stderr=subprocess.PIPE

      # ProteinGym Version
      seq_name, seq = parse_fasta(target_seqs_file,return_names=True, clean="unalign")
      # ref_name, ref = parse_fasta(reference_seqs_file,return_names=True, clean="unalign")

      df_target = pd.DataFrame({"id":seq_name, "sequence":seq})
      # identify mutations
      df_target = df_target[df_target['sequence'].apply(lambda x: len(x) == len(orig_seq))]
      df_target['mutant'] = df_target['sequence'].apply(lambda x: identify_mutation(orig_seq, x, sep=":"))
      # remove nan
      df_target = df_target[df_target['mutant'] != np.nan]
      df_perfect_target = df_target[df_target['mutant'] == np.nan]

      if df_target.shape[0] > 0:
        # create a temp file path for target csv and reference MSA
        with tempfile.TemporaryDirectory() as temp_dir:
          df_target.to_csv(os.path.join(temp_dir, "target.csv"), index=False)
          df_target = os.path.join(temp_dir, "target.csv")
          
          if msa_weights is not None:
            proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "ProteinGym/proteingym/baselines/esm/compute_fitness.py"), "--sequence", orig_seq, "--dms-input", df_target, "--dms-output", outfile, "--mutation-col", "mutant", "--model-location", os.path.expanduser("~/esm_msa1b_t12_100M_UR50S.pt"), "--msa-path", reference_seqs_file, "--msa-weights-folder", msa_weights, "--filter-msa", "--overwrite-prior-scores"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
          else:
            proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "ProteinGym/proteingym/baselines/esm/compute_fitness.py"), "--sequence", orig_seq, "--dms-input", df_target, "--dms-output", outfile, "--mutation-col", "mutant", "--model-location", os.path.expanduser("~/esm_msa1b_t12_100M_UR50S.pt"), "--msa-path", reference_seqs_file, "--filter-msa", "--overwrite-prior-scores"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      else:
        outfile = None

    except subprocess.CalledProcessError as e:
      print(e.stderr.decode('utf-8'))
      print(e.stdout.decode('utf-8'))
      raise e

    # debug
    # print(proc.stdout)
    # print(proc.stderr)

    df = pd.read_csv(outfile) if outfile else pd.DataFrame()
    if df_perfect_target.shape[0] > 0:
      df_perfect_target['esm_msa1b_t12_100M_UR50S_ensemble'] = 1
      df = pd.concat([df, df_perfect_target], ignore_index=True)
      df = df.drop_duplicates(subset=['id'])
      print(f'ESM-MSA result.shape: {df.shape} (should be 100)')

    # print(f'ESM-MSA.columns: {df.columns}')
    # print(f'P-Gym ESM-MSA results: {df.head()}')
    for i, row in df.iterrows():
      add_metric(results, row["id"], "ESM-MSA", row["esm_msa1b_t12_100M_UR50S_ensemble"])
    del df

# substitution score
def substitution_score(target_seqs_file, reference_seqs_file, substitution_matrix:str, Substitution_matrix_score_mean_of_mutated_positions:bool, Identity_to_closest_reference:bool, results, gap_open:int = 10, gap_extend:int = 2,):
  #SEARCH #/tmp/mat.mat
  assert substitution_matrix in ["BLOSUM62", "PFASUM15"], "substitution_matrix must be 'BLOSUM62' or 'PFASUM15'"
  search_results_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"tmp/ggsearch_results_{substitution_matrix}.txt")
  substitution_matrix_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'tmp/{substitution_matrix}.mat')
  # print("Matrix file: ", substitution_matrix_file)
  ggsearch_path = str(os.path.join(os.path.dirname(os.path.realpath(__file__)), "fasta3/ggsearch36"))
  
  with open(search_results_file,"w") as fh:
    try:
      proc = subprocess.run([ggsearch_path, '-f', str(gap_open), '-g', str(gap_extend), '-s', substitution_matrix_file, '-b,' '1', target_seqs_file, reference_seqs_file], stdout=subprocess.PIPE, check=True, stderr=subprocess.PIPE)
      print(proc.stdout.decode('utf-8'), file=fh)
    except subprocess.CalledProcessError as e:
      print(e.stderr.decode('utf-8'))
      print(e.stdout.decode('utf-8'))
      raise e

  df = pd.read_csv(substitution_matrix_file, delimiter=r"\s+")
  blosum62 = {}
  for aa1 in df.columns:
    for aa2 in df.index:
      blosum62[(aa1, aa2)] = df.loc[aa1, aa2]
  i = np.random.randint(100000)
  n_train, s_train = parse_fasta(reference_seqs_file,return_names=True)
  n_query,s_query = parse_fasta(target_seqs_file,return_names=True)
  n_query = [nq.strip() for nq in n_query]
  train_seqs = {nt: st for st, nt in zip(s_train,n_train)}
  query_seqs = {nq: sq for sq, nq in zip(s_query,n_query)}

  # with open(output[0],'w') as f: # ell is the number of non-gaps in the query sequence.
      # f.write(f'id,{substitution_matrix},{substitution_matrix}_n_aligned,{substitution_matrix}_ell,{substitution_matrix}_mut_mean,{substitution_matrix}_n_mut,{substitution_matrix}_worst,closest\n')

  with open(search_results_file) as f:
    lines = f.readlines()
  train_coming = False
  qns = [] #query names
  tns = [] #target names
  for i, line in enumerate(lines):
    if '!! No sequences' in line:
      tns.append(None)

    if not train_coming:
      if 'The best scores are:' in line:
        train_coming = True
    else:
      tns.append(line.split()[0])
      train_coming = False
    if 'Library: ' in line:
      qns.append(lines[i - 1].split('>')[-1].split()[0])
  for qn, tn in zip(qns,tns):
    if tn is None: #No hits found
      average_score = str(0) #TODO: I'm not sure what sensible defaults for any of these are
      n_aligned = 0
      mutant_score = 0
      worst_score = 0
      # ell = sum([s != '-' for s in query_seqs[qn]]) # just the length of the query sequence.
      ell = len(query_seqs[qn]) - query_seqs[qn].count('-')
      tn=""
      identity = 0
    else:
      # write a temporary fasta file
      with tempfile.TemporaryDirectory() as output_dir:
        pairwise_result_file = output_dir + '/pairwise_result.fasta'
        pairwise_target_fasta = output_dir + '/a_sequence.fasta'
        pairwise_query_fasta = output_dir + '/b_sequence.fasta'
        with open(pairwise_target_fasta,'w') as f_target, open(pairwise_query_fasta,'w') as f_query:
          f_target.write('>' + tn + '\n' + train_seqs[tn] + '\n')
          f_query.write('>' + qn + '\n' + query_seqs[qn] + '\n')
        # with open(pairwise_query_fasta,'w') as f:
        #   f.write('>' + qn + '\n' + query_seqs[qn] + '\n')
        needle_cline = NeedleCommandline(asequence=pairwise_target_fasta, bsequence=pairwise_query_fasta, gapopen=str(gap_open), gapextend=str(gap_extend), datafile=substitution_matrix_file, aformat='fasta', outfile=pairwise_result_file)
        stdout, stderr = needle_cline()
        # subprocess.run(["needle", "-gapopen", str(gap_open), "-gapextend", str(gap_extend), "-datafile", substitution_matrix_file, pairwise_target_fasta, pairwise_query_fasta, '-aformat', 'fasta', '-outfile', pairwise_result_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # score it
        seqs = parse_fasta(pairwise_result_file)
        global_score = 0
        mutant_score = 0
        worst_score = np.inf
        n_mutants = 0
        n_aligned = 0
        # print("Closest sequence:", seqs[0])
        seqs_array = np.array([[ord(ss) for ss in list(s)] for s in seqs])
        dist = pdist(seqs_array, metric='hamming')[0]
        identity = 1.0 - dist
        for aa1, aa2 in zip(*seqs):
          if '-' in aa1 + aa2:
            continue
          score = blosum62[(aa1, aa2)]
          global_score += score
          n_aligned += 1
          if aa1 != aa2:
            mutant_score += score
            n_mutants += 1
            if score < worst_score:
                worst_score = score
        if n_mutants > 0:
          mutant_score /= n_mutants
        ell = len(seqs[1]) - seqs[1].count('-')
        # ell = sum([s != '-' for s in seqs[1]]) # ell is the number of non-gaps in the query sequence.
        if n_aligned == 0: #TODO: same here, are these defaults sensible?
          mutant_score = 0
          worst_score = 0
          average_score = str(0)
          tn=""
        else:
          average_score = str(global_score / n_aligned)
    add_metric(results, qn, f"Closest training sequence ({substitution_matrix})", tn)
    if Substitution_matrix_score_mean_of_mutated_positions:
      add_metric(results, qn, substitution_matrix, mutant_score)
    if Identity_to_closest_reference: 
      try:   
        add_metric(results, qn, "Identity", identity)
        add_metric(results, qn, "SD", dist)
      except UnboundLocalError:
        add_metric(results, qn, "Identity", None)
        add_metric(results, qn, "SD", None)

def EVmutation(target_files, orig_seq, results, model_params):
  # Load Model
  c = CouplingsModel(model_params)
  # Load targets
  for targets_fasta in target_files:
    with open(targets_fasta) as fasta_file:  # Will close handle cleanly
      identifiers = []
      seqeunce = []
      for title, sequence in SimpleFastaParser(fasta_file):
          identifiers.append(title.split(None, 1)[0])  # First word is ID
          seqeunce.append(sequence)
      targets = pd.DataFrame({"id":identifiers,"mutated_sequence":seqeunce})
  # Calculate delta E (log-likehood of mutation)
  for i, row in targets.iterrows():
    target = row['mutated_sequence']
    # if len(target) > len(orig_seq):
    #   target = target[:len(orig_seq)]
    # else:
    #   target = target + "X"*(len(orig_seq)-len(target))
    mutations = identify_mutation(orig_seq, target)
    extract = extract_mutations(mutations)
    delta_E, delta_E_couplings, delta_E_fields = c.delta_hamiltonian(extract)
    add_metric(results, row['id'], "EVmutation", delta_E)



    
