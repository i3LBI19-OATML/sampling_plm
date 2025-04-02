device = 'cuda:0'
import random
from glob import glob
from pgen.utils import parse_fasta
import pandas as pd
import os
import argparse
from random import randint
import tempfile

from scoring_metrics import structure_metrics as st_metrics
from scoring_metrics import single_sequence_metrics as ss_metrics
from scoring_metrics import alignment_based_metrics as ab_metrics
from scoring_metrics import fid_score as fid
from scoring_metrics import esmfold
import time

#Reset calculated metrics (creates a new datastructure to store results, clearing any existing results)
results = dict()
start_time = time.time()

# Default directories
file_dir = os.path.dirname(os.path.realpath(__file__))
default_pdb_dir = os.path.join(file_dir, "pdbs")
default_reference_dir = os.path.join(file_dir, "reference_seqs")
default_target_dir = os.path.join(file_dir, "target_seqs")

os.makedirs(default_pdb_dir) if not os.path.exists(default_pdb_dir) else None
os.makedirs(default_reference_dir) if not os.path.exists(default_reference_dir) else None
os.makedirs(default_target_dir) if not os.path.exists(default_target_dir) else None

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pdb_dir", type=str, default=default_pdb_dir, help="Directory containing pdb files")
parser.add_argument("--reference_dir", type=str, required=True, help="Directory containing reference fasta files")
parser.add_argument("--msa_weights_dir", type=str, required=True, help="Directory containing MSA weights files (Obtain from ProteinGym repo)")
parser.add_argument("--reference_pdb", type=str, required=True, help="Reference pdb file")
parser.add_argument("--target_dir", type=str, required=True, help="Directory containing target fasta files")
parser.add_argument("--sub_matrix", type=str, choices=["blosum62", "pfasum15"], default="blosum62", help="Substitution matrix to use for alignment-based metrics")
parser.add_argument("--remove_sub_score_mean", action="store_false", help="Whether to not score the mean of the scores for mutated sequences")
parser.add_argument("--remove_identity", action="store_false", help="Whether to not score the identity of the mutated sequence to the closest reference sequence")
parser.add_argument("--sub_gap_open", type=int, default=10, help="Gap open penalty for alignment-based metrics")
parser.add_argument("--sub_gap_extend", type=int, default=2, help="Gap extend penalty for alignment-based metrics")
parser.add_argument("--remove_repeat_score_1", action="store_false", help="Whether to not score the first repeat")
parser.add_argument("--remove_repeat_score_2", action="store_false", help="Whether to not score the second repeat")
parser.add_argument("--remove_repeat_score_3", action="store_false", help="Whether to not score the third repeat")
parser.add_argument("--remove_repeat_score_4", action="store_false", help="Whether to not score the fourth repeat")
parser.add_argument("--score_existing_structure", action="store_true", help="Whether to score existing pdb files")
parser.add_argument("--use_tranception", action="store_true", help="Whether to use Tranception")
parser.add_argument("--use_evmutation", action="store_true", help="Whether to use EVmutation")
parser.add_argument("--skip_FID", action="store_true", help="Whether to not calculate FID")
parser.add_argument("--model_params", type=str, help="Model params to use for EVmutation")
parser.add_argument("--orig_seq", required=True, type=str, help="Original sequence to use for Tranception or EVmutation")
parser.add_argument('--output_name', type=str, required=True, help='Output file name (Just name with no extension!)')
args = parser.parse_args()

# Checks
if args.use_tranception or args.use_evmutation:
  assert args.orig_seq, "Must specify original sequence if using Tranception or EVmutation"
if args.use_evmutation:
  assert args.model_params, "Must specify model params if using EVmutation"
  assert os.path.exists(args.model_params), f"Model params {args.model_params} does not exist"

# Check that the required directories exist
pdb_dir = os.path.abspath(args.pdb_dir) if args.score_existing_structure else None
reference_pdb = os.path.abspath(args.reference_pdb)
reference_dir = os.path.abspath(args.reference_dir)
target_dir = os.path.abspath(args.target_dir)
msa_weights_dir = os.path.abspath(args.msa_weights_dir)

if args.score_existing_structure: assert os.path.exists(pdb_dir), f"PDB directory {pdb_dir} does not exist" 
assert os.path.isfile(reference_pdb), f"Reference pdb file {reference_pdb} does not exist"
assert os.path.exists(reference_dir), f"Reference directory {reference_dir} does not exist"
assert os.path.exists(target_dir), f"Target directory {target_dir} does not exist"
assert os.path.exists(msa_weights_dir), f"MSA weights directory {msa_weights_dir} does not exist"

# Check that the required files exist
pdb_files = glob(pdb_dir + "/*.pdb") if args.score_existing_structure else None
reference_files = glob(reference_dir + "/*.fasta")
# msat_reference_files = glob(reference_dir + "/*.csv")
target_files = glob(target_dir + "/*.fasta")
msa_weights_files = glob(msa_weights_dir + "/*.npy")[0]

if args.score_existing_structure: assert len(pdb_files) > 0, f"No pdb files found in {pdb_dir}"
assert len(reference_files) > 0, f"No reference fasta files found in {reference_dir}"
# assert len(msat_reference_files) > 0, f"No reference csv files (for MSAT) found in {reference_dir}"
assert len(target_files) > 0, f"No target fasta files found in {target_dir}"
assert len(msa_weights_files) > 0, f"No MSA weights files found in {msa_weights_dir}"

sub_matrix = args.sub_matrix.upper()
score_mean = args.remove_sub_score_mean
identity = args.remove_identity
sub_gap_open = args.sub_gap_open
sub_gap_extend = args.sub_gap_extend
mask_distance = round(len(args.orig_seq)/len(args.orig_seq)*0.15) # mask distance is 15% of the length of the original sequence

rand_id = randint(10000, 99999) # Necessary for parallelization
# print("===========================================")
print(f"Using random ID {rand_id} for temporary files")

# Temporary files
with tempfile.TemporaryDirectory() as output_dir:
  for directory in ["random_unalign_ref_cache", "unalign_ref_cache", "raw_ref_cache", "target_cache"]:
    os.makedirs(os.path.join(output_dir, directory), exist_ok=True)
    
  reference_seqs_file = os.path.join(output_dir, f"random_unalign_ref_cache/reference_seqs_{rand_id}.fasta")
  full_reference_seqs_file = os.path.join(output_dir, f"unalign_ref_cache/full_reference_seqs_{rand_id}.fasta")
  raw_reference_seqs_file = os.path.join(output_dir, f"raw_ref_cache/raw_reference_seqs_{rand_id}.fasta")
  target_seqs_file = os.path.join(output_dir, f"target_cache/target_seqs_{rand_id}.fasta")

  # Reference sequences
  # concatenate reference sequences
  n = 400  # default value for quick analysis; replace with the number of sequences you want
  sequences = []

  for idf, reference_fasta in enumerate(reference_files):
    # idf to alphabet
    did = chr(65 + idf)

    # Parse once for raw sequences
    parsed = list(zip(*parse_fasta(reference_fasta, return_names=True, clean=None, full_name=True)))
    raw_lines = [f">{did}{idx}_{name}\n{seq}\n" for idx, (name, seq) in enumerate(parsed)]

    # Parse once for cleaned sequences
    parsed_clean = list(zip(*parse_fasta(reference_fasta, return_names=True, clean="unalign")))
    full_lines = [f">{did}{idx}_{name}\n{seq}\n" for idx, (name, seq) in enumerate(parsed_clean)]

    # Collect sequences for further processing
    sequences.extend((f"{did}{idx}_{name}", seq) for idx, (name, seq) in enumerate(parsed_clean))

  # Write raw and full reference sequences
  with open(raw_reference_seqs_file, "w") as fh:
      fh.writelines(raw_lines)

  with open(full_reference_seqs_file, "w") as fh:
      fh.writelines(full_lines)

  # Random sampling and writing selected sequences
  sample_size = min(n, len(sequences))
  selected_sequences = random.sample(sequences, sample_size)
  print(f"Selected {sample_size} sequences from {len(sequences)} sequences")

  with open(reference_seqs_file, "w") as fh:
      fh.writelines(f">{name}_{idx}\n{seq}\n" for idx, (name, seq) in enumerate(selected_sequences))

  # Target sequences
  with open(target_seqs_file,"w") as fh:
    for idf, target_fasta in enumerate(target_files):
      did = chr(65 + idf)
      # generate unique names for each fasta file
      for idx, (name, seq) in enumerate(zip(*parse_fasta(target_fasta, return_names=True, clean="unalign"))):
        name = f"{did}{idx}_{name}"
        print(f">{name}\n{seq}", file=fh)
  
  structure_time = time.time()
  if args.score_existing_structure:
    # Structure metrics
    # ESM-IF, ProteinMPNN, MIF-ST, AlphaFold2 pLDDT, TM-score
    st_metrics.TM_score(pdb_files, reference_pdb, results)
    st_metrics.ESM_IF(pdb_files, results)
    st_metrics.ProteinMPNN(pdb_files, results)
    st_metrics.MIF_ST(pdb_files, results, device)
    st_metrics.AlphaFold2_pLDDT(pdb_files, results)
  else:
    esm_target = os.path.dirname(target_seqs_file)
    esmfold.predict_structure(esm_target, reference_pdb, save_dir=None, copies=1, num_recycles=3, keep_pdb=False, 
                      verbose=0, collect_output=True, results=results)
  print(f"############ STRUCTURE METRICS DONE! ({time.time() - structure_time}s) ############")

  # Alignment-based metrics
  # ESM-MSA, Identity to closest reference, Subtitution matix (BLOSUM62 or PFASUM15) score mean of mutated positions
  # FID (ESM-1v), EVmutation
  alignment_time = time.time()
  ab_metrics.ESM_MSA(target_seqs_file, raw_reference_seqs_file, results, orig_seq=args.orig_seq.upper(), msa_weights=msa_weights_files)
  ab_metrics.substitution_score(target_seqs_file, reference_seqs_file,
                                substitution_matrix=sub_matrix, 
                                Substitution_matrix_score_mean_of_mutated_positions=score_mean, 
                                Identity_to_closest_reference=identity,
                                results=results,
                                gap_open=sub_gap_open,
                                gap_extend=sub_gap_extend,)
  print(f"############ ALIGNMENT-BASED METRICS DONE! ({time.time() - alignment_time}s) ############")

  if args.use_evmutation:
    ab_metrics.EVmutation(target_files=[target_seqs_file], orig_seq=args.orig_seq.upper(), results=results, model_params=args.model_params)

  # Single sequence metrics
  # ESM-1v, ESM-1v-mask6, CARP-640m-logp, Repeat-1, Repeat-2, Repeat-3, Repeat-4, Tranception

  repeat_score = dict()
  repeat_score['repeat_1'] = args.remove_repeat_score_1
  repeat_score['repeat_2'] = args.remove_repeat_score_2
  repeat_score['repeat_3'] = args.remove_repeat_score_3
  repeat_score['repeat_4'] = args.remove_repeat_score_4

  single_time = time.time()
  ss_metrics.CARP_640m_logp(target_seqs_file, results, device)
  ss_metrics.ESM_1v(target_seqs_file, results, device, orig_seq=args.orig_seq.upper()) # ProteinGym ESM-1v model
  esm1v_pred = ss_metrics.ESM_1v_unmask(target_seqs_file, results, device, return_pred=True)
  ss_metrics.Progen2(target_seqs_file, results, device)
  ss_metrics.ESM_1v_mask6([target_seqs_file], results, device)
  ss_metrics.Repeat([target_seqs_file], repeat_score, results)
  if args.use_tranception:
    past_key_values = None
    past_key_values = ss_metrics.Tranception(target_files=[target_seqs_file], orig_seq=args.orig_seq.upper(), results=results, device=device, model_type="Large", local_model=os.path.expanduser("~/Tranception_Large"))
  print(f"############ SINGLE SEQUENCE METRICS DONE! ({time.time() - single_time}s) ############")

  # Download results
  df = pd.DataFrame.from_dict(results, orient="index")
  if not args.skip_FID:
    fid_time = time.time()
    fretchet_score = fid.calculate_fid_given_paths(esm1v_pred, full_reference_seqs_file, device=device, name=reference_dir, orig_seq=args.orig_seq.upper())
    # fretchet_score = fid.calculate_fid_given_paths(target_seqs_file, full_reference_seqs_file, device=device, name=reference_dir, orig_seq=args.orig_seq.upper())
    df["FID"] = fretchet_score
    print(f"FID took {time.time() - fid_time} seconds")
  else:
    fretchet_score = None

# # delete temporary files
# os.remove(reference_seqs_file)
# os.remove(full_reference_seqs_file)
# os.remove(target_seqs_file)

if args.score_existing_structure:
  save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "{}.csv".format(args.output_name))
else :
  save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "{}_no_structure.csv".format(args.output_name))
os.makedirs(os.path.dirname(os.path.realpath(save_path))) if not os.path.exists(os.path.dirname(os.path.realpath(save_path))) else None

df.to_csv(save_path)

print("===========================================")
print(f"SCORING COMPLETED SUCCESSFULLY | SAVED: {save_path} | TIME: {time.time() - start_time} seconds")
print("===========================================")

