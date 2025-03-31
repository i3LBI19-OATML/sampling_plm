from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import torch
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
from scipy.special import softmax
import gc
import tqdm
import tmscoring
from glob import glob
import pandas as pd
from pgen.utils import parse_fasta
import argparse
import time
import shutil
import tempfile
from .util import add_metric

# parser = argparse.ArgumentParser()
# parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results")
# parser.add_argument("--target_dir", type=str, required=True, help="Directory containing target fasta files")
# parser.add_argument("--reference_pdb", type=str, required=True, help="Reference pdb file")
# parser.add_argument("--copies", default=1, type=int, help="Number of copies of the sequence")
# parser.add_argument("--num_recycles", default=3, type=int, choices=[0, 1, 2, 3, 6, 12, 24], help="Number of recycles")
# parser.add_argument("--keep_pdb", action='store_true', help="Keep pdb files")
# parser.add_argument("--verbose", default=0, type=int, help="Verbose level")
# args = parser.parse_args()

# copies = args.copies
# num_recycles = args.num_recycles

# Load model
model_weights = "esmfold.model"
model_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_weights)
if not os.path.isfile(f'{model_name}'):
  print(f"Downloading {model_weights}...")
  os.system(f"aria2c -q -x 16 https://colabfold.steineggerlab.workers.dev/esm/{model_weights} -d {os.path.dirname(os.path.realpath(__file__))}")

def parse_output(output):
  pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
  plddt = output["plddt"][0,:,1]

  bins = np.append(0,np.linspace(2.3125,21.6875,63))
  sm_contacts = softmax(output["distogram_logits"],-1)[0]
  sm_contacts = sm_contacts[...,bins<8].sum(-1)
  xyz = output["positions"][-1,0,:,1]
  mask = output["atom37_atom_exists"][0,:,1] == 1
  o = {"pae":pae[mask,:][:,mask],
       "plddt":plddt[mask],
       "sm_contacts":sm_contacts[mask,:][:,mask],
       "xyz":xyz[mask]}
  return o

def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()
alphabet_list = list(ascii_uppercase+ascii_lowercase)

id_list, plddt_list, tm_list = [], [], []

def predict_structure(target_dir, reference_pdb, save_dir=None, copies=1, num_recycles=3, keep_pdb=False, verbose=0, collect_output=True, results=None):

  save_to_drive = save_dir
  if save_to_drive is not None:
    path_prefix = save_to_drive
  else:
    path_prefix = ""

  # target_dir = args.target_dir
  fasta_dir = glob(target_dir+"/*.fasta")
  assert len(fasta_dir) > 0, f"No fasta files found in {target_dir}"
  data_df = pd.DataFrame()
  for fasta_path in fasta_dir:
    batch = fasta_path.split('/')[-3]
    sampling = fasta_path.split('/')[-2]
    print(f'Batch: {batch}/{sampling}')
    name, seq = parse_fasta(fasta_path, return_names=True, clean="unalign")
    df = pd.DataFrame(list(zip(name, seq)), columns = ["name", "seq"])
    data_df = pd.concat([data_df, df], ignore_index=True)
    # data_df = data_df.head()
  print(f'Total sequences: {len(data_df)}')

  model = torch.load(f'{model_name}')
  model.eval().cuda().requires_grad_(False)
  model_name_ = model_name

  for _, row in tqdm.tqdm(data_df.iterrows(), total=len(data_df), desc=f"Predicting {batch}/{sampling}"):
    start_time = time.time()
    jobname = row['name']
    jobname = re.sub(r'\W+', '', jobname)[:50]

    sequence = row['seq']
    sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
    sequence = re.sub(":+",":",sequence)
    sequence = re.sub("^[:]+","",sequence)
    sequence = re.sub("[:]+$","",sequence)
    copies = 1 
    if copies == "" or copies <= 0: copies = 1
    sequence = ":".join([sequence] * copies)
    num_recycles = 3
    chain_linker = 25

    if save_to_drive is not None:
      ID = path_prefix+'/'+batch+'/'+sampling+'/'+jobname
    else:
      tmpfolder = tempfile.mkdtemp()
      ID = tmpfolder+'/'+batch+'/'+sampling+'/'+jobname
    destination_path = '/'.join(ID.split('/')[:-1])
    # print(f'Destination path: {destination_path}')

    seqs = sequence.split(":")
    lengths = [len(s) for s in seqs]
    length = sum(lengths)
    # print("length",length)

    u_seqs = list(set(seqs))
    if len(seqs) == 1: mode = "mono"
    elif len(u_seqs) == 1: mode = "homo"
    else: mode = "hetero"

    if "model" not in dir() or model_name != model_name_:
      if "model" in dir():
        # delete old model from memory
        del model
        gc.collect()
        if torch.cuda.is_available():
          torch.cuda.empty_cache()

    # optimized for Tesla T4
    if length > 700:
      model.set_chunk_size(64)
    else:
      model.set_chunk_size(128)

    torch.cuda.empty_cache()
    output = model.infer(sequence,
                        num_recycles=num_recycles,
                        chain_linker="X"*chain_linker,
                        residue_index_offset=512)

    pdb_str = model.output_to_pdb(output)[0]
    output = tree_map(lambda x: x.cpu().numpy(), output)
    ptm = output["ptm"][0]
    plddt = output["plddt"][0,...,1].mean()

    id_list.append(ID.split('/')[-1]) if not collect_output else None
    plddt_list.append(plddt) if not collect_output else add_metric(results, ID.split('/')[-1], "pLDDT", plddt)
    end_time = time.time() - start_time

    O = parse_output(output)

    # Saving pdbs
    os.system(f"mkdir -p {ID}")
    prefix = f"{ID}/ptm{ptm:.3f}_r{num_recycles}_default"
    np.savetxt(f"{prefix}.pae.txt",O["pae"],"%.3f")
    with open(f"{prefix}.pdb","w") as out:
      out.write(pdb_str)

    # TM Score
    # reference_pdb = args.reference_pdb
    pdb_file = f"{prefix}.pdb"
    tmscore = tmscoring.get_tm(reference_pdb, pdb_file)
    tm_list.append(tmscore) if not collect_output else add_metric(results, ID.split('/')[-1], "TM-score", tmscore)

    # delete pdb folder
    if not keep_pdb:
      shutil.rmtree(tmpfolder) if save_dir is None else shutil.rmtree(destination_path)

    print(f'ptm: {ptm:.3f} plddt: {plddt:.3f} tmscore: {tmscore:.3f} time: {end_time:.3f}') if verbose == 1 else None

  if not collect_output:
    results_df = pd.DataFrame(list(zip(id_list, plddt_list, tm_list)), columns = ["seq_name", "plddt", "tm_score"])
    if path_prefix is not None:
      os.system(f"mkdir -p {path_prefix}/{batch}/results")
      results_df.to_csv(f'{path_prefix}/{batch}/results/{sampling}.csv')
    else:
      os.system(f"mkdir -p {batch}/results")
      results_df.to_csv(f'{batch}/results/{sampling}.csv')

    mean_plddt = results_df['plddt'].mean()
    mean_tm = results_df['tm_score'].mean()
    print(f'avg plddt: {mean_plddt:.3f}; avg tm score: {mean_tm:.3f}')
    print(f'=========Batch: {batch}/{sampling} done!=============')