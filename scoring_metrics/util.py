from biotite.structure import get_residue_count
from biotite.structure.residues import get_residues
from biotite.structure.io import pdb
from biotite.sequence import ProteinSequence
import numpy as np

def add_metric(metrics_dict, protein_name, metric_name, value):
  if protein_name not in metrics_dict:
    metrics_dict[protein_name] = dict()
  metrics_dict[protein_name][metric_name] = value

def add_sequence(metrics_dict, protein_name, sequence):
  if protein_name not in metrics_dict:
    metrics_dict[protein_name] = dict()
  metrics_dict[protein_name]['sequence'] = sequence

def get_pdb_sequence(pdb_path):
    with open(pdb_path) as f:
        pdb_file = pdb.PDBFile.read(pdb_path)
        atoms  = pdb_file.get_structure()
        residues = get_residues(atoms)[1]
    return ''.join([ProteinSequence.convert_letter_3to1(r) for r in residues])

def residues_in_pdb(pdb_path):
    with open(pdb_path) as f:
        pdb_file = pdb.PDBFile.read(pdb_path)
        atoms  = pdb_file.get_structure()
    return get_residue_count(atoms)

def identify_mutation(reference, target, sep=','):
    str1vec = list(reference)
    str2vec = list(target)
    assert len(str1vec) == len(str2vec), 'Sequence lengths must be equal'
    iMut = [i for i in range(len(str1vec)) if str1vec[i] != str2vec[i]]
    if len(iMut) == 0:
        return np.nan
    else:
        return f'{sep}'.join([str1vec[i] + str(i+1) + str2vec[i] for i in iMut])

def extract_mutations(mutation_string, offset=0):
  """
  Turns a string containing mutations of the format I100V into a list of tuples with
  format (100, 'I', 'V') (index, from, to)
  Parameters
  ----------
  mutation_string : str
      Comma-separated list of one or more mutations (e.g. "K50R,I100V")
  offset : int, default: 0
      Offset to be added to the index/position of each mutation
  Returns
  -------
  list of tuples
      List of tuples of the form (index+offset, from, to)
  """
  # if mutation_string is np.nan, consider as wild type
  mutation_string = "" if mutation_string is np.nan else mutation_string
  if mutation_string.lower() not in ["wild", "wt", ""]:
      mutations = mutation_string.split(",")
      return list(map(
          lambda x: (int(x[1:-1]) + offset, x[0], x[-1]),
          mutations
      ))
  else:
      return []