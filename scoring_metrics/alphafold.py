import os
import re
import hashlib
import random
from glob import glob
import json
import time
import shutil
import tempfile
import tmscoring
import pandas as pd
from pgen.utils import parse_fasta
from .util import add_metric
import tqdm


from sys import version_info

import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from Bio import BiopythonDeprecationWarning
warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)
from pathlib import Path
from colabfold.download import download_alphafold_params, default_data_dir
from colabfold.utils import setup_logging
from colabfold.batch import get_queries, run, set_model_type
from colabfold.plot import plot_msa_v2

import os
import numpy as np

from colabfold.colabfold import plot_protein
from pathlib import Path
import matplotlib.pyplot as plt


def add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]


def predict_AFstructure(target_dir, reference_pdb, binder_sequence=None, save_dir=None, num_recycles=3, keep_pdb=False, verbose=0, results=None):

    # target_dir = args.target_dir
    fasta_dir = glob(target_dir+"/*.fasta")
    assert len(fasta_dir) > 0, f"No fasta files found in {target_dir}"
    data_df = pd.DataFrame()
    for fasta_path in fasta_dir:
        batch = fasta_path.split('/')[-3]
        sampling = fasta_path.split('/')[-2]
        # print(f'Batch: {batch}/{sampling}')
        name, seq = parse_fasta(fasta_path, return_names=True, clean="unalign")
        df = pd.DataFrame(list(zip(name, seq)), columns = ["name", "seq"])
        data_df = pd.concat([data_df, df], ignore_index=True)
        # data_df = data_df.head()
    print(f'Total sequences: {len(data_df)}')
    
    model_type = "auto"
    num_recycles = num_recycles
    recycle_early_stop_tolerance = "auto"
    relax_max_iterations = 200
    pairing_strategy = "greedy"
    calc_extra_ptm = True
    save_all = False 
    save_recycles = False 
    dpi = 200 
    display_images = False 
    data_path = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'alphafold_data'))

    for _, row in tqdm.tqdm(data_df.iterrows(), total=len(data_df), desc=f"Predicting {batch}/{sampling}"):
        start_time = time.time()

        query_sequence = f"{row['seq']}: {binder_sequence}" if binder_sequence is not None else row['seq']
        #@markdown  - Use `:` to specify inter-protein chainbreaks for **modeling complexes** (supports homo- and hetro-oligomers). For example **PI...SK:PI...SK** for a homodimer
        jobname = row['name']
        jobname = re.sub(r'\W+', '', jobname)[:50]
        # number of models to use
        num_relax = 0
        template_mode = "none"
        use_amber = False

        # remove whitespaces
        query_sequence = "".join(query_sequence.split())

        tmpfolder = tempfile.mkdtemp()
        IDname = tmpfolder+'/'+batch+'/'+sampling+'/'+jobname
        destination_path = '/'.join(IDname.split('/')[:-1])

        # make directory to save results
        os.makedirs(IDname, exist_ok=True)

        # save queries
        queries_path = os.path.join(destination_path, f"{jobname}.csv")
        with open(queries_path, "w") as text_file:
            text_file.write(f"id,sequence\n{jobname},{query_sequence}")

        custom_template_path = None
        use_templates = False

        msa_mode = "mmseqs2_uniref_env"
        pair_mode = "unpaired_paired" 

        a3m_file = os.path.join(destination_path,f"{jobname}.a3m")

        max_msa = "auto" 
        num_seeds = 1 
        use_dropout = False 

        num_recycles = None if num_recycles == "auto" else int(num_recycles)
        recycle_early_stop_tolerance = None if (recycle_early_stop_tolerance == "auto") or (recycle_early_stop_tolerance is None) else float(recycle_early_stop_tolerance)
        if max_msa == "auto": max_msa = None

        result_dir = destination_path
        log_filename = os.path.join(destination_path,"log.txt")
        setup_logging(Path(log_filename))

        queries, is_complex = get_queries(queries_path)
        model_type = set_model_type(is_complex, model_type)

        if "multimer" in model_type and max_msa is not None:
            use_cluster_profile = False
        else:
            use_cluster_profile = True
        
        download_alphafold_params(model_type, data_path)
        AFresults = run(
            queries=queries,
            result_dir=result_dir,
            use_templates=use_templates,
            custom_template_path=custom_template_path,
            num_relax=num_relax,
            msa_mode=msa_mode,
            model_type=model_type,
            num_models=5,
            num_recycles=num_recycles,
            relax_max_iterations=relax_max_iterations,
            recycle_early_stop_tolerance=recycle_early_stop_tolerance,
            num_seeds=num_seeds,
            use_dropout=use_dropout,
            model_order=[1,2,3,4,5],
            is_complex=is_complex,
            data_dir=data_path,
            keep_existing_results=False,
            rank_by="auto",
            pair_mode=pair_mode,
            pairing_strategy=pairing_strategy,
            stop_at_score=float(100),
            prediction_callback=None,
            dpi=dpi,
            zip_results=False,
            save_all=save_all,
            max_seq=1000,
            use_cluster_profile=use_cluster_profile,
            input_features_callback=None,
            save_recycles=save_recycles,
            user_agent="colabfold/google-colab-main",
            calc_extra_ptm=calc_extra_ptm,
        )
        end_time = time.time() - start_time
        # results_zip = f"{jobname}.result.zip"
        # os.system(f"zip -r {results_zip} {jobname}")

        # print(glob(f"{destination_path}/*"))
        # print(jobname)
        # print(AFresults['rank'])

        jsonname = f"{destination_path}/{jobname}_scores_{AFresults['rank'][0][0]}.json"
        pdbname = f"{destination_path}/{jobname}_unrelaxed_{AFresults['rank'][0][0]}.pdb"

        with open(jsonname, 'r') as file:
            data = json.load(file)

        pae = data['pae']
        plddt = data['plddt']
        ptm = data['ptm']
        iptm = data['iptm']

        mean_pae = np.mean(pae)
        mean_plddt = np.mean(plddt)
        mean_ptm = np.mean(ptm)
        mean_iptm = np.mean(iptm)

        tmscore = tmscoring.get_tm(reference_pdb, pdbname)

        print(f'pae: {mean_pae:.3f} ptm: {mean_ptm:.3f} plddt: {mean_plddt:.3f} tmscore: {tmscore:.3f} time: {end_time:.3f}') if verbose == 1 else None
        
        add_metric(results, jobname, "AF_TM-score", tmscore)
        add_metric(results, jobname, "AF_pLDDT", mean_plddt)
        add_metric(results, jobname, "AF_pTM", mean_ptm)
        add_metric(results, jobname, "AF_iPTM", mean_iptm)
        add_metric(results, jobname, "AF_PAE", mean_pae)

        if not keep_pdb:
            shutil.rmtree(tmpfolder)