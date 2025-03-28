# import os
# import argparse
import tqdm 

# from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
# from scipy.stats import spearmanr
import numpy as np
# import pandas as pd

import torch
from torch.nn import CrossEntropyLoss

def calc_fitness(model, prots, tokenizer, device='cuda:0', model_context_len=1023):
    loss_list = []
    loss_fn = CrossEntropyLoss()
    with torch.no_grad():
        for prot in tqdm.tqdm(prots, position=0, leave=True):
            loss_val = 0
            
            sequence_chunks=[]
            if len(prot) < model_context_len:
                sequence_chunks = [prot]
            else:
                len_target_seq = len(prot)
                num_windows = 1 + int( len_target_seq / model_context_len)
                start=0
                for window_index in range(1, num_windows+1):
                    sequence_chunks.append(prot[start:start+model_context_len])
                    start += model_context_len
            
            for chunk in sequence_chunks:
                for p in [chunk, chunk[::-1]]:
                    ids = torch.tensor([tokenizer.encode(p)]).to(device)
                    input_ids = ids[:, :-1]
                    targets   = ids[:, 1:]
                    
                    logits=model(input_ids).logits
                    loss = loss_fn(target=targets.view(-1), input=logits.view(-1,logits.size(-1)))
                    loss_val += -loss.item()
                
            loss_list += [loss_val]
    return np.array(loss_list)

def get_mutated_sequence(focus_seq, mutant, start_idx=1, AA_vocab="ACDEFGHIKLMNPQRSTVWY"):
    """
    Helper function that mutates an input sequence (focus_seq) via an input mutation triplet (substitutions only).
    Mutation triplet are typically based on 1-indexing: start_idx is used for switching to 0-indexing.
    """
    mutated_seq = list(focus_seq)
    for mutation in mutant.split(":"):
        try:
            from_AA, position, to_AA = mutation[0], int(mutation[1:-1]), mutation[-1]
        except:
            print("Issue with mutant: "+str(mutation))
        relative_position = position - start_idx
        assert (from_AA==focus_seq[relative_position]), "Invalid from_AA or mutant position: "+str(mutation)+" from_AA: "+str(from_AA) + " relative pos: "+str(relative_position) + " focus_seq: "+str(focus_seq)
        assert (to_AA in AA_vocab) , "Mutant to_AA is invalid: "+str(mutation)
        mutated_seq[relative_position] = to_AA
    return "".join(mutated_seq)