import app
import argparse
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoModelForCausalLM
from tranception import config, model_pytorch
import tranception
import pandas as pd
import os
import util
from AR_sampling import ARtop_k_sampling, ARtemperature_sampler, ARtop_p_sampling, ARtypical_sampling, ARmirostat_sampling, ARrandom_sampling, ARbeam_search
import time
import AR_MCTS
from EVmutation.model import CouplingsModel
from tqdm.auto import tqdm
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=str, help='Sequence to do mutation or DE')
parser.add_argument('--model', type=str, choices=['small', 'medium', 'large'], help='Tranception model size')
parser.add_argument('--Tmodel', type=str, help='Tranception model path')
parser.add_argument('--model_name', type=str, choices=['Tranception', 'RITA'], required=True, help='Model name')
parser.add_argument('--use_scoring_mirror', action='store_true', help='Whether to score the sequence from both ends')
parser.add_argument('--batch', type=int, default=20, help='Batch size for scoring')
parser.add_argument('--max_pos', type=int, default=50, help='Maximum number of positions per heatmap')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for dataloader')
parser.add_argument('--with_heatmap', action='store_true', help='Whether to generate heatmap')
parser.add_argument('--save_scores', action='store_true', help='Whether to save scores')

parser.add_argument('--sampling_method', type=str, choices=['top_k', 'top_p', 'typical', 'mirostat', 'random', 'greedy', 'beam_search', 'mcts'], required=True, help='Sampling method')
parser.add_argument('--sampling_threshold', type=float, help='Sampling threshold (k for top_k, p for top_p, tau for mirostat, beam_width in beam_search, etc.)')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for final sampling; 1.0 equals to random sampling')
parser.add_argument('--filter', type=str, choices=['hpf'], help='Filter to use for MCTS/Beam Search')
parser.add_argument('--intermediate_threshold', type=int, default=96, help='Intermediate threshold for MCTS/Beam Search')
parser.add_argument('--evmutation_model_dir', type=str, help='Path to EVmutation model directory')
parser.add_argument('--sequence_num', type=int, required=True, help='Number of sequences to generate')
parser.add_argument('--seq_length', type=int, required=True, help='Length of each sequence to generate')
parser.add_argument('--max_length', type=int, help='Number of search levels in beam search or MCTS')
parser.add_argument('--extension_factor', type=int, default=1, help='Number of AAs to add to extend the sequence in each round')
parser.add_argument('--output_name', type=str, required=True, help='Output file name (Just name with no extension!)')
parser.add_argument('--save_df', action='store_true', help='Whether to save the metadata dataframe')
parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')
args = parser.parse_args()

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tranception/utils/tokenizers/Basic_tokenizer"),
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )

# Load model
model_name = args.model_name
if model_name == 'Tranception':
    assert args.model or args.Tmodel, "Either model size or model path must be specified"
    model_type = args.model.capitalize() if args.model else None
    try:
        model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.Tmodel, local_files_only=True)
        print("Model successfully loaded from local")
    except:
        print("Model not found locally, downloading from HuggingFace")
        if model_type=="Small":
            model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Small")
        elif model_type=="Medium":
            model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Medium")
        elif model_type=="Large":
            model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Large")
elif model_name == 'RITA':
    assert args.Tmodel, "Model path must be specified"
    tokenizer = AutoTokenizer.from_pretrained(args.Tmodel)
    model = AutoModelForCausalLM.from_pretrained(args.Tmodel, local_files_only=True, trust_remote_code=True)
else:
    raise ValueError(f"Model {model_name} not supported")

if args.sampling_method == 'beam_search' or args.sampling_method == 'mcts':
    assert args.max_length is not None, "Maximum length must be specified for beam_search or MCTS sampling method"

sequence_num = args.sequence_num
seq_length = args.seq_length
AA_extension = args.extension_factor
generated_sequence = []
sequence_iteration = []
generated_sequence_name = []
mutation_list = []
generation_duration = []
samplings = []
mutants = []
subsamplings = []
samplingtheshold = []
subsamplingtheshold = []
past_key_values=None

if args.sampling_method in ['top_k', 'top_p', 'typical', 'mirostat', 'beam_search']:
    assert args.sampling_threshold is not None, "Sampling threshold must be specified for top_k, top_p, typical, mirostat, and beam_search sampling methods"
if args.sampling_method == 'beam_search' or args.sampling_method == 'mcts':
    assert args.max_length is not None, "Maximum length must be specified for beam_search or MCTS sampling method"

pbar1 = tqdm(total=sequence_num, desc="Generating", position=0, leave=True)
while len(generated_sequence) < sequence_num:

    if not args.sequence: seq = random.choice(AA_vocab)
    else: seq = args.sequence.upper()
    sequence_length = len(seq)
    start_time = time.time()
    mutation_history = []
    iteration = 0
    
    pbar1.set_description(f"Generating {len(generated_sequence) + 1}/{sequence_num}")
    while sequence_length < seq_length:
        pbar1.set_postfix_str(f"Length: {sequence_length}/{seq_length}")
        if args.verbose == 1:
            print(f"Sequence {len(generated_sequence) + 1} of {sequence_num}, Length {sequence_length} of {seq_length}")
            print("=========================================")

        if args.sampling_method == 'mcts':
            sampling_strat = args.sampling_method
            sampling_threshold = args.max_length
            mutation, past_key_values = AR_MCTS.UCT_search(seq, max_length=args.max_length, tokenizer=tokenizer, AA_vocab=AA_vocab, extension_factor=AA_extension, Tmodel=model, past_key_values=past_key_values, filter=args.filter, intermediate_sampling_threshold=args.intermediate_threshold, batch=args.batch)
            # print("MCTS mutation: ", mutation)
        
        else:
            # Generate possible mutations
            extended_seq = app.extend_sequence_by_n(seq, AA_extension, AA_vocab, output_sequence=True)

            # Score using Tranception (app.score_multi_mutations works for scoring AR sequences)
            scores, _, past_key_values = app.score_multi_mutations(sequence=None,
                                                        extra_mutants=extended_seq,
                                                        mutation_range_start=None, 
                                                        mutation_range_end=None, 
                                                        scoring_mirror=args.use_scoring_mirror, 
                                                        batch_size_inference=args.batch, 
                                                        max_number_positions_per_heatmap=args.max_pos, 
                                                        num_workers=args.num_workers, 
                                                        AA_vocab=AA_vocab, 
                                                        tokenizer=tokenizer,
                                                        AR_mode=True,
                                                        Tranception_model=model,
                                                        past_key_values=past_key_values,
                                                        verbose=args.verbose,
                                                        model_type=model_name)

            # Save scores
            if args.save_scores:
                save_path_scores = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output_scores.csv")
                scores.to_csv(save_path_scores)
                print(f"Scores saved to {save_path_scores}")

            # 2. Sample mutation from scores
            final_sampler = ARtemperature_sampler(args.temperature)
            sampling_strat = args.sampling_method
            sampling_threshold = args.sampling_threshold

            if sampling_strat == 'top_k':
                assert int(sampling_threshold) <= len(scores), "Top-k sampling threshold must be less than or equal to the number of mutations ({}).".format(len(scores))
                mutation = ARtop_k_sampling(scores, k=int(sampling_threshold), sampler=final_sampler)
            elif sampling_strat == 'beam_search':
                assert args.max_length < seq_length, "Maximum length must be less than the length of the final sequence"
                # print(f'IST ar_gen 146: {args.intermediate_threshold}')
                mutation, past_key_values = ARbeam_search(scores, beam_width=int(sampling_threshold), max_length=args.max_length, tokenizer=tokenizer, sampler=final_sampler, Tmodel=model, batch=args.batch, past_key_values=past_key_values, extension_factor=AA_extension, filter=args.filter, IST=args.intermediate_threshold)
            elif sampling_strat == 'top_p':
                assert float(sampling_threshold) <= 1.0 and float(sampling_threshold) > 0, "Top-p sampling threshold must be between 0 and 1"
                mutation = ARtop_p_sampling(scores, p=float(sampling_threshold), sampler=final_sampler)
            elif sampling_strat == 'typical':
                assert float(sampling_threshold) < 1.0 and float(sampling_threshold) > 0, "Typical sampling threshold must be between 0 and 1"
                mutation = ARtypical_sampling(scores, mass=float(sampling_threshold), sampler=final_sampler)
            elif sampling_strat == 'mirostat':
                mutation = ARmirostat_sampling(scores, tau=float(sampling_threshold), sampler=final_sampler)
            elif sampling_strat == 'random':
                mutation = ARrandom_sampling(scores, sampler=final_sampler)
            elif sampling_strat == 'greedy':
                mutation = ARtop_k_sampling(scores, k=1, sampler=final_sampler)
            else:
                raise ValueError(f"Sampling strategy {sampling_strat} not supported")
            print(f"Using {sampling_strat} sampling strategy with threshold {sampling_threshold}") if args.verbose == 1 else None
            # print("Sampled mutation: ", mutation)

        # 3. Get Mutated Sequence
        mutated_sequence = mutation
        # mutated_sequence = app.get_mutated_protein(seq, mutation) # TODO: Check if this is correct
        if len(mutated_sequence) > seq_length:
            mutated_sequence = mutated_sequence[:seq_length]
        mutation_history += [mutated_sequence.replace(seq, '')]

        if args.verbose == 1:
            print("Original Sequence: ", seq)
            # print("Mutation: ", mutation)
            print("Mutated Sequence: ", mutated_sequence)
            print("=========================================")

        seq = mutated_sequence
        sequence_length = len(seq)
        iteration += 1

    generated_sequence.append(mutated_sequence)
    sequence_iteration.append(iteration)
    samplings.append(sampling_strat)
    samplingtheshold.append(sampling_threshold) 
    seq_name = 'AR{}_{}AA_{}'.format(model_name, iteration+1, len(generated_sequence))
    generated_sequence_name.append(seq_name)
    mutants.append('1')
    subsamplings.append('NA')
    subsamplingtheshold.append('NA')
    mutation_list.append(''.join(mutation_history))
    generation_time = time.time() - start_time
    generation_duration.append(generation_time)
    pbar1.write(f"Sequence {len(generated_sequence)}/{sequence_num}: {generation_time} seconds")
    # print("=========================================") if args.verbose == 1 else None
    pbar1.update(1)

pbar1.close()   
print(f'===========Generated {len(generated_sequence)} sequences of length {seq_length} in {sum(generation_duration)} seconds============')
generated_sequence_df = pd.DataFrame({'name': generated_sequence_name,'sequence': generated_sequence, 'sampling': samplings, 'threshold': samplingtheshold, 'subsampling':subsamplings, 'subthreshold': subsamplingtheshold, 'iterations': sequence_iteration, 'mutants': mutants, 'mutations': mutation_list, 'time': generation_duration})

if args.save_df:
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ARgenerated_metadata/{}.csv".format(args.output_name))
    os.makedirs(os.path.dirname(os.path.realpath(save_path))) if not os.path.exists(os.path.dirname(os.path.realpath(save_path))) else None
    if os.path.exists(save_path):
        generated_sequence_df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        generated_sequence_df.to_csv(save_path, index=False)
    print(f"Generated sequences saved to {save_path}")

save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ARgenerated_sequence/{}.fasta".format(args.output_name))
os.makedirs(os.path.dirname(os.path.realpath(save_path))) if not os.path.exists(os.path.dirname(os.path.realpath(save_path))) else None
util.save_as_fasta(generated_sequence_df, save_path)
print(f"Generated sequences saved to {save_path}")