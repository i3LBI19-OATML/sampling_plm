import app
import argparse
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, AutoTokenizer, XLNetLMHeadModel, XLNetTokenizer
from tranception import config, model_pytorch
import tranception
import pandas as pd
import os
import util
from sampling import top_k_sampling, temperature_sampler, top_p_sampling, typical_sampling, mirostat_sampling, random_sampling, beam_search
import time
import MCTS
from EVmutation.model import CouplingsModel
from app import process_prompt_protxlnet

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=str, help='Sequence to do mutation or DE', required=True)
parser.add_argument('--seq_id', type=str, help='Sequence ID for mutation', required=True)
parser.add_argument('--mutation_start', type=int, default=None, help='Mutation start position')
parser.add_argument('--mutation_end', type=int, default=None, help='Mutation end position')
parser.add_argument('--model', type=str, choices=['small', 'medium', 'large'], help='Tranception model size')
parser.add_argument('--Tmodel', type=str, help='Tranception model path')
parser.add_argument('--model_name', type=str, choices=['Tranception', 'RITA', 'ProtXLNet'], help='Model name', required=True)
parser.add_argument('--use_scoring_mirror', action='store_true', help='Whether to score the sequence from both ends')
parser.add_argument('--batch', type=int, default=20, help='Batch size for scoring')
parser.add_argument('--max_pos', type=int, default=50, help='Maximum number of positions per heatmap')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
parser.add_argument('--with_heatmap', action='store_true', help='Whether to generate heatmap')
parser.add_argument('--save_scores', action='store_true', help='Whether to save scores')

parser.add_argument('--sampling_method', type=str, choices=['top_k', 'top_p', 'typical', 'mirostat', 'random', 'greedy', 'beam_search', 'mcts'], required=True, help='Sampling method')
parser.add_argument('--sampling_threshold', type=float, help='Sampling threshold (k for top_k, p for top_p, tau for mirostat, etc.)')
parser.add_argument('--evmutation_model_dir', type=str, help='Path to EVmutation model directory')
parser.add_argument('--filter', type=str, choices=['hpf', 'qff', 'ams'], help='Filter to use for MCTS/Beam Search')
parser.add_argument('--intermediate_threshold', type=int, default=96, help='Intermediate threshold for MCTS/Beam Search')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for final sampling; 1.0 equals to random sampling')
parser.add_argument('--sequence_num', type=int, required=True, help='Number of sequences to generate')
parser.add_argument('--max_length', type=int, help='Number of search levels in beam search or MCTS')
parser.add_argument('--evolution_cycles', type=int, required=True, help='Number of evolution cycles per generated sequence')
parser.add_argument('--output_name', type=str, required=True, help='Output file name (Just name with no extension!)')
parser.add_argument('--save_df', action='store_true', help='Whether to save the dataframe')
parser.add_argument('--verbose', action='store_true', help='Whether to print verbose output')
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
elif model_name == 'ProtXLNet':
    tokenizer = XLNetTokenizer.from_pretrained(args.Tmodel)
    model = XLNetLMHeadModel.from_pretrained(args.Tmodel, mem_len=512)
else:
    raise ValueError(f"Model {model_name} not supported")
    
mutation_start = args.mutation_start
mutation_end = args.mutation_end
sequence_num = args.sequence_num
evolution_cycles = args.evolution_cycles
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
    assert args.sampling_threshold is not None, "Sampling threshold must be specified for top_k, top_p, and mirostat sampling methods"
if args.sampling_method == 'beam_search' or args.sampling_method == 'mcts':
    assert args.max_length is not None, "Maximum length must be specified for beam_search or MCTS sampling method"
    if args.filter == 'qff' or args.filter == 'ams':
        ev_dir = os.path.join(f"{args.evmutation_model_dir}")
        assert os.path.exists(ev_dir), f"Model directory {ev_dir} does not exist"
        ev_model = CouplingsModel(ev_dir)
    else:
        ev_model = None

while len(generated_sequence) < sequence_num:

    iteration = 0
    seq = args.sequence if model_name != 'ProtXLNet' else process_prompt_protxlnet(args.sequence)
    sequence_id = args.seq_id
    start_time = time.time()
    mutation_history = []

    while iteration < evolution_cycles:
        if args.verbose:
            print(f"Sequence {len(generated_sequence) + 1} of {sequence_num}, Iteration {iteration + 1} of {evolution_cycles}")
            print("=========================================")

        if args.sampling_method == 'mcts':
            mutation, past_key_values = MCTS.UCT_search(seq, max_length=args.max_length, extra=1, tokenizer=tokenizer, AA_vocab=AA_vocab, Tmodel=model, past_key_values=past_key_values, filter=args.filter, ev_model=ev_model, intermediate_sampling_threshold=args.intermediate_threshold)
            sampling_strat = args.sampling_method
            sampling_threshold = args.max_length
        else:
            # 1. Get scores of suggested mutation
            score_heatmap, suggested_mutation, scores, _, past_key_values = app.score_and_create_matrix_all_singles(seq, Tranception_model=model, 
                                                                                        mutation_range_start=mutation_start, mutation_range_end=mutation_end, 
                                                                                        scoring_mirror=args.use_scoring_mirror, 
                                                                                        batch_size_inference=args.batch, 
                                                                                        max_number_positions_per_heatmap=args.max_pos, 
                                                                                        num_workers=args.num_workers, 
                                                                                        AA_vocab=AA_vocab, 
                                                                                        tokenizer=tokenizer,
                                                                                        with_heatmap=args.with_heatmap,
                                                                                        past_key_values=past_key_values,
                                                                                        model_type=model_name)

            # Save heatmap
            if args.with_heatmap and args.save_scores:
                save_path_heatmap = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output_heatmap.csv")
                pd.DataFrame(score_heatmap, columns =['score_heatmap']).to_csv(save_path_heatmap)
                print(f"Results saved to {save_path_heatmap}")

            # Save scores
            if args.save_scores:
                save_path_scores = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output_scores.csv")
                scores.to_csv(save_path_scores)
                print(f"Scores saved to {save_path_scores}")

            # 2. Sample mutation from suggested mutation scores
            final_sampler = temperature_sampler(args.temperature)
            sampling_strat = args.sampling_method
            sampling_threshold = args.sampling_threshold

            if sampling_strat == 'top_k':
                mutation = top_k_sampling(scores, k=int(sampling_threshold), sampler=final_sampler)
            elif sampling_strat == 'beam_search':
                mutation, past_key_values = beam_search(scores, beam_width=int(sampling_threshold), max_length=args.max_length, tokenizer=tokenizer, sampler=final_sampler, Tmodel=model, past_key_values=past_key_values, filter=args.filter, ev_model=ev_model, IST=args.intermediate_threshold)
            elif sampling_strat == 'top_p':
                assert float(sampling_threshold) <= 1.0 and float(sampling_threshold) > 0, "Top-p sampling threshold must be between 0 and 1"
                mutation = top_p_sampling(scores, p=float(sampling_threshold), sampler=final_sampler)
            elif sampling_strat == 'typical':
                assert float(sampling_threshold) < 1.0 and float(sampling_threshold) > 0, "Typical sampling threshold must be between 0 and 1"
                mutation = typical_sampling(scores, mass=float(sampling_threshold), sampler=final_sampler)
            elif sampling_strat == 'mirostat':
                mutation = mirostat_sampling(scores, tau=float(sampling_threshold), sampler=final_sampler)
            elif sampling_strat == 'random':
                mutation = random_sampling(scores, sampler=final_sampler)
            elif sampling_strat == 'greedy':
                mutation = top_k_sampling(scores, k=1, sampler=final_sampler)
            else:
                raise ValueError(f"Sampling strategy {sampling_strat} not supported")
            print(f"Using {sampling_strat} sampling strategy with threshold {sampling_threshold}") if args.verbose else None

        # 3. Get Mutated Sequence
        mutated_sequence = app.get_mutated_protein(seq, mutation)
        mutation_history += [mutation]

        if args.verbose:
            print("Original Sequence: ", seq)
            print("Mutation: ", mutation)
            print("Mutated Sequence: ", mutated_sequence)
            print("=========================================")

        seq = mutated_sequence

        iteration += 1

    generated_sequence.append(mutated_sequence)
    sequence_iteration.append(iteration)
    samplings.append(sampling_strat)
    samplingtheshold.append(sampling_threshold) 
    seq_name = '{}_{}_{}x_{}'.format(model_name, sequence_id, iteration, len(generated_sequence))
    generated_sequence_name.append(seq_name)
    mutants.append('1')
    subsamplings.append('NA')
    subsamplingtheshold.append('NA')
    mutation_list.append(';'.join(mutation_history))
    generation_time = time.time() - start_time
    generation_duration.append(generation_time)
    print(f"Sequence {len(generated_sequence)}/{sequence_num}: {generation_time} seconds with {iteration} evolution cycles")
    print("=========================================") if args.verbose else None
    
print(f'===========Mutated {len(generated_sequence)} sequences in {sum(generation_duration)} seconds============')
generated_sequence_df = pd.DataFrame({'name': generated_sequence_name,'sequence': generated_sequence, 'sampling': samplings, 'threshold': samplingtheshold, 'subsampling':subsamplings, 'subthreshold': subsamplingtheshold, 'iterations': sequence_iteration, 'mutants': mutants, 'mutations': mutation_list, 'time': generation_duration})

if args.save_df:
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "generated_metadata/{}.csv".format(args.output_name))
    os.makedirs(os.path.dirname(os.path.realpath(save_path))) if not os.path.exists(os.path.dirname(os.path.realpath(save_path))) else None
    if os.path.exists(save_path):
        generated_sequence_df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        generated_sequence_df.to_csv(save_path, index=False)
    print(f"Generated sequences saved to {save_path}")

save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "generated_sequence/{}.fasta".format(args.output_name))
os.makedirs(os.path.dirname(os.path.realpath(save_path))) if not os.path.exists(os.path.dirname(os.path.realpath(save_path))) else None
util.save_as_fasta(generated_sequence_df, save_path)
print(f"Generated sequences saved to {save_path}")
