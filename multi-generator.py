import app
import argparse
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, AutoTokenizer, XLNetLMHeadModel, XLNetTokenizer
from tranception import config, model_pytorch
import tranception
import pandas as pd
import os
import util
import tensorflow as tf
from sampling import top_k_sampling, temperature_sampler, top_p_sampling, typical_sampling, mirostat_sampling, random_sampling
from proteinbert.model_generation import InputEncoder
import time
from EVmutation.model import CouplingsModel
from app import process_prompt_protxlnet

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=str, help='Sequence for mutation', required=True)
parser.add_argument('--seq_id', type=str, help='Sequence ID for mutation', required=True)
parser.add_argument('--mutation_start', type=int, default=None, help='Mutation start position')
parser.add_argument('--mutation_end', type=int, default=None, help='Mutation end position')
parser.add_argument('--model', type=str, choices=['small', 'medium', 'large'], help='Tranception model size')
parser.add_argument('--Tmodel', type=str, help='Tranception model path')
# parser.add_argument('--AMSmodel', type=str, help='Tranception model path for Attention-Matrix Sampling')
parser.add_argument('--model_name', type=str, choices=['Tranception', 'RITA', 'ProtXLNet'], help='Model name', required=True)
parser.add_argument('--use_scoring_mirror', action='store_true', help='Whether to score the sequence from both ends')
parser.add_argument('--batch', type=int, default=20, help='Batch size for scoring')
parser.add_argument('--max_pos', type=int, default=50, help='Maximum number of positions per heatmap')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
parser.add_argument('--with_heatmap', action='store_true', help='Whether to generate heatmap')
parser.add_argument('--mutations', type=int, default=2, help='Number of mutations to generate')
parser.add_argument('--save_scores', action='store_true', help='Whether to save scores')

parser.add_argument('--sampling_method', type=str, choices=['top_k', 'top_p', 'typical', 'mirostat', 'random', 'greedy'], required=True, help='Sampling method')
parser.add_argument('--sampling_threshold', type=float, help='Sampling threshold (k for top_k, p for top_p, tau for mirostat, etc.)')
parser.add_argument('--intermediate_threshold', type=int, required=True, help='Final Top-K threshold for intermediate sampling')
parser.add_argument('--use_qff', action='store_true', help='Whether to use Quantitative-Function Filtering')
parser.add_argument('--use_hpf', action='store_true', help='Whether to use High-Probability Filtering')
parser.add_argument('--use_rsf', action='store_true', help='Whether to use Random-Stratified Filtering')
parser.add_argument('--use_ams', action='store_true', help='Whether to use Attention-Matrix Sampling')
parser.add_argument('--proteinbert', action='store_true', help='Whether to use ProteinBERT for Quantitative-Function Filtering')
parser.add_argument('--evmutation', action='store_true', help='Whether to use EVmutation for Quantitative-Function Filtering')
parser.add_argument('--saved_model_dir', type=str, help='ProteinBERT saved model directory')
parser.add_argument('--evmutation_model_dir', type=str, help='EVmutation model directory')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for final sampling; 1.0 equals to random sampling')
parser.add_argument('--sequence_num', type=int, required=True, help='Number of sequences to generate')
parser.add_argument('--evolution_cycles', type=int, required=True, help='Number of evolution cycles per generated sequence')
parser.add_argument('--output_name', type=str, required=True, help='Output file name (Just name with no extension!)')
parser.add_argument('--save_df', action='store_true', help='Whether to save the dataframe')
parser.add_argument('--verbose', action='store_true', help='Verbose mode')
args = parser.parse_args()

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
tranception_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tranception/utils/tokenizers/Basic_tokenizer"),
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
    tokenizer = tranception_tokenizer
elif model_name == 'RITA':
    assert args.Tmodel, "Model path must be specified"
    tokenizer = AutoTokenizer.from_pretrained(args.Tmodel)
    model = AutoModelForCausalLM.from_pretrained(args.Tmodel, local_files_only=True, trust_remote_code=True)
elif model_name == 'ProtXLNet':
    tokenizer = XLNetTokenizer.from_pretrained(args.Tmodel)
    model = XLNetLMHeadModel.from_pretrained(args.Tmodel, mem_len=512)
else:
    raise ValueError(f"Model {model_name} not supported")


# example_sequence = {'MDH_A0A075B5H0': 'MTQRKKISLIGAGNIGGTLAHLIAQKELGDVVLFDIVEGMPQGKALDISHSSPIMGSNVKITGTNNYEDIKGSDVVIITAGIPRKPGKSDKEWSRDDLLSVNAKIMKDVAENIKKYCPNAFVIVVTNPLDVMVYVLHKYSGLPHNKVCGMAGVLDSSRFRYFLAEKLNVSPNDVQAMVIGGHGDTMVPLTRYCTVGGIPLTEFIKQGWITQEEIDEIVERTRNAGGEIVNLLKTGSAYFAPAASAIEMAESYLKDKKRILPCSAYLEGQYGVKDLFVGVPVIIGKNGVEKIIELELTEEEQEMFDKSVESVRELVETVKKLNALEHHHHHH',
#                     'MDH_A0A2V9QQ45': 'MRKKVTIVGSGNVGATAAQRIVDKELADVVLIDIIEGVPQGKGLDLLQSGPIEGYDSHVLGTNDYKDTANSDIVVITAGLPRRPGMSRDDLLIKNYEIVKGVTEQVVKYSPHSILIVVSNPLDAMVQTAFKISGFPKNRVIGMAGVLDSARFRTFIAMELNVSVENIHAFVLGGHGDTMVPLPRYSTVAGIPITELLPRERIDALVKRTRDGGAEIVGLLKTGSAYYAPSAATVEMVEAIFKDKKKILPCAAYLEGEYGISGSYVGVPVKLGKSGVEEIIQIKLTPEENAALKKSANAVKELVDIIKV',
#                     'avGFP': 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'}

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
subsamplings = []
mutants = []
samplingthreshold = []
subsamplingthreshold = []
past_key_values=None

if args.sampling_method in ['top_k', 'top_p', 'typical', 'mirostat']:
    assert args.sampling_threshold is not None, "Sampling threshold must be specified for top_k, top_p, and mirostat sampling methods"
assert args.intermediate_threshold <= 100, "Intermediate sampling threshold cannot be greater than 100!"

assert args.use_qff or args.use_hpf or args.use_ams or args.use_rsf, "Please specify at least one filter-sampling method!"
if args.use_qff:
    if args.proteinbert:
        assert args.saved_model_dir is not None, "Please specify the saved model directory for Quantitative Filter!"
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"{args.saved_model_dir}")
        
        input_encoder = InputEncoder(n_annotations=8943) # Check this number
        proteinbert_model = app.load_savedmodel(model_path=model_path)
        strat = "ProteinBERT"
    
    if args.evmutation:
        ev_dir = os.path.join(f"{args.evmutation_model_dir}")
        assert os.path.exists(ev_dir), f"Model directory {ev_dir} does not exist"
        ev_model = CouplingsModel(ev_dir)
        strat = "EVmutation"

    print(f"{strat} Quantitative-Function Filter will be used!")
if args.use_hpf:
    strat = "High-Probability Filter"
    print("High-Probability Filter will be used!")
if args.use_ams:
    ev_dir = os.path.join(f"{args.evmutation_model_dir}")
    assert os.path.exists(ev_dir), f"Model directory {ev_dir} does not exist"
    ev_model = CouplingsModel(ev_dir)
    strat = "Attention-Matrix Sampling"
    print("Attention-Matrix Sampling will be used!")
    # if args.model_name == 'RITA':
    #     tranception_model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.AMSmodel, local_files_only=True)
    # else:
    #     tranception_model = model
if args.use_rsf:
    ev_dir = os.path.join(f"{args.evmutation_model_dir}")
    assert os.path.exists(ev_dir), f"Model directory {ev_dir} does not exist"
    ev_model = CouplingsModel(ev_dir)
    strat = "Random-Stratified Filter"
    print("Random-Stratified Filter will be used!")

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

        mutation_count = 0
        while mutation_count < args.mutations:
            mutation_count += 1
            print(f"Mutation {mutation_count} of {args.mutations}") if args.verbose else None
            # First Mutation
            if mutation_count == 1:
                # 1. Generate and score suggested mutation
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
                                                                                            model_type=model_name,)

                # 2. Define intermediate sampling threshold
                final_sampler = temperature_sampler(args.temperature)
                intermediate_sampling_threshold = args.intermediate_threshold
                assert intermediate_sampling_threshold > 0, "Intermediate sampling threshold must be greater than 0!"
            
            # Subsequent Mutations
            if mutation_count > 1 and mutation_count < args.mutations:
                # 1. Generate extra mutations
                last_mutation_round_DMS = scores
                print(f"Generating 1 extra mutations after {len(last_mutation_round_DMS['mutant'][0].split(':'))} rounds to make {mutation_count} rounds in total") if args.verbose else None
                assert len(last_mutation_round_DMS['mutant'][0].split(':')) == mutation_count-1, "Mutation step not consistent with previous mutation round"
                
                # 2. Sample from extra mutations
                if args.use_qff:
                    mutation = top_k_sampling(last_mutation_round_DMS, k=int(100), sampler=final_sampler, multi=True)
                    all_extra_mutants = app.apply_gen_1extra(DMS=mutation)
                    if args.proteinbert:
                        extra_mutants = app.predict_proteinBERT(model=proteinbert_model, DMS=all_extra_mutants,input_encoder=input_encoder, top_n=intermediate_sampling_threshold, batch_size=128)
                    if args.evmutation:
                        extra_mutants = app.predict_evmutation(DMS=all_extra_mutants, top_n=intermediate_sampling_threshold, ev_model=ev_model)
                
                if args.use_hpf:
                    mutation = top_k_sampling(last_mutation_round_DMS, k=int(100), sampler=final_sampler, multi=True)
                    all_extra_mutants = app.apply_gen_1extra(DMS=mutation)
                    trimmed = app.trim_DMS(DMS_data=all_extra_mutants, sampled_mutants=mutation, mutation_rounds=mutation_count)
                    # _, scored_trimmed, trimmed, past_key_values = app.score_multi_mutations(seq,extra_mutants=all_extra_mutants,mutation_range_start=mutation_start, mutation_range_end=mutation_end, 
                    #                                         scoring_mirror=args.use_scoring_mirror, batch_size_inference=args.batch, 
                    #                                         max_number_positions_per_heatmap=args.max_pos, num_workers=args.num_workers, 
                    #                                         AA_vocab=AA_vocab, tokenizer=tokenizer, Tranception_model=model, past_key_values=past_key_values)
                    # extra_mutants = top_k_sampling(scored_trimmed, k=intermediate_sampling_threshold, sampler=final_sampler, multi=True)[['mutant', 'mutated_sequence']]
                    extra_mutants = trimmed.sample(n=intermediate_sampling_threshold)

                if args.use_rsf:
                    mutation = top_k_sampling(last_mutation_round_DMS, k=int(100), sampler=final_sampler, multi=True)
                    all_extra_mutants = app.apply_gen_1extra(DMS=mutation)
                    ev_scored = app.predict_evmutation(DMS=all_extra_mutants, top_n=len(all_extra_mutants), ev_model=ev_model, return_evscore=True)
                    try:
                        extra_mutants = app.stratified_filtering(ev_scored, threshold=intermediate_sampling_threshold, column_name='EVmutation')
                    except:
                        print("Retrying...") if args.verbose else None
                        ev_scored = app.predict_evmutation(DMS=all_extra_mutants, top_n=len(all_extra_mutants), ev_model=ev_model, return_evscore=True)
                        extra_mutants = app.stratified_filtering(ev_scored, threshold=intermediate_sampling_threshold, column_name='EVmutation')


                if args.use_ams:
                    mutation = top_k_sampling(last_mutation_round_DMS, k=int(100), sampler=final_sampler, multi=True)
                    att_mutations = app.get_attention_mutants(DMS=mutation, AMSmodel=model, focus='highest', top_n=5, tokenizer=tokenizer, model_type=model_name) #top_n is the number of attention positions to focus on
                    extra_mutants = app.predict_evmutation(DMS=att_mutations, top_n=intermediate_sampling_threshold, ev_model=ev_model)
                
                print(f"Using {len(extra_mutants)} variants for scoring") if args.verbose else None

                # 3. Get scores of sampled mutation
                suggested_mutation, scores, _, past_key_values = app.score_multi_mutations(seq,
                                                                        extra_mutants=extra_mutants,
                                                                        mutation_range_start=mutation_start, 
                                                                        mutation_range_end=mutation_end, 
                                                                        scoring_mirror=args.use_scoring_mirror, 
                                                                        batch_size_inference=args.batch, 
                                                                        max_number_positions_per_heatmap=args.max_pos, 
                                                                        num_workers=args.num_workers, 
                                                                        AA_vocab=AA_vocab, 
                                                                        tokenizer=tokenizer,
                                                                        Tranception_model=model,
                                                                        past_key_values=past_key_values,
                                                                        model_type=model_name,)

                # 4. Define intermediate sampling threshold
                final_sampler = temperature_sampler(args.temperature)
                intermediate_sampling_threshold = args.intermediate_threshold
                assert intermediate_sampling_threshold > 0, "Intermediate sampling threshold must be greater than 0!"

            # Last Mutation
            if mutation_count == args.mutations:
                # 1. Generate extra mutations
                last_mutation_round_DMS = scores
                print(f"Generating 1 extra mutations after {len(last_mutation_round_DMS['mutant'][0].split(':'))} rounds to make {mutation_count} rounds in total") if args.verbose else None
                assert len(last_mutation_round_DMS['mutant'][0].split(':')) == mutation_count-1, "Mutation step not consistent with previous mutation round"
                # 2. Sample from extra mutations
                if args.use_qff:
                    mutation = top_k_sampling(last_mutation_round_DMS, k=int(100), sampler=final_sampler, multi=True)
                    all_extra_mutants = app.apply_gen_1extra(DMS=mutation)
                    if args.proteinbert:
                        extra_mutants = app.predict_proteinBERT(model=proteinbert_model, DMS=all_extra_mutants,input_encoder=input_encoder, top_n=intermediate_sampling_threshold, batch_size=128)
                    if args.evmutation:
                        extra_mutants = app.predict_evmutation(DMS=all_extra_mutants, top_n=intermediate_sampling_threshold, ev_model=ev_model)
                
                if args.use_hpf:
                    mutation = top_k_sampling(last_mutation_round_DMS, k=int(100), sampler=final_sampler, multi=True)
                    all_extra_mutants = app.apply_gen_1extra(DMS=mutation)
                    trimmed = app.trim_DMS(DMS_data=all_extra_mutants, sampled_mutants=mutation, mutation_rounds=mutation_count)
                    # _, scored_trimmed, trimmed, past_key_values = app.score_multi_mutations(seq,extra_mutants=all_extra_mutants,mutation_range_start=mutation_start, mutation_range_end=mutation_end, 
                    #                                         scoring_mirror=args.use_scoring_mirror, batch_size_inference=args.batch, 
                    #                                         max_number_positions_per_heatmap=args.max_pos, num_workers=args.num_workers, 
                    #                                         AA_vocab=AA_vocab, tokenizer=tokenizer, Tranception_model=model, past_key_values=past_key_values)
                    # extra_mutants = top_k_sampling(scored_trimmed, k=intermediate_sampling_threshold, sampler=final_sampler, multi=True)[['mutant', 'mutated_sequence']]
                    extra_mutants = trimmed.sample(n=intermediate_sampling_threshold)

                if args.use_rsf:
                    mutation = top_k_sampling(last_mutation_round_DMS, k=int(100), sampler=final_sampler, multi=True)
                    all_extra_mutants = app.apply_gen_1extra(DMS=mutation)
                    ev_scored = app.predict_evmutation(DMS=all_extra_mutants, top_n=len(all_extra_mutants), ev_model=ev_model, return_evscore=True)
                    try:
                        extra_mutants = app.stratified_filtering(ev_scored, threshold=intermediate_sampling_threshold, column_name='EVmutation')
                    except:
                        print("Retrying...")
                        ev_scored = app.predict_evmutation(DMS=all_extra_mutants, top_n=len(all_extra_mutants), ev_model=ev_model, return_evscore=True)
                        extra_mutants = app.stratified_filtering(ev_scored, threshold=intermediate_sampling_threshold, column_name='EVmutation')

                if args.use_ams:
                    mutation = top_k_sampling(last_mutation_round_DMS, k=int(100), sampler=final_sampler, multi=True)
                    att_mutations = app.get_attention_mutants(DMS=mutation, AMSmodel=model, focus='highest', top_n=5, tokenizer=tokenizer, model_type=model_name) #top_n is the number of attention positions to focus on
                    extra_mutants = app.predict_evmutation(DMS=att_mutations, top_n=intermediate_sampling_threshold, ev_model=ev_model)
                
                print(f"Using {len(extra_mutants)} variants for scoring") if args.verbose else None

                # 3. Get scores of sampled mutation
                suggested_mutation, scores, _, past_key_values = app.score_multi_mutations(seq,
                                                                        extra_mutants=extra_mutants,
                                                                        mutation_range_start=mutation_start, 
                                                                        mutation_range_end=mutation_end, 
                                                                        scoring_mirror=args.use_scoring_mirror, 
                                                                        batch_size_inference=args.batch, 
                                                                        max_number_positions_per_heatmap=args.max_pos, 
                                                                        num_workers=args.num_workers, 
                                                                        AA_vocab=AA_vocab, 
                                                                        tokenizer=tokenizer,
                                                                        Tranception_model=model,
                                                                        past_key_values=past_key_values,
                                                                        model_type=model_name,)

                # 4. Final Sampling mutation from suggested mutation scores
                final_sampler = temperature_sampler(args.temperature)
                sampling_strat = args.sampling_method
                sampling_threshold = args.sampling_threshold

                if sampling_strat == 'top_k':
                    mutation = top_k_sampling(scores, k=int(sampling_threshold), sampler=final_sampler)
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
                print(f"Using {sampling_strat} as final sampling strategy with threshold {sampling_threshold}") if args.verbose else None

        # Get Mutated Sequence
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
    samplingthreshold.append(sampling_threshold)
    if args.use_qff:
        subsamplings.append(f'QFF {strat}')
    if args.use_hpf:
        subsamplings.append('HPF')
    if args.use_ams:
        subsamplings.append('AMS')
    if args.use_rsf:
        subsamplings.append('RSF')
    subsamplingthreshold.append(intermediate_sampling_threshold)
    mutants.append(mutation_count)
    seq_name = f'{model_name}_{sequence_id}_{iteration}x_{len(generated_sequence)}'
    generated_sequence_name.append(seq_name)
    mutation_list.append(';'.join(mutation_history))
    generation_time = time.time() - start_time
    generation_duration.append(generation_time)
    print(f"Sequence {len(generated_sequence)}/{sequence_num}: {generation_time} seconds using {strat} on {mutation_count} multi-mutants and {iteration} evolution cycles")
    print("=========================================") if args.verbose else None
    
print(f'===========Mutated {len(generated_sequence)} sequences in {sum(generation_duration)} seconds============')
generated_sequence_df = pd.DataFrame({'name': generated_sequence_name,'sequence': generated_sequence, 'sampling': samplings, 'threshold': samplingthreshold, 'subsampling':subsamplings, 'subthreshold': subsamplingthreshold, 'iterations': sequence_iteration, 'mutants': mutants, 'mutations': mutation_list, 'time': generation_duration})

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