# agreement_surprisal.py
# Author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

import os
import torch
import pandas as pd
import tqdm
import argparse
from numpy.random import default_rng
from models.modeling_gpt2 import GPT2LMHeadModel
from itertools import zip_longest, takewhile
from glob import glob
from utils import (
    CHECKPOINT_PATH,
    PERTURBATIONS,
    BABYLM_DATA_PATH,
    gpt2_original_tokenizer,
    compute_surprisals,
)


MAX_TRAINING_STEPS = 3000
CHECKPOINTS = [300, 600, 900, 1200, 1500, 3000]


def load_test_file(test_file):
    file = open(test_file, 'r')
    token_sequences = [
        [int(s) for s in l.split()] + [EOS_TOKEN] for l in file.readlines()]
    return token_sequences


def subsample_file(token_sequences, sample_indices):
    return [token_sequences[i] for i in sample_indices]

def filter_files(l1, l2):
    new_l1 = []
    new_l2 = []
    for seq1, seq2 in zip(l1, l2):
        seq1 = seq1[:MAX_SEQ_LEN]
        seq2 = seq2[:MAX_SEQ_LEN]
        if seq1 != seq2:
            new_l1.append(seq1)
            new_l2.append(seq2)
    
    return new_l1, new_l2

def longest_common_prefix(list1, list2):
    prefix = []
    for a, b in zip(list1, list2):
        if a == b:
            prefix.append(a)
        else:
            break
    return prefix


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Get surprisal of correct/incorrect verb agreement",
        description="Verb agreement surprisal")
    parser.add_argument('perturbation_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=PERTURBATIONS.keys(),
                        help='Perturbation function used to transform BabyLM dataset')
    parser.add_argument('train_set',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=["100M", "10M"],
                        help='BabyLM train set')
    parser.add_argument('run_name', type=str, default="run_name", help="Run name")
    parser.add_argument('random_seed', type=int, help="Random seed")

    # Get args
    args = parser.parse_args()

    if "agreement" not in args.perturbation_type:
        raise Exception(
            "'{args.perturbation_type}' is not a valid hop perturbation")

    # Get path to model
    model = f"{args.run_name}_seed{args.random_seed}"
    model_path = f"{CHECKPOINT_PATH}/{args.perturbation_type}_{args.train_set}/{model}/checkpoints/checkpoint-"

    # Get perturbed test files
    if args.perturbation_type == "control_agreement":
        correct_filename = "control_agreement"
        incorrect_filename = "local_agreement"
    else:
        correct_filename = "local_agreement"
        incorrect_filename = "control_agreement"
    correct_files = sorted(glob(BABYLM_DATA_PATH +
        "/babylm_data_perturbed/babylm_{}/babylm_test_affected/*".format(correct_filename)))
    incorrect_files = sorted(glob(BABYLM_DATA_PATH +
        "/babylm_data_perturbed/babylm_{}/babylm_test_affected/*".format(incorrect_filename)))

    EOS_TOKEN = gpt2_original_tokenizer.eos_token_id
    FILE_SAMPLE_SIZE = 1000
    MAX_SEQ_LEN = 1024
    rng = default_rng(args.random_seed)

    correct_token_sequences = []
    incorrect_token_sequences = []
    target_indices = []

    # Iterate over data files to get surprisal data
    print("Sampling BabyLM affected test files to extract surprisals...")
    for correct_file, incorrect_file in zip(correct_files, incorrect_files):
        print(f"Correct file: {correct_file}")
        print(f"Incorrect file: {incorrect_file}")

        # Load correct and incorrect token sequences
        file_correct_token_sequences = load_test_file(correct_file)
        file_incorrect_token_sequences = load_test_file(incorrect_file)

        # Filter out sequences that are the same
        file_correct_token_sequences, file_incorrect_token_sequences = \
            filter_files(file_correct_token_sequences, file_incorrect_token_sequences)

        # Extract target indices
        for correct_seq, incorrect_seq in zip(file_correct_token_sequences, file_incorrect_token_sequences):
            # Find common prefix to identify target index
            target_index = len(longest_common_prefix(correct_seq, incorrect_seq))
            if target_index == len(correct_seq) or target_index == len(incorrect_seq):
                continue
            assert (target_index is not None)

            target_indices.append(target_index)
            correct_token_sequences.append(correct_seq)
            incorrect_token_sequences.append(incorrect_seq)

    # For logging/debugging, include decoded sentence
    correct_sents = [gpt2_original_tokenizer.decode(
        toks) for toks in correct_token_sequences]
    incorrect_sents = [gpt2_original_tokenizer.decode(
        toks) for toks in incorrect_token_sequences]
    correct_targets = [gpt2_original_tokenizer.decode(
        toks[idx]) for toks, idx in zip(correct_token_sequences, target_indices)]
    incorrect_targets = [gpt2_original_tokenizer.decode(
        toks[idx]) for toks, idx in zip(incorrect_token_sequences, target_indices)]

    surprisal_df = pd.DataFrame({
        "Grammatical Sentences": correct_sents,
        "Ungrammatical Sentences": incorrect_sents,
        "Grammatical Targets": correct_targets,
        "Ungrammatical Targets": incorrect_targets,
    })

    BATCH_SIZE = 32
    device = "cuda"
    for ckpt in CHECKPOINTS:
        print(f"Checkpoint: {ckpt}")

        # Load model
        model = GPT2LMHeadModel.from_pretrained(
            model_path + str(ckpt)).to(device)

        # Init lists for tracking correct/wrong surprisals for each example
        correct_surprisals = []
        incorrect_surprisals = []

        # Iterate over data in batches
        for i in tqdm.tqdm(range(0, len(correct_token_sequences), BATCH_SIZE)):

            # Extract data for batch and pad the sequences
            correct_batch = correct_token_sequences[i:i+BATCH_SIZE]
            correct_batch_padded = zip(
                *zip_longest(*correct_batch, fillvalue=gpt2_original_tokenizer.eos_token_id))
            correct_batch_tensors = torch.tensor(
                list(correct_batch_padded)).to(device)

            # Do the same for wrong batch
            incorrect_batch = incorrect_token_sequences[i:i+BATCH_SIZE]
            incorrect_batch_padded = zip(
                *zip_longest(*incorrect_batch, fillvalue=gpt2_original_tokenizer.eos_token_id))
            incorrect_batch_tensors = torch.tensor(
                list(incorrect_batch_padded)).to(device)

            # Get target indices in a batch
            targets_batch = target_indices[i:i+BATCH_SIZE]

            # Compute correct/incorrect surprisals for batches
            correct_surprisal_sequences = compute_surprisals(
                model, correct_batch_tensors)
            incorrect_surprisal_sequences = compute_surprisals(
                model, incorrect_batch_tensors)

            # Extract surprisals for target token
            for correct_seq, incorrect_seq, idx in \
                    zip(correct_surprisal_sequences, incorrect_surprisal_sequences, targets_batch):
                correct_surprisals.append(correct_seq[idx])
                incorrect_surprisals.append(incorrect_seq[idx])

        # Add surprisals to df
        ckpt_df = pd.DataFrame(
            list(zip(correct_surprisals, incorrect_surprisals)),
            columns=[f'Grammatical Token Surprisals (ckpt {ckpt})',
                     f'Ungrammatical Token Surprisals (ckpt {ckpt})',]
        )
        surprisal_df = pd.concat((surprisal_df, ckpt_df), axis=1)

    # Write results to CSV
    directory = f"agreement_surprisal_results/{args.perturbation_type}_{args.train_set}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file = directory + f"/{args.run_name}_seed{args.random_seed}.csv"
    print(f"Writing results to CSV: {file}")
    surprisal_df.to_csv(file)