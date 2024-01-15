# perplexities.py
# Author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

from transformers import GPT2LMHeadModel
from gpt2_no_positional_encoding_model import GPT2NoPositionalEncodingLMHeadModel
from utils import CHECKPOINT_READ_PATH, PERTURBATIONS, BABYLM_DATA_PATH, \
    PAREN_MODELS, gpt2_original_tokenizer
from tqdm import tqdm
from glob import glob
from numpy.random import default_rng
import pandas as pd
import torch
import itertools
import argparse
import os


MAX_TRAINING_STEPS = 3000
CHECKPOINTS = list(range(100, MAX_TRAINING_STEPS+1, 100))


def create_attention_mask(token_lists):
    seq_length = max([len(i) for i in token_lists])
    batch_size = len(token_lists)
    mask = torch.full((batch_size, seq_length), 0)

    for i, tokens in enumerate(token_lists):
        mask[i, 0:len(tokens)] = 1

    return mask


def create_input_ids(token_lists, pad_token_id):
    padded = zip(*itertools.zip_longest(*token_lists, fillvalue=pad_token_id))
    return torch.tensor(list(padded))


def get_perplexities(model, token_lists, pad_token_id, device="cuda"):

    # Prepare data
    input_ids = create_input_ids(token_lists, pad_token_id).to(device)
    labels = input_ids.clone()  # GPT-2 uses input as labels for CLM task
    attention_mask = create_attention_mask(token_lists).to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels,
                    attention_mask=attention_mask)

    # The "shifted" nature of labels in GPT-2 (next token prediction)
    # Shift logits, labels, and attention mask by one position
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()

    # Instantiate loss function with no reduction
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    # Calculate per-token loss
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    # Reshape back to the original batch size and sequence length
    loss = loss.view(shift_labels.size())

    # Apply the attention mask - only calculate loss where mask is 1
    loss = loss * shift_attention_mask

    # Sum the loss over the sequence length, get per-example perplexity
    per_example_loss = loss.sum(dim=1) / shift_attention_mask.sum(dim=1)
    return torch.exp(per_example_loss).tolist()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Edge probing',
        description='Edge probing experiments')
    parser.add_argument('perturbation_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=PERTURBATIONS.keys(),
                        help='Perturbation function used to transform BabyLM dataset')
    parser.add_argument('test_perturbation_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=PERTURBATIONS.keys(),
                        help='Perturbation function used to transform test BabyLM dataset')
    parser.add_argument('train_set',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=["100M", "10M"],
                        help='BabyLM train set')
    parser.add_argument('random_seed', type=int, help="Random seed")
    parser.add_argument('paren_model',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=list(PAREN_MODELS.keys()) + ["randinit"],
                        help='Parenthesis model')
    parser.add_argument('-np', '--no_pos_encodings', action='store_true',
                        help="Train GPT-2 with no positional encodings")

    # Get args
    args = parser.parse_args()
    no_pos_encodings_underscore = "_no_positional_encodings" if args.no_pos_encodings else ""

    # Get path to model
    model = f"babylm_{args.perturbation_type}_{args.train_set}_{args.paren_model}{no_pos_encodings_underscore}_seed{args.random_seed}"
    model_path = f"{CHECKPOINT_READ_PATH}/babylm_{args.perturbation_type}_{args.train_set}_{args.paren_model}{no_pos_encodings_underscore}/{model}/runs/{model}/checkpoint-"

    # Get perturbed test files
    test_files = sorted(glob(
        f"{BABYLM_DATA_PATH}/babylm_data_perturbed/babylm_{args.test_perturbation_type}/babylm_test_affected/*"))

    FILE_SAMPLE_SIZE = 1000
    rng = default_rng(args.random_seed)

    # Iterate over data files to get perplexity data
    print("Sampling BabyLM affected test files to extract surprisals...")
    token_sequences = []
    for test_file in test_files:
        print(test_file)

        # Get tokens from test file and subsample
        f = open(test_file, 'r')
        file_token_sequences = [
            [int(s) for s in l.split()] for l in f.readlines()]
        sample_indices = rng.choice(
            list(range(len(file_token_sequences))), FILE_SAMPLE_SIZE, replace=False)
        file_token_sequences = [file_token_sequences[i]
                                for i in sample_indices]
        token_sequences.extend(file_token_sequences)

    # For logging/debugging, include decoded sentence
    test_sents = [gpt2_original_tokenizer.decode(
        toks) for toks in token_sequences]

    ppl_df = pd.DataFrame({
        "Sentences": test_sents
    })

    BATCH_SIZE = 8
    device = "cuda"
    for ckpt in CHECKPOINTS:
        print(f"Checkpoint: {ckpt}")

        # Load model
        if args.no_pos_encodings:
            model = GPT2NoPositionalEncodingLMHeadModel.from_pretrained(
                model_path + str(ckpt)).to(device)
        else:
            model = GPT2LMHeadModel.from_pretrained(
            model_path + str(ckpt)).to(device)

        # Get perplexities
        perplexities = []
        for i in tqdm(range(0, len(token_sequences), BATCH_SIZE)):
            batch = token_sequences[i:i+BATCH_SIZE]
            ppls = get_perplexities(
                model, batch, gpt2_original_tokenizer.eos_token_id)
            perplexities.extend(ppls)

        # Add ppls to df
        ppl_df[f'Perplexities (ckpt {ckpt})'] = perplexities

    # Write results to CSV
    directory = f"perplexity_results/{args.perturbation_type}_{args.train_set}{no_pos_encodings_underscore}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file = directory + \
        f"/{args.paren_model}_seed{args.random_seed}_test_{args.test_perturbation_type}.csv"
    print(f"Writing results to CSV: {file}")
    ppl_df.to_csv(file)
