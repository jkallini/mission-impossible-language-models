# edge_probing.py
# Author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

from utils import CHECKPOINT_READ_PATH, PERTURBATIONS, PAREN_MODELS, get_gpt2_tokenizer_with_markers
from gpt2_no_positional_encoding_model import GPT2NoPositionalEncodingLMHeadModel
from transformers import GPT2LMHeadModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import zip_longest
import torch
import tqdm
import argparse
import pandas as pd
import os


MAX_TRAINING_STEPS = 3000
CHECKPOINTS = list(range(200, MAX_TRAINING_STEPS+1, 200))
LAYERS = [1, 3, 6, 9, 12, "Avg Last 4"]


def get_layer_embedding(model, token_sequences, indices, layer=None):

    # Pad input token sequences
    input_ids = zip(*zip_longest(*token_sequences,
                    fillvalue=gpt2_tokenizer.eos_token_id))
    input_ids = torch.tensor(list(input_ids)).to(device)

    # Get GPT2 model's output
    with torch.no_grad():
        output = model(input_ids)

    # Either get the hidden state of the specified layer or
    # get the average of the last 4 hidden states
    if layer is not None:
        hidden_states = output.hidden_states[layer]
    else:
        hidden_states = output.hidden_states[-4:]
        hidden_states = sum(hidden_states) / 4

    # Create mask using start and end indices
    batch_size, seq_length = input_ids.shape
    mask = torch.full((batch_size, seq_length), 0).to(device)
    for i, (start_idx, end_idx) in enumerate(indices):
        mask[i, start_idx:end_idx] = 1

    # Mask out embeddings of tokens outside indices
    mask_expanded = mask.unsqueeze(-1).expand(hidden_states.size())
    hidden_states = hidden_states * mask_expanded

    return hidden_states


def max_pooling(tensor, index_tuples):
    pooled_results = []
    for i, (start, end) in enumerate(index_tuples):
        # Extracting the embeddings corresponding to the specified range
        embeddings = tensor[i, start:end, :]

        # Performing max pooling
        max_pooled = torch.max(embeddings, dim=0)[0]

        pooled_results.append(max_pooled)
    return torch.stack(pooled_results)


def mean_pooling(tensor, index_tuples):
    batch_size, seq_len, embedding_size = tensor.shape
    output = torch.empty(batch_size, embedding_size,
                         device=tensor.device, dtype=tensor.dtype)

    for i, (start, end) in enumerate(index_tuples):
        embeddings = tensor[i, start:end, :]
        output[i, :] = torch.mean(embeddings, dim=0)

    return output


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
    parser.add_argument('pooling_operation',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=["mean", "max"],
                        help='Pooling operation to compute on embeddings')
    parser.add_argument('-np', '--no_pos_encodings', action='store_true',
                        help="Train GPT-2 with no positional encodings")

    # Get args
    args = parser.parse_args()

    if args.pooling_operation == "mean":
        pooling_function = mean_pooling
    elif args.pooling_operation == "max":
        pooling_function = max_pooling
    else:
        raise Exception("Pooling operation undefined")

    # Init tokenizer
    gpt2_tokenizer = get_gpt2_tokenizer_with_markers([])
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    # Get path to model
    no_pos_encodings_underscore = "_no_positional_encodings" if args.no_pos_encodings else ""
    model = f"babylm_{args.perturbation_type}_{args.train_set}_{args.paren_model}{no_pos_encodings_underscore}_seed{args.random_seed}"
    model_path = f"{CHECKPOINT_READ_PATH}/babylm_{args.perturbation_type}_{args.train_set}_{args.paren_model}{no_pos_encodings_underscore}/{model}/runs/{model}/checkpoint-"

    # Get constituency parse data
    if "hop" in args.perturbation_type:
        phrase_df = pd.read_csv("phrase_data/hop_phrase_data.csv")
    elif "reverse" in args.perturbation_type:
        phrase_df = pd.read_csv("phrase_data/reverse_phrase_data.csv")
    else:
        raise Exception("Phrase data not found")

    token_sequences = list(phrase_df["Sentence Tokens"])
    if args.perturbation_type == "reverse_full":
        indices = list(
            zip(phrase_df["Rev Start Index"], phrase_df["Rev End Index"]))
    else:
        indices = list(zip(phrase_df["Start Index"], phrase_df["End Index"]))
    labels = list(phrase_df["Category"])

    BATCH_SIZE = 32
    device = "cuda"

    edge_probing_df = pd.DataFrame(LAYERS, columns=["GPT-2 Layer"])
    for ckpt in CHECKPOINTS:

        # Load model
        if args.no_pos_encodings:
            model = GPT2LMHeadModel.from_pretrained(
                model_path + str(ckpt), output_hidden_states=True).to(device)
        else:
            model = GPT2NoPositionalEncodingLMHeadModel.from_pretrained(
                model_path + str(ckpt), output_hidden_states=True).to(device)

        layer_accuracies = []
        for layer in LAYERS:
            print(f"Checkpoint: {ckpt}, Layer: {layer}")
            print("Computing span embeddings...")

            # Iterate over token sequences and indices to get embeddings
            spans = []
            for i in tqdm.tqdm(list(range(0, len(token_sequences), BATCH_SIZE))):

                tokens_batch = [[int(tok) for tok in seq.split()]
                                for seq in token_sequences[i:i+BATCH_SIZE]]
                if args.perturbation_type == "reverse_full":
                    tokens_batch = [toks[::-1] for toks in tokens_batch]

                index_batch = indices[i:i+BATCH_SIZE]

                # Extract embeddings
                if layer == "Avg Last 4":
                    embeddings = get_layer_embedding(
                        model, tokens_batch, index_batch, None)
                else:
                    embeddings = get_layer_embedding(
                        model, tokens_batch, index_batch, layer)
                pooled_results = pooling_function(embeddings, index_batch)
                spans.extend(list(pooled_results))

            # Get features and ground truth
            X = torch.vstack(spans).detach().cpu().numpy()
            y = labels

            # Split the data; since we pass random seed, it
            # will be the same split every time
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=args.random_seed)

            # Fit L2-regularized linear classifier
            clf = LogisticRegression(max_iter=10,
                                     random_state=args.random_seed).fit(X_train, y_train)

            # Get probe accuracy
            y_test_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_test_pred)
            layer_accuracies.append(acc)
            print(f"Accuracy: {acc}")

        edge_probing_df[f"Accuracy (ckpt {ckpt})"] = layer_accuracies

    # Write results to CSV
    nps = '_no_pos_encodings' if args.no_pos_encodings else ''
    directory = f"edge_probing_results/{args.perturbation_type}_{args.train_set}{nps}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    file = directory + \
        f"/{args.paren_model}_{args.pooling_operation}_pooling_seed{args.random_seed}.csv"
    print(f"Writing results to CSV: {file}")
    edge_probing_df.to_csv(file)
