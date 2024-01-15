# hop_interventions.py
# Author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

# align-transformers
PATH_TO_ALIGN_TRANSFORMERS = "/nlp/scr/kallini/align-transformers/"
sys.path.append(PATH_TO_ALIGN_TRANSFORMERS)

import pandas as pd
from models.utils import embed_to_distrib
from models.configuration_alignable_model import AlignableRepresentationConfig, AlignableConfig
from models.alignable_base import AlignableModel
from models.interventions import VanillaIntervention
from utils import CHECKPOINT_READ_PATH, marker_sg_token, marker_pl_token, \
    PERTURBATIONS, PAREN_MODELS
from tqdm import tqdm
from transformers import GPT2Model
from gpt2_no_positional_encoding_model import GPT2NoPositionalEncodingModel
import os
import torch
import argparse


MAX_TRAINING_STEPS = 3000
CHECKPOINTS = list(range(100, MAX_TRAINING_STEPS+1, 100))


def simple_position_config(model_type, intervention_type, layer):
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,             # layer
                intervention_type,  # intervention type
                "pos",             # intervention unit
                1                  # max number of unit
            ),
        ],
        alignable_interventions_type=VanillaIntervention,
    )
    return alignable_config


def get_model(perturbation_type, train_set, seed, paren_model, ckpt, no_pos_encodings=False):

    # Get path to model
    no_pos_encodings = "_no_positional_encodings" if no_pos_encodings else ""
    model = f"babylm_{perturbation_type}_{train_set}_{paren_model}{no_pos_encodings}_seed{seed}"
    model_path = f"{CHECKPOINT_READ_PATH}/babylm_{perturbation_type}_{train_set}_{paren_model}{no_pos_encodings}/{model}/runs/{model}/checkpoint-{ckpt}"

    # Load appropriate GPT-2 model
    if no_pos_encodings:
        return GPT2NoPositionalEncodingModel.from_pretrained(model_path).to(device)
    else:
        return GPT2Model.from_pretrained(model_path).to(device)


def run_interventions(model, base_input_ids, source_input_ids):

    tokens = [marker_sg_token, marker_pl_token]

    data = []
    BATCH_SIZE = 16
    for batch_i in tqdm(range(0, len(base_input_ids), BATCH_SIZE)):

        # Get base and source batches
        base_batch = base_input_ids[batch_i:batch_i+BATCH_SIZE]
        source_batch = source_input_ids[batch_i:batch_i+BATCH_SIZE]

        # Iterate over GPT-2 layers
        for layer_i in range(model.config.n_layer):

            # Get block_output config for this layer
            alignable_config = simple_position_config(
                type(model), "block_output", layer_i)
            alignable = AlignableModel(alignable_config, model)

            # Iterate over token positions
            for pos_i in range(len(base_batch[0])):

                _, counterfactual_outputs = alignable(
                    {"input_ids": torch.tensor(base_batch).to(device)},
                    [{"input_ids": torch.tensor(source_batch).to(device)}],
                    {"sources->base": ([[[pos_i]] * len(base_batch)],
                                       [[[pos_i]] * len(base_batch)])}
                )
                distrib = embed_to_distrib(
                    model, counterfactual_outputs.last_hidden_state,
                    logits=False
                )
                for i in range(len(base_batch)):
                    for token in tokens:
                        data.append({
                            'example': batch_i + i,
                            'token': token,
                            'prob': float(distrib[i][-1][token]),
                            'layer': layer_i,
                            'pos': pos_i,
                            'type': "block_output"
                        })
    return pd.DataFrame(data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Run intervention tests for subject-verb agreement on hop models',
        description='Run interventions for subject-verb agreement on hop models')
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
    parser.add_argument('-np', '--no_pos_encodings', action='store_true',
                        help="Train GPT-2 with no positional encodings")

    # Get args
    args = parser.parse_args()

    if "hop" not in args.perturbation_type:
        raise Exception(
            "'{args.perturbation_type}' is not a valid hop perturbation")

    # Get examples to run interventions
    data_df = pd.read_csv("hop_agreement_data.csv")
    bases = [[int(tok) for tok in seq.split()]
             for seq in list(data_df["Singular"])]
    sources = [[int(tok) for tok in seq.split()]
               for seq in list(data_df["Plural"])]

    # Only get first three tokens of each example for control model
    if args.perturbation_type == "hop_control":
        bases = [row[:3] for row in bases]
        sources = [row[:3] for row in sources]

    # Get model and run intervention experiments
    device = "cuda"
    result_df = None
    for ckpt in CHECKPOINTS:
        print(f"Checkpoint: {ckpt}")
        model = get_model(args.perturbation_type, args.train_set,
                        args.random_seed, args.paren_model, ckpt,
                        args.no_pos_encodings)
        if result_df is None:
            result_df = run_interventions(model, bases, sources)
            result_df["ckpt"] = ckpt
        else:
            ckpt_df = run_interventions(model, bases, sources)
            ckpt_df["ckpt"] = ckpt
            result_df = pd.concat((result_df, ckpt_df), axis=0)

    # Create directory for results
    nps = '_no_pos_encodings' if args.no_pos_encodings else ''
    result_directory = f"hop_intervention_results/{args.perturbation_type}_{args.train_set}{nps}/"
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    # Write results
    result_df.to_csv(result_directory + f"{args.paren_model}_seed{args.random_seed}.csv", index=False)
