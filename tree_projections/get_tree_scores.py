# get_tree_scores.py
# Author: Julie Kallini

import sys
sys.path.append("..")

import argparse
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

from utils import (
    PERTURBATIONS,
    CHECKPOINT_PATH,
    BABYLM_DATA_PATH,
)
from models.modeling_gpt2 import GPT2Model
from tree_projection import TreeProjection

FILE_SAMPLE_SIZE = 1000
MAX_SEQ_LEN = 1024

def load_model(run_name, random_seed, perturbation_type, train_set, ckpt=3000):
    model_dir = f"{CHECKPOINT_PATH}/{perturbation_type}_{train_set}/{run_name}_seed{random_seed}/checkpoints"
    model = GPT2Model.from_pretrained(f"{model_dir}/checkpoint-{ckpt}").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model

def load_samples(perturbation_type, rng):
    test_files = sorted(glob(f"{BABYLM_DATA_PATH}/babylm_data_perturbed/babylm_{perturbation_type}/babylm_test_affected/*"))
    samples = []
    for file in tqdm(test_files, desc="Loading Test Files"):
        f = open(file, 'r')
        file_token_sequences = [
            [int(s) for s in l.split()] for l in f.readlines()]
        file_token_sequences = [
            toks for toks in file_token_sequences if len(toks) < MAX_SEQ_LEN]
        if len(file_token_sequences) >= FILE_SAMPLE_SIZE:
            sample_indices = rng.choice(len(file_token_sequences), FILE_SAMPLE_SIZE, replace=False)
            file_token_sequences = [file_token_sequences[i] for i in sample_indices]
        samples.extend(file_token_sequences)
    return samples

def compute_tree_metric(model, samples):
    tree_projector = TreeProjection(model=model)
    all_scores = []
    for input_ids in tqdm(samples, desc="Computing Tree Metric"):
        sci_chart = tree_projector.compute_sci_chart(
            input_ids,
            [1] * len(input_ids),
            st_threshold=4,
            layer_id=11
        )
        _, score = tree_projector(
            sci_chart=sci_chart,
            input_ids=input_ids,
            projection_algorithm="dp"
        )
        all_scores.append(score / len(input_ids))  # Normalize score by sequence length
    return np.mean(all_scores), np.std(all_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('perturbation_type', choices=PERTURBATIONS.keys(), help='Perturbation function used to transform BabyLM dataset')
    parser.add_argument('train_set', choices=['100M', '10M'], help='BabyLM train set')
    parser.add_argument('run_name', type=str, help='Run name')
    parser.add_argument('random_seed', type=int, help='Random seed')
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    model = load_model(args.run_name, args.random_seed, args.perturbation_type, args.train_set)
    samples = load_samples(args.perturbation_type, rng)
    mean_score, std_dev = compute_tree_metric(model, samples)
    print(f"Tree Metric: {mean_score:.4f} ± {std_dev:.4f}")

if __name__ == '__main__':
    main()
