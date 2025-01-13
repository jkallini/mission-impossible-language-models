# syntaxgym_eval.py
# Author: Julie Kallini

import sys
sys.path.append('..')

import argparse
import torch
import pandas as pd
from models.modeling_gpt2 import GPT2LMHeadModel
from utils import PERTURBATIONS, CHECKPOINT_PATH
from syntaxgym_utils import SYNTAXGYM_EVALS, load_syntaxgym_data


def main(args):
    # Set device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f"{args.run_name}_seed{args.random_seed}"
    model_path = f"{CHECKPOINT_PATH}/{args.perturbation_type}_{args.train_set}/{model_name}/checkpoints/checkpoint-{args.checkpoint}"

    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)
    tokenizer = PERTURBATIONS[args.perturbation_type]["gpt2_tokenizer"]
    results = []

    for suite_name in SYNTAXGYM_EVALS[args.task_type]:
        print(f"Evaluating {suite_name}...")
        subset = load_syntaxgym_data(suite_name)
        suite = SYNTAXGYM_EVALS[args.task_type][suite_name]
        eval_function = suite["eval_function"]
        target_loc = suite["target_loc"]
        result = eval_function(tokenizer, model, subset, target_loc, args.local_agreement)

        for res in result:
            print(f"{res}: {result[res]}")
            results.append([suite_name, res, result[res]])

    results_df = pd.DataFrame(results, columns=["Suite Name", "Condition", "Results"])
    results_df.to_csv(f"results/{model_name}_{args.task_type}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 model on SyntaxGym data")
    parser.add_argument('task_type', default='all', const='all', nargs='?',
                        choices=SYNTAXGYM_EVALS.keys(),
                        help='SyntaxGym task type')
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
    parser.add_argument('--checkpoint', type=int, default=3000, help="Checkpoint number")
    parser.add_argument('--local_agreement', action='store_true',
                        help="Evaluate model with local agreement")

    args = parser.parse_args()
    main(args)