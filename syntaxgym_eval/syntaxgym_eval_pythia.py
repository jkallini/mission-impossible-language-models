# syntaxgym_eval_pythia.py
# Author: Julie Kallini

import sys
sys.path.append('..')

import argparse
import torch
import pandas as pd
from syntaxgym_utils import SYNTAXGYM_EVALS, load_syntaxgym_data
from transformers import GPTNeoXForCausalLM, AutoTokenizer


def main(args):
    # Set device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{args.model_size}",
    )

    CHECKPOINTS = [0, 128, 256, 512, 1000, 2000, 3000, 4000, 10000, 143000]

    results = []

    for ckpt in CHECKPOINTS:
        print(f"--------------Evaluating checkpoint {ckpt}--------------")

        # Load model
        model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/pythia-{args.model_size}",
            revision=f"step{ckpt}",
            use_cache=False,
        ).to(DEVICE)

        for suite_name in SYNTAXGYM_EVALS[args.task_type]:
            print(f"Evaluating {suite_name}...")
            subset = load_syntaxgym_data(suite_name)
            suite = SYNTAXGYM_EVALS[args.task_type][suite_name]
            eval_function = suite["eval_function"]
            target_loc = suite["target_loc"]
            result = eval_function(tokenizer, model, subset, target_loc, pythia=True)

            for res in result:
                print(f"{res}: {result[res]}")
                results.append([suite_name, ckpt, res, result[res]])
        
        print()

    results_df = pd.DataFrame(results, columns=["Suite Name", "Checkpoint", "Condition", "Results"])
    results_df.to_csv(f"results/pythia-{args.model_size}_{args.task_type}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pythia model on SyntaxGym data")
    parser.add_argument('task_type', default='all', nargs='?',
                        choices=SYNTAXGYM_EVALS.keys(),
                        help='SyntaxGym task type')
    parser.add_argument('model_size', type=str, help="model name")

    args = parser.parse_args()
    main(args)