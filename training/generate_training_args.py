import sys
import argparse
import random
from itertools import product

sys.path.append("..")
from utils import PERTURBATIONS

perturbations = ["shuffle_nondeterministic", "shuffle_deterministic57", "shuffle_local3", "shuffle_control",
                 "hop_words4", "hop_tokens4", "hop_control",
                 "reverse_full", "reverse_partial", "reverse_control",
                 "agreement_local", "agreement_control",
                 "negation_linear", "negation_control"]
arch_options = ["", "--alibi", "--rope"] #  "--geometric_attention",  "--no_pos_encodings"

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds to generate for each config")
args = parser.parse_args()
num_seeds = args.num_seeds

used_seeds = set()

def get_unique_seed():
    while True:
        seed = random.randint(0, 9999)
        if seed not in used_seeds:
            used_seeds.add(seed)
            return seed

output_filename = f"args_{num_seeds}.txt"

with open(output_filename, "w") as f:
    for perturb, arch in product(perturbations, arch_options):
        assert(perturb in PERTURBATIONS.keys()), f"Unknown perturbation: {perturb}"
        for _ in range(num_seeds):
            seed = get_unique_seed()
            arch_suffix = arch.strip("--") if arch else "gpt2"
            run_name = f"{perturb}_{arch_suffix}_seed{seed}"
            f.write(f"{perturb} 100M {run_name} {seed} {arch}\n")
