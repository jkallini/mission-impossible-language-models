# load_phrase_data.py
# author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

from utils import PERTURBATIONS
import argparse
import pandas as pd
import os
from tqdm import tqdm
from nltk import Tree
from numpy.random import default_rng
import random

def get_span(tokens, sub_tokens):
    for i in range(len(tokens)):
        if tokens[i:i+len(sub_tokens)] == sub_tokens:
            start_idx, end_idx = i, i + len(sub_tokens)
            return start_idx, end_idx

def extract_constituents(tree):
    """Extract all true constituent phrases (multiword only)."""
    constituents = set()
    phrases = []

    for subtree in tree.subtrees():
        leaves = subtree.leaves()
        if len(leaves) > 1:
            phrase = " ".join(leaves)
            span = tuple(leaves)
            constituents.add(span)
            phrases.append(phrase)

    return constituents, phrases

def process_file(file):
    """Read file, extract constituents and non-constituents with labels."""
    results = []

    with open(file, 'r') as f:
        lines = f.readlines()

        for i in tqdm(range(0, len(lines) - 1, 2)):
            sentence = lines[i].strip()
            if len(sentence.split()) < 5:
                continue

            tree_str = lines[i + 1].strip()
            tree = Tree.fromstring(tree_str)

            tokens = sentence.split()
            constituents, constituent_phrases = extract_constituents(tree)
            constituent_phrases_set = set(constituent_phrases)

            # Add positive examples
            for phrase in constituent_phrases:
                results.append((sentence, phrase, 1))

            # Add negative examples: one for each constituent
            tokens_len = len(tokens)
            used_phrases = set(constituent_phrases)
            negatives_added = 0
            attempts = 0
            max_attempts = 1000

            while negatives_added < len(constituent_phrases) and attempts < max_attempts:
                start = random.randint(0, tokens_len - 2)
                end = random.randint(start + 2, tokens_len + 1)  # length >= 2
                if end > tokens_len:
                    attempts += 1
                    continue

                span = tuple(tokens[start:end])
                phrase = " ".join(span)

                if phrase not in used_phrases:
                    results.append((sentence, phrase, 0))
                    used_phrases.add(phrase)
                    negatives_added += 1

                attempts += 1

    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Get phrase spans for edge probing',
        description='Get spans of constituents and non-constituents for probing')
    parser.add_argument('perturbation_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=PERTURBATIONS.keys(),
                        help='Perturbation function used to transform BabyLM dataset')

    args = parser.parse_args()

    perturbation_class = None
    if "reverse" in args.perturbation_type:
        perturbation_class = "reverse"
    elif "hop" in args.perturbation_type:
        perturbation_class = "hop"
    else:
        raise Exception("Perturbation class not implemented")

    print("Extracting phrases from constituency parses")
    data = process_file(f"test_constituency_parses/{perturbation_class}_parses.test")

    SAMPLE_SIZE = 10000
    RANDOM_STATE = 62
    rng = default_rng(RANDOM_STATE)
    rng.shuffle(data)

    tokenizer = PERTURBATIONS[args.perturbation_type]["gpt2_tokenizer"]
    span_sample_data = []
    print("Getting spans of tokens for phrases")
    for sentence, phrase, label in tqdm(data):
        tokens = tokenizer.encode(sentence)
        if len(tokens) > 1024:
            continue
        sub_tokens = tokenizer.encode(phrase)

        span = get_span(tokens, sub_tokens)
        if span is None:
            sub_tokens = tokenizer.encode(" " + phrase)
            span = get_span(tokens, sub_tokens)

        if span is not None:
            start_idx, end_idx = span
            rev_start_index, rev_end_index = len(tokens) - end_idx, len(tokens) - start_idx
            span_sample_data.append(
                (sentence, phrase, label, " ".join([str(t) for t in tokens]),
                 start_idx, end_idx, rev_start_index, rev_end_index))

    sample_df = pd.DataFrame(data=span_sample_data, columns=[
        "Sentence", "Phrase", "IsConstituent",
        "Sentence Tokens", "Start Index", "End Index",
        "Rev Start Index", "Rev End Index"])

    final_sample_df = sample_df.sample(frac=1, random_state=RANDOM_STATE)  # shuffle
    if len(final_sample_df) > SAMPLE_SIZE:
        final_sample_df = final_sample_df.iloc[:SAMPLE_SIZE]

    phrases_directory = f"phrase_data/"
    if not os.path.exists(phrases_directory):
        os.makedirs(phrases_directory)
    phrases_file = phrases_directory + f"{perturbation_class}_constituent_span_data.csv"
    final_sample_df.to_csv(phrases_file, index=False)
