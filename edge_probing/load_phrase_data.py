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


def get_span(tokens, sub_tokens):
    for i in range(len(tokens)):
        if tokens[i:i+len(sub_tokens)] == sub_tokens:
            start_idx, end_idx = i, i + len(sub_tokens)
            return start_idx, end_idx


def extract_phrases(tree, categories):
    """Extract phrases that belong to the specified categories."""
    results = []

    for subtree in tree.subtrees():
        if subtree.label() in categories:
            words = subtree.leaves()
            if len(words) > 1:
                phrase = ' '.join(words)
                results.append((phrase, subtree.label()))

    return results


def process_file(file):
    """Read file, extract phrases and return results."""
    results = []

    categories = ["NP", "VP", "ADJP", "ADVP", "PP"]

    # Get all files from the given path
    with open(file, 'r') as f:
        lines = f.readlines()

        # Process every two lines
        for i in tqdm(range(0, len(lines) - 1, 2)):
            sentence = lines[i].strip()

            if len(sentence.split()) < 5:
                continue

            tree_str = lines[i + 1].strip()
            tree = Tree.fromstring(tree_str)

            phrases = extract_phrases(tree, categories)
            for phrase, category in phrases:
                if phrase in sentence:
                    results.append((sentence, phrase, category))

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Get phrase spans for edge probing',
        description='Get spans of text from constituency parses for edge probing experiments')
    parser.add_argument('perturbation_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=PERTURBATIONS.keys(),
                        help='Perturbation function used to transform BabyLM dataset')

    # Get args
    args = parser.parse_args()

    # Get class of perturbations
    perturbation_class = None
    if "reverse" in args.perturbation_type:
        perturbation_class = "reverse" 
    elif "hop" in args.perturbation_type:
        perturbation_class = "hop"
    else:
        raise Exception("Perturbation class not implemented")

    # Process constituency parses
    print("Extracting phrases from constituency parses")
    data = process_file(
        f"test_constituency_parses/{perturbation_class}_parses.test")

    # Get a sufficiently large sample of phrases
    SAMPLE_SIZE = 10000
    RANDOM_STATE = 62
    rng = default_rng(RANDOM_STATE)
    rng.shuffle(data)

    # Find the start and end indices of the substring's tokens within the sentence's tokens
    tokenizer = PERTURBATIONS[args.perturbation_type]["gpt2_tokenizer"]
    span_sample_data = []
    print("Getting spans of tokens for constituents")
    for sentence, phrase, category in tqdm(data):

        # Tokenize both the full sentence and the substring
        tokens = tokenizer.encode(sentence)
        if len(tokens) > 1024:
            continue
        sub_tokens = tokenizer.encode(phrase)

        span = get_span(tokens, sub_tokens)

        # If span is not found, append space to substring
        if span is None:
            sub_tokens = tokenizer.encode(" " + phrase)
            span = get_span(tokens, sub_tokens)

        if span is not None:
            start_idx, end_idx = span
            rev_start_index, rev_end_index = len(tokens) - end_idx, len(tokens) - start_idx
            span_sample_data.append(
                (sentence, phrase, category, " ".join([str(t) for t in tokens]),
                 start_idx, end_idx, rev_start_index, rev_end_index))

    # Create DataFrame and write stratefied random sample
    sample_df = pd.DataFrame(data=span_sample_data, columns=[
                             "Sentence", "Phrase", "Category",
                             "Sentence Tokens", "Start Index", "End Index",
                             "Rev Start Index", "Rev End Index"])
    final_sample_df = sample_df.groupby('Category', group_keys=False).apply(
        lambda x: x.sample(SAMPLE_SIZE // 5, random_state=RANDOM_STATE))
    final_sample_df = final_sample_df.sample(frac=1).reset_index(drop=True)

    # Create directory and write CSV
    phrases_directory = f"phrase_data/"
    if not os.path.exists(phrases_directory):
        os.makedirs(phrases_directory)
    phrases_file = phrases_directory + f"{perturbation_class}_phrase_data.csv"
    final_sample_df.to_csv(phrases_file, index=False)
