# load_phrase_data.py
# author: Julie Kallini 

import sys, os, argparse
sys.path.append("..")

from utils import PERTURBATIONS
import pandas as pd
from tqdm import tqdm
from nltk import Tree
from numpy.random import default_rng


def get_span(tokens, sub_tokens):
    """Return (start, end) indices of sub_tokens inside tokens or None."""
    for i in range(len(tokens) - len(sub_tokens) + 1):
        if tokens[i : i + len(sub_tokens)] == sub_tokens:
            return i, i + len(sub_tokens)
    return None


def extract_constituents(tree, cats=("NP", "VP", "ADJP", "ADVP", "PP"), min_len=2):
    """Return all multi‑word constituent strings whose label ∈ cats."""
    return [
        " ".join(t.leaves())
        for t in tree.subtrees()
        if t.label() in cats and len(t.leaves()) > min_len
    ]


def sample_negative_span(span_len, n_tokens, constituent_spans, rng):
    """Sample a random span of length span_len that is *not* a constituent."""
    candidates = [
        (s, s + span_len)
        for s in range(0, n_tokens - span_len + 1)
        if (s, s + span_len) not in constituent_spans
    ]
    return rng.choice(candidates) if candidates else None


def process_file(path, tokenizer, rng):
    """Yield (sentence, span_text, label, tokens_str, start, end, rev_start, rev_end)."""
    data = []

    with open(path, "r") as f:
        lines = f.readlines()

    for i in tqdm(range(0, len(lines) - 1, 2)):
        sentence = lines[i].strip()
        if len(sentence.split()) < 5:
            continue

        tree = Tree.fromstring(lines[i + 1].strip())
        constituents = extract_constituents(tree)
        if not constituents:
            continue

        tokens = tokenizer.encode(sentence)
        if len(tokens) > 1024:          # skip extremely long sentences
            continue

        span_tokens_str = " ".join(map(str, tokens))
        constituent_spans = set()       # {(start,end), …}
        positives = []

        # build positive examples & record their spans
        for phrase in constituents:
            sub_tok = tokenizer.encode(phrase)
            span = get_span(tokens, sub_tok) or get_span(tokens, tokenizer.encode(" " + phrase))
            if not span:
                continue
            s, e = span
            constituent_spans.add(span)
            rev_s, rev_e = len(tokens) - e, len(tokens) - s
            positives.append((sentence, phrase, 1, span_tokens_str, s, e, rev_s, rev_e))

        if not positives:
            continue

        # for every positive span, sample one negative span of the same length
        for sent, phrase, _, tok_str, s, e, rev_s, rev_e in positives:
            span_len = e - s
            neg = sample_negative_span(span_len, len(tokens), constituent_spans, rng)
            if neg is None:
                continue
            ns, ne = neg
            n_phrase = tokenizer.decode(tokens[ns:ne])
            n_rev_s, n_rev_e = len(tokens) - ne, len(tokens) - ns
            data.append((sent, n_phrase, 0, tok_str, ns, ne, n_rev_s, n_rev_e))

        data.extend(positives)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Get span data for constituent probing (binary)",
        description="Create constituent vs non‑constituent span dataset",
    )
    parser.add_argument('perturbation_class',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=["reverse", "hop", "negation", "agreement"],
                        help='Perturbation function used to transform BabyLM dataset')
    args = parser.parse_args()

    tokenizer = PERTURBATIONS[args.perturbation_class + "_control"]["gpt2_tokenizer"]
    rng = default_rng(seed=62)

    print("Building span dataset...")
    examples = process_file(f"../constituency_parses/{args.perturbation_class}_parses.test", tokenizer, rng)

    # balance, shuffle, and subsample
    SAMPLE_SIZE = 10_000       # final size (50 % positive, 50 % negative)
    df = pd.DataFrame(
        examples,
        columns=[
            "Sentence",
            "Span Text",
            "Label",            # 1 = constituent, 0 = non‑constituent
            "Sentence Tokens",
            "Start Index",
            "End Index",
            "Rev Start Index",
            "Rev End Index",
        ],
    )
    balanced = (
        df.groupby("Label", group_keys=False)
        .sample(n=SAMPLE_SIZE // 2, random_state=62, replace=False)
        .sample(frac=1, random_state=62)
        .reset_index(drop=True)
    )

    out_dir = "phrase_data/"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.perturbation_class}_constituent_probe_spans.csv")
    balanced.to_csv(out_file, index=False)
    print(f"✓ Wrote {len(balanced)} examples to {out_file}")
