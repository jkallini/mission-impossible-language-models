# get_constituency_parses.py
# Author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

import os
import argparse
import stanza
import json
import tqdm
import numpy as np
from utils import write_file, merge_part_tokens, BABYLM_DATA_PATH
from glob import glob


def __get_constituency_parse(sent, nlp, perturbation_class):
    try:
        parse_doc = nlp(sent)
        parsed_sent = parse_doc.sentences[0]
        if perturbation_class == "reverse":
            new_sent = sent
        elif perturbation_class in ("hop", "negation", "agreement"):
            words = [w.text for w in parsed_sent.words]
            new_sent = " ".join(merge_part_tokens(words))
        else:
            raise Exception("Perturbation class is not implemented")
        return str(parsed_sent.constituency), new_sent
    except:
        return None, None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Parse BabyLM test data',
        description='Get constituency parses of BabyLM test data for probing experiments')
    parser.add_argument('perturbation_class',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=["reverse", "hop", "negation", "agreement"],
                        help='Perturbation function used to transform BabyLM dataset')

    # Get args
    args = parser.parse_args()

    # Get all relevant test sentences
    test_sentences = []
    print("Getting sentences to parse...")
    if args.perturbation_class == "reverse":
        # For reversal, load original test sentences
        babylm_data = glob(f"{BABYLM_DATA_PATH}/babylm_data/babylm_test/*_dep.json")
        for file in babylm_data:
            print(file)
            f = open(file)
            data = json.load(f)
            f.close()

            # Get untagged test sentences
            for line in tqdm.tqdm(data):
                for sent in line["sent_annotations"]:
                    test_sentences.append(sent["sent_text"])
    else:
        # For other perturbations, get unaffected test sentences
        babylm_data = glob(
            f"{BABYLM_DATA_PATH}/babylm_data_perturbed/babylm_{args.perturbation_class}_control/babylm_test_unaffected_sents/*")
        for file in babylm_data:
            print(file)
            f = open(file)
            data = f.readlines()
            f.close()
            test_sentences.extend([line.strip() for line in data])

    # Remove short sentences
    MIN_SENTENCE_LEN = 50
    test_sentences = [sent for sent in test_sentences if len(
        sent) >= MIN_SENTENCE_LEN]

    # Init rng for sampling
    rng = np.random.default_rng(seed=15)
    N = len(test_sentences) if len(test_sentences) < 50000 else 50000
    test_sentences = rng.choice(test_sentences, size=N, replace=False)

    # Init Stanza NLP tools
    nlp = stanza.Pipeline(lang='en',
                          processors='tokenize,pos,constituency',
                          package="default_accurate",
                          use_gpu=True)

    # Get constituency parses
    parse_data = []
    for sent in tqdm.tqdm(test_sentences):
        constituency_parse, new_sent = __get_constituency_parse(
            sent, nlp, args.perturbation_class)
        if constituency_parse is not None:
            parse_data.append(new_sent + "\n")
            parse_data.append(constituency_parse + "\n")

    # Create directory
    parses_file = f"{args.perturbation_class}_parses.test"
    write_file("", parses_file, parse_data)
