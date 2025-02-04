# tag.py
# Author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

import pytest
import glob
import tqdm
import os
import argparse
import stanza
import json


test_all_files = sorted(glob.glob("babylm_data/babylm_*/*"))
test_original_files = [f for f in test_all_files if ".json" not in f]
test_json_files = [f for f in test_all_files if "_parsed.json" in f]
test_cases = list(zip(test_original_files, test_json_files))


@pytest.mark.parametrize("original_file, json_file", test_cases)
def test_equivalent_lines(original_file, json_file):

    # Read lines of file and remove all whitespace
    original_file = open(original_file)
    original_data = "".join(original_file.readlines())
    original_data = "".join(original_data.split())

    json_file = open(json_file)
    json_lines = json.load(json_file)
    json_data = ""
    for line in json_lines:
        for sent in line["sent_annotations"]:
            json_data += sent["sent_text"]
    json_data = "".join(json_data.split())

    # Test equivalence
    assert (original_data == json_data)


def __get_constituency_parse(sent, nlp):

    # Try parsing the doc
    try:
        parse_doc = nlp(sent.text)
    except:
        return None
    
    # Get set of constituency parse trees
    parse_trees = [str(sent.constituency) for sent in parse_doc.sentences]

    # Join parse trees and add ROOT
    constituency_parse = "(ROOT " + " ".join(parse_trees) + ")"
    return constituency_parse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Tag BabyLM dataset',
        description='Tag BabyLM dataset using Stanza')
    parser.add_argument('path', type=argparse.FileType('r'),
                        nargs='+', help="Path to file(s)")
    parser.add_argument('--constituency_parse', action='store_true',
                        help="Include constituency parse")
    parser.add_argument('--dependency_parse', action='store_true',
                        help="Include dependency parse")
    # Get args
    args = parser.parse_args()

    # Init Stanza NLP tools
    nlp1 = stanza.Pipeline(
        lang='en',
        processors='tokenize,pos,lemma,depparse' if args.dependency_parse else 'tokenize,pos,lemma',
        # package="default_accurate",
        use_gpu=True)

    # If constituency parse is needed, init second Stanza parser
    if args.constituency_parse:
        nlp2 = stanza.Pipeline(lang='en',
                               processors='tokenize,pos,constituency',
                               package="default_accurate",
                               use_gpu=True)

    BATCH_SIZE = 5000

    # Iterate over BabyLM files
    for file in args.path:

        print(file.name)
        lines = file.readlines()

        # Strip lines and join text
        print("Concatenating lines...")
        lines = [l.strip() for l in lines]
        line_batches = [lines[i:i + BATCH_SIZE]
                        for i in range(0, len(lines), BATCH_SIZE)]
        text_batches = [" ".join(l) for l in line_batches]

        # Iterate over lines in file and track annotations
        line_annotations = []
        print("Segmenting and parsing text batches...")
        for text in tqdm.tqdm(text_batches):
            # Tokenize text with stanza
            doc = nlp1(text)

            # Iterate over sents in the line and track annotations
            sent_annotations = []
            for sent in doc.sentences:

                # Iterate over words in sent and track annotations
                word_annotations = []
                for token, word in zip(sent.tokens, sent.words):
                    wa = {
                        'id': word.id,
                        'text': word.text,
                        'lemma': word.lemma,
                        'upos': word.upos,
                        'xpos': word.xpos,
                        'feats': word.feats,
                        'start_char': token.start_char,
                        'end_char': token.end_char,
                        # Add dependency parse-specific fields
                        'feats': word.feats if args.dependency_parse else None,
                        'head': word.head if args.dependency_parse else None,
                        'deprel': word.deprel if args.dependency_parse else None,
                    }
                    word_annotations.append(wa)  # Track word annotation

                # Get constituency parse if needed
                if args.constituency_parse:
                    constituency_parse = __get_constituency_parse(sent, nlp2)
                    sa = {
                        'sent_text': sent.text,
                        'constituency_parse': constituency_parse,
                        'word_annotations': word_annotations,
                    }
                else:
                    sa = {
                        'sent_text': sent.text,
                        'word_annotations': word_annotations,
                    }
                sent_annotations.append(sa)  # Track sent annotation

            la = {
                'sent_annotations': sent_annotations
            }
            line_annotations.append(la)  # Track line annotation

        # Write annotations to file as a JSON
        print("Writing JSON outfile...")
        ext = ".json"
        if args.constituency_parse:
            ext = "_const" + ext
        if args.dependency_parse:
            ext = "_dep" + ext
        json_filename = os.path.splitext(file.name)[0] + ext
        with open(json_filename, "w") as outfile:
            json.dump(line_annotations, outfile, indent=4)
