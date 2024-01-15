# create__agreement_data.py
# Author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

import pandas as pd
from glob import glob
import re
from utils import gpt2_hop_tokenizer, BABYLM_DATA_PATH, marker_sg_token
from pluralizer import Pluralizer


if __name__ == "__main__":

    get_vocab_dict = []
    vocab = gpt2_hop_tokenizer.vocab

    # Patterns for finding simple "The SUBJ VERB x x x x MARKER" sequences
    control_pattern = f"(?:^| )(?:{vocab['The']}|{vocab['Ġthe']}) [1-9]+ [1-9]+ {marker_sg_token} [1-9]+ [1-9]+ [1-9]+ [1-9]+"
    words4_pattern = f"(?:^| )(?:{vocab['The']}|{vocab['Ġthe']}) [1-9]+ [1-9]+ [1-9]+ [1-9]+ [1-9]+ [1-9]+ {marker_sg_token}"

    # Get test file paths
    test_file_path = f'{BABYLM_DATA_PATH}/babylm_data_perturbed/' + \
        'babylm_hop_{}/babylm_test_affected/*'
    control_files = sorted(glob(test_file_path.format("control")))
    words4_files = sorted(glob(test_file_path.format("words4")))

    # Iterate over files and get candidate sequences for interventions
    candidate_sequences = []
    for control_file_path, words4_file_path in zip(control_files, words4_files):
        print(control_file_path.split("/")[-1])
        assert control_file_path.split(
            "/")[-1] == words4_file_path.split("/")[-1]

        # Iterate over pairs of lines in file (control and words4)
        control_lines = open(control_file_path, 'r').readlines()
        words4_lines = open(words4_file_path, 'r').readlines()
        for cl, wl in zip(control_lines, words4_lines):

            # Find all matches to patterns
            cseqs = re.findall(control_pattern, cl)
            wseqs = re.findall(words4_pattern, wl)

            # See if there is a shared pattern in control and words4 line
            for cseq, wseq in zip(re.findall(control_pattern, cl), re.findall(words4_pattern, wl)):
                if cseq.replace(" " + str(marker_sg_token), "") == wseq.replace(" " + str(marker_sg_token), ""):
                    candidate_sequences.append(
                        [int(s) for s in wseq.replace(
                            str(marker_sg_token), "").split()]
                    )

    # Init pluralizer
    pluralizer = Pluralizer()

    # Iterate over candidate sequences
    data = []
    for seq in candidate_sequences:

        # Get string version of sequence to get the subject
        splitted = gpt2_hop_tokenizer.decode(seq).split()

        # Get plural of subject (the word at index 1)
        splitted[1] = pluralizer.pluralize(splitted[1], 2, False)

        # Get GPT-2 tokens of plural version of sequence
        plur_seq = gpt2_hop_tokenizer.encode(" ".join(splitted))

        # If new subject form has a different number of tokens, skip
        if len(plur_seq) != len(seq):
            continue

        # In case " the" has changed to "the", reset the first token
        plur_seq[0] = seq[0]

        # If singular and plural sequence are the same, skip
        if seq == plur_seq:
            continue

        data.append([" ".join([str(s) for s in seq]),
                     " ".join([str(s) for s in plur_seq])])

    df = pd.DataFrame(data, columns=["Singular", "Plural"])
    df.to_csv("hop_agreement_data.csv")
