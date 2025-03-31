# babylm_dataset.py
# author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

import datasets
import logging
import os
import glob
import tqdm
from itertools import product

from utils import PERTURBATIONS, BABYLM_DATA_PATH

datasets.logging.set_verbosity_info()
logger = datasets.logging.get_logger(__name__)
logger.setLevel(logging.INFO)

_DESCRIPTION = """\
    Pre-tokenized BabyLM HuggingFace dataset for verb perturbations.
"""
_PERTURBED_DATA_PATH = f"{BABYLM_DATA_PATH}/babylm_data_perturbed"
_PERTURBATIONS = PERTURBATIONS.keys()
_TRAIN_SETS = ["100M", "10M"]
_EOS_TOKEN_ID = 50256


class BabyConfig(datasets.BuilderConfig):

    def __init__(self, data_dir, babylm_train_set, **kwargs):
        """BuilderConfig for IzParens

        Args:
          data_dir: path to directory of tokenized, perturbed BabyLM dataset
        """
        super(BabyConfig, self).__init__(
            **kwargs,
        )
        self.data_dir = data_dir
        self.babylm_train_set = babylm_train_set


class BabyLMCorpus(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BabyConfig(
            name=f"babylm_{perturbation}_{train_set}",
            data_dir=os.path.join(
                _PERTURBED_DATA_PATH, "babylm_" + perturbation),
            babylm_train_set=train_set,
        ) for perturbation, train_set in list(product(_PERTURBATIONS, _TRAIN_SETS))
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "input_ids": datasets.Sequence(datasets.Value(dtype="int32"))
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": os.path.join(
                    self.config.data_dir, "babylm_" + self.config.babylm_train_set), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_dir": os.path.join(
                    self.config.data_dir, "babylm_dev"), "split": "valid"},
            )
        ]

    def __chunk(self, sentences, eos_token):

        # Tokenize each sentence
        logger.info("Loading pre-tokenized data")
        tokenized_sentences = []
        for sent in tqdm.tqdm(sentences):
            tokenized_sentences.append([int(tok) for tok in sent.split()])

        # Concatenate the tokenized sentences using the EOS token
        logger.info("Concatenating tokenized data using EOS token")
        all_tokens = []
        for tokens in tqdm.tqdm(tokenized_sentences):
            all_tokens.extend(tokens)
            all_tokens.append(eos_token)

        # Chunk the tokens into sublists of max_seq_len tokens each
        logger.info("Chunking tokens into sublists of 1024")
        max_seq_len = 1024
        chunked_tokens = []
        for i in tqdm.tqdm(range(0, len(all_tokens), max_seq_len)):
            chunked_tokens.append(all_tokens[i:i + max_seq_len])

        # Drop last line if not a multiple of max_seq_len
        if len(chunked_tokens[-1]) < max_seq_len:
            chunked_tokens.pop()

        return chunked_tokens

    def _generate_examples(self, data_dir, split):
        """This function returns the BabyLM text in the discretized, tokenized form."""

        logger.info("Generating examples from = %s", data_dir)
        infiles = sorted(glob.glob(os.path.join(data_dir, "*")))

        # Extend sentences
        all_sentences = []
        for infile in infiles:
            f = open(infile, encoding="utf-8")
            all_sentences.extend(f.readlines())
        logger.info("Total sentences: {}".format(len(all_sentences)))

        # Tokenize and chunk
        tokenized_lines = self.__chunk(all_sentences, _EOS_TOKEN_ID)

        # Generate data
        logger.info("Writing dataset as space-separated sequences of tokens")
        for idx, line in enumerate(tokenized_lines):
            yield idx, {"input_ids": line}