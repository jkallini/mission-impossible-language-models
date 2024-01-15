# utils.py
# Author: Julie Kallini

from collections import deque
from string import punctuation
from transformers import AutoTokenizer, AddedToken
from functools import partial
from numpy.random import default_rng
from nltk.tree import ParentedTree
import torch


##############################################################################
# CONSTANTS
##############################################################################


BABYLM_SPLITS = ['100M', '10M', 'dev', 'test', 'unittest']
SEEDS = [21, 57, 84]
CHECKPOINTS = list(range(50, 501, 50))
GENRES = {
    "aochildes": "CHILDES",
    "bnc_spoken": "British National Corpus (BNC)",
    "cbt": "Children’s Book Test",
    "children_stories": "Children’s Stories Text Corpus",
    "gutenberg": "Standardized Project Gutenberg Corpus",
    "open_subtitles": "OpenSubtitles",
    "qed": "QCRI Educational Domain Corpus",
    "simple_wikipedia": "Simple Wikipedia",
    "switchboard": "Switchboard Dialog Act Corpus",
    "wikipedia": "Wikipedia"
}
CHECKPOINT_WRITE_PATH = "/nlp/scr3/nlp/llms-in-llms/babylm_models"
CHECKPOINT_READ_PATH = "/nlp/scr3/nlp/llms-in-llms/babylm_models"
BABYLM_DATA_PATH = "/nlp/scr3/nlp/llms-in-llms/babylm_data"
MARKER_HOP_SING = "🅂"
MARKER_HOP_PLUR = "🄿"
MARKER_REV = "🅁"
BOS_TOKEN = "<BOS_TOKEN>"
PART_TOKENS = set(["n't", "'ll", "'s", "'re", "'ve", "'m"])
PUNCT_TOKENS = set(punctuation)


##############################################################################
# PARENS MODELS (Structurally-pretrained)
##############################################################################


PAREN_MODEL_PATH = "/u/scr/isabelvp//tilt-stuff/tilt-finetuning/pretrained_checkpoints/"
PAREN_MODELS = {
    "CROSS": "flat-parens_vocab500-uniform_deplength-nesting-nolimit",
    "NEST": "nested-parens0.49_vocab500-uniform",
    "RAND": "random_vocab500-uniform",
}


##############################################################################
# HELPER FUNCTIONS
##############################################################################


def write_file(directory, filename, lines):
    f = open(directory + filename, "w")
    f.writelines(lines)
    f.close()


def get_gpt2_tokenizer_with_markers(marker_list):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # If no new markers to add, return normal tokenizer
    if len(marker_list) == 0:
        return tokenizer

    # Create tokens and return modified tokenizer
    new_tokens = []
    for marker in marker_list:
        new_tokens.append(AddedToken(marker, lstrip=True, rstrip=False))
    tokenizer.add_tokens(new_tokens)
    return tokenizer


gpt2_original_tokenizer = get_gpt2_tokenizer_with_markers([])


# GPT-2 hop tokenization
gpt2_hop_tokenizer = get_gpt2_tokenizer_with_markers(
    [MARKER_HOP_SING, MARKER_HOP_PLUR])
# Get ids of marker tokens
marker_sg_token = gpt2_hop_tokenizer.get_added_vocab()[
    MARKER_HOP_SING]
marker_pl_token = gpt2_hop_tokenizer.get_added_vocab()[
    MARKER_HOP_PLUR]


# GPT-2 reverse tokenization
gpt2_rev_tokenizer = get_gpt2_tokenizer_with_markers(
    [MARKER_REV])
# Get ids of marker tokens
marker_rev_token = gpt2_rev_tokenizer.get_added_vocab()[
    MARKER_REV]

# GPT-2 determiner tokenization
gpt2_det_tokenizer = get_gpt2_tokenizer_with_markers(
    [BOS_TOKEN])
# Get id of BOS token
bos_token_id = gpt2_det_tokenizer.get_added_vocab()[BOS_TOKEN]


MARKER_TOKEN_IDS = [marker_sg_token, marker_pl_token, marker_rev_token]


def compute_surprisals(model, input_ids):
    # Get the log probabilities from the model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1]
        shifted_input_ids = input_ids[:, 1:]

        # Get the log probabilities for the actual next tokens
        log_probs = torch.log2(torch.nn.functional.softmax(logits, dim=-1))
        true_log_probs = log_probs.gather(
            2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

    # Get the negative log probabilities
    neg_log_probs = (-true_log_probs).tolist()
    surprisals = [[None] + probs for probs in neg_log_probs]
    return surprisals


def compute_token_probabilities(model, input_ids, token_id, pad_token_id):
    # Get the log probabilities from the model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1]
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get the probabilities for the specified token at each position
        token_probs = probs[:, :, token_id]

    # Convert to list and add None at the beginning to align with input tokens
    # Put null probability for instances of pad token
    token_probs_list = []
    for batch_i, probs in enumerate(token_probs):
        input_ids_seq = input_ids[batch_i].tolist() + [pad_token_id]
        filtered = [p if input_ids_seq[pos_i+1] !=
                    pad_token_id else None for pos_i, p in enumerate(probs.tolist())]
        token_probs_list.append([None] + filtered)

    return token_probs_list


def merge_part_tokens(words):
    result = []
    for s in words:
        if result and s in PART_TOKENS and len(result) > 0:
            result[-1] += s
        else:
            result.append(s)
    return result


def __affect_hop_word(word):
    return word["feats"] and "Person=3" in word["feats"] \
        and "Tense=Pres" in word["feats"] \
        and "VerbForm=Fin" in word["feats"] \
        and "Number" in word["feats"]


def __perturb_hop_words(sent, num_hops, marker_sg, marker_pl):
    perturbed_tokens, _ = __perturb_hop_words_complete_hops(
        sent, num_hops, marker_sg, marker_pl)
    return perturbed_tokens


def check_word_hops_completed(sent, num_hops=4, marker=MARKER_HOP_SING):
    _, hops_completed = __perturb_hop_words_complete_hops(
        sent, num_hops, marker, marker)
    return hops_completed


def __perturb_hop_words_complete_hops(sent, num_hops, marker_sg, marker_pl):

    word_annotations = sent["word_annotations"].copy()
    word_annotations.reverse()

    hop_completed = []
    new_sent = []
    for word in word_annotations:

        # Identify 3.pres verbs
        if __affect_hop_word(word):

            # Lemmatize verb if possible
            new_sent.append(
                word["lemma"] if word["lemma"] is not None else word["text"])

            # Marker hopping logic
            insert_index = len(new_sent)-1
            skipped_words = 0
            while skipped_words < num_hops and insert_index > 0:

                # Handle edge case when punctuation (or sequence of
                # punctuation) begin the sentence
                if (not any([c.isalnum() for c in
                             "".join(new_sent[:insert_index])])):
                    break

                # Count word as skipped if it is not a special token
                if (new_sent[insert_index] not in PART_TOKENS) and \
                        (not set(new_sent[insert_index]).issubset(PUNCT_TOKENS)):
                    skipped_words += 1
                insert_index -= 1

            # Handle edge case when insert index is punctuation (and this is not
            # sentence-initial punctuation)
            if any([c.isalnum() for c in
                    "".join(new_sent[:insert_index])]):
                while insert_index != 0 and (new_sent[insert_index] in PART_TOKENS
                                             or set(new_sent[insert_index]).issubset(PUNCT_TOKENS)):
                    insert_index -= 1

            # Handle edge case when token before insert index is part/aux token
            if insert_index != 0 and new_sent[insert_index-1] in PART_TOKENS:
                insert_index -= 1

            # Log if this sentence had all full hops
            hop_completed.append(skipped_words == num_hops)

            # Use correct marker for singular vs. plural
            if "Number=Sing" in word["feats"]:
                new_sent.insert(insert_index, marker_sg)
            elif "Number=Plur" in word["feats"]:
                new_sent.insert(insert_index, marker_pl)
            else:
                raise Exception(
                    "Number not in verb features\n" + sent["sent_text"])

        else:
            new_sent.append(word["text"])

    new_sent.reverse()
    sent_string = " ".join(merge_part_tokens(new_sent))
    tokens = gpt2_hop_tokenizer.encode(sent_string)
    return tokens, all(hop_completed) and len(hop_completed) > 0


def __perturb_hop_tokens(sent, num_hops):

    word_annotations = sent["word_annotations"].copy()
    word_annotations.reverse()

    new_sent = deque()
    tokens = []
    for word in word_annotations:

        # Identify 3.pres verbs
        if __affect_hop_word(word):

            # Lemmatize verb if possible
            lemma = word["lemma"] if word["lemma"] is not None else word["text"]

            if len(new_sent) > 0 and new_sent[0] in PART_TOKENS:
                lemma = lemma + new_sent[0]
                new_sent.popleft()

            if len(new_sent) > 0:
                sent_string = " ".join(merge_part_tokens(new_sent))
                tokens = gpt2_hop_tokenizer.encode(
                    " " + sent_string) + tokens

            # Use correct marker for singular vs. plural
            if "Number=Sing" in word["feats"]:
                tokens.insert(num_hops, marker_sg_token)
            elif "Number=Plur" in word["feats"]:
                tokens.insert(num_hops, marker_pl_token)
            else:
                raise Exception(
                    "Number not in verb features\n" + sent["sent_text"])

            new_sent = deque()
            new_sent.append(lemma)

        else:
            new_sent.appendleft(word["text"])

    if len(new_sent) > 0:
        sent_string = " ".join(merge_part_tokens(new_sent))
        tokens = gpt2_hop_tokenizer.encode(sent_string) + tokens
    return tokens


def __perturb_reverse(sent, rng, reverse, full):

    # Get sentence text and GPT-2 tokens
    tokens = gpt2_rev_tokenizer.encode(sent["sent_text"])

    # Pick random index to insert REV token
    i = rng.choice(len(tokens)+1)
    tokens.insert(i, marker_rev_token)

    # Extract tokens before/after the marker, and reverse tokens after
    tokens_before = tokens[:i+1]
    tokens_after = tokens[i+1:]
    if reverse:
        tokens_after.reverse()
    new_tokens = tokens_before + tokens_after
    if full:
        assert not reverse
        new_tokens.reverse()

    return new_tokens


def __perturb_shuffle_deterministic(sent, seed, shuffle):
    # Get sentence text and GPT-2 tokens
    tokens = gpt2_original_tokenizer.encode(sent["sent_text"])
    if shuffle:
        default_rng(seed).shuffle(tokens)
    return tokens


def __perturb_shuffle_nondeterministic(sent, rng):
    # Get sentence text and GPT-2 tokens
    tokens = gpt2_original_tokenizer.encode(sent["sent_text"])
    rng.shuffle(tokens)
    return tokens


def __perturb_shuffle_local(sent, seed, window=5):
    # Get sentence text and GPT-2 tokens
    tokens = gpt2_original_tokenizer.encode(sent["sent_text"])

    # Shuffle tokens in batches of size window
    shuffled_tokens = []
    for i in range(0, len(tokens), window):
        batch = tokens[i:i+window].copy()
        default_rng(seed).shuffle(batch)
        shuffled_tokens += batch

    return shuffled_tokens


def __perturb_shuffle_even_odd(sent):
    # Get sentence text and GPT-2 tokens
    tokens = gpt2_original_tokenizer.encode(sent["sent_text"])
    even = [tok for i, tok in enumerate(tokens) if i % 2 == 0]
    odd = [tok for i, tok in enumerate(tokens) if i % 2 != 0]
    return even + odd


##############################################################################
# AFFECT FUNCTIONS
# These functions define when a perturbation has been applied to a sentence
# not. This is used for identifying which test sentences have been
# altered to separate affected vs. unaffected senences. Affect functions are
# functions of the input sentence object and return a boolean.
##############################################################################


def affect_hop(sent):
    return any([__affect_hop_word(word) for word in sent['word_annotations']]) \
        and sent["constituency_parse"] is not None


def affect_reverse(sent):
    return True


def affect_shuffle(sent):
    return True


def affect_none(sent):
    return False


##############################################################################
# FILTER FUNCTIONS
# These functions define when an affected sentence should be included in the
# final dataset. For instance, hop perturbations where the marker is placed
# at the end of the sentence should be excluded. A filter function returns
# True if an affected sentence should be included in the dataset.
##############################################################################


def filter_hop(sent):
    # Assertion needed since filter function is only defined for affected
    # sentences
    assert (affect_hop(sent))
    return check_word_hops_completed(sent, 4)


def filter_reverse(sent):
    return True


def filter_shuffle(sent):
    tokens = gpt2_original_tokenizer.encode(sent["sent_text"])
    return len(tokens) > 1 and len(tokens) <= 350


def filter_none(sent):
    return False


##############################################################################
# PERTURBATION FUNCTIONS
# These functions define how a perturbation will affect a sentence. They
# take in a sentence object and an optional marker
# for verb transformations. They return a string representing the transformed
# sentence.
##############################################################################


def perturb_hop_words4(sent):
    return __perturb_hop_words(sent, 4, MARKER_HOP_SING, MARKER_HOP_PLUR)


def perturb_hop_tokens4(sent):
    return __perturb_hop_tokens(sent, 4)


def perturb_hop_control(sent):
    return __perturb_hop_tokens(sent, 0)


def perturb_reverse(sent, rng, reverse=True, full=False):
    return __perturb_reverse(sent, rng, reverse, full)


def perturb_shuffle_deterministic(sent, seed=None, shuffle=True):
    return __perturb_shuffle_deterministic(sent, seed, shuffle)


def perturb_shuffle_nondeterministic(sent, rng):
    return __perturb_shuffle_nondeterministic(sent, rng)


def perturb_shuffle_local(sent, seed, window):
    return __perturb_shuffle_local(sent, seed, window)


def perturb_shuffle_even_odd(sent):
    return __perturb_shuffle_even_odd(sent)


##############################################################################
# PERTURBATIONS
# This dict maps the name of a perturbation to its perturbation and filter
# functions. The names and functions in this dict are used throughout the
# repo.
##############################################################################


PERTURBATIONS = {
    "shuffle_control": {
        "perturbation_function": partial(perturb_shuffle_deterministic, seed=None, shuffle=False),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": gpt2_original_tokenizer,
        "color": "#606060",
    },
    "shuffle_nondeterministic": {
        "perturbation_function": partial(perturb_shuffle_nondeterministic, rng=default_rng(0)),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": gpt2_original_tokenizer,
        "color": "#E8384F",
    },
    "shuffle_deterministic21": {
        "perturbation_function": partial(perturb_shuffle_deterministic, seed=21, shuffle=True),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": gpt2_original_tokenizer,
        "color": "#FFB000",
    },
    "shuffle_deterministic57": {
        "perturbation_function": partial(perturb_shuffle_deterministic, seed=57, shuffle=True),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": gpt2_original_tokenizer,
        "color": "#8db000",
    },
    "shuffle_deterministic84": {
        "perturbation_function": partial(perturb_shuffle_deterministic, seed=84, shuffle=True),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": gpt2_original_tokenizer,
        "color": "#62BB35",
    },
    "shuffle_local3": {
        "perturbation_function": partial(perturb_shuffle_local, seed=0, window=3),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": gpt2_original_tokenizer,
        "color": "#208EA3",
    },
    "shuffle_local5": {
        "perturbation_function": partial(perturb_shuffle_local, seed=0, window=5),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": gpt2_original_tokenizer,
        "color": "#4178BC",
    },
    "shuffle_local10": {
        "perturbation_function": partial(perturb_shuffle_local, seed=0, window=10),
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": gpt2_original_tokenizer,
        "color": "#AA71FF",
    },
    "shuffle_even_odd": {
        "perturbation_function": perturb_shuffle_even_odd,
        "affect_function": affect_shuffle,
        "filter_function": filter_shuffle,
        "gpt2_tokenizer": gpt2_original_tokenizer,
        "color": "#E37CFF",
    },
    "reverse_control": {
        "perturbation_function": partial(perturb_reverse, rng=default_rng(21), reverse=False, full=False),
        "affect_function": affect_reverse,
        "filter_function": filter_reverse,
        "gpt2_tokenizer": gpt2_rev_tokenizer,
        "color": "#606060",
    },
    "reverse_partial": {
        "perturbation_function": partial(perturb_reverse, rng=default_rng(21), reverse=True, full=False),
        "affect_function": affect_reverse,
        "filter_function": filter_reverse,
        "gpt2_tokenizer": gpt2_rev_tokenizer,
        "color": "#E5A836",
    },
    "reverse_full": {
        "perturbation_function": partial(perturb_reverse, rng=default_rng(21), reverse=False, full=True),
        "affect_function": affect_reverse,
        "filter_function": filter_reverse,
        "gpt2_tokenizer": gpt2_rev_tokenizer,
        "color": "#A348A6",
    },
    "hop_control": {
        "perturbation_function": perturb_hop_control,
        "affect_function": affect_hop,
        "filter_function": filter_hop,
        "gpt2_tokenizer": gpt2_hop_tokenizer,
        "color": "#606060",
    },
    "hop_tokens4": {
        "perturbation_function": perturb_hop_tokens4,
        "affect_function": affect_hop,
        "filter_function": filter_hop,
        "gpt2_tokenizer": gpt2_hop_tokenizer,
        "color": "#fa8128", 
    },
    "hop_words4": {
        "perturbation_function": perturb_hop_words4,
        "affect_function": affect_hop,
        "filter_function": filter_hop,
        "gpt2_tokenizer": gpt2_hop_tokenizer,
        "color": "#03a0ff",
    },
}
