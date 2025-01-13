# syntaxgym_utils.py
# Author: Julie Kallini

from datasets import load_dataset
from utils import compute_surprisals


def load_syntaxgym_data(suite_name):
    dataset = load_dataset("cpllab/syntaxgym", trust_remote_code=True)
    return dataset["test"].filter(lambda example: example["suite_name"] == suite_name)


def get_target_surprisal(tokenizer, model, content, target_loc, pythia=False):
    bos = '<|endoftext|> ' if pythia else ''
    prefix_len = len(tokenizer.encode(bos + " ".join(content[:target_loc]), return_tensors="pt")[0])
    tokenized_content = tokenizer.encode(bos + " ".join(content), return_tensors="pt")
    surprisals = compute_surprisals(model, tokenized_content.to(model.device))[0]
    return surprisals[prefix_len]


def get_condition_to_region_map(item):
    return {
        condition_name: region
        for condition_name, region in zip(item['conditions']['condition_name'], item['conditions']['regions'])
    }


def evaluate_agreement(tokenizer, model, subset, target_loc, local_agreement=False, pythia=False):
    total = 0
    sing_correct = 0
    plural_correct = 0

    for item in subset:
        condition_to_region_map = get_condition_to_region_map(item)

        match_sing = get_target_surprisal(tokenizer, model, condition_to_region_map['match_sing']['content'], target_loc, pythia)
        mismatch_sing = get_target_surprisal(tokenizer, model, condition_to_region_map['mismatch_sing']['content'], target_loc, pythia)
        match_plural = get_target_surprisal(tokenizer, model, condition_to_region_map['match_plural']['content'], target_loc, pythia)
        mismatch_plural = get_target_surprisal(tokenizer, model, condition_to_region_map['mismatch_plural']['content'], target_loc, pythia)

        if not local_agreement:
            if match_sing < mismatch_sing:
                sing_correct += 1
            if match_plural < mismatch_plural:
                plural_correct += 1
        else:
            if match_sing > mismatch_sing:
                sing_correct += 1
            if match_plural > mismatch_plural:
                plural_correct += 1
        total += 1

    return {
        "singular_acc": sing_correct / total,
        "plural_acc": plural_correct / total
    }


def evaluate_npi_licensing(tokenizer, model, subset, target_loc, local_agreement=False, pythia=False):

    def swap_has_have(content):
        if "have" in content[target_loc-1]:
            content[target_loc-1] = content[target_loc-1].replace("have", "has")
        elif "has" in content[target_loc-1]:
            content[target_loc-1] = content[target_loc-1].replace("has", "have")
        return content

    total = 0
    pred1_correct = 0
    pred2_correct = 0
    pred3_correct = 0

    for item in subset:
        condition_to_region_map = get_condition_to_region_map(item)

        pos_pos_content = condition_to_region_map['pos_pos']['content']
        pos_neg_content = condition_to_region_map['pos_neg']['content']
        neg_pos_content = condition_to_region_map['neg_pos']['content']
        neg_neg_content = condition_to_region_map['neg_neg']['content']

        if local_agreement:
            pos_pos_content = swap_has_have(pos_pos_content)
            pos_neg_content = swap_has_have(pos_neg_content)
            neg_pos_content = swap_has_have(neg_pos_content)
            neg_neg_content = swap_has_have(neg_neg_content)

        pos_pos = get_target_surprisal(tokenizer, model, pos_pos_content, target_loc, pythia)
        pos_neg = get_target_surprisal(tokenizer, model, pos_neg_content, target_loc, pythia)
        neg_pos = get_target_surprisal(tokenizer, model, neg_pos_content, target_loc, pythia)
        neg_neg = get_target_surprisal(tokenizer, model, neg_neg_content, target_loc, pythia)

        if neg_neg < pos_neg:
            pred1_correct += 1
        if neg_pos < pos_neg:
            pred2_correct += 1
        if neg_pos < pos_pos:
            pred3_correct += 1
        total += 1

    return {
        "pred1_acc": pred1_correct / total,
        "pred2_acc": pred2_correct / total,
        "pred3_acc": pred3_correct / total
    }


def evaluate_reflexive(tokenizer, model, subset, target_loc, local_agreement=False, pythia=False):
    total = 0
    sing_correct = 0
    plural_correct = 0

    for item in subset:
        condition_to_region_map = get_condition_to_region_map(item)

        match_sing = get_target_surprisal(tokenizer, model, condition_to_region_map['match_sing']['content'], target_loc, pythia)
        mismatch_sing = get_target_surprisal(tokenizer, model, condition_to_region_map['mismatch_sing']['content'], target_loc, pythia)
        match_plural = get_target_surprisal(tokenizer, model, condition_to_region_map['match_plural']['content'], target_loc, pythia)
        mismatch_plural = get_target_surprisal(tokenizer, model, condition_to_region_map['mismatch_plural']['content'], target_loc, pythia)

        if match_sing < mismatch_sing:
            sing_correct += 1
        if match_plural < mismatch_plural:
            plural_correct += 1
        total += 1

    return {
        "singular_acc": sing_correct / total,
        "plural_acc": plural_correct / total
    }


SYNTAXGYM_EVALS = {
    "agreement": {
        "number_prep": {
            "eval_function": evaluate_agreement,
            "target_loc": 5,
        },
        "number_orc": {
            "eval_function": evaluate_agreement,
            "target_loc": 6,
        },
        "number_src": {
            "eval_function": evaluate_agreement,
            "target_loc": 6,
        },
    },
    "licensing": {
        "npi_orc_any": {
            "eval_function": evaluate_npi_licensing,
            "target_loc": 7,
        },
        "npi_src_any": {
            "eval_function": evaluate_npi_licensing,
            "target_loc": 7,
        },
        "npi_orc_ever": {
            "eval_function": evaluate_npi_licensing,
            "target_loc": 7,
        },
        "npi_src_ever": {
            "eval_function": evaluate_npi_licensing,
            "target_loc": 7,
        },
        "reflexive_prep_fem": {
            "eval_function": evaluate_reflexive,
            "target_loc": 6,
        },
        "reflexive_orc_fem": {
            "eval_function": evaluate_reflexive,
            "target_loc": 7,
        },
        "reflexive_src_fem": {
            "eval_function": evaluate_reflexive,
            "target_loc": 7,
        },
        "reflexive_prep_masc": {
            "eval_function": evaluate_reflexive,
            "target_loc": 6,
        },
        "reflexive_orc_masc": {
            "eval_function": evaluate_reflexive,
            "target_loc": 7,
        },
        "reflexive_src_masc": {
            "eval_function": evaluate_reflexive,
            "target_loc": 7,
        },
    }
}