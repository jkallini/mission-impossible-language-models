# tree_projection.py
# Authors: Ananth Agarwal, Julie Kallini
# Adapted from: https://github.com/MurtyShikhar/TreeProjections/blob/main/tree_projection_src/tree_projection.py
"""
Modified tree projection code that supports:
    (1) HF models.
    (2) Pre-tokenized input.
"""

from typing import List, Dict, Tuple, Union, Any, Callable
from tqdm import tqdm
import torch
import torch.nn.functional as F
import random
from scipy.spatial import distance
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np

device = torch.device("cuda")


def decode_parse(data: Union[int, Tuple[Any, Any]], tokenizer: PreTrainedTokenizer) -> Union[str, Tuple]:
    """
    Recursively decodes nested tuples of token IDs into human-readable text.

    Args:
        data: A nested tuple of integers (token IDs) or more tuples.
        tokenizer: A Hugging Face tokenizer.

    Returns:
        A nested structure where token IDs are replaced with decoded strings.
    """
    if isinstance(data, int):
        return tokenizer.decode([data])  # Decode single token ID
    elif isinstance(data, tuple):
        return tuple(decode_parse(sub, tokenizer) for sub in data)
    else:
        raise ValueError("Unsupported data type. Expected int or tuple.")


def get_all_hidden_states_scratch(
    model: PreTrainedModel,
    input_list: List[List[int]],
    attention_mask_list: List[List[int]],
    input_masks: Union[None, List[List[int]]] = None,
    sum_all: bool = False,
    tqdm_disable: bool = False,
    pre_tokenized: Tuple[List[int], List[Tuple[int, int]]] = None,
    start_relax_layer: int = 0,
    layer_id: int = -1,
) -> List[torch.Tensor]:
    """
    Extracts hidden states from a transformer model.

    Args:
        model: The pre-trained transformer model.
        input_list: List of tokenized input sequences.
        attention_mask_list: List of attention masks.
        input_masks: Optional custom attention masks for specific layers.
        sum_all: If True, sums all hidden states.
        tqdm_disable: If True, disables progress bars.
        pre_tokenized: Tuple containing pre-tokenized words and their indices.
        start_relax_layer: Layer index where masking starts being applied.
        layer_id: Layer index from which to extract hidden states.

    Returns:
        A list of extracted hidden state tensors.
    """
    if not pre_tokenized:
        raise ValueError("pre_tokenized must be specified")

    _, idxs = pre_tokenized
    st_p = idxs[0][0][0]
    en_p = idxs[0][-1][-1]
    assert len(set(map(lambda x: (x[0][0], x[-1][-1]), idxs))) == 1

    def t_shaped_mask(mask, relax_mask, num_layers):
        ### relax mask only masks padded stuff
        #### relax mask from 0 ... start_relax_layer-1,
        #### mask from start_relax_layer to num_layers - 1
        return [relax_mask] * start_relax_layer + [mask] * (
            num_layers - start_relax_layer
        )

    hidden_states_all = []
    batch_size = 128
    st = 0

    if hasattr(model.config, "n_layer"):
        num_layers = model.config.n_layer
    elif hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    with tqdm(total=len(input_list), disable=tqdm_disable) as progress_bar:
        while st < len(input_list):
            en = min(len(input_list), st + batch_size)
            cslice = input_list[st:en]
            input_ids = torch.tensor(input_list[st:en], device=device)
            attention_mask = torch.tensor(attention_mask_list[st:en], device=device)
            # input masks specify the inner context
            if input_masks is not None:
                masks_curr = input_masks[st:en]
                masks_padded = []
                input_len = input_ids.shape[-1]
                for mask in masks_curr:
                    mask_padded = F.pad(
                        torch.tensor(mask), (0, input_len - len(mask))
                    ).unsqueeze(0)
                    masks_padded.append(mask_padded)
                tree_mask = torch.cat(masks_padded, dim=0).to(device)
                attention_masks = t_shaped_mask(
                    tree_mask, attention_mask, num_layers
                )
                mask_mult = tree_mask.unsqueeze(-1)
            else:
                attention_masks = [attention_mask] * num_layers
                mask_mult = attention_mask.unsqueeze(-1)
            model.eval()
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, attention_masks=attention_masks, output_hidden_states=True
                )

            # remove vectors for masked stuff
            hidden_states = (outputs.hidden_states[layer_id] * mask_mult).cpu()
            if sum_all:
                hidden_states = hidden_states[st_p:en_p].sum(dim=-2)
            # For simplicity, flatten batch dimension to fit other code
            for batch_item_hs in hidden_states:
                hidden_states_all.append(batch_item_hs)
            progress_bar.update(en - st)
            st = en

    return hidden_states_all


def measure_sim_factory(distance_fn: Callable[[np.ndarray, np.ndarray], float]) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Factory function to create a similarity measurement function based on a distance function.
    
    Args:
        distance_fn: A distance function (e.g., Euclidean or Cosine).
    
    Returns:
        A function that computes similarity between two matrices.
    """
    def measure_sim(m1, m2):
        if m1.ndim == 2:
            assert len(m1) == len(m2)
            return sum(measure_sim(m1[idx], m2[idx]) for idx in range(len(m1))) / (
                1.0 * len(m1)
            )
        elif distance_fn == distance.cosine:
            return 1.0 - distance_fn(m1, m2)
        else:
            return -1.0 * distance_fn(m1, m2)

    return measure_sim


def get_pre_tokenized_info(input_ids: List[int]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Generate a trivial list of start and end indices for words.
    Assumes each token is a separate word (no subword splitting).

    Args:
        input_ids: List of token IDs from a tokenizer.
    
    Returns:
        Tuple containing:
            - The input token IDs (unchanged).
            - A list of (start, end) indices for each token.
    """
    idxs = [(i, i + 1) for i in range(len(input_ids))]  # Each token is its own word
    return input_ids, idxs


class TreeProjection:
    def __init__(self, model: PreTrainedModel, sim_fn: str = "cosine", normalize: bool = True) -> None:
        """
        Initialize TreeProjection class for computing hierarchical structures from token representations.

        Args:
            model: A Hugging Face transformer model.
            sim_fn: Similarity function to use for computing distances.
            normalize: If True, subtracts a baseline to prevent trivial encodings from scoring highly.
        """

        self.model = model
        self.sim_fn = sim_fn
        self.normalize = normalize

    def _create_all_possible_attention_masks(self, input_ids: List[int]) -> Dict[Tuple[int, int], List[int]]:
        """
        Compute all possible attention masks for tokenized input.
        Each mask at index (i, j) allows attention only within the span (i, j).
        
        Args:
            input_ids: Tokenized input sequence.
        
        Returns:
            A dictionary mapping (start, end) indices to attention masks.
        """
        num_tokens = len(input_ids)
        
        def generate_attention_mask(st, en):
            """ Generate an attention mask where only tokens from index st to en-1 can be attended to. """
            mask = [0] * num_tokens
            for i in range(st, en):
                mask[i] = 1  # Allow attention within the span
            return mask
        
        all_inputs = {}
        # Generate masks for all spans (i, j)
        for l in range(1, num_tokens + 1):
            for st in range(num_tokens - l + 1):
                en = st + l  # End index (exclusive)
                all_inputs[(st, en - 1)] = generate_attention_mask(st, en)
        
        return all_inputs

    ### implements Algorithm-1 from the Appendix
    def _tree_projection(
        self,
        chart_values,
        input_ids,
        projection_algorithm,
        normalize=False,
        is_leaf_fn=None,
        is_invalid_fn=None,
    ):
        num_words = len(input_ids)

        def tree_projection_dp(word_list, st, en, randomize=False):
            if randomize:
                raise NotImplementedError("Randomized DP not implemented")
            tree_scores = {}
            for l in range(1, num_words + 1):
                for st in range(num_words - l + 1):
                    en = st + l - 1
                    if st == en:
                        tree_scores[(st, en)] = (word_list[st], 0.0)
                        continue
                    t_score_per_k = []
                    parses = {}
                    for k in range(st, en):
                        p1, s1 = tree_scores[(st, k)]
                        p2, s2 = tree_scores[(k + 1, en)]
                        curr_val = chart_values[(st, k)] + chart_values[(k + 1, en)]
                        t_score_per_k.append(curr_val + s1 + s2)
                        parses[k] = (p1, p2)
                    argmax = np.argmax(t_score_per_k)
                    best_val = t_score_per_k[argmax]
                    best_split = st + argmax
                    if normalize:
                        rand_split = random.choice(range(st, en))
                        rand_val = (
                            chart_values[(st, rand_split)]
                            + chart_values[(rand_split + 1, en)]
                        )
                        best_val -= rand_val
                    tree_scores[(st, en)] = (parses[best_split], best_val)
            return tree_scores[(0, num_words - 1)]

        def tree_projection_recurse(word_list, st, en, randomize=False):
            if is_leaf_fn is not None and is_leaf_fn(word_list, st, en):
                return " ".join(word_list[st : en + 1])
            elif st == en:
                return word_list[st], 0.0
            else:
                curr_split = st
                best_val = -10000
                if randomize:
                    curr_split = random.choice(range(st, en))
                else:
                    for k in range(st, en):
                        if is_invalid_fn is not None and is_invalid_fn(
                            word_list, st, k, en
                        ):
                            continue
                        curr_val = chart_values[(st, k)] + chart_values[(k + 1, en)]
                        if curr_val > best_val:
                            best_val = curr_val
                            curr_split = k
                p1, s1 = tree_projection_recurse(word_list, st, curr_split)
                p2, s2 = tree_projection_recurse(word_list, curr_split + 1, en)
                if normalize:
                    rand_split = random.choice(range(st, en))
                    rand_val = (
                        chart_values[(st, rand_split)]
                        + chart_values[(rand_split + 1, en)]
                    )
                    best_val -= rand_val

                return (p1, p2), s1 + s2 + best_val

        if projection_algorithm == "dp":
            f = tree_projection_dp
        else:
            f = tree_projection_recurse
        parse, score = f(input_ids, 0, num_words - 1)

        return parse, score

    def compute_sci_chart(
        self,
        input_ids: List[int],
        attention_mask: List[int],
        st_threshold: int,
        layer_id: int = -1,
        span_formula: bool = False,
        mask_free_sci: bool = False
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute the Structural Contextual Invariance (SCI) chart for an input sequence.
        
        Args:
            input_ids: Tokenized input sequence.
            attention_mask: Attention mask for input.
            st_threshold: Layer threshold for relaxation.
            layer_id: Layer at which to extract embeddings.
            span_formula: Whether to use a specific span formulation.
            mask_free_sci: If True, computes SCI without masking outer context.
        
        Returns:
            A dictionary mapping spans (start, end) to their SCI scores.
        """
        sent_tokens, idxs = get_pre_tokenized_info(input_ids)
        # contextual vectors (no mask)
        outer_context_vecs = get_all_hidden_states_scratch(
            self.model,
            [input_ids],
            [attention_mask],
            tqdm_disable=True,
            pre_tokenized=([sent_tokens], [idxs]),
            layer_id=layer_id,
        )


        all_input_masks_dict = self._create_all_possible_attention_masks(input_ids)
        span_idxs = list(all_input_masks_dict.keys())
        if not mask_free_sci:
            input_masks = list(all_input_masks_dict.values())
            num_input_masks = len(input_masks)
            # one for each mask
            input_list = [input_ids] * num_input_masks
            attention_mask_list = [attention_mask] * num_input_masks

            # context free vectors
            inner_context_vecs = get_all_hidden_states_scratch(
                self.model,
                input_list,
                attention_mask_list,
                input_masks,
                sum_all=False,
                start_relax_layer=st_threshold,
                tqdm_disable=True,
                pre_tokenized=([sent_tokens] * num_input_masks, [idxs] * num_input_masks),
                layer_id=layer_id,
            )

            sci_chart = {}
            all_vector_idxs = outer_context_vecs[0]
            for idx, (st, en) in enumerate(span_idxs):
                context_free_vectors = inner_context_vecs[idx][st : en + 1]
                if self.sim_fn == "euclidean_of_span_mean":
                    contextual_vectors = all_vector_idxs[st : en + 1]
                    assert context_free_vectors.shape == contextual_vectors.shape
                    contextual_vectors = contextual_vectors.mean(axis=0)
                    # mask positions have been zeroed out
                    context_free_vectors = context_free_vectors.mean(axis=0)
                    measure_sci = measure_sim_factory(distance.euclidean)
                elif self.sim_fn == "euclidean_mean":
                    if span_formula:
                        if st:
                            contextual_vectors = (
                                all_vector_idxs[en] - all_vector_idxs[st - 1]
                            )
                        else:
                            contextual_vectors = all_vector_idxs[en]
                        context_free_vectors = context_free_vectors[-1]
                    else:
                        contextual_vectors = all_vector_idxs[st : en + 1]
                    assert context_free_vectors.shape == contextual_vectors.shape
                    measure_sci = measure_sim_factory(distance.euclidean)
                elif self.sim_fn == "cosine":
                    contextual_vectors = all_vector_idxs[st : en + 1]
                    assert context_free_vectors.shape == contextual_vectors.shape
                    contextual_vectors = contextual_vectors.sum(axis=0)
                    # mask positions have been zeroed out
                    context_free_vectors = context_free_vectors.sum(axis=0)
                    measure_sci = measure_sim_factory(distance.cosine)
                else:
                    raise ValueError(f"Unexpected sim_fn {self.sim_fn}")
                # only consider the words inside the context
                sci_chart[(st, en)] = measure_sci(context_free_vectors, contextual_vectors)
        else:
            sci_chart = {}
            all_vector_idxs = outer_context_vecs[0]
            for idx, (st, en) in enumerate(span_idxs):
                contextual_vectors = all_vector_idxs[st: en + 1]
                if st != 0:
                    residual_context = all_vector_idxs[:st]
                    # if this distance is small, then the context is not important, and so SCI is high
                    if self.sim_fn == "cosine":
                        measure_sci = measure_sim_factory(distance.cosine)
                        # convert cosine similarity back to cosine distance
                        # large distance is good.
                        dist = 1.0 - measure_sci(contextual_vectors.mean(axis=0), residual_context.mean(axis=0))
                    elif self.sim_fn == "euclidean":
                        measure_sci = measure_sim_factory(distance.euclidean)
                        dist = -1.0 * measure_sci(contextual_vectors.mean(axis=0), residual_context.mean(axis=0))
                    elif self.sim_fn == "cosine_mean":
                        ## take mean over all pairs of vectors
                        measure_sci = measure_sim_factory(distance.cosine)
                        dist_fn = lambda x, y: 1.0 - measure_sci(x, y)
                        dist = np.mean([dist_fn(x, y) for x in contextual_vectors for y in residual_context])
                    elif self.sim_fn == "euclidean_mean":
                        measure_sci = measure_sim_factory(distance.euclidean)
                        dist_fn = lambda x, y: -1.0*measure_sci(x, y)
                        dist = np.mean([dist_fn(x, y) for x in contextual_vectors for y in residual_context])
                    else:
                        raise ValueError(f"Unexpected sim_fn {self.sim_fn}")
                    sci_chart[(st, en)] = dist
                else:
                    # random large number...
                    sci_chart[(st, en)] = 100.0


        return sci_chart


    def __call__(
        self, 
        sci_chart: Dict[Tuple[int, int], float], 
        input_ids: List[int], 
        projection_algorithm: str
    ) -> Tuple[Tuple, float]:
        """
        Compute the tree projection for an input sequence.
        
        Args:
            sci_chart: A dictionary containing SCI scores.
            input_ids: Tokenized input sequence.
            projection_algorithm: Algorithm to use for tree projection.
        
        Returns:
            A tuple containing the parse tree and its associated score.
        """
        parse, score = self._tree_projection(
            sci_chart,
            input_ids,
            projection_algorithm=projection_algorithm,
            normalize=self.normalize,
        )
        return parse, score