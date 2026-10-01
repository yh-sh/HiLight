from typing import List, Tuple
import torch
import numpy as np

START_TAG = "<start_important>"
END_TAG   = "<end_important>"

def select_spans_from_probs(
    probs: torch.Tensor,             # (T,)  probabilities in [0,1]
    attention_mask: torch.Tensor,    # (T,)  bool
    strategy: str = "top_p",         # "threshold" | "top_p" | "top_k"
    threshold: float = 0.7,
    top_p: float = 0.3,
    top_k: int = 128,
) -> List[Tuple[int, int]]:
    """
    Convert per-token probs into contiguous spans [l, r] (inclusive indices).
    """
    assert probs.ndim == 1
    assert attention_mask.ndim == 1
    valid = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return []

    idx = valid
    p = probs[idx]

    if strategy == "threshold":
        chosen = (p >= threshold).nonzero(as_tuple=False).squeeze(-1)
        chosen_idx = idx[chosen]
    elif strategy == "top_k":
        k = min(int(top_k), idx.numel())
        topv, topi = torch.topk(p, k)
        chosen_idx = idx[topi.sort().values]
    else:
        # top_p by probability mass
        sorted_p, sorted_i = torch.sort(p, descending=True)
        cum = torch.cumsum(sorted_p, dim=0)
        cut = (cum <= top_p).nonzero(as_tuple=False).squeeze(-1)
        if cut.numel() == 0:
            cut_k = 1
        else:
            cut_k = int(cut[-1].item()) + 1
        chosen_idx = idx[sorted_i[:cut_k].sort().values]

    # merge into spans
    spans = []
    if chosen_idx.numel() == 0:
        return spans
    start = int(chosen_idx[0].item())
    prev = start
    for t in chosen_idx[1:].tolist():
        if t == prev + 1:
            prev = t
        else:
            spans.append((start, prev))
            start = t
            prev = t
    spans.append((start, prev))
    return spans

def inject_tags_into_text(tokens: List[str], spans: List[Tuple[int,int]]) -> str:
    """
    Insert tags around token spans.
    Assumes tokens are already detokenized in a simple joinable form.
    """
    # convert token list to a char-level stream by join with space; we’ll tag by token index
    # Simpler: insert tags around token indices directly.
    tagged = []
    span_map = {}
    for (l, r) in spans:
        span_map.setdefault(l, []).append("START")
        span_map.setdefault(r, []).append("END")

    for i, tk in enumerate(tokens):
        if i in span_map and "START" in span_map[i]:
            tagged.append(START_TAG)
        tagged.append(tk)
        if i in span_map and "END" in span_map[i]:
            tagged.append(END_TAG)

    return " ".join(tagged)

def apply_markers(text: str, char_spans,
                  start_tok: str = "<start_important>",
                  end_tok: str = "<end_important>"):
    if not char_spans:
        return text
    # insert from right to left so indices remain valid
    srt = sorted(char_spans, key=lambda x: x[0], reverse=True)
    out = text
    for a, b in srt:
        out = out[:b] + end_tok + out[b:]
        out = out[:a] + start_tok + out[a:]
    return out

def topk_spans_from_weights(
    weights,              # 1D np.array, length T
    offsets,              # list of (char_start, char_end)
    budget_ratio=0.15,
    win=6,
    max_spans=5,
):
    """
    Select a few high-weight windows, then map them to char spans.
    """
    T = len(weights)
    idx = np.argsort(weights)[::-1].tolist()  # descending
    intervals = []
    used = 0
    max_tokens = max(1, int(budget_ratio * T))

    for j in idx:
        s = max(0, j - win)
        e = min(T - 1, j + win)
        # avoid overlapping windows
        if intervals and not (s > intervals[-1][1] + 1):
            continue
        length = e - s + 1
        if len(intervals) >= max_spans or used + length > max_tokens:
            continue
        intervals.append((s, e))
        used += length

    spans = []
    for s, e in intervals:
        cs = offsets[s][0]
        ce = offsets[e][1]
        spans.append((cs, ce))
    return spans

def spans_from_binary_mask(
    actions_1d: torch.Tensor,        # (T,) 0/1 from Bernoulli
    attention_mask_1d: torch.Tensor, # (T,) 0/1 or bool
) -> List[Tuple[int, int]]:
    """
    Convert a binary mask over tokens into contiguous spans [l, r] (inclusive).
    Only positions with attention_mask_1d == 1 are considered.
    """
    assert actions_1d.ndim == 1
    assert attention_mask_1d.ndim == 1
    assert actions_1d.shape == attention_mask_1d.shape

    actions_1d = actions_1d.to(torch.float32)
    attention_mask_1d = attention_mask_1d.to(torch.bool)

    chosen = ((actions_1d > 0.5) & attention_mask_1d).nonzero(as_tuple=False).squeeze(-1)
    if chosen.numel() == 0:
        return []

    spans: List[Tuple[int, int]] = []
    start = int(chosen[0].item())
    prev  = start
    for t in chosen[1:].tolist():
        if t == prev + 1:
            prev = t
        else:
            spans.append((start, prev))
            start = t
            prev  = t
    spans.append((start, prev))
    return spans

def token_offsets(tok_fast, text: str):
    """
    Returns:
      - input_ids: List[int]
      - attention_mask: List[int]
      - offsets: List[(char_start, char_end)]
    """
    enc = tok_fast(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    return enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]