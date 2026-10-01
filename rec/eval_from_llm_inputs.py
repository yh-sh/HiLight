#!/usr/bin/env python3
import os, json, argparse, re, math, random, collections, time
from typing import List, Tuple, Dict, Any
import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

import json
# Hugging Face auth: set the HF_TOKEN environment variable (see README); never hard-code tokens.
# def parse_pairs(text: str):
#     """
#     Safer parser: looks only at the tail of the output, finds the last [...] block,
#     and tries to json.loads() that. No regex, no catastrophic backtracking.
#     """
#     # Quick sanity: if these aren't present, bail fast
#     if "[" not in text or "]" not in text:
#         return []

#     # Only look at the last N chars to avoid earlier junk
#     tail = text[-2000:]  # adjust if needed

#     # Find last closing ']' in the tail
#     right = tail.rfind("]")
#     if right == -1:
#         return []

#     # Find the nearest '[' before that
#     left = tail.rfind("[", 0, right)
#     if left == -1:
#         return []

#     candidate = tail[left:right+1]

#     try:
#         arr = json.loads(candidate)
#     except Exception:
#         return []

#     if not isinstance(arr, list):
#         return []

#     pairs = []
#     for o in arr:
#         if isinstance(o, dict) and "id" in o and "score" in o:
#             try:
#                 pairs.append((int(o["id"]), float(o["score"])))
#             except Exception:
#                 continue

#     return pairs
def parse_pairs(text: str):
    """
    Parse (id, score) pairs from LLM output.

    Priority:
      1) If <FINAL_JSON>...</FINAL_JSON> is present, parse ONLY the contents.
      2) Otherwise, fall back to scanning the tail for the last JSON array [...].
    Returns:
      List[Tuple[int, float]]
    """
    # ---------- 1) Try <FINAL_JSON> ... </FINAL_JSON> ----------
    start_tag = "<FINAL_JSON>"
    end_tag   = "</FINAL_JSON>"

    start = text.find(start_tag)
    end   = text.find(end_tag, start + len(start_tag)) if start != -1 else -1

    candidate = None

    if start != -1 and end != -1:
        # Extract everything between the tags
        candidate = text[start + len(start_tag):end].strip()
        # Optional: strip code fences if model insists on ```json
        if candidate.startswith("```"):
            # naive strip of fenced block
            candidate = candidate.strip("`")
            # if it still has 'json' prefix, remove it
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()

    # ---------- 2) Fallback: use last [...] in the tail ----------
    if candidate is None or not candidate:
        if "[" not in text or "]" not in text:
            return []

        tail = text[-2000:]  # adjust if needed

        right = tail.rfind("]")
        if right == -1:
            return []

        left = tail.rfind("[", 0, right)
        if left == -1:
            return []

        candidate = tail[left:right+1]

    # ---------- 3) Try to parse JSON ----------
    try:
        arr = json.loads(candidate)
    except Exception:
        return []

    if not isinstance(arr, list):
        return []

    # ---------- 4) Sanitize objects to (int(id), float(score)) ----------
    pairs = []
    for o in arr:
        if not isinstance(o, dict):
            continue
        if "id" not in o or "score" not in o:
            continue
        try:
            cid   = int(o["id"])
            score = float(o["score"])
        except Exception:
            continue
        pairs.append((cid, score))

    return pairs

# ---------------------------- Loaders ----------------------------

def load_llm_inputs(path: str) -> Dict[str, str]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            j = json.loads(line)
            uid = j.get("user_id"); inp = j.get("input")
            if uid and inp: out[uid] = inp
    return out

def load_last_label(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rows.append(json.loads(line))
    return rows

def load_adj_jsonl(path: str) -> Dict[str, List[Tuple[str, float]]]:
    adj = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            j = json.loads(line)
            adj[j["iid"]] = [(b, float(w)) for (b, w) in j["neighbors"]]
    return adj

def load_compact_meta(path: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            j = json.loads(line); out[j["iid"]] = j
    return out

# ----------------------- Candidate generation --------------------

def covis_candidates(history: List[str], adj: Dict[str, List[Tuple[str,float]]],
                     K: int = 200, alpha_recency: float = 0.85) -> List[str]:
    seen = set(history)
    recent = list(reversed(history[-10:]))
    scores = collections.defaultdict(float)
    for r_idx, iid in enumerate(recent):
        for nb, w in adj.get(iid, []):
            if nb in seen: 
                continue
            scores[nb] += (alpha_recency ** r_idx) * w
    return [iid for iid,_ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:K]]


def precompute_popular(adj: Dict[str, List[Tuple[str,float]]]) -> List[str]:
    pop = collections.Counter()
    for _, lst in adj.items():
        for b, w in lst:
            pop[b] += w
    # only compute once
    return [iid for iid, _ in pop.most_common()]

def backfill_popular_from_list(popular_iids: List[str], exclude: set, need: int) -> List[str]:
    out = []
    for iid in popular_iids:
        if iid in exclude:
            continue
        out.append(iid)
        if len(out) >= need:
            break
    return out

def backfill_popular(adj: Dict[str, List[Tuple[str,float]]], exclude: set, need: int) -> List[str]:
    pop = collections.Counter()
    for _, lst in adj.items():
        for b, w in lst: pop[b] += w
    out = []
    for iid, _ in pop.most_common():
        if iid in exclude: continue
        out.append(iid)
        if len(out) >= need: break
    return out

# ------------------------ Packing + prompts ----------------------

def pack_candidates_table(cand_ids: List[str], iid2meta: Dict[str, Dict[str,Any]], max_rows: int = 40) -> List[Dict[str, Any]]:
    rows = []
    for i, iid in enumerate(cand_ids[:max_rows], start=1):
        c = iid2meta.get(iid, {})
        rows.append({
            "id": i,
            "iid": iid,
            "title": (c.get("title","") or "")[:120],
            "brand": c.get("brand","") or "",
            "cat": c.get("cat","") or "",
            "price_band": c.get("price_band","") or "",
            "avg_rating": c.get("avg_rating", None),
            "rating_count": c.get("rating_count", None),
            "phrases": (c.get("phrases") or [])[:3],
        })
    return rows

HINT_PRESETS = {
    "none": "",
    "cat_brand_price": "brand/cat/price-band match; novelty vs. most recent; light popularity via avg rating & count.",
    "category_only": "Prioritize category continuity with last 3 items; break ties by avg_rating then rating_count.",
    "brand_loyalty": "Prefer same-brand or sibling-brand items; if none, use category then price-band.",
    "novelty": "Prefer items that are similar in category but different brand from the last item (novelty), penalize identical titles.",
}

def build_history_summary_prompt(raw_block: str) -> str:
    """
    Build a prompt for the LLM to summarize a user's interaction history.

    raw_block: the long, raw history text (uid2block[uid]).
    Returns: prompt string to send to the LLM.
    """
    lines = []
    lines.append("You are assisting a recommender system.")
    lines.append("Your task is to summarize a user's interaction history.")
    lines.append("")
    lines.append("Write a concise, high-level summary of this user's preferences and behavior.")
    lines.append("")
    lines.append("Output format:")
    lines.append("- Use 1-3 short bullet points.")
    lines.append("- Output bullet points, nothing before or after.")
    lines.append("")
    lines.append("[RAW_USER_HISTORY]")
    lines.append(raw_block.strip())
    return "\n".join(lines)

# def build_llm_prompt_from_block(user_block: str, rows: List[Dict[str,Any]], hint_text: str) -> str:
#     lines = []
#     lines.append("You are a recommender re-ranker.")
#     lines.append("Goal: score each candidate 0–10 for next-item likelihood for this user.")
#     lines.append('Return JSON only, for example: [{"id":1,"score":3}, ...].')
#     lines.append("")
#     lines.append("[USER_HISTORY_SUMMARY]")
#     lines.append(user_block.strip())
#     lines.append("")
#     lines.append("[CANDIDATES]")
#     for cid, r in enumerate(rows, start=1):
#         phr = ", ".join(r["phrases"])
#         star = f'{r["avg_rating"]}({r["rating_count"]})'
#         # DO NOT show iid / ASIN at all
#         lines.append(
#             f'cid={cid} | title="{r["title"]}" | brand="{r["brand"]}" | '
#             f'cat="{r["cat"]}" | price_band="{r["price_band"]}" | '
#             f'rating={star} | {phr}'
#         )
#     lines.append("Rules:")
#     lines.append("1. Output ONLY the JSON array, nothing before or after.")
#     lines.append("2. keep the format strictly valid JSON")
#     return "\n".join(lines)

def build_llm_prompt_from_block(user_summary, rows):
    lines = []
    lines.append("You are a recommender re-ranker.")
    lines.append("Goal: based on the user history, score each candidate 0–10 for next-item likelihood for this user.")
    lines.append("")
    lines.append("Your answer MUST follow this structure exactly:")
    lines.append("<FINAL_JSON>")
    lines.append('[{"id": 1,"score": 8.5}, {"id":2,"score":6.0}, ...]\n')
    lines.append("</FINAL_JSON>")
    lines.append("")
    lines.append("The JSON above is only an example. In your real answer, use the correct ids and scores.")
    lines.append("")
    lines.append("Rules for <FINAL_JSON>:")
    lines.append('- It must be a single valid JSON array of objects with fields id and score.')
    lines.append('- No comments, no extra fields, no text outside the array.')
    lines.append('- Do NOT use backticks or markdown fences around the JSON.')
    lines.append("")
    lines.append("[USER_HISTORY_SUMMARY]")
    lines.append(user_summary.strip())
    lines.append("")

    lines.append("[CANDIDATES]")
    for cid, r in enumerate(rows, start=1):
        phr = ", ".join(r["phrases"])
        star = f'{r["avg_rating"]}({r["rating_count"]})'
        lines.append(
            f'id={cid} | title="{r["title"]}" | brand="{r["brand"]}" | '
            f'cat="{r["cat"]}" | price_band="{r["price_band"]}" | '
            f'rating={star} | {phr}'
        )

    lines.append("Output:")

    return "\n".join(lines)


# -------------------------- Scoring ------------------------------
def llm_summaryize_history(prompt: str, tok, model, max_new_tokens: int = 500) -> str:
    with torch.inference_mode():
        x = tok(prompt, return_tensors="pt").to(model.device)
        y = model.generate(
            **x,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )

    # Only keep the newly generated tokens (strip the prompt part)
    gen_ids = y[0][x["input_ids"].shape[1]:]
    # If you wanted the full sequence, you’d use:
    # gen_ids = y[0]

    out = tok.decode(gen_ids, skip_special_tokens=True)
    # print("LLM summary output:", out)
    return out

def llm_score_candidates_with_model(
    prompt: str,
    tok,
    model,
    max_new_tokens: int = 1000,
):
    with torch.inference_mode():
        x = tok(prompt, return_tensors="pt").to(model.device)
        y = model.generate(
            **x,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )

    # Only keep the newly generated tokens (strip the prompt part)
    gen_ids = y[0][x["input_ids"].shape[1]:]
    # If you wanted the full sequence, you’d use:
    # gen_ids = y[0]

    out = tok.decode(gen_ids, skip_special_tokens=True)
    # print("LLM raw output:", out)
    m = parse_pairs(out)
    if not m:
        print("Warning: failed to parse LLM output, skip this sample")
    return m if m else []

def heuristic_scores(rows: List[Dict[str,Any]], history: List[str], iid2meta: Dict[str,Dict[str,Any]]) -> List[Tuple[int,float]]:
    recent = list(reversed(history[-5:]))
    cats = {iid2meta.get(i,{}).get("cat","") for i in recent if iid2meta.get(i)}
    brands = {iid2meta.get(i,{}).get("brand","") for i in recent if iid2meta.get(i)}
    bands = {iid2meta.get(i,{}).get("price_band","") for i in recent if iid2meta.get(i)}

    out = []
    for r in rows:
        s = 0.0
        if r["cat"] in cats and r["cat"]: s += 2.0
        if r["brand"] in brands and r["brand"]: s += 1.5
        if r["price_band"] in bands and r["price_band"]: s += 1.0
        try:
            if r["avg_rating"] is not None:  s += 0.2 * float(r["avg_rating"])
            if r["rating_count"] is not None: s += 0.5 * math.log1p(float(r["rating_count"]))
        except Exception:
            pass
        out.append((r["id"], s))
    return out

# -------------------------- Metrics -----------------------------

def hr_at_k(rank: int, k: int) -> float:
    return 1.0 if (rank and 1 <= rank <= k) else 0.0

def ndcg_at_k(rank: int, k: int) -> float:
    if rank and 1 <= rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0

# ---------------------- Bucketing utilities ---------------------

def choose_bin(hlen: int, edges: List[int]) -> str:
    # edges like [4,8,12] => bins: [1-4], [5-8], [9-12], [13+]
    if hlen <= 0: return "len0"
    prev = 1
    for e in edges:
        if hlen <= e:
            return f"{prev}-{e}"
        prev = e + 1
    return f"{prev}+"

def sample_by_bins(rows, per_bin: int, edges: List[int]) -> Dict[str, List[Dict[str,Any]]]:
    bins = collections.defaultdict(list)
    for r in rows:
        h = r.get("history") or []
        b = choose_bin(len(h), edges)
        bins[b].append(r)
    rng = random.Random(42)
    out = {}
    for b, lst in bins.items():
        rng.shuffle(lst)
        out[b] = lst[:per_bin]
    return out

# ---------------------------- Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-inputs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--adj", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--candidates", type=int, default=60)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--mode", choices=["llm","heuristic"], default="heuristic")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    # 4 bins [1–4],[5–8],[9–12],[13+]
    ap.add_argument("--len-edges", default="4,8,12",
                    help="Comma-separated bucket edges (inclusive).")
    ap.add_argument("--per-bin", type=int, default=75,
                    help="Max users per length bin (≈300 total if 4 bins).")
    ap.add_argument("--hints", default="cat_brand_price,category_only,brand_loyalty,novelty",
                    help="Comma-separated hint names from presets (or 'none').")
    args = ap.parse_args()

    uid2block = load_llm_inputs(args.llm_inputs)
    labels = load_last_label(args.labels)
    adj = load_adj_jsonl(args.adj)
    iid2meta = load_compact_meta(args.meta)

    print("[INFO] Loading popularity...")
    popular_iids = precompute_popular(adj)
    print(f"[INFO] Precomputed popularity list of size {len(popular_iids)}")
    
    rows = [r for r in labels
            if r["user_id"] in uid2block and r.get("history") and r.get("label")]

    edges = [int(x) for x in args.len_edges.split(",") if x.strip()]
    bins = collections.defaultdict(list)
    for r in rows:
        hlen = len(r["history"])
        b = ("1-4" if hlen <= 4 else
             "5-8" if hlen <= 8 else
             "9-12" if hlen <= 12 else
             "13+")
        bins[b].append(r)

    # sample per bin
    rng = random.Random(42)
    cohort = []
    skipped_no_label = 0
    print("[INFO] Sampling from ALL bins")
    # bin_order = ["9-12", "13+", "1-4", "5-8"]
    bin_order = list(bins.keys())
    for bin_name in bin_order:
        lst = bins.get(bin_name, [])
        if not lst:
            continue

        rng.shuffle(lst)
        taken_in_bin = 0

        print(f"[INFO] Sampling bin={bin_name} (max {args.per_bin})")

        for r in lst:
            uid = r["user_id"]
            history = r["history"]
            label = r["label"]

            cand_ids = covis_candidates(history, adj, K=args.candidates)
            if len(cand_ids) < args.candidates:
                need = args.candidates - len(cand_ids)
                cand_ids += backfill_popular_from_list(
                    popular_iids,
                    exclude=set(history) | set(cand_ids),
                    need=need,
                )

            if label not in cand_ids:
                skipped_no_label += 1
                continue

            cohort.append((bin_name, r, cand_ids))
            taken_in_bin += 1

            if taken_in_bin >= args.per_bin:
                break

    print(f"[INFO] Built cohort of size={len(cohort)}, skipped_no_label={skipped_no_label}")

    hint_names = [h.strip() for h in args.hints.split(",") if h.strip()]
    for h in hint_names:
        if h not in HINT_PRESETS:
            print(f"[WARN] Unknown hint '{h}', falling back to 'none'")
    hint_names = [h if h in HINT_PRESETS else "none" for h in hint_names]
    hint_names = list(dict.fromkeys(hint_names))  # de-dup, keep order

    tok = model = None
    if args.mode == "llm":
        tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    # metrics[hints][bin] -> stats
    Stats = lambda: {"hits": 0.0, "dcg": 0.0, "n": 0}
    metrics = {h: collections.defaultdict(Stats) for h in hint_names}
    G_steps = 0

    for hint in hint_names:
        for (bin_name, row, cand_ids) in cohort:
            uid = row["user_id"]
            history = row["history"]
            label = row["label"]
            G_steps += 1

            rows_tbl = pack_candidates_table(cand_ids, iid2meta, max_rows=args.candidates)

            if args.mode == "llm":
                prompt = build_llm_prompt_from_block(uid2block[uid], rows_tbl)

                pairs = llm_score_candidates_with_model(prompt, tok, model)
                print("Generated scores:", pairs)
                if not pairs:
                    pairs = heuristic_scores(rows_tbl, history, iid2meta)
            else:
                pairs = heuristic_scores(rows_tbl, history, iid2meta)

            id2iid = {r["id"]: r["iid"] for r in rows_tbl}
            scored = [(id2iid.get(rid), sc) for (rid, sc) in pairs if id2iid.get(rid)]
            scored_iids = {iid for iid, _ in scored}
            for r in rows_tbl:
                if r["iid"] not in scored_iids:
                    scored.append((r["iid"], -1e9))

            ranked = sorted(scored, key=lambda kv: kv[1], reverse=True)

            try:
                pos = [iid for (iid, _) in ranked].index(label) + 1
            except ValueError:
                pos = None

            m = metrics[hint][bin_name]
            m["n"] += 1
            m["hits"] += hr_at_k(pos, args.topk)
            m["dcg"] += ndcg_at_k(pos, args.topk)
            n = m["n"]
            print(
                f"[{hint}] uid-{uid} bin={bin_name:>6s}  n={n:4d}  "
                f"HR@{args.topk}={m['hits']/n:.4f}  NDCG@{args.topk}={m['dcg']/n:.4f} ",
                end="\r",
            )
    # summary
    print("\n=== Test Summary without Skipped Ones===")
    total_hits = 0.0
    total_dcg  = 0.0
    total_eval_n = 0  # only those we actually evaluated (label in cand_ids)

    for b in ["1-4", "5-8", "9-12", "13+"]:
        m = metrics["none"][b]
        n = m["n"]
        if n == 0:
            continue

        hr_b = m["hits"] / n
        nd_b = m["dcg"]  / n

        total_hits   += m["hits"]
        total_dcg    += m["dcg"]
        total_eval_n += n

        print(f"{b:>4s}  n={n:4d}  HR@{args.topk}={hr_b:.4f}  NDCG@{args.topk}={nd_b:.4f}")
    # now include skipped ones as failures
    overall_n_with_skips = total_eval_n + skipped_no_label
    if overall_n_with_skips:
        overall_hr_sk   = total_hits / overall_n_with_skips          # skipped → 0 hit
        overall_ndcg_sk = total_dcg  / overall_n_with_skips          # skipped → 0 dcg
        print(
            f"OVERALL (including {skipped_no_label} skipped_no_label) "
            f"n={overall_n_with_skips}  "
            f"HR@{args.topk}={overall_hr_sk:.4f}  NDCG@{args.topk}={overall_ndcg_sk:.4f}"
        )
if __name__ == "__main__":
    main()
