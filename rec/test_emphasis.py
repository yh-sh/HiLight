#!/usr/bin/env python3
# test_emphasis.py
import os, json, argparse, re, math, random, collections
from typing import List, Tuple, Dict, Any
import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from actors.actor import EmphasisActor, EmphasisConfig  # type: ignore
from rec.eval_from_llm_inputs import (
    load_llm_inputs, load_last_label, load_adj_jsonl, load_compact_meta,
    covis_candidates, precompute_popular, backfill_popular_from_list, pack_candidates_table,
    llm_score_candidates_with_model, heuristic_scores, hr_at_k, ndcg_at_k
)
from peft import set_peft_model_state_dict  # only if you saved LoRA

def build_prompt(user_summary, rows, start_tok="<start_important>", end_tok="<end_important>", qwen3=False) -> str:
    lines = []
    lines.append("You are a recommender re-ranker.")
    lines.append("Given a user's history summary and a list of candidate items, re-rank the candidates according to their likelihood of being the next item of interest to the user based on the user's historical interactions.")
    lines.append("Each entry should contain an \"id\" and a \"score\" reflecting the predicted relevance of the candidate to the user. Ensure that the JSON is well-formed and encapsulated within `<FINAL_JSON>` tags. ")
    lines.append(f"Some parts of the history are wrapped in {start_tok} ... {end_tok}. Treat those spans as especially important signals about the user preferences.")
    lines.append("Example User Block:")
    lines.append("[USER_HISTORY_SUMMARY]")
    lines.append("[USER] AE4735AXL3NI7PGY6EGLYHW2BK2A")
    lines.append('[H1] asin="B008B4TIPA", parent_asin="B07N4NY2F8", rating=4.0, verified_purchase=true, helpful_vote=0, timestamp=1397989865000, title="very good", text="it\'s really good and excellent for the soft and hard hear to be used, also it is reliable and long life."')
    lines.append("Example Candidates:")
    lines.append("[CANDIDATES]")
    lines.append('cid=1 | title="Product Title 1" | brand="Brand 1" | cat="Category 1" | phrases=["phrase1", "phrase2", ...]')
    lines.append('cid=2 | title="Product Title 2" | brand="Brand 2" | cat="Category 2" | phrases=["phrase3", "phrase4", ...]')
    lines.append("Example Final JSON Output:\n")
    lines.append("<FINAL_JSON>\n")
    lines.append('[{"id": 1,"score": 8.5}, {"id":2,"score":6.0}, {"id": 10, "score": 0.7}, {"id": 5, "score": 0.6}]\n')
    lines.append("</FINAL_JSON>\n")
    lines.append("Please adhere to the provided structure and ensure the JSON output is strictly formatted as shown.\n")
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
    if qwen3:
        lines.append("Output ONLY <FINAL_JSON>...</FINAL_JSON>. No other text. /no_think.\n")
    lines.append("Final JSON Output:")
    return "\n".join(lines)

# ---------------- Emphasis (deterministic at test) ----------------
def token_offsets(tok_fast, text: str):
    enc = tok_fast(text, return_offsets_mapping=True, add_special_tokens=False)
    return enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]

def topk_spans_from_weights(weights, offsets, budget_ratio=0.06, win=6, max_spans=5):
    import numpy as np
    T = len(weights)
    idx = np.argsort(weights)[::-1].tolist()
    intervals = []
    used = 0
    max_tokens = max(1, int(budget_ratio * T))
    for j in idx:
        s = max(0, j - win); e = min(T-1, j + win)
        if intervals and not (s > intervals[-1][1] + 1):
            continue
        length = e - s + 1
        if len(intervals) >= max_spans or used + length > max_tokens:
            continue
        intervals.append((s, e))
        used += length
    spans = []
    for s, e in intervals:
        cs = offsets[s][0]; ce = offsets[e][1]
        spans.append((cs, ce))
    return spans

def apply_markers(text: str, char_spans, start_tok="<start_important>", end_tok="<end_important>"):
    if not char_spans: return text
    srt = sorted(char_spans, key=lambda x: x[0], reverse=True)
    out = text
    for a, b in srt:
        out = out[:b] + end_tok + out[b:]
        out = out[:a] + start_tok + out[a:]
    return out

# ================================== TEST ==================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-inputs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--adj", required=True)
    ap.add_argument("--meta", required=True)

    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--actor_param", required=True, help="path to actor_emph.pt")
    ap.add_argument("--actor", default="default") 

    ap.add_argument("--candidates", type=int, default=60)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--len-edges", default="4,8,12")
    ap.add_argument("--per-bin", type=int, default=75)
    ap.add_argument("--start-tok", default="<start_important>")
    ap.add_argument("--end-tok", default="<end_important>")

    ap.add_argument("--budget-ratio", type=float, default=0.25)
    ap.add_argument("--win", type=int, default=6)
    ap.add_argument("--max-spans", type=int, default=7)
    # ap.add_argument("--bin", dest="target_bin",
    #                 choices=["1-4", "5-8", "9-12", "13+"],
    #                 required=True,
    #                 help="Which history-length bin to evaluate in this run.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()
    
    # Print arguments nicely
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed); torch.manual_seed(args.seed)

    uid2block = load_llm_inputs(args.llm_inputs)
    labels = load_last_label(args.labels)
    adj = load_adj_jsonl(args.adj)
    iid2meta = load_compact_meta(args.meta)

    
    print("[INFO] Loading popularity...")
    popular_iids = precompute_popular(adj)
    print(f"[INFO] Precomputed popularity list of size {len(popular_iids)}")
    
    rows = [r for r in labels if r["user_id"] in uid2block and r.get("history") and r.get("label")]
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
    # sample ONLY the requested bin
    rng = random.Random(42)
    cohort = []
    skipped_no_label = 0
    
    print("[INFO] Sampling from ALL bins")
    # bin_order = ["5-8", "1-4", "9-12", "13+"]
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

    
    # LLM
    llm_tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, device_map="auto", dtype="auto", low_cpu_mem_usage=True
    )
    qwen3 = "Qwen" in args.model_id
    
    # Actor
    ckpt = torch.load(args.actor_param, map_location=args.device)
    if args.actor == "default":
        vocab_size = ckpt.get("vocab_size", llm_tok.vocab_size)
        cfg = EmphasisConfig(vocab_size=vocab_size, d_model=256, n_heads=4, n_layers=8, ffw_mult=4, dropout=0.1)
        actor = EmphasisActor(cfg).to(args.device)
        actor.load_state_dict(ckpt["state_dict"])
    elif args.actor == "qwen":
        from actors.qwen_rec import QwenEmphasisActor
        base_model_name = ckpt["base_model_name"]
        prob_temp = ckpt["prob_temp"]

        # recreate frozen backbone
        actor = QwenEmphasisActor(
            base_model_name=base_model_name,
            prob_temp=prob_temp,
            freeze_backbone=True,
            use_lora=True,
            device_map="auto",
        ).to(args.device)
        actor.ln_f.load_state_dict(ckpt["ln_f"])
        actor.policy.load_state_dict(ckpt["policy"])
        actor.value.load_state_dict(ckpt["value"])
        actor_tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=True)
        if "peft_model" in ckpt:
            set_peft_model_state_dict(actor.lm, ckpt["peft_model"])
    actor.eval()

    # metrics
    def Stats(): return {"hits":0.0,"dcg":0.0,"n":0}
    metrics = collections.defaultdict(Stats)

    G_steps = 0
    for (bin_name, row, cand_ids) in cohort:
        begin_time = time.time()
        uid = row["user_id"]; history = row["history"]; label = row["label"]
        G_steps += 1
            
        rows_tbl = pack_candidates_table(cand_ids, iid2meta, max_rows=args.candidates)

        user_block = uid2block[uid]
        enc = actor_tok(
            user_block,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        

        ids = enc["input_ids"]           # list[int]
        mask = enc["attention_mask"]     # list[int]
        offsets = enc["offset_mapping"]  # list[(start, end)]
        
        input_ids = torch.tensor([ids], device=args.device)
        attn_mask = torch.tensor([mask], device=args.device)

        with torch.no_grad():
            logits, probs, values = actor(input_ids, attn_mask)
            weights = torch.softmax(logits.masked_fill(attn_mask==0, -1e9), dim=-1)[0].cpu().numpy()
        # print(probs)
        spans = topk_spans_from_weights(weights, offsets, budget_ratio=args.budget_ratio, win=args.win, max_spans=args.max_spans)
        user_block_emph = apply_markers(user_block, spans, start_tok=args.start_tok, end_tok=args.end_tok)
        
        prompt = build_prompt(user_block_emph, rows_tbl, start_tok=args.start_tok, end_tok=args.end_tok, qwen3=qwen3)
        # print("================================== Begin Prompt ============================")
        # print(f"\n[DEBUG] prompt for uid-{uid}:\n{prompt}\n")
        # print("=================================== End Prompt =============================\n")
        pairs = llm_score_candidates_with_model(prompt, llm_tok, llm_model)
        print("Generated scores:", pairs)
        if not pairs:
            pairs = heuristic_scores(rows_tbl, history, iid2meta)
            
        id2iid = {r["id"]: r["iid"] for r in rows_tbl}
        scored = [(id2iid.get(rid), sc) for (rid, sc) in pairs if id2iid.get(rid)]
        scored_iids = {iid for iid,_ in scored}
        for r in rows_tbl:
            if r["iid"] not in scored_iids:
                scored.append((r["iid"], -1e9))
        ranked = sorted(scored, key=lambda kv: kv[1], reverse=True)
        
        try:
            pos = [iid for (iid,_) in ranked].index(label) + 1
            print(f"[DEBUG] uid-{uid} label={label} found at pos={pos} in ranked list.")
        except ValueError:
            pos = None

        m = metrics[bin_name]
        m["n"] += 1
        m["hits"] += hr_at_k(pos, args.topk)
        m["dcg"]  += ndcg_at_k(pos, args.topk)
        n = m["n"]
        time_end = time.time()
        elapsed = time_end - begin_time
        print(f"[test] uid-{uid} bin={bin_name:>4s} n={n:4d}  HR@{args.topk}={m['hits']/n:.4f}  NDCG@{args.topk}={m['dcg']/n:.4f}  Time={elapsed:.2f}s", end="\r")

    # summary
    print("\n=== Test Summary without Skipped Ones===")
    total_hits = 0.0
    total_dcg  = 0.0
    total_eval_n = 0  # only those we actually evaluated (label in cand_ids)

    for b in ["1-4", "5-8", "9-12", "13+"]:
        m = metrics[b]
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
