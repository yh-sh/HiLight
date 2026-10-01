# train_driver.py
from collections import defaultdict
import os, math, random, argparse
import torch
import torch.optim as optim

from actors.actor import EmphasisActor, EmphasisConfig
from utils.prompt_enhancer import topk_spans_from_weights, apply_markers, token_offsets
from algorithms.ppo_trainer import PPOTrainer, PPOConfig
from algorithms.grpo_trainer import GRPOConfig, GRPOTrainer
from peft import get_peft_model_state_dict
from rec.eval_from_llm_inputs import (
    load_llm_inputs, load_last_label, load_adj_jsonl, load_compact_meta,
    covis_candidates, pack_candidates_table, llm_score_candidates_with_model
)
# Hugging Face auth: set the HF_TOKEN environment variable (see README); never hard-code tokens.

def save_actor_ckpt(path, actor, tok):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "state_dict": actor.state_dict(),
        "vocab_size": getattr(tok, "vocab_size", None) or tok.get_vocab_size(),
    }
    torch.save(ckpt, path)
    
    

def save_heads_ckpt(path, actor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "base_model_name": actor.base_model_name,
        "prob_temp": actor.prob_temp,
        "ln_f": actor.ln_f.state_dict(),
        "policy": actor.policy.state_dict(),
        "value": actor.value.state_dict(),
    }
    if hasattr(actor.lm, "peft_config"):
        ckpt["peft_model"] = get_peft_model_state_dict(actor.lm)
    torch.save(ckpt, path)
    
def get_bin(hlen):
    return ("1-4"  if hlen <= 4  else
            "5-8"  if hlen <= 8  else
            "9-12" if hlen <= 12 else
            "13+")
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-inputs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--adj", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--task", default="amazon-beauty")
    # Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct 
    # google/gemma-3-27b-it
    # meta-llama/Meta-Llama-3-70B-Instruct
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--candidates", type=int, default=40)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--train-steps", type=int, default=10000)
    ap.add_argument("--algorithm", choices=["PPO", "GRPO"], default="PPO")
    ap.add_argument("--actor", default="default") # or lm
    ap.add_argument("--ckpt", type=int, default=0)
    #  meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct
    # google/gemma-3-1b-it google/gemma-3-4b-it
    # not causal mistralai/Ministral-3-3B-Instruct-2512
    
    # Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct
    ap.add_argument("--actor-model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--gpu", type=int, default=0)
    
    ap.add_argument("--budget-ratio", type=float, default=0.25)
    ap.add_argument("--win", type=int, default=6)
    ap.add_argument("--max-spans", type=int, default=5)
    args = ap.parse_args()
    
    # Print arguments nicely
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, 
        device_map="auto",
        dtype="auto", low_cpu_mem_usage=True
    )
        # Build actor
    if args.actor == "default":
        vocab_size = tok.vocab_size
        cfg = EmphasisConfig(vocab_size=vocab_size, d_model=256, n_heads=4, n_layers=8, ffw_mult=4, dropout=0.1)
        actor = EmphasisActor(cfg).to(device)
        opt = optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=0.01)
    elif args.actor == "lm":
        from actors.qwen_rec import QwenEmphasisActor
        actor = QwenEmphasisActor(
            base_model_name=args.actor_model_id,
            prob_temp=1.0,
            freeze_backbone=True,
            use_lora=True,
            device_map="auto",            # 🔑 spread across multiple GPUs
            torch_dtype=torch.float32,   # or float16 if your GPUs support it
        )
        if args.ckpt > 0:
            ckpt_path = os.path.join(
                args.save_dir, args.task, args.model_id,
                args.algorithm, args.actor_model_id,
                f"actor_emph_step{args.ckpt}.pt"
            )
            print(f"Loading actor emphasis checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            actor.ln_f.load_state_dict(ckpt["ln_f"])
            actor.policy.load_state_dict(ckpt["policy"])
            actor.value.load_state_dict(ckpt["value"])
            if "peft_model" in ckpt:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(actor.lm, ckpt["peft_model"])
            print("Loaded checkpoint into actor.")
            
        opt = optim.AdamW(
            filter(lambda p: p.requires_grad, actor.parameters()),
            lr=1e-4,
            weight_decay=0.01,
        )
        actor_tokenizer = AutoTokenizer.from_pretrained(args.actor_model_id, trust_remote_code=True, use_fast=True)
        
    uid2block = load_llm_inputs(args.llm_inputs)
    labels = load_last_label(args.labels)
    adj = load_adj_jsonl(args.adj)
    iid2meta = load_compact_meta(args.meta)

    rows = [r for r in labels if r["user_id"] in uid2block and r.get("history") and r.get("label")]
    rng = random.Random(42)
    rng.shuffle(rows)
    
    bins = defaultdict(list)
    for r in rows:
        hlen = len(r["history"])
        b = get_bin(hlen)
        bins[b].append(r)
    
    
    split = args.train_steps - args.ckpt
    bin_names = ["1-4", "5-8", "9-12", "13+"]
    print(f"Training bins {bin_names} in {split} steps")
    n_bins = len(bin_names)
    base = split // n_bins
    extra = split % n_bins  # distribute remainder over first few bins

    train_rows = []
    for i, b in enumerate(bin_names):
        target = base + (1 if i < extra else 0)   # e.g. 2500 or 2501 if split=10000
        candidates = bins[b]

        if not candidates:
            continue

        if len(candidates) >= target:
            # sample without replacement
            train_rows.extend(rng.sample(candidates, target))
        else:
            # not enough data in this bin: take all, then oversample with replacement
            train_rows.extend(candidates)
            train_rows.extend(rng.choices(candidates, k=target - len(candidates)))

    # 4) Shuffle final training set so bins are mixed
    rng.shuffle(train_rows)

    print("number of training data", len(train_rows)) # should be == split (args.train_steps)

    base_dtype = next(llm.parameters()).dtype

    
    
    if args.algorithm == "PPO":
        # Trainer scaffolding
        ppo_cfg = PPOConfig(device=str(device), batch_size=args.batch_size, updates_per_batch=2)
        trainer = PPOTrainer(
            actor=actor,
            optimizer=opt,
            cfg=ppo_cfg,
            build_prompt_fn=None,     # we’ll pass per-example builder via batch entries
            score_prompt_fn=None,     # wrapped per-example
        )
    else:
        # GRPO Trainer scaffolding
        grpo_cfg = GRPOConfig(
            device=str(device),
            lr=1e-4,
            group_size=4,        # try 4 or 8
            ent_coef=0.01,
            len_reg_coef=1.0,    # adjust if too dense / too sparse
            target_frac=0.2,    # aim for ~20% tokens tagged
            max_grad_norm=1.0,
        )
        trainer = GRPOTrainer(
            actor=actor,
            optimizer=opt,
            cfg=grpo_cfg,
            build_prompt_fn=None,   # still unused; we rely on ex["build_fn"]
            score_prompt_fn=None,   # still unused; we rely on ex["score_fn"]
        )

    def build_example(uid: str, history: list, label: str):
        # 1) candidates
        cand_ids = covis_candidates(history, adj, K=args.candidates)
        if label not in cand_ids:
            cand_ids = [label] + [x for x in cand_ids if x != label]
            
        rows_tbl = pack_candidates_table(cand_ids, iid2meta, max_rows=args.candidates)
        # 2) user block
        user_block = uid2block[uid].strip()

        # 3) tokenize the *user block only* for emphasis (not candidates)
        base_tok = tok if args.actor == "default" else actor_tokenizer
        ids, mask, offsets = token_offsets(base_tok, user_block)  # lists

        input_ids = torch.tensor(ids, device=device)        # (T,)
        attention_mask = torch.tensor(mask, device=device)  # (T,)
        # 4) id mapping to compute reward later
        id2iid = {r["id"]: r["iid"] for r in rows_tbl}
        label_id = next((rid for rid, iid in id2iid.items() if iid == label), None)

        # 5) per-example wrapped scorer
        def example_score(prompt_text: str):
            pairs = llm_score_candidates_with_model(prompt_text, tok, llm)
            if not pairs:
                # hard penalty for non-JSON
                return 0.0, {"empty": True}

            ranked = sorted(pairs, key=lambda kv: kv[1], reverse=True)
            try:
                pos = [rid for (rid, _) in ranked].index(label_id) + 1
            except ValueError:
                pos = None

            hr = 1.0 if (pos and pos <= args.topk) else 0.0
            ndcg = (1.0 / math.log2(pos + 1)) if (pos and pos <= args.topk) else 0.0

            # keep coefficients modest so reward scale stays ~[-1, 1]
            reward = 0.7 * hr + 0.3 * ndcg
            return reward, {}
        
        def example_build(ex, actions_tensor):
            probs = actions_tensor.to(torch.float32)
            weights  = probs.cpu().numpy()
            char_spans = topk_spans_from_weights(
                weights, offsets,
                budget_ratio=args.budget_ratio,
                win=args.win,
                max_spans=args.max_spans,
            )
            tagged_hist = apply_markers(user_block, char_spans)

            lines = []
            lines.append("You are a recommender re-ranker.")
            lines.append("Goal: score each candidate 0–10 for next-item likelihood for this user.")
            lines.append("Given a user's history summary and a list of candidate items, re-rank the candidates according to their likelihood of being the next item of interest to the user based on the user's historical interactions.")
            lines.append("Each entry should contain an \"id\" and a \"score\" reflecting the predicted relevance of the candidate to the user. Ensure that the JSON is well-formed and encapsulated within `<FINAL_JSON>` tags. ")
            lines.append("Example User Block:\n")
            lines.append("<FINAL_JSON>\n")
            lines.append("[USER_HISTORY_SUMMARY]\n")
            lines.append("[USER] AE4735AXL3NI7PGY6EGLYHW2BK2A")
            lines.append("[H1] asin=\"B008B4TIPA\", parent_asin=\"B07N4NY2F8\", rating=4.0, verified_purchase=true, helpful_vote=0, timestamp=1397989865000, title=\"very good\", text=\"it's really good and excellent for the soft and hard hear to be used, also it is reliable and long life.\"")
            lines.append("</FINAL_JSON>\n")
            lines.append("Example Candidates:\n")
            lines.append("[CANDIDATES]\n")
            lines.append('id=1 | title="Product Title 1" | brand="Brand 1" | cat="Category 1" | phrases=["phrase1", "phrase2", ...]\n')
            lines.append('id=2 | title="Product Title 2" | brand="Brand 2" | cat="Category 2" | phrases=["phrase3", "phrase4", ...]\n')
            lines.append("Example Final JSON Output:\n")
            lines.append("<FINAL_JSON>\n")
            lines.append('[{"id": 1,"score": 8.5}, {"id":2,"score":6.0}, {"id": 10, "score": 0.7}, {"id": 5, "score": 0.6}, {"id": 34, "score": 0.5}]\n')
            lines.append("</FINAL_JSON>\n")
            lines.append("Please adhere to the provided structure and ensure the JSON output is strictly formatted as shown.\n")
            lines.append("[USER_HISTORY_SUMMARY]\n")
            lines.append(tagged_hist)
            lines.append("")
            lines.append("[CANDIDATES]")
            for cid, r in enumerate(rows_tbl, start=1):
                phr = ", ".join(r["phrases"])
                star = f'{r["avg_rating"]}({r["rating_count"]})'
                # DO NOT show iid / ASIN at all
                lines.append(
                    f'id={cid} | title="{r["title"]}" | brand="{r["brand"]}" | '
                    f'cat="{r["cat"]}" | price_band="{r["price_band"]}" | '
                    f'rating={star} | {phr}'
                )
            lines.append("Final JSON Output:")
            return "\n".join(lines)


        return {
            "user_id": uid,
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "rows_tbl": rows_tbl,
            "build_fn": example_build,
            "score_fn": example_score,
        }

    # === Train ===
    # make sure save_dir exists
    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0  # counts PPO "rounds" (update_on_batch calls)
    if args.ckpt > 0:
        global_step = args.ckpt
        print(f"Resuming from global step {global_step}")
    # === Train ===
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        batch = []
        for i, row in enumerate(train_rows, 1):
            ex = build_example(row["user_id"], row["history"], row["label"])
            # bind the trainer’s callbacks per-example
            trainer.build_prompt_fn = lambda ex_dict, act, ex=ex: ex["build_fn"](ex_dict, act)
            trainer.score_prompt_fn = ex["score_fn"]

            batch.append(ex)
            if len(batch) >= args.batch_size:
                logs = trainer.update_on_batch(batch)
                global_step += 1

                print(
                    f"[epoch {epoch} |  | step {i} | round {global_step}] "
                    f"R={logs['R']:.4f} loss={logs['loss']:.3f} "
                    f"pol={logs['pol']:.3f} val={logs['val']:.3f} ent={logs['ent']:.3f}"
                )

                # save every 1k rounds
                if global_step % 1000 == 0:
                    ckpt_path = os.path.join(
                        args.save_dir, args.task, args.model_id,
                        args.algorithm, args.actor_model_id,
                        f"actor_emph_step{global_step}.pt"
                    )
                    save_heads_ckpt(ckpt_path, trainer.actor)
                    print(f"[checkpoint] Saved actor at round {global_step} to {ckpt_path}")

                batch = []

    # === (Optional) Evaluate on test_rows using your existing eval script ===
    print("Training complete. Use your existing evaluation script on test split.")

    final_path = os.path.join(args.save_dir, args.task, args.model_id,
                              args.algorithm, args.actor_model_id, 
                              f"actor_emph_{args.train_steps}.pt")
    if args.actor == "qwen": 
        save_heads_ckpt(final_path, trainer.actor)
    else:
        save_actor_ckpt(final_path, trainer.actor, tok)
    print(f"Saved final actor to {final_path}")
    print("Training complete. Use test_emphasis.py with --actor ckpts_emph/best_actor.pt (or the final).")

    
if __name__ == "__main__":
    main()