# ppo_trainer.py
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple
import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

@dataclass
class PPOConfig:
    device: str = "cpu"
    lr: float = 3e-4
    gamma: float = 1.0
    lam: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    batch_size: int = 4
    updates_per_batch: int = 2

class PPOTrainer:
    """
    Generic PPO wrapper that:
      1) Gets token logits/probs/values from the actor.
      2) Samples emphasis mask (Bernoulli per token).
      3) Builds a tagged prompt via a callback.
      4) Calls an external scorer to get reward (e.g., HR/NDCG on LLM ranking).
    Your callbacks:
      - build_prompt(batch, masks) -> List[str]
      - score_prompt(prompt) -> reward float, aux dict
    """
    def __init__(self, actor, optimizer, cfg: PPOConfig,
                 build_prompt_fn: Callable[[Dict[str,Any], torch.Tensor], List[str]],
                 score_prompt_fn: Callable[[str], Tuple[float, Dict[str,Any]]]):
        self.actor = actor
        self.opt = optimizer
        self.cfg = cfg
        self.build_prompt_fn = build_prompt_fn
        self.score_prompt_fn = score_prompt_fn

    def _logprob_and_entropy(self, logits: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor):
        """
        logits: (B,T)
        actions: (B,T) in {0,1}
        mask: (B,T) bool
        Returns mean logprob, entropy over selected positions (masked mean).
        """
        dist = Bernoulli(logits=logits)
        logp = dist.log_prob(actions.float())  # (B,T)
        ent = dist.entropy()                   # (B,T)
        maskf = mask.float()
        denom = maskf.sum().clamp_min(1.0)
        return (logp * maskf).sum() / denom, (ent * maskf).sum() / denom

    def update_on_batch(self, batch: List[Dict[str,Any]]) -> Dict[str, float]:
        device = self.cfg.device

        # -------- Rollout collection (no grad): sample actions, cache old stats --------
        input_ids_list, attn_list, actions_list, masks_list = [], [], [], []
        old_logps_list, old_values_list, rewards_list = [], [], []
        build_fns, score_fns = [], []

        with torch.no_grad():
            for ex in batch:
                input_ids = ex["input_ids"].to(device)           # (T,)
                attn_mask = ex["attention_mask"].to(device)      # (T,)
                logits, probs, values = self.actor(input_ids.unsqueeze(0), attn_mask.unsqueeze(0))
                logits = logits[0]                                # (T,)
                probs  = probs[0]
                value  = values[0]                                # ()

                # Bernoulli sampling per token
                dist = Bernoulli(probs=probs)
                
                # Sample K masks at once: shape (K, T)
                actions_K = dist.sample((4,))  # (K, T)
                # zero out padding positions
                actions_K = actions_K * attn_mask.float().unsqueeze(0)

                # ---- Debug: check sparsity / saturation ----
                valid = attn_mask.bool()          # (T,)
                probs_valid   = probs[valid]      # (T_valid,)
                actions_valid = actions_K[:, valid]  # (K, T_valid)

                mean_prob      = float(probs_valid.mean().item()) if probs_valid.numel() > 0 else 0.0
                mean_action    = float(actions_valid.float().mean().item()) if actions_valid.numel() > 0 else 0.0
                frac_very_high = float((probs_valid > 0.5).float().mean().item()) if probs_valid.numel() > 0 else 0.0
                frac_very_low  = float((probs_valid < 0.1).float().mean().item()) if probs_valid.numel() > 0 else 0.0

                print(
                    f"[DBG] mean_prob={mean_prob:.3f} mean_action={mean_action:.3f} "
                    f">0.5={frac_very_high:.3f} <0.1={frac_very_low:.3f}"
                )
                
                
                actions = dist.sample()                           # (T,)
                actions = actions * attn_mask.float()
                # per-seq mean logp (masked)
                tok_logp = dist.log_prob(actions).masked_fill(~attn_mask.bool(), 0.0)
                denom = attn_mask.float().sum().clamp_min(1.0)
                old_logp = tok_logp.sum() / denom                # scalar

                # Build prompt and score (reward) with this sampled emphasis
                prompt_text = ex["build_fn"](ex, actions.cpu())
                
                r, _aux = ex["score_fn"](prompt_text)
                r = float(r)

                # Accumulate detached rollout pieces
                input_ids_list.append(input_ids)
                attn_list.append(attn_mask)
                actions_list.append(actions)
                masks_list.append(attn_mask)                      # same mask
                old_logps_list.append(torch.tensor(r, device=device))  # placeholder, fix below
                old_values_list.append(value.detach())
                rewards_list.append(torch.tensor(r, device=device, dtype=torch.float32))
                build_fns.append(ex["build_fn"])
                score_fns.append(ex["score_fn"])

                # We stored reward into old_logps_list by mistake; fix after loop.
                old_logps_list[-1] = old_logp.detach()

        # Pad/stack rollout tensors
        logits_b_inputs = input_ids_list                        # keep raw inputs to re-forward
        masks_b         = torch.nn.utils.rnn.pad_sequence(masks_list,   batch_first=True)
        actions_b       = torch.nn.utils.rnn.pad_sequence(actions_list, batch_first=True)
        attn_b          = torch.nn.utils.rnn.pad_sequence(attn_list,    batch_first=True)
        old_logps       = torch.stack(old_logps_list)                          # (B,)
        old_values      = torch.stack(old_values_list)                         # (B,)
        rewards_b       = torch.stack(rewards_list)                            # (B,)

        # One-step advantage/return
        advantages = (rewards_b - old_values).detach()                         # (B,)
        returns    = rewards_b.detach()                                        # (B,)

        logs = {}
        for _ in range(self.cfg.updates_per_batch):
            # -------- Re-forward current policy on the same inputs (fresh graph) --------
            new_logits_list, new_values_list = [], []
            for input_ids, attn_mask in zip(input_ids_list, attn_list):
                logits, probs, values = self.actor(input_ids.unsqueeze(0), attn_mask.unsqueeze(0))
                new_logits_list.append(logits[0])
                new_values_list.append(values[0])

            logits_b = torch.nn.utils.rnn.pad_sequence(new_logits_list, batch_first=True)  # (B,T)
            values_new = torch.stack(new_values_list)                                      # (B,)

            # New log-prob / entropy as masked per-seq mean
            dist = Bernoulli(logits=logits_b)
            logp_tok = dist.log_prob(actions_b).masked_fill(~attn_b.bool(), 0.0)           # (B,T)
            denom = attn_b.float().sum(dim=1).clamp_min(1.0)                                # (B,)
            logp_new = (logp_tok.sum(dim=1) / denom)                                        # (B,)
            ent_tok  = dist.entropy().masked_fill(~attn_b.bool(), 0.0)
            entropy  = (ent_tok.sum(dim=1) / denom).mean()                                  # scalar

            # PPO objective
            ratio = torch.exp(logp_new - old_logps)                                         # (B,)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            value_loss = F.mse_loss(values_new, returns)
            loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.opt.step()

            logs = {
                "loss": float(loss.item()),
                "pol": float(policy_loss.item()),
                "val": float(value_loss.item()),
                "ent": float(entropy.item()),
                "R":   float(rewards_b.mean().item()),
            }

        return logs
