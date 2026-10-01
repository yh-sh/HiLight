# grpo_trainer.py
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple

import torch
from torch.distributions import Bernoulli


@dataclass
class GRPOConfig:
    device: str = "cpu"
    lr: float = 3e-4

    group_size: int = 4

    # --- entropy anneal ---
    ent_coef_start: float = 0.02
    ent_coef_end: float = 0.01
    ent_coef: float = 0.02
    ent_anneal_steps: int = 2000  # decay over first N updates

    # --- length regularization ---
    len_reg_coef: float = 1.0
    target_frac: float = 0.30

    max_grad_norm: float = 1.0


class GRPOTrainer:
    """
    Expects each ex in batch to contain:
      - "input_ids": (T,)
      - "attention_mask": (T,)
      - optional "policy_mask": (T,) (1=policy-controlled tokens)
      - "build_fn":  (ex, actions_tensor(T,)) -> messages (chat format)
      - "score_fn":  messages -> (reward_float, aux_dict)
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: GRPOConfig,
        build_prompt_fn: Callable[[Dict[str, Any], torch.Tensor], str] = None,
        score_prompt_fn: Callable[[str], Tuple[float, Dict[str, Any]]] = None,
    ):
        self.actor = actor
        self.opt = optimizer
        self.cfg = cfg
        self.build_prompt_fn = build_prompt_fn
        self.score_prompt_fn = score_prompt_fn
        self.global_step = 0

    def _actor_device(self) -> torch.device:
        try:
            return next(self.actor.parameters()).device
        except StopIteration:
            return torch.device(self.cfg.device)

    def _ent_coef_now(self) -> float:
        # linear schedule
        return self.cfg.ent_coef_end

    def update_on_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        device = self._actor_device()
        K = int(self.cfg.group_size)

        # -------- rollout (no grad) --------
        input_ids_list: List[torch.Tensor] = []
        attn_list: List[torch.Tensor] = []
        policy_list: List[torch.Tensor] = []
        actions_list: List[torch.Tensor] = []
        rewards_list: List[torch.Tensor] = []

        for ex in batch:
            input_ids = ex["input_ids"].to(device)       # (T,)
            attn_mask = ex["attention_mask"].to(device)  # (T,)
            build_fn = ex["build_fn"]
            score_fn = ex["score_fn"]

            policy_mask = ex.get("policy_mask", attn_mask).to(device)  # (T,)
            policy_mask_bool = policy_mask.bool()

            with torch.no_grad():
                _logits, probs, _values = self.actor(
                    input_ids.unsqueeze(0),
                    attn_mask.unsqueeze(0),
                )
                probs = probs[0]  # (T,)

                # All rollout tensors are already on `device`.

                dist = Bernoulli(probs=probs)
                actions_K = dist.sample((K,))  # (K,T)

                # only allow actions on policy tokens
                actions_K = actions_K * policy_mask.float().unsqueeze(0)

                # debug on policy tokens
                valid = policy_mask_bool
                probs_valid = probs[valid]
                if probs_valid.numel() > 0:
                    print(
                        f"[DBG] mean_prob={probs_valid.mean().item():.3f} "
                        f"mean_action={actions_K[:, valid].float().mean().item():.3f} "
                        f">0.5={(probs_valid>0.5).float().mean().item():.3f} "
                        f"<0.1={(probs_valid<0.1).float().mean().item():.3f} "
                        f"max={probs_valid.max().item():.3f} min={probs_valid.min().item():.3f}"
                    )

            rewards_K: List[float] = []
            for k in range(K):
                actions_k = actions_K[k].cpu()
                messages = build_fn(ex, actions_k)
                r, _aux = score_fn(messages)
                rewards_K.append(float(r))

            rewards_K_t = torch.tensor(rewards_K, device=device, dtype=torch.float32)

            input_ids_list.append(input_ids)
            attn_list.append(attn_mask)
            policy_list.append(policy_mask)
            actions_list.append(actions_K.to(device))
            rewards_list.append(rewards_K_t)

        B = len(batch)
        rewards_b = torch.cat(rewards_list, dim=0)  # (B*K,)

        # -------- update (with grad) --------
        all_logps: List[torch.Tensor] = []
        entropy_terms: List[torch.Tensor] = []
        len_reg_terms: List[torch.Tensor] = []

        for input_ids, attn_mask, policy_mask, actions_K in zip(
            input_ids_list, attn_list, policy_list, actions_list
        ):
            _logits, probs, _values = self.actor(
                input_ids.unsqueeze(0).to(device),
                attn_mask.unsqueeze(0).to(device),
            )
            probs = probs[0]  # (T,)

            actions_K = actions_K.to(device)
            policy_mask = policy_mask.to(device)

            dist = Bernoulli(probs=probs)
 
            logp_tok = dist.log_prob(actions_K)  # (K,T)
            pol_mask_KT = policy_mask.bool().unsqueeze(0).expand_as(actions_K)
            logp_tok = logp_tok.masked_fill(~pol_mask_KT, 0.0)

            valid_len = policy_mask.float().sum().clamp_min(1.0)
            logp_K = logp_tok.sum(dim=1) / valid_len
            all_logps.append(logp_K)

            ent_tok = dist.entropy().masked_fill(~policy_mask.bool(), 0.0)
            entropy_ex = ent_tok.sum() / valid_len
            entropy_terms.append(entropy_ex)

            probs_valid = probs[policy_mask.bool()]
            if probs_valid.numel() > 0:
                mean_p = probs_valid.mean()
                len_reg_terms.append((mean_p - self.cfg.target_frac) ** 2)
            else:
                len_reg_terms.append(torch.tensor(0.0, device=device))

        logps_b = torch.cat(all_logps, dim=0)  # (B*K,)
        entropy_mean = torch.stack(entropy_terms).mean()
        len_reg = torch.stack(len_reg_terms).mean()

        rewards_b = rewards_b.to(logps_b.device)
        rewards_reshaped = rewards_b.view(B, K)
        baselines = rewards_reshaped.mean(dim=1, keepdim=True)
        adv = rewards_reshaped - baselines
        advantages = adv.view(-1).clamp(-5.0, 5.0)

        pg_loss = -(advantages.detach() * logps_b).mean()

        ent_coef = self._ent_coef_now()
        loss = pg_loss + self.cfg.len_reg_coef * len_reg - ent_coef * entropy_mean

        self.opt.zero_grad(set_to_none=True)
        loss.backward()


        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        self.global_step += 1

        return {
            "loss": float(loss.item()),
            "pol": float(pg_loss.item()),
            "val": 0.0,
            "ent": float(entropy_mean.item()),
            "R": float(rewards_b.mean().item()),
            "ent_coef": float(ent_coef),
        }


# from dataclasses import dataclass
# from typing import Callable, Dict, Any, List, Tuple

# import torch
# import torch.nn.functional as F
# from torch.distributions import Bernoulli


# @dataclass
# class GRPOConfig:
#     device: str = "cpu"
#     lr: float = 3e-4

#     # how many actions per state in a GRPO group
#     group_size: int = 2

#     # entropy bonus (encourage non-degenerate Bernoulli at start)
#     ent_coef: float = 0.01

#     # length regularization: push mean prob toward target_frac
#     len_reg_coef: float = 4.0
#     target_frac: float = 0.1  # e.g. 20% tokens active on average

#     # gradient clipping
#     max_grad_norm: float = 1.0


# class GRPOTrainer:
#     """
#     GRPO trainer for your EmphasisActor.

#     Expects each `ex` in batch to contain:
#       - "input_ids": (T,)
#       - "attention_mask": (T,)
#       - "build_fn":  (ex, actions_tensor) -> prompt_text
#       - "score_fn":  prompt_text -> (reward_float, aux_dict)
#     """

#     def __init__(
#         self,
#         actor: torch.nn.Module,
#         optimizer: torch.optim.Optimizer,
#         cfg: GRPOConfig,
#         build_prompt_fn: Callable[[Dict[str, Any], torch.Tensor], str] = None,
#         score_prompt_fn: Callable[[str], Tuple[float, Dict[str, Any]]] = None,
#     ):
#         self.actor = actor
#         self.opt = optimizer
#         self.cfg = cfg
#         # kept for interface parity, but you already pass build_fn/score_fn via ex
#         self.build_prompt_fn = build_prompt_fn
#         self.score_prompt_fn = score_prompt_fn

#     def update_on_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
#         device = self.cfg.device
#         K = self.cfg.group_size

#         # ------------- Rollout (no grad): sample K masks, get rewards -------------
#         input_ids_list: List[torch.Tensor] = []
#         attn_list: List[torch.Tensor] = []
#         actions_list: List[torch.Tensor] = []   # each: (K, T)
#         rewards_list: List[torch.Tensor] = []   # each: (K,)

#         for ex in batch:
#             input_ids = ex["input_ids"].to(device)       # (T,)
#             attn_mask = ex["attention_mask"].to(device)  # (T,)
#             build_fn = ex["build_fn"]
#             score_fn = ex["score_fn"]

#             with torch.no_grad():
#                 logits, probs, _values = self.actor(
#                     input_ids.unsqueeze(0),   # (1, T)
#                     attn_mask.unsqueeze(0),   # (1, T)
#                 )
#                 logits = logits[0]  # (T,)
#                 probs  = probs[0]   # (T,)

#                 # make sure mask is on same device as probs/actions
#                 attn_mask_dev = attn_mask.to(probs.device)  # (T,)

#                 dist = Bernoulli(probs=probs)

#                 # Sample K masks at once: shape (K, T) on probs.device
#                 actions_K = dist.sample((K,))  # (K, T)
#                 # zero out padding positions using device-aligned mask
#                 actions_K = actions_K * attn_mask_dev.float().unsqueeze(0)

#                 # ---- Debug: check sparsity / saturation ----
#                 valid = attn_mask_dev.bool()          # (T,) ON THE SAME DEVICE AS probs
#                 probs_valid   = probs[valid]          # (T_valid,)
#                 actions_valid = actions_K[:, valid]   # (K, T_valid)

#                 # move to cpu just for printing
#                 if probs_valid.numel() > 0:
#                     probs_valid_cpu   = probs_valid.detach().cpu()
#                     actions_valid_cpu = actions_valid.detach().cpu()

#                     mean_prob      = float(probs_valid_cpu.mean().item())
#                     mean_action    = float(actions_valid_cpu.float().mean().item())
#                     frac_very_high = float((probs_valid_cpu > 0.5).float().mean().item())
#                     frac_very_low  = float((probs_valid_cpu < 0.1).float().mean().item())
#                     max_prob       = float(probs_valid_cpu.max().item())
#                     min_prob       = float(probs_valid_cpu.min().item())
#                 else:
#                     mean_prob = mean_action = frac_very_high = frac_very_low = max_prob = min_prob = 0.0

#                 print(
#                     f"[DBG] mean_prob={mean_prob:.3f} mean_action={mean_action:.3f} "
#                     f">0.5={frac_very_high:.3f} <0.1={frac_very_low:.3f} "
#                     f"max={max_prob:.3f} min={min_prob:.3f}"
#                 )

#             # Query external scorer K times
#             rewards_K: List[float] = []
#             for k in range(K):
#                 actions_k = actions_K[k].cpu()
#                 prompt_text = build_fn(ex, actions_k)
#                 r, _aux = score_fn(prompt_text)
#                 rewards_K.append(float(r))

#             rewards_K_t = torch.tensor(rewards_K, device=device, dtype=torch.float32)  # (K,)

#             input_ids_list.append(input_ids)
#             attn_list.append(attn_mask)
#             actions_list.append(actions_K.to(device))  # (K,T)
#             rewards_list.append(rewards_K_t)

#         # flatten rewards for later logging and shape
#         B = len(batch)
#         rewards_b = torch.cat(rewards_list, dim=0)  # (B*K,)

#         # ------------- Policy update (with grad) -------------
#         all_logps: List[torch.Tensor] = []   # list of (K,) per example
#         len_reg_terms: List[torch.Tensor] = []
#         entropy_terms: List[torch.Tensor] = []

#         for input_ids, attn_mask, actions_K in zip(input_ids_list, attn_list, actions_list):
#             # Forward again, now with grad – actor decides its own device
#             logits, probs, _values = self.actor(
#                 input_ids.unsqueeze(0),    # (1, T)
#                 attn_mask.unsqueeze(0),    # (1, T)
#             )
#             logits = logits[0]            # (T,)
#             probs  = probs[0]             # (T,)

#             dev = probs.device  # actor / backbone device (could be cuda:0, cuda:2, etc.)

#             # Make sure actions and mask live on the SAME device as probs
#             actions_K_dev   = actions_K.to(dev)        # (K, T)
#             attn_mask_dev   = attn_mask.to(dev)        # (T,)
#             attn_bool_dev   = attn_mask_dev.bool()     # (T,)

#             dist = Bernoulli(probs=probs)

#             # log π(a) for each of the K masks
#             logp_tok = dist.log_prob(actions_K_dev)    # (K, T) on dev

#             # Mask out padding positions
#             mask = attn_bool_dev.unsqueeze(0).expand_as(actions_K_dev)   # (K, T)
#             logp_tok = logp_tok.masked_fill(~mask, 0.0)

#             # Normalize by number of valid tokens
#             valid_len = attn_mask_dev.float().sum().clamp_min(1.0)       # scalar on dev
#             logp_K = logp_tok.sum(dim=1) / valid_len                     # (K,)

#             all_logps.append(logp_K)

#             # Entropy per example (averaged over tokens)
#             ent_tok = dist.entropy().masked_fill(~attn_bool_dev, 0.0)    # (T,)
#             entropy_ex = ent_tok.sum() / valid_len                       # scalar
#             entropy_terms.append(entropy_ex)

#             # Length regularizer term: mean prob -> target_frac
#             probs_valid = probs[attn_bool_dev]                           # (T_valid,)
#             if probs_valid.numel() > 0:
#                 mean_p = probs_valid.mean()
#                 len_reg_terms.append((mean_p - self.cfg.target_frac) ** 2)
#             else:
#                 len_reg_terms.append(torch.tensor(0.0, device=dev))

#         # Stack everything on the correct device
#         logps_b = torch.cat(all_logps, dim=0)               # (B*K,)
#         entropy_mean = torch.stack(entropy_terms).mean()    # scalar
#         len_reg = torch.stack(len_reg_terms).mean()         # scalar

#         # ---- GRPO advantages: group-wise baselines ----
#         # make sure rewards_b is on same device as logps_b
#         rewards_b = rewards_b.to(logps_b.device)            # (B*K,)
#         B = len(input_ids_list)
#         K = self.cfg.group_size

#         rewards_reshaped = rewards_b.view(B, K)             # (B, K)
#         baselines = rewards_reshaped.mean(dim=1, keepdim=True)  # (B, 1)
#         adv = rewards_reshaped - baselines                  # (B, K)
#         advantages = adv.view(-1)                           # (B*K,)

#         # (optional) clip advantages to avoid crazy updates
#         advantages = torch.clamp(advantages, -5.0, 5.0)

#         # Policy gradient loss
#         pg_loss = -(advantages.detach() * logps_b).mean()

#         # Total loss = PG + length_reg - entropy_bonus
#         loss = pg_loss + self.cfg.len_reg_coef * len_reg - self.cfg.ent_coef * entropy_mean

#         self.opt.zero_grad(set_to_none=True)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
#         self.opt.step()

#         logs = {
#             "loss": float(loss.item()),
#             "pol": float(pg_loss.item()),
#             "val": 0.0,  # no critic here
#             "ent": float(entropy_mean.item()),
#             "R":   float(rewards_b.mean().item()),
#         }
#         return logs

