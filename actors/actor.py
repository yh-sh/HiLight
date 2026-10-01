# actor.py
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class EmphasisConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    ffw_mult: int = 4
    dropout: float = 0.1
    max_len: int = 4096
    # probability temperature for Bernoulli head (lower = sharper)
    prob_temp: float = 1.0

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffw_mult: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ffw_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffw_mult, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + self.dropout(h))
        h = self.ff(x)
        x = self.ln2(x + self.dropout(h))
        return x

class EmphasisActor(nn.Module):
    """
    Produces per-token emphasis Bernoulli logits and a scalar value for PPO.
    You pass token *embeddings* from a frozen tokenizer/embedding (or use internal embed layer).
    """
    def __init__(self, cfg: EmphasisConfig, tie_input_embed: Optional[nn.Embedding] = None):
        super().__init__()
        if tie_input_embed is not None:
            self.embed = tie_input_embed  # (vocab_size, d_model)
            d_model = tie_input_embed.embedding_dim
            if d_model != cfg.d_model:
                raise ValueError(f"Embed dim {d_model} != cfg.d_model {cfg.d_model}")
        else:
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffw_mult, cfg.dropout)
        for _ in range(cfg.n_layers)])

        self.ln_f = nn.LayerNorm(cfg.d_model)

        # policy head: per-token Bernoulli logit
        self.policy = nn.Linear(cfg.d_model, 1)
        # value head: pooled -> scalar
        self.value = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.Tanh(),
            nn.Linear(cfg.d_model, 1),
        )

        self.prob_temp = cfg.prob_temp

    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) bool or 0/1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits: (B, T) Bernoulli logits (before sigmoid)
          probs:  (B, T) probabilities (after sigmoid / temp)
          values:(B,)    state value
        """
        B, T = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.embed(input_ids) + self.pos(pos)

        # torch MHA expects key_padding_mask True for padding positions
        key_padding_mask = ~attention_mask  # True where pad
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        x = self.ln_f(x)

        logits = self.policy(x).squeeze(-1)               # (B, T)
        probs = torch.sigmoid(logits / max(1e-6, self.prob_temp))

        # masked mean pooling for value
        mask_f = attention_mask.float()
        pooled = (x * mask_f.unsqueeze(-1)).sum(dim=1) / (mask_f.sum(dim=1).clamp_min(1.0)).unsqueeze(-1)
        values = self.value(pooled).squeeze(-1)           # (B,)

        return logits, probs, values
