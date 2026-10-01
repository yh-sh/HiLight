import torch
import torch.nn as nn
from transformers import AutoModel


class EmphasisActorFromBert(nn.Module):
    """
    Encoder-based emphasis actor (e.g., roberta-base) with chunking support.

    - Works with arbitrarily long sequences by processing in chunks of at most
      `window_size <= config.max_position_embeddings`.
    - Interface:
        forward(input_ids, attention_mask=None) -> (logits, probs, values)
          logits: (B, T_total)
          probs:  (B, T_total)
          values: (B,)
    """
    def __init__(
        self,
        model_name: str = "roberta-base",
        prob_temp: float = 1.0,
        freeze_encoder: bool = False,
        max_window: int | None = None,  # optional manual cap per chunk
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config
        d_model = self.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # determine window size: do not exceed model's max_position_embeddings
        model_max = getattr(self.config, "max_position_embeddings", 512)
        if max_window is None:
            self.window_size = model_max
        else:
            self.window_size = min(max_window, model_max)

        self.ln_f = nn.LayerNorm(d_model)

        # per-token Bernoulli logit head
        self.policy = nn.Linear(d_model, 1)

        # pooled -> scalar value head
        self.value = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        self.prob_temp = prob_temp

        # pad token id (RoBERTa usually has pad_token_id=1)
        self.pad_token_id = getattr(self.config, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = 1  # reasonable default for roberta-like models

    # --- helper: single-chunk forward ---
    def _forward_window(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Single-window encoder pass:
          input_ids:      (B, Tw)
          attention_mask: (B, Tw)
        Returns:
          x: (B, Tw, d_model) after encoder + layernorm
        """
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        x = enc_out.last_hidden_state  # (B, Tw, d_model)
        x = self.ln_f(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, T_total)
        attention_mask: torch.Tensor | None = None,   # (B, T_total) or None
    ):
        """
        Returns:
          logits: (B, T_total)
          probs:  (B, T_total)
          values: (B,)
        """
        B, T_total = input_ids.shape

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        if attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()

        W = self.window_size

        # === Fast path: fits in one encoder call ===
        if T_total <= W:
            x = self._forward_window(input_ids, attention_mask)   # (B, T_total, d)
            logits = self.policy(x).squeeze(-1)                   # (B, T_total)
            probs = torch.sigmoid(logits / max(1e-6, self.prob_temp))

            mask_f = attention_mask.float()
            denom = mask_f.sum(dim=1).clamp_min(1.0)              # (B,)
            pooled = (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)  # (B, d)
            values = self.value(pooled).squeeze(-1)               # (B,)

            return logits, probs, values

        # === Long path: chunk into multiple windows ===
        logits_chunks = []
        mask_chunks = []

        global_sum = None        # (B, d_model)
        global_mask_sum = None   # (B,)

        for start in range(0, T_total, W):
            end = min(start + W, T_total)
            ids_chunk = input_ids[:, start:end]          # (B, Tw)
            mask_chunk = attention_mask[:, start:end]    # (B, Tw)
            mask_f = mask_chunk.float()

            x_chunk = self._forward_window(ids_chunk, mask_chunk)   # (B, Tw, d)

            # per-token logits for this chunk
            logits_chunk = self.policy(x_chunk).squeeze(-1)         # (B, Tw)
            logits_chunks.append(logits_chunk)
            mask_chunks.append(mask_chunk)

            # accumulate for global pooled value
            chunk_sum = (x_chunk * mask_f.unsqueeze(-1)).sum(dim=1)  # (B, d)
            chunk_mask_sum = mask_f.sum(dim=1)                       # (B,)

            if global_sum is None:
                global_sum = chunk_sum
                global_mask_sum = chunk_mask_sum
            else:
                global_sum = global_sum + chunk_sum
                global_mask_sum = global_mask_sum + chunk_mask_sum

        # concat all chunk logits / masks back to full length
        logits = torch.cat(logits_chunks, dim=1)        # (B, T_total)
        full_mask = torch.cat(mask_chunks, dim=1)       # (B, T_total)  # noqa: F841 (if unused)

        probs = torch.sigmoid(logits / max(1e-6, self.prob_temp))

        # global pooled representation for value
        denom = global_mask_sum.clamp_min(1.0).unsqueeze(-1)  # (B, 1)
        pooled = global_sum / denom                           # (B, d)
        values = self.value(pooled).squeeze(-1)               # (B,)

        return logits, probs, values
