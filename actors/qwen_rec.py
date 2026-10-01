import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class QwenEmphasisActor(nn.Module):
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        prob_temp: float = 1.0,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        device_map: str | dict | None = "auto",  # or None for single GPU / manual placement
        torch_dtype: torch.dtype = torch.float32,  # or float16/float32
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.prob_temp = prob_temp
        self.use_lora = use_lora

        if device_map is None:
            # single GPU (or you control placement outside)
            lm = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch_dtype,
            )
        else:
            # model-parallel across multiple GPUs
            lm = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch_dtype,
                device_map=device_map,   # e.g. "auto"
            )

        # Force base config to return hidden states
        if hasattr(lm, "config"):
            lm.config.output_hidden_states = True
        if hasattr(lm, "model") and hasattr(lm.model, "config"):
            lm.model.config.output_hidden_states = True

        if freeze_backbone:
            lm.requires_grad_(False)

        if use_lora:
            lora_cfg = LoraConfig(
                r=4,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            )
            lm = get_peft_model(lm, lora_cfg)

            if hasattr(lm, "base_model") and hasattr(lm.base_model, "model"):
                if hasattr(lm.base_model.model, "config"):
                    lm.base_model.model.config.output_hidden_states = True

        self.lm = lm
        d_model = self.lm.config.hidden_size

        # heads are small; we'll move them to the right device in forward()
        self.ln_f = nn.LayerNorm(d_model)
        self.policy = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        self.pad_token_id = self.lm.config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.lm.config.eos_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        if attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()

        # put inputs on the same device as the *first* lm parameter
        first_param_device = next(self.lm.parameters()).device
        input_ids = input_ids.to(first_param_device)
        attention_mask = attention_mask.to(first_param_device)

        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("hidden_states is None; config.output_hidden_states was not set?")

        x = hidden_states[-1]  # (B, T, d_model)
        # final hidden state lives on one device (e.g. last layer GPU)
        dev = x.device

        # move heads lazily to that device (cost is tiny, done once in practice)
        if self.ln_f.weight.device != dev:
            self.ln_f.to(dev)
            self.policy.to(dev)
            self.value.to(dev)

        x = self.ln_f(x)
        logits = self.policy(x).squeeze(-1)  # (B, T)
        probs = torch.sigmoid(logits / max(1e-6, self.prob_temp))

        mask_f = attention_mask.to(dev).float()
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
        values = self.value(pooled).squeeze(-1)  # (B,)

        return logits, probs, values
