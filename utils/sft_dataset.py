# sft_dataset.py
import json
import glob
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

class JsonlSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_glob: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ):
        """
        jsonl_glob: e.g. 'filtered_data/filtered_shard*.jsonl'
        Each line must have some variant of "Input"/"input" and "Output"/"output".
        max_length: should match model.config.max_position_embeddings (or smaller).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict[str, str]] = []

        files = sorted(glob.glob(jsonl_glob))
        if not files:
            raise ValueError(f"No JSONL files matched glob: {jsonl_glob}")

        total_lines = 0
        kept_lines = 0

        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_lines += 1
                    obj = json.loads(line)

                    # case-insensitive mapping for input/output keys
                    key_map = {k.lower(): k for k in obj.keys()}
                    if "input" in key_map and "output" in key_map:
                        inp_key = key_map["input"]
                        out_key = key_map["output"]
                        self.samples.append(
                            {"input": obj[inp_key], "output": obj[out_key]}
                        )
                        kept_lines += 1

        print(
            f"[SFT] Loaded {len(self.samples)} samples from {len(files)} files. "
            f"(total_lines={total_lines}, kept_lines={kept_lines})"
        )

        if not self.samples:
            raise ValueError(
                f"[SFT] No valid samples loaded from {files}. "
                f"Check that your JSONL has some form of 'Input'/'input' and 'Output'/'output' keys."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        inp = ex["input"]   # full prompt (system + history + candidates + tags)
        out = ex["output"]  # full <think> ... <FINAL_JSON> ... </FINAL_JSON>

        # 1) tokenize both fully (no artificial truncation)
        inp_ids = self.tokenizer(
            inp,
            add_special_tokens=False,
        )["input_ids"]

        out_ids = self.tokenizer(
            out,
            add_special_tokens=False,
        )["input_ids"]

        # 2) concatenate
        input_ids = inp_ids + out_ids

        # 3) cap only by model context window
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        # 4) labels: ignore loss on the prompt part
        #    but if truncation cut off part of the prompt, clamp n_inp
        n_inp = min(len(inp_ids), len(input_ids))
        labels = [-100] * n_inp + input_ids[n_inp:]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }



def sft_collate_fn(batch, tokenizer: PreTrainedTokenizerBase, max_length: int = 2048):
    """
    Simple collate: pad to max seq length in batch.
    """
    # find max length in this batch (cap at max_length)
    max_len = min(
        max(len(ex["input_ids"]) for ex in batch),
        max_length,
    )

    input_ids_padded = []
    labels_padded = []
    attention_masks = []

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # Qwen often uses eos as pad
        pad_id = tokenizer.eos_token_id

    for ex in batch:
        ids = ex["input_ids"][:max_len]
        labs = ex["labels"][:max_len]

        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = ids + [pad_id] * pad_len
            labs = labs + [-100] * pad_len

        input_ids_padded.append(ids)
        labels_padded.append(labs)
        attention_masks.append([1] * max_len)

    import torch
    return {
        "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
        "labels": torch.tensor(labels_padded, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }
