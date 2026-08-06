"""Standalone LoRA fine-tuning for TimesFM-200M on a GPU.

Usage:
  python experiments/lora_finetune.py [--epochs 5] [--lr 1e-5] [--rank 8]
  [--batch 2] [--device cuda] [--loss mse|pinball]

Trains LoRA adapters on the q_proj/v_proj attention projections of the TimesFM
transformer encoder using 252d log-return windows → 21d cumulative return label.
Saves adapter weights to artifacts/timesfm/lora_adapter_{loss}/.

Requires: peft, bitsandbytes, accelerate, torch (CUDA).
"""
import os, sys, argparse, warnings, pickle, time, json
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import TimesFm2_5ModelForPrediction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from algo_trading.timesfm_engine import CACHE_ROOT, _cache_key

CTX = 256
HOR = 21
TRAIN_END = pd.Timestamp("2020-12-31")
FEAT_COL = "close"


class ReturnWindowDataset(Dataset):
    """252d log-return windows → 21d cumulative forward return (labels)."""

    def __init__(self, close_df, start, end):
        logret = np.log(close_df).diff().dropna(how="all")
        sessions = logret.index
        self.samples = []
        self.labels = []
        self.symbols = []
        for sym in logret.columns:
            s = logret[sym].dropna().to_numpy()
            if s.size < CTX + HOR + 1:
                continue
            for i in range(0, s.size - CTX - HOR + 1):
                d = sessions[i + CTX]
                if pd.isna(d) or d < start or d > end:
                    continue
                window = s[i:i + CTX].astype(np.float32)
                label = float(s[i + CTX:i + CTX + HOR].sum())  # cum log-ret
                self.samples.append(window)
                self.labels.append(label)
                self.symbols.append(sym)
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels, dtype=np.float32)
        print(f"[lora] Dataset: {len(self.samples)} windows, "
              f"{len(set(self.symbols))} symbols, "
              f"label range [{self.labels.min():.4f}, {self.labels.max():.4f}]",
              flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))


def pinball_loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
    """Multi-quantile pinball loss."""
    loss = 0.0
    for i, q in enumerate(quantiles):
        err = target - pred[..., i]
        loss += torch.mean(torch.max(q * err, (q - 1) * err))
    return loss / len(quantiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loss", choices=["mse", "pinball"], default="mse")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[lora] Device: {device}", flush=True)

    # ---- load close prices ----
    from experiment_utils import setup
    cfg, ctx, _, _, _ = setup("2015-01-01")
    close = ctx.prices.close
    start = pd.Timestamp("2015-01-01")
    max_samples = int(os.environ.get("TFM_LORA_MAX_SAMPLES", "200"))  # tiny for slow GPU

    dataset = ReturnWindowDataset(close, start, TRAIN_END)
    if len(dataset) > max_samples:
        import random
        idx = random.Random(42).sample(range(len(dataset)), max_samples)
        dataset.samples = dataset.samples[idx]
        dataset.labels = dataset.labels[idx]
        dataset.symbols = [dataset.symbols[i] for i in idx]
        print(f"[lora] Subsampled to {max_samples} windows", flush=True)
    if len(dataset) == 0:
        print("[lora] No training samples! Check date range / symbols.", flush=True)
        return
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)

    # ---- load TimesFM model ----
    model_id = "google/timesfm-2.5-200m-transformers"
    print(f"[lora] Loading {model_id} ...", flush=True)
    from transformers import TimesFm2_5ModelForPrediction
    model = TimesFm2_5ModelForPrediction.from_pretrained(model_id, trust_remote_code=True)

    # FREEZE the entire base model first
    for param in model.parameters():
        param.requires_grad_(False)

    # Manually inject LoRA adapters into q_proj and v_proj
    from peft.tuners.lora import Linear as LoRALinear
    from peft.utils import get_peft_model_state_dict
    from peft import LoraConfig
    import copy
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    for name, module in model.named_modules():
        if not any(marker in name for marker in ["q_proj", "v_proj"]):
            continue
        if not isinstance(module, nn.Linear):
            continue
        # Wrap a copy of the module so original stays frozen
        wrapped = LoRALinear(
            base_layer=copy.deepcopy(module),
            adapter_name="default",
            config=lora_config,
            r=args.rank,
            lora_alpha=args.rank * 2,
        )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, wrapped)

    # Move to GPU AFTER injection
    model.to(device)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)

    # ---- training ----
    if args.loss == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = lambda pred, targ: pinball_loss(pred, targ)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0; n_batches = 0
        t0 = time.time()
        log_every = 50
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(past_values=inputs)
                if args.loss == "mse":
                    pred = outputs.mean_predictions[:, :HOR]  # [B, HOR]
                    pred = pred.mean(dim=-1)  # scalar per batch
                    loss = criterion(pred, targets)
                else:
                    pred = outputs.full_predictions[:, :HOR, :]  # [B, HOR, 10]
                    pred = pred.mean(dim=1)  # [B, 10]
                    loss = criterion(pred, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
            if n_batches % log_every == 0:
                avg_so_far = total_loss / n_batches
                elapsed = time.time() - t0
                rate = n_batches / elapsed if elapsed > 0 else 0
                eta = (len(loader) - n_batches) / rate if rate > 0 else 0
                print(f"[lora] E{epoch+1} batch {n_batches}/{len(loader)} "
                      f"loss={avg_so_far:.6f} ({rate:.1f}b/s, ETA {eta:.0f}s)", flush=True)
        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0
        print(f"[lora] Epoch {epoch+1}/{args.epochs} loss={avg_loss:.6f} "
              f"({elapsed:.0f}s, {n_batches} batches)", flush=True)

    # ---- save adapter ----
    loss_label = args.loss
    save_dir = os.path.join("artifacts", "timesfm", f"lora_adapter_{loss_label}")
    os.makedirs(save_dir, exist_ok=True)
    # Save only LoRA params
    lora_state_dict = get_peft_model_state_dict(model, adapter_name="default")
    torch.save(lora_state_dict, os.path.join(save_dir, "adapter_model.safetensors"))
    # Save training config
    cfg_to_save = {
        "r": args.rank,
        "lora_alpha": args.rank * 2,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1,
        "loss": args.loss,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch,
    }
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(cfg_to_save, f)
    print(f"[lora] Adapter saved to {save_dir}", flush=True)
    print("[lora] DONE", flush=True)


if __name__ == "__main__":
    main()
