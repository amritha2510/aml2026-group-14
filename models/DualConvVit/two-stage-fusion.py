"""
Two-stage dual-branch pipeline + late-fusion ensemble, with MULTI-SEED reporting.

Stage A — solo-train the ViT and CNN branches SEPARATELY, each with its own
          hyperparameters (the optima the baselines used).
Stage B — freeze both trained branches and train ONLY the fusion head on top
          (concat and cross-attention) — the "proposed model", done right.
Ensemble — average the two solo branches' probabilities (decision-level).

Every model is evaluated on the TEST split BOTH with and without TTA, across
N seeds, and reported as mean +/- std. This makes the comparison meaningful
(run-to-run variance was larger than the gaps between models) and shows
directly whether TTA helps or hurts each class (especially normal).

NOTE: the solo branches here are re-implementations with the baseline
hyperparameters but a different pipeline (15% internal val fold, cosine
schedule, TTA on both branches), so they will NOT exactly reproduce the
teammates' reported ViT/CNN numbers. To match those, load the teammates'
trained checkpoints into the branches instead of re-training.

Run:  python models/DualConvVit/two_stage_fusion.py
"""

import os
import sys
import copy
import json
import random
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as tvm
import timm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score

from constants import LABEL_TO_ID
from data.image_transforms import get_image_aug_config
from data.data_reader import get_required_config_path, load_config, load_metadata
from DualBranchConvViT import PreprocessedChestXrayDataset, DualBranchConvViT, tta_predict

CONFIG_PATH = Path(os.environ.get("DUAL_BRANCH_CONFIG_PATH", "config.yaml"))
CLASSES = [0, 1, 2]  # normal, bacterial, viral

# Per-branch hyperparameters = the solo optima from the baselines.
VIT_HP    = {"lr": 5e-5, "weight_decay": 0.01, "epochs": 20, "freeze_epochs": 5, "dropout": 0.2, "label_smoothing": 0.1}
CNN_HP    = {"lr": 1e-3, "weight_decay": 1e-4, "epochs": 20, "freeze_epochs": 3, "dropout": 0.2, "label_smoothing": 0.0}
FUSION_HP = {"lr": 1e-4, "weight_decay": 1e-3, "epochs": 15, "dropout": 0.2, "label_smoothing": 0.1}
FUSION_TYPES = ["concat", "attention"]

SEEDS = [42, 7, 123]   # add more for tighter error bars (each seed ~= 70 training epochs)
OUTPUT_DIR = Path("./outputs/dual_conv_vit/two_stage")


# ───────────────────────── solo branch wrappers ─────────────────────────

class ViTSolo(nn.Module):
    def __init__(self, vit_branch, vit_dim, num_classes=3, dropout=0.2):
        super().__init__()
        self.vit_branch = vit_branch
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(vit_dim, num_classes)

    def forward(self, x):
        cls = self.vit_branch.forward_features(x)[:, 0]
        return self.head(self.drop(cls))


class CNNSolo(nn.Module):
    def __init__(self, cnn_features, cnn_dim, num_classes=3, dropout=0.2):
        super().__init__()
        self.cnn_features = cnn_features
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(cnn_dim, num_classes)

    def forward(self, x):
        feat = F.relu(self.cnn_features(x))
        return self.head(self.drop(feat.mean(dim=(2, 3))))


# ───────────────────────── data / seeding ─────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(config, dl_cfg, device):
    preprocessed_path = get_required_config_path(config, CONFIG_PATH, "preprocessed_metadata_output_path")
    df = load_metadata(preprocessed_path)
    df = df[df["label"].isin(LABEL_TO_ID.keys()) & df["filepath"].notna()].copy()
    df["filepath"] = df["filepath"].astype(str)
    df = df[df["filepath"].map(lambda p: Path(p).exists())].copy()

    train_df_full = df[df["split"] == "train"].copy()
    test_df       = df[df["split"] == "test"].copy()

    random_state = dl_cfg.get("random_state", 42)        # fold is FIXED across seeds
    batch_size   = dl_cfg.get("batch_size", 32)
    num_workers  = dl_cfg.get("num_workers", 4)
    internal_val_frac = dl_cfg.get("internal_val_frac", 0.15)

    train_df, val_df = train_test_split(
        train_df_full, test_size=internal_val_frac,
        stratify=train_df_full["label"], random_state=random_state,
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    aug_config = get_image_aug_config(config, model_key="dual_conv_vit")

    def loader(frame, training=False):
        return DataLoader(
            PreprocessedChestXrayDataset(
                frame, transform=transform,
                aug_cfg=aug_config if training else None,
                is_training=training, random_state=random_state,
            ),
            batch_size=batch_size, shuffle=training, num_workers=num_workers, pin_memory=True,
        )

    loaders = {"train": loader(train_df, training=True),
               "val": loader(val_df), "test": loader(test_df)}

    y_train = train_df["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    _w = compute_class_weight("balanced", classes=np.arange(len(LABEL_TO_ID)), y=y_train)
    class_weight = torch.tensor(_w, dtype=torch.float32).to(device)
    return loaders, class_weight, (len(train_df), len(val_df), len(test_df))


# ───────────────────────── train / predict / metrics ─────────────────────────

@torch.no_grad()
def predict_probs(net, loader, device, use_amp, use_tta):
    net.eval()
    all_p, all_y = [], []
    for images, labels in loader:
        images = images.to(device)
        if use_tta:
            p = tta_predict(net, images, use_amp)
        else:
            with torch.amp.autocast("cuda", enabled=use_amp):
                p = torch.softmax(net(images).float(), dim=1)
        all_p.append(p.cpu().numpy())
        all_y.append(labels.numpy())
    return np.concatenate(all_p), np.concatenate(all_y)


def macro_recall(labels, preds):
    return recall_score(labels, preds, average="macro", labels=CLASSES, zero_division=0)


def fit(net, loaders, criterion, optimizer, scheduler, device, use_amp, epochs,
        unfreeze_epoch=None, backbone=None, tag=""):
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_recall, best_state = -1.0, copy.deepcopy(net.state_dict())
    for epoch in range(epochs):
        if unfreeze_epoch is not None and backbone is not None and epoch == unfreeze_epoch:
            for p in backbone.parameters():
                p.requires_grad = True
        net.train()
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(net(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        probs, ys = predict_probs(net, loaders["val"], device, use_amp, use_tta=False)
        rec = macro_recall(ys, probs.argmax(1))
        if rec > best_recall:
            best_recall, best_state = rec, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    print(f"  [{tag}] best internal-val recall={best_recall:.4f}")
    return best_recall


def metrics_from_probs(probs, labels):
    preds = probs.argmax(1)
    per = recall_score(labels, preds, average=None, labels=CLASSES, zero_division=0)
    return {
        "macro_recall": float(macro_recall(labels, preds)),
        "macro_f1":     float(f1_score(labels, preds, average="macro", labels=CLASSES, zero_division=0)),
        "normal":       float(per[0]),
        "bacterial":    float(per[1]),
        "viral":        float(per[2]),
    }


# ───────────────────────── one full run (one seed) ─────────────────────────

def run_once(config, dl_cfg, device, use_amp, seed):
    set_seed(seed)
    loaders, class_weight, sizes = build_loaders(config, dl_cfg, device)
    print(f"  splits -> train(fit)={sizes[0]} internal-val={sizes[1]} test={sizes[2]}")

    vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
    cnn = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT).features
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        vit_dim = vit.forward_features(dummy).shape[-1]
        cnn_dim = cnn(dummy).shape[1]

    # Stage A — solo branches
    vit_net = ViTSolo(vit, vit_dim, dropout=VIT_HP["dropout"]).to(device)
    for p in vit_net.vit_branch.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(vit_net.parameters(), lr=VIT_HP["lr"], weight_decay=VIT_HP["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=VIT_HP["epochs"])
    crit = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=VIT_HP["label_smoothing"])
    fit(vit_net, loaders, crit, opt, sch, device, use_amp, VIT_HP["epochs"],
        unfreeze_epoch=VIT_HP["freeze_epochs"], backbone=vit_net.vit_branch, tag="ViT-solo")

    cnn_net = CNNSolo(cnn, cnn_dim, dropout=CNN_HP["dropout"]).to(device)
    for p in cnn_net.cnn_features.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(cnn_net.parameters(), lr=CNN_HP["lr"], weight_decay=CNN_HP["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CNN_HP["epochs"])
    crit = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=CNN_HP["label_smoothing"])
    fit(cnn_net, loaders, crit, opt, sch, device, use_amp, CNN_HP["epochs"],
        unfreeze_epoch=CNN_HP["freeze_epochs"], backbone=cnn_net.cnn_features, tag="CNN-solo")

    def both(net):
        raw, y = predict_probs(net, loaders["test"], device, use_amp, use_tta=False)
        tta, _ = predict_probs(net, loaders["test"], device, use_amp, use_tta=True)
        return raw, tta, y

    vit_raw, vit_tta, y = both(vit_net)
    cnn_raw, cnn_tta, _ = both(cnn_net)

    out = {
        "ViT (solo)": {"raw": metrics_from_probs(vit_raw, y), "tta": metrics_from_probs(vit_tta, y)},
        "CNN (solo)": {"raw": metrics_from_probs(cnn_raw, y), "tta": metrics_from_probs(cnn_tta, y)},
        "Ensemble (equal)": {
            "raw": metrics_from_probs(0.5 * vit_raw + 0.5 * cnn_raw, y),
            "tta": metrics_from_probs(0.5 * vit_tta + 0.5 * cnn_tta, y),
        },
    }

    # Stage B — fusion on frozen branches
    for ft in FUSION_TYPES:
        dual = DualBranchConvViT(fusion_type=ft, noise_dropout_rates=FUSION_HP["dropout"]).to(device)
        dual.vit_branch.load_state_dict(vit.state_dict())
        dual.cnn_features.load_state_dict(cnn.state_dict())
        for n, p in dual.named_parameters():
            p.requires_grad = not (n.startswith("vit_branch") or n.startswith("cnn_features"))
        fp = [p for p in dual.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(fp, lr=FUSION_HP["lr"], weight_decay=FUSION_HP["weight_decay"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FUSION_HP["epochs"])
        crit = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=FUSION_HP["label_smoothing"])
        fit(dual, loaders, crit, opt, sch, device, use_amp, FUSION_HP["epochs"], tag=f"Fusion-{ft}")
        f_raw, f_tta, _ = both(dual)
        out[f"Fusion-{ft}"] = {"raw": metrics_from_probs(f_raw, y), "tta": metrics_from_probs(f_tta, y)}

    return out


# ───────────────────────── aggregate + report ─────────────────────────

METRIC_KEYS = ["macro_recall", "macro_f1", "normal", "bacterial", "viral"]


def aggregate(per_seed):
    names = list(per_seed[0].keys())
    agg = {}
    for name in names:
        agg[name] = {}
        for mode in ("tta", "raw"):
            agg[name][mode] = {}
            for k in METRIC_KEYS:
                vals = [s[name][mode][k] for s in per_seed]
                agg[name][mode][k] = (float(np.mean(vals)), float(np.std(vals)))
    return agg, names


def print_table(title, mode, agg, names):
    nw = max(len(n) for n in names) + 2
    cols = [("MacroRec", "macro_recall"), ("MacroF1", "macro_f1"),
            ("Normal", "normal"), ("Bacterial", "bacterial"), ("Viral", "viral")]
    cw = 15
    header = f"{'Model':<{nw}}" + "".join(f"{c[0]:>{cw}}" for c in cols)
    line = "═" * len(header)
    print("\n" + line)
    print(f"{title:^{len(header)}}")
    print(line)
    print(header)
    print("─" * len(header))
    for n in names:
        row = f"{n:<{nw}}"
        for _, key in cols:
            m, s = agg[n][mode][key]
            row += f"{m:.3f}±{s:.3f}".rjust(cw)
        print(row)
    print(line)


def main():
    config = load_config(CONFIG_PATH)
    dl_cfg = config.get("dual_conv_vit", {})
    if not dl_cfg:
        raise Exception("No dual_conv_vit configuration")

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    use_amp = device.type == "cuda"
    print(f"Using device: {device} | seeds: {SEEDS}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in SEEDS:
        print(f"\n{'#'*30} SEED {seed} {'#'*30}")
        per_seed.append(run_once(config, dl_cfg, device, use_amp, seed))

    agg, names = aggregate(per_seed)

    print_table(f"TEST RESULTS — WITH TTA  (mean±std over {len(SEEDS)} seeds)", "tta", agg, names)
    print_table(f"TEST RESULTS — WITHOUT TTA (raw)  (mean±std over {len(SEEDS)} seeds)", "raw", agg, names)

    print("\n[TTA effect]  positive = TTA helps, negative = TTA hurts  (mean over seeds)")
    print(f"  {'Model':<20}{'ΔNormal':>12}{'ΔMacroRec':>12}")
    for n in names:
        d_norm  = agg[n]["tta"]["normal"][0]       - agg[n]["raw"]["normal"][0]
        d_macro = agg[n]["tta"]["macro_recall"][0] - agg[n]["raw"]["macro_recall"][0]
        print(f"  {n:<20}{d_norm:>+12.4f}{d_macro:>+12.4f}")

    best = max(names, key=lambda n: agg[n]["tta"]["macro_recall"][0])
    bm, bs = agg[best]["tta"]["macro_recall"]
    print(f"\n[INFO] Best by mean TTA macro recall: {best} ({bm:.4f} ± {bs:.4f})")
    print(f"[INFO] Baselines to beat — CNN: 0.8146 | ViT: 0.8489")

    with open(OUTPUT_DIR / "multiseed_results.json", "w") as f:
        json.dump({"seeds": SEEDS, "per_seed": per_seed, "aggregated": agg}, f, indent=2)
    print(f"[INFO] Saved → {OUTPUT_DIR / 'multiseed_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())