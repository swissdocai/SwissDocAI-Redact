"""
Fine-tune docTR's word recognizer (PARSeq) for the Swiss documents the app reads.

The Swiss recognizer is docTR's pretrained PARSeq with the Swiss vocabulary
(swissdocai/vocab.py; the output head re-initialized). Round 1 learnt on our
renders alone: it read umlauts and ß (XFUND-de words 88.6% → 95.4%) but real
French and English scans worse than docTR's (−4 points, mostly t read as f
and letters in the wrong case). Round 2 (--checkpoint
output/doctr_models/parseq_swiss.pt) mixes word crops from real pages,
labelled with their reference text (export_doctr_data.py --real-reco), into
every batch; a checkpoint counts only while our words and those with umlauts
or ß hold, and is chosen by real crops and real scans (FUNSD training forms,
evaluation only).

The weights drop into docTR's predictor (swissdocai DoctrWordReader:
--ocr-reco <file>.pt), and scripts/export_onnx.py turns them into the app's
ONNX model.

Training checkpoints every --save-every steps, at every evaluation and on
Ctrl+C into --run-dir; run the same command again to resume where it
stopped. The best checkpoint is <run-dir>/best.pt with its .json.

Data: scripts/export_doctr_data.py. Scan-like augmentation (scan_augment.py:
blur, rotation, noise, JPEG, contrast) is applied on the fly.

Usage:
    uv run --no-sync python scripts/train_doctr.py --data output/doctr_data \
        --checkpoint output/doctr_models/parseq_swiss.pt --run-dir output/doctr_runs/reco_v2
    uv run --no-sync python scripts/train_doctr.py --eval-only              # docTR's parseq
    uv run --no-sync python scripts/train_doctr.py --eval-only --checkpoint output/doctr_runs/reco_v2/best.pt
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from scan_augment import scan_like  # noqa: E402  (scripts/, next to this file)
from swissdocai.vocab import SWISS_VOCAB  # noqa: E402

logger = logging.getLogger(__name__)


def _collate(batch):
    """Lists per field (crops, texts); module level so spawned loader workers can pickle it."""
    return tuple(list(field) for field in zip(*batch))


_VOCAB = set(SWISS_VOCAB)
_UMLAUTS = set("äöüÄÖÜß")


class CropDataset(torch.utils.data.Dataset):
    """Word crops and their texts (export_doctr_data.py: labels.json + images/); ``augment``:
    the strength of the scan-like augmentation, 0 for none."""

    def __init__(self, folder: Path, augment: float = 0.0, max_items: int = 0, seed: int = 0):
        labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        self.items = sorted(labels.items())
        if max_items and len(self.items) > max_items:
            self.items = random.Random(seed).sample(self.items, max_items)
        self.folder = folder / "images"
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, text = self.items[idx]
        image = Image.open(self.folder / name).convert("RGB")
        if self.augment:
            image = scan_like(image, random.Random(), strength=self.augment)
        return np.asarray(image), text


class FormCropDataset(torch.utils.data.Dataset):
    """Word crops from real scanned forms — a FUNSD split directory — cut from the true boxes
    with a margin like the detector's. Evaluation only (non-commercial license)."""

    def __init__(self, folder: Path, max_items: int = 0, seed: int = 0, margin: float = 0.1):
        self.items = []
        for ann in sorted((folder / "annotations").glob("*.json")):
            for entity in json.loads(ann.read_text(encoding="utf-8"))["form"]:
                for word in entity["words"]:
                    text = word["text"].strip()
                    if text and len(text) <= 30 and set(text) <= _VOCAB:
                        self.items.append((str(folder / "images" / f"{ann.stem}.png"), tuple(word["box"]), text))
        if max_items and len(self.items) > max_items:
            self.items = sorted(random.Random(seed).sample(self.items, max_items))
        self.margin = margin
        self._page = (None, None)  # the last page opened: words come page by page

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, (x0, y0, x1, y1), text = self.items[idx]
        if self._page[0] != path:
            self._page = (path, Image.open(path).convert("RGB"))
        image = self._page[1]
        pad = self.margin * (y1 - y0)
        crop = image.crop((max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad)))
        return np.asarray(crop), text


def _reco_batches(model, crops: list[np.ndarray]):
    """docTR's recognition preprocessing: aspect-preserving resize + normalize."""
    from doctr.models.preprocessor import PreProcessor

    pre = PreProcessor(model.cfg["input_shape"][-2:], batch_size=len(crops), preserve_aspect_ratio=True,
                       mean=model.cfg["mean"], std=model.cfg["std"])
    return pre(crops)[0]


def evaluate_reco(model, loader, device) -> dict:
    """Exact and caseless word accuracy, CER, and the accuracy on words with an umlaut or ß."""
    was_training = model.training
    model.eval()
    exact = caseless = total = umlaut = umlaut_total = 0
    distance = 0.0
    with torch.no_grad():
        for crops, texts in loader:
            preds = model(_reco_batches(model, list(crops)).to(device), return_preds=True)["preds"]
            for (pred, _conf), text in zip(preds, texts):
                total += 1
                exact += pred == text
                caseless += pred.lower() == text.lower()
                distance += Levenshtein.normalized_distance(pred, text)
                if set(text) & _UMLAUTS:
                    umlaut_total += 1
                    umlaut += pred == text
    if was_training:
        model.train()
    scores = {"exact": exact / total, "caseless": caseless / total, "cer": distance / total}
    if umlaut_total:
        scores["umlaut"] = umlaut / umlaut_total
    return {**{k: round(float(v), 4) for k, v in scores.items()}, "n": total}


def train_reco(args, device):
    from doctr.models import parseq

    data = Path(args.data) / "reco"
    if args.eval_only and not args.checkpoint:
        model = parseq(pretrained=True)  # docTR's own, with its own vocabulary
    else:
        model = parseq(pretrained=True, vocab=SWISS_VOCAB)  # the output head re-initialized for ours
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)

    def loader(dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=256, collate_fn=_collate, num_workers=2)

    ours_val = loader(CropDataset(data / "val", max_items=args.val_items, seed=1))
    umlauts = CropDataset(data / "val")  # every validation word with an umlaut or ß, up to 2000
    umlauts.items = [item for item in umlauts.items if set(item[1]) & _UMLAUTS][:2000]
    umlaut_val = loader(umlauts)
    real_dir = data / "real"
    has_real = {split: (real_dir / split / "labels.json").exists() for split in ("train", "val")}
    real_val = loader(CropDataset(real_dir / "val", max_items=args.val_items, seed=1)) if has_real["val"] else None
    scans = loader(FormCropDataset(Path(args.guard), max_items=args.val_items, seed=1))

    def evaluate() -> dict:
        """Our validation crops, those with umlauts or ß, real crops, real scans."""
        scores = {"ours": evaluate_reco(model, ours_val, device), "umlauts": evaluate_reco(model, umlaut_val, device),
                  "real_scans": evaluate_reco(model, scans, device)}
        if real_val is not None:
            scores["real"] = evaluate_reco(model, real_val, device)
        return scores

    if args.eval_only:
        logger.info("%s: %s", args.checkpoint or "docTR parseq", evaluate())
        return

    ours = CropDataset(data / "train", augment=0.8)
    n_real = round(args.batch_size * args.real_share) if has_real["train"] else 0
    real = CropDataset(real_dir / "train", augment=0.4) if n_real else None

    def batches(step: int):
        mine = _batches(ours, args.batch_size - n_real, args.workers, args.seed, step)
        if not n_real:
            yield from mine
            return
        theirs = _batches(real, n_real, max(1, args.workers // 2), args.seed + 1, step)
        for (crops, texts), (real_crops, real_texts) in zip(mine, theirs):
            yield crops + real_crops, texts + real_texts

    def train_step(batch):
        crops, texts = batch
        with _autocast(device):
            return model(_reco_batches(model, crops).to(device), target=texts)["loss"], ""

    def better(scores: dict, best: dict | None, before: dict) -> bool:
        """Our words and those with umlauts or ß must hold; real crops and real scans decide."""
        holds = all(scores[name]["exact"] >= before[name]["exact"] - args.guard_tolerance for name in ("ours", "umlauts"))

        def gain(s: dict) -> float:
            return s.get("real", {}).get("exact", 0.0) + s["real_scans"]["exact"]

        return holds and gain(scores) > gain(best or before)

    _run(args, model, evaluate, batches, train_step, better,
         {"arch": "parseq", "task": "reco", "vocab": SWISS_VOCAB}, device)


# --- the training loop -----------------------------------------------------------------


def _batches(dataset, batch_size: int, workers: int, seed: int, step: int):
    """Endless batches in a fixed order per epoch (seed + epoch), starting at ``step``: a resumed
    run sees the same batches it would have seen."""
    per_epoch = len(dataset) // batch_size
    epoch, offset = divmod(step, per_epoch)
    while True:
        order = np.random.default_rng(seed + epoch).permutation(len(dataset))[offset * batch_size:
                                                                               per_epoch * batch_size]
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=order.tolist(),
                                             collate_fn=_collate, num_workers=workers, drop_last=True)
        yield from loader
        epoch, offset = epoch + 1, 0


def _save(path: Path, state: dict) -> None:
    """Atomic: a save cut short leaves the previous file intact."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _autocast(device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def _run(args, model, evaluate, batches, train_step, better, config: dict, device) -> None:
    """AdamW, warmup and cosine decay over --steps. Every
    --eval-every steps an evaluation (weights kept as step_<n>.pt), and best.pt + best.json
    (``config``, the scores, the scores before training) whenever ``better(scores, best, before)``.
    Resumable: <run-dir>/last.pt holds everything, saved every --save-every steps, at every
    evaluation and on Ctrl+C; the same command resumes."""
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                  weight_decay=args.weight_decay)
    warmup = max(1, int(0.05 * args.steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))),
    )
    last = run / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, history, best = state["step"], state["history"], state["best"]
        logger.info("resumed %s at step %d", last, step)
    else:
        step, history, best = 0, [{"step": 0, **evaluate()}], None
        logger.info("before training: %s", history[0])
        (run / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")

    def state() -> dict:
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step, "history": history, "best": best}

    stream = batches(step)
    model.train()
    started, start_step = time.time(), step
    try:
        while step < args.steps:
            loss, note = train_step(next(stream))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                logger.info("step %d/%d loss %.5f%s, lr %.2e (%.2f steps/s)", step, args.steps, loss.item(), note,
                            scheduler.get_last_lr()[0], (step - start_step) / (time.time() - started))
            if step % args.eval_every == 0 or step == args.steps:
                scores = {"step": step, **evaluate()}
                model.train()
                history.append(scores)
                logger.info("eval @ %d: %s", step, scores)
                with open(run / "metrics.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(scores) + "\n")
                torch.save(model.state_dict(), run / f"step_{step:05d}.pt")
                if better(scores, best, history[0]):
                    best = scores
                    torch.save(model.state_dict(), run / "best.pt")
                    (run / "best.json").write_text(json.dumps({**config, "before": history[0], **scores},
                                                              ensure_ascii=False, indent=1), encoding="utf-8")
                    logger.info("best so far: step %d", step)
                _save(last, state())
            elif step % args.save_every == 0:
                _save(last, state())
    except KeyboardInterrupt:
        _save(last, state())
        logger.warning("interrupted at step %d; saved %s — run the same command to resume", step, last)
        return
    _save(last, state())
    if best:
        logger.info("done: best step %d %s (before training %s)", best["step"], best, history[0])
    else:
        logger.warning("done: no checkpoint beat the model before training; best.pt not written")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune docTR's recognizer on swissdocai data")
    parser.add_argument("--data", default="output/doctr_data")
    parser.add_argument("--run-dir", default="output/doctr_runs/reco",
                        help="Checkpoints, metrics, best.pt; an existing run resumes")
    parser.add_argument("--checkpoint", default=None,
                        help="Start from this fine-tuned .pt (e.g. output/doctr_models/parseq_swiss.pt); "
                             "with --eval-only: score it")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=100, help="Steps between resumable checkpoints")
    parser.add_argument("--val-items", type=int, default=5000, help="Validation crops per set")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--real-share", type=float, default=0.5, help="Share of real crops in a batch")
    parser.add_argument("--eval-only", action="store_true", help="Score docTR's parseq (or --checkpoint) only")
    parser.add_argument("--guard", default="output/benchmarks/funsd/dataset/training_data",
                        help="Real scans scored at every evaluation (FUNSD split dir; evaluation only)")
    parser.add_argument("--guard-tolerance", type=float, default=0.005,
                        help="How far our words or those with umlauts may fall below the model before training")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_reco(args, device)


if __name__ == "__main__":
    main()
