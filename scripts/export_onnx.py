"""
Export a fine-tuned docTR model (scripts/train_doctr.py: <name>.pt + <name>.json)
to ONNX for words.OnnxWordReader — the desktop app runs without PyTorch.

Writes <out>/<name>.onnx and <out>/<name>.json (architecture, task, vocab),
optionally also an 8-bit version (<name>_int8.onnx, dynamic quantization:
a quarter of the size).

Usage:
    uv run --no-sync python scripts/export_onnx.py output/doctr_models/parseq_swiss.pt --out swissdocai/models --int8
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def export(pt: Path, out_dir: Path, int8: bool) -> list[Path]:
    import doctr.models
    import torch
    from doctr.models.utils import export_model_to_onnx

    config = json.loads(pt.with_suffix(".json").read_text(encoding="utf-8"))
    kwargs = {"vocab": config["vocab"]} if config["task"] == "reco" else {}
    model = getattr(doctr.models, config["arch"])(pretrained=False, pretrained_backbone=False, exportable=True,
                                                  **kwargs)
    model.load_state_dict(torch.load(pt, map_location="cpu"))
    model.eval()
    shape = (1, 3, 32, 128) if config["task"] == "reco" else (1, 3, 1024, 1024)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{pt.stem}.onnx"
    exported = Path(export_model_to_onnx(model, str(target.with_suffix("")), torch.rand(shape)))
    if exported != target:
        shutil.move(exported, target)
    meta = {key: config[key] for key in ("arch", "task", "vocab") if key in config}
    meta.update({key: config[key] for key in ("scores",) if key in config})
    target.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    written = [target]
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized = out_dir / f"{pt.stem}_int8.onnx"
        quantize_dynamic(str(target), str(quantized), weight_type=QuantType.QUInt8)
        quantized.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        written.append(quantized)
    return written


def main():
    parser = argparse.ArgumentParser(description="Export a fine-tuned docTR model to ONNX")
    parser.add_argument("model", help="<name>.pt with <name>.json next to it")
    parser.add_argument("--out", default="swissdocai/models")
    parser.add_argument("--int8", action="store_true", help="Also write an 8-bit quantized copy")
    args = parser.parse_args()
    for path in export(Path(args.model), Path(args.out), args.int8):
        print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
