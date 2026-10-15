"""Loading fine-tuned docTR checkpoints (scripts/train_doctr.py output)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import doctr.models  # noqa: E402
import torch  # noqa: E402

from swissdocai.words import load_doctr_model  # noqa: E402


def test_finetuned_recognizer_loads_with_its_vocabulary(tmp_path):
    vocab = "abcäöüß0123456789"
    trained = doctr.models.crnn_mobilenet_v3_small(pretrained=False, pretrained_backbone=False, vocab=vocab)
    torch.save(trained.state_dict(), tmp_path / "reco.pt")
    (tmp_path / "reco.json").write_text(
        json.dumps({"arch": "crnn_mobilenet_v3_small", "task": "reco", "vocab": vocab}), encoding="utf-8")

    model = load_doctr_model(str(tmp_path / "reco.pt"), "reco")

    assert model.vocab == vocab
    for name, weights in trained.state_dict().items():
        assert torch.equal(model.state_dict()[name], weights), name


def test_architecture_names_pass_through():
    assert load_doctr_model("db_resnet50", "det") == "db_resnet50"
