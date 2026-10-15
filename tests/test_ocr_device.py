"""Where the OCR runs: ONNX Runtime's providers by hardware, and the CPU when a GPU provider
fails (no models, no GPU)."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

ort = pytest.importorskip("onnxruntime")

from PIL import Image  # noqa: E402

from swissdocai import engine, words  # noqa: E402
from swissdocai.engine import GPU, Hardware  # noqa: E402

CPU = ["CPUExecutionProvider"]


def test_the_ocr_device_follows_the_hardware(monkeypatch):
    monkeypatch.delenv("SWISSDOCAI_OCR_DEVICE", raising=False)
    hw = Hardware("Windows", "x86_64", 32.0, [GPU("amd", "RX 7900 XTX", 24)])
    monkeypatch.setattr(engine, "detect_hardware", lambda: hw)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    assert words.onnx_providers() == ["DmlExecutionProvider", "CPUExecutionProvider"]
    hw.gpus = []  # e.g. a virtual machine: not DirectML on Windows' software renderer
    assert words.onnx_providers() == CPU

    hw.system, hw.machine, hw.gpus = "Darwin", "arm64", [GPU("apple", "Apple silicon", 16)]
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    assert words.onnx_providers() == CPU  # CoreML cannot build the recognizer


class Fail(Exception):
    """Like ONNX Runtime's own errors, which are no RuntimeError."""


SHAPE_ERROR = "model_builder.cc:851 RegisterModelInputOutput Unable to get shape for output: /Squeeze_output_0"


def result():
    word = SimpleNamespace(value="Kuhn", confidence=0.99, geometry=((0.1, 0.1), (0.2, 0.15)))
    return SimpleNamespace(pages=[SimpleNamespace(blocks=[SimpleNamespace(lines=[SimpleNamespace(words=[word])])])])


@pytest.mark.parametrize("fails_when", ["loading", "reading"])
def test_a_failing_gpu_provider_falls_back_to_the_cpu(monkeypatch, fails_when):
    monkeypatch.setattr(words, "onnx_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    built = []

    def build(self, providers):
        built.append(providers[0])
        on_gpu = providers != CPU
        if on_gpu and fails_when == "loading":
            raise Fail(SHAPE_ERROR)

        def predict(pages):
            if on_gpu:
                raise Fail(SHAPE_ERROR)
            return result()
        return predict

    monkeypatch.setattr(words.OnnxWordReader, "_build", build)
    reader = words.OnnxWordReader(reco_arch="parseq")
    page = reader(Image.new("RGB", (1000, 800), "white"))
    assert [w.text for w in page.words] == ["Kuhn"] and page.words[0].bbox == (100.0, 80.0, 200.0, 120.0)
    assert built == ["DmlExecutionProvider", "CPUExecutionProvider"] and reader.providers == CPU
    reader(Image.new("RGB", (1000, 800), "white"))
    assert len(built) == 2  # stays on the CPU


def test_a_failing_cpu_is_reported(monkeypatch):
    monkeypatch.setattr(words, "onnx_providers", lambda: CPU)

    def build(self, providers):
        raise Fail("broken model")

    monkeypatch.setattr(words.OnnxWordReader, "_build", build)
    with pytest.raises(Fail):
        words.OnnxWordReader(reco_arch="parseq")(Image.new("RGB", (100, 100), "white"))
