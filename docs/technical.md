# SwissDocAI Redact — technical documentation

What the app does and how to install it is in the README:
[Deutsch](../README.md) · [Français](../README.fr.md) · [Italiano](../README.it.md) · [English](../README.en.md).
This page is for developers: the pipeline, results, hardware, the installers, the command
line, the development setup and the recognizer's training. The processes and modules are
drawn in [architecture.md](architecture.md).

## How it works

Every page is read twice, and the two readings check each other.

1. **Words and boxes.** The PDF text layer gives exact word boxes, and OCR covers
   whatever it lacks: scans, photos, text in images such as letterheads. The OCR is
   docTR's detector with our Swiss recognizer (umlauts, ß, accents) through ONNX
   Runtime, with no PyTorch.
2. **Page reader.** GLM-OCR transcribes each page; this is the text the PII step
   reads. Page words missing from the transcription are added as plain lines, since
   readers skip text. Alternatives: the SwissDocAI OCR service, or none (page words
   only).
3. **Personal data.** A language model finds personal data in the transcription,
   guided by two editable lists: what to black out and what to keep (organizations,
   letter dates, appointments). Locally this is Gemma 4 E4B (QAT, 4-bit) through
   llama.cpp: one pass, asked again if the answer comes back empty; or the SwissDocAI
   PII service, which takes the same texts and lists. Rules with checksums add AHV
   numbers, IBANs, UIDs, labelled identifiers and birth dates
   after "geb."/"*", and optionally phone numbers, emails and names after Dr./Prof.
   Answers are cleaned: sentences, lone titles, organizations, birth dates in the
   future.
4. **Locating.** Every finding is searched on the page words: on their letters and
   digits, immune to spacing, line breaks and glued words, and through the
   transcription's links to the words, which catches OCR misreadings. Where the boxes
   under a value lack characters, its line is read again in cut-outs; a word is never
   taken for its position alone. What stays incomplete is marked for review.
5. **Review** (app). Findings can be toggled per value or per word, signatures and
   QR codes kept, and boxes added by hand.
6. **Output and check.** Padded black bars go over the findings, and small images
   (signatures) next to a redacted name are blacked out whole, and so is every QR code
   unless that is switched off (settings; `--no-qr-codes`): a QR-bill's holds names, addresses
   and the IBAN. The pages are written as an image-only PDF with a report. The result
   is read again to make sure no finding is still readable.

Nothing leaves the machine unless a SwissDocAI service (in Switzerland) is chosen: the
OCR service as page reader, or the PII service in place of the local model. Each costs
one page of credit per document page, so a page with both costs two; the app says so in
its settings.

## Results

On a Radeon RX 7900 XTX (Vulkan / DirectML), everything local unless noted.

**The PII step**: Documents with 598 personal values and
287 look-alikes that should stay readable (organizations and their contact data).
The documents are invoices, bank, authority, medical, insurance and employment
letters in de/fr/it/en.

| PII step | Values found | Documents fully redacted | Look-alikes blacked out | Time per document |
|---|---|---|---|---|
| **Gemma 4 E4B, local, German lists (app and CLI default)** | **99.8%** | **97.6%** | 31% | **3.6 s** |

"Look-alikes blacked out" includes the organizations' phone numbers and emails, which the
contact rule always blacks out (on by default).

**End to end** (`scripts/eval_dpi.py`): the same texts typeset as scan-like PDFs at
200 dpi, with words, page reader, PII step, locating and bars.

| | Values blacked out | Documents fully redacted | Other words blacked out | Time per page |
|---|---|---|---|---|
| 200 dpi (app default) | 99.5% | 93% | 1.7% | 9.8 s |

The remaining misses are a few phone numbers on which docTR's detector skips a digit
group or cuts a box short. The app shows them for review.

**Word layer on real scans** (`scripts/eval_word_layer.py`). The XFUND validation
forms (de/fr/it) and the FUNSD test set (en) are never trained on: their licenses
are non-commercial, so they are used for evaluation only. Words read right, matched
words, through ONNX Runtime as in the app:

| Recognizer | German | of which ä/ö/ü / ß | French | Italian | English |
|---|---|---|---|---|---|
| docTR's PARSeq | 88.6% | 0% / 0% | 91.4% | 94.1% | 85.8% |
| **Swiss (+ real pages, shipped)** | **97.0%** | **95.9% / 98.0%** | **90.5%** | **94.7%** | **85.0%** |

## Desktop app

`desktop/` is a Tauri 2 + React window. It starts the Python engine
(`python -m swissdocai.app.api`, a local HTTP API on 127.0.0.1 with a random port and
token) and stops it, with its model servers, on exit.

From the repository, with uv, Node.js and Rust installed
([Tauri's prerequisites](https://v2.tauri.app/start/prerequisites/)):

```bash
powershell -ExecutionPolicy Bypass -File start.ps1   # Windows
bash start.sh                                        # macOS, Linux
```

The script fetches the Swiss recognizer from Git LFS if the checkout lacks it, creates
`.venv` with the engine's packages (`uv pip install -r pyproject.toml`, again whenever
`pyproject.toml` changes; an existing venv such as the ROCm one below is only added to),
runs `npm ci` in `desktop/` when `package-lock.json` changes, and starts `npm run tauri dev`
with `SWISSDOCAI_ENGINE` pointing at `.venv`'s Python (hot reload of the React UI).

On first start the app downloads what it runs locally, each file checked against
its SHA-256 and resumable:
- the llama.cpp server (release b11541), the build for this machine's GPU (below):
  33 MB for Vulkan, 0.6–0.7 GB for CUDA with its runtime;
- Gemma 4 E4B QAT (5.2 GB);
- GLM-OCR (1.4 GB with its vision projector);
- docTR's detector (100 MB).

They go to `%LOCALAPPDATA%\swissdocai` (or `$XDG_CACHE_HOME/swissdocai`,
`SWISSDOCAI_CACHE`). The Swiss recognizer ships with the package
(`swissdocai/models/parseq_swiss.onnx`, Git LFS).

### Hardware

The app finds out on first start what the language models run on, and says so in
its setup dialog. Nothing needs to be installed beyond the normal graphics driver.

| Machine | llama.cpp build | Fallback |
|---|---|---|
| NVIDIA GPU (driver 528+, 4 GB+) | CUDA 13 (driver 580+, RTX 20 series or newer), else CUDA 12; the CUDA runtime comes with it | Vulkan |
| AMD GPU | ROCm, only where AMD's ROCm libraries are installed (HIP SDK on Windows, ROCm on Linux), since llama.cpp's ROCm builds do not ship hipBLAS/rocBLAS | Vulkan |
| Apple silicon | Metal | CPU |
| Intel or other GPU | Vulkan | CPU |
| no usable GPU | CPU | — |

Each build is asked which GPUs it sees (`llama-server --list-devices`) before it is
kept; one that sees none is deleted and the next is tried. The choice is stored in
`llama.cpp/backend.json` in the cache (delete it to detect again). A model that does
not fit in the GPU's memory runs on the CPU, and a model server that fails on the GPU
is started again on the CPU. The OCR runs on DirectML (Windows, any GPU), else on
the CPU, also on Apple silicon: ONNX Runtime's CoreML cannot build the recognizer.
When a GPU fails the OCR, it reads on the CPU. `swissdocai doctor` shows all of it.

On Windows the app puts the Visual C++ runtime next to llama.cpp, which needs it but
does not ship it, so PCs without the Visual C++ Redistributable work too.

### Installer

```bash
uv run --no-project python installer/build.py      # needs uv, Node.js and Rust
```

1. A clean Python environment with the engine's dependencies only (`build/engine-venv`,
   no PyTorch).
2. The engine frozen with PyInstaller (`installer/swissdocai-engine.spec`, 450 MB
   uncompressed). The script starts it once to check that it answers, detects the
   hardware and ends with its app.
3. The Tauri app with the engine as a resource (`desktop/src-tauri/tauri.release.conf.json`),
   bundled into `desktop/src-tauri/target/release/bundle/`:
   - Windows: `nsis/SwissDocAI Redact_<version>_x64-setup.exe`, installed for the
     current user without administrator rights. Languages: German, French, Italian,
     English. Start menu and desktop shortcuts.
   - macOS: `dmg/` (`--bundles dmg`, Apple silicon).
   - Linux: `appimage/` and `deb/` (`--bundles appimage,deb`).

The installer holds no models and is the same for every machine. The workflow
`.github/workflows/installers.yml` builds all three on GitHub (by hand, or for a
`v*` tag as a draft release; publishing that draft puts the installers on the
Releases page the READMEs link to).

Uninstalling keeps the downloaded models unless "delete the application data" is
ticked; then `%LOCALAPPDATA%\swissdocai` goes too (`desktop/src-tauri/windows/hooks.nsh`).

**Signing.** The installers are unsigned. Windows SmartScreen then warns
("Windows protected your PC" → More info → Run anyway), and macOS blocks the app
until it is allowed under System Settings → Privacy & Security. For customers, sign
them: on Windows with a code-signing certificate (e.g. Azure Trusted Signing, through
`bundle > windows > signCommand`), on macOS with an Apple Developer ID. The workflow
signs and notarizes the macOS build when the `APPLE_*` secrets are set.

## Command line

```bash
swissdocai redact letter.pdf -o letter.redacted.pdf --preview review/   # as the app, all local
swissdocai redact letter.pdf --policy lists.json                        # own lists instead of the German ones
swissdocai redact letter.pdf --llm swissdocai                           # PII step at the SwissDocAI service
swissdocai redact invoice.pdf --no-qr-codes                             # leave QR codes readable
swissdocai doctor                                                       # hardware, what would run
swissdocai analyze invoice.pdf --model zai-org/GLM-OCR -o invoice.json  # words + markdown JSON
```

From the repository: `uv run --no-sync python -m swissdocai ...`, in the `.venv` that
`start.ps1`/`start.sh` set up, or without Node.js and Rust:

```bash
uv venv
uv pip install -r pyproject.toml
uv run --no-sync python -m swissdocai redact letter.pdf
```

Like the app, the first run downloads the models.

`redact` writes a report next to the PDF (`<output>.report.json`). With
`--report-text` it also lists the redacted strings, and then itself contains
personal data. `--preview` writes review images with red boxes instead of black
bars. `--policy` takes a JSON file `{"redact": [...], "keep": [...], "notes": ""}`,
the standard lists of `de`/`fr`/`it`/`en`, or `none` for the categories alone.

`analyze` writes a Document-Intelligence-style JSON: words with boxes, the page
markdown, paragraphs, tables and checkboxes, linked by spans. It needs the `doctr`
and `vlm` extras (PyTorch).

### Configuration

Keys go into the environment or a git-ignored `.env` in the working directory.
Never commit them.

| Variable | For |
|---|---|
| `SWISSDOCAI_API_KEY` | the SwissDocAI services: OCR as page reader (`POST /v1/ocr`), and the PII step (`POST /v1/pii`, `--llm swissdocai`); the calls are in `swissdocai/saas.py`. The app stores the token encrypted (Windows DPAPI) when it is entered in the settings. |
| `SWISSDOCAI_API_URL` | another address for those services |
| `SWISSDOCAI_LLAMA_BACKEND` | force a llama.cpp build: `cuda`, `rocm`, `vulkan`, `sycl`, `metal`, `cpu` (default: chosen from the hardware) |
| `SWISSDOCAI_ROCM_LIBS` | a folder with AMD's hipBLAS/rocBLAS for the ROCm build, e.g. the ROCm wheels' `_rocm_sdk_libraries\bin` in the dev venv |
| `SWISSDOCAI_OCR_DEVICE=cpu` | ONNX Runtime on the CPU even with a GPU |
| `SWISSDOCAI_CACHE` | where models and logs go |
| `SWISSDOCAI_ENGINE` | the engine the app starts, `program\|arg\|arg` (set by `start.ps1`/`start.sh`); default: the frozen engine shipped with the app |

## Setup (development)

### Windows with AMD ROCm (RX 7900 XTX, gfx1100, ROCm 10.0)

Prerequisites: **AMD Software: Adrenalin Edition**, **uv** (https://docs.astral.sh/uv/).

> **Use uv's managed Python.** A venv created from the Anaconda interpreter
> breaks ROCm DLL initialization (WinError 1114 on `amdhip64_7.dll`).

```bash
uv venv --python 3.12.13     # .python-version pins 3.12.13 (managed)
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ -r requirements-rocm.txt
uv pip install --index-strategy unsafe-any-match \
    --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ \
    -r requirements.txt
uv run --no-sync python scripts/check_rocm.py   # PyTorch sees the GPU, matmul, a training step
SWISSDOCAI_OCR_DEVICE=cpu uv run --no-sync python -m pytest -q
```

> ⚠️ **Never run `uv sync` in this repo.** Dependencies are managed with `uv pip`
> (the AMD wheel index does not support uv's lockfile resolution); `uv sync` would
> prune the ROCm stack out of the venv. Use `uv run --no-sync`.

PyTorch is only needed for docTR with PyTorch, the transformers readers, and
recognizer training. The engine itself needs just `pip install .`:
- ONNX Runtime comes as `onnxruntime-directml` on Windows and `onnxruntime`
  elsewhere;
- the two provide the same module, so never install both;
- extras: `doctr`, `vlm`, `train`, `dev`.

## Benchmarks

```bash
# end to end on scan-like PDFs (the PII benchmark texts, output/pii_benchmark/benchmark.jsonl), per resolution
uv run --no-sync python scripts/eval_dpi.py --dpi 150 --dpi 200
# the word layer on real scans: detection, words read right, umlauts, digit groups
uv run --no-sync python scripts/eval_word_layer.py --dataset output/benchmarks/xfund/de.val.json \
    output/benchmarks/xfund/fr.val.json output/benchmarks/xfund/it.val.json \
    output/benchmarks/funsd/dataset/testing_data output/dpi_benchmark/truth.json \
    --ocr-reco output/doctr_models/parseq_swiss.pt
```

XFUND: `<lang>.val.json` + `<lang>.val.zip` from github.com/doc-analysis/XFUND/releases,
images extracted next to the json. FUNSD: guillaumejaume.github.io/FUNSD.

## Training the recognizer

docTR's PARSeq with the Swiss vocabulary (`swissdocai/vocab.py`; the output head
re-initialized):

```bash
# word crops from datasets with word boxes, and from real pages with reference text
uv run --no-sync python scripts/export_doctr_data.py \
    --datasets output/dataset_paged output/fedlex_dataset --out output/doctr_data
uv run --no-sync python scripts/export_doctr_data.py --out output/doctr_data \
    --real-reco output/unified_dataset_v4 --real-reco-pages 2000
# train from the current model; checkpoints every 100 steps, the same command resumes
uv run --no-sync python scripts/train_doctr.py --data output/doctr_data \
    --checkpoint output/doctr_models/parseq_swiss.pt --run-dir output/doctr_runs/reco_v3
# the app's model: best.pt and best.json copied to output/doctr_models/parseq_swiss.pt / .json
uv run --no-sync python scripts/export_onnx.py output/doctr_models/parseq_swiss.pt --out swissdocai/models
```

Round 1 learnt on our renders alone. It read umlauts but real French and English
scans worse than docTR's model: t read as f, letters in the wrong case. Round 2 mixes
in word crops from 2,000 real pages, labelled by aligning each page's reference text
to its word boxes. Labels the reference text more likely got wrong than the
recognizer are dropped: re-cased headings, cut boxes, quote style.

A checkpoint only counts while our words and those with umlauts or ß hold; real
crops and real scans (the FUNSD training forms, evaluation only) choose it. Before
it replaces the shipped model, it must hold on the real-scan benchmark above.
`export_onnx.py` names the ONNX after the `.pt`, hence the copy.

docTR's detector stays as it is. Two fine-tuning rounds (tag `pre-cleanup`) made it
worse on real scans: boxes on specks and fill lines, and box sizes drifting with
docTR's loss. Its "missed" digit groups are mostly merged with their neighbours into
one box, which is harmless for redaction.

### Data

| Source | Used for | License |
|---|---|---|
| Synthetic Swiss documents (generated here, de/en/fr/it) | recognizer crops | own |
| Fedlex, Swiss federal law (de/fr/it/rm) | recognizer crops | Art. 5 CopA; Art. 49 PublO notice on redistribution |
| olmOCR-mix-1025, DoclingMatix (real pages) | recognizer crops with reference text | ODC-BY, CDLA-Permissive-2.0 |
| XFUND, FUNSD | evaluation only | non-commercial |

The datasets live in `output/` (git-ignored). The generators that built them were
removed with the cleanup; they are at the tag `pre-cleanup`.

The recognizer `swissdocai/models/parseq_swiss.onnx` is licensed under the Apache License
2.0, like the code. It is docTR's pretrained PARSeq (Apache 2.0) fine-tuned on word crops
from the first three sources only: 502,909 from our renders and Fedlex, 109,453 from
olmOCR-mix-1025 and DoclingMatix pages. XFUND and FUNSD are not in its training data or in
this repository; they only measure it: the benchmark above, and the FUNSD training forms
scored while it trains.

## Project structure

```
SwissDocAI-Redact/
├── swissdocai/                 # the engine (pip install .)
│   ├── redact.py               #   find, locate, review, black out (the product)
│   ├── pii.py, policy.py       #   PII step: prompt, rules, plausibility; the editable lists
│   ├── engine.py               #   local models: llama.cpp, downloads, hardware
│   ├── llm.py, saas.py         #   chat with the local llama.cpp server; the SwissDocAI API (OCR, PII)
│   ├── words.py                #   word layer: PDF text layer, OnnxTR, docTR
│   ├── models/                 #   the Swiss recognizer (ONNX, Git LFS)
│   ├── pipeline.py, result.py  #   pages of a document; the analyze JSON
│   ├── markdown_doc.py, align.py, selection_marks.py, otsl.py, text.py, vocab.py
│   ├── vlm.py, prompts.py      #   page readers and their prompts
│   ├── app/                    #   the desktop app's engine: API, session, token storage
│   └── cli.py                  #   swissdocai redact / analyze / doctor
├── desktop/                    # Tauri 2 + React desktop app
├── installer/                  # the engine frozen with PyInstaller, build.py for the installers
├── scripts/                    # benchmarks, recognizer training, check_rocm.py
├── tests/
├── docs/technical.md           # this page
├── docs/architecture.md        # pipeline and processes
├── start.ps1, start.sh         # the app from the repository (Windows; macOS, Linux)
├── README.md                   # for users: Deutsch; README.fr.md, README.it.md, README.en.md
└── requirements-rocm.txt, requirements.txt, pyproject.toml
```
