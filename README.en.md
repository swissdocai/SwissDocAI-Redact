<p align="center">
  <img src="desktop/public/logo.png" alt="SwissDocAI" width="340">
</p>

<h1 align="center">SwissDocAI Redact</h1>

<p align="center">
  <strong>Black out personal data in documents automatically – on your own computer.</strong>
</p>

<p align="center">
  <a href="README.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.it.md">Italiano</a> · <b>English</b>
</p>

---

SwissDocAI Redact finds the personal data in Swiss documents and blacks it out. You drop a
document into the window, check what the app found, and save a redacted PDF.

## What is blacked out

- Names, birth dates and addresses
- Phone numbers and email addresses
- AHV numbers (Swiss social security), IBANs, and patient, policy and case numbers
- Signatures next to a blacked-out name
- QR codes: the payment part of a QR-bill holds names, addresses and the IBAN

Organizations, letter dates and appointments stay readable. You can change what is blacked out
and what is not in the settings.

The app reads PDFs, scans and phone photos (JPG, PNG, TIFF) in German, French, Italian and
English. The result is a new PDF made of page images only: no text layer, metadata,
attachments or comments in which blacked-out data could live on.

## Your data stays with you

The app works entirely on your computer. After the one-time setup it needs no internet
connection, and no document leaves the computer.

If you like, you can hand single steps to **SwissDocAI Cloud**, processed in Switzerland
([swissdocai.ch](https://swissdocai.ch)) – for instance when your computer has no suitable
graphics card. It is an option under "Settings" → "General", off by default:

- **Read the pages:** the page images are read in Switzerland.
- **Find personal data:** only the text of the pages is checked in Switzerland, no images.

Each service costs one page of credit per document page, both together two. You need an API
token from the SwissDocAI console.

## Installation

Download the installer for your system from the
**[releases page](https://github.com/swissdocai/SwissDocAI-Redact/releases/latest)**:

| System | File |
|---|---|
| Windows 10 and 11 | `…_x64-setup.exe` – installs without administrator rights |
| Mac with an Apple chip (M1 or newer), macOS 12 or later | `….dmg` |
| Linux | `….AppImage` or `….deb` (Debian, Ubuntu) |

The installers are not signed yet, so your system warns you the first time you open them:

- **Windows:** if "Windows protected your PC" appears, click "More info", then "Run anyway".
- **macOS:** open the app once. If it is blocked, go to System Settings → Privacy & Security
  and click "Open Anyway".
- **Linux:** mark the AppImage file as executable in its properties (or `chmod +x` in a
  terminal).

## First start

On its first start the app sets itself up once: it downloads its models, about 7 GB – it shows
the exact size. The download can be interrupted and resumed later. Plan for about 8 GB of free
disk space. The models are stored in `%LOCALAPPDATA%\swissdocai` (Windows) or
`~/.cache/swissdocai` (macOS, Linux); when you uninstall on Windows, you can have them deleted
too.

The app finds out by itself whether a graphics card can speed up the work (NVIDIA, AMD, Apple,
Intel), and shows what it uses. Apart from the normal graphics driver there is nothing to
install. Without a suitable graphics card everything runs on the processor – that works, but
is slower.

## Redacting a document

1. Drop PDFs or images into the window, or click "Choose files".
2. Click "Find personal data" – for several documents, "Analyze all".
3. Check the personal data found. Untick a value and it stays readable everywhere. On the page,
   click a word to black it out or keep it readable, and drag a box for a redaction of your
   own. Look especially carefully at the places under "Please check".
4. Click "Save redacted PDF". The app reads the result again and tells you if a blacked-out
   value is still readable.

Under "Settings" → "What to black out" you adjust the lists: what is blacked out and what stays
readable. Under "Further notes" you can describe the documents, for example: "These are medical
reports. Do not black out the names of doctors." The app's language is chosen under "General".

> [!IMPORTANT]
> Automatic redaction is very reliable, but not infallible: in our tests with invoices and letters from
> banks, public authorities, medical practices, insurers and employers, 99.5% of personal data was
> redacted. By outsourcing to SwissDocAI, we achieved 99.9%. Please review every document before sharing it.

## How it works

Every page is read twice, and the two readings check each other:

1. Text recognition (OCR) finds every word and its position on the page. It uses a model
   trained for Swiss documents, which also reads umlauts and accents.
2. A reading model transcribes the whole page, so that the context is kept: whose name is this?
3. A language model looks for the personal data in that text, guided by your lists. Rules with
   check digits add AHV numbers, IBANs and other numbers.
4. Every finding is located on the page and covered in black. Finally, the result is read
   again.

All the details – measurements, hardware, command line, development – are in the
[technical documentation](docs/technical.md).

## Running without the installer

For developers, and for anyone who wants to start the app from source. You need:

- [Git](https://git-scm.com) with [Git LFS](https://git-lfs.com) (included in Git for Windows)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python
- [Node.js](https://nodejs.org) (LTS)
- [Rust](https://rustup.rs) with [Tauri's prerequisites](https://v2.tauri.app/start/prerequisites/)
  for your system (Windows: Microsoft C++ Build Tools; macOS: Xcode Command Line Tools;
  Linux: the WebKitGTK packages)

```bash
git clone https://github.com/swissdocai/SwissDocAI-Redact.git
cd SwissDocAI-Redact
```

Then on Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

On macOS and Linux:

```bash
bash start.sh
```

The script sets up whatever is missing – the Python environment, the Node.js packages and the
text recognition model from Git LFS – and starts the app. The first start takes a few minutes
because the app is compiled; after that it is quick.

Command line only, without Node.js and Rust:

```bash
uv venv
uv pip install -r pyproject.toml
uv run --no-sync python -m swissdocai redact letter.pdf
```

The redacted PDF is saved next to it as `letter.redacted.pdf`. More commands are in the
[technical documentation](docs/technical.md#command-line).

## Contact

Questions, feedback or a problem with the app? Write to us at
[support@swissdocai.ch](mailto:support@swissdocai.ch).

## License

SwissDocAI Redact is open source under the [Apache License 2.0](LICENSE).
Copyright 2026 SwissDocAI.

---

<p align="center">SwissDocAI · <a href="https://swissdocai.ch">swissdocai.ch</a> · <a href="mailto:support@swissdocai.ch">support@swissdocai.ch</a></p>
