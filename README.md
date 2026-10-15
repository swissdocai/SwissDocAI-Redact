<p align="center">
  <img src="desktop/public/logo.png" alt="SwissDocAI" width="340">
</p>

<h1 align="center">SwissDocAI Redact</h1>

<p align="center">
  <strong>Personendaten in Dokumenten automatisch schwärzen – auf Ihrem eigenen Computer.</strong>
</p>

<p align="center">
  <b>Deutsch</b> · <a href="README.fr.md">Français</a> · <a href="README.it.md">Italiano</a> · <a href="README.en.md">English</a>
</p>

---

SwissDocAI Redact findet Personendaten in Schweizer Dokumenten und schwärzt sie. Sie ziehen
ein Dokument ins Fenster, prüfen, was die App gefunden hat, und speichern ein geschwärztes PDF.

## Was geschwärzt wird

- Namen, Geburtsdaten und Adressen
- Telefonnummern und E-Mail-Adressen
- AHV-Nummern, IBAN sowie Patienten-, Policen- und Fallnummern
- Unterschriften neben einem geschwärzten Namen
- QR-Codes: Der Zahlteil einer QR-Rechnung enthält Namen, Adressen und IBAN

Organisationen, Briefdaten und Termine bleiben lesbar. Was geschwärzt wird und was nicht,
können Sie in den Einstellungen anpassen.

Die App liest PDFs, Scans und Handyfotos (JPG, PNG, TIFF) auf Deutsch, Französisch,
Italienisch und Englisch. Das Ergebnis ist ein neues PDF aus reinen Seitenbildern: ohne
Textebene, Metadaten, Anhänge oder Kommentare, in denen geschwärzte Angaben weiterleben
könnten.

## Ihre Daten bleiben bei Ihnen

Die App arbeitet vollständig auf Ihrem Computer. Nach der einmaligen Einrichtung braucht sie
kein Internet mehr, und kein Dokument verlässt den Computer.

Wer möchte, kann einzelne Schritte an **SwissDocAI Cloud** abgeben, verarbeitet in der Schweiz
([swissdocai.ch](https://swissdocai.ch)) – etwa wenn der Computer keine passende Grafikkarte
hat. Das ist eine Option unter «Einstellungen» → «Allgemein» und standardmässig ausgeschaltet:

- **Seiten lesen:** Die Seitenbilder werden in der Schweiz gelesen.
- **Personendaten finden:** Nur der Text der Seiten wird in der Schweiz geprüft, keine Bilder.

Jeder Dienst kostet pro Dokumentseite eine Seite Guthaben, beide zusammen zwei. Dafür brauchen
Sie einen API-Token aus der SwissDocAI Console.

## Installation

Laden Sie die Installationsdatei für Ihr System von der
**[Release-Seite](https://github.com/swissdocai/SwissDocAI-Redact/releases/latest)** herunter:

| System | Datei |
|---|---|
| Windows 10 und 11 | `…_x64-setup.exe` – installiert ohne Administratorrechte |
| Mac mit Apple-Chip (M1 oder neuer), ab macOS 12 | `….dmg` |
| Linux | `….AppImage` oder `….deb` (Debian, Ubuntu) |

Die Installationsdateien sind noch nicht signiert. Darum warnt Ihr System beim ersten Öffnen:

- **Windows:** Erscheint «Der Computer wurde durch Windows geschützt», klicken Sie auf
  «Weitere Informationen» und dann auf «Trotzdem ausführen».
- **macOS:** Öffnen Sie die App einmal. Wird sie blockiert, gehen Sie zu Systemeinstellungen →
  Datenschutz & Sicherheit und klicken auf «Dennoch öffnen».
- **Linux:** Markieren Sie die AppImage-Datei in ihren Eigenschaften als ausführbar
  (oder `chmod +x` im Terminal).

## Erster Start

Beim ersten Start richtet sich die App einmalig ein: Sie lädt ihre Modelle herunter, etwa
7 GB – die genaue Grösse zeigt sie an. Der Download lässt sich unterbrechen und später
fortsetzen. Planen Sie rund 8 GB freien Speicherplatz ein. Die Modelle liegen unter
`%LOCALAPPDATA%\swissdocai` (Windows) bzw. `~/.cache/swissdocai` (macOS, Linux); beim
Deinstallieren unter Windows können Sie sie mitlöschen lassen.

Die App erkennt selbst, ob eine Grafikkarte die Arbeit beschleunigen kann (NVIDIA, AMD, Apple,
Intel), und zeigt an, was sie verwendet. Ausser dem normalen Grafiktreiber müssen Sie nichts
installieren. Ohne passende Grafikkarte läuft alles auf dem Prozessor – das funktioniert, ist
aber langsamer.

## So schwärzen Sie ein Dokument

1. Ziehen Sie PDFs oder Bilder ins Fenster, oder klicken Sie auf «Dateien auswählen».
2. Klicken Sie auf «Personendaten finden» – bei mehreren Dokumenten auf «Alle analysieren».
3. Prüfen Sie die gefundenen Personendaten. Entfernen Sie einen Haken, bleibt der Wert überall
   lesbar. Auf der Seite klicken Sie ein Wort an, um es zu schwärzen oder lesbar zu lassen, und
   ziehen einen Rahmen für eine eigene Schwärzung. Die Stellen unter «Bitte prüfen» sollten Sie
   sich besonders genau ansehen.
4. Klicken Sie auf «Geschwärztes PDF speichern». Die App liest das Ergebnis noch einmal und
   meldet, falls ein geschwärzter Wert noch lesbar ist.

Unter «Einstellungen» → «Was schwärzen» passen Sie die Listen an: was geschwärzt wird und was
lesbar bleibt. Unter «Weitere Hinweise» können Sie die Dokumente beschreiben, zum Beispiel:
«Es sind Arztberichte. Namen von Ärztinnen und Ärzten nicht schwärzen.» Die Sprache der App
wählen Sie unter «Allgemein».

> [!IMPORTANT]
> Die automatische Schwärzung ist sehr zuverlässig, aber nicht unfehlbar: In unseren Tests mit
> Rechnungen und Briefen von Banken, Behörden, Arztpraxen, Versicherungen und Arbeitgebern
> wurden 99,5 % der Personendaten geschwärzt, mit der Auslagerung an SwissDocAI erreichten wir 99,9 %.
> Sehen Sie jedes Dokument durch, bevor Sie es weitergeben.

## So funktioniert es

Jede Seite wird zweimal gelesen, und die beiden Lesungen prüfen sich gegenseitig:

1. Eine Texterkennung (OCR) findet jedes Wort und seine Position auf der Seite. Sie nutzt ein
   eigens für Schweizer Dokumente trainiertes Modell, das auch Umlaute und Akzente liest.
2. Ein Lesemodell schreibt die ganze Seite ab, damit der Zusammenhang erhalten bleibt: Wessen
   Name steht da?
3. Ein Sprachmodell sucht in diesem Text die Personendaten, nach Ihren Listen. Regeln mit
   Prüfziffern ergänzen AHV-Nummern, IBAN und weitere Nummern.
4. Jeder Fund wird auf der Seite gesucht und schwarz überdeckt. Zum Schluss wird das Ergebnis
   noch einmal gelesen.

Alle Details – Messergebnisse, Hardware, Kommandozeile, Entwicklung – stehen in der
[technischen Dokumentation](docs/technical.md) (Englisch).

## Ohne Installer starten

Für Entwicklerinnen und Entwickler und alle, die die App aus dem Quellcode starten möchten.
Sie brauchen:

- [Git](https://git-scm.com) mit [Git LFS](https://git-lfs.com) (bei Git für Windows dabei)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) für Python
- [Node.js](https://nodejs.org) (LTS)
- [Rust](https://rustup.rs) mit den [Voraussetzungen von Tauri](https://v2.tauri.app/start/prerequisites/)
  für Ihr System (Windows: Microsoft C++ Build Tools; macOS: Xcode Command Line Tools;
  Linux: die WebKitGTK-Pakete)

```bash
git clone https://github.com/swissdocai/SwissDocAI-Redact.git
cd SwissDocAI-Redact
```

Dann unter Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

Unter macOS und Linux:

```bash
bash start.sh
```

Das Skript richtet ein, was fehlt – die Python-Umgebung, die Node.js-Pakete und das
Texterkennungsmodell aus Git LFS –, und startet die App. Der erste Start dauert einige Minuten,
weil die App dabei kompiliert wird; danach geht es schnell.

Nur die Kommandozeile, ohne Node.js und Rust:

```bash
uv venv
uv pip install -r pyproject.toml
uv run --no-sync python -m swissdocai redact Brief.pdf
```

Das geschwärzte PDF wird als `Brief.redacted.pdf` daneben gespeichert. Weitere Befehle stehen in
der [technischen Dokumentation](docs/technical.md#command-line).

## Kontakt

Fragen, Rückmeldungen oder ein Problem mit der App? Schreiben Sie uns an
[support@swissdocai.ch](mailto:support@swissdocai.ch).

## Lizenz

SwissDocAI Redact ist Open Source unter der [Apache-Lizenz 2.0](LICENSE).
Copyright 2026 SwissDocAI.

---

<p align="center">SwissDocAI · <a href="https://swissdocai.ch">swissdocai.ch</a> · <a href="mailto:support@swissdocai.ch">support@swissdocai.ch</a></p>
