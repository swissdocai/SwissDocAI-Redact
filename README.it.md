<p align="center">
  <img src="desktop/public/logo.png" alt="SwissDocAI" width="340">
</p>

<h1 align="center">SwissDocAI Redact</h1>

<p align="center">
  <strong>Oscurare automaticamente i dati personali nei documenti – sul tuo computer.</strong>
</p>

<p align="center">
  <a href="README.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <b>Italiano</b> · <a href="README.en.md">English</a>
</p>

---

SwissDocAI Redact trova i dati personali nei documenti svizzeri e li oscura. Trascini un
documento nella finestra, verifichi ciò che l'app ha trovato e salvi un PDF oscurato.

## Cosa viene oscurato

- Nomi, date di nascita e indirizzi
- Numeri di telefono e indirizzi e-mail
- Numeri AVS, IBAN e numeri di paziente, di polizza e di pratica
- Firme accanto a un nome oscurato
- Codici QR: la sezione pagamento di una QR-fattura contiene nomi, indirizzi e IBAN

Le organizzazioni, le date delle lettere e gli appuntamenti restano leggibili. Nelle
impostazioni puoi adattare ciò che viene oscurato e ciò che non lo è.

L'app legge PDF, scansioni e foto scattate con il telefono (JPG, PNG, TIFF) in tedesco,
francese, italiano e inglese. Il risultato è un nuovo PDF composto solo da immagini delle
pagine: senza livello di testo, metadati, allegati o commenti in cui i dati oscurati potrebbero
sopravvivere.

## I tuoi dati restano tuoi

L'app lavora interamente sul tuo computer. Dopo la configurazione iniziale non ha più bisogno di
Internet e nessun documento lascia il computer.

Se lo desideri, puoi affidare singole fasi a **SwissDocAI Cloud**, con elaborazione in Svizzera
([swissdocai.ch](https://swissdocai.ch)) – per esempio se il computer non ha una scheda grafica
adatta. È un'opzione in «Impostazioni» → «Generale», disattivata per impostazione predefinita:

- **Lettura delle pagine:** le immagini delle pagine vengono lette in Svizzera.
- **Ricerca dei dati personali:** in Svizzera viene verificato solo il testo delle pagine,
  nessuna immagine.

Ogni servizio costa una pagina di credito per ogni pagina del documento, entrambi insieme due.
Serve un token API della console SwissDocAI.

## Installazione

Scarica il file di installazione per il tuo sistema dalla
**[pagina delle release](https://github.com/swissdocai/SwissDocAI-Redact/releases/latest)**:

| Sistema | File |
|---|---|
| Windows 10 e 11 | `…_x64-setup.exe` – si installa senza diritti di amministratore |
| Mac con chip Apple (M1 o successivo), da macOS 12 | `….dmg` |
| Linux | `….AppImage` o `….deb` (Debian, Ubuntu) |

I file di installazione non sono ancora firmati. Per questo il sistema mostra un avviso alla
prima apertura:

- **Windows:** se compare «PC protetto da Windows», clicca su «Ulteriori informazioni» e poi su
  «Esegui comunque».
- **macOS:** apri l'app una volta. Se viene bloccata, vai in Impostazioni di Sistema → Privacy e
  sicurezza e clicca su «Apri comunque».
- **Linux:** contrassegna il file AppImage come eseguibile nelle sue proprietà
  (oppure `chmod +x` nel terminale).

## Primo avvio

Al primo avvio l'app si configura una sola volta: scarica i suoi modelli, circa 7 GB – ne indica
la dimensione esatta. Il download può essere interrotto e ripreso più tardi. Prevedi circa 8 GB
di spazio libero. I modelli si trovano in `%LOCALAPPDATA%\swissdocai` (Windows) o
`~/.cache/swissdocai` (macOS, Linux); disinstallando l'app su Windows puoi eliminarli insieme
all'app.

L'app riconosce da sola se una scheda grafica può accelerare il lavoro (NVIDIA, AMD, Apple,
Intel) e indica cosa utilizza. Oltre al normale driver grafico non devi installare nulla. Senza
una scheda grafica adatta tutto gira sul processore – funziona, ma è più lento.

## Come oscurare un documento

1. Trascina PDF o immagini nella finestra, oppure clicca su «Scegli file».
2. Clicca su «Trova i dati personali» – con più documenti, su «Analizza tutto».
3. Verifica i dati personali trovati. Se togli la spunta a un valore, resta leggibile ovunque.
   Sulla pagina, clicca su una parola per oscurarla o lasciarla leggibile e trascina un riquadro
   per un oscuramento manuale. Esamina con particolare attenzione i punti sotto «Da verificare».
4. Clicca su «Salva il PDF oscurato». L'app rilegge il risultato e segnala se un valore oscurato
   è ancora leggibile.

In «Impostazioni» → «Cosa oscurare» adatti gli elenchi: ciò che viene oscurato e ciò che resta
leggibile. In «Altre indicazioni» puoi descrivere i documenti, per esempio: «Sono referti
medici. Non oscurare i nomi dei medici.» La lingua dell'app si sceglie in «Generale».

> [!IMPORTANT]
> L’oscuramento automatico è molto affidabile, ma non infallibile: nei nostri test con fatture e
> lettere di banche, autorità, studi medici, assicurazioni e datori di lavoro è stato oscurato il 99,5 %
> dei dati personali; con l’esternalizzazione a SwissDocAI abbiamo raggiunto il 99,9 %. Controlli
> ogni documento prima di inoltrarlo.

## Come funziona

Ogni pagina viene letta due volte e le due letture si controllano a vicenda:

1. Un riconoscimento del testo (OCR) trova ogni parola e la sua posizione sulla pagina. Usa un
   modello addestrato appositamente per i documenti svizzeri, che legge anche dieresi e accenti.
2. Un modello di lettura trascrive l'intera pagina, così da conservare il contesto: di chi è quel
   nome?
3. Un modello linguistico cerca i dati personali in questo testo, secondo i tuoi elenchi. Regole
   con cifre di controllo aggiungono numeri AVS, IBAN e altri numeri.
4. Ogni dato trovato viene localizzato sulla pagina e coperto di nero. Alla fine il risultato
   viene riletto.

Tutti i dettagli – risultati delle misurazioni, hardware, riga di comando, sviluppo – si trovano
nella [documentazione tecnica](docs/technical.md) (in inglese).

## Avviare senza programma di installazione

Per sviluppatrici e sviluppatori e per chi vuole avviare l'app dal codice sorgente. Ti servono:

- [Git](https://git-scm.com) con [Git LFS](https://git-lfs.com) (incluso in Git per Windows)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) per Python
- [Node.js](https://nodejs.org) (LTS)
- [Rust](https://rustup.rs) con i [prerequisiti di Tauri](https://v2.tauri.app/start/prerequisites/)
  per il tuo sistema (Windows: Microsoft C++ Build Tools; macOS: Xcode Command Line Tools;
  Linux: i pacchetti WebKitGTK)

```bash
git clone https://github.com/swissdocai/SwissDocAI-Redact.git
cd SwissDocAI-Redact
```

Poi, su Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

Su macOS e Linux:

```bash
bash start.sh
```

Lo script configura ciò che manca – l'ambiente Python, i pacchetti Node.js e il modello di
riconoscimento del testo da Git LFS – e avvia l'app. Il primo avvio richiede alcuni minuti
perché l'app viene compilata; dopo è rapido.

Solo la riga di comando, senza Node.js e Rust:

```bash
uv venv
uv pip install -r pyproject.toml
uv run --no-sync python -m swissdocai redact lettera.pdf
```

Il PDF oscurato viene salvato accanto, come `lettera.redacted.pdf`. Gli altri comandi si trovano
nella [documentazione tecnica](docs/technical.md#command-line).

## Contatto

Domande, osservazioni o un problema con l'app? Scrivici a
[support@swissdocai.ch](mailto:support@swissdocai.ch).

## Licenza

SwissDocAI Redact è open source con [licenza Apache 2.0](LICENSE).
Copyright 2026 SwissDocAI.

---

<p align="center">SwissDocAI · <a href="https://swissdocai.ch">swissdocai.ch</a> · <a href="mailto:support@swissdocai.ch">support@swissdocai.ch</a></p>
