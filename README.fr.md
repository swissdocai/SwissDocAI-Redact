<p align="center">
  <img src="desktop/public/logo.png" alt="SwissDocAI" width="340">
</p>

<h1 align="center">SwissDocAI Redact</h1>

<p align="center">
  <strong>Caviarder automatiquement les données personnelles de vos documents – sur votre propre ordinateur.</strong>
</p>

<p align="center">
  <a href="README.md">Deutsch</a> · <b>Français</b> · <a href="README.it.md">Italiano</a> · <a href="README.en.md">English</a>
</p>

---

SwissDocAI Redact trouve les données personnelles dans les documents suisses et les caviarde.
Vous glissez un document dans la fenêtre, vérifiez ce que l'application a trouvé et enregistrez
un PDF caviardé.

## Ce qui est caviardé

- Noms, dates de naissance et adresses
- Numéros de téléphone et adresses e-mail
- Numéros AVS, IBAN ainsi que numéros de patient, de police et de dossier
- Signatures à côté d'un nom caviardé
- Codes QR : la section paiement d'une QR-facture contient des noms, des adresses et l'IBAN

Les organisations, les dates des lettres et les rendez-vous restent lisibles. Vous pouvez
adapter dans les paramètres ce qui est caviardé et ce qui ne l'est pas.

L'application lit les PDF, les scans et les photos prises avec un téléphone (JPG, PNG, TIFF)
en allemand, en français, en italien et en anglais. Le résultat est un nouveau PDF composé
uniquement d'images des pages : sans couche de texte, métadonnées, pièces jointes ni
commentaires dans lesquels des données caviardées pourraient subsister.

## Vos données restent chez vous

L'application fonctionne entièrement sur votre ordinateur. Après l'installation unique, elle
n'a plus besoin d'Internet, et aucun document ne quitte l'ordinateur.

Si vous le souhaitez, vous pouvez confier certaines étapes à **SwissDocAI Cloud**, avec un
traitement en Suisse ([swissdocai.ch](https://swissdocai.ch)) – par exemple si votre
ordinateur n'a pas de carte graphique adaptée. C'est une option sous « Paramètres » →
« Général », désactivée par défaut :

- **Lecture des pages :** les images des pages sont lues en Suisse.
- **Recherche des données personnelles :** seul le texte des pages est vérifié en Suisse,
  aucune image.

Chaque service coûte une page de crédit par page de document, les deux ensemble deux. Il vous
faut pour cela un jeton API de la console SwissDocAI.

## Installation

Téléchargez le fichier d'installation pour votre système sur la
**[page des versions](https://github.com/swissdocai/SwissDocAI-Redact/releases/latest)** :

| Système | Fichier |
|---|---|
| Windows 10 et 11 | `…_x64-setup.exe` – s'installe sans droits d'administrateur |
| Mac avec puce Apple (M1 ou plus récent), à partir de macOS 12 | `….dmg` |
| Linux | `….AppImage` ou `….deb` (Debian, Ubuntu) |

Les fichiers d'installation ne sont pas encore signés. Votre système affiche donc un
avertissement à la première ouverture :

- **Windows :** si « Windows a protégé votre ordinateur » s'affiche, cliquez sur
  « Informations complémentaires », puis sur « Exécuter quand même ».
- **macOS :** ouvrez l'application une fois. Si elle est bloquée, allez dans Réglages Système →
  Confidentialité et sécurité et cliquez sur « Ouvrir quand même ».
- **Linux :** marquez le fichier AppImage comme exécutable dans ses propriétés
  (ou `chmod +x` dans le terminal).

## Premier démarrage

Au premier démarrage, l'application s'installe une fois pour toutes : elle télécharge ses
modèles, environ 7 Go – elle en indique la taille exacte. Le téléchargement peut être
interrompu et repris plus tard. Prévoyez environ 8 Go d'espace libre. Les modèles se trouvent
dans `%LOCALAPPDATA%\swissdocai` (Windows) ou `~/.cache/swissdocai` (macOS, Linux) ; lors de
la désinstallation sous Windows, vous pouvez les supprimer en même temps.

L'application détecte elle-même si une carte graphique peut accélérer le travail (NVIDIA, AMD,
Apple, Intel) et indique ce qu'elle utilise. À part le pilote graphique habituel, vous n'avez
rien à installer. Sans carte graphique adaptée, tout tourne sur le processeur – cela
fonctionne, mais plus lentement.

## Caviarder un document

1. Glissez des PDF ou des images dans la fenêtre, ou cliquez sur « Choisir des fichiers ».
2. Cliquez sur « Rechercher les données personnelles » – pour plusieurs documents, sur
   « Tout analyser ».
3. Vérifiez les données personnelles trouvées. Si vous décochez une valeur, elle reste lisible
   partout. Sur la page, cliquez sur un mot pour le caviarder ou le laisser lisible, et tirez un
   cadre pour un caviardage manuel. Examinez avec une attention particulière les passages
   sous « À vérifier ».
4. Cliquez sur « Enregistrer le PDF caviardé ». L'application relit le résultat et signale si
   une valeur caviardée est encore lisible.

Sous « Paramètres » → « Que caviarder », vous adaptez les listes : ce qui est caviardé et ce
qui reste lisible. Sous « Autres indications », vous pouvez décrire les documents, par exemple :
« Ce sont des rapports médicaux. Ne pas caviarder les noms des médecins. » La langue de
l'application se choisit sous « Général ».

> [!IMPORTANT]
> Le caviardage automatique est très fiable, mais pas infaillible : lors de nos tests avec des factures
> et des courriers de banques, d’autorités, de cabinets médicaux, d’assurances et d’employeurs, 99,5 %
> des données personnelles ont été caviardées ; en externalisant le traitement à SwissDocAI,
> nous avons atteint 99,9 %. Veuillez vérifier chaque document avant de le transmettre.

## Comment ça marche

Chaque page est lue deux fois, et les deux lectures se contrôlent mutuellement :

1. Une reconnaissance de texte (OCR) trouve chaque mot et sa position sur la page. Elle utilise
   un modèle entraîné spécialement pour les documents suisses, qui lit aussi les trémas et les
   accents.
2. Un modèle de lecture retranscrit toute la page, afin de conserver le contexte : à qui est ce
   nom ?
3. Un modèle de langage cherche les données personnelles dans ce texte, selon vos listes. Des
   règles avec chiffres de contrôle ajoutent les numéros AVS, les IBAN et d'autres numéros.
4. Chaque élément trouvé est localisé sur la page et recouvert de noir. Pour finir, le résultat
   est relu.

Tous les détails – résultats de mesure, matériel, ligne de commande, développement – se
trouvent dans la [documentation technique](docs/technical.md) (en anglais).

## Démarrer sans installateur

Pour les développeuses et développeurs, et pour toutes les personnes qui souhaitent lancer
l'application à partir du code source. Il vous faut :

- [Git](https://git-scm.com) avec [Git LFS](https://git-lfs.com) (inclus dans Git pour Windows)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) pour Python
- [Node.js](https://nodejs.org) (LTS)
- [Rust](https://rustup.rs) avec les [prérequis de Tauri](https://v2.tauri.app/start/prerequisites/)
  pour votre système (Windows : Microsoft C++ Build Tools ; macOS : Xcode Command Line Tools ;
  Linux : les paquets WebKitGTK)

```bash
git clone https://github.com/swissdocai/SwissDocAI-Redact.git
cd SwissDocAI-Redact
```

Ensuite, sous Windows (PowerShell) :

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

Sous macOS et Linux :

```bash
bash start.sh
```

Le script installe ce qui manque – l'environnement Python, les paquets Node.js et le modèle de
reconnaissance de texte depuis Git LFS – et lance l'application. Le premier démarrage prend
quelques minutes, car l'application est compilée ; ensuite, c'est rapide.

Uniquement la ligne de commande, sans Node.js ni Rust :

```bash
uv venv
uv pip install -r pyproject.toml
uv run --no-sync python -m swissdocai redact lettre.pdf
```

Le PDF caviardé est enregistré à côté, sous le nom `lettre.redacted.pdf`. Les autres commandes
figurent dans la [documentation technique](docs/technical.md#command-line).

## Contact

Des questions, des remarques ou un problème avec l'application ? Écrivez-nous à
[support@swissdocai.ch](mailto:support@swissdocai.ch).

## Licence

SwissDocAI Redact est un logiciel open source sous [licence Apache 2.0](LICENSE).
Copyright 2026 SwissDocAI.

---

<p align="center">SwissDocAI · <a href="https://swissdocai.ch">swissdocai.ch</a> · <a href="mailto:support@swissdocai.ch">support@swissdocai.ch</a></p>
