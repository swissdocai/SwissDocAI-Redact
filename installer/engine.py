"""The desktop app's engine as the installer ships it (frozen by PyInstaller, see
swissdocai-engine.spec): what ``python -m swissdocai.app.api`` runs in development."""

from swissdocai.app.api import main

if __name__ == "__main__":
    main()
