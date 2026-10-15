; SwissDocAI Redact: additions to Tauri's NSIS installer (bundle > windows > nsis > installerHooks).

; End an engine left running from this installation, which would keep its files locked, and the
; model servers it started (they run from the model cache). Normally the engine ends with the app.
!macro SWISSDOCAI_STOP_ENGINE
  nsExec::Exec `powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "Get-Process swissdocai-engine, llama-server -ErrorAction SilentlyContinue | Where-Object { $$_.Path -like '$INSTDIR\*' -or $$_.Path -like '$LOCALAPPDATA\swissdocai\*' } | Stop-Process -Force"`
  Pop $0
!macroend

!macro NSIS_HOOK_PREINSTALL
  !insertmacro SWISSDOCAI_STOP_ENGINE
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  !insertmacro SWISSDOCAI_STOP_ENGINE
!macroend

; "Delete the application data" ticked (never on updates or silent uninstalls): also the
; downloaded models (several GB), settings and logs in %LOCALAPPDATA%\swissdocai, which Tauri's
; own clean-up does not know about.
!macro NSIS_HOOK_POSTUNINSTALL
  ${If} $DeleteAppDataCheckboxState = 1
  ${AndIf} $UpdateMode <> 1
    SetShellVarContext current
    RmDir /r "$LOCALAPPDATA\swissdocai"
  ${EndIf}
!macroend
