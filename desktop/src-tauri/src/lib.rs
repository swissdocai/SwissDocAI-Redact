//! The app shell: a native window around the React UI, and the Python engine
//! (`swissdocai.app.api`) started in the background on a free local port with a
//! fresh token. The window asks for both (`engine_info`) and talks to the engine
//! directly; on exit the engine is asked to stop its model servers, then ended.
//! The engine also watches this process (`--parent-pid`): if the app is killed
//! or crashes, the engine stops itself and its model servers.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{Manager, RunEvent};

#[derive(Clone, Serialize)]
struct EngineInfo {
    url: String,
    token: String,
}

#[derive(Default)]
struct Engine {
    info: Mutex<Option<EngineInfo>>,
    error: Mutex<Option<String>>,
    child: Mutex<Option<Child>>,
}

#[tauri::command]
fn engine_info(engine: tauri::State<Arc<Engine>>) -> Option<EngineInfo> {
    engine.info.lock().unwrap().clone()
}

#[tauri::command]
fn engine_error(engine: tauri::State<Arc<Engine>>) -> Option<String> {
    engine.error.lock().unwrap().clone()
}

/// The engine to start: `SWISSDOCAI_ENGINE` ("program|arg|arg", for development, e.g.
/// `.venv\Scripts\python.exe|-m|swissdocai.app.api`), else `swissdocai-engine` shipped with the app.
fn engine_command(app: &tauri::AppHandle) -> Result<(PathBuf, Vec<String>), String> {
    if let Ok(spec) = std::env::var("SWISSDOCAI_ENGINE") {
        let mut parts = spec.split('|').map(String::from);
        let program = parts.next().filter(|p| !p.is_empty()).ok_or("SWISSDOCAI_ENGINE is empty")?;
        return Ok((PathBuf::from(program), parts.collect()));
    }
    let name = if cfg!(windows) { "swissdocai-engine.exe" } else { "swissdocai-engine" };
    let mut candidates = Vec::new();
    if let Some(dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(PathBuf::from)) {
        candidates.push(dir.join(name));
        candidates.push(dir.join("engine").join(name));
    }
    if let Ok(resources) = app.path().resource_dir() {
        candidates.push(resources.join("engine").join(name));
    }
    candidates
        .into_iter()
        .find(|path| path.exists())
        .map(|path| (path, Vec::new()))
        .ok_or_else(|| format!("{name} was not found next to the app"))
}

fn start_engine(app: tauri::AppHandle, engine: Arc<Engine>) {
    std::thread::spawn(move || {
        let token = uuid::Uuid::new_v4().simple().to_string();
        let result = (|| -> Result<(), String> {
            let (program, mut args) = engine_command(&app)?;
            args.extend(["--port".into(), "0".into(), "--token".into(), token.clone()]);
            args.extend(["--parent-pid".into(), std::process::id().to_string()]);
            let mut command = Command::new(&program);
            command.args(&args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
            }
            let mut child = command.spawn().map_err(|e| format!("{}: {e}", program.display()))?;
            let stdout = child.stdout.take().ok_or("the engine has no output")?;
            *engine.child.lock().unwrap() = Some(child);
            // read the announcement, then keep draining so the engine never blocks on a full pipe
            for line in BufReader::new(stdout).lines() {
                let line = line.map_err(|e| e.to_string())?;
                if let Some(port) = line.trim().strip_prefix("SWISSDOCAI_ENGINE_READY ") {
                    *engine.info.lock().unwrap() =
                        Some(EngineInfo { url: format!("http://127.0.0.1:{port}"), token: token.clone() });
                }
            }
            Err("the engine stopped".into())
        })();
        if let Err(error) = result {
            *engine.error.lock().unwrap() = Some(error);
        }
    });
}

/// Ask the engine to stop its model servers and exit; end it if it does not within a few seconds.
fn stop_engine(engine: &Engine) {
    if let Some(info) = engine.info.lock().unwrap().clone() {
        let address = info.url.trim_start_matches("http://").to_string();
        if let Ok(mut stream) = TcpStream::connect(&address) {
            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
            let _ = write!(
                stream,
                "POST /api/shutdown HTTP/1.1\r\nHost: {address}\r\nX-Token: {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                info.token
            );
            let _ = stream.read(&mut [0u8; 512]);
        }
    }
    if let Some(mut child) = engine.child.lock().unwrap().take() {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = child.kill();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let engine = Arc::new(Engine::default());
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(engine.clone())
        .invoke_handler(tauri::generate_handler![engine_info, engine_error])
        .setup({
            let engine = engine.clone();
            move |app| {
                start_engine(app.handle().clone(), engine);
                Ok(())
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building the app");
    app.run(move |_, event| {
        if let RunEvent::Exit = event {
            stop_engine(&engine);
        }
    });
}
