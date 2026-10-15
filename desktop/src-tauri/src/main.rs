// No console window next to the app in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    swissdocai_redact_lib::run()
}
