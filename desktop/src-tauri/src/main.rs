// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use tauri::{AppHandle, Manager, State, WindowEvent};

struct AppState {
    backend_port: Mutex<Option<u16>>,
    backend_process: Mutex<Option<Child>>,
}

#[tauri::command]
fn get_backend_port(state: State<AppState>) -> Result<u16, String> {
    let lock = state.backend_port.lock().map_err(|e| e.to_string())?;
    match *lock {
        Some(port) => Ok(port),
        None => Err("Cổng backend chưa sẵn sàng".into()),
    }
}

fn spawn_backend() -> (Child, u16) {
    // Ưu tiên chạy binary đóng gói độc lập trong portable folder: ./backend/backend-server.exe
    // Nếu không có (môi trường dev), fallback chạy python -m backend.api.main
    let mut cmd = if std::path::Path::new("backend/backend-server.exe").exists() {
        Command::new("backend/backend-server.exe")
    } else if std::path::Path::new("../backend/backend-server.exe").exists() {
        Command::new("../backend/backend-server.exe")
    } else {
        let mut c = Command::new("python");
        c.args(["-m", "backend.api.main"]);
        c
    };

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::inherit());

    let mut child = cmd.spawn().expect("Không thể khởi chạy backend process");
    let stdout = child.stdout.take().expect("Không thể mở stdout của backend");
    let reader = BufReader::new(stdout);

    let mut port = 8000;
    for line in reader.lines() {
        if let Ok(l) = line {
            println!("[Backend Output] {}", l);
            if l.starts_with("PORT:") {
                let port_str = l.trim_start_matches("PORT:").trim();
                if let Ok(p) = port_str.parse::<u16>() {
                    port = p;
                    break;
                }
            }
        }
    }

    (child, port)
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            backend_port: Mutex::new(None),
            backend_process: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![get_backend_port])
        .setup(|app| {
            let app_handle = app.handle().clone();

            thread::spawn(move || {
                let (child, port) = spawn_backend();
                println!("Đã khởi động backend thành công trên cổng: {}", port);

                // Lưu vào AppState
                let state: State<AppState> = app_handle.state();
                if let Ok(mut lock_port) = state.backend_port.lock() {
                    *lock_port = Some(port);
                }
                if let Ok(mut lock_proc) = state.backend_process.lock() {
                    *lock_proc = Some(child);
                }

                // Health check đợi backend sẵn sàng
                let client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(2))
                    .build()
                    .unwrap();

                let health_url = format!("http://127.0.0.1:{}/api/health", port);
                for _ in 0..30 {
                    if let Ok(resp) = client.get(&health_url).send() {
                        if resp.status().is_success() {
                            println!("Backend health check OK!");
                            break;
                        }
                    }
                    thread::sleep(Duration::from_millis(500));
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                // Diệt tiến trình con sạch sẽ khi đóng app, giải phóng VRAM GPU
                let state: State<AppState> = window.state();
                if let Ok(mut lock_proc) = state.backend_process.lock() {
                    if let Some(mut child) = lock_proc.take() {
                        let _ = child.kill();
                        println!("Đã tắt tiến trình con backend thành công.");
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("Lỗi khi khởi chạy ứng dụng Tauri");
}
