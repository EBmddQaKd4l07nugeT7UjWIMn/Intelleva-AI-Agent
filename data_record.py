import tkinter as tk
from tkinter import ttk, messagebox
import psutil
import time
import threading
import os
from pathlib import Path
import json
from mss import mss
from mss.tools import to_png
from pynput import keyboard, mouse
import queue
import math
import platform
import traceback
GPU_LIB = None
try:
    import pynvml
    GPU_LIB = "pynvml"
    try:
        print("Initializing pynvml...")
        pynvml.nvmlInit()
        print("pynvml initialized successfully.")
        try:
             device_count = pynvml.nvmlDeviceGetCount()
             if device_count == 0:
                 print("WARN: pynvml initialized, but no NVIDIA devices found.")
                 GPU_LIB = None
             else:
                 print(f"Found {device_count} NVIDIA device(s). Using pynvml for GPU monitoring.")
        except pynvml.NVMLError_LibraryNotFound:
             print("WARN: NVML Library found but potentially not functional (driver issue?). Disabling pynvml.")
             GPU_LIB = None
             try: pynvml.nvmlShutdown()
             except: pass
        except pynvml.NVMLError as e:
             print(f"WARN: NVML Initialization check failed: {e}. Disabling pynvml.")
             GPU_LIB = None
             try: pynvml.nvmlShutdown()
             except: pass
        except Exception as e:
             print(f"WARN: Unexpected error during NVML init check: {e}. Disabling pynvml.")
             GPU_LIB = None
             try: pynvml.nvmlShutdown()
             except: pass
    except pynvml.NVMLError as e:
        GPU_LIB = None
        print(f"WARNING: Error initializing pynvml: {e}. GPU information will be unavailable.")
    except ImportError:
        GPU_LIB = None
        print("pynvml not found, checking for gpustat...")
    except Exception as e:
        GPU_LIB = None
        print(f"WARNING: An unexpected error occurred during pynvml import/init: {e}. GPU info unavailable.")
    if GPU_LIB is None:
        print("Attempting to use gpustat as fallback...")
        try:
            import gpustat
            gpustat.new_query()
            GPU_LIB = "gpustat"
            print("Using gpustat for GPU monitoring.")
        except ImportError:
            GPU_LIB = None
            print("WARNING: Cannot import gpustat either. GPU information will be unavailable.")
        except Exception as e:
            GPU_LIB = None
            print(f"WARNING: gpustat found but failed during initial query: {e}. GPU info unavailable.")
except ImportError:
    GPU_LIB = None
    print("WARNING: Cannot import pynvml or gpustat. GPU information will be unavailable.")
except Exception as e:
    GPU_LIB = None
    print(f"WARNING: An unexpected error occurred during GPU library check: {e}. GPU info unavailable.")
SCRIPT_DIR = Path(__file__).parent.resolve()
EXPERIENCE_POOL_DIR = SCRIPT_DIR / "experience_pool"
SCREENSHOT_DIR = EXPERIENCE_POOL_DIR / "screenshots"
KEYBOARD_LOG_FILE = EXPERIENCE_POOL_DIR / "keyboard_log.jsonl"
MOUSE_LOG_FILE = EXPERIENCE_POOL_DIR / "mouse_log.jsonl"
RESULTS_LOG_FILE = EXPERIENCE_POOL_DIR / "results_log.jsonl"
gui_update_queue = queue.Queue()
data_collection_paused = threading.Event()
data_collection_paused.clear()
def get_gpu_usage():
    gpu_percent = "N/A"
    vram_percent = "N/A"
    global GPU_LIB
    if not GPU_LIB:
        return gpu_percent, vram_percent
    try:
        if GPU_LIB == "gpustat":
            stats = gpustat.new_query()
            if stats:
                gpu_info = stats[0]
                gpu_percent = gpu_info.utilization if gpu_info.utilization is not None else 0.0
                vram_total = gpu_info.memory_total
                vram_used = gpu_info.memory_used
                if vram_total and vram_total > 0:
                    vram_percent = round((vram_used / vram_total) * 100, 1)
                else:
                    vram_percent = 0.0
            else:
                gpu_percent = "No Device"
                vram_percent = "No Device"
        elif GPU_LIB == "pynvml":
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_percent = float(utilization.gpu)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if mem_info.total > 0:
                    vram_percent = round((mem_info.used / mem_info.total) * 100, 1)
                else:
                    vram_percent = 0.0
            else:
                 gpu_percent = "No Device"
                 vram_percent = "No Device"
    except pynvml.NVMLError as e:
        if not getattr(get_gpu_usage, "nvml_error_logged", False):
             print(f"ERROR getting GPU info via pynvml: {e}. Check NVIDIA drivers/runtime. Disabling NVML for this session.")
             get_gpu_usage.nvml_error_logged = True
             GPU_LIB = None
             try: pynvml.nvmlShutdown()
             except: pass
        gpu_percent = "Error NVML"
        vram_percent = "Error NVML"
    except Exception as e:
        library_name = GPU_LIB or "unavailable library"
        if not getattr(get_gpu_usage, f"{library_name}_error_logged", False):
            print(f"ERROR getting GPU info (using {library_name}): {e}. Will not log further generic GPU errors for this library.")
            setattr(get_gpu_usage, f"{library_name}_error_logged", True)
        gpu_percent = "Error"
        vram_percent = "Error"
    return gpu_percent, vram_percent
def ensure_experience_pool():
    print("Checking experience pool structure...")
    try:
        if not EXPERIENCE_POOL_DIR.exists():
            print(f"Creating directory: {EXPERIENCE_POOL_DIR}")
            EXPERIENCE_POOL_DIR.mkdir(parents=True, exist_ok=True)
        if not SCREENSHOT_DIR.exists():
            print(f"Creating directory: {SCREENSHOT_DIR}")
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        for fpath in [KEYBOARD_LOG_FILE, MOUSE_LOG_FILE, RESULTS_LOG_FILE]:
            if not fpath.exists():
                print(f"Creating file: {fpath}")
                fpath.touch()
        print("Experience pool structure checked/created successfully.")
    except OSError as e:
        print(f"ERROR: Could not create experience pool directories/files: {e}")
        raise
def count_log_lines(filepath):
    count = 0
    if not filepath.exists(): return 0
    try:
        with filepath.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception as e:
        print(f"ERROR reading line count for {filepath}: {e}")
    return count
def count_screenshots():
    count = 0
    if not SCREENSHOT_DIR.exists(): return 0
    try:
        count = sum(1 for f in SCREENSHOT_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.png')
    except Exception as e:
        print(f"ERROR counting screenshots in {SCREENSHOT_DIR}: {e}")
    return count
log_lock = threading.Lock()
def write_log(filepath, data):
    try:
        with log_lock:
            with filepath.open('a', encoding='utf-8') as f:
                json.dump(data, f, separators=(',', ':'))
                f.write('\n')
        return True
    except Exception as e:
        print(f"ERROR writing log to {filepath}: {e}")
        traceback.print_exc()
        return False
class MonitoringWindow:
    BASE_WIDTH = 450
    BASE_HEIGHT = 350
    BASE_FONT_SIZE = 10
    MIN_FONT_SIZE = 7
    MAX_FONT_SIZE = 20
    def __init__(self, root):
        self.root = root
        self.root.title("System Status & Data Monitor")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        win_width = max(self.BASE_WIDTH, min(int(screen_width * 0.25), 700))
        win_height = max(self.BASE_HEIGHT, min(int(screen_height * 0.35), 600))
        win_x = screen_width - win_width - 30
        win_y = 30
        self.root.geometry(f"{win_width}x{win_height}+{win_x}+{win_y}")
        self.root.minsize(380, 300)
        self.current_width = win_width
        self.current_height = win_height
        self.frame = ttk.Frame(root, padding="10")
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.stat_labels = {}
        self.title_labels = []
        stats_to_display = [
            ("CPU Usage:", "cpu_usage"),
            ("RAM Usage:", "ram_usage"),
            ("GPU Usage:", "gpu_usage"),
            ("VRAM Usage:", "vram_usage"),
            ("Screenshot Freq (Hz):", "screenshot_freq"),
            ("--- Data Counts ---", None),
            ("Screenshots:", "screenshot_count"),
            ("Keyboard Events:", "keyboard_count"),
            ("Mouse Events:", "mouse_count"),
            ("Game Results:", "result_count")
        ]
        self.frame.grid_columnconfigure(0, weight=1, uniform="labelcol")
        self.frame.grid_columnconfigure(1, weight=3, uniform="valuecol")
        initial_scale = min(win_width / self.BASE_WIDTH, win_height / self.BASE_HEIGHT)
        initial_font_size = max(self.MIN_FONT_SIZE, min(self.MAX_FONT_SIZE, int(self.BASE_FONT_SIZE * initial_scale)))
        self.initial_font = ('Segoe UI', initial_font_size)
        self.initial_title_font = ('Segoe UI', initial_font_size, 'bold')
        self._last_applied_font_size = initial_font_size
        for i, (text, key) in enumerate(stats_to_display):
            self.frame.grid_rowconfigure(i, weight=1)
            if key:
                label_text = ttk.Label(self.frame, text=text, anchor="w", font=self.initial_font)
                label_text.grid(row=i, column=0, sticky="nsew", padx=(0, 5), pady=2)
                label_value = ttk.Label(self.frame, text="N/A", anchor="w", relief=tk.SUNKEN, padding=(5,2), font=self.initial_font)
                label_value.grid(row=i, column=1, sticky="nsew", padx=(5, 0), pady=2)
                self.stat_labels[key] = label_value
                self.title_labels.append(label_text)
                self.title_labels.append(label_value)
            else:
                sep_label = ttk.Label(self.frame, text=text, anchor="w", font=self.initial_title_font)
                sep_label.grid(row=i, column=0, columnspan=2, sticky="ew", padx=0, pady=(10, 2))
                self.title_labels.append(sep_label)
        self.frame.bind("<Configure>", self._on_resize)
        self.check_queue()
    def update_stats(self, stats_dict):
        for key, value in stats_dict.items():
            if key in self.stat_labels:
                formatted_value = "N/A"
                try:
                    if isinstance(value, (int, float)):
                        if key.endswith("_usage") or key == "vram_usage":
                            formatted_value = f"{value:.1f}%"
                        elif key.endswith("_freq"):
                             formatted_value = f"{value:.1f}"
                        elif key.endswith("_count"):
                            formatted_value = f"{int(value):,}"
                        else:
                            formatted_value = str(value)
                    elif isinstance(value, str) and (value.startswith("Error") or value == "No Device"):
                        formatted_value = value
                    elif value is None:
                         formatted_value = "N/A"
                    else:
                        formatted_value = str(value)
                except (ValueError, TypeError) as e:
                     print(f"Warning: Could not format value for {key}: {value} ({type(value)}) - Error: {e}")
                     formatted_value = "Error fmt"
                self.stat_labels[key].config(text=formatted_value)
    def check_queue(self):
        try:
            while True:
                update_data = gui_update_queue.get_nowait()
                self.update_stats(update_data)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error processing GUI update queue: {e}")
            traceback.print_exc()
        finally:
            self.root.after(150, self.check_queue)
    def _on_resize(self, event):
        new_width = event.width
        new_height = event.height
        width_scale = new_width / self.BASE_WIDTH
        height_scale = new_height / self.BASE_HEIGHT
        scale = min(width_scale, height_scale)
        new_size = max(self.MIN_FONT_SIZE, min(self.MAX_FONT_SIZE, int(self.BASE_FONT_SIZE * scale)))
        if new_size == self._last_applied_font_size:
            return
        new_font_tuple = ('Segoe UI', new_size)
        new_bold_font_tuple = ('Segoe UI', new_size, 'bold')
        for label in self.title_labels:
            try:
                current_font_config = label.cget("font")
                is_bold = 'bold' in str(current_font_config).lower()
                if is_bold:
                    label.config(font=new_bold_font_tuple)
                else:
                    label.config(font=new_font_tuple)
            except tk.TclError:
                pass
            except Exception as e:
                 print(f"Error updating font for label {label}: {e}")
        self._last_applied_font_size = new_size
        self.current_width = new_width
        self.current_height = new_height
class ResultWindow:
    _instance = None
    def __new__(cls, parent_root):
        if cls._instance is None or not cls._instance.window.winfo_exists():
            cls._instance = super(ResultWindow, cls).__new__(cls)
            return cls._instance
        else:
            print("Result window already exists. Bringing to front.")
            try:
                cls._instance.window.lift()
                cls._instance.window.focus_force()
                cls._instance.window.grab_set()
            except tk.TclError as e:
                 print(f"Error focusing existing result window (may be closing): {e}")
                 cls._instance = None
            return cls._instance
    def __init__(self, parent_root):
        if hasattr(self, 'initialized') and self.initialized:
             return
        self.parent_root = parent_root
        self.result = None
        self.selected_button_var = tk.StringVar(value="")
        if not data_collection_paused.is_set():
             print("Pausing data collection for result selection...")
             data_collection_paused.set()
        self.window = tk.Toplevel(parent_root)
        self.window.title("Select Game Result")
        self.window.transient(parent_root)
        self.window.protocol("WM_DELETE_WINDOW", self._handle_close_attempt)
        self.window.resizable(False, False)
        button_width_chars = 10
        button_internal_padx = 15
        button_internal_pady = 5
        button_external_padx = 10
        frame_padx = 20
        frame_pady = 15
        button_font_size = 11
        est_btn_width_pixels = button_width_chars * button_font_size + 2 * button_internal_padx
        win_width = (est_btn_width_pixels * 3) + (button_external_padx * 2) + (frame_padx * 2)
        win_width = max(win_width, 450)
        est_btn_height_pixels = 2 * button_internal_pady + button_font_size * 2
        win_height = (est_btn_height_pixels * 2) + frame_pady * 3
        win_height = max(win_height, 180)
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        win_width = min(win_width, int(screen_width * 0.6))
        win_height = min(win_height, int(screen_height * 0.3))
        win_x = (screen_width // 2) - (win_width // 2)
        win_y = (screen_height // 2) - (win_height // 2)
        self.window.geometry(f"{win_width}x{win_height}+{win_x}+{win_y}")
        top_frame = tk.Frame(self.window)
        top_frame.pack(expand=True, fill=tk.X, padx=frame_padx, pady=(frame_pady, frame_pady // 2))
        bottom_frame = tk.Frame(self.window)
        bottom_frame.pack(expand=True, fill=tk.X, padx=frame_padx, pady=(frame_pady // 2, frame_pady))
        top_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="top_buttons")
        bottom_frame.grid_columnconfigure((0, 1), weight=1, uniform="bottom_buttons")
        top_frame.grid_rowconfigure(0, weight=1)
        bottom_frame.grid_rowconfigure(0, weight=1)
        button_font = ('Segoe UI', button_font_size, 'bold')
        radio_style = ttk.Style()
        radio_style_name = 'Result.TRadiobutton'
        radio_style.configure(radio_style_name,
                              font=button_font,
                              padding=(button_internal_padx, button_internal_pady),
                              anchor=tk.CENTER,
                              indicatoron=0,
                              relief=tk.RAISED,
                              borderwidth=2)
        results = [("LOSE", "lose", "red"), ("DRAW", "draw", "green"), ("WIN", "win", "blue")]
        self.result_buttons = {}
        for i, (text, value, color) in enumerate(results):
            rb = ttk.Radiobutton(top_frame, text=text, variable=self.selected_button_var, value=value,
                                 style=radio_style_name,
                                 command=self._on_result_select)
            rb.grid(row=0, column=i, sticky="nsew", padx=button_external_padx // 2, pady=5)
            self.result_buttons[value] = rb
            try:
                s = ttk.Style()
                dynamic_style_name = f'{text}.Result.TRadiobutton'
                s.configure(dynamic_style_name,
                            background=color,
                            foreground='white',
                            font=button_font,
                            padding=(button_internal_padx, button_internal_pady),
                            anchor=tk.CENTER,
                            relief=tk.RAISED,
                            borderwidth=2)
                s.map(dynamic_style_name,
                      background=[('active', color), ('selected', color), ('!selected', color)],
                      foreground=[('active', 'white'), ('selected', 'white'), ('!selected', 'white')])
                rb.config(style=dynamic_style_name)
            except Exception as style_e:
                print(f"Warning: Could not apply custom style colors for {text} button: {style_e}")
                try:
                     rb.config(background=color, foreground='white')
                except:
                     pass
        common_options = {
            "font": button_font,
            "relief": tk.RAISED,
            "borderwidth": 2,
            "width": button_width_chars,
            "padx": button_internal_padx,
            "pady": button_internal_pady,
        }
        skip_btn = tk.Button(bottom_frame, text="SKIP", command=self.skip,
                             bg="black", fg="white",
                             activebackground="gray20", activeforeground="white",
                             **common_options)
        skip_btn.grid(row=0, column=0, sticky="nsew", padx=button_external_padx // 2, pady=5)
        confirm_btn = tk.Button(bottom_frame, text="CONFIRM", command=self.confirm,
                                bg="white", fg="black",
                                activebackground="gray90", activeforeground="black",
                                **common_options)
        confirm_btn.grid(row=0, column=1, sticky="nsew", padx=button_external_padx // 2, pady=5)
        self.window.lift()
        self.window.focus_force()
        self.window.grab_set()
        self.initialized = True
    def _on_result_select(self):
        self.result = self.selected_button_var.get()
        print(f"Result selected: {self.result}")
        for value, button in self.result_buttons.items():
             try:
                 if value == self.result:
                     button.state(['pressed'])
                 else:
                     button.state(['!pressed'])
             except tk.TclError: pass
    def skip(self):
        print("User chose to SKIP result logging.")
        self.result = "skip"
        self.close_window()
    def confirm(self):
        selected_result = self.selected_button_var.get()
        if selected_result and selected_result in ["lose", "draw", "win"]:
            confirm_time_ns = time.time_ns()
            print(f"User CONFIRMED result: {selected_result} at {confirm_time_ns}")
            data = {
                "timestamp_ns": confirm_time_ns,
                "result": selected_result
            }
            if write_log(RESULTS_LOG_FILE, data):
                gui_update_queue.put({"result_count": count_log_lines(RESULTS_LOG_FILE)})
            self.close_window()
        else:
            messagebox.showwarning("Selection Required",
                                   "Please select 'LOSE', 'DRAW', or 'WIN' before confirming.",
                                   parent=self.window)
            print("Confirm clicked without a valid result selection.")
            self.window.grab_set()
            self.window.focus_force()
    def _handle_close_attempt(self):
        print("Result window close attempted - interpreting as SKIP.")
        self.skip()
    def close_window(self):
        if data_collection_paused.is_set():
            print("Resuming data collection.")
            data_collection_paused.clear()
        if self.window and self.window.winfo_exists():
            self.window.grab_release()
            self.window.destroy()
        ResultWindow._instance = None
        self.initialized = False
class BackgroundTaskManager:
    def __init__(self, root_tk):
        self.root = root_tk
        self.stop_event = threading.Event()
        self.threads = []
        self.base_screenshot_delay_sec = 1.0
        self.current_screenshot_delay_sec = self.base_screenshot_delay_sec
        self.min_screenshot_delay_sec = 0.2
        self.max_screenshot_delay_sec = 5.0
        self.key_press_times = {}
        self.last_ctrl_press_time_ns = 0
        self.last_alt_press_time_ns = 0
        self.ctrl_alt_threshold_ns = 950_000_000
        self.mouse_press_info = {}
        self.long_press_threshold_ns = 500_000_000
        self.drag_threshold_pixels = 10
        self.keyboard_listener = None
        self.mouse_listener = None
        self.sct_instance = None
    def start_all(self):
        self.stop_event.clear()
        data_collection_paused.clear()
        if hasattr(get_gpu_usage, "nvml_error_logged"): delattr(get_gpu_usage, "nvml_error_logged")
        if GPU_LIB and hasattr(get_gpu_usage, f"{GPU_LIB}_error_logged"):
             delattr(get_gpu_usage, f"{GPU_LIB}_error_logged")
        thread_targets = {
            "SystemMonitor": self.monitor_system,
            "ScreenshotWorker": self.screenshot_worker,
            "KeyboardListener": self.keyboard_listener_worker,
            "MouseListener": self.mouse_listener_worker
        }
        self.threads = []
        for name, target in thread_targets.items():
            thread = threading.Thread(target=target, name=name, daemon=True)
            self.threads.append(thread)
            thread.start()
        print(f"Started {len(self.threads)} background tasks.")
    def stop_all(self):
        print("Stopping background tasks...")
        self.stop_event.set()
        if self.keyboard_listener and self.keyboard_listener.is_alive():
             print("Requesting keyboard listener stop...")
             try:
                 pass
             except Exception as e: print(f" Minor error requesting kbd listener stop: {e}")
        if self.mouse_listener and self.mouse_listener.is_alive():
             print("Requesting mouse listener stop...")
             try:
                 pass
             except Exception as e: print(f" Minor error requesting mouse listener stop: {e}")
        print("Waiting for threads to join...")
        active_threads_before_join = [t.name for t in self.threads if t.is_alive()]
        if active_threads_before_join:
             print(f"  Active threads: {', '.join(active_threads_before_join)}")
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=3.0)
                if thread.is_alive():
                      print(f"  WARNING: Thread {thread.name} did not stop gracefully after timeout.")
        self.threads = []
        print("All background threads processed.")
        if GPU_LIB == "pynvml":
            try:
                print("Attempting to shut down pynvml...")
                pynvml.nvmlShutdown()
                print("pynvml shutdown successful.")
            except pynvml.NVMLError as e:
                 if e.value != pynvml.NVML_ERROR_UNINITIALIZED and e.value != pynvml.NVML_ERROR_NOT_FOUND:
                      print(f"WARN: Error shutting down pynvml: {e}")
            except Exception as e:
                 print(f"WARN: Unexpected error during pynvml shutdown: {e}")
        print("Background tasks stopped.")
    def monitor_system(self):
        print("System monitor thread started.")
        while not self.stop_event.is_set():
            loop_start_time = time.monotonic()
            try:
                cpu_usage = psutil.cpu_percent()
                ram = psutil.virtual_memory()
                ram_usage = ram.percent
                gpu_usage, vram_usage = get_gpu_usage()
                load_factor = 0.0
                if isinstance(cpu_usage, (int, float)):
                    load_factor += max(0.0, min(1.0, (cpu_usage - 70) / 30.0))
                if isinstance(ram_usage, (int, float)):
                    load_factor += max(0.0, min(1.0, (ram_usage - 75) / 25.0))
                if isinstance(gpu_usage, (int, float)):
                    load_factor += max(0.0, min(1.0, (gpu_usage - 75) / 25.0)) * 1.2
                if isinstance(vram_usage, (int, float)):
                    load_factor += max(0.0, min(1.0, (vram_usage - 80) / 20.0)) * 0.8
                delay_multiplier = 1.0 + load_factor
                target_delay = self.base_screenshot_delay_sec * delay_multiplier
                self.current_screenshot_delay_sec = max(self.min_screenshot_delay_sec, min(self.max_screenshot_delay_sec, target_delay))
                current_freq = 1.0 / self.current_screenshot_delay_sec if self.current_screenshot_delay_sec > 0 else float('inf')
                stats_update = {
                    "cpu_usage": cpu_usage,
                    "ram_usage": ram_usage,
                    "gpu_usage": gpu_usage,
                    "vram_usage": vram_usage,
                    "screenshot_freq": current_freq,
                }
                gui_update_queue.put(stats_update)
            except Exception as e:
                print(f"ERROR in monitor_system loop: {e}")
                traceback.print_exc()
                wait_time = 5.0
            else:
                 elapsed = time.monotonic() - loop_start_time
                 wait_time = max(0.05, 1.0 - elapsed)
            self.stop_event.wait(wait_time)
        print("System monitor thread finished.")
    def screenshot_worker(self):
        print("Screenshot worker thread started.")
        self.sct_instance = None
        monitor_definition = None
        try:
            self.sct_instance = mss()
            print("mss instance created for screenshots.")
            if len(self.sct_instance.monitors) > 1:
                monitor_definition = self.sct_instance.monitors[1]
                print(f"Using monitor 1 (Primary physical): {monitor_definition}")
            elif len(self.sct_instance.monitors) == 1:
                monitor_definition = self.sct_instance.monitors[0]
                print(f"Using monitor 0 (Only one found - Virtual?): {monitor_definition}")
            else:
                print("ERROR: mss found no monitors!")
                raise RuntimeError("No monitors detected by mss")
            while not self.stop_event.is_set():
                loop_start_time = time.monotonic()
                current_delay = self.current_screenshot_delay_sec
                if data_collection_paused.is_set():
                    self.stop_event.wait(0.2)
                    continue
                try:
                    timestamp_ns = time.time_ns()
                    filename = SCREENSHOT_DIR / f"{timestamp_ns}.png"
                    sct_img = self.sct_instance.grab(monitor_definition)
                    to_png(sct_img.rgb, sct_img.size, output=str(filename))
                    if (timestamp_ns // 1_000_000_000) % 2 == 0:
                          gui_update_queue.put({"screenshot_count": count_screenshots()})
                except Exception as e:
                    print(f"ERROR during screenshot capture/save: {e}")
                    traceback.print_exc()
                    wait_time = max(current_delay, 3.0)
                    self.stop_event.wait(wait_time)
                    continue
                elapsed = time.monotonic() - loop_start_time
                wait_time = max(0.01, current_delay - elapsed)
                self.stop_event.wait(wait_time)
        except Exception as e:
             print(f"FATAL ERROR initializing mss or in screenshot loop: {e}")
             traceback.print_exc()
        finally:
            if hasattr(self.sct_instance, 'close') and callable(self.sct_instance.close):
                 try:
                     self.sct_instance.close()
                     print("mss instance closed.")
                 except Exception as e_close:
                      print(f"Error closing mss instance: {e_close}")
            self.sct_instance = None
            print("Screenshot worker finished.")
    def _get_key_repr(self, key):
        try:
            if isinstance(key, keyboard.Key):
                return key.name
            elif isinstance(key, keyboard.KeyCode):
                if key.char and key.char.isprintable():
                    return key.char
                if hasattr(key, 'name'): return key.name
                if hasattr(key, 'vk'): return f"vk_{key.vk}"
                return str(key)
            else:
                return str(key)
        except Exception:
             return repr(key)
    def _on_press(self, key):
        press_time_ns = time.time_ns()
        if data_collection_paused.is_set(): return
        try:
            key_repr = self._get_key_repr(key)
            self.key_press_times[key_repr] = press_time_ns
            if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
                       keyboard.Key.caps_lock, keyboard.Key.cmd, keyboard.Key.cmd_l,
                       keyboard.Key.cmd_r, keyboard.Key.scroll_lock, keyboard.Key.num_lock):
                return
            is_ctrl = key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)
            is_alt = key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr)
            combo_detected = False
            if is_ctrl:
                current_ctrl_time = press_time_ns
                if self.last_alt_press_time_ns > 0 and (current_ctrl_time - self.last_alt_press_time_ns) < self.ctrl_alt_threshold_ns:
                    print(f"Ctrl+Alt detected (Alt then Ctrl)! Diff: {(current_ctrl_time - self.last_alt_press_time_ns)/1e6:.1f} ms")
                    combo_detected = True
                self.last_ctrl_press_time_ns = current_ctrl_time
            elif is_alt:
                current_alt_time = press_time_ns
                if self.last_ctrl_press_time_ns > 0 and (current_alt_time - self.last_ctrl_press_time_ns) < self.ctrl_alt_threshold_ns:
                    print(f"Alt+Ctrl detected (Ctrl then Alt)! Diff: {(current_alt_time - self.last_ctrl_press_time_ns)/1e6:.1f} ms")
                    combo_detected = True
                self.last_alt_press_time_ns = current_alt_time
            if combo_detected:
                self.root.after(0, self._trigger_result_window)
                self.last_alt_press_time_ns = 0
                self.last_ctrl_press_time_ns = 0
            cleanup_threshold_ns = self.ctrl_alt_threshold_ns + 100_000_000
            if self.last_ctrl_press_time_ns > 0 and (press_time_ns - self.last_ctrl_press_time_ns) > cleanup_threshold_ns:
                 self.last_ctrl_press_time_ns = 0
            if self.last_alt_press_time_ns > 0 and (press_time_ns - self.last_alt_press_time_ns) > cleanup_threshold_ns:
                 self.last_alt_press_time_ns = 0
        except Exception as e:
            print(f"ERROR in _on_press callback for key {key}: {e}")
            traceback.print_exc()
    def _on_release(self, key):
        release_time_ns = time.time_ns()
        if data_collection_paused.is_set(): return
        try:
            key_repr = self._get_key_repr(key)
            key_name = key_repr
            if key_repr in self.key_press_times:
                press_time_ns = self.key_press_times.pop(key_repr)
                if release_time_ns < press_time_ns:
                    print(f"WARN: Key '{key_name}' release time ({release_time_ns}) is before press time ({press_time_ns}). Skipping log.")
                else:
                    key_data = {
                        "timestamp_ns": press_time_ns,
                        "press_time_ns": press_time_ns,
                        "release_time_ns": release_time_ns,
                        "key_name": key_name
                    }
                    if write_log(KEYBOARD_LOG_FILE, key_data):
                        gui_update_queue.put({"keyboard_count": count_log_lines(KEYBOARD_LOG_FILE)})
                pass
            if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self.last_ctrl_press_time_ns = 0
            elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
                self.last_alt_press_time_ns = 0
        except Exception as e:
             print(f"ERROR in _on_release callback for key {key}: {e}")
             traceback.print_exc()
    def keyboard_listener_worker(self):
        print("Keyboard listener thread started.")
        listener = None
        try:
            listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self.keyboard_listener = listener
            print("Keyboard listener created and starting...")
            listener.start()
            print("Keyboard listener running.")
            while listener.is_alive() and not self.stop_event.is_set():
                 self.stop_event.wait(0.5)
            print("Keyboard listener stop condition met.")
            if listener.is_alive():
                print("Requesting explicit listener stop...")
                listener.stop()
            print("Waiting for listener join...")
            listener.join()
            print("Listener joined.")
        except Exception as e:
             print(f"FATAL ERROR in keyboard listener worker: {e}")
             traceback.print_exc()
        finally:
             print("Keyboard listener worker finished.")
             self.keyboard_listener = None
    def _get_mouse_button_name(self, button):
        try:
             return button.name
        except AttributeError:
             return str(button)
    def _on_move(self, x, y):
        if data_collection_paused.is_set(): return
        current_time_ns = time.time_ns()
        for button_name in list(self.mouse_press_info.keys()):
            if button_name in self.mouse_press_info:
                 info = self.mouse_press_info[button_name]
                 info['trajectory'].append((x, y, current_time_ns))
                 MAX_TRAJ_POINTS = 2000
                 if len(info['trajectory']) > MAX_TRAJ_POINTS:
                     info['trajectory'].pop(1)
    def _on_click(self, x, y, button, pressed):
        current_time_ns = time.time_ns()
        if data_collection_paused.is_set(): return
        try:
            button_name = self._get_mouse_button_name(button)
            if button_name not in ['left', 'right']:
                 return
            if pressed:
                self.mouse_press_info[button_name] = {
                    'press_time_ns': current_time_ns,
                    'start_pos': (int(x), int(y)),
                    'trajectory': [(int(x), int(y), current_time_ns)]
                }
            else:
                if button_name in self.mouse_press_info:
                    info = self.mouse_press_info.pop(button_name)
                    press_time_ns = info['press_time_ns']
                    start_pos = info['start_pos']
                    end_pos = (int(x), int(y))
                    trajectory = info['trajectory']
                    release_time_ns = current_time_ns
                    if release_time_ns < press_time_ns:
                        print(f"WARN: Mouse '{button_name}' release time ({release_time_ns}) is before press time ({press_time_ns}). Skipping log.")
                        return
                    duration_ns = release_time_ns - press_time_ns
                    dx = end_pos[0] - start_pos[0]
                    dy = end_pos[1] - start_pos[1]
                    distance = math.hypot(dx, dy)
                    action_type = "click"
                    if distance >= self.drag_threshold_pixels:
                        action_type = "drag"
                    elif duration_ns >= self.long_press_threshold_ns:
                        action_type = "long_press"
                    mouse_data = {
                        "timestamp_ns": press_time_ns,
                        "press_time_ns": press_time_ns,
                        "release_time_ns": release_time_ns,
                        "action": action_type,
                        "button": button_name,
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                    }
                    if action_type == "drag":
                        if not trajectory or trajectory[-1][:2] != end_pos:
                             trajectory.append((end_pos[0], end_pos[1], release_time_ns))
                        mouse_data["trajectory"] = trajectory
                    if write_log(MOUSE_LOG_FILE, mouse_data):
                        gui_update_queue.put({"mouse_count": count_log_lines(MOUSE_LOG_FILE)})
                    pass
        except Exception as e:
             print(f"ERROR in _on_click callback for button {button} ({pressed}): {e}")
             traceback.print_exc()
    def _on_scroll(self, x, y, dx, dy):
        if data_collection_paused.is_set(): return
        pass
    def mouse_listener_worker(self):
        print("Mouse listener thread started.")
        listener = None
        try:
            listener = mouse.Listener(on_move=self._on_move,
                                      on_click=self._on_click,
                                      on_scroll=self._on_scroll)
            self.mouse_listener = listener
            print("Mouse listener created and starting...")
            listener.start()
            print("Mouse listener running.")
            while listener.is_alive() and not self.stop_event.is_set():
                self.stop_event.wait(0.5)
            print("Mouse listener stop condition met.")
            if listener.is_alive():
                 print("Requesting explicit listener stop...")
                 listener.stop()
            print("Waiting for listener join...")
            listener.join()
            print("Listener joined.")
        except Exception as e:
             print(f"FATAL ERROR in mouse listener worker: {e}")
             traceback.print_exc()
        finally:
             print("Mouse listener worker finished.")
             self.mouse_listener = None
    def _trigger_result_window(self):
        print("Attempting to trigger result window...")
        try:
             instance = ResultWindow(self.root)
             if instance is None or not instance.window.winfo_exists():
                  print("Failed to create or focus ResultWindow instance.")
             else:
                  print("ResultWindow triggered/focused successfully.")
        except Exception as e:
            print(f"ERROR trying to create/focus ResultWindow: {e}")
            traceback.print_exc()
            if data_collection_paused.is_set():
                 print("Resuming data collection due to ResultWindow trigger error.")
                 data_collection_paused.clear()
if __name__ == "__main__":
    print("========================================")
    print(" Starting Data Recorder Application")
    print("========================================")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version: {platform.python_version()}")
    try:
        ensure_experience_pool()
    except Exception as e:
        message = f"CRITICAL: Failed to set up experience pool at\n{EXPERIENCE_POOL_DIR}\n\nError: {e}\n\nApplication cannot continue."
        print(message)
        try:
            root_err = tk.Tk()
            root_err.withdraw()
            messagebox.showerror("Startup Error", message, parent=None)
            root_err.destroy()
        except Exception:
            pass
        exit(1)
    root = None
    task_manager = None
    try:
        root = tk.Tk()
        root.withdraw()
        app = MonitoringWindow(root)
        root.deiconify()
        print("Getting initial data counts...")
        initial_counts = {
            "screenshot_count": count_screenshots(),
            "keyboard_count": count_log_lines(KEYBOARD_LOG_FILE),
            "mouse_count": count_log_lines(MOUSE_LOG_FILE),
            "result_count": count_log_lines(RESULTS_LOG_FILE)
        }
        app.update_stats(initial_counts)
        root.update_idletasks()
        print("Initial counts displayed.")
        task_manager = BackgroundTaskManager(root)
        def on_closing():
            print("\nShutdown requested...")
            print("Proceeding with shutdown sequence...")
            if task_manager:
                task_manager.stop_all()
            if root:
                print("Destroying Tkinter root window...")
                root.destroy()
                print("Tkinter root window destroyed.")
            print("Application shutdown complete.")
        root.protocol("WM_DELETE_WINDOW", on_closing)
        print("Starting background tasks...")
        task_manager.start_all()
        print("========================================")
        print(" Application setup complete. Running...")
        print(" Press Ctrl+Alt (<1s) to log results.")
        print(" Close the Status window to quit.")
        print("========================================")
        root.mainloop()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Initiating graceful shutdown...")
        on_closing()
    except Exception as e:
        print("\n" + "="*40)
        print(f" UNEXPECTED ERROR in main execution block: {e}")
        print(" Traceback:")
        traceback.print_exc()
        print("="*40 + "\n")
        try:
            if root and root.winfo_exists():
                 messagebox.showerror("Fatal Runtime Error", f"An unexpected error occurred:\n\n{e}\n\nPlease check the console output.\nAttempting shutdown.", parent=root)
            else:
                 root_err = tk.Tk()
                 root_err.withdraw()
                 messagebox.showerror("Fatal Runtime Error", f"An unexpected error occurred:\n\n{e}\n\nPlease check the console output.\nAttempting shutdown.", parent=None)
                 root_err.destroy()
        except Exception as e_msg:
             print(f"(Could not display error messagebox: {e_msg})")
        print("Attempting shutdown after error...")
        try:
            if task_manager:
                task_manager.stop_all()
        except Exception as shutdown_e:
            print(f"Error during shutdown after exception: {shutdown_e}")
        finally:
            try:
                if root and root.winfo_exists():
                    root.destroy()
            except Exception:
                pass
        print("Shutdown attempted.")
    finally:
        print("Application final cleanup.")
        if task_manager and any(t.is_alive() for t in task_manager.threads):
             print("Forcing stop on any remaining threads...")
             task_manager.stop_event.set()
             time.sleep(0.1)
    print("Application finished.")