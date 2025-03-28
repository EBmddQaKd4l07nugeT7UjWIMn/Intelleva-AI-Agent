import os
import sys
import json
import base64
import threading
import time
from io import BytesIO
import psutil
import platform
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from pynput import keyboard, mouse
from PIL import ImageGrab, Image
from screeninfo import get_monitors
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pynvml

# 设置基础目录和所需文件的路径（所有文件必须与本程序存放在同一目录下）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIENCE_POOL_FILE = os.path.join(BASE_DIR, "experience_pool.json")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
KEYBOARD_AI_MODEL_FILE = os.path.join(BASE_DIR, "keyboard_ai_model.pth")
MOUSE_AI_MODEL_FILE = os.path.join(BASE_DIR, "mouse_ai_model.pth")

# 全局状态变量
mode = "learning"
last_input_time = 0
waiting_window_active = False
terminate_flag = False
experience_pool_lock = threading.Lock()
synthetic_events_lock = threading.Lock()
input_event_lock = threading.Lock()
print_lock = threading.Lock()
synthetic_keyboard_events = []
synthetic_mouse_events = []
keyboard_ai_version = 1
mouse_ai_version = 1
esc_press_start_time = None
esc_currently_pressed = False
keyboard_ai_counter = 0
mouse_ai_counter = 0
global_latest_screenshot = None

# 全局记录键盘按下时间和鼠标运动数据的字典
keyboard_press_times = {}
mouse_press_times = {}
mouse_move_data = {}

# 定义有效按键（AI输出不包括Esc、Ctrl、Alt）
KEYBOARD_KEYS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
    "Shift", "Ctrl", "Alt", "Tab", "CapsLock", "Esc", "Space", "Enter", "Backspace"
]
VALID_KEYBOARD_KEYS = [k for k in KEYBOARD_KEYS if k not in ("Esc", "Ctrl", "Alt")]

keyboard_training_details = None
mouse_training_details = None

def load_config():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except:
        config = {}
    required_keys = {
        "screenshot_interval": 1,
        "cpu_threshold": 50,
        "memory_threshold": 80,
        "esc_long_press_duration_ns": 3000000000,
        "screenshot_min_interval": 0.5,
        "default_training_sample_count": 1000,
        "default_training_epochs": 10,
        "default_training_batch_size": 32,
        "keyboard_hold_duration_ms": 100,
        "ai_task_interval_ms": 200,
        "resource_monitor_window_width": 300,
        "resource_monitor_window_height": 250,
        "waiting_window_width": 400,
        "waiting_window_height": 200,
        "waiting_window_progress_length": 300,
        "waiting_window_total_steps": 300,
        "waiting_window_progress_interval_ms": 50,
        "keyboard_model_conv_layers": [
            {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "pool_kernel_size": 2},
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "pool_kernel_size": 2},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "pool_kernel_size": 0}
        ],
        "keyboard_model_fc_out_features": len(VALID_KEYBOARD_KEYS),
        "keyboard_model_dropout": 0.5,
        "keyboard_model_fc_hidden_features": 128,
        "mouse_model_conv_layers": [
            {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "pool_kernel_size": 2},
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "pool_kernel_size": 2},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "pool_kernel_size": 0}
        ],
        "mouse_model_fc_out_features": 8,
        "mouse_model_dropout": 0.5,
        "mouse_model_fc_hidden_features": 128,
        "ai_key_release_offset_ns": 100000000,
        "ai_mouse_release_offset_ns": 100000000,
        "synthetic_event_expiry_ns": 500000000,
        "mouse_drag_trajectory_steps": 10,
        "mouse_long_press_threshold_ns": 1000000000,
        "screenshot_mode": "L",
        "screenshot_width": 64,
        "screenshot_height": 64,
        "default_screen_width": 1920,
        "default_screen_height": 1080,
        "mouse_offset_scale": 100,
        "mouse_adj_delta": 5,
        "training_learning_rate": 0.001,
        "training_weight_decay": 1e-4,
        "training_grad_clip": 1.0,
        "mouse_drag_threshold_px": 5
    }
    updated = False
    for key, value in required_keys.items():
        if key not in config:
            config[key] = value
            updated = True
    if updated:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    return config

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs, flush=True)

def process_screenshot_to_tensor(screenshot):
    cfg = load_config()
    img = screenshot.convert(cfg["screenshot_mode"]).resize((cfg["screenshot_width"], cfg["screenshot_height"]))
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return tensor

class KeyboardAIModel(nn.Module):
    def __init__(self):
        super(KeyboardAIModel, self).__init__()
        cfg = load_config()
        layers = []
        in_channels = 1
        for layer_cfg in cfg["keyboard_model_conv_layers"]:
            layers.append(nn.Conv2d(in_channels, layer_cfg["out_channels"], kernel_size=layer_cfg["kernel_size"],
                                    stride=layer_cfg["stride"], padding=layer_cfg["padding"]))
            layers.append(nn.BatchNorm2d(layer_cfg["out_channels"]))
            layers.append(nn.ReLU())
            if cfg["keyboard_model_dropout"] > 0:
                layers.append(nn.Dropout(cfg["keyboard_model_dropout"]))
            if layer_cfg["pool_kernel_size"] > 0:
                layers.append(nn.MaxPool2d(layer_cfg["pool_kernel_size"]))
            in_channels = layer_cfg["out_channels"]
        if cfg["keyboard_model_conv_layers"][-1]["pool_kernel_size"] == 0:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        if cfg["keyboard_model_fc_hidden_features"] > 0:
            self.fc1 = nn.Linear(in_channels, cfg["keyboard_model_fc_hidden_features"])
            self.dropout = nn.Dropout(cfg["keyboard_model_dropout"])
            self.fc2 = nn.Linear(cfg["keyboard_model_fc_hidden_features"], cfg["keyboard_model_fc_out_features"])
        else:
            self.fc = nn.Linear(in_channels, cfg["keyboard_model_fc_out_features"])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        cfg = load_config()
        if hasattr(self, "fc1"):
            x = self.fc1(x)
            x = self.dropout(x)
            logits = self.fc2(x)
        else:
            logits = self.fc(x)
        return logits

class MouseAIModel(nn.Module):
    def __init__(self):
        super(MouseAIModel, self).__init__()
        cfg = load_config()
        layers = []
        in_channels = 1
        for layer_cfg in cfg["mouse_model_conv_layers"]:
            layers.append(nn.Conv2d(in_channels, layer_cfg["out_channels"], kernel_size=layer_cfg["kernel_size"],
                                    stride=layer_cfg["stride"], padding=layer_cfg["padding"]))
            layers.append(nn.BatchNorm2d(layer_cfg["out_channels"]))
            layers.append(nn.ReLU())
            if cfg["mouse_model_dropout"] > 0:
                layers.append(nn.Dropout(cfg["mouse_model_dropout"]))
            if layer_cfg["pool_kernel_size"] > 0:
                layers.append(nn.MaxPool2d(layer_cfg["pool_kernel_size"]))
            in_channels = layer_cfg["out_channels"]
        if cfg["mouse_model_conv_layers"][-1]["pool_kernel_size"] == 0:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        if cfg["mouse_model_fc_hidden_features"] > 0:
            self.fc1 = nn.Linear(in_channels, cfg["mouse_model_fc_hidden_features"])
            self.dropout = nn.Dropout(cfg["mouse_model_dropout"])
            self.fc2 = nn.Linear(cfg["mouse_model_fc_hidden_features"], cfg["mouse_model_fc_out_features"])
        else:
            self.fc = nn.Linear(in_channels, cfg["mouse_model_fc_out_features"])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        cfg = load_config()
        if hasattr(self, "fc1"):
            x = self.fc1(x)
            x = self.dropout(x)
            output = self.fc2(x)
        else:
            output = self.fc(x)
        return output

def regularize_model(model):
    for param in model.parameters():
        if torch.isnan(param).any():
            param.data.zero_()
    return model

keyboard_ai_model = None
mouse_ai_model = None

def check_hardware_info():
    safe_print("Checking computer hardware information...")
    safe_print("Platform: {}".format(platform.platform()))
    safe_print("CPU Count: {}".format(psutil.cpu_count(logical=True)))
    safe_print("Total Memory: {} bytes".format(psutil.virtual_memory().total))
    if torch.cuda.is_available():
        safe_print("GPU is available")
    else:
        safe_print("GPU is not available")

def check_files():
    cfg = load_config()
    if not os.path.exists(EXPERIENCE_POOL_FILE):
        initial_data = {"keyboard": {}, "mouse": {}, "screenshot": {}, "ai_keyboard": {}, "ai_mouse": {}}
        with open(EXPERIENCE_POOL_FILE, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=4)
    if not os.path.exists(KEYBOARD_AI_MODEL_FILE):
        model = KeyboardAIModel()
        torch.save(model.state_dict(), KEYBOARD_AI_MODEL_FILE)
    if not os.path.exists(MOUSE_AI_MODEL_FILE):
        model = MouseAIModel()
        torch.save(model.state_dict(), MOUSE_AI_MODEL_FILE)
    if not os.path.exists(CONFIG_FILE):
        default_config = load_config()
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, ensure_ascii=False, indent=4)
    required_files = [EXPERIENCE_POOL_FILE, KEYBOARD_AI_MODEL_FILE, MOUSE_AI_MODEL_FILE, CONFIG_FILE]
    for file in required_files:
        if os.path.dirname(os.path.abspath(file)) != BASE_DIR:
            safe_print("Error: File {} is not in the program directory!".format(file))
            sys.exit(1)

def preload_models():
    global keyboard_ai_model, mouse_ai_model
    safe_print("Preloading keyboard and mouse AI model files...")
    cfg = load_config()
    keyboard_ai_model = KeyboardAIModel()
    state_dict = torch.load(KEYBOARD_AI_MODEL_FILE, map_location="cpu")
    keyboard_ai_model.load_state_dict(state_dict)
    mouse_ai_model = MouseAIModel()
    state_dict = torch.load(MOUSE_AI_MODEL_FILE, map_location="cpu")
    mouse_ai_model.load_state_dict(state_dict)

def get_key_object(key_str):
    mapping = {
        "Shift": keyboard.Key.shift, "Ctrl": keyboard.Key.ctrl, "Alt": keyboard.Key.alt, "Tab": keyboard.Key.tab,
        "CapsLock": keyboard.Key.caps_lock, "Esc": keyboard.Key.esc, "Space": keyboard.Key.space, "Enter": keyboard.Key.enter,
        "Backspace": keyboard.Key.backspace, "F1": keyboard.Key.f1, "F2": keyboard.Key.f2, "F3": keyboard.Key.f3, "F4": keyboard.Key.f4,
        "F5": keyboard.Key.f5, "F6": keyboard.Key.f6, "F7": keyboard.Key.f7, "F8": keyboard.Key.f8, "F9": keyboard.Key.f9,
        "F10": keyboard.Key.f10, "F11": keyboard.Key.f11, "F12": keyboard.Key.f12
    }
    return mapping.get(key_str, key_str)

def get_mouse_button(button_str):
    mapping = {"left": mouse.Button.left, "right": mouse.Button.right}
    return mapping.get(button_str, mouse.Button.left)

def is_synthetic_keyboard(event_type, key_str, current_time):
    with synthetic_events_lock:
        for event in synthetic_keyboard_events:
            if event.get("marker") == "AI" and event["type"] == event_type and event["key"] == key_str and event["time"] <= current_time <= event["expiry"]:
                synthetic_keyboard_events.remove(event)
                return True
    return False

def is_synthetic_mouse(event_type, identifier, current_time):
    with synthetic_events_lock:
        for event in synthetic_mouse_events:
            if event.get("marker") == "AI" and event["type"] == event_type and event["id"] == identifier and event["time"] <= current_time <= event["expiry"]:
                synthetic_mouse_events.remove(event)
                return True
    return False

class ResourceMonitorWindow:
    def __init__(self, master):
        # 创建唯一的系统资源监控窗口（Toplevel窗口）
        self.top = tk.Toplevel(master)
        cfg = load_config()
        monitors = get_monitors()
        if monitors:
            screen_width = monitors[0].width
            screen_height = monitors[0].height
        else:
            screen_width = cfg["default_screen_width"]
            screen_height = cfg["default_screen_height"]
        width = cfg["resource_monitor_window_width"]
        height = cfg["resource_monitor_window_height"]
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.top.geometry(f"{width}x{height}+{x}+{y}")
        self.top.title("Resource Monitor")
        self.top.protocol("WM_DELETE_WINDOW", self.on_close)
        self.cpu_label = tk.Label(self.top, text="")
        self.mem_label = tk.Label(self.top, text="")
        self.gpu_label = tk.Label(self.top, text="")
        self.screenshot_label = tk.Label(self.top, text="")
        self.keyboard_label = tk.Label(self.top, text="")
        self.mouse_label = tk.Label(self.top, text="")
        self.cpu_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=2)
        self.mem_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=2)
        self.gpu_label.grid(row=2, column=0, sticky="nsew", padx=10, pady=2)
        self.screenshot_label.grid(row=3, column=0, sticky="nsew", padx=10, pady=2)
        self.keyboard_label.grid(row=4, column=0, sticky="nsew", padx=10, pady=2)
        self.mouse_label.grid(row=5, column=0, sticky="nsew", padx=10, pady=2)
        for i in range(6):
            self.top.grid_rowconfigure(i, weight=1)
        self.top.grid_columnconfigure(0, weight=1)
        self.running = True
        self.default_font_size = 12
        self.font = tkFont.Font(size=self.default_font_size)
        self.cpu_label.config(font=self.font)
        self.mem_label.config(font=self.font)
        self.gpu_label.config(font=self.font)
        self.screenshot_label.config(font=self.font)
        self.keyboard_label.config(font=self.font)
        self.mouse_label.config(font=self.font)
        self.resize_after_id = None
        self.top.bind("<Configure>", self.on_resize)
        self.nvml_handle = None
        self.update_resources()

    def on_resize(self, event):
        if self.resize_after_id is not None:
            self.top.after_cancel(self.resize_after_id)
        self.resize_after_id = self.top.after(100, self.update_font_size_callback)

    def update_font_size_callback(self):
        new_size = max(8, min(self.top.winfo_width() // 20, self.top.winfo_height() // 6))
        self.font.configure(size=new_size)
        self.top.update_idletasks()

    def update_resources(self):
        if not self.running or terminate_flag:
            return
        cpu_usage = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        self.cpu_label.config(text="CPU Usage: {}%".format(cpu_usage))
        self.mem_label.config(text="Memory Usage: {}%".format(mem.percent))
        if torch.cuda.is_available():
            try:
                if self.nvml_handle is None:
                    pynvml.nvmlInit()
                    self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                gpu_usage = util.gpu
                self.gpu_label.config(text="GPU Usage: {}%".format(gpu_usage))
            except Exception as e:
                self.gpu_label.config(text="GPU Usage: Error")
        else:
            self.gpu_label.config(text="GPU Not Detected")
        try:
            with experience_pool_lock:
                with open(EXPERIENCE_POOL_FILE, "r", encoding="utf-8") as f:
                    pool = json.load(f)
            screenshot_count = len(pool.get("screenshot", {}))
            keyboard_count = len(pool.get("keyboard", {}))
            mouse_count = len(pool.get("mouse", {}))
        except:
            screenshot_count = keyboard_count = mouse_count = 0
        self.screenshot_label.config(text="Screenshots: {}".format(screenshot_count))
        self.keyboard_label.config(text="Keyboard Data: {}".format(keyboard_count))
        self.mouse_label.config(text="Mouse Data: {}".format(mouse_count))
        self.top.after(500, self.update_resources)

    def on_close(self):
        self.running = False
        self.top.destroy()

class WaitingWindow:
    def __init__(self, master, config):
        self.top = tk.Toplevel(master)
        monitors = get_monitors()
        if monitors:
            screen_width = monitors[0].width
            screen_height = monitors[0].height
        else:
            screen_width = config["default_screen_width"]
            screen_height = config["default_screen_height"]
        width = config["waiting_window_width"]
        height = config["waiting_window_height"]
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.top.geometry(f"{width}x{height}+{x}+{y}")
        self.top.title("Waiting Window")
        self.top.protocol("WM_DELETE_WINDOW", lambda: None)  # 禁止用户手动关闭
        self.progress = ttk.Progressbar(self.top, orient="horizontal", length=config["waiting_window_progress_length"], mode="determinate")
        self.progress.pack(padx=20, pady=20)
        self.total_steps = config["waiting_window_total_steps"]
        self.current_step = 0
        self.config = config
        self.optim_thread = threading.Thread(target=self.run_offline_optimization, daemon=True)
        self.optim_thread.start()
        self.update_progress()

    def update_progress(self):
        if self.optim_thread.is_alive():
            if self.current_step < self.total_steps:
                self.current_step += 1
                self.progress["value"] = (self.current_step / self.total_steps) * 100
            self.top.after(self.config["waiting_window_progress_interval_ms"], self.update_progress)
        else:
            self.progress["value"] = 100
            safe_print("Offline optimization completed. Training details:")
            if keyboard_training_details is not None:
                safe_print("Keyboard AI - Sample Count: {}, Epochs: {}, Batch Size: {}, Final Average Loss: {:.4f}".format(
                    keyboard_training_details["sample_count"], keyboard_training_details["epochs"],
                    keyboard_training_details["batch_size"], keyboard_training_details["final_loss"]))
            if mouse_training_details is not None:
                safe_print("Mouse AI - Sample Count: {}, Epochs: {}, Batch Size: {}, Final Average Loss: {:.4f}".format(
                    mouse_training_details["sample_count"], mouse_training_details["epochs"],
                    mouse_training_details["batch_size"], mouse_training_details["final_loss"]))
            global mode, waiting_window_active
            mode = "learning"
            waiting_window_active = False
            self.top.after(0, self.close_window)

    def run_offline_optimization(self):
        try:
            optimize_keyboard_ai()
        except Exception as e:
            safe_print("Keyboard AI offline optimization error: {}".format(e))
        try:
            optimize_mouse_ai()
        except Exception as e:
            safe_print("Mouse AI offline optimization error: {}".format(e))
        try:
            optimize_experience_pool()
        except Exception as e:
            safe_print("Experience pool offline optimization error: {}".format(e))
        try:
            preload_models()
        except Exception as e:
            safe_print("Model preloading error: {}".format(e))

    def close_window(self):
        self.top.destroy()

def get_dynamic_training_params(config):
    mem = psutil.virtual_memory()
    mem_ratio = mem.available / mem.total
    sample_count = max(500, int(config["default_training_sample_count"] * mem_ratio))
    epochs = max(1, int(config["default_training_epochs"] * mem_ratio))
    batch_size = max(1, int(config["default_training_batch_size"] * mem_ratio))
    return (sample_count, epochs, batch_size)

def screenshot_task():
    global global_latest_screenshot
    if terminate_flag:
        return
    config = load_config()
    if waiting_window_active:
        main_root.after(100, screenshot_task)
        return
    try:
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        interval = config["screenshot_interval"] * (max(cpu_usage, memory_usage) / config["cpu_threshold"])
        if interval < config["screenshot_min_interval"]:
            interval = config["screenshot_min_interval"]
        screenshot = ImageGrab.grab()
        global_latest_screenshot = screenshot.copy()
        timestamp = time.time_ns()
        buf = BytesIO()
        screenshot.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        data_entry = {"timestamp": timestamp, "image": image_base64}
        with experience_pool_lock:
            with open(EXPERIENCE_POOL_FILE, "r+", encoding="utf-8") as f:
                pool = json.load(f)
                pool["screenshot"][str(timestamp)] = data_entry
                f.seek(0)
                json.dump(pool, f, ensure_ascii=False, indent=4)
                f.truncate()
    except Exception as e:
        safe_print("Screenshot task encountered exception: {}".format(e))
    delay_ms = int(interval * 1000)
    main_root.after(delay_ms, screenshot_task)

def time_ns_val():
    try:
        return time.time_ns()
    except:
        return int(time.time() * 1000000000)

def mode_monitor_task():
    if terminate_flag:
        return
    main_root.after(500, mode_monitor_task)

def real_keyboard_ai_task():
    global keyboard_ai_counter
    if terminate_flag or mode != "training" or waiting_window_active:
        main_root.after(load_config()["ai_task_interval_ms"], real_keyboard_ai_task)
        return
    config = load_config()
    if global_latest_screenshot is None:
        main_root.after(config["ai_task_interval_ms"], real_keyboard_ai_task)
        return
    tensor = process_screenshot_to_tensor(global_latest_screenshot)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)
    keyboard_ai_model.to(device)
    keyboard_ai_model.eval()
    with torch.no_grad():
        logits = keyboard_ai_model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    max_prob = float(probs.max())
    if max_prob < 0.5:
        num_keys = 1
    elif max_prob < 0.7:
        num_keys = 2
    else:
        num_keys = 3
    base_time = time_ns_val()
    hold_duration = config["keyboard_hold_duration_ms"]
    buf = BytesIO()
    global_latest_screenshot.save(buf, format="PNG")
    screenshot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    for idx, index in enumerate(probs.argsort()[-num_keys:][::-1]):
        key_val = VALID_KEYBOARD_KEYS[index]
        press_time = base_time + idx
        release_time = press_time + config["ai_key_release_offset_ns"]
        confidence = round(float(probs[index]), 2)
        safe_print("Keyboard AI output operation: confidence: {:.2f}, key: {}, press time: {}, release time: {}".format(confidence, key_val, press_time, release_time))
        key_obj = get_key_object(key_val)
        with synthetic_events_lock:
            synthetic_keyboard_events.append({"type": "press", "key": key_val, "time": press_time, "expiry": press_time + config["synthetic_event_expiry_ns"], "marker": "AI"})
            synthetic_keyboard_events.append({"type": "release", "key": key_val, "time": release_time, "expiry": release_time + config["synthetic_event_expiry_ns"], "marker": "AI"})
        keyboard_controller.press(key_obj)
        main_root.after(hold_duration, lambda k=key_obj: keyboard_controller.release(k))
        data_entry = {"key": key_val, "press_time": press_time, "release_time": release_time, "confidence": confidence, "source": "AI", "screenshot": screenshot_base64}
        with experience_pool_lock:
            with open(EXPERIENCE_POOL_FILE, "r+", encoding="utf-8") as f:
                pool = json.load(f)
                pool["ai_keyboard"][str(press_time) + "_" + key_val] = data_entry
                f.seek(0)
                json.dump(pool, f, ensure_ascii=False, indent=4)
                f.truncate()
    keyboard_ai_counter += 1
    main_root.after(config["ai_task_interval_ms"], real_keyboard_ai_task)

def real_mouse_ai_task():
    global mouse_ai_counter
    if terminate_flag or mode != "training" or waiting_window_active:
        main_root.after(load_config()["ai_task_interval_ms"], real_mouse_ai_task)
        return
    config = load_config()
    if global_latest_screenshot is None:
        main_root.after(config["ai_task_interval_ms"], real_mouse_ai_task)
        return
    tensor = process_screenshot_to_tensor(global_latest_screenshot)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)
    mouse_ai_model.to(device)
    mouse_ai_model.eval()
    with torch.no_grad():
        output = mouse_ai_model(tensor).squeeze(0).cpu()
    out_conf = output[0]
    out_op_logits = output[1:4]
    out_offsets = output[4:8]
    confidence = round(float(torch.sigmoid(out_conf).item()), 2)
    op_index = int(torch.argmax(out_op_logits).item())
    op_mapping = {0: "click", 1: "long_press", 2: "drag"}
    op = op_mapping.get(op_index, "click")
    scale = config["mouse_offset_scale"]
    offsets = (out_offsets * scale).detach().numpy()
    monitors = get_monitors()
    if monitors:
        screen_width = monitors[0].width
        screen_height = monitors[0].height
    else:
        screen_width = config["default_screen_width"]
        screen_height = config["default_screen_height"]
    center_x = screen_width // 2
    center_y = screen_height // 2
    press_position = {"x": int(center_x + offsets[0]), "y": int(center_y + offsets[1])}
    release_position = {"x": int(center_x + offsets[2]), "y": int(center_y + offsets[3])}
    if confidence < 0.5:
        num_ops = 1
    elif confidence < 0.7:
        num_ops = 2
    else:
        num_ops = 3
    operations = []
    buf = BytesIO()
    global_latest_screenshot.save(buf, format="PNG")
    screenshot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    for i in range(num_ops):
        adj_press = {"x": press_position["x"] + i * config["mouse_adj_delta"], "y": press_position["y"] + i * config["mouse_adj_delta"]}
        adj_release = {"x": release_position["x"] - i * config["mouse_adj_delta"], "y": release_position["y"] - i * config["mouse_adj_delta"]}
        op_type = op
        if op_type == "drag":
            steps = config["mouse_drag_trajectory_steps"]
            trajectory = []
            for j in range(steps + 1):
                inter_x = int(adj_press["x"] + (adj_release["x"] - adj_press["x"]) * j / steps)
                inter_y = int(adj_press["y"] + (adj_release["y"] - adj_press["y"]) * j / steps)
                trajectory.append({"x": inter_x, "y": inter_y})
        else:
            trajectory = []
        current_press_time = time_ns_val()
        current_release_time = current_press_time + config["ai_mouse_release_offset_ns"]
        operations.append({"operation": op_type, "button": "left", "press_time": current_press_time, "release_time": current_release_time, "press_position": adj_press, "release_position": adj_release, "trajectory": trajectory, "confidence": confidence, "screenshot": screenshot_base64})
        with synthetic_events_lock:
            synthetic_mouse_events.append({"type": "press", "id": "left", "time": time_ns_val(), "expiry": time_ns_val() + config["synthetic_event_expiry_ns"], "marker": "AI"})
            synthetic_mouse_events.append({"type": "release", "id": "left", "time": time_ns_val() + config["ai_mouse_release_offset_ns"], "expiry": time_ns_val() + config["ai_mouse_release_offset_ns"] + config["synthetic_event_expiry_ns"], "marker": "AI"})
            synthetic_mouse_events.append({"type": "move", "id": f"{adj_press['x']}_{adj_press['y']}", "time": time_ns_val(), "expiry": time_ns_val() + config["synthetic_event_expiry_ns"], "marker": "AI"})
            if op_type == "drag":
                synthetic_mouse_events.append({"type": "move", "id": f"{adj_release['x']}_{adj_release['y']}", "time": time_ns_val() + config["ai_mouse_release_offset_ns"], "expiry": time_ns_val() + config["ai_mouse_release_offset_ns"] + config["synthetic_event_expiry_ns"], "marker": "AI"})
        button_obj = get_mouse_button("left")
        if op_type == "click":
            mouse_controller.position = (adj_press["x"], adj_press["y"])
            mouse_controller.click(button_obj, 1)
        elif op_type == "long_press":
            mouse_controller.position = (adj_press["x"], adj_press["y"])
            mouse_controller.press(button_obj)
            main_root.after(config["keyboard_hold_duration_ms"], lambda b=button_obj: mouse_controller.release(b))
        elif op_type == "drag":
            mouse_controller.position = (adj_press["x"], adj_press["y"])
            mouse_controller.press(button_obj)
            main_root.after(config["keyboard_hold_duration_ms"], lambda b=button_obj, pos=(adj_release["x"], adj_release["y"]): (setattr(mouse_controller, "position", pos), mouse_controller.release(b)))
        with experience_pool_lock:
            with open(EXPERIENCE_POOL_FILE, "r+", encoding="utf-8") as f:
                pool = json.load(f)
                key_id = str(time_ns_val()) + "_" + "left" + "_" + op_type + "_" + str(i)
                pool["ai_mouse"][key_id] = operations[i]
                f.seek(0)
                json.dump(pool, f, ensure_ascii=False, indent=4)
                f.truncate()
    printed_ops = []
    for op_item in operations:
        op_copy = {k: op_item[k] for k in op_item if k in ("operation", "button", "press_time", "release_time", "press_position", "release_position")}
        if op_item["operation"] == "drag":
            op_copy["trajectory"] = op_item["trajectory"]
        printed_ops.append(op_copy)
    safe_print("Mouse AI output operation: confidence: {:.2f}, number of operations: {}, details: {}".format(confidence, num_ops, printed_ops))
    mouse_ai_counter += 1
    main_root.after(config["ai_task_interval_ms"], real_mouse_ai_task)

def optimize_keyboard_ai():
    global keyboard_ai_model, keyboard_ai_version, keyboard_training_details
    try:
        with experience_pool_lock:
            with open(EXPERIENCE_POOL_FILE, "r", encoding="utf-8") as f:
                pool = json.load(f)
        data = pool.get("ai_keyboard", {})
        samples = []
        labels = []
        for key, entry in data.items():
            if "screenshot" in entry and "key" in entry:
                try:
                    img_data = base64.b64decode(entry["screenshot"])
                    img = Image.open(BytesIO(img_data))
                    tensor = process_screenshot_to_tensor(img)
                    samples.append(tensor)
                    if entry["key"] in VALID_KEYBOARD_KEYS:
                        labels.append(VALID_KEYBOARD_KEYS.index(entry["key"]))
                except:
                    continue
        if not samples or not labels:
            return
        X = torch.cat(samples, dim=0)
        y = torch.tensor(labels, dtype=torch.long)
        config = load_config()
        sample_count, epochs, batch_size = get_dynamic_training_params(config)
        dataset = torch.utils.data.TensorDataset(X, y)
        safe_print("Keyboard AI offline optimization started: Sample Count: {}, Epochs: {}, Batch Size: {}".format(len(dataset), epochs, batch_size))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keyboard_ai_model.to(device)
        keyboard_ai_model.train()
        optimizer = optim.Adam(keyboard_ai_model.parameters(), lr=config["training_learning_rate"], weight_decay=config["training_weight_decay"])
        loss_fn = nn.CrossEntropyLoss()
        epoch_losses = []
        for _ in range(epochs):
            epoch_loss = 0
            batch_count = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = keyboard_ai_model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(keyboard_ai_model.parameters(), config["training_grad_clip"])
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            if batch_count > 0:
                epoch_losses.append(epoch_loss / batch_count)
        keyboard_ai_model = regularize_model(keyboard_ai_model)
        torch.save(keyboard_ai_model.state_dict(), KEYBOARD_AI_MODEL_FILE)
        mem_ratio = psutil.virtual_memory().available / psutil.virtual_memory().total
        if mem_ratio > 0.5 and keyboard_ai_version <= 3:
            backup_file = KEYBOARD_AI_MODEL_FILE.replace(".pth", f"_v{keyboard_ai_version}.pth")
            torch.save(keyboard_ai_model.state_dict(), backup_file)
            keyboard_ai_version += 1
        final_loss = epoch_losses[-1] if epoch_losses else 0
        keyboard_training_details = {"sample_count": len(dataset), "epochs": epochs, "batch_size": batch_size, "final_loss": final_loss}
    except Exception as e:
        safe_print("Keyboard AI optimization error: {}".format(e))

def optimize_mouse_ai():
    global mouse_ai_model, mouse_ai_version, mouse_training_details
    try:
        with experience_pool_lock:
            with open(EXPERIENCE_POOL_FILE, "r", encoding="utf-8") as f:
                pool = json.load(f)
        data = pool.get("ai_mouse", {})
        samples = []
        conf_targets = []
        op_targets = []
        offset_targets = []
        monitors = get_monitors()
        config = load_config()
        if monitors:
            screen_width = monitors[0].width
            screen_height = monitors[0].height
        else:
            screen_width = config["default_screen_width"]
            screen_height = config["default_screen_height"]
        center_x = screen_width // 2
        center_y = screen_height // 2
        op_mapping = {"click": 0, "long_press": 1, "drag": 2}
        for key, entry in data.items():
            if "screenshot" in entry and "operation" in entry and "confidence" in entry and "press_position" in entry and "release_position" in entry:
                try:
                    img_data = base64.b64decode(entry["screenshot"])
                    img = Image.open(BytesIO(img_data))
                    tensor = process_screenshot_to_tensor(img)
                    samples.append(tensor)
                    conf = float(entry.get("confidence", 0.5))
                    conf_targets.append([conf])
                    op_label = op_mapping.get(entry["operation"], 0)
                    op_targets.append(op_label)
                    press = entry["press_position"]
                    release = entry["release_position"]
                    offset_px = [(press["x"] - center_x) / 100.0, (press["y"] - center_y) / 100.0, (release["x"] - center_x) / 100.0, (release["y"] - center_y) / 100.0]
                    offset_targets.append(offset_px)
                except:
                    continue
        if not samples or not conf_targets or not op_targets or not offset_targets:
            return
        X = torch.cat(samples, dim=0)
        conf_targets = torch.tensor(conf_targets, dtype=torch.float32)
        op_targets = torch.tensor(op_targets, dtype=torch.long)
        offset_targets = torch.tensor(offset_targets, dtype=torch.float32)
        _, epochs, batch_size = get_dynamic_training_params(config)
        dataset = torch.utils.data.TensorDataset(X, conf_targets, op_targets, offset_targets)
        safe_print("Mouse AI offline optimization started: Sample Count: {}, Epochs: {}, Batch Size: {}".format(len(dataset), epochs, batch_size))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mouse_ai_model.to(device)
        mouse_ai_model.train()
        optimizer = optim.Adam(mouse_ai_model.parameters(), lr=config["training_learning_rate"], weight_decay=config["training_weight_decay"])
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        epoch_losses = []
        for _ in range(epochs):
            epoch_loss = 0
            batch_count = 0
            for batch in loader:
                batch_x, batch_conf, batch_op, batch_offset = batch
                batch_x, batch_conf, batch_op, batch_offset = batch_x.to(device), batch_conf.to(device), batch_op.to(device), batch_offset.to(device)
                optimizer.zero_grad()
                output = mouse_ai_model(batch_x)
                out_conf = torch.sigmoid(output[:, 0])
                out_op_logits = output[:, 1:4]
                out_offsets = output[:, 4:8]
                loss_conf = mse_loss(out_conf, batch_conf.squeeze(1))
                loss_op = ce_loss(out_op_logits, batch_op)
                loss_offset = mse_loss(out_offsets, batch_offset)
                loss = loss_conf + loss_op + loss_offset
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mouse_ai_model.parameters(), config["training_grad_clip"])
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            if batch_count > 0:
                epoch_losses.append(epoch_loss / batch_count)
        mouse_ai_model = regularize_model(mouse_ai_model)
        torch.save(mouse_ai_model.state_dict(), MOUSE_AI_MODEL_FILE)
        mem_ratio = psutil.virtual_memory().available / psutil.virtual_memory().total
        if mem_ratio > 0.5 and mouse_ai_version <= 3:
            backup_file = MOUSE_AI_MODEL_FILE.replace(".pth", f"_v{mouse_ai_version}.pth")
            torch.save(mouse_ai_model.state_dict(), backup_file)
            mouse_ai_version += 1
        final_loss = epoch_losses[-1] if epoch_losses else 0
        mouse_training_details = {"sample_count": len(dataset), "epochs": epochs, "batch_size": batch_size, "final_loss": final_loss}
    except Exception as e:
        safe_print("Mouse AI optimization error: {}".format(e))

def optimize_experience_pool():
    try:
        with experience_pool_lock:
            with open(EXPERIENCE_POOL_FILE, "r+", encoding="utf-8") as f:
                pool = json.load(f)
                f.seek(0)
                json.dump(pool, f, ensure_ascii=False, indent=4)
                f.truncate()
    except Exception as e:
        safe_print("Experience pool optimization error: {}".format(e))

def on_activate_waiting():
    global waiting_window_active
    if waiting_window_active:
        return
    waiting_window_active = True
    safe_print("Detected Ctrl+Alt combination. Displaying waiting window and starting offline optimization.")
    main_root.after(0, lambda: WaitingWindow(main_root, load_config()))

def start_global_hotkey_listener():
    hotkey = keyboard.GlobalHotKeys({'<ctrl>+<alt>': on_activate_waiting})
    hotkey.start()

def on_key_press(key):
    global last_input_time, esc_press_start_time, esc_currently_pressed
    try:
        current_time = time_ns_val()
        try:
            key_str = key.char
        except:
            key_str = str(key)
        if waiting_window_active and key != keyboard.Key.esc:
            last_input_time = time.time()
            return
        if key == keyboard.Key.esc:
            esc_currently_pressed = True
            if esc_press_start_time is None:
                esc_press_start_time = current_time
        else:
            with input_event_lock:
                keyboard_press_times.setdefault(key_str, []).append(current_time)
        last_input_time = time.time()
    except Exception as e:
        safe_print("Keyboard press event error: {}".format(e))

def on_key_release(key):
    global last_input_time, terminate_flag, esc_press_start_time, mode, esc_currently_pressed
    try:
        current_time = time_ns_val()
        try:
            key_str = key.char
        except:
            key_str = str(key)
        if waiting_window_active and key != keyboard.Key.esc:
            last_input_time = time.time()
            return
        if key == keyboard.Key.esc:
            esc_currently_pressed = False
            config = load_config()
            duration = current_time - esc_press_start_time if esc_press_start_time is not None else 0
            if duration >= config["esc_long_press_duration_ns"]:
                safe_print("Detected long press on Esc. Program will terminate immediately.")
                terminate_flag = True
                os._exit(0)
            else:
                if mode == "learning" and not waiting_window_active:
                    safe_print("Detected Esc key press, switching to training mode.")
                    mode = "training"
                    real_keyboard_ai_task()
                    real_mouse_ai_task()
                elif mode == "training":
                    safe_print("Detected Esc key press, switching to learning mode.")
                    mode = "learning"
            esc_press_start_time = None
            return
        last_input_time = time.time()
        press_time = None
        with input_event_lock:
            if key_str in keyboard_press_times and keyboard_press_times[key_str]:
                press_time = keyboard_press_times[key_str].pop(0)
                if not keyboard_press_times[key_str]:
                    del keyboard_press_times[key_str]
        if press_time is None:
            return
        data_entry = {"key": key_str, "press_time": press_time, "release_time": current_time, "source": "AI" if mode=="training" else "user"}
        pool_key = "ai_keyboard" if mode=="training" else "keyboard"
        with experience_pool_lock:
            with open(EXPERIENCE_POOL_FILE, "r+", encoding="utf-8") as f:
                pool = json.load(f)
                pool[pool_key][str(press_time)] = data_entry
                f.seek(0)
                json.dump(pool, f, ensure_ascii=False, indent=4)
                f.truncate()
    except Exception as e:
        safe_print("Keyboard release event error: {}".format(e))

def on_mouse_press(x, y, button, pressed):
    try:
        current_time = time_ns_val()
        if waiting_window_active:
            global last_input_time
            last_input_time = time.time()
            return
        if is_synthetic_mouse("press", str(button), current_time):
            return
        last_input_time = time.time()
        press_time = current_time
        with input_event_lock:
            mouse_press_times.setdefault(str(button), []).append(press_time)
            mouse_move_data.setdefault(str(button), [])
            mouse_move_data[str(button)].clear()
            mouse_move_data[str(button)].append((x, y))
    except Exception as e:
        safe_print("Mouse press event error: {}".format(e))

def on_mouse_release(x, y, button, pressed):
    try:
        current_time = time_ns_val()
        if waiting_window_active:
            global last_input_time
            last_input_time = time.time()
            return
        if is_synthetic_mouse("release", str(button), current_time):
            return
        last_input_time = time.time()
        release_time = current_time
        btn_str = str(button)
        with input_event_lock:
            if btn_str in mouse_press_times and mouse_press_times[btn_str]:
                press_time = mouse_press_times[btn_str].pop(0)
                if not mouse_press_times[btn_str]:
                    del mouse_press_times[btn_str]
            else:
                return
            trajectory = mouse_move_data.get(btn_str, [])
            if not trajectory:
                trajectory = [(x, y)]
        config = load_config()
        operation_type = "click"
        if len(trajectory) > 1:
            dx = abs(trajectory[-1][0] - trajectory[0][0])
            dy = abs(trajectory[-1][1] - trajectory[0][1])
            if dx > config.get("mouse_drag_threshold_px", 5) or dy > config.get("mouse_drag_threshold_px", 5):
                operation_type = "drag"
            else:
                if (release_time - press_time) > config["mouse_long_press_threshold_ns"]:
                    operation_type = "long_press"
        data_entry = {
            "operation": operation_type,
            "button": btn_str,
            "press_time": press_time,
            "release_time": release_time,
            "press_position": {"x": trajectory[0][0], "y": trajectory[0][1]},
            "release_position": {"x": x, "y": y},
            "trajectory": trajectory if operation_type == "drag" else [],
            "source": "AI" if mode=="training" else "user"
        }
        pool_key = "ai_mouse" if mode=="training" else "mouse"
        with experience_pool_lock:
            with open(EXPERIENCE_POOL_FILE, "r+", encoding="utf-8") as f:
                pool = json.load(f)
                pool[pool_key][str(press_time)] = data_entry
                f.seek(0)
                json.dump(pool, f, ensure_ascii=False, indent=4)
                f.truncate()
        with input_event_lock:
            if btn_str in mouse_move_data:
                del mouse_move_data[btn_str]
    except Exception as e:
        safe_print("Mouse release event error: {}".format(e))

def on_mouse_move(x, y):
    try:
        current_time = time_ns_val()
        if waiting_window_active:
            global last_input_time
            last_input_time = time.time()
            return
        if is_synthetic_mouse("move", f"{x}_{y}", current_time):
            return
        last_input_time = time.time()
        with input_event_lock:
            for btn in list(mouse_move_data.keys()):
                mouse_move_data[btn].append((x, y))
    except Exception as e:
        safe_print("Mouse move event error: {}".format(e))

def check_esc_long_press():
    global esc_press_start_time, terminate_flag
    try:
        if terminate_flag:
            return
        if esc_currently_pressed and esc_press_start_time is not None:
            if time_ns_val() - esc_press_start_time >= load_config()["esc_long_press_duration_ns"]:
                safe_print("Detected long press on Esc. Program will terminate immediately.")
                terminate_flag = True
                os._exit(0)
    except Exception as e:
        safe_print("Error checking Esc long press: {}".format(e))
    main_root.after(100, check_esc_long_press)

def safe_on_mouse_click(x, y, button, pressed):
    try:
        if pressed:
            on_mouse_press(x, y, button, pressed)
        else:
            on_mouse_release(x, y, button, pressed)
    except Exception:
        pass

def safe_on_mouse_move(x, y):
    try:
        on_mouse_move(x, y)
    except Exception:
        pass

def main():
    check_hardware_info()
    check_files()
    load_config()
    preload_models()
    safe_print("Preloading complete, entering learning mode.")
    global last_input_time
    last_input_time = time.time()
    # 创建主Tkinter窗口并立即隐藏
    global main_root
    main_root = tk.Tk()
    main_root.withdraw()
    # 创建唯一的系统资源监控窗口
    ResourceMonitorWindow(main_root)
    config = load_config()
    screenshot_task()
    mode_monitor_task()
    check_esc_long_press()
    global keyboard_controller, mouse_controller
    keyboard_controller = keyboard.Controller()
    mouse_controller = mouse.Controller()
    kb_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    kb_listener.start()
    mouse_listener = mouse.Listener(on_click=safe_on_mouse_click, on_move=safe_on_mouse_move)
    mouse_listener.start()
    start_global_hotkey_listener()
    def check_terminate():
        if terminate_flag:
            main_root.quit()
        else:
            main_root.after(100, check_terminate)
    main_root.after(100, check_terminate)
    main_root.mainloop()

if __name__ == "__main__":
    main()
