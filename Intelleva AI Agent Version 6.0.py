import os
import sys
import json
import pickle
import threading
import time
import gc
import tkinter
import psutil
import pyautogui
import keyboard
import mouse
import io
import logging
import queue
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import jieba

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ----------------------- Global Variables -----------------------
global_config = {}         # configuration dictionary
script_directory = ""      # current script directory

# Locks for synchronizing AI model updates and calls
keyboard_ai_lock = threading.Lock()
mouse_ai_lock = threading.Lock()

# Global AI model variables for keyboard and mouse
keyboard_ai_model = None
mouse_ai_model = None

# Experience pool: screenshots, keyboard, mouse, text data keyed by nanosecond timestamp
experience_pool = {
    "screenshot": {},
    "keyboard": {},
    "mouse": {},
    "text": {}
}
# Locks for each data type
experience_pool_locks = {
    "screenshot": threading.Lock(),
    "keyboard": threading.Lock(),
    "mouse": threading.Lock(),
    "text": threading.Lock()
}
experience_pool_unsaved_count = 0
experience_pool_last_save_time = time.time()
experience_pool_save_lock = threading.Lock()

# Latest screenshot and text timestamps
latest_screenshot_ts = None
latest_text_ts = None

mode = "learning"          # current mode, initial "learning"
last_activity_time = time.time()  # last user activity (in seconds)
window_active = False       # input window flag
optimization_in_progress = False  # offline optimization flag
user_input_text = ""       # text entered by the user

# Current keyboard events (list of dicts with unique id, down and up timestamps)
current_keyboard_events = []
current_keyboard_events_lock = threading.Lock()  # lock for keyboard events

# Current mouse events (dict keyed by mouse button)
current_mouse_events = {}

# Tkinter root window and thread-safe queue for Tkinter operations
tk_root = None
tk_queue = queue.Queue()

# Global AI reward factor and adaptive loss scale
ai_reward_factor = 0.0
adaptive_loss_scale = 1.0

# ----------------------- Helper Functions -----------------------
def get_event_position(event):
    """Get screen coordinates from an event; fallback to current mouse position."""
    try:
        return (event.x, event.y)
    except AttributeError:
        return pyautogui.position()

def get_recent_ai_actions():
    """Retrieve AI's recent actions within ai_memory_window_seconds and join them into a string."""
    now_ns = time.time_ns()
    window_ns = global_config.get("ai_memory_window_seconds", 10) * 10**9
    ai_actions = []
    with experience_pool_locks["keyboard"]:
        for ts, rec in experience_pool["keyboard"].items():
            if rec.get("source") == "ai" and now_ns - ts <= window_ns:
                up_ts = rec.get("up_timestamp", "unknown")
                ai_actions.append(f"keyboard:{rec['key_name']} down:{rec['down_timestamp']} up:{up_ts}")
    with experience_pool_locks["mouse"]:
        for ts, rec in experience_pool["mouse"].items():
            if rec.get("source") == "ai" and now_ns - ts <= window_ns:
                up_ts = rec.get("up_timestamp", "unknown")
                ai_actions.append(f"mouse:{rec['operation']} button:{rec['button']} down:{rec['down_timestamp']} up:{up_ts}")
    return " ".join(ai_actions)

# ----------------------- AI Model Definitions -----------------------
class ImageEncoder(nn.Module):
    def __init__(self, common_dim, image_size, pretrained=True):
        super(ImageEncoder, self).__init__()
        if pretrained:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        resnet = torchvision.models.resnet18(weights=weights)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(512, common_dim)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, common_dim, num_layers, num_heads):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, common_dim)
    def forward(self, x):
        # x should be a 2D tensor of shape (batch_size, sequence_length)
        embedded = self.embedding(x)
        transformer_out = self.transformer_encoder(embedded)
        out = transformer_out.mean(dim=1)
        out = self.fc(out)
        return out

class MultiModalFusion(nn.Module):
    def __init__(self, common_dim, num_heads):
        super(MultiModalFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, batch_first=True)
    def forward(self, image_feature, text_feature):
        image_feature = image_feature.unsqueeze(1)
        text_feature = text_feature.unsqueeze(1)
        attn_output, _ = self.attention(query=text_feature, key=image_feature, value=image_feature)
        fused = attn_output.squeeze(1)
        return fused

class DragSequenceGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length):
        super(DragSequenceGenerator, self).__init__()
        self.sequence_length = sequence_length
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, features):
        batch_size = features.size(0)
        hidden = features.unsqueeze(0)
        zeros = torch.zeros(batch_size, self.sequence_length, features.size(1), device=features.device)
        out, _ = self.gru(zeros, hidden)
        offsets = self.fc(out)
        return offsets

class KeyboardAIDecisionModel(nn.Module):
    def __init__(self, vocab_size, text_embed_dim, common_dim, text_transformer_layers, text_transformer_heads,
                 num_keyboard_keys, image_size, pretrained_image=True):
        super(KeyboardAIDecisionModel, self).__init__()
        self.image_encoder = ImageEncoder(common_dim, image_size, pretrained=pretrained_image)
        self.text_encoder = TextEncoder(vocab_size, text_embed_dim, common_dim,
                                        num_layers=text_transformer_layers, num_heads=text_transformer_heads)
        self.fusion = MultiModalFusion(common_dim, num_heads=text_transformer_heads)
        self.fc1 = nn.Linear(common_dim, common_dim)
        self.fc2 = nn.Linear(common_dim, common_dim // 2)
        self.keyboard_head = nn.Linear(common_dim // 2, num_keyboard_keys)
        self.version = 1
    def forward(self, image, text):
        img_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        fused = self.fusion(img_features, text_features)
        x = torch.relu(self.fc1(fused))
        x = torch.relu(self.fc2(x))
        keyboard_logits = self.keyboard_head(x)
        return keyboard_logits

class MouseAIDecisionModel(nn.Module):
    def __init__(self, vocab_size, text_embed_dim, common_dim, text_transformer_layers, text_transformer_heads,
                 num_mouse_output, image_size, drag_sequence_length, pretrained_image=True):
        super(MouseAIDecisionModel, self).__init__()
        self.image_encoder = ImageEncoder(common_dim, image_size, pretrained=pretrained_image)
        self.text_encoder = TextEncoder(vocab_size, text_embed_dim, common_dim,
                                        num_layers=text_transformer_layers, num_heads=text_transformer_heads)
        self.fusion = MultiModalFusion(common_dim, num_heads=text_transformer_heads)
        self.fc1 = nn.Linear(common_dim, common_dim)
        self.fc2 = nn.Linear(common_dim, common_dim // 2)
        self.max_parallel_mouse_ops = num_mouse_output
        self.mouse_op_head = nn.Linear(common_dim // 2, 4 * num_mouse_output)
        self.drag_sequence_generator = DragSequenceGenerator(common_dim // 2, common_dim // 2, drag_sequence_length * num_mouse_output)
        self.version = 1
    def forward(self, image, text):
        img_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        fused = self.fusion(img_features, text_features)
        x = torch.relu(self.fc1(fused))
        x = torch.relu(self.fc2(x))
        raw_mouse_logits = self.mouse_op_head(x)
        mouse_op_logits = raw_mouse_logits.view(x.size(0), self.max_parallel_mouse_ops, 4)
        raw_drag_offsets = self.drag_sequence_generator(x)
        drag_offsets = raw_drag_offsets.view(x.size(0), self.max_parallel_mouse_ops, -1, 2)
        return {"op_logits": mouse_op_logits, "drag_offsets": drag_offsets}

# ----------------------- Model Creation Functions -----------------------
def create_new_keyboard_model():
    default_keys = global_config.get("default_keyboard_keys", [])
    keyboard_output_dim = len(default_keys)
    vocab_size = global_config.get("text_vocab_size", 5000)
    text_embed_dim = global_config.get("text_embed_dim", 128)
    common_dim = global_config.get("common_feature_dim", 256)
    text_transformer_layers = global_config.get("text_transformer_layers", 2)
    text_transformer_heads = global_config.get("text_transformer_heads", 4)
    image_size = global_config.get("image_size", 224)
    model = KeyboardAIDecisionModel(vocab_size, text_embed_dim, common_dim, text_transformer_layers,
                                    text_transformer_heads, keyboard_output_dim, image_size, pretrained_image=True)
    return model

def create_new_mouse_model():
    default_mouse_ops = global_config.get("default_mouse_operations", ["click", "long_press", "drag"])
    num_mouse_output = global_config.get("max_parallel_mouse_ops", 3)
    vocab_size = global_config.get("text_vocab_size", 5000)
    text_embed_dim = global_config.get("text_embed_dim", 128)
    common_dim = global_config.get("common_feature_dim", 256)
    text_transformer_layers = global_config.get("text_transformer_layers", 2)
    text_transformer_heads = global_config.get("text_transformer_heads", 4)
    image_size = global_config.get("image_size", 224)
    drag_sequence_length = global_config.get("drag_sequence_length", 10)
    model = MouseAIDecisionModel(vocab_size, text_embed_dim, common_dim, text_transformer_layers,
                                  text_transformer_heads, num_mouse_output, image_size, drag_sequence_length, pretrained_image=True)
    return model

def get_or_create_keyboard_model():
    global keyboard_ai_model, script_directory, global_config
    model_filename = global_config["required_files"]["keyboard_ai_model"]
    model_path = os.path.join(script_directory, model_filename)
    if not os.path.exists(model_path):
        keyboard_ai_model = create_new_keyboard_model()
        try:
            torch.save(keyboard_ai_model, model_path)
            logging.info("已自动生成键盘AI模型文件。")
        except Exception as e:
            logging.error("保存新的键盘模型失败：%s", e)
    else:
        try:
            keyboard_ai_model = torch.load(model_path)
            logging.info("成功预加载键盘AI模型文件。")
        except Exception as e:
            logging.error("加载键盘AI模型失败：%s", e)
            backup_model_path = os.path.join(script_directory, "keyboard_ai_model_backup.pkl")
            if os.path.exists(backup_model_path):
                try:
                    keyboard_ai_model = torch.load(backup_model_path)
                    torch.save(keyboard_ai_model, model_path)
                    logging.info("使用备份键盘模型恢复成功。")
                except Exception as ex:
                    logging.error("加载备份键盘模型失败：%s", ex)
                    keyboard_ai_model = create_new_keyboard_model()
                    torch.save(keyboard_ai_model, model_path)
            else:
                keyboard_ai_model = create_new_keyboard_model()
                torch.save(keyboard_ai_model, model_path)

def get_or_create_mouse_model():
    global mouse_ai_model, script_directory, global_config
    model_filename = global_config["required_files"]["mouse_ai_model"]
    model_path = os.path.join(script_directory, model_filename)
    if not os.path.exists(model_path):
        mouse_ai_model = create_new_mouse_model()
        try:
            torch.save(mouse_ai_model, model_path)
            logging.info("已自动生成鼠标AI模型文件。")
        except Exception as e:
            logging.error("保存新的鼠标模型失败：%s", e)
    else:
        try:
            mouse_ai_model = torch.load(model_path)
            logging.info("成功预加载鼠标AI模型文件。")
        except Exception as e:
            logging.error("加载鼠标AI模型失败：%s", e)
            backup_model_path = os.path.join(script_directory, "mouse_ai_model_backup.pkl")
            if os.path.exists(backup_model_path):
                try:
                    mouse_ai_model = torch.load(backup_model_path)
                    torch.save(mouse_ai_model, model_path)
                    logging.info("使用备份鼠标模型恢复成功。")
                except Exception as ex:
                    logging.error("加载备份鼠标模型失败：%s", ex)
                    mouse_ai_model = create_new_mouse_model()
                    torch.save(mouse_ai_model, model_path)
            else:
                mouse_ai_model = create_new_mouse_model()
                torch.save(mouse_ai_model, model_path)

# ----------------------- File and Memory Management -----------------------
def save_experience_pool():
    try:
        experience_pool_path = os.path.join(script_directory, global_config["required_files"]["experience_pool"])
        with open(experience_pool_path, "wb") as f:
            pickle.dump(experience_pool, f)
    except Exception as e:
        logging.error("保存经验池失败：%s", e)

def calculate_data_value(record, data_type):
    try:
        if data_type == "screenshot":
            return len(record) * 0.001 if isinstance(record, bytes) else 0
        elif data_type == "keyboard":
            if "up_timestamp" in record and "down_timestamp" in record:
                duration = (record["up_timestamp"] - record["down_timestamp"]) / 1e6
                return duration if duration >= 50 else 0
            return 0
        elif data_type == "mouse":
            if "down_position" in record and "up_position" in record:
                dx = record["up_position"][0] - record["down_position"][0]
                dy = record["up_position"][1] - record["down_position"][1]
                distance = (dx**2 + dy**2)**0.5
                track_length = len(record.get("movement_track", []))
                return distance + track_length * 0.1
            return 0
        elif data_type == "text":
            if "text" in record:
                return len(record["text"])
            return 0
    except Exception as e:
        logging.error("计算数据价值时出错：%s", e)
        return 0

def clean_experience_pool():
    try:
        base_max_entries = global_config.get("max_experience_entries", {"screenshot": 500, "keyboard": 500, "mouse": 500, "text": 500})
        threshold_bytes = global_config.get("experience_pool_intelligent_optimization_min_bytes", 1073741824)
        current_size = len(pickle.dumps(experience_pool))
        current_mem_percent = psutil.virtual_memory().percent
        for data_type, base_max in base_max_entries.items():
            dynamic_max = max(50, int(base_max * (100 - current_mem_percent) / 100))
            current_pool = experience_pool.get(data_type, {})
            if len(current_pool) > dynamic_max:
                if current_size > threshold_bytes:
                    value_list = []
                    for key, record in current_pool.items():
                        value_score = calculate_data_value(record, data_type)
                        value_list.append((key, value_score))
                    value_list.sort(key=lambda x: x[1])
                    keys_to_delete = [key for key, score in value_list[:len(current_pool) - dynamic_max]]
                    for key in keys_to_delete:
                        del current_pool[key]
                else:
                    sorted_keys = sorted(current_pool.keys())
                    keys_to_delete = sorted_keys[:len(current_pool) - dynamic_max]
                    for key in keys_to_delete:
                        del current_pool[key]
    except Exception as e:
        logging.error("清理经验池时出错：%s", e)

def mark_experience_pool_unsaved(count=1):
    global experience_pool_unsaved_count
    with experience_pool_save_lock:
        experience_pool_unsaved_count += count

def experience_pool_saver_loop():
    global experience_pool_unsaved_count, experience_pool_last_save_time
    base_interval_min = global_config.get("experience_pool_save_interval_min", 5)
    base_interval_max = global_config.get("experience_pool_save_interval_max", 30)
    batch_count_min = global_config.get("experience_pool_batch_count_min", 3)
    batch_count_max = global_config.get("experience_pool_batch_count_max", 10)
    while True:
        try:
            time.sleep(1)
            current_time = time.time()
            with experience_pool_save_lock:
                elapsed = current_time - experience_pool_last_save_time
                unsaved = experience_pool_unsaved_count
                batch_count = batch_count_max if unsaved >= batch_count_max else batch_count_min
                ratio = unsaved / float(batch_count_max)
                save_interval = base_interval_max - (base_interval_max - base_interval_min) * min(ratio, 1.0)
                if unsaved >= batch_count or elapsed >= save_interval:
                    for key in sorted(experience_pool_locks.keys()):
                        experience_pool_locks[key].acquire()
                    try:
                        save_experience_pool()
                        clean_experience_pool()
                        experience_pool_unsaved_count = 0
                        experience_pool_last_save_time = time.time()
                    finally:
                        for key in sorted(experience_pool_locks.keys()):
                            experience_pool_locks[key].release()
        except Exception as e:
            logging.error("经验池保存线程错误：%s", e)
            time.sleep(1)

# ----------------------- Check and Create Files -----------------------
def check_and_create_files():
    global global_config, script_directory, experience_pool, adaptive_loss_scale, mode
    try:
        script_directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_directory, "config.json")
        # Updated required files to include keyboard and mouse AI models
        if not os.path.exists(config_path):
            default_config = {
                "required_files": {
                    "config": "config.json",
                    "keyboard_ai_model": "keyboard_ai_model.pkl",
                    "mouse_ai_model": "mouse_ai_model.pkl",
                    "experience_pool": "experience_pool.pkl"
                },
                "hotkey": "ctrl+alt",
                "max_parallel_mouse_ops": 3,
                "max_parallel_keyboard_ops": 3,
                "memory_threshold_percent": 90,
                "memory_check_interval": 1,
                "screen_capture_interval_base": 0.1,
                "screen_capture_cpu_low_threshold": 20,
                "screen_capture_cpu_high_threshold": 80,
                "screen_capture_interval_multiplier_low": 0.5,
                "screen_capture_interval_multiplier_high": 2.0,
                "screen_capture_interval_min_bound": 0.05,
                "screen_capture_interval_max_bound": 1.0,
                "default_keyboard_keys": [
                    "esc", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
                    "print_screen", "scroll_lock", "pause",
                    "tilde", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "minus", "equal", "backspace",
                    "tab", "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "left_bracket", "right_bracket", "backslash",
                    "caps_lock", "a", "s", "d", "f", "g", "h", "j", "k", "l", "semicolon", "quote", "enter",
                    "left_shift", "z", "x", "c", "v", "b", "n", "m", "comma", "period", "slash", "right_shift",
                    "left_ctrl", "left_windows", "left_alt", "space", "right_alt", "right_windows", "menu", "right_ctrl"
                ],
                "default_mouse_operations": ["click", "long_press", "drag"],
                "long_press_duration": 1,
                "drag_distance": 100,
                "drag_duration": 0.5,
                "drag_sequence_length": 10,
                "text_vocab_size": 5000,
                "text_embed_dim": 128,
                "text_hidden_dim": 256,
                "image_size": 224,
                "common_feature_dim": 256,
                "text_transformer_layers": 2,
                "text_transformer_heads": 4,
                "optimization_epochs": 100,
                "optimizer_lr": 0.001,
                "max_experience_entries": {
                    "screenshot": 500,
                    "keyboard": 500,
                    "mouse": 500,
                    "text": 500
                },
                "experience_pool_save_interval_min": 5,
                "experience_pool_save_interval_max": 30,
                "experience_pool_batch_count_min": 3,
                "experience_pool_batch_count_max": 10,
                "memory_for_capture_threshold": 70,
                "optimization_batch_size": 32,
                "lr_decay_factor": 0.9,
                "lr_decay_patience": 5,
                "max_screenshot_age_ns": 5000000000,
                "experience_pool_intelligent_optimization_min_bytes": 1073741824,
                "keyboard_press_duration": 0.05,
                "mouse_click_duration": 0.05,
                "ai_decision_threshold_factor": 0.5,
                "ai_decision_threshold_min": 0.2,
                "inactivity_duration_threshold": 10,
                "esc_termination_duration": 3,
                "offline_optimization_patience": 10,
                "optimization_duration_seconds": 10,
                "offline_optimization_sample_threshold": 10,
                "tk_queue_interval_ms": 100,
                "ai_training_loop_interval": 0.1,
                "input_window_update_interval_ms": 10,
                "ai_decision_interval": 0.1,
                "mouse_drag_min_distance": 5,
                "online_optimizer_lr": 1e-4,
                "online_optimization_interval": 10,
                "ai_memory_window_seconds": 10,
                "adaptive_loss_scale": 1.0,
                "input_window_initial_width_ratio": 0.3,
                "input_window_initial_height_ratio": 0.1,
                "input_window_min_width": 300,
                "input_window_min_height": 100,
                "experience_pool_saver_sleep": 1,
                "screen_capture_error_sleep": 0.5,
                "inactivity_monitor_interval": 1,
                "esc_monitor_interval": 0.1,
                "cognitive_active_interval": 30,
                "cognitive_active_error_sleep": 5,
                "keyboard_timeout_duration": 2,
                "input_window_update_interval_ms": 10
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=4)
            logging.info("已自动生成配置文件。")
        with open(config_path, "r", encoding="utf-8") as f:
            global_config = json.load(f)
        adaptive_loss_scale = global_config.get("adaptive_loss_scale", 1.0)
        get_or_create_keyboard_model()
        get_or_create_mouse_model()
        exp_pool_filename = global_config["required_files"]["experience_pool"]
        exp_pool_path = os.path.join(script_directory, exp_pool_filename)
        if not os.path.exists(exp_pool_path):
            with open(exp_pool_path, "wb") as f:
                pickle.dump(experience_pool, f)
            logging.info("已自动生成经验池文件。")
        else:
            try:
                with open(exp_pool_path, "rb") as f:
                    experience_pool_data = pickle.load(f)
                    if isinstance(experience_pool_data, dict):
                        for key in experience_pool:
                            if key in experience_pool_data and isinstance(experience_pool_data[key], dict):
                                experience_pool[key].update(experience_pool_data[key])
            except Exception:
                experience_pool.clear()
                experience_pool.update({
                    "screenshot": {},
                    "keyboard": {},
                    "mouse": {},
                    "text": {}
                })
                mark_experience_pool_unsaved()
        mode = "learning"
        logging.info("文件结构验证完毕，进入学习模式。")
    except Exception as e:
        logging.error("文件检查/创建失败：%s", e)
        sys.exit(1)

def memory_monitor():
    check_interval = global_config.get("memory_check_interval", 1)
    threshold = global_config.get("memory_threshold_percent", 90)
    while True:
        try:
            mem = psutil.virtual_memory()
            if mem.percent >= threshold:
                gc.collect()
            time.sleep(check_interval)
        except Exception as e:
            logging.error("内存监控错误：%s", e)
            time.sleep(check_interval)

# ----------------------- Data Preprocessing -----------------------
def preprocess_screenshot(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_size = global_config.get("image_size", 224)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        return img
    except Exception as e:
        logging.error("截图预处理错误：%s", e)
        image_size = global_config.get("image_size", 224)
        return torch.zeros((3, image_size, image_size), dtype=torch.float32)

def preprocess_text(text):
    try:
        tokens = jieba.lcut(text)
        vocab_size = global_config.get("text_vocab_size", 5000)
        indices = [abs(hash(token)) % vocab_size for token in tokens]
        if len(indices) == 0:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor(indices, dtype=torch.long)
    except Exception as e:
        logging.error("文本预处理错误：%s", e)
        return torch.tensor([], dtype=torch.long)

# ----------------------- Screenshot and Event Capture -----------------------
def screenshot_capture_loop():
    global latest_screenshot_ts
    base_interval = global_config.get("screen_capture_interval_base")
    mem_capture_threshold = global_config.get("memory_for_capture_threshold")
    cpu_low_threshold = global_config.get("screen_capture_cpu_low_threshold")
    cpu_high_threshold = global_config.get("screen_capture_cpu_high_threshold")
    multiplier_low = global_config.get("screen_capture_interval_multiplier_low")
    multiplier_high = global_config.get("screen_capture_interval_multiplier_high")
    min_bound = global_config.get("screen_capture_interval_min_bound")
    max_bound = global_config.get("screen_capture_interval_max_bound")
    while True:
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            mem_usage = psutil.virtual_memory().percent
            if cpu_usage <= cpu_low_threshold and mem_usage < mem_capture_threshold:
                multiplier = multiplier_low
            elif cpu_usage >= cpu_high_threshold or mem_usage >= mem_capture_threshold:
                multiplier = multiplier_high
            else:
                multiplier = multiplier_low + (cpu_usage - cpu_low_threshold) / (cpu_high_threshold - cpu_low_threshold) * (multiplier_high - multiplier_low)
            interval = base_interval * multiplier
            interval = max(min_bound, min(interval, max_bound))
            screenshot = pyautogui.screenshot()
            buf = io.BytesIO()
            screenshot.save(buf, format='PNG')
            img_bytes = buf.getvalue()
            timestamp = time.time_ns()
            with experience_pool_locks["screenshot"]:
                experience_pool["screenshot"][timestamp] = img_bytes
                latest_screenshot_ts = timestamp
            mark_experience_pool_unsaved()
            time.sleep(interval)
        except Exception as e:
            logging.error("截图捕获错误：%s", e)
            time.sleep(global_config.get("screen_capture_error_sleep", 0.5))

# ----------------------- Keyboard and Mouse Event Handlers -----------------------
def on_key_down(event):
    global last_activity_time, mode, current_keyboard_events
    try:
        last_activity_time = time.time()
        if mode != "learning":
            mode = "learning"
        event_id = time.time_ns()
        record = {
            "id": event_id,
            "key_name": event.name.lower(),
            "scan_code": event.scan_code if hasattr(event, "scan_code") else None,
            "down_timestamp": event_id
        }
        with current_keyboard_events_lock:
            current_keyboard_events.append(record)
    except Exception as e:
        logging.error("按键按下事件错误：%s", e)

def on_key_up(event):
    global last_activity_time, mode, current_keyboard_events
    try:
        last_activity_time = time.time()
        if mode != "learning":
            mode = "learning"
        up_time = time.time_ns()
        matched_record = None
        with current_keyboard_events_lock:
            for rec in reversed(current_keyboard_events):
                if rec["key_name"] == event.name.lower() and rec.get("scan_code") == (event.scan_code if hasattr(event, "scan_code") else None) and "up_timestamp" not in rec:
                    matched_record = rec
                    break
            if matched_record:
                matched_record["up_timestamp"] = up_time
                matched_record["source"] = "user"
                with experience_pool_locks["keyboard"]:
                    experience_pool["keyboard"][matched_record["id"]] = matched_record
                current_keyboard_events.remove(matched_record)
                mark_experience_pool_unsaved()
    except Exception as e:
        logging.error("按键松开事件错误：%s", e)

def on_keyboard_event(event):
    """Unified keyboard event callback to capture both down and up events."""
    if event.event_type == "down":
        on_key_down(event)
    elif event.event_type == "up":
        on_key_up(event)

def mouse_event_handler(event):
    global last_activity_time, mode, current_mouse_events
    try:
        last_activity_time = time.time()
        if mode != "learning":
            mode = "learning"
        event_type = event.event_type if hasattr(event, "event_type") else "move"
        if event_type == "down":
            event_id = time.time_ns()
            record = {
                "id": event_id,
                "operation": None,
                "button": event.button,
                "down_timestamp": event_id,
                "down_position": get_event_position(event),
                "movement_track": []
            }
            current_mouse_events.setdefault(event.button, []).append(record)
        elif event_type == "move":
            for rec_list in current_mouse_events.values():
                for rec in rec_list:
                    rec["movement_track"].append({"timestamp": time.time_ns(), "position": get_event_position(event)})
        elif event_type == "up":
            up_time = time.time_ns()
            if event.button in current_mouse_events and current_mouse_events[event.button]:
                rec = current_mouse_events[event.button].pop(0)
                rec["up_timestamp"] = up_time
                rec["up_position"] = get_event_position(event)
                duration = (rec["up_timestamp"] - rec["down_timestamp"]) / 1e9
                dx = rec["up_position"][0] - rec["down_position"][0]
                dy = rec["up_position"][1] - rec["down_position"][1]
                distance = (dx**2 + dy**2)**0.5
                if distance > global_config.get("mouse_drag_min_distance", 5) and len(rec["movement_track"]) > 0:
                    rec["operation"] = "drag"
                elif duration >= global_config.get("long_press_duration", 1):
                    rec["operation"] = "long_press"
                else:
                    rec["operation"] = "click"
                rec["source"] = "user"
                with experience_pool_locks["mouse"]:
                    experience_pool["mouse"][rec["id"]] = rec
                mark_experience_pool_unsaved()
    except Exception as e:
        logging.error("鼠标事件处理错误：%s", e)

def inactivity_monitor():
    global mode
    inactivity_threshold = global_config.get("inactivity_duration_threshold", 10)
    while True:
        try:
            if not window_active and mode == "learning" and (time.time() - last_activity_time >= inactivity_threshold):
                mode = "training"
                logging.info("检测到%d秒无操作，切换至训练模式。", inactivity_threshold)
            time.sleep(global_config.get("inactivity_monitor_interval", 1))
        except Exception as e:
            logging.error("无操作监控错误：%s", e)
            time.sleep(global_config.get("inactivity_monitor_interval", 1))

# ----------------------- AI Decision and Operation Functions -----------------------
def ai_keyboard_decision():
    try:
        with experience_pool_locks["screenshot"]:
            if latest_screenshot_ts is not None:
                screenshot_bytes = experience_pool["screenshot"].get(latest_screenshot_ts, None)
            else:
                return []
        if screenshot_bytes is None:
            return []
        img_tensor = preprocess_screenshot(screenshot_bytes).unsqueeze(0)
        with experience_pool_locks["text"]:
            if latest_text_ts is not None:
                text_record = experience_pool["text"].get(latest_text_ts, {"text": ""})
                text_data = text_record.get("text", "")
            else:
                text_data = ""
        if mode == "training":
            ai_actions = get_recent_ai_actions()
            text_data = text_data + " " + ai_actions
        text_tensor = preprocess_text(text_data)
        if text_tensor.dim() == 1:
            text_tensor = text_tensor.unsqueeze(0)
        with keyboard_ai_lock:
            keyboard_ai_model.eval()
            with torch.no_grad():
                keyboard_logits = keyboard_ai_model(img_tensor, text_tensor)
        threshold = max(global_config.get("ai_decision_threshold_factor") * (1 - ai_reward_factor),
                        global_config.get("ai_decision_threshold_min"))
        keyboard_probs = torch.softmax(keyboard_logits, dim=1).squeeze(0)
        keys_to_press = []
        default_keys = global_config.get("default_keyboard_keys", [])
        for i, prob in enumerate(keyboard_probs):
            if prob.item() > threshold and i < len(default_keys):
                keys_to_press.append(default_keys[i])
        operations = []
        if keys_to_press:
            operations.append({"type": "keyboard", "keys": keys_to_press, "timestamp": time.time_ns()})
        logging.info("键盘AI决策输出：%s", keys_to_press)
        return operations
    except Exception as e:
        logging.error("键盘AI决策错误：%s", e)
        return []

def ai_mouse_decision():
    try:
        with experience_pool_locks["screenshot"]:
            if latest_screenshot_ts is not None:
                screenshot_bytes = experience_pool["screenshot"].get(latest_screenshot_ts, None)
            else:
                return []
        if screenshot_bytes is None:
            return []
        img_tensor = preprocess_screenshot(screenshot_bytes).unsqueeze(0)
        with experience_pool_locks["text"]:
            if latest_text_ts is not None:
                text_record = experience_pool["text"].get(latest_text_ts, {"text": ""})
                text_data = text_record.get("text", "")
            else:
                text_data = ""
        if mode == "training":
            ai_actions = get_recent_ai_actions()
            text_data = text_data + " " + ai_actions
        text_tensor = preprocess_text(text_data)
        if text_tensor.dim() == 1:
            text_tensor = text_tensor.unsqueeze(0)
        with mouse_ai_lock:
            mouse_ai_model.eval()
            with torch.no_grad():
                mouse_output = mouse_ai_model(img_tensor, text_tensor)
        threshold = max(global_config.get("ai_decision_threshold_factor") * (1 - ai_reward_factor),
                        global_config.get("ai_decision_threshold_min"))
        operations = []
        mouse_op_logits = mouse_output["op_logits"]
        if mouse_op_logits.size(0) == 1:
            mouse_op_logits = mouse_op_logits[0]
        num_mouse_ops = mouse_op_logits.size(0)
        screen_width, screen_height = pyautogui.size()
        logging.info("鼠标AI决策：当前屏幕尺寸为 %d x %d", screen_width, screen_height)
        for i in range(num_mouse_ops):
            op_probs = torch.softmax(mouse_op_logits[i], dim=0)
            if op_probs.max().item() > threshold:
                op_index = torch.argmax(op_probs).item()
                op_names = ["none", "click", "long_press", "drag"]
                op_name = op_names[op_index]
                if op_name != "none":
                    if op_name == "drag":
                        offsets = mouse_output["drag_offsets"]
                        if offsets.size(0) == 1:
                            offsets = offsets[0]
                        offsets_list = offsets[i].detach().cpu().numpy().tolist()
                        # 使用屏幕中心作为起始位置
                        start_pos = (screen_width // 2, screen_height // 2)
                        mouse_op = {"type": "mouse", "operation": "drag", "button": "left",
                                    "start_position": start_pos,
                                    "drag_sequence": offsets_list,
                                    "timestamp": time.time_ns()}
                    else:
                        start_pos = (screen_width // 2, screen_height // 2)
                        mouse_op = {"type": "mouse", "operation": op_name, "button": "left",
                                    "start_position": start_pos,
                                    "timestamp": time.time_ns()}
                    operations.append(mouse_op)
        logging.info("鼠标AI决策输出：%s", operations)
        return operations
    except Exception as e:
        logging.error("鼠标AI决策错误：%s", e)
        return []

def ai_decision():
    keyboard_ops = ai_keyboard_decision()
    mouse_ops = ai_mouse_decision()
    combined_ops = keyboard_ops + mouse_ops
    sensory_input = []
    with experience_pool_locks["text"]:
        for ts, rec in experience_pool["text"].items():
            sensory_input.append(rec)
    enhanced_operations = cognitive_architecture.process_decision(combined_ops, sensory_input)
    return enhanced_operations

def perform_keyboard_operation(keys, source="user"):
    try:
        max_keys = global_config.get("max_parallel_keyboard_ops", 3)
        if len(keys) > max_keys:
            keys = keys[:max_keys]
        keyboard_press_duration = global_config.get("keyboard_press_duration", 0.05)
        for key in keys:
            down_timestamp = time.time_ns()
            pyautogui.keyDown(key)
            time.sleep(keyboard_press_duration)
            pyautogui.keyUp(key)
            up_timestamp = time.time_ns()
            record = {
                "key_name": key,
                "down_timestamp": down_timestamp,
                "up_timestamp": up_timestamp,
                "source": source
            }
            with experience_pool_locks["keyboard"]:
                experience_pool["keyboard"][down_timestamp] = record
            mark_experience_pool_unsaved()
        if source == "ai":
            logging.info("AI执行了键盘操作：%s", keys)
    except Exception as e:
        logging.error("键盘操作错误：%s", e)

def perform_mouse_operation(operation, source="user"):
    try:
        if operation["operation"] == "click":
            pyautogui.mouseDown()
            time.sleep(global_config.get("mouse_click_duration", 0.05))
            pyautogui.mouseUp()
        elif operation["operation"] == "long_press":
            pyautogui.mouseDown()
            time.sleep(global_config.get("long_press_duration", 1))
            pyautogui.mouseUp()
        elif operation["operation"] == "drag":
            screen_width, screen_height = pyautogui.size()
            image_size = global_config.get("image_size", 224)
            scale_x = screen_width / image_size
            scale_y = screen_height / image_size
            start_pos = operation.get("start_position", (screen_width//2, screen_height//2))
            drag_seq = operation.get("drag_sequence", None)
            if drag_seq is not None:
                if not drag_seq:
                    operation["operation"] = "click"
                    pyautogui.mouseDown()
                    time.sleep(global_config.get("mouse_click_duration", 0.05))
                    pyautogui.mouseUp()
                else:
                    pyautogui.moveTo(start_pos[0], start_pos[1])
                    pyautogui.mouseDown()
                    current_pos = start_pos
                    seq_len = len(drag_seq)
                    duration_per_step = global_config.get("drag_duration", 0.5) / seq_len
                    for offset in drag_seq:
                        offset_x = int(round(offset[0] * scale_x))
                        offset_y = int(round(offset[1] * scale_y))
                        next_pos = (min(max(0, current_pos[0] + offset_x), screen_width - 1),
                                    min(max(0, current_pos[1] + offset_y), screen_height - 1))
                        pyautogui.dragTo(next_pos[0], next_pos[1], duration=duration_per_step)
                        current_pos = next_pos
                    pyautogui.mouseUp()
            else:
                end_pos = (min(max(0, start_pos[0] + global_config.get("drag_distance", 100)), screen_width - 1),
                           min(max(0, start_pos[1] + global_config.get("drag_distance", 100)), screen_height - 1))
                pyautogui.moveTo(start_pos[0], start_pos[1])
                pyautogui.mouseDown()
                pyautogui.dragTo(end_pos[0], end_pos[1], duration=global_config.get("drag_duration", 0.5))
                pyautogui.mouseUp()
        down_timestamp = time.time_ns()
        up_timestamp = time.time_ns()
        record = {
            "operation": operation["operation"],
            "button": operation.get("button", "left"),
            "down_timestamp": down_timestamp,
            "down_position": operation.get("start_position", pyautogui.position()),
            "up_timestamp": up_timestamp,
            "up_position": operation.get("end_position", pyautogui.position()),
            "movement_track": [],
            "source": source
        }
        with experience_pool_locks["mouse"]:
            experience_pool["mouse"][down_timestamp] = record
        mark_experience_pool_unsaved()
        if source == "ai":
            logging.info("AI执行了鼠标操作：%s，起始位置：%s，屏幕尺寸：%s", operation["operation"], operation.get("start_position"), pyautogui.size())
    except Exception as e:
        logging.error("鼠标操作错误（%s）：%s", operation, e)

def evaluate_model(model, samples, loss_fn):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for img_tensor, txt_tensor, target_tensor in samples:
            if txt_tensor.dim() == 1:
                txt_tensor = txt_tensor.unsqueeze(0)
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            if hasattr(model, 'keyboard_head'):
                logits = model(img_tensor, txt_tensor)
            else:
                mouse_output = model(img_tensor, txt_tensor)
                logits = mouse_output["op_logits"][:, 0, :]
            if target_tensor.dim() == 1:
                target_tensor = target_tensor.unsqueeze(0)
            if target_tensor.shape[-1] < logits.shape[-1]:
                pad_size = logits.shape[-1] - target_tensor.shape[-1]
                target_tensor = nn.functional.pad(target_tensor, (0, pad_size), "constant", 0)
            elif target_tensor.shape[-1] > logits.shape[-1]:
                target_tensor = target_tensor[..., :logits.shape[-1]]
            loss = loss_fn(logits, target_tensor)
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else float("inf")

class OfflineOptimizationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    images, texts, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)
    return images, texts, targets

def offline_optimize_models():
    global optimization_in_progress, mode, keyboard_ai_model, mouse_ai_model, ai_reward_factor, adaptive_loss_scale
    try:
        optimization_in_progress = True
        logging.info("开始对两个模型进行离线优化...")
        keyboard_model_filename = global_config["required_files"]["keyboard_ai_model"]
        keyboard_model_path = os.path.join(script_directory, keyboard_model_filename)
        keyboard_backup_path = os.path.join(script_directory, "keyboard_ai_model_backup.pkl")
        shutil.copyfile(keyboard_model_path, keyboard_backup_path)
        mouse_model_filename = global_config["required_files"]["mouse_ai_model"]
        mouse_model_path = os.path.join(script_directory, mouse_model_filename)
        mouse_backup_path = os.path.join(script_directory, "mouse_ai_model_backup.pkl")
        shutil.copyfile(mouse_model_path, mouse_backup_path)
        
        with keyboard_ai_lock:
            keyboard_model = torch.load(keyboard_model_path)
        with mouse_ai_lock:
            mouse_model = torch.load(mouse_model_path)
        
        keyboard_model.train()
        mouse_model.train()
        
        default_keys = global_config.get("default_keyboard_keys", [])
        num_keyboard = len(default_keys)
        keyboard_samples = []
        skipped_keyboard_samples = 0
        with experience_pool_locks["keyboard"]:
            keyboard_records = list(experience_pool["keyboard"].items())
        with experience_pool_locks["screenshot"]:
            all_screenshots = {k: v for k, v in experience_pool["screenshot"].items()}
        with experience_pool_locks["text"]:
            all_texts = {k: v for k, v in experience_pool["text"].items()}
        base_max_age = 18000000000000
        total_records = len(keyboard_records)
        if total_records < 100:
            max_age_multiplier = 2.0
        elif total_records < 500:
            max_age_multiplier = 1.5
        elif total_records < 1000:
            max_age_multiplier = 1.0
        else:
            max_age_multiplier = 0.5
        dynamic_max_age = int(base_max_age * max_age_multiplier)
        for ts, record in keyboard_records:
            if record.get("source", "user") == "user":
                closest_screenshot = None
                pre_screenshots = {k: v for k, v in all_screenshots.items() if k <= ts}
                if pre_screenshots:
                    closest_screenshot = pre_screenshots[max(pre_screenshots.keys())]
                else:
                    post_screenshots = {k: v for k, v in all_screenshots.items() if k > ts}
                    if post_screenshots:
                        closest_screenshot = post_screenshots[min(post_screenshots.keys())]
                if closest_screenshot is None:
                    skipped_keyboard_samples += 1
                    continue
                ref_ts = max(pre_screenshots.keys()) if pre_screenshots else min(post_screenshots.keys())
                if abs(ts - ref_ts) > dynamic_max_age:
                    skipped_keyboard_samples += 1
                    continue
                img_tensor = preprocess_screenshot(closest_screenshot)
                closest_text = ""
                if all_texts:
                    pre_texts = {k: v for k, v in all_texts.items() if k <= ts}
                    if pre_texts:
                        closest_text = pre_texts[max(pre_texts.keys())].get("text", "")
                    else:
                        post_texts = {k: v for k, v in all_texts.items() if k > ts}
                        if post_texts:
                            closest_text = post_texts[min(post_texts.keys())].get("text", "")
                if not closest_text:
                    closest_text = " "
                txt_tensor = preprocess_text(closest_text)
                label = [0.0] * num_keyboard
                key_name = record.get("key_name", "")
                if key_name in default_keys:
                    index = default_keys.index(key_name)
                    if index < num_keyboard:
                        label[index] = 1.0
                target_tensor = torch.tensor(label, dtype=torch.float32)
                keyboard_samples.append((img_tensor, txt_tensor, target_tensor))
        logging.info("键盘样本数量：%d（跳过：%d）", len(keyboard_samples), skipped_keyboard_samples)
        
        default_mouse_ops = global_config.get("default_mouse_operations", ["click", "long_press", "drag"])
        mouse_samples = []
        skipped_mouse_samples = 0
        with experience_pool_locks["mouse"]:
            mouse_records = list(experience_pool["mouse"].items())
        for ts, record in mouse_records:
            if record.get("source", "user") == "user":
                operation = record.get("operation")
                if not operation:
                    skipped_mouse_samples += 1
                    continue
                supported_ops = set(default_mouse_ops)
                if operation not in supported_ops:
                    mapping = {"点击": "click", "长按": "long_press", "拖拽": "drag"}
                    if operation in mapping:
                        operation = mapping[operation]
                        record["operation"] = operation
                    else:
                        skipped_mouse_samples += 1
                        continue
                closest_screenshot = None
                pre_screenshots = {k: v for k, v in all_screenshots.items() if k <= ts}
                if pre_screenshots:
                    closest_screenshot = pre_screenshots[max(pre_screenshots.keys())]
                else:
                    post_screenshots = {k: v for k, v in all_screenshots.items() if k > ts}
                    if post_screenshots:
                        closest_screenshot = post_screenshots[min(post_screenshots.keys())]
                if closest_screenshot is None:
                    skipped_mouse_samples += 1
                    continue
                ref_ts = max(pre_screenshots.keys()) if pre_screenshots else min(post_screenshots.keys())
                if abs(ts - ref_ts) > dynamic_max_age:
                    skipped_mouse_samples += 1
                    continue
                img_tensor = preprocess_screenshot(closest_screenshot)
                closest_text = ""
                if all_texts:
                    pre_texts = {k: v for k, v in all_texts.items() if k <= ts}
                    if pre_texts:
                        closest_text = pre_texts[max(pre_texts.keys())].get("text", "")
                    else:
                        post_texts = {k: v for k, v in all_texts.items() if k > ts}
                        if post_texts:
                            closest_text = post_texts[min(post_texts.keys())].get("text", "")
                if not closest_text:
                    closest_text = " "
                txt_tensor = preprocess_text(closest_text)
                target = [0.0, 0.0, 0.0, 0.0]
                if record["operation"] == "drag" and not record.get("movement_track"):
                    record["operation"] = "click"
                    target = [0.0, 1.0, 0.0, 0.0]
                else:
                    if record["operation"] == "click":
                        target[1] = 1.0
                    elif record["operation"] == "long_press":
                        target[2] = 1.0
                    elif record["operation"] == "drag":
                        target[3] = 1.0
                    else:
                        target[0] = 1.0
                target_tensor = torch.tensor(target, dtype=torch.float32)
                mouse_samples.append((img_tensor, txt_tensor, target_tensor))
        logging.info("鼠标样本数量：%d（跳过：%d）", len(mouse_samples), skipped_mouse_samples)
        
        sample_thresh = global_config.get("offline_optimization_sample_threshold", 10)
        if len(keyboard_samples) < sample_thresh or len(mouse_samples) < sample_thresh:
            logging.info("经验数据不足（键盘样本：%d，鼠标样本：%d）；中止离线优化。", len(keyboard_samples), len(mouse_samples))
            optimization_in_progress = False
            mode = "learning"
            return
        
        dynamic_batch_size = min(
            max(16, int(min(len(keyboard_samples), len(mouse_samples)) / 10)),
            global_config.get("optimization_batch_size", 32)
        )
        logging.info("使用的批量大小：%d", dynamic_batch_size)
        
        keyboard_dataset = OfflineOptimizationDataset(keyboard_samples)
        mouse_dataset = OfflineOptimizationDataset(mouse_samples)
        
        keyboard_loader = DataLoader(keyboard_dataset, batch_size=dynamic_batch_size, shuffle=True, collate_fn=collate_fn)
        mouse_loader = DataLoader(mouse_dataset, batch_size=dynamic_batch_size, shuffle=True, collate_fn=collate_fn)
        
        loss_fn = nn.BCEWithLogitsLoss()
        dataset_size = len(keyboard_samples) + len(mouse_samples)
        weight_decay = 1e-4 * (100 / max(dataset_size, 100))
        logging.info("使用的权重衰减：%.8f", weight_decay)
        
        optimizer_keyboard = optim.Adam(keyboard_model.parameters(), lr=global_config.get("optimizer_lr", 0.001), weight_decay=weight_decay)
        optimizer_mouse = optim.Adam(mouse_model.parameters(), lr=global_config.get("optimizer_lr", 0.001), weight_decay=weight_decay)
        
        scheduler_keyboard = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_keyboard,
            factor=global_config.get("lr_decay_factor", 0.9),
            patience=global_config.get("lr_decay_patience", 5)
        )
        scheduler_mouse = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_mouse,
            factor=global_config.get("lr_decay_factor", 0.9),
            patience=global_config.get("lr_decay_patience", 5)
        )
        
        dynamic_epochs = min(
            max(50, int(1000 / max(dataset_size, 1))),
            global_config.get("optimization_epochs", 100)
        )
        logging.info("计划训练轮数：%d", dynamic_epochs)
        
        best_loss_keyboard = float("inf")
        best_loss_mouse = float("inf")
        best_state_keyboard = None
        best_state_mouse = None
        patience = global_config.get("offline_optimization_patience", 10)
        counter_keyboard = 0
        counter_mouse = 0
        
        for epoch in range(dynamic_epochs):
            keyboard_model.train()
            mouse_model.train()
            epoch_loss_keyboard = 0.0
            epoch_loss_mouse = 0.0
            batch_count = 0
            
            for images, texts, targets in keyboard_loader:
                optimizer_keyboard.zero_grad()
                keyboard_logits = keyboard_model(images, texts)
                loss = loss_fn(keyboard_logits, targets) * adaptive_loss_scale
                loss.backward()
                torch.nn.utils.clip_grad_norm_(keyboard_model.parameters(), max_norm=1.0)
                optimizer_keyboard.step()
                epoch_loss_keyboard += loss.item()
                batch_count += 1
            
            for images, texts, targets in mouse_loader:
                optimizer_mouse.zero_grad()
                mouse_output = mouse_model(images, texts)
                mouse_logits = mouse_output["op_logits"][:, 0, :]
                loss = loss_fn(mouse_logits, targets) * adaptive_loss_scale
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mouse_model.parameters(), max_norm=1.0)
                optimizer_mouse.step()
                epoch_loss_mouse += loss.item()
                batch_count += 1
            
            avg_loss_keyboard = epoch_loss_keyboard / (batch_count if batch_count else 1)
            avg_loss_mouse = epoch_loss_mouse / (batch_count if batch_count else 1)
            scheduler_keyboard.step(avg_loss_keyboard)
            scheduler_mouse.step(avg_loss_mouse)
            
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == dynamic_epochs - 1:
                logging.info("离线优化轮次 %d/%d，键盘平均损失：%.6f，鼠标平均损失：%.6f", epoch + 1, dynamic_epochs, avg_loss_keyboard, avg_loss_mouse)
            
            if avg_loss_keyboard < best_loss_keyboard:
                best_loss_keyboard = avg_loss_keyboard
                best_state_keyboard = {k: v.cpu().detach().clone() for k, v in keyboard_model.state_dict().items()}
                counter_keyboard = 0
            else:
                counter_keyboard += 1
            
            if avg_loss_mouse < best_loss_mouse:
                best_loss_mouse = avg_loss_mouse
                best_state_mouse = {k: v.cpu().detach().clone() for k, v in mouse_model.state_dict().items()}
                counter_mouse = 0
            else:
                counter_mouse += 1
            
            if counter_keyboard >= patience and counter_mouse >= patience:
                logging.info("在第%d轮提前停止，最佳键盘损失：%.6f，最佳鼠标损失：%.6f", epoch + 1 - min(counter_keyboard, counter_mouse), best_loss_keyboard, best_loss_mouse)
                break
        
        if best_state_keyboard is not None:
            keyboard_model.load_state_dict(best_state_keyboard)
            logging.info("加载了最佳键盘模型状态。")
        if best_state_mouse is not None:
            mouse_model.load_state_dict(best_state_mouse)
            logging.info("加载了最佳鼠标模型状态。")
        
        new_loss_keyboard = best_loss_keyboard
        new_loss_mouse = best_loss_mouse
        with keyboard_ai_lock:
            backup_keyboard = torch.load(keyboard_backup_path)
        with mouse_ai_lock:
            backup_mouse = torch.load(mouse_backup_path)
        eval_sample_count = min(len(keyboard_samples) + len(mouse_samples), 1000)
        eval_samples = (keyboard_samples + mouse_samples)[:eval_sample_count]
        backup_loss_keyboard = evaluate_model(backup_keyboard, eval_samples, loss_fn)
        backup_loss_mouse = evaluate_model(backup_mouse, eval_samples, loss_fn)
        meta_avg = cognitive_architecture.monitor_self((new_loss_keyboard + new_loss_mouse)/2)
        adjustment = cognitive_architecture.adjust_self((new_loss_keyboard + new_loss_mouse)/2)
        
        improvement_threshold = 0.01
        if new_loss_keyboard < backup_loss_keyboard * (1 - improvement_threshold) and new_loss_mouse < backup_loss_mouse * (1 - improvement_threshold):
            ai_reward_factor = min(ai_reward_factor + 0.1, 0.5)
            adaptive_loss_scale = min(adaptive_loss_scale * 1.05, 2.0)
            keyboard_model.version += 1
            mouse_model.version += 1
            with keyboard_ai_lock:
                torch.save(keyboard_model, keyboard_model_path)
                global keyboard_ai_model
                keyboard_ai_model = keyboard_model
            with mouse_ai_lock:
                torch.save(mouse_model, mouse_model_path)
                global mouse_ai_model
                mouse_ai_model = mouse_model
            logging.info("离线优化成功：键盘新损失 %.6f，鼠标新损失 %.6f，改进：键盘 %.2f%%，鼠标 %.2f%%", best_loss_keyboard, best_loss_mouse,
                         (backup_loss_keyboard-new_loss_keyboard)/backup_loss_keyboard*100, (backup_loss_mouse-new_loss_mouse)/backup_loss_mouse*100)
            logging.info("键盘模型版本：%d，鼠标模型版本：%d，奖励因子：%.2f，自适应损失比例：%.2f", keyboard_model.version, mouse_model.version, ai_reward_factor, adaptive_loss_scale)
            logging.info("元认知平均值：%.6f，调整：%s", meta_avg, adjustment)
        else:
            ai_reward_factor = max(ai_reward_factor - 0.05, 0.0)
            adaptive_loss_scale = max(adaptive_loss_scale * 0.95, 0.5)
            shutil.copyfile(keyboard_backup_path, keyboard_model_path)
            shutil.copyfile(mouse_backup_path, mouse_model_path)
            with keyboard_ai_lock:
                keyboard_ai_model = torch.load(keyboard_model_path)
            with mouse_ai_lock:
                mouse_ai_model = torch.load(mouse_model_path)
            backup_version_keyboard = backup_keyboard.version if hasattr(backup_keyboard, "version") else 1
            backup_version_mouse = backup_mouse.version if hasattr(backup_mouse, "version") else 1
            logging.info("离线优化未达到预期：键盘新损失 %.6f 对比备份损失 %.6f，鼠标新损失 %.6f 对比备份损失 %.6f", new_loss_keyboard, backup_loss_keyboard, new_loss_mouse, backup_loss_mouse)
            logging.info("已回退至键盘版本 %d，鼠标版本 %d，奖励因子：%.2f，自适应损失比例：%.2f", backup_version_keyboard, backup_version_mouse, ai_reward_factor, adaptive_loss_scale)
        
        best_state_keyboard = None
        best_state_mouse = None
        keyboard_samples = None
        mouse_samples = None
        gc.collect()
        
        optimization_in_progress = False
        mode = "learning"
        logging.info("两个模型的离线优化已完成。")
    except Exception as e:
        logging.error("离线优化错误：%s", e)
        logging.exception("详细错误信息：")
        optimization_in_progress = False
        mode = "learning"

def ai_online_learning_loop():
    global keyboard_ai_model, mouse_ai_model
    while True:
        try:
            if mode == "training" and not optimization_in_progress and not window_active:
                samples = []
                now_ns = time.time_ns()
                max_screenshot_age = global_config.get("max_screenshot_age_ns", 5000000000)
                with experience_pool_locks["keyboard"]:
                    keyboard_items = list(experience_pool["keyboard"].items())
                default_keys = global_config.get("default_keyboard_keys", [])
                num_keys = len(default_keys)
                for ts, record in keyboard_items:
                    if record.get("source", "user") == "user" and now_ns - ts < max_screenshot_age:
                        with experience_pool_locks["screenshot"]:
                            valid_screenshots = {k: v for k, v in experience_pool["screenshot"].items() if k <= ts}
                        if not valid_screenshots:
                            continue
                        latest_ss = max(valid_screenshots.keys())
                        if ts - latest_ss > max_screenshot_age:
                            continue
                        img_bytes = valid_screenshots[latest_ss]
                        img_tensor = preprocess_screenshot(img_bytes)
                        with experience_pool_locks["text"]:
                            valid_texts = {k: v for k, v in experience_pool["text"].items() if k <= ts}
                        if valid_texts:
                            text_data = valid_texts[max(valid_texts.keys())].get("text", "")
                        else:
                            text_data = ""
                        txt_tensor = preprocess_text(text_data)
                        label = [0] * num_keys
                        if record["key_name"] in default_keys:
                            index = default_keys.index(record["key_name"])
                            label[index] = 1
                        target_tensor = torch.tensor(label, dtype=torch.float32)
                        samples.append((img_tensor, txt_tensor, target_tensor))
                        if len(samples) >= global_config.get("online_optimization_batch_size", 16):
                            break
                if len(samples) > 0:
                    loss_fn = nn.BCEWithLogitsLoss()
                    images = torch.stack([s[0] for s in samples], dim=0)
                    texts = pad_sequence([s[1] for s in samples], batch_first=True, padding_value=0)
                    targets = torch.stack([s[2] for s in samples], dim=0)
                    with keyboard_ai_lock:
                        keyboard_ai_model.train()
                        optimizer = optim.Adam(keyboard_ai_model.parameters(), lr=global_config.get("online_optimizer_lr", 1e-4))
                        optimizer.zero_grad()
                        keyboard_logits = keyboard_ai_model(images, texts)
                        loss = loss_fn(keyboard_logits, targets)
                        loss.backward()
                        optimizer.step()
                        logging.info("在线优化更新了键盘模型，损失：%.6f", loss.item())
                samples_mouse = []
                with experience_pool_locks["mouse"]:
                    mouse_items = list(experience_pool["mouse"].items())
                default_mouse_ops = global_config.get("default_mouse_operations", ["click", "long_press", "drag"])
                for ts, record in mouse_items:
                    if record.get("source", "user") == "user" and now_ns - ts < max_screenshot_age:
                        with experience_pool_locks["screenshot"]:
                            valid_screenshots = {k: v for k, v in experience_pool["screenshot"].items() if k <= ts}
                        if not valid_screenshots:
                            continue
                        latest_ss = max(valid_screenshots.keys())
                        if ts - latest_ss > max_screenshot_age:
                            continue
                        img_bytes = valid_screenshots[latest_ss]
                        img_tensor = preprocess_screenshot(img_bytes)
                        with experience_pool_locks["text"]:
                            valid_texts = {k: v for k, v in experience_pool["text"].items() if k <= ts}
                        if valid_texts:
                            text_data = valid_texts[max(valid_texts.keys())].get("text", "")
                        else:
                            text_data = ""
                        txt_tensor = preprocess_text(text_data)
                        target = [0.0, 0.0, 0.0, 0.0]
                        if record["operation"] == "click":
                            target[1] = 1.0
                        elif record["operation"] == "long_press":
                            target[2] = 1.0
                        elif record["operation"] == "drag":
                            target[3] = 1.0
                        else:
                            target[0] = 1.0
                        target_tensor = torch.tensor(target, dtype=torch.float32)
                        samples_mouse.append((img_tensor, txt_tensor, target_tensor))
                        if len(samples_mouse) >= global_config.get("online_optimization_batch_size", 16):
                            break
                if len(samples_mouse) > 0:
                    loss_fn = nn.BCEWithLogitsLoss()
                    images = torch.stack([s[0] for s in samples_mouse], dim=0)
                    texts = pad_sequence([s[1] for s in samples_mouse], batch_first=True, padding_value=0)
                    targets = torch.stack([s[2] for s in samples_mouse], dim=0)
                    with mouse_ai_lock:
                        mouse_ai_model.train()
                        optimizer = optim.Adam(mouse_ai_model.parameters(), lr=global_config.get("online_optimizer_lr", 1e-4))
                        optimizer.zero_grad()
                        mouse_output = mouse_ai_model(images, texts)
                        mouse_logits = mouse_output["op_logits"][:, 0, :]
                        loss = loss_fn(mouse_logits, targets)
                        loss.backward()
                        optimizer.step()
                        logging.info("在线优化更新了鼠标模型，损失：%.6f", loss.item())
            time.sleep(global_config.get("online_optimization_interval", 10))
        except Exception as e:
            logging.error("在线优化错误：%s", e)
            time.sleep(global_config.get("online_optimization_interval", 10))

def show_input_window():
    global window_active, user_input_text, mode, tk_root, latest_text_ts
    try:
        window_active = True
        screen_width = tk_root.winfo_screenwidth()
        screen_height = tk_root.winfo_screenheight()
        initial_width_ratio = global_config.get("input_window_initial_width_ratio", 0.3)
        initial_height_ratio = global_config.get("input_window_initial_height_ratio", 0.1)
        min_width = global_config.get("input_window_min_width", 300)
        min_height = global_config.get("input_window_min_height", 100)
        win_width = max(int(screen_width * initial_width_ratio), min_width)
        win_height = max(int(screen_height * initial_height_ratio), min_height)
        pos_x = int((screen_width - win_width) / 2)
        pos_y = int((screen_height - win_height) / 2)
        geometry_str = f"{win_width}x{win_height}+{pos_x}+{pos_y}"
        input_win = tkinter.Toplevel(tk_root)
        input_win.title("Input Window")
        input_win.geometry(geometry_str)
        input_win.resizable(True, True)
        input_win.grab_set()
        input_win.rowconfigure(0, weight=1)
        input_win.rowconfigure(1, weight=0)
        input_win.columnconfigure(0, weight=1)
        text_widget = tkinter.Text(input_win, wrap="word")
        text_widget.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        button = tkinter.Button(input_win, text="Confirm", command=lambda: on_confirm(input_win, text_widget, button))
        button.grid(row=1, column=0, sticky="ew", padx=10, pady=(0,10))
        def update_button_state():
            if optimization_in_progress:
                button.config(state="disabled")
            else:
                button.config(state="normal")
            input_win.after(global_config.get("input_window_update_interval_ms", 10), update_button_state)
        update_button_state()
        input_win.wait_window()
        window_active = False
    except Exception as e:
        logging.error("输入窗口错误：%s", e)
        window_active = False

def on_confirm(window, text_widget, button):
    global user_input_text, mode, latest_text_ts, keyboard_ai_model, mouse_ai_model
    try:
        confirm_timestamp = time.time_ns()
        user_input_text = text_widget.get("1.0", "end-1c")
        with experience_pool_locks["text"]:
            experience_pool["text"][confirm_timestamp] = {"text": user_input_text, "source": "user"}
            latest_text_ts = confirm_timestamp
        mark_experience_pool_unsaved()
        keyboard_model_filename = global_config["required_files"]["keyboard_ai_model"]
        keyboard_model_path = os.path.join(script_directory, keyboard_model_filename)
        with keyboard_ai_lock:
            keyboard_ai_model = torch.load(keyboard_model_path)
        mouse_model_filename = global_config["required_files"]["mouse_ai_model"]
        mouse_model_path = os.path.join(script_directory, mouse_model_filename)
        with mouse_ai_lock:
            mouse_ai_model = torch.load(mouse_model_path)
        mode = "learning"
        window.destroy()
    except Exception as e:
        logging.error("确认按钮错误：%s", e)

def on_hotkey_trigger():
    try:
        if not optimization_in_progress:
            threading.Thread(target=offline_optimize_models, daemon=True).start()
        tk_queue.put("show_input")
    except Exception as e:
        logging.error("热键触发错误：%s", e)

def esc_monitor_loop():
    while True:
        try:
            if keyboard.is_pressed("esc"):
                start = time.time()
                while keyboard.is_pressed("esc"):
                    if time.time() - start >= global_config.get("esc_termination_duration", 3):
                        os._exit(0)
                    time.sleep(global_config.get("esc_monitor_interval", 0.1))
            time.sleep(global_config.get("esc_monitor_interval", 0.1))
        except Exception as e:
            logging.error("Esc监控错误：%s", e)
            time.sleep(global_config.get("esc_monitor_interval", 0.1))

def process_tk_queue():
    try:
        while not tk_queue.empty():
            cmd = tk_queue.get_nowait()
            if cmd == "show_input":
                show_input_window()
    except Exception as e:
        logging.error("Tk队列处理错误：%s", e)
    finally:
        tk_root.after(global_config.get("tk_queue_interval_ms", 100), process_tk_queue)

def ai_training_loop():
    while True:
        try:
            if mode == "training" and not optimization_in_progress and not window_active:
                ops = ai_decision()
                if ops:
                    for op in ops:
                        if op["type"] == "keyboard":
                            keys = op["keys"]
                            max_keys = global_config.get("max_parallel_keyboard_ops", 3)
                            if len(keys) > max_keys:
                                keys = keys[:max_keys]
                            perform_keyboard_operation(keys, source="ai")
                        elif op["type"] == "mouse":
                            threading.Thread(target=perform_mouse_operation, args=(op, "ai"), daemon=True).start()
                        elif op["type"] == "symbolic":
                            logging.info("AI符号决策：%s", op["operation"])
                        elif op["type"] == "world_state":
                            logging.info("当前世界状态：%s", op["state"])
                        elif op["type"] == "causal":
                            logging.info("因果链：%s", op["chain"])
                        elif op["type"] == "goal":
                            logging.info("自主目标：%s", op["goal"])
                        elif op["type"] == "creative":
                            logging.info("创造性解决方案：%s", op["solution"])
                        elif op["type"] == "transfer":
                            logging.info("迁移学习状态：%s", op["status"])
                    for op in ops:
                        timestamp = time.time_ns()
                        cognitive_architecture.update_memory(timestamp, op)
                time.sleep(global_config.get("ai_training_loop_interval", 0.1))
            else:
                time.sleep(global_config.get("ai_training_loop_interval", 0.1))
        except Exception as e:
            logging.error("AI训练循环错误：%s", e)
            time.sleep(global_config.get("ai_training_loop_interval", 0.1))

# ----------------------- Cognitive Architecture Modules -----------------------
class AdvancedMemorySystem:
    def __init__(self):
        self.working_memory = {}
        self.short_term_memory = {}
        self.long_term_memory = {}
        self.semantic_memory = {}
        self.memory_lock = threading.Lock()
        self.working_capacity = 1000
        self.short_term_capacity = 5000
        self.long_term_capacity = 20000
    def store_experience(self, timestamp, experience):
        with self.memory_lock:
            self.working_memory[timestamp] = experience
            if 'text' in experience:
                key = experience['text']
                self.semantic_memory[key] = experience
            if len(self.working_memory) > self.working_capacity:
                for ts in sorted(self.working_memory.keys())[:len(self.working_memory) - self.working_capacity]:
                    exp = self.working_memory.pop(ts)
                    self.short_term_memory[ts] = exp
            if len(self.short_term_memory) > self.short_term_capacity:
                for ts in sorted(self.short_term_memory.keys())[:len(self.short_term_memory) - self.short_term_capacity]:
                    exp = self.short_term_memory.pop(ts)
                    self.long_term_memory[ts] = exp
            if len(self.long_term_memory) > self.long_term_capacity:
                keys = sorted(self.long_term_memory.keys())
                for ts in keys[:len(self.long_term_memory) - self.long_term_capacity]:
                    del self.long_term_memory[ts]
    def retrieve_memory(self, query):
        with self.memory_lock:
            results = []
            for key, exp in self.semantic_memory.items():
                if query in key:
                    results.append(exp)
            return results
    def consolidate_memory(self):
        with self.memory_lock:
            for ts, exp in list(self.working_memory.items()):
                self.short_term_memory[ts] = exp
            self.working_memory.clear()

class NeuroSymbolicHybridSystem:
    def __init__(self):
        self.symbolic_rules = {"click": "execute_click", "drag": "execute_drag", "long_press": "execute_long_press"}
        self.feedback_rules = {}
        self.lock = threading.Lock()
    def symbolic_reasoning(self, neural_output):
        refined_output = neural_output.copy()
        for op in neural_output:
            if op["type"] == "keyboard":
                for key in op["keys"]:
                    if key in self.symbolic_rules:
                        refined_output.append({"type": "symbolic", "operation": self.symbolic_rules[key], "key": key, "timestamp": op["timestamp"]})
            elif op["type"] == "mouse":
                if op["operation"] in self.symbolic_rules:
                    refined_output.append({"type": "symbolic", "operation": self.symbolic_rules[op["operation"]], "timestamp": op["timestamp"]})
        return refined_output
    def provide_feedback(self, neural_loss):
        with self.lock:
            if neural_loss > 1.0:
                self.feedback_rules["adjustment"] = "increase_regularization"
            else:
                self.feedback_rules["adjustment"] = "decrease_regularization"
            return self.feedback_rules
    def integrate_symbolic_into_neural(self, neural_model):
        with self.lock:
            adjustment = self.feedback_rules.get("adjustment", None)
            if adjustment == "increase_regularization":
                for param in neural_model.parameters():
                    param.data = param.data * 0.99
            elif adjustment == "decrease_regularization":
                for param in neural_model.parameters():
                    param.data = param.data * 1.01
        return neural_model

class WorldModel:
    def __init__(self):
        self.state = {"time": time.time(), "environment": "default", "context": {}}
        self.lock = threading.Lock()
    def simulate_action(self, action):
        with self.lock:
            if action["type"] == "keyboard":
                self.state["last_keyboard"] = action["keys"]
            elif action["type"] == "mouse":
                self.state["last_mouse"] = action["operation"]
            self.state["time"] = time.time()
            return self.state
    def get_current_state(self):
        with self.lock:
            return self.state.copy()
    def update_context(self, key, value):
        with self.lock:
            self.state["context"][key] = value

class CausalGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
    def add_event(self, event_id, event):
        self.nodes[event_id] = event
    def add_causal_link(self, cause_id, effect_id, weight=1.0):
        self.edges[(cause_id, effect_id)] = weight
    def get_effects(self, cause_id):
        effects = []
        for (c, e), w in self.edges.items():
            if c == cause_id:
                effects.append((self.nodes.get(e, None), w))
        return effects
    def counterfactual_reasoning(self, event_id):
        if event_id in self.nodes:
            return {"counterfactual": f"If event {event_id} had not occurred, the effect might be different."}
        return {}

class CausalReasoningSystem:
    def __init__(self):
        self.causal_graph = CausalGraph()
        self.lock = threading.Lock()
    def infer_causality(self, events):
        with self.lock:
            if len(events) >= 2:
                event_ids = []
                for idx, event in enumerate(events):
                    event_id = f"event_{idx}"
                    self.causal_graph.add_event(event_id, event)
                    event_ids.append(event_id)
                self.causal_graph.add_causal_link(event_ids[0], event_ids[-1], weight=1.0)
                chain = {"cause": events[0], "effect": events[-1]}
                return chain
            return {}
    def perform_counterfactual(self, event_id):
        with self.lock:
            return self.causal_graph.counterfactual_reasoning(event_id)

class MetaCognitiveModule:
    def __init__(self):
        self.performance_history = []
    def self_monitor(self, performance_metric):
        self.performance_history.append(performance_metric)
        avg = sum(self.performance_history[-10:]) / len(self.performance_history[-10:])
        return avg
    def adjust_parameters(self, current_loss):
        if current_loss > 1.0:
            return {"adjustment": "increase_learning_rate"}
        else:
            return {"adjustment": "decrease_learning_rate"}

class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
    def cooperate(self, decision):
        decision["agent_id"] = self.agent_id
        return decision

class MultiAgentSystem:
    def __init__(self, num_agents=3):
        self.agents = [Agent(i) for i in range(num_agents)]
    def aggregate_decisions(self, decisions):
        aggregated = []
        for agent in self.agents:
            for decision in decisions:
                agent_decision = agent.cooperate(decision.copy())
                aggregated.append(agent_decision)
        return aggregated

class SelfSupervisedActiveLearning:
    def __init__(self):
        self.active_learning_triggered = False
    def trigger_active_learning(self, memory_system, performance_metric):
        if performance_metric > 1.0:
            self.active_learning_triggered = True
            samples = []
            for ts, exp in memory_system.working_memory.items():
                if "text" in exp and len(exp["text"]) > 5:
                    samples.append(exp)
            return samples
        return []

class AutonomousGoalSystem:
    def __init__(self):
        self.goals = []
        self.intrinsic_motivation = 0.0
        self.lock = threading.Lock()
    def generate_goal(self, context):
        with self.lock:
            new_goal = {"goal": "Optimize Performance", "priority": 1, "timestamp": time.time_ns()}
            self.goals.append(new_goal)
            return new_goal
    def update_goal(self, feedback):
        with self.lock:
            for goal in self.goals:
                if "adjustment" in feedback.get("adjustment", ""):
                    goal["priority"] += 1
            return self.goals
    def get_current_goals(self):
        with self.lock:
            return self.goals.copy()

class CreativeThinkingModule:
    def __init__(self):
        self.lock = threading.Lock()
    def blend_concepts(self, concept1, concept2):
        with self.lock:
            return concept1 + "+" + concept2
    def reason_by_analogy(self, problem, analogy):
        with self.lock:
            return f"By analogy with {analogy}, solve {problem}"
    def evaluate_solution(self, solution):
        with self.lock:
            score = len(solution) % 10
            return score

class TransferLearningModule:
    def __init__(self):
        self.domain_models = {}
        self.lock = threading.Lock()
    def register_domain(self, domain_name, model_parameters):
        with self.lock:
            self.domain_models[domain_name] = model_parameters
    def transfer_learn(self, source_domain, target_domain):
        with self.lock:
            if source_domain in self.domain_models:
                self.domain_models[target_domain] = self.domain_models[source_domain].copy()
                return True
            return False

class CognitiveArchitecture:
    def __init__(self):
        self.memory_system = AdvancedMemorySystem()
        self.neuro_symbolic = NeuroSymbolicHybridSystem()
        self.world_model = WorldModel()
        self.causal_reasoning = CausalReasoningSystem()
        self.meta_cognitive = MetaCognitiveModule()
        self.multi_agent = MultiAgentSystem()
        self.active_learning = SelfSupervisedActiveLearning()
        self.autonomous_goal = AutonomousGoalSystem()
        self.creative_thinking = CreativeThinkingModule()
        self.transfer_learning = TransferLearningModule()
    def process_decision(self, original_decision, sensory_input):
        enhanced_decision = original_decision.copy()
        symbolic_decision = self.neuro_symbolic.symbolic_reasoning(original_decision)
        enhanced_decision.extend(symbolic_decision)
        state = self.world_model.get_current_state()
        enhanced_decision.append({"type": "world_state", "state": state, "timestamp": time.time_ns()})
        if len(sensory_input) > 0:
            causal = self.causal_reasoning.infer_causality(sensory_input)
            if causal:
                enhanced_decision.append({"type": "causal", "chain": causal, "timestamp": time.time_ns()})
        goal = self.autonomous_goal.generate_goal(state)
        enhanced_decision.append({"type": "goal", "goal": goal, "timestamp": time.time_ns()})
        if "problem" in state.get("context", {}):
            analogy = self.creative_thinking.reason_by_analogy(state["context"]["problem"], "historical cases")
            enhanced_decision.append({"type": "creative", "solution": analogy, "timestamp": time.time_ns()})
        transfer_success = self.transfer_learning.transfer_learn("DomainA", "DomainB")
        enhanced_decision.append({"type": "transfer", "status": "success" if transfer_success else "failure", "timestamp": time.time_ns()})
        aggregated = self.multi_agent.aggregate_decisions(enhanced_decision)
        return aggregated
    def update_memory(self, timestamp, experience):
        self.memory_system.store_experience(timestamp, experience)
    def perform_active_learning(self, performance_metric):
        return self.active_learning.trigger_active_learning(self.memory_system, performance_metric)
    def monitor_self(self, current_loss):
        return self.meta_cognitive.self_monitor(current_loss)
    def adjust_self(self, current_loss):
        adjustment = self.meta_cognitive.adjust_parameters(current_loss)
        self.neuro_symbolic.provide_feedback(current_loss)
        return adjustment

cognitive_architecture = CognitiveArchitecture()

def cognitive_architecture_active_loop():
    while True:
        try:
            current_loss = 0.5
            samples = cognitive_architecture.perform_active_learning(current_loss)
            if samples:
                logging.info("触发自监督主动学习，样本数：%d", len(samples))
            cognitive_architecture.memory_system.consolidate_memory()
            time.sleep(global_config.get("cognitive_active_interval", 30))
        except Exception as e:
            logging.error("认知架构主动循环错误：%s", e)
            time.sleep(global_config.get("cognitive_active_error_sleep", 5))

# ----------------------- Main Entry -----------------------
def main():
    check_and_create_files()
    threading.Thread(target=memory_monitor, daemon=True).start()
    threading.Thread(target=screenshot_capture_loop, daemon=True).start()
    threading.Thread(target=experience_pool_saver_loop, daemon=True).start()
    keyboard.hook(on_keyboard_event, suppress=False)
    mouse.hook(mouse_event_handler)
    threading.Thread(target=inactivity_monitor, daemon=True).start()
    threading.Thread(target=ai_training_loop, daemon=True).start()
    threading.Thread(target=esc_monitor_loop, daemon=True).start()
    threading.Thread(target=ai_online_learning_loop, daemon=True).start()
    threading.Thread(target=cognitive_architecture_active_loop, daemon=True).start()
    hotkey = global_config.get("hotkey", "ctrl+alt")
    keyboard.add_hotkey(hotkey, on_hotkey_trigger)
    global tk_root
    tk_root = tkinter.Tk()
    tk_root.withdraw()
    tk_root.after(global_config.get("tk_queue_interval_ms", 100), process_tk_queue)
    tk_root.mainloop()

if __name__ == "__main__":
    main()
