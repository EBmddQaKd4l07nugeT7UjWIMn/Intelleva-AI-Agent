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

# ----------------------- 全局变量定义 -----------------------
global_config = {}         # 配置字典
script_directory = ""      # 脚本所在目录

# 用于线程同步 AI 模型更新和调用
ai_model_lock = threading.Lock()

# 经验池：截图、键盘、鼠标、文本数据均以精确到纳秒的时间戳为键存储
experience_pool = {
    "screenshot": {},
    "keyboard": {},
    "mouse": {},
    "text": {}
}
# 各数据类型独立的线程锁
experience_pool_locks = {
    "screenshot": threading.Lock(),
    "keyboard": threading.Lock(),
    "mouse": threading.Lock(),
    "text": threading.Lock()
}
# 记录未保存修改次数及上次保存时间（用于批量保存机制）
experience_pool_unsaved_count = 0
experience_pool_last_save_time = time.time()
experience_pool_save_lock = threading.Lock()

# 记录最新的截图和文本数据时间戳（None表示尚未记录）
latest_screenshot_ts = None
latest_text_ts = None

mode = "learning"          # 当前模式，初始为“学习模式”
last_activity_time = time.time()  # 上一次用户操作时间（秒级，用于检测无操作时间）
window_active = False       # 输入窗口存在时置 True（仅用于 AI 输出判断，不影响数据采集）
optimization_in_progress = False  # 离线优化是否正在进行
user_input_text = ""       # 用户在输入窗口中输入的文本

# 当前正在进行的键盘事件（按下–保持–松起过程视为一条数据，支持同时多个按键）
current_keyboard_events = []
# 当前正在进行的鼠标事件（按下–保持–松起过程视为一条数据，支持多条数据）
current_mouse_events = {}

# Tkinter 主窗口及线程安全队列（用于调度 Tkinter 相关操作）
tk_root = None
tk_queue = queue.Queue()

# 全局 AI 奖励因子（用于调整 AI 输出的激进程度，范围0~0.5）
ai_reward_factor = 0.0

# ----------------------- 辅助函数 -----------------------
def get_event_position(event):
    """
    获取事件的屏幕位置坐标；若事件无此属性，则返回当前鼠标位置
    """
    try:
        return (event.x, event.y)
    except AttributeError:
        return pyautogui.position()

def get_recent_ai_actions():
    """
    获取最近10秒内记录的 AI 自己输出的键盘和鼠标操作，拼接为字符串
    """
    now_ns = time.time_ns()
    window_ns = 10 * 10**9  # 10秒（单位：纳秒）
    ai_actions = []
    with experience_pool_locks["keyboard"]:
        for ts, rec in experience_pool["keyboard"].items():
            if rec.get("source") == "ai" and now_ns - ts <= window_ns:
                up_ts = rec.get("up_timestamp", "未知")
                ai_actions.append(f"键盘:{rec['key_name']}下{rec['down_timestamp']}上{up_ts}")
    with experience_pool_locks["mouse"]:
        for ts, rec in experience_pool["mouse"].items():
            if rec.get("source") == "ai" and now_ns - ts <= window_ns:
                up_ts = rec.get("up_timestamp", "未知")
                ai_actions.append(f"鼠标:{rec['operation']}按钮{rec['button']}下{rec['down_timestamp']}上{up_ts}")
    return " ".join(ai_actions)

# ----------------------- AI 模型相关定义 -----------------------
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

class AIDecisionModel(nn.Module):
    def __init__(self, vocab_size, text_embed_dim, common_dim, text_transformer_layers, text_transformer_heads,
                 num_keyboard_keys, num_mouse_output, image_size, drag_sequence_length, pretrained_image=True):
        super(AIDecisionModel, self).__init__()
        self.image_encoder = ImageEncoder(common_dim, image_size, pretrained=pretrained_image)
        self.text_encoder = TextEncoder(vocab_size, text_embed_dim, common_dim,
                                        num_layers=text_transformer_layers, num_heads=text_transformer_heads)
        self.fusion = MultiModalFusion(common_dim, num_heads=text_transformer_heads)
        self.fc1 = nn.Linear(common_dim, common_dim)
        self.fc2 = nn.Linear(common_dim, common_dim // 2)
        self.keyboard_head = nn.Linear(common_dim // 2, num_keyboard_keys)
        # 对于鼠标操作，输出维度为 4（代表 none, click, long_press, drag）乘以最大并行输出数
        self.max_parallel_mouse_ops = num_mouse_output
        self.mouse_op_head = nn.Linear(common_dim // 2, 4 * num_mouse_output)
        # 拖拽序列生成器输出每个并行操作对应的拖拽轨迹，轨迹长度为 drag_sequence_length
        self.drag_sequence_generator = DragSequenceGenerator(common_dim // 2, common_dim // 2, drag_sequence_length * num_mouse_output)
        self.version = 1

    def forward(self, image, text):
        img_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        fused = self.fusion(img_features, text_features)
        x = torch.relu(self.fc1(fused))
        x = torch.relu(self.fc2(x))
        keyboard_logits = self.keyboard_head(x)
        raw_mouse_logits = self.mouse_op_head(x)
        mouse_op_logits = raw_mouse_logits.view(self.max_parallel_mouse_ops, 4)
        raw_drag_offsets = self.drag_sequence_generator(x)
        drag_offsets = raw_drag_offsets.view(self.max_parallel_mouse_ops, -1, 2)
        return keyboard_logits, {"op_logits": mouse_op_logits, "drag_offsets": drag_offsets}

# 全局 AI 模型变量
ai_model = None

def get_or_create_model():
    global ai_model, script_directory, global_config
    ai_model_filename = global_config["required_files"]["ai_model"]
    ai_model_path = os.path.join(script_directory, ai_model_filename)
    default_keys = global_config.get("default_keyboard_keys", [])
    keyboard_output_dim = len(default_keys)
    vocab_size = global_config.get("text_vocab_size", 5000)
    text_embed_dim = global_config.get("text_embed_dim", 128)
    common_dim = global_config.get("common_feature_dim", 256)
    text_transformer_layers = global_config.get("text_transformer_layers", 2)
    text_transformer_heads = global_config.get("text_transformer_heads", 4)
    image_size = global_config.get("image_size", 224)
    drag_sequence_length = global_config.get("drag_sequence_length", 10)
    num_mouse_output = global_config.get("max_parallel_mouse_ops", 3)
    with ai_model_lock:
        if not os.path.exists(ai_model_path):
            model = AIDecisionModel(vocab_size, text_embed_dim, common_dim, text_transformer_layers,
                                    text_transformer_heads, keyboard_output_dim, num_mouse_output, image_size,
                                    drag_sequence_length, pretrained_image=True)
            try:
                torch.save(model, ai_model_path)
            except Exception as e:
                logging.error("保存新模型失败：%s", e)
            ai_model = model
        else:
            try:
                ai_model = torch.load(ai_model_path)
            except Exception as e:
                logging.error("加载 AI 模型失败：%s", e)
                backup_model_path = os.path.join(script_directory, "ai_model_backup.pkl")
                if os.path.exists(backup_model_path):
                    try:
                        ai_model = torch.load(backup_model_path)
                        torch.save(ai_model, ai_model_path)
                        logging.info("使用备份模型恢复。")
                    except Exception as ex:
                        logging.error("加载备份模型失败：%s", ex)
                        ai_model = AIDecisionModel(vocab_size, text_embed_dim, common_dim,
                                                   text_transformer_layers, text_transformer_heads,
                                                   keyboard_output_dim, num_mouse_output, image_size,
                                                   drag_sequence_length, pretrained_image=True)
                        torch.save(ai_model, ai_model_path)
                else:
                    ai_model = AIDecisionModel(vocab_size, text_embed_dim, common_dim,
                                               text_transformer_layers, text_transformer_heads,
                                               keyboard_output_dim, num_mouse_output, image_size,
                                               drag_sequence_length, pretrained_image=True)
                    torch.save(ai_model, ai_model_path)

# ----------------------- 文件与内存管理 -----------------------
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
        logging.error("计算数据价值异常：%s", e)
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
        logging.error("经验池清理异常：%s", e)

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
            logging.error("经验池保存线程异常：%s", e)
            time.sleep(1)

def check_and_create_files():
    global global_config, script_directory, experience_pool
    try:
        script_directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_directory, "config.json")
        if not os.path.exists(config_path):
            default_config = {
                "required_files": {
                    "config": "config.json",
                    "ai_model": "ai_model.pkl",
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
                "tk_queue_interval_ms": 100,
                "ai_training_loop_interval": 0.1,
                "input_window_update_interval_ms": 10,
                "ai_decision_interval": 0.1,
                "mouse_drag_min_distance": 5,
                "online_optimizer_lr": 1e-4,
                "online_optimization_interval": 10,
                "online_optimization_batch_size": 16
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=4)
        with open(config_path, "r", encoding="utf-8") as f:
            global_config = json.load(f)
        get_or_create_model()
        exp_pool_filename = global_config["required_files"]["experience_pool"]
        exp_pool_path = os.path.join(script_directory, exp_pool_filename)
        if os.path.exists(exp_pool_path):
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
        else:
            mark_experience_pool_unsaved()
    except Exception as e:
        logging.error("文件检查与创建失败：%s", e)
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
            logging.error("内存监控异常：%s", e)
            time.sleep(1)

# ----------------------- 数据预处理函数 -----------------------
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
        logging.error("截图预处理异常：%s", e)
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
        logging.error("文本预处理异常：%s", e)
        return torch.tensor([], dtype=torch.long)

# ----------------------- 截图与事件采集 -----------------------
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
            logging.error("截图采集异常：%s", e)
            time.sleep(0.5)

def on_key_down(event):
    global last_activity_time, mode, current_keyboard_events
    try:
        last_activity_time = time.time()
        if mode != "learning":
            mode = "learning"
        key_name = event.name
        current_keyboard_events.append({
            "key_name": key_name,
            "down_timestamp": time.time_ns()
        })
    except Exception as e:
        logging.error("键盘按下事件异常：%s", e)

def on_key_up(event):
    global last_activity_time, current_keyboard_events
    try:
        last_activity_time = time.time()
        if mode != "learning":
            mode = "learning"
        key_name = event.name
        record_index = None
        for i, rec in enumerate(current_keyboard_events):
            if rec["key_name"] == key_name:
                record_index = i
                break
        if record_index is not None:
            rec = current_keyboard_events.pop(record_index)
            rec["up_timestamp"] = time.time_ns()
            rec["source"] = "user"
            with experience_pool_locks["keyboard"]:
                experience_pool["keyboard"][rec["down_timestamp"]] = rec
            mark_experience_pool_unsaved()
    except Exception as e:
        logging.error("键盘抬起事件异常：%s", e)

def mouse_event_handler(event):
    global last_activity_time, mode, current_mouse_events
    try:
        last_activity_time = time.time()
        if mode != "learning":
            mode = "learning"
        event_type = event.event_type if hasattr(event, "event_type") else "move"
        if event_type == "down":
            timestamp = time.time_ns()
            record = {
                "operation": None,
                "button": event.button,
                "down_timestamp": timestamp,
                "down_position": get_event_position(event),
                "up_timestamp": None,
                "up_position": None,
                "movement_track": []
            }
            current_mouse_events[event.button] = record
        elif event_type == "move":
            for btn, rec in current_mouse_events.items():
                rec["movement_track"].append({"timestamp": time.time_ns(), "position": get_event_position(event)})
        elif event_type == "up":
            if event.button in current_mouse_events:
                rec = current_mouse_events.pop(event.button)
                rec["up_timestamp"] = time.time_ns()
                rec["up_position"] = get_event_position(event)
                duration = (rec["up_timestamp"] - rec["down_timestamp"]) / 1e9
                dx = rec["up_position"][0] - rec["down_position"][0]
                dy = rec["up_position"][1] - rec["down_position"][1]
                distance = (dx**2 + dy**2)**0.5
                if distance > global_config.get("mouse_drag_min_distance", 5) and len(rec["movement_track"]) > 0:
                    rec["operation"] = "拖拽"
                elif duration >= global_config.get("long_press_duration", 1):
                    rec["operation"] = "长按"
                else:
                    rec["operation"] = "点击"
                rec["source"] = "user"
                with experience_pool_locks["mouse"]:
                    experience_pool["mouse"][rec["down_timestamp"]] = rec
                mark_experience_pool_unsaved()
    except Exception as e:
        logging.error("鼠标事件处理异常：%s", e)

def inactivity_monitor():
    global mode
    inactivity_threshold = global_config.get("inactivity_duration_threshold", 10)
    while True:
        try:
            if not window_active and mode == "learning" and (time.time() - last_activity_time >= inactivity_threshold):
                mode = "training"
                logging.info("检测到%d秒无操作，模式由学习切换为训练。", inactivity_threshold)
            time.sleep(1)
        except Exception as e:
            logging.error("无操作监控异常：%s", e)
            time.sleep(1)

def ai_decision():
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
        # 在训练模式下，将最近10秒内 AI 自己的键盘和鼠标操作拼接进文本数据
        if mode == "training":
            ai_actions = get_recent_ai_actions()
            text_data = text_data + " " + ai_actions
        text_tensor = preprocess_text(text_data)
        if text_tensor.dim() == 1:
            text_tensor = text_tensor.unsqueeze(0)
        with ai_model_lock:
            ai_model.eval()
            with torch.no_grad():
                keyboard_logits, mouse_output = ai_model(img_tensor, text_tensor)
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
        mouse_ops = []
        mouse_op_logits = mouse_output["op_logits"]
        drag_offsets = mouse_output["drag_offsets"]
        num_mouse_ops = mouse_op_logits.size(0)
        for i in range(num_mouse_ops):
            op_probs = torch.softmax(mouse_op_logits[i], dim=0)
            if op_probs.max().item() > threshold:
                op_index = torch.argmax(op_probs).item()
                op_names = ["none", "click", "long_press", "drag"]
                op_name = op_names[op_index]
                if op_name != "none":
                    if op_name == "drag":
                        offsets = drag_offsets[i]
                        offsets_list = offsets.detach().cpu().numpy().tolist()
                        mouse_op = {"type": "mouse", "operation": "drag", "button": "left",
                                    "start_position": pyautogui.position(),
                                    "drag_sequence": offsets_list,
                                    "timestamp": time.time_ns()}
                    else:
                        mouse_op = {"type": "mouse", "operation": op_name, "button": "left",
                                    "timestamp": time.time_ns()}
                    mouse_ops.append(mouse_op)
        if mouse_ops:
            operations.extend(mouse_ops)
        logging.info("AI决策输出 - 键盘: %s, 鼠标: %s", keys_to_press, mouse_ops)
        return operations
    except Exception as e:
        logging.error("AI决策异常：%s", e)
        return []

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
            logging.info("AI执行键盘操作: %s", keys)
    except Exception as e:
        logging.error("键盘操作执行异常：%s", e)

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
            start_pos = operation.get("start_position", pyautogui.position())
            drag_seq = operation.get("drag_sequence", None)
            if drag_seq is not None:
                pyautogui.moveTo(start_pos)
                pyautogui.mouseDown()
                current_pos = start_pos
                seq_len = len(drag_seq)
                duration_per_step = global_config.get("drag_duration", 0.5) / seq_len
                for offset in drag_seq:
                    next_pos = (current_pos[0] + int(round(offset[0])), current_pos[1] + int(round(offset[1])))
                    pyautogui.dragTo(next_pos[0], next_pos[1], duration=duration_per_step)
                    current_pos = next_pos
                pyautogui.mouseUp()
            else:
                end_pos = operation.get("end_position", (start_pos[0] + global_config.get("drag_distance", 100),
                                                         start_pos[1] + global_config.get("drag_distance", 100)))
                pyautogui.moveTo(start_pos)
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
            logging.info("AI执行鼠标操作: %s", operation["operation"])
    except Exception as e:
        logging.error("鼠标操作执行异常（%s）：%s", operation, e)

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
            target_tensor = target_tensor.unsqueeze(0) if target_tensor.dim() == 1 else target_tensor
            keyboard_logits, _ = model(img_tensor, txt_tensor)
            loss = loss_fn(keyboard_logits, target_tensor)
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

def ai_offline_optimization():
    global optimization_in_progress, mode, ai_model, ai_reward_factor
    try:
        optimization_in_progress = True
        ai_model_filename = global_config["required_files"]["ai_model"]
        ai_model_path = os.path.join(script_directory, ai_model_filename)
        backup_model_path = os.path.join(script_directory, "ai_model_backup.pkl")
        shutil.copyfile(ai_model_path, backup_model_path)
        with ai_model_lock:
            model = torch.load(ai_model_path)
        model.train()
        default_keys = global_config.get("default_keyboard_keys", [])
        num_keys = len(default_keys)
        samples = []
        max_screenshot_age = global_config.get("max_screenshot_age_ns", 5000000000)
        with experience_pool_locks["keyboard"]:
            keyboard_items = list(experience_pool["keyboard"].items())
        for ts, record in keyboard_items:
            if record.get("source", "user") == "user":
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
                text_data = valid_texts[max(valid_texts.keys())]["text"] if valid_texts else ""
                if not text_data:
                    text_data = " "
                txt_tensor = preprocess_text(text_data)
                if txt_tensor.dim() == 0:
                    txt_tensor = torch.tensor([], dtype=torch.long)
                label = [0] * num_keys
                if record["key_name"] in default_keys:
                    index = default_keys.index(record["key_name"])
                    label[index] = 1
                target_tensor = torch.tensor(label, dtype=torch.float32)
                samples.append((img_tensor, txt_tensor, target_tensor))
        loss_fn = nn.BCEWithLogitsLoss()
        if len(samples) == 0:
            logging.info("经验数据不足，离线优化结束。")
            best_loss = float("inf")
            time.sleep(global_config.get("optimization_duration_seconds", 10))
        else:
            batch_size = global_config.get("optimization_batch_size", 32)
            dataset = OfflineOptimizationDataset(samples)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            weight_decay = 1e-4 * (100 / max(len(samples), 100))
            optimizer = optim.Adam(model.parameters(), lr=global_config.get("optimizer_lr", 0.001), weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             factor=global_config.get("lr_decay_factor", 0.9),
                                                             patience=global_config.get("lr_decay_patience", 5))
            best_loss = float("inf")
            patience = global_config.get("offline_optimization_patience", 10)
            counter = 0
            epochs = global_config.get("optimization_epochs", 100)
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                batch_count = 0
                for images, texts, targets in dataloader:
                    optimizer.zero_grad()
                    keyboard_logits, _ = model(images, texts)
                    loss = loss_fn(keyboard_logits, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_count += 1
                avg_loss = epoch_loss / batch_count if batch_count else float("inf")
                scheduler.step(avg_loss)
                logging.info("离线优化 epoch %d, 平均损失: %.6f", epoch + 1, avg_loss)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    counter = 0
                else:
                    counter += 1
                if counter >= patience:
                    logging.info("提前停止训练，最佳损失: %.6f", best_loss)
                    break
            new_loss = evaluate_model(model, samples, loss_fn)
            with ai_model_lock:
                backup_model = torch.load(backup_model_path)
            backup_loss = evaluate_model(backup_model, samples, loss_fn)
            if new_loss < backup_loss:
                ai_reward_factor = min(ai_reward_factor + 0.1, 0.5)
                if hasattr(model, "version"):
                    model.version += 1
                else:
                    model.version = 1
                with ai_model_lock:
                    torch.save(model, ai_model_path)
                    ai_model = model
                logging.info("离线优化成功，新模型损失: %.6f, 模型版本: %d, 奖励因子调整为: %.2f", best_loss, model.version, ai_reward_factor)
            else:
                ai_reward_factor = max(ai_reward_factor - 0.1, 0.0)
                shutil.copyfile(backup_model_path, ai_model_path)
                with ai_model_lock:
                    ai_model = torch.load(ai_model_path)
                backup_version = backup_model.version if hasattr(backup_model, "version") else 1
                logging.info("离线优化未达预期，新模型损失 %.6f 高于备份模型 %.6f，已回滚至版本 %d, 奖励因子调整为: %.2f",
                             new_loss, backup_loss, backup_version, ai_reward_factor)
        optimization_in_progress = False
        mode = "learning"
    except Exception as e:
        logging.error("离线优化异常：%s", e)
        optimization_in_progress = False

def ai_online_learning_loop():
    global ai_model
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
                        if txt_tensor.dim() == 1:
                            txt_tensor = txt_tensor.unsqueeze(0)
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
                    texts = pad_sequence([s[1].squeeze(0) for s in samples], batch_first=True, padding_value=0)
                    targets = torch.stack([s[2] for s in samples], dim=0)
                    with ai_model_lock:
                        ai_model.train()
                        optimizer = optim.Adam(ai_model.parameters(), lr=global_config.get("online_optimizer_lr", 1e-4))
                        optimizer.zero_grad()
                        keyboard_logits, _ = ai_model(images, texts)
                        loss = loss_fn(keyboard_logits, targets)
                        loss.backward()
                        optimizer.step()
                        logging.info("在线优化更新完成，损失：%.6f", loss.item())
            time.sleep(global_config.get("online_optimization_interval", 10))
        except Exception as e:
            logging.error("在线优化异常：%s", e)
            time.sleep(1)

def show_input_window():
    global window_active, user_input_text, mode, tk_root, latest_text_ts
    try:
        window_active = True
        screen_width = tk_root.winfo_screenwidth()
        screen_height = tk_root.winfo_screenheight()
        win_width = int(screen_width * 0.3)
        win_height = int(screen_height * 0.1)
        pos_x = int((screen_width - win_width) / 2)
        pos_y = int((screen_height - win_height) / 2)
        geometry_str = f"{win_width}x{win_height}+{pos_x}+{pos_y}"
        input_win = tkinter.Toplevel(tk_root)
        input_win.title("输入窗口")
        input_win.geometry(geometry_str)
        input_win.grab_set()
        entry = tkinter.Entry(input_win, width=50)
        entry.pack(pady=10)
        button = tkinter.Button(input_win, text="确定", command=lambda: on_confirm(input_win, entry, button))
        button.pack()
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
        logging.error("输入窗口异常：%s", e)
        window_active = False

def on_confirm(window, entry, button):
    global user_input_text, mode, latest_text_ts, ai_model
    try:
        confirm_timestamp = time.time_ns()
        user_input_text = entry.get()
        with experience_pool_locks["text"]:
            experience_pool["text"][confirm_timestamp] = {"text": user_input_text, "source": "user"}
            latest_text_ts = confirm_timestamp
        mark_experience_pool_unsaved()
        ai_model_filename = global_config["required_files"]["ai_model"]
        ai_model_path = os.path.join(script_directory, ai_model_filename)
        with ai_model_lock:
            ai_model = torch.load(ai_model_path)
        mode = "learning"
        window.destroy()
    except Exception as e:
        logging.error("确定按钮处理异常：%s", e)

def on_hotkey_trigger():
    try:
        if not optimization_in_progress:
            threading.Thread(target=ai_offline_optimization, daemon=True).start()
        tk_queue.put("show_input")
    except Exception as e:
        logging.error("热键触发处理异常：%s", e)

def esc_monitor_loop():
    while True:
        try:
            if keyboard.is_pressed("esc"):
                start = time.time()
                while keyboard.is_pressed("esc"):
                    if time.time() - start >= global_config.get("esc_termination_duration", 3):
                        os._exit(0)
                    time.sleep(0.1)
            time.sleep(0.1)
        except Exception as e:
            logging.error("Esc 监控异常：%s", e)
            time.sleep(0.1)

def process_tk_queue():
    try:
        while not tk_queue.empty():
            cmd = tk_queue.get_nowait()
            if cmd == "show_input":
                show_input_window()
    except Exception as e:
        logging.error("处理 Tk 队列异常：%s", e)
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
                time.sleep(global_config.get("ai_training_loop_interval", 0.1))
            else:
                time.sleep(global_config.get("ai_training_loop_interval", 0.1))
        except Exception as e:
            logging.error("AI训练模式异常：%s", e)
            time.sleep(0.1)

def main():
    check_and_create_files()
    threading.Thread(target=memory_monitor, daemon=True).start()
    threading.Thread(target=screenshot_capture_loop, daemon=True).start()
    threading.Thread(target=experience_pool_saver_loop, daemon=True).start()
    keyboard.on_press(on_key_down)
    keyboard.on_release(on_key_up)
    mouse.hook(mouse_event_handler)
    threading.Thread(target=inactivity_monitor, daemon=True).start()
    threading.Thread(target=ai_training_loop, daemon=True).start()
    threading.Thread(target=esc_monitor_loop, daemon=True).start()
    threading.Thread(target=ai_online_learning_loop, daemon=True).start()
    hotkey = global_config.get("hotkey", "ctrl+alt")
    keyboard.add_hotkey(hotkey, on_hotkey_trigger)
    global tk_root
    tk_root = tkinter.Tk()
    tk_root.withdraw()
    tk_root.after(global_config.get("tk_queue_interval_ms", 100), process_tk_queue)
    tk_root.mainloop()

if __name__ == "__main__":
    main()
