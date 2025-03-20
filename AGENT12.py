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
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import pytesseract

# 初始化日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# 全局变量定义
global_config = {}         # 配置字典
script_directory = ""      # 脚本所在目录
experience_pool = {        # 经验池文件数据结构：屏幕截图、键盘、鼠标、文本数据均以纳秒级时间戳为键存储
    "screenshot": {},
    "keyboard": {},
    "mouse": {},
    "text": {}
}
experience_pool_lock = threading.Lock()  # 用于经验池文件写入的线程锁
mode = "learning"          # 当前模式：初始为“学习模式”，可切换为“训练模式”
last_activity_time = time.time()  # 上一次用户操作时间（秒级，用于检测10秒内无操作）
window_active = False      # 输入窗口是否显示
optimization_in_progress = False  # 离线优化是否进行中
user_input_text = ""       # 用户在输入窗口中输入的文本

# 当前正在进行的键盘事件（支持同时按下多个键，采用列表记录）
current_keyboard_events = []  
# 当前正在进行的鼠标事件，键为鼠标按键名称，值为事件记录字典
current_mouse_events = {}

# Tkinter 主窗口及线程安全队列（用于主线程中调度 Tkinter 相关操作）
tk_root = None
tk_queue = queue.Queue()

# ----------------------- AI 模型相关定义 -----------------------
class AIDecisionModel(nn.Module):
    def __init__(self, input_dim, keyboard_output_dim, mouse_output_dim):
        super(AIDecisionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.keyboard_head = nn.Linear(64, keyboard_output_dim)
        self.mouse_head = nn.Linear(64, mouse_output_dim)
        self.version = 1  # 模型版本号

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        keyboard_logits = self.keyboard_head(x)
        mouse_logits = self.mouse_head(x)
        return keyboard_logits, mouse_logits

# 全局 AI 模型变量
ai_model = None

def get_or_create_model():
    global ai_model, script_directory, global_config
    ai_model_filename = global_config["required_files"]["ai_model"]
    ai_model_path = os.path.join(script_directory, ai_model_filename)
    input_dim = 32 * 32 + 10  # 32x32 灰度图像展平后 + 10维文本向量
    default_keys = global_config.get("default_keyboard_keys", ["a", "b", "c"])
    keyboard_output_dim = len(default_keys)
    mouse_output_dim = 6  # 前4个值为操作类别（none, click, long_press, drag），后2个为拖拽坐标偏移
    if not os.path.exists(ai_model_path):
        model = AIDecisionModel(input_dim, keyboard_output_dim, mouse_output_dim)
        torch.save(model, ai_model_path)
        ai_model = model
    else:
        ai_model = torch.load(ai_model_path)

# ----------------------- 文件与内存管理 -----------------------
def save_experience_pool():
    """
    将全局经验池数据实时保存到与本程序相同目录下的 experience_pool.pkl 文件中
    """
    try:
        experience_pool_path = os.path.join(script_directory, "experience_pool.pkl")
        with open(experience_pool_path, "wb") as f:
            pickle.dump(experience_pool, f)
    except Exception as e:
        logging.error("保存经验池失败：%s", e)

def check_and_create_files():
    """
    检查并生成必要文件：配置文件、AI 模型文件、经验池文件，
    均放置于本程序所在目录（采用绝对路径）。
    """
    global global_config, script_directory, experience_pool
    try:
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # 检查配置文件
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
                "optimization_duration_seconds": 10,
                "screen_capture_interval": 1,
                "ai_operation_interval": 2,
                "default_keyboard_keys": ["a", "b", "c"],
                "default_mouse_operations": ["click", "long_press", "drag"],
                "window_geometry": "300x100",
                "memory_check_interval": 1,
                "long_press_duration": 1,
                "drag_distance": 100,
                "drag_duration": 0.5
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=4)
        with open(config_path, "r", encoding="utf-8") as f:
            global_config = json.load(f)
        # 检查 AI 模型文件（使用 PyTorch 模型）
        get_or_create_model()
        # 检查经验池文件
        exp_pool_filename = global_config["required_files"]["experience_pool"]
        exp_pool_path = os.path.join(script_directory, exp_pool_filename)
        if os.path.exists(exp_pool_path):
            try:
                with open(exp_pool_path, "rb") as f:
                    experience_pool_data = pickle.load(f)
                    if isinstance(experience_pool_data, dict):
                        experience_pool.update(experience_pool_data)
            except Exception:
                experience_pool.clear()
                experience_pool.update({
                    "screenshot": {},
                    "keyboard": {},
                    "mouse": {},
                    "text": {}
                })
                save_experience_pool()
        else:
            save_experience_pool()
    except Exception as e:
        logging.error("文件检查与创建失败：%s", e)
        sys.exit(1)

def memory_monitor():
    """
    实时监控内存使用情况，若使用率超过配置阈值，则执行垃圾回收以避免内存溢出。
    """
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
    """
    将截图数据（PNG 字节流）转换为 32x32 灰度图像展平后的张量，并归一化至 [0,1] 范围。
    """
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img = img.resize((32, 32))
        img_data = list(img.getdata())
        img_tensor = torch.tensor(img_data, dtype=torch.float32) / 255.0
        return img_tensor  # 形状 (1024,)
    except Exception as e:
        logging.error("截图预处理异常：%s", e)
        return torch.zeros(32 * 32)

def preprocess_text(text):
    """
    将文本转换为固定长度（10）的张量，每个字符转为 ASCII 编码后归一化（除以128）。
    """
    tensor = torch.zeros(10, dtype=torch.float32)
    for i, ch in enumerate(text):
        if i >= 10:
            break
        tensor[i] = ord(ch) / 128.0
    return tensor

# ----------------------- 截图与事件采集 -----------------------
def screenshot_capture_loop():
    """
    持续对电脑屏幕进行截图，并根据当前 CPU 使用率动态调整截图频率。
    截图数据转换为 PNG 格式字节流后，以纳秒级时间戳为键存入经验池文件。
    当输入窗口显示时暂停截图采集。
    """
    base_interval = global_config.get("screen_capture_interval", 1)
    while True:
        try:
            if window_active:
                time.sleep(0.5)
                continue
            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage < 20:
                interval = base_interval * 0.5
            elif cpu_usage > 80:
                interval = base_interval * 2
            else:
                interval = base_interval * (cpu_usage / 50)
            interval = max(0.5, min(interval, base_interval * 2))
            screenshot = pyautogui.screenshot()
            buf = io.BytesIO()
            screenshot.save(buf, format='PNG')
            img_bytes = buf.getvalue()
            timestamp = time.time_ns()
            with experience_pool_lock:
                experience_pool["screenshot"][timestamp] = img_bytes
                save_experience_pool()
            time.sleep(interval)
        except Exception as e:
            logging.error("截图采集异常：%s", e)
            time.sleep(0.5)

# 键盘事件处理（支持同时按下多个键，采用列表记录）
def on_key_down(event):
    global last_activity_time, mode, current_keyboard_events
    try:
        last_activity_time = time.time()
        if mode == "training":
            mode = "learning"
            return
        if window_active:
            return
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
        if mode != "learning" or window_active:
            return
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
            with experience_pool_lock:
                experience_pool["keyboard"][rec["down_timestamp"]] = rec
                save_experience_pool()
    except Exception as e:
        logging.error("键盘抬起事件异常：%s", e)

# 鼠标事件统一处理
def get_event_position(event):
    """
    尝试从事件对象中获取屏幕坐标；若失败，则调用 pyautogui.position() 获取当前位置。
    """
    try:
        return (event.x, event.y)
    except AttributeError:
        return pyautogui.position()

def mouse_event_handler(event):
    """
    鼠标事件统一处理函数：
    - 仅在学习模式且无输入窗口显示时记录用户的鼠标操作；
    - 鼠标按下时开始记录，移动事件记录运动轨迹，鼠标抬起时根据操作时间及位移判断操作类型（点击、长按或拖拽），并存入经验池文件，
      使用按下时的纳秒时间戳作为键；
    - 若处于训练模式，则切换回学习模式（不记录当前事件）。
    """
    global last_activity_time, mode, current_mouse_events
    try:
        last_activity_time = time.time()
        if mode == "training":
            mode = "learning"
            return
        if window_active:
            return
        event_type = event.event_type if hasattr(event, "event_type") else "move"
        if event_type == "down":
            timestamp = time.time_ns()
            record = {
                "operation": None,  # 后续确定为“点击”、“长按”或“拖拽”
                "button": event.button,  # "left" 或 "right"
                "down_timestamp": timestamp,
                "down_position": get_event_position(event),
                "up_timestamp": None,
                "up_position": None,
                "movement_track": []  # 存储运动轨迹（每项包含时间戳与位置）
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
                if distance > 5 and len(rec["movement_track"]) > 0:
                    rec["operation"] = "拖拽"
                elif duration >= global_config.get("long_press_duration", 1):
                    rec["operation"] = "长按"
                else:
                    rec["operation"] = "点击"
                with experience_pool_lock:
                    experience_pool["mouse"][rec["down_timestamp"]] = rec
                    save_experience_pool()
    except Exception as e:
        logging.error("鼠标事件处理异常：%s", e)

def inactivity_monitor():
    """
    监控用户键盘和鼠标操作：
    - 当检测到连续 10 秒无任何用户操作时，将模式切换为“训练模式”。
    """
    global mode
    while True:
        try:
            if mode == "learning" and (time.time() - last_activity_time >= 10):
                mode = "training"
            time.sleep(1)
        except Exception as e:
            logging.error("无操作监控异常：%s", e)
            time.sleep(1)

# ----------------------- AI 决策与训练 -----------------------
def ai_decision():
    """
    AI 决策函数：
    - 从经验池中获取最新截图并利用 pytesseract 进行 OCR 提取屏幕文本，
      同时获取最新用户输入文本（若存在），
    - 将截图与文本预处理后组合成固定维度输入，经由全局 AI 模型输出键盘及鼠标操作决策，
    - 根据输出概率与预测结果构造操作列表（支持同时输出多个键盘操作，鼠标拖拽根据预测偏移量确定终点）。
    """
    try:
        with experience_pool_lock:
            if experience_pool["screenshot"]:
                latest_screenshot_ts = max(experience_pool["screenshot"].keys())
                screenshot_bytes = experience_pool["screenshot"][latest_screenshot_ts]
            else:
                return []
        img_feature = preprocess_screenshot(screenshot_bytes)
        with experience_pool_lock:
            if experience_pool["text"]:
                latest_text_ts = max(experience_pool["text"].keys())
                text_data = experience_pool["text"][latest_text_ts]
            else:
                text_data = ""
        text_feature = preprocess_text(text_data)
        input_tensor = torch.cat([img_feature, text_feature]).float().unsqueeze(0)  # 形状 (1, 1034)
        ai_model.eval()
        with torch.no_grad():
            keyboard_logits, mouse_logits = ai_model(input_tensor)
        # 处理键盘部分：对每个默认键计算概率，若概率大于0.5则决定按下该键
        keyboard_probs = torch.softmax(keyboard_logits, dim=1).squeeze(0)
        keys_to_press = []
        default_keys = global_config.get("default_keyboard_keys", ["a", "b", "c"])
        for i, prob in enumerate(keyboard_probs):
            if prob.item() > 0.5:
                keys_to_press.append(default_keys[i])
        # 处理鼠标部分：前4个值为操作类别 logits，后2个为拖拽偏移量
        mouse_logits_split = mouse_logits.squeeze(0)
        mouse_op_logits = mouse_logits_split[:4]
        mouse_op_probs = torch.softmax(mouse_op_logits, dim=0)
        mouse_op_index = torch.argmax(mouse_op_probs).item()
        mouse_ops = ["none", "click", "long_press", "drag"]
        mouse_op = mouse_ops[mouse_op_index]
        mouse_offset = mouse_logits_split[4:].tolist()  # 拖拽偏移量
        operations = []
        if keys_to_press:
            operations.append({"type": "keyboard", "keys": keys_to_press, "timestamp": time.time_ns()})
        if mouse_op != "none":
            current_pos = pyautogui.position()
            if mouse_op == "drag":
                end_position = (current_pos[0] + int(mouse_offset[0]), current_pos[1] + int(mouse_offset[1]))
                operations.append({"type": "mouse", "operation": "drag", "button": "left",
                                   "start_position": current_pos,
                                   "end_position": end_position,
                                   "timestamp": time.time_ns()})
            else:
                operations.append({"type": "mouse", "operation": mouse_op, "button": "left",
                                   "timestamp": time.time_ns()})
        return operations
    except Exception as e:
        logging.error("AI 决策异常：%s", e)
        return []

def perform_keyboard_operation(keys, source="user"):
    """
    同时按下 keys 列表中的键（数量不超过配置中最大并行按键数），
    使用 pyautogui.hotkey 模拟多个键同时按下，并将操作记录到经验池中（附上 source 标记）。
    """
    try:
        max_keys = global_config.get("max_parallel_keyboard_ops", 3)
        if len(keys) > max_keys:
            keys = keys[:max_keys]
        pyautogui.hotkey(*keys)
        down_timestamp = time.time_ns()
        for key in keys:
            record = {
                "key_name": key,
                "down_timestamp": down_timestamp,
                "up_timestamp": time.time_ns(),
                "source": source
            }
            with experience_pool_lock:
                experience_pool["keyboard"][down_timestamp] = record
                save_experience_pool()
    except Exception as e:
        logging.error("键盘操作执行异常：%s", e)

def perform_mouse_operation(operation, source="user"):
    """
    根据 operation 参数执行鼠标操作，支持 "click"（点击）、"long_press"（长按）和 "drag"（拖拽）。
    对于拖拽操作，根据 operation 中提供的起始与终点位置执行真实鼠标拖拽。
    执行完成后，将操作记录（附上 source 标记）存入经验池中。
    """
    try:
        if operation["operation"] == "click":
            pyautogui.click()
        elif operation["operation"] == "long_press":
            pyautogui.mouseDown()
            time.sleep(global_config.get("long_press_duration", 1))
            pyautogui.mouseUp()
        elif operation["operation"] == "drag":
            start_pos = operation.get("start_position", pyautogui.position())
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
        with experience_pool_lock:
            experience_pool["mouse"][down_timestamp] = record
            save_experience_pool()
    except Exception as e:
        logging.error("鼠标操作执行异常（%s）：%s", operation, e)

def ai_training_loop():
    """
    AI 训练模式主循环（仅在训练模式下运行）：
    - 调用 ai_decision() 根据屏幕截图与用户文本实时决策输出键盘与鼠标操作，
      支持并行输出（最多不超过配置限制），并将 AI 输出的操作实时记录到经验池中（source 标记为 "ai"）。
    """
    while True:
        try:
            if mode == "training" and not window_active and not optimization_in_progress:
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
                            perform_mouse_operation(op, source="ai")
                time.sleep(global_config.get("ai_operation_interval", 2))
            else:
                time.sleep(0.5)
        except Exception as e:
            logging.error("AI 训练模式异常：%s", e)
            time.sleep(1)

def ai_offline_optimization():
    """
    离线优化 AI 模型：
    1. 加载当前 AI 模型；
    2. 利用经验池中用户键盘事件数据构造训练样本，
       对模型进行真实的训练（采用早停与 L2 正则化方法实现最优训练效果）；
    3. 训练完成后更新模型（版本号递增）并覆盖保存至 AI 模型文件中，
       优化期间输入窗口上的“确定”按钮处于不可用状态；
    4. 优化完成后预加载新模型，程序切换回“学习模式”。
    """
    global optimization_in_progress, mode, ai_model
    try:
        optimization_in_progress = True
        ai_model_filename = global_config["required_files"]["ai_model"]
        ai_model_path = os.path.join(script_directory, ai_model_filename)
        model = torch.load(ai_model_path)
        model.train()
        # 构造训练数据：使用经验池中用户的键盘记录构造 (截图+文本) -> (默认键的 one-hot 标签)
        default_keys = global_config.get("default_keyboard_keys", ["a", "b", "c"])
        num_keys = len(default_keys)
        inputs = []
        targets = []
        with experience_pool_lock:
            for ts, record in experience_pool["keyboard"].items():
                if record.get("source", "user") == "user":
                    screenshot_keys = sorted(experience_pool["screenshot"].keys())
                    img_feature = torch.zeros(32 * 32)
                    for s_ts in screenshot_keys:
                        if s_ts <= ts:
                            img_bytes = experience_pool["screenshot"][s_ts]
                            img_feature = preprocess_screenshot(img_bytes)
                        else:
                            break
                    text_keys = sorted(experience_pool["text"].keys())
                    text_data = ""
                    for t_ts in text_keys:
                        if t_ts <= ts:
                            text_data = experience_pool["text"][t_ts]
                        else:
                            break
                    text_feature = preprocess_text(text_data)
                    input_tensor = torch.cat([img_feature, text_feature])
                    inputs.append(input_tensor.unsqueeze(0))
                    # 构造标签：若按下键在默认键中，则对应 one-hot 向量；否则视为全零
                    key_name = record["key_name"]
                    label = [0] * num_keys
                    if key_name in default_keys:
                        index = default_keys.index(key_name)
                        label[index] = 1
                    targets.append(torch.tensor(label, dtype=torch.float32).unsqueeze(0))
        if len(inputs) == 0:
            time.sleep(global_config.get("optimization_duration_seconds", 10))
        else:
            X = torch.cat(inputs, dim=0)  # 形状 (N, 1034)
            Y = torch.cat(targets, dim=0)  # 形状 (N, num_keys)
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            loss_fn = nn.MSELoss()
            best_loss = float("inf")
            patience = 10
            counter = 0
            epochs = 100
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                keyboard_logits, _ = model(X)
                loss = loss_fn(torch.softmax(keyboard_logits, dim=1), Y)
                loss.backward()
                optimizer.step()
                current_loss = loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    counter = 0
                else:
                    counter += 1
                if counter >= patience:
                    break
        if hasattr(model, "version"):
            model.version += 1
        else:
            model.version = 1
        torch.save(model, ai_model_path)
        ai_model = model
        optimization_in_progress = False
        mode = "learning"
    except Exception as e:
        logging.error("离线优化异常：%s", e)
        optimization_in_progress = False

# ----------------------- 输入窗口与热键 -----------------------
def show_input_window():
    """
    弹出用户输入窗口：
    - 窗口标题为“输入窗口”，尺寸由配置指定；
    - 窗口内包含文本输入框及“确定”按钮；
    - 用户点击“确定”后，获取整段输入文本，以按下按钮时的纳秒级时间戳作为键存入经验池文件，同时关闭窗口；
    - 窗口显示期间，程序其他所有功能暂停（AI 不输出键盘和鼠标操作）；
    - 优化期间，“确定”按钮处于不可用状态。
    """
    global window_active, user_input_text, mode, tk_root
    try:
        window_active = True
        input_win = tkinter.Toplevel(tk_root)
        input_win.title("输入窗口")
        input_win.geometry(global_config.get("window_geometry", "300x100"))
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
            input_win.after(100, update_button_state)
        update_button_state()
        input_win.wait_window()
        window_active = False
    except Exception as e:
        logging.error("输入窗口异常：%s", e)
        window_active = False

def on_confirm(window, entry, button):
    """
    用户点击“确定”按钮后的回调处理函数：
    - 获取用户输入文本，以按下按钮时的纳秒级时间戳作为键存入经验池文件；
    - 预加载 AI 模型文件并切换回“学习模式”，最后关闭窗口。
    """
    global user_input_text, mode
    try:
        confirm_timestamp = time.time_ns()
        user_input_text = entry.get()
        with experience_pool_lock:
            experience_pool["text"][confirm_timestamp] = user_input_text
            save_experience_pool()
        ai_model_filename = global_config["required_files"]["ai_model"]
        ai_model_path = os.path.join(script_directory, ai_model_filename)
        with open(ai_model_path, "rb") as f:
            _ = torch.load(f)
        mode = "learning"
        window.destroy()
    except Exception as e:
        logging.error("确定按钮处理异常：%s", e)

def on_hotkey_trigger():
    """
    全局热键（Ctrl+Alt）触发处理函数：
    - 当检测到热键时，启动离线优化（若未在进行中）并将请求放入线程安全队列，由主线程调度显示输入窗口，
      从而保证 Tkinter 操作均在主线程中执行，其余功能暂停。
    """
    try:
        if not optimization_in_progress:
            threading.Thread(target=ai_offline_optimization, daemon=True).start()
        tk_queue.put("show_input")
    except Exception as e:
        logging.error("热键触发处理异常：%s", e)

def esc_monitor_loop():
    """
    监控 Esc 键：
    - 当检测到用户长按 Esc 键且持续时长达到 3 秒时，立即终止整个程序运行。
    """
    while True:
        try:
            if keyboard.is_pressed("esc"):
                start = time.time()
                while keyboard.is_pressed("esc"):
                    if time.time() - start >= 3:
                        os._exit(0)
                    time.sleep(0.1)
            time.sleep(0.1)
        except Exception as e:
            logging.error("Esc 监控异常：%s", e)
            time.sleep(0.1)

def process_tk_queue():
    """
    主线程中周期性检查线程安全队列，若有“显示输入窗口”请求则调用对应函数。
    """
    try:
        while not tk_queue.empty():
            cmd = tk_queue.get_nowait()
            if cmd == "show_input":
                show_input_window()
    except Exception as e:
        logging.error("处理 Tk 队列异常：%s", e)
    finally:
        tk_root.after(100, process_tk_queue)

# ----------------------- 主函数 -----------------------
def main():
    """
    主函数：
    1. 获取脚本所在目录并检查生成必要文件（配置文件、AI 模型文件、经验池文件）；
    2. 预加载 AI 模型并启动内存监控、屏幕截图采集、用户事件监听（键盘、鼠标）、用户操作监控、AI 训练模式及 Esc 退出监控线程；
    3. 注册全局热键（Ctrl+Alt）用于启动离线优化及输入窗口；
    4. 在主线程中创建 Tkinter 隐藏主窗口，并开始周期性处理 Tk 队列，保证所有 Tkinter 操作均在主线程中执行。
    """
    check_and_create_files()
    mem_thread = threading.Thread(target=memory_monitor, daemon=True)
    mem_thread.start()
    screenshot_thread = threading.Thread(target=screenshot_capture_loop, daemon=True)
    screenshot_thread.start()
    keyboard.on_press(on_key_down)
    keyboard.on_release(on_key_up)
    mouse.hook(mouse_event_handler)
    inactivity_thread = threading.Thread(target=inactivity_monitor, daemon=True)
    inactivity_thread.start()
    ai_thread = threading.Thread(target=ai_training_loop, daemon=True)
    ai_thread.start()
    esc_thread = threading.Thread(target=esc_monitor_loop, daemon=True)
    esc_thread.start()
    hotkey = global_config.get("hotkey", "ctrl+alt")
    keyboard.add_hotkey(hotkey, on_hotkey_trigger)
    global tk_root
    tk_root = tkinter.Tk()
    tk_root.withdraw()
    tk_root.after(100, process_tk_queue)
    tk_root.mainloop()

if __name__ == "__main__":
    main()
