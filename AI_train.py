import os
import json
import time
import gc
import random
import shutil
from pathlib import Path
from datetime import timedelta
import traceback
import platform
from collections import deque
import warnings
import pickle
import sys
import math
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`torch.cuda.amp.GradScaler.*is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`torch.cuda.amp.autocast.*is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*The verbose parameter is deprecated.*")
warnings.filterwarnings("ignore", message=".*Detected call of `lr_scheduler.step()` before `optimizer.step()`.*")
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.*")
warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*encoder_layer.norm_first was True.*")
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.amp import autocast, GradScaler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import Dataset, DataLoader, Subset
    from torchvision import transforms
    try:
        import timm
        TIMM_AVAILABLE = True
        timm_version = tuple(map(int, timm.__version__.split('.')[:2]))
        print(f"Found 'timm' library (version: {timm.__version__}) for Vision Transformer models.")
    except ImportError:
        print("ERROR: 'timm' library not found. This script requires ViT models.")
        print("Install with: pip install timm")
        print("Script cannot proceed without 'timm'. Exiting.")
        sys.exit(1)
    from PIL import Image, UnidentifiedImageError, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        from safetensors.torch import load_file as load_safetensors_file
        SAFETENSORS_AVAILABLE = True
        print("Found 'safetensors' library.")
    except ImportError:
        SAFETENSORS_AVAILABLE = False
        print("Warning: 'safetensors' library not found. Will rely on torch.load for .bin files (less secure).")
        print("         Install with: pip install safetensors")
except ImportError as e:
    print(f"ERROR: Missing required library: {e.name}")
    print("Please install the necessary libraries:")
    print("pip install torch torchvision torchaudio pandas numpy Pillow timm safetensors")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during initial imports: {e}")
    traceback.print_exc()
    sys.exit(1)
SCRIPT_DIR = Path(__file__).parent.resolve()
EXPERIENCE_POOL_DIR = SCRIPT_DIR / "experience_pool"
SCREENSHOT_DIR = EXPERIENCE_POOL_DIR / "screenshots"
KEYBOARD_LOG_FILE = EXPERIENCE_POOL_DIR / "keyboard_log.jsonl"
MOUSE_LOG_FILE = EXPERIENCE_POOL_DIR / "mouse_log.jsonl"
RESULTS_LOG_FILE = EXPERIENCE_POOL_DIR / "results_log.jsonl"
PRETRAINED_MODEL_DIR = SCRIPT_DIR / "pretrained_models"
KEYBOARD_MODEL_FILE = SCRIPT_DIR / "keyboard_ai_model_transformer.pth"
MOUSE_MODEL_FILE = SCRIPT_DIR / "mouse_ai_model_transformer.pth"
OLD_KEYBOARD_MODEL_FILE = SCRIPT_DIR / "keyboard_ai_model_transformer_old.pth"
OLD_MOUSE_MODEL_FILE = SCRIPT_DIR / "mouse_ai_model_transformer_old.pth"
print("\n--- Verifying Environment ---")
essential_paths = {
    "Experience Pool Dir": EXPERIENCE_POOL_DIR,
    "Screenshots Dir": SCREENSHOT_DIR,
    "Keyboard Log": KEYBOARD_LOG_FILE,
    "Mouse Log": MOUSE_LOG_FILE,
    "Results Log": RESULTS_LOG_FILE,
    "Pretrained Models Dir": PRETRAINED_MODEL_DIR,
}
found_missing_critical = False
for name, path in essential_paths.items():
    is_dir = "Dir" in name
    exists = path.is_dir() if is_dir else path.is_file()
    status = "Found" if exists else "NOT FOUND"
    print(f"  {name:<25}: {path} ({status})")
    if not exists:
        if "Log" in name:
            if name in ["Keyboard Log", "Mouse Log"]:
                 print(f"    Warning: Log file '{path.name}' not found. The corresponding AI ('{name.split()[0]}') might not be trainable without data.")
            else:
                 print(f"    Warning: Optional log file '{path.name}' not found. Rewards may default to neutral.")
        elif name in ["Pretrained Models Dir", "Screenshots Dir", "Experience Pool Dir"]:
             print(f"    ERROR: Critical directory '{path.name}' is missing.")
             found_missing_critical = True
if PRETRAINED_MODEL_DIR.is_dir():
    pt_bin = PRETRAINED_MODEL_DIR / "pytorch_model.bin"
    pt_safe = PRETRAINED_MODEL_DIR / "model.safetensors"
    pt_bin_exists = pt_bin.is_file()
    pt_safe_exists = pt_safe.is_file()
    print(f"  Pretrained '.bin' file      : {pt_bin} ({'Found' if pt_bin_exists else 'NOT FOUND'})")
    print(f"  Pretrained '.safetensors' file: {pt_safe} ({'Found' if pt_safe_exists else 'NOT FOUND'})")
    if not pt_bin_exists and not pt_safe_exists:
        print(f"ERROR: Critical - Neither 'pytorch_model.bin' nor 'model.safetensors' found in {PRETRAINED_MODEL_DIR}.")
        found_missing_critical = True
if found_missing_critical:
    print("\nERROR: One or more essential directories or pretrained files are missing. Please correct the paths or files.")
    sys.exit(1)
else:
    print("Environment verification passed (required directories/files found or optional files noted).")
print("-" * 30)
VISION_MODEL_NAME = 'vit_small_patch16_224.augreg_in21k_ft_in1k'
IMG_SIZE = (224, 224)
SEQUENCE_LENGTH = 8
TRANSFORMER_D_MODEL = 384
TRANSFORMER_NHEAD = 6
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_DIM_FEEDFORWARD = TRANSFORMER_D_MODEL * 4
TRANSFORMER_DROPOUT = 0.1
BATCH_SIZE_INITIAL = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
EPOCHS_MAX = 50
PATIENCE_EARLY_STOPPING = 5
GRADIENT_CLIP_VALUE = 1.0
VAL_SPLIT = 0.15
TRAINING_TIME_LIMIT_PER_MODEL_SEC = None
FORCE_NUM_WORKERS = None
_cpu_count = os.cpu_count() or 1
DEFAULT_NUM_WORKERS_NON_WINDOWS = min(8, _cpu_count // 2) if _cpu_count > 1 else 0
DEFAULT_NUM_WORKERS = 0 if platform.system() == "Windows" else DEFAULT_NUM_WORKERS_NON_WINDOWS
NUM_DATALOADER_WORKERS = FORCE_NUM_WORKERS if FORCE_NUM_WORKERS is not None else DEFAULT_NUM_WORKERS
PERSISTENT_WORKERS = (NUM_DATALOADER_WORKERS > 0) and (platform.system() != "Windows")
PIN_MEMORY = torch.cuda.is_available()
print(f"DataLoader Config: Workers={NUM_DATALOADER_WORKERS}, Persistent={PERSISTENT_WORKERS}, PinMemory={PIN_MEMORY}")
ACTION_PREDICTION_WINDOW_MS = 100
REWARD_LOOKBACK_WINDOW_SEC = 60
REWARD_MAPPING = {"win": 1.5, "draw": 1.0, "lose": 0.5, "skip": 1.0}
MIN_VALID_SEQUENCES_FOR_TRAINING = max(10, int(BATCH_SIZE_INITIAL / (1.0 - VAL_SPLIT)) + 1, BATCH_SIZE_INITIAL * 2)
print(f"Minimum *valid sequences* required to attempt training (after alignment): {MIN_VALID_SEQUENCES_FOR_TRAINING}")
MAX_PARALLEL_KEYS_OUTPUT = 3
FORBIDDEN_KEYS = {
    'ctrl', 'ctrl_l', 'ctrl_r', 'alt', 'alt_l', 'alt_r', 'alt_gr',
    'shift', 'shift_l', 'shift_r', 'caps_lock',
    'cmd', 'cmd_l', 'cmd_r', 'win', 'meta', 'windows', 'gui', 'gui_l', 'gui_r',
    'apps', 'menu', 'compose', 'scroll_lock', 'num_lock', 'insert', 'delete',
    'home', 'end', 'page_up', 'page_down', 'print_screen', 'pause', 'sleep', 'wakeup', 'esc',
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
    'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24',
    'media_volume_mute', 'media_volume_down', 'media_volume_up', 'media_play_pause',
    'media_previous', 'media_next', 'media_stop',
    'kana', 'kanji', 'convert', 'nonconvert', 'yen', 'hiragana', 'katakana',
    'unknown', 'unidentified', None, '', ' ', 'none', 'spacebar',
}
MOUSE_ACTION_TYPES_TARGET = ['no_action', 'click_left', 'click_right', 'drag_left', 'drag_right']
MOUSE_ACTION_MAPPING = {'click': 'click', 'long_press': 'click', 'drag': 'drag'}
SCREEN_RESOLUTION = None
def get_screen_resolution(image_path):
    try:
        with Image.open(image_path) as img:
            res = img.size
            if isinstance(res, tuple) and len(res) == 2 and all(isinstance(d, int) and d > 0 for d in res):
                return res
            else: print(f"Warning: Invalid resolution tuple read from {image_path}: {res}")
    except FileNotFoundError: print(f"Warning: Sample image for resolution not found: {image_path}")
    except UnidentifiedImageError: print(f"Warning: Could not read sample image (unidentified format or corrupted?): {image_path}")
    except Exception as e: print(f"Warning: Error inferring screen resolution from {image_path}: {e}")
    return None
def load_jsonl(filepath):
    data = []
    lines_read = 0
    invalid_lines = 0
    if not filepath.exists():
        print(f"Info: Log file not found: {filepath}. Returning empty DataFrame.")
        return pd.DataFrame()
    try:
        with filepath.open('r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                lines_read += 1
                line_content = line.strip()
                if not line_content: continue
                try:
                    data.append(json.loads(line_content))
                except json.JSONDecodeError:
                    invalid_lines += 1
                    if invalid_lines < 5 or invalid_lines % 1000 == 0:
                         print(f"Warning: Skipping invalid JSON line {i+1} in {filepath.name} (error {invalid_lines}). Content: '{line_content[:100]}...'")
        print(f"Loaded {len(data)} records from {filepath.name} ({lines_read} lines read, {invalid_lines} invalid lines skipped).")
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        if 'timestamp_ns' not in df.columns:
            print(f"ERROR: Crucial 'timestamp_ns' column missing in {filepath.name}. Returning empty DataFrame.")
            return pd.DataFrame()
        initial_rows = len(df)
        df['timestamp_ns'] = pd.to_numeric(df['timestamp_ns'], errors='coerce')
        df.dropna(subset=['timestamp_ns'], inplace=True)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"Warning: Dropped {dropped_rows} rows from {filepath.name} due to non-numeric 'timestamp_ns'.")
        if df.empty: return pd.DataFrame()
        df['timestamp_ns'] = df['timestamp_ns'].astype('int64')
        required_cols = {'timestamp_ns'}
        if 'keyboard' in filepath.name:
            required_cols.update(['key_name', 'press_time_ns', 'release_time_ns'])
            missing_cols = required_cols - set(df.columns)
            if missing_cols: print(f"Warning: Missing expected keyboard columns in {filepath.name}: {missing_cols}.")
            if 'key_name' in df.columns: df['key_name'] = df['key_name'].fillna('').astype(str).str.lower().str.strip()
            else: df['key_name'] = ''
            for col in ['press_time_ns', 'release_time_ns']:
                 if col in df.columns:
                     df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                 else: df[col] = pd.NA
        elif 'mouse' in filepath.name:
            required_cols.update(['action', 'button', 'start_pos', 'end_pos', 'press_time_ns', 'release_time_ns'])
            missing_cols = required_cols - set(df.columns)
            if missing_cols: print(f"Warning: Missing expected mouse columns in {filepath.name}: {missing_cols}.")
            for col in ['action', 'button']:
                 if col in df.columns: df[col] = df[col].fillna('unknown').astype(str).str.lower().str.strip()
                 else: df[col] = 'unknown'
            for col in ['press_time_ns', 'release_time_ns']:
                 if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                 else: df[col] = pd.NA
            for col in ['start_pos', 'end_pos']:
                 if col in df.columns:
                     df[col] = df[col].apply(lambda p: p if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(n, (int, float)) and not math.isnan(n) for n in p) else None)
                 else: df[col] = None
            if 'trajectory' not in df.columns: df['trajectory'] = None
        elif 'results' in filepath.name:
            required_cols.add('result')
            if 'result' not in df.columns: print(f"Warning: Missing expected 'result' column in {filepath.name}.")
            if 'result' in df.columns: df['result'] = df['result'].fillna('unknown').astype(str).str.lower().str.strip()
            else: df['result'] = 'unknown'
        return df
    except Exception as e:
        print(f"ERROR loading or processing {filepath.name}: {e}")
        traceback.print_exc()
        return pd.DataFrame()
def cleanup_memory():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e_cleanup:
        print(f"Warning: Memory cleanup (GC/CUDA Cache) failed: {e_cleanup}")
def preprocess_data():
    global SCREEN_RESOLUTION
    print("\n--- Starting Data Preprocessing ---")
    t_start = time.time()
    print("Loading log files...")
    df_keyboard = load_jsonl(KEYBOARD_LOG_FILE)
    df_mouse = load_jsonl(MOUSE_LOG_FILE)
    df_results = load_jsonl(RESULTS_LOG_FILE)
    print(f"Scanning screenshots in {SCREENSHOT_DIR}...")
    screenshots = []
    sample_img_path_for_res = None
    checked_files = 0
    png_count = 0
    invalid_name_count = 0
    inferred_res = None
    if not SCREENSHOT_DIR.is_dir():
        print(f"ERROR: Screenshot directory not found or not a directory: {SCREENSHOT_DIR}")
        return pd.DataFrame(), [], {}, {}, {}, False
    try:
        potential_screenshots = []
        for f in SCREENSHOT_DIR.iterdir():
            checked_files += 1
            if f.is_file() and f.suffix.lower() == '.png':
                png_count += 1
                try:
                    ts_str = f.stem
                    if ts_str.isdigit():
                        ts_ns = int(ts_str)
                        potential_screenshots.append({'timestamp_ns': ts_ns, 'filepath': str(f), 'path_obj': f})
                    else:
                        invalid_name_count +=1
                        if invalid_name_count < 5 or invalid_name_count % 1000 == 0:
                            print(f"Warning: Skipping screenshot file with non-numeric name: {f.name} (Count: {invalid_name_count})")
                except (ValueError, TypeError) as e_parse:
                    print(f"Warning: Skipping screenshot file {f.name} due to timestamp parsing error: {e_parse}")
                except Exception as e_other:
                    print(f"Warning: Skipping screenshot file {f.name} due to unexpected error: {e_other}")
        screenshots = sorted(potential_screenshots, key=lambda x: x['timestamp_ns'])
        print(f"Checked {checked_files} files. Found {png_count} PNG files, {len(screenshots)} with valid numeric timestamp names.")
        if invalid_name_count > 0: print(f"  ({invalid_name_count} PNGs skipped due to non-numeric names).")
    except Exception as e:
        print(f"ERROR scanning screenshot directory {SCREENSHOT_DIR}: {e}")
        return pd.DataFrame(), [], {}, {}, {}, False
    if not screenshots:
        print("ERROR: No valid screenshots found (PNG files with numeric timestamp names). Cannot proceed.")
        return pd.DataFrame(), [], {}, {}, {}, False
    if SCREEN_RESOLUTION is None:
        print("Attempting to infer screen resolution...")
        for ss_info in screenshots:
             is_likely_png = False
             try:
                 with open(ss_info['path_obj'], 'rb') as img_f: header = img_f.read(8)
                 if header.startswith(b'\x89PNG\r\n\x1a\n'): is_likely_png = True
             except Exception: pass
             if is_likely_png:
                 inferred_res_attempt = get_screen_resolution(ss_info['path_obj'])
                 if inferred_res_attempt:
                     SCREEN_RESOLUTION = inferred_res_attempt
                     sample_img_path_for_res = ss_info['path_obj']
                     print(f"Successfully inferred screen resolution {SCREEN_RESOLUTION} from {sample_img_path_for_res.name}")
                     break
        if SCREEN_RESOLUTION is None:
            print("ERROR: Could not infer screen resolution from any valid PNG file. Using default (1920, 1080), but this may lead to incorrect mouse coordinates.")
            SCREEN_RESOLUTION = (1920, 1080)
    print(f"Using Screen resolution: {SCREEN_RESOLUTION}")
    try:
        screen_w, screen_h = SCREEN_RESOLUTION
        if not (isinstance(screen_w, int) and screen_w > 0 and isinstance(screen_h, int) and screen_h > 0):
            raise ValueError("Invalid screen resolution components.")
    except (TypeError, ValueError) as e_res:
        print(f"ERROR: Invalid screen resolution determined: {SCREEN_RESOLUTION}. Error: {e_res}. Cannot normalize coordinates.")
        return pd.DataFrame(), [], {}, {}, {}, False
    df_screenshots = pd.DataFrame(screenshots)
    print("Converting timestamps and validating data...")
    valid_dfs = {}
    min_timestamp_overall = pd.Timestamp.max.tz_localize(None)
    max_timestamp_overall = pd.Timestamp.min.tz_localize(None)
    for df_name, df in [('screenshots', df_screenshots), ('keyboard', df_keyboard), ('mouse', df_mouse), ('results', df_results)]:
        if df is None or df.empty:
            print(f"Info: No data loaded for {df_name}.")
            continue
        if 'timestamp_ns' not in df.columns:
            print(f"Warning: 'timestamp_ns' column missing in {df_name} DataFrame. Skipping.")
            continue
        initial_count = len(df)
        df['timestamp'] = pd.to_datetime(df['timestamp_ns'], unit='ns', errors='coerce')
        failed_conversion_mask = df['timestamp'].isna()
        dropped_datetime = failed_conversion_mask.sum()
        if dropped_datetime > 0:
            print(f"Warning: Dropped {dropped_datetime}/{initial_count} rows from {df_name} due to invalid datetime conversion from timestamp_ns.")
            df.dropna(subset=['timestamp'], inplace=True)
        if not df.empty and pd.api.types.is_datetime64_any_dtype(df['timestamp']) and df['timestamp'].dt.tz is not None:
            try:
                df['timestamp'] = df['timestamp'].dt.tz_convert(None)
            except Exception as e_tz:
                print(f"Warning: Could not make timestamp column timezone naive for {df_name}: {e_tz}")
        if not df.empty:
            valid_dfs[df_name] = df
            try:
                min_ts, max_ts = df['timestamp'].min(), df['timestamp'].max()
                min_timestamp_overall = min(min_ts, min_timestamp_overall)
                max_timestamp_overall = max(max_ts, max_timestamp_overall)
            except Exception as e_ts_range:
                print(f"Warning: Could not get time range for {df_name}: {e_ts_range}")
        else:
            print(f"Info: No valid records remaining for {df_name} after timestamp conversion.")
    if 'screenshots' not in valid_dfs or valid_dfs['screenshots'].empty:
        print("ERROR: No valid screenshot data after timestamp processing. Cannot proceed.")
        return pd.DataFrame(), [], {}, {}, {}, False
    df_screenshots = valid_dfs['screenshots']
    kb_cols = ['timestamp', 'timestamp_ns', 'key_name', 'press_time_ns', 'release_time_ns']
    df_keyboard = valid_dfs.get('keyboard', pd.DataFrame(columns=kb_cols))
    for col in kb_cols:
        if col not in df_keyboard.columns: df_keyboard[col] = pd.NA
    mouse_cols = ['timestamp', 'timestamp_ns', 'action', 'button', 'start_pos', 'end_pos', 'press_time_ns', 'release_time_ns', 'trajectory']
    df_mouse = valid_dfs.get('mouse', pd.DataFrame(columns=mouse_cols))
    for col in mouse_cols:
         if col not in df_mouse.columns: df_mouse[col] = pd.NA
    results_cols = ['timestamp', 'timestamp_ns', 'result']
    df_results = valid_dfs.get('results', pd.DataFrame(columns=results_cols))
    if 'result' not in df_results.columns: df_results['result'] = pd.NA
    print("Sorting dataframes by timestamp...")
    df_screenshots.sort_values('timestamp', inplace=True, kind='mergesort')
    if not df_keyboard.empty: df_keyboard.sort_values('timestamp', inplace=True, kind='mergesort')
    if not df_mouse.empty: df_mouse.sort_values('timestamp', inplace=True, kind='mergesort')
    if not df_results.empty: df_results.sort_values('timestamp', inplace=True, kind='mergesort')
    action_window_td = pd.Timedelta(milliseconds=ACTION_PREDICTION_WINDOW_MS)
    reward_window_td = pd.Timedelta(seconds=REWARD_LOOKBACK_WINDOW_SEC)
    results_map = None
    if not df_results.empty and 'result' in df_results.columns and df_results['result'].notna().any():
        df_results['reward_value'] = df_results['result'].map(REWARD_MAPPING).fillna(1.0)
        df_results = df_results.dropna(subset=['timestamp'])
        if not df_results.empty:
            df_results = df_results.drop_duplicates(subset=['timestamp'], keep='last')
            df_results = df_results.set_index('timestamp').sort_index()
            results_map = df_results['reward_value']
            print(f"Created results map with {len(results_map)} entries for reward lookup.")
        else: print("Info: No valid results data remained after dropping NaT timestamps.")
    else: print("Info: No valid results data found. Using default reward weight 1.0 for all samples.")
    key_to_idx, idx_to_key = {}, {}
    num_keyboard_classes = 0
    if not df_keyboard.empty and 'key_name' in df_keyboard.columns and df_keyboard['key_name'].notna().any():
        potential_keys = df_keyboard['key_name'].dropna().astype(str).str.lower().str.strip()
        valid_keys_series = potential_keys[~potential_keys.isin(FORBIDDEN_KEYS) & (potential_keys != '')]
        unique_valid_keys = sorted(list(valid_keys_series.unique()))
        if unique_valid_keys:
             key_to_idx = {k: i for i, k in enumerate(unique_valid_keys)}
             idx_to_key = {v: k for k, v in key_to_idx.items()}
             num_keyboard_classes = len(key_to_idx)
             print(f"Keyboard vocabulary size: {num_keyboard_classes} (Valid keys found: {len(unique_valid_keys)})")
        else: print("Warning: No valid keyboard keys found in logs after filtering forbidden keys.")
    else: print("Info: No keyboard data or 'key_name' column found. Keyboard vocabulary empty.")
    mouse_action_to_idx = {}
    num_mouse_action_classes = 0
    if not df_mouse.empty and 'action' in df_mouse.columns and 'button' in df_mouse.columns:
        df_mouse_valid = df_mouse.dropna(subset=['action', 'button'])
        if not df_mouse_valid.empty:
            df_mouse_valid['action_mapped'] = df_mouse_valid['action'].map(MOUSE_ACTION_MAPPING).fillna('unknown')
            combined_actions = df_mouse_valid['action_mapped'] + '_' + df_mouse_valid['button']
            detected_target_actions = set()
            for target_action in MOUSE_ACTION_TYPES_TARGET:
                if target_action != 'no_action' and (combined_actions == target_action).any():
                    detected_target_actions.add(target_action)
            final_mouse_actions = ['no_action'] + sorted(list(detected_target_actions))
            mouse_action_to_idx = {action: i for i, action in enumerate(final_mouse_actions)}
            num_mouse_action_classes = len(mouse_action_to_idx)
            if 'action_mapped' in df_mouse_valid.columns:
                df_mouse = pd.merge(df_mouse, df_mouse_valid[['action_mapped']], left_index=True, right_index=True, how='left')
        else: print("Info: Mouse data found but no rows with valid action/button combinations.")
    if num_mouse_action_classes == 0:
        mouse_action_to_idx = {'no_action': 0}
        num_mouse_action_classes = 1
        print("Warning: No valid mouse actions found/mapped. Mouse AI vocabulary contains only 'no_action'.")
    print(f"Final Mouse action types ({num_mouse_action_classes}): {list(mouse_action_to_idx.keys())}")
    kb_use_press_time = (not df_keyboard.empty and 'press_time_ns' in df_keyboard.columns and df_keyboard['press_time_ns'].notna().sum() > 0.5 * df_keyboard['timestamp_ns'].notna().sum())
    mouse_use_press_time = (not df_mouse.empty and 'press_time_ns' in df_mouse.columns and df_mouse['press_time_ns'].notna().sum() > 0.5 * df_mouse['timestamp_ns'].notna().sum())
    kb_index_col = 'press_time_ns' if kb_use_press_time else 'timestamp_ns'
    mouse_index_col = 'press_time_ns' if mouse_use_press_time else 'timestamp_ns'
    print(f"Using '{kb_index_col}' for keyboard action alignment.")
    print(f"Using '{mouse_index_col}' for mouse action alignment.")
    keyboard_actions = None
    if not df_keyboard.empty and kb_index_col in df_keyboard.columns:
        df_keyboard_indexed = df_keyboard.dropna(subset=[kb_index_col])
        if not df_keyboard_indexed.empty:
            df_keyboard_indexed = df_keyboard_indexed[pd.to_numeric(df_keyboard_indexed[kb_index_col], errors='coerce').notna()]
            if not df_keyboard_indexed.empty:
                 df_keyboard_indexed[kb_index_col] = df_keyboard_indexed[kb_index_col].astype(np.int64)
                 df_keyboard_indexed = df_keyboard_indexed[~df_keyboard_indexed[kb_index_col].duplicated(keep='last')]
                 keyboard_actions = df_keyboard_indexed.set_index(kb_index_col).sort_index()
    if keyboard_actions is None: print("Warning: Could not create keyboard actions index.")
    mouse_actions = None
    if not df_mouse.empty and mouse_index_col in df_mouse.columns:
         df_mouse_indexed = df_mouse.dropna(subset=[mouse_index_col])
         if not df_mouse_indexed.empty:
             df_mouse_indexed = df_mouse_indexed[pd.to_numeric(df_mouse_indexed[mouse_index_col], errors='coerce').notna()]
             if not df_mouse_indexed.empty:
                 df_mouse_indexed[mouse_index_col] = df_mouse_indexed[mouse_index_col].astype(np.int64)
                 if 'action_mapped' not in df_mouse_indexed.columns: df_mouse_indexed['action_mapped'] = pd.NA
                 df_mouse_indexed = df_mouse_indexed[~df_mouse_indexed[mouse_index_col].duplicated(keep='last')]
                 mouse_actions = df_mouse_indexed.set_index(mouse_index_col).sort_index()
    if mouse_actions is None: print("Warning: Could not create mouse actions index.")
    kb_index_vals = keyboard_actions.index.values if keyboard_actions is not None else np.array([])
    mouse_index_vals = mouse_actions.index.values if mouse_actions is not None else np.array([])
    df_screenshots.reset_index(drop=True, inplace=True)
    screenshot_list_full = list(df_screenshots[['timestamp', 'filepath', 'timestamp_ns']].itertuples(index=True, name=None))
    df_screenshots = df_screenshots.drop(columns=['path_obj'], errors='ignore')
    aligned_data = []
    print("Aligning actions and rewards to screenshots...")
    skipped_no_action = 0
    skipped_invalid_mouse_pos = 0
    total_screenshots = len(screenshot_list_full)
    start_align_time = time.time()
    action_window_ns = int(ACTION_PREDICTION_WINDOW_MS * 1e6)
    no_action_idx_mouse = mouse_action_to_idx.get('no_action', 0)
    for df_idx, img_ts_pd, img_path, img_ts_ns in screenshot_list_full:
        if df_idx % 10000 == 0 and df_idx > 0:
             elapsed = time.time() - start_align_time
             rate = df_idx / elapsed if elapsed > 0 else 0
             eta_sec = (total_screenshots - df_idx) / rate if rate > 0 else 0
             eta_str = str(timedelta(seconds=int(eta_sec))) if eta_sec > 0 else "N/A"
             print(f"  Aligning progress: {df_idx}/{total_screenshots} ({rate:.1f} screenshots/sec, ETA: {eta_str})")
        action_window_end_ns = img_ts_ns + action_window_ns
        reward_window_start_pd = img_ts_pd - reward_window_td
        kb_target_indices = []
        if keyboard_actions is not None and num_keyboard_classes > 0 and len(kb_index_vals) > 0:
            try:
                kb_slice_idx_start = np.searchsorted(kb_index_vals, img_ts_ns, side='left')
                kb_slice_idx_end = np.searchsorted(kb_index_vals, action_window_end_ns, side='right')
                if kb_slice_idx_start < kb_slice_idx_end:
                    kb_slice = keyboard_actions.iloc[kb_slice_idx_start:kb_slice_idx_end]
                    if not kb_slice.empty and 'key_name' in kb_slice.columns:
                        keys_in_slice = kb_slice['key_name'].dropna()
                        keys_present = set(k for k in keys_in_slice if k in key_to_idx)
                        if keys_present:
                            kb_target_indices = sorted([key_to_idx[k] for k in keys_present])
            except Exception as e_kb_align:
                if df_idx % 5000 == 0: print(f"Warning: Error aligning keyboard action at index {df_idx}: {e_kb_align}")
        mouse_target_action_idx = no_action_idx_mouse
        mouse_target_pos_normalized = np.array([0.5, 0.5], dtype=np.float32)
        if mouse_actions is not None and num_mouse_action_classes > 1 and len(mouse_index_vals) > 0:
             try:
                 mouse_slice_idx_start = np.searchsorted(mouse_index_vals, img_ts_ns, side='left')
                 mouse_slice_idx_end = np.searchsorted(mouse_index_vals, action_window_end_ns, side='right')
                 if mouse_slice_idx_start < mouse_slice_idx_end:
                      mouse_slice = mouse_actions.iloc[mouse_slice_idx_start:mouse_slice_idx_end]
                      if not mouse_slice.empty and 'action_mapped' in mouse_slice.columns and 'button' in mouse_slice.columns:
                          action_button_combo_slice = mouse_slice['action_mapped'].fillna('unknown') + '_' + mouse_slice['button'].fillna('unknown')
                          found_action_in_window = False
                          for target_action_str in MOUSE_ACTION_TYPES_TARGET:
                              if target_action_str == 'no_action': continue
                              matches = (action_button_combo_slice == target_action_str)
                              if matches.any():
                                  first_match_idx = matches.idxmax()
                                  action_row = mouse_slice.loc[first_match_idx]
                                  pos_col_name = 'end_pos' if 'drag' in target_action_str else 'start_pos'
                                  pos_to_normalize = action_row.get(pos_col_name)
                                  if isinstance(pos_to_normalize, (list, tuple)) and len(pos_to_normalize) == 2:
                                      try:
                                          px, py = pos_to_normalize
                                          if isinstance(px, (int, float)) and isinstance(py, (int, float)) and not (math.isnan(px) or math.isnan(py)):
                                              x_norm = max(0.0, min(1.0, float(px) / screen_w))
                                              y_norm = max(0.0, min(1.0, float(py) / screen_h))
                                              if target_action_str in mouse_action_to_idx:
                                                  mouse_target_action_idx = mouse_action_to_idx[target_action_str]
                                                  mouse_target_pos_normalized = np.array([x_norm, y_norm], dtype=np.float32)
                                                  found_action_in_window = True
                                                  break
                                              else: print(f"Warning: Detected action '{target_action_str}' not in vocab!")
                                          else: skipped_invalid_mouse_pos += 1
                                      except (ValueError, TypeError, ZeroDivisionError) as e_norm: skipped_invalid_mouse_pos += 1
                                  else: skipped_invalid_mouse_pos += 1
             except Exception as e_mouse_align:
                 if df_idx % 5000 == 0: print(f"Warning: Error aligning mouse action at index {df_idx}: {e_mouse_align}")
        reward_weight = 1.0
        if results_map is not None:
             try:
                 relevant_results = results_map.loc[reward_window_start_pd : img_ts_pd]
                 if not relevant_results.empty:
                     reward_weight = relevant_results.iloc[-1]
             except Exception as e_reward: pass
        has_kb_action = bool(kb_target_indices)
        has_mouse_action = (mouse_target_action_idx != no_action_idx_mouse)
        if has_kb_action or has_mouse_action:
             aligned_data.append({
                 'img_path': img_path,
                 'kb_actions': kb_target_indices,
                 'mouse_action': mouse_target_action_idx,
                 'mouse_pos': mouse_target_pos_normalized.tolist(),
                 'reward_weight': float(reward_weight),
                 'original_df_index': df_idx
             })
        else:
            skipped_no_action += 1
    del df_keyboard, df_mouse, df_results, df_screenshots
    del keyboard_actions, mouse_actions, results_map
    cleanup_memory()
    t_end = time.time()
    print(f"--- Data Preprocessing Finished ({t_end - t_start:.2f}s) ---")
    print(f"Total aligned samples with actions: {len(aligned_data)}")
    print(f"Screenshots skipped (no relevant actions followed): {skipped_no_action}")
    if skipped_invalid_mouse_pos > 0: print(f"Mouse actions skipped/defaulted due to invalid/missing position data: {skipped_invalid_mouse_pos}")
    if not aligned_data:
         print(f"\nWARNING: No aligned data points with actions were generated. Training cannot proceed.")
         return pd.DataFrame(), screenshot_list_full, key_to_idx, idx_to_key, mouse_action_to_idx, False
    df_aligned = pd.DataFrame(aligned_data)
    return df_aligned, screenshot_list_full, key_to_idx, idx_to_key, mouse_action_to_idx, True
class ActionDatasetSequence(Dataset):
    def __init__(self, aligned_data_df, screenshot_list_full, sequence_length, img_transform,
                 target_type='keyboard', key_vocab_size=0, mouse_action_vocab_size=0, mouse_action_to_idx=None):
        self.sequence_length = sequence_length
        self.transform = img_transform
        self.target_type = target_type
        self.key_vocab_size = key_vocab_size
        self.mouse_action_vocab_size = mouse_action_vocab_size
        self.mouse_action_to_idx = mouse_action_to_idx if mouse_action_to_idx is not None else {}
        self.no_action_idx = self.mouse_action_to_idx.get('no_action', 0) if self.mouse_action_vocab_size > 0 else 0
        self.aligned_items = []
        self.screenshot_lookup = {}
        self.valid_indices_in_aligned = []
        if screenshot_list_full:
            for i, (df_idx, _ts_pd, path, _ts_ns) in enumerate(screenshot_list_full):
                self.screenshot_lookup[df_idx] = {'path': str(path), 'list_pos': i}
        else:
            print(f"Warning: Initializing {target_type} dataset with empty screenshot_list_full.")
        if aligned_data_df is None or aligned_data_df.empty:
            print(f"Warning: Initializing {target_type} dataset with empty aligned_data_df.")
        elif not self.screenshot_lookup:
             print(f"Warning: Initializing {target_type} dataset - screenshot lookup is empty despite having aligned data.")
        else:
            try:
                required_cols_aligned = {'original_df_index', 'kb_actions', 'mouse_action', 'mouse_pos', 'reward_weight', 'img_path'}
                if not required_cols_aligned.issubset(aligned_data_df.columns):
                    raise KeyError(f"Missing expected columns in aligned_data_df: {required_cols_aligned - set(aligned_data_df.columns)}")
                self.aligned_items = aligned_data_df[list(required_cols_aligned)].to_dict('records')
            except KeyError as e:
                 print(f"ERROR: Failed reading aligned_data_df for {target_type} dataset: {e}.")
                 self.aligned_items = []
            except Exception as e_conv:
                 print(f"ERROR converting aligned_data_df to records for {target_type} dataset: {e_conv}.")
                 self.aligned_items = []
            num_aligned = len(self.aligned_items)
            if num_aligned > 0:
                print(f"Building valid sequence indices for {target_type} dataset from {num_aligned} aligned samples...")
                for aligned_idx, item in enumerate(self.aligned_items):
                     target_original_df_idx = item['original_df_index']
                     lookup_info = self.screenshot_lookup.get(target_original_df_idx)
                     if not lookup_info: continue
                     target_frame_list_pos = lookup_info['list_pos']
                     if target_frame_list_pos >= self.sequence_length - 1:
                          sequence_indices_exist = True
                          for k in range(self.sequence_length):
                              expected_original_df_idx = target_original_df_idx - (self.sequence_length - 1 - k)
                              if expected_original_df_idx not in self.screenshot_lookup:
                                  sequence_indices_exist = False; break
                          if sequence_indices_exist:
                               self.valid_indices_in_aligned.append(aligned_idx)
                print(f"Finished building valid sequence indices for {target_type}.")
        num_valid_seq = len(self.valid_indices_in_aligned)
        print(f"Dataset '{self.target_type}': Input aligned samples={len(self.aligned_items)}, Found {num_valid_seq} valid sequence end points.")
        if not self.valid_indices_in_aligned and len(self.aligned_items) > 0:
            print(f"WARNING: No valid sequences could be formed for {self.target_type} dataset! Check SEQUENCE_LENGTH ({self.sequence_length}) vs data continuity.")
        self._dummy_image = torch.zeros((3, IMG_SIZE[0], IMG_SIZE[1]), dtype=torch.float32)
        self._dummy_image_seq = self._dummy_image.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)
        self._dummy_kb_target = torch.zeros(self.key_vocab_size, dtype=torch.float32) if self.key_vocab_size > 0 else torch.empty(0, dtype=torch.float32)
        self._dummy_mouse_target_action = torch.zeros(self.mouse_action_vocab_size, dtype=torch.float32) if self.mouse_action_vocab_size > 0 else torch.empty(0, dtype=torch.float32)
        if self.mouse_action_vocab_size > 0 and 0 <= self.no_action_idx < self.mouse_action_vocab_size:
             self._dummy_mouse_target_action[self.no_action_idx] = 1.0
        self._dummy_mouse_target_pos = torch.tensor([0.5, 0.5], dtype=torch.float32)
        self._dummy_weight = torch.tensor(1.0, dtype=torch.float32)
    def __len__(self):
        return len(self.valid_indices_in_aligned)
    def __getitem__(self, idx):
        if not self.valid_indices_in_aligned or idx < 0 or idx >= len(self.valid_indices_in_aligned):
            return self._return_dummy()
        aligned_item_index = -1
        target_original_df_idx = -1
        try:
            aligned_item_index = self.valid_indices_in_aligned[idx]
            item = self.aligned_items[aligned_item_index]
            target_original_df_idx = item['original_df_index']
            image_sequence = []
            load_errors_in_seq = 0
            for i in range(self.sequence_length):
                 current_df_idx = target_original_df_idx - (self.sequence_length - 1 - i)
                 img_tensor = self._dummy_image
                 found_path_info = self.screenshot_lookup.get(current_df_idx)
                 if found_path_info:
                     img_path = found_path_info['path']
                     try:
                         if not Path(img_path).is_file(): raise FileNotFoundError(f"File not found: {img_path}")
                         with Image.open(img_path) as img:
                             img_rgb = img.convert('RGB')
                             img_tensor = self.transform(img_rgb)
                     except Exception as img_err:
                         load_errors_in_seq += 1
                         if load_errors_in_seq < 3 or load_errors_in_seq % 50 == 0:
                             print(f"Warning: Img load failed (Dataset: {self.target_type}, Item: {idx}, Frame Idx: {current_df_idx}, Path: {img_path}). Using dummy. Err: {type(img_err).__name__}")
                 else:
                     load_errors_in_seq += 1
                     if load_errors_in_seq < 3 or load_errors_in_seq % 50 == 0:
                        print(f"Warning: Screenshot lookup failed for expected df_idx {current_df_idx} (Item {idx}). Using dummy frame.")
                 image_sequence.append(img_tensor)
            if load_errors_in_seq > self.sequence_length // 2:
                 return self._return_dummy()
            try:
                image_sequence_tensor = torch.stack(image_sequence, dim=0)
            except Exception as stack_err:
                print(f"ERROR: Failed to stack image tensors (Item {idx}, Type: {self.target_type}). Error: {stack_err}")
                return self._return_dummy()
            try:
                weight_val = item.get('reward_weight', 1.0)
                weight = torch.tensor(float(weight_val) if isinstance(weight_val, (int, float)) and not math.isnan(weight_val) else 1.0, dtype=torch.float32)
            except (ValueError, TypeError):
                weight = self._dummy_weight
            if self.target_type == 'keyboard':
                if self.key_vocab_size == 0: return image_sequence_tensor, self._dummy_kb_target, weight
                kb_target = torch.zeros(self.key_vocab_size, dtype=torch.float32)
                kb_indices = item.get('kb_actions', [])
                if kb_indices:
                     try:
                         valid_indices = [int(i) for i in kb_indices if isinstance(i, (int, np.integer)) and 0 <= int(i) < self.key_vocab_size]
                         if valid_indices: kb_target[valid_indices] = 1.0
                     except (ValueError, TypeError): pass
                return image_sequence_tensor, kb_target, weight
            elif self.target_type == 'mouse':
                 if self.mouse_action_vocab_size == 0: return image_sequence_tensor, (self._dummy_mouse_target_action, self._dummy_mouse_target_pos), weight
                 action_target = self._dummy_mouse_target_action.clone()
                 action_idx_raw = item.get('mouse_action', self.no_action_idx)
                 try:
                     valid_action_idx = int(action_idx_raw) if isinstance(action_idx_raw, (int, np.integer)) and 0 <= int(action_idx_raw) < self.mouse_action_vocab_size else self.no_action_idx
                     action_target = torch.zeros(self.mouse_action_vocab_size, dtype=torch.float32)
                     action_target[valid_action_idx] = 1.0
                 except (ValueError, TypeError): action_target = self._dummy_mouse_target_action.clone()
                 pos_target = self._dummy_mouse_target_pos.clone()
                 raw_pos = item.get('mouse_pos', [0.5, 0.5])
                 if isinstance(raw_pos, (list, tuple)) and len(raw_pos) == 2:
                      try:
                          x = max(0.0, min(1.0, float(raw_pos[0])))
                          y = max(0.0, min(1.0, float(raw_pos[1])))
                          if not (math.isnan(x) or math.isnan(y)):
                            pos_target = torch.tensor([x, y], dtype=torch.float32)
                      except (ValueError, TypeError): pass
                 return image_sequence_tensor, (action_target, pos_target), weight
            else:
                print(f"CRITICAL ERROR: Invalid target_type '{self.target_type}' in __getitem__")
                return self._return_dummy()
        except IndexError as e:
             print(f"ERROR: IndexError in __getitem__ (idx {idx}, aligned_idx {aligned_item_index}, Type: {self.target_type}). Error: {e}")
             return self._return_dummy()
        except KeyError as e:
            print(f"ERROR: KeyError in __getitem__ (idx {idx}, Type: {self.target_type}). Missing key '{e}'.")
            return self._return_dummy()
        except Exception as e:
            print(f"CRITICAL ERROR in ActionDatasetSequence.__getitem__ (idx {idx}, Type: {self.target_type}): {type(e).__name__} - {e}")
            traceback.print_exc(limit=1)
            return self._return_dummy()
    def _return_dummy(self):
        if self.target_type == 'keyboard':
            return self._dummy_image_seq.clone(), self._dummy_kb_target.clone(), self._dummy_weight.clone()
        elif self.target_type == 'mouse':
            return self._dummy_image_seq.clone(), (self._dummy_mouse_target_action.clone(), self._dummy_mouse_target_pos.clone()), self._dummy_weight.clone()
        else:
            return self._dummy_image_seq.clone(), torch.empty(0), self._dummy_weight.clone()
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            pe_slice = self.pe[:, :x.size(1), :]
            x = x + pe_slice
            return self.dropout(x)
        except RuntimeError as e_pe:
            print(f"ERROR applying Positional Encoding: {e_pe}")
            print(f"  Input shape: {x.shape}, PE slice shape attempt: (1, {x.size(1)}, {self.pe.shape[2]})")
            raise
class VisionSequenceModel(nn.Module):
    def __init__(self, vision_model_name, pretrained_model_path, img_size, sequence_length,
                 transformer_d_model, transformer_nhead, transformer_num_layers,
                 transformer_dim_feedforward, transformer_dropout=0.1,
                 num_keyboard_classes=0, num_mouse_actions=0, freeze_vision=False):
        super().__init__()
        self.sequence_length = sequence_length
        self.transformer_d_model = transformer_d_model
        self.vision_model_name_used = vision_model_name
        self.num_keyboard_classes = num_keyboard_classes
        self.num_mouse_actions = num_mouse_actions
        self.freeze_vision = freeze_vision
        self.vision_feature_dim = 0
        print(f"\nInitializing Vision Backbone: {vision_model_name}")
        model_weights_path_safe = pretrained_model_path / "model.safetensors"
        model_weights_path_bin = pretrained_model_path / "pytorch_model.bin"
        weights_path_to_use = None
        load_safe = False
        if model_weights_path_safe.is_file() and SAFETENSORS_AVAILABLE:
            weights_path_to_use = model_weights_path_safe; load_safe = True
            print(f"Using preferred weights file: {weights_path_to_use.name}")
        elif model_weights_path_bin.is_file():
            weights_path_to_use = model_weights_path_bin
            print(f"Using fallback weights file: {weights_path_to_use.name}")
        else:
            raise FileNotFoundError(f"CRITICAL: Could not find 'model.safetensors' or 'pytorch_model.bin' in {pretrained_model_path}")
        try:
            print(f"Creating vision backbone '{vision_model_name}' structure (pretrained=False)...")
            self.vision_backbone = timm.create_model(
                vision_model_name, pretrained=False, num_classes=0, global_pool='avg'
            )
            self.vision_feature_dim = self.vision_backbone.num_features
            if self.vision_feature_dim <= 0: raise ValueError(f"ViT feature dimension <= 0: {self.vision_feature_dim}")
            print(f"Vision backbone created. Feature Dimension: {self.vision_feature_dim}")
            print(f"Loading local weights from: {weights_path_to_use.name}...")
            if load_safe:
                state_dict = load_safetensors_file(weights_path_to_use, device='cpu')
            else:
                try:
                    state_dict = torch.load(weights_path_to_use, map_location='cpu', weights_only=True)
                except (RuntimeError, pickle.UnpicklingError, AttributeError, TypeError):
                    print("  Warning: Loading .bin with weights_only=True failed. Trying weights_only=False (less secure).")
                    try:
                        state_dict = torch.load(weights_path_to_use, map_location='cpu', weights_only=False)
                    except pickle.UnpicklingError as e_pickle:
                        print(f"\nERROR: UnpicklingError loading {weights_path_to_use.name}. File may be corrupt or malicious.\nDetails: {e_pickle}\n")
                        raise RuntimeError("UnpicklingError loading weights") from e_pickle
                    except Exception as e_load_bin_unsafe:
                         print(f"ERROR loading .bin weights (unsafe mode) {weights_path_to_use.name}: {e_load_bin_unsafe}")
                         raise RuntimeError("Failed to load .bin weights") from e_load_bin_unsafe
            if isinstance(state_dict, dict):
                wrapper_keys = ['state_dict', 'model', 'module']; processed_state_dict = state_dict
                for key in wrapper_keys:
                    if key in state_dict and isinstance(state_dict[key], dict):
                         processed_state_dict = state_dict[key]; break
                state_dict = processed_state_dict
                prefixes_to_remove = ['module.', 'backbone.', '_orig_mod.']; cleaned_state_dict = {}
                for k, v in state_dict.items():
                    temp_k = k
                    for prefix in prefixes_to_remove:
                        if temp_k.startswith(prefix): temp_k = temp_k[len(prefix):]
                    cleaned_state_dict[temp_k] = v
                state_dict = cleaned_state_dict
            print("Loading state dict into model (strict=False)...")
            load_result = self.vision_backbone.load_state_dict(state_dict, strict=False)
            print(f"  Weight loading analysis:")
            expected_missing = {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            missing_in_backbone = [k for k in load_result.missing_keys if k not in expected_missing]
            if missing_in_backbone: print(f"    Warning: Missing Keys expected in backbone: {missing_in_backbone}")
            elif load_result.missing_keys: print(f"    Note: Missing keys match expected (removed head/fc_norm): {load_result.missing_keys}")
            else: print("    No missing keys.")
            expected_unexpected = {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias', 'norm.weight', 'norm.bias'}
            unexpected_actual = [k for k in load_result.unexpected_keys if k not in expected_unexpected]
            if unexpected_actual: print(f"    Warning: Unexpected Keys found in checkpoint file: {unexpected_actual}")
            elif load_result.unexpected_keys: print(f"    Note: Unexpected keys in file match expected (removed head/norm), ignored: {load_result.unexpected_keys}")
            else: print("    No unexpected keys.")
            print(f"Successfully attempted to load weights into '{vision_model_name}'.")
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f"ERROR during vision backbone setup: {e}"); raise
        except Exception as e_load:
            print(f"ERROR creating/loading vision backbone '{vision_model_name}': {e_load}")
            traceback.print_exc(); raise RuntimeError("Cannot create/load vision backbone") from e_load
        if self.freeze_vision:
            print("Freezing vision backbone weights.")
            for param in self.vision_backbone.parameters(): param.requires_grad = False
            self.vision_backbone.eval()
        if self.vision_feature_dim != self.transformer_d_model:
            self.input_proj = nn.Linear(self.vision_feature_dim, self.transformer_d_model)
            print(f"Added Input Projection Layer: {self.vision_feature_dim} -> {self.transformer_d_model}")
        else:
            self.input_proj = nn.Identity()
            print("ViT feature dimension matches Transformer d_model. Using Identity projection.")
        self.pos_encoder = PositionalEncoding(self.transformer_d_model, max(50, sequence_length * 2), transformer_dropout)
        print(f"Initialized Positional Encoding (d_model={self.transformer_d_model}, max_len={self.pos_encoder.pe.size(1)})")
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.transformer_d_model, nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout,
                activation="gelu", batch_first=True, norm_first=True
            )
            encoder_norm = nn.LayerNorm(self.transformer_d_model)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers, norm=encoder_norm)
            print(f"Initialized Transformer Encoder: Layers={transformer_num_layers}, Heads={transformer_nhead}, d_ff={transformer_dim_feedforward}, Norm=True")
        except Exception as e_transformer:
             print(f"ERROR initializing Transformer Encoder: {e_transformer}")
             print(f"  Check Params: d_model={self.transformer_d_model}, nhead={transformer_nhead}, ff={transformer_dim_feedforward}, layers={transformer_num_layers}")
             raise RuntimeError("Failed to initialize Transformer Encoder") from e_transformer
        self.keyboard_head = None
        if num_keyboard_classes > 0:
             try:
                 self.keyboard_head = nn.Linear(self.transformer_d_model, num_keyboard_classes)
                 print(f"Initialized Keyboard Head (Input: {self.transformer_d_model}, Output: {num_keyboard_classes})")
             except Exception as e_head: raise RuntimeError("Failed to initialize Keyboard Head") from e_head
        self.mouse_action_head = None
        self.mouse_pos_head = None
        if num_mouse_actions > 0:
             try:
                 self.mouse_action_head = nn.Linear(self.transformer_d_model, num_mouse_actions)
                 print(f"Initialized Mouse Action Head (Input: {self.transformer_d_model}, Output: {num_mouse_actions})")
                 pos_head_intermediate = max(64, self.transformer_d_model // 4)
                 self.mouse_pos_head = nn.Sequential(
                     nn.Linear(self.transformer_d_model, pos_head_intermediate), nn.GELU(),
                     nn.Dropout(0.1), nn.Linear(pos_head_intermediate, 2), nn.Sigmoid()
                 )
                 print(f"Initialized Mouse Position Head (Input: {self.transformer_d_model}, Output: 2, Intermediate: {pos_head_intermediate})")
             except Exception as e_head: raise RuntimeError("Failed to initialize Mouse Heads") from e_head

    def forward(self, x_seq):
        if x_seq.ndim != 5: raise ValueError(f"Input tensor shape error: Expected 5 dims, got {x_seq.ndim} ({x_seq.shape})")
        batch_size, seq_len, C, H, W = x_seq.shape
        x_flat = x_seq.view(batch_size * seq_len, C, H, W)
        vision_features_flat = None
        try:
            with torch.set_grad_enabled(not self.freeze_vision):
                 original_bb_mode = self.vision_backbone.training
                 if self.freeze_vision and original_bb_mode: self.vision_backbone.eval()
                 vision_features_flat = self.vision_backbone(x_flat)
                 if self.freeze_vision and original_bb_mode: self.vision_backbone.train()
        except Exception as e_vit_fwd:
            print(f"ERROR during Vision Backbone forward pass: {e_vit_fwd}"); raise
        try:
            vision_features_seq = vision_features_flat.view(batch_size, seq_len, self.vision_feature_dim)
        except RuntimeError as e_reshape:
            print(f"ERROR reshaping ViT features. Flat shape: {vision_features_flat.shape if vision_features_flat is not None else 'None'}. Target: ({batch_size}, {seq_len}, {self.vision_feature_dim}). Error: {e_reshape}"); raise
        final_features = None
        try:
            transformer_input = self.input_proj(vision_features_seq)
            transformer_input_pe = self.pos_encoder(transformer_input)
            memory = self.transformer_encoder(transformer_input_pe)
            if memory is not None and memory.ndim == 3 and memory.shape[1] > 0:
                 final_features = memory[:, -1, :]
            else: raise ValueError(f"Transformer output (memory) unexpected shape: {memory.shape if memory is not None else 'None'}")
        except Exception as e_transformer_fwd:
            print(f"ERROR during Transformer forward pass: {e_transformer_fwd}"); raise
        keyboard_logits, mouse_action_logits, mouse_pos_pred = None, None, None
        try:
            if final_features is None: raise ValueError("final_features is None before output heads.")
            if self.keyboard_head is not None: keyboard_logits = self.keyboard_head(final_features)
            if self.mouse_action_head is not None: mouse_action_logits = self.mouse_action_head(final_features)
            if self.mouse_pos_head is not None: mouse_pos_pred = self.mouse_pos_head(final_features)
        except Exception as e_head_fwd:
            print(f"ERROR during Output Head forward pass: {e_head_fwd}"); raise
        return keyboard_logits, mouse_action_logits, mouse_pos_pred
def train_model(model, model_name, train_loader, val_loader, optimizer, scheduler,
                keyboard_criterion=None, mouse_action_criterion=None, mouse_pos_criterion=None,
                device='cpu', batch_size_start=16, time_limit_sec=None):
    print(f"\n--- Starting Training: {model_name} ---")
    time_limit_str = str(timedelta(seconds=time_limit_sec)) if time_limit_sec else "None"
    print(f"  Device: {device}, Initial Batch Size: {batch_size_start}, Time Limit: {time_limit_str}")
    print(f"  Max Epochs: {EPOCHS_MAX}, Early Stopping Patience: {PATIENCE_EARLY_STOPPING}")
    train_batches = len(train_loader) if train_loader is not None else 0
    val_batches = len(val_loader) if val_loader is not None else 0
    print(f"  Train Batches: {train_batches}, Val Batches: {val_batches}")
    print(f"  Gradient Clipping Norm: {GRADIENT_CLIP_VALUE}")
    if not train_loader or train_batches == 0:
         print("ERROR: Train loader is None or empty. Cannot start training.")
         return None, 0, batch_size_start, None, False, "Skipped - No training data"
    if not model:
         print("ERROR: Model object is None. Cannot start training.")
         return None, 0, batch_size_start, None, False, "Skipped - Model is None"
    training_start_time = time.time()
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    actual_epochs_run = 0
    training_stopped_reason = f"Reached max epochs ({EPOCHS_MAX})"
    current_batch_size = batch_size_start
    best_model_state_cpu = None
    saved_best_state_flag = False
    use_amp = (str(device) == 'cuda')
    scaler = GradScaler(enabled=use_amp)
    autocast_device_type = device.type
    print(f"  AMP Enabled: {use_amp} (Autocast Device Type: {autocast_device_type})")
    if use_amp: cleanup_memory()
    try:
        for epoch in range(EPOCHS_MAX):
            epoch_start_time = time.time()
            actual_epochs_run = epoch + 1
            print(f"\n--- Epoch {actual_epochs_run}/{EPOCHS_MAX} ---")
            model.train()
            if hasattr(model, 'freeze_vision') and model.freeze_vision and hasattr(model, 'vision_backbone'):
                 model.vision_backbone.eval()
            train_loss_accum = 0.0
            samples_processed_train = 0
            nan_skipped_batches = 0
            oom_occurred_epoch_train = False
            other_error_batches_train = 0
            batch_iterator = iter(train_loader)
            batch_idx = 0
            while True:
                 if time_limit_sec is not None and (time.time() - training_start_time > time_limit_sec):
                     print(f"\nINFO: Time limit ({timedelta(seconds=time_limit_sec)}) reached during training epoch {epoch + 1}. Stopping.")
                     training_stopped_reason = "Time limit reached"
                     return model, actual_epochs_run -1, current_batch_size, best_model_state_cpu, saved_best_state_flag, training_stopped_reason
                 batch_data = None
                 batch_processed_successfully = False
                 try:
                     try:
                         batch_data = next(batch_iterator); batch_idx += 1
                     except StopIteration: break
                     except (OSError, ConnectionResetError, TimeoutError, BrokenPipeError, EOFError) as e_load_batch_io:
                         print(f"Warning: I/O/Worker Error loading train batch {batch_idx+1}: {type(e_load_batch_io).__name__}. Skipping."); other_error_batches_train += 1; cleanup_memory(); continue
                     except Exception as e_load_batch:
                         print(f"Warning: Error loading train batch {batch_idx+1}: {e_load_batch}. Skipping."); other_error_batches_train += 1; cleanup_memory(); continue
                     if not isinstance(batch_data, (list, tuple)) or len(batch_data) < 2:
                         print(f"Warn: Invalid batch data structure (train batch {batch_idx}). Skipping."); other_error_batches_train += 1; cleanup_memory(); continue
                     images_seq = batch_data[0]
                     if not isinstance(images_seq, torch.Tensor):
                         print(f"Warn: Image sequence is not a tensor (train batch {batch_idx}, Type: {type(images_seq)}). Skipping."); other_error_batches_train += 1; continue
                     if images_seq.shape == torch.Size([BATCH_SIZE_INITIAL, SEQUENCE_LENGTH, 3, IMG_SIZE[0], IMG_SIZE[1]]) and torch.all(images_seq == 0):
                          other_error_batches_train += 1; continue
                     images_seq = images_seq.to(device, non_blocking=PIN_MEMORY)
                     weights = batch_data[-1].to(device, non_blocking=PIN_MEMORY)
                     current_batch_size_actual = images_seq.size(0)
                     if current_batch_size_actual == 0: continue
                     expected_shape = (current_batch_size_actual, SEQUENCE_LENGTH, 3, IMG_SIZE[0], IMG_SIZE[1])
                     if images_seq.shape != expected_shape:
                         print(f"Warning: Unexpected image batch shape {images_seq.shape} vs expected {expected_shape}. Skipping."); other_error_batches_train += 1; cleanup_memory(); continue
                     targets_kb, targets_mouse_act, targets_mouse_pos = None, None, None
                     valid_targets = True
                     if model_name == 'Keyboard AI':
                         if not isinstance(batch_data[1], torch.Tensor): valid_targets = False
                         else:
                             targets_kb = batch_data[1].to(device, non_blocking=PIN_MEMORY)
                             if targets_kb.shape[0] != current_batch_size_actual or targets_kb.ndim != 2 or targets_kb.shape[1] != model.num_keyboard_classes: valid_targets = False
                     elif model_name == 'Mouse AI':
                         if not isinstance(batch_data[1], (list, tuple)) or len(batch_data[1]) != 2: valid_targets = False
                         else:
                             targets_mouse_act_raw, targets_mouse_pos_raw = batch_data[1]
                             if not isinstance(targets_mouse_act_raw, torch.Tensor) or not isinstance(targets_mouse_pos_raw, torch.Tensor): valid_targets = False
                             else:
                                 targets_mouse_act = targets_mouse_act_raw.to(device, non_blocking=PIN_MEMORY)
                                 targets_mouse_pos = targets_mouse_pos_raw.to(device, non_blocking=PIN_MEMORY)
                                 if targets_mouse_act.shape[0] != current_batch_size_actual or targets_mouse_act.ndim != 2 or targets_mouse_act.shape[1] != model.num_mouse_actions: valid_targets = False
                                 if targets_mouse_pos.shape[0] != current_batch_size_actual or tuple(targets_mouse_pos.shape[1:]) != (2,): valid_targets = False
                     if not valid_targets:
                          print(f"Warn: Invalid target data structure/shape (train batch {batch_idx}). Skipping."); other_error_batches_train += 1; cleanup_memory(); continue
                     optimizer.zero_grad(set_to_none=True)
                     with autocast(device_type=autocast_device_type, enabled=use_amp):
                         kb_logits, mouse_act_logits, mouse_pos_pred = model(images_seq)
                         loss = torch.tensor(0.0, device=device)
                         batch_loss_unweighted = torch.tensor(0.0, device=device)
                         loss_calculated = False
                         try:
                             if model_name == 'Keyboard AI' and keyboard_criterion is not None and kb_logits is not None:
                                 if kb_logits.shape == targets_kb.shape:
                                     batch_loss_per_item_kb = keyboard_criterion(kb_logits, targets_kb)
                                     if batch_loss_per_item_kb.ndim == 2:
                                          batch_loss_unweighted = batch_loss_per_item_kb.mean(dim=1); loss_calculated = True
                                     else: print(f"Warn: Unexpected KB loss shape {batch_loss_per_item_kb.shape}.")
                             elif model_name == 'Mouse AI':
                                 loss_action, loss_pos = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                                 action_loss_ok, pos_loss_ok = False, False
                                 if mouse_action_criterion is not None and mouse_act_logits is not None and mouse_act_logits.shape == targets_mouse_act.shape:
                                     loss_action_raw = mouse_action_criterion(mouse_act_logits, targets_mouse_act)
                                     if loss_action_raw.ndim == 2: loss_action = loss_action_raw.mean(dim=1); action_loss_ok = True
                                 if mouse_pos_criterion is not None and mouse_pos_pred is not None and mouse_pos_pred.shape == targets_mouse_pos.shape:
                                     loss_pos_raw = mouse_pos_criterion(mouse_pos_pred, targets_mouse_pos)
                                     if loss_pos_raw.ndim == 2: loss_pos = loss_pos_raw.mean(dim=1); pos_loss_ok = True
                                 valid_mouse_losses = []
                                 if action_loss_ok: valid_mouse_losses.append(loss_action)
                                 if pos_loss_ok: valid_mouse_losses.append(loss_pos * 0.5)
                                 if valid_mouse_losses:
                                     batch_loss_unweighted = torch.stack(valid_mouse_losses).sum(dim=0); loss_calculated = True
                             if loss_calculated and batch_loss_unweighted.shape == weights.shape:
                                 loss = (batch_loss_unweighted * weights.float()).mean()
                             elif loss_calculated:
                                 print(f"Warn: Loss/Weight shape mismatch ({batch_loss_unweighted.shape} vs {weights.shape}). Using unweighted mean."); loss = batch_loss_unweighted.mean()
                         except Exception as e_loss_calc:
                             print(f"ERROR calculating loss (train batch {batch_idx}): {e_loss_calc}"); loss = torch.tensor(float('nan'), device=device)
                     if not torch.isfinite(loss).all():
                         print(f"Warning: NaN/Inf loss detected (train batch {batch_idx}). Loss: {loss.item()}. Skipping step."); nan_skipped_batches += 1; cleanup_memory(); continue
                     try: scaler.scale(loss).backward()
                     except torch.cuda.OutOfMemoryError:
                         print(f"\nERROR: CUDA OOM during backward() (train batch {batch_idx})!"); oom_occurred_epoch_train = True; cleanup_memory(); optimizer.zero_grad(set_to_none=True); break
                     except RuntimeError as e_backward:
                         print(f"ERROR: RuntimeError during backward() (train batch {batch_idx}): {e_backward}"); other_error_batches_train += 1; cleanup_memory(); optimizer.zero_grad(set_to_none=True); continue
                     grads_are_finite = False
                     try:
                         scaler.unscale_(optimizer)
                         found_nan_grad = False; param_with_nan = None
                         for name, param in model.named_parameters():
                             if param.grad is not None and not torch.isfinite(param.grad).all():
                                 found_nan_grad = True; param_with_nan = name; break
                         if not found_nan_grad: grads_are_finite = True
                         else: print(f"Warning: NaN/Inf gradient in '{param_with_nan}' (train batch {batch_idx}). Skipping step.")
                     except RuntimeError as e_unscale: print(f"ERROR during scaler.unscale_() (train batch {batch_idx}): {e_unscale}")
                     if grads_are_finite:
                         total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
                         if not torch.isfinite(total_norm):
                              print(f"Warning: NaN/Inf grad norm ({total_norm:.2f}) *after* clipping (train batch {batch_idx}). Skipping step."); nan_skipped_batches += 1; optimizer.zero_grad(set_to_none=True); cleanup_memory(); continue
                         scaler.step(optimizer)
                         scaler.update()
                         batch_processed_successfully = True
                     else:
                         nan_skipped_batches += 1; optimizer.zero_grad(set_to_none=True); cleanup_memory(); continue
                     if batch_processed_successfully:
                         train_loss_accum += loss.item() * current_batch_size_actual
                         samples_processed_train += current_batch_size_actual
                     del loss, images_seq, weights, batch_data, kb_logits, mouse_act_logits, mouse_pos_pred
                     if targets_kb is not None: del targets_kb
                     if targets_mouse_act is not None: del targets_mouse_act, targets_mouse_pos, targets_mouse_act_raw, targets_mouse_pos_raw
                 except torch.cuda.OutOfMemoryError:
                     print(f"\nERROR: CUDA OOM in training (epoch {epoch+1}, batch {batch_idx})!"); oom_occurred_epoch_train = True; cleanup_memory(); optimizer.zero_grad(set_to_none=True); break
                 except RuntimeError as e_runtime:
                     print(f"ERROR: Runtime error during train batch {batch_idx}: {e_runtime}"); traceback.print_exc(limit=1); other_error_batches_train += 1; cleanup_memory(); optimizer.zero_grad(set_to_none=True); continue
                 except Exception as e_batch:
                     print(f"ERROR: Unexpected error during train batch {batch_idx}: {type(e_batch).__name__} - {e_batch}"); traceback.print_exc(limit=1); other_error_batches_train += 1; cleanup_memory(); optimizer.zero_grad(set_to_none=True); continue
            if oom_occurred_epoch_train: print(f"Epoch {epoch+1} training interrupted by OOM.")
            if other_error_batches_train > 0: print(f"Epoch {epoch+1} had {other_error_batches_train} non-OOM training batch errors.")
            if nan_skipped_batches > 0: print(f"Epoch {epoch+1} had {nan_skipped_batches} batches skipped due to NaN/Inf.")
            avg_train_loss = train_loss_accum / samples_processed_train if samples_processed_train > 0 else float('inf')
            current_lr = optimizer.param_groups[0]['lr']
            avg_val_loss = float('inf')
            val_samples_processed = 0
            oom_during_val = False
            other_error_batches_val = 0
            run_validation = (val_loader is not None and val_batches > 0 and not oom_occurred_epoch_train and samples_processed_train > 0)
            if not run_validation:
                print(f"Epoch {epoch+1}: Skipping validation.")
                if oom_occurred_epoch_train: reason = "OOM in train"
                elif samples_processed_train == 0: reason = "No train batches succeeded"
                else: reason = "No validation data/loader"
                print(f"  Reason: {reason}")
            else:
                model.eval()
                val_loss_accum = 0.0
                val_batch_idx = 0
                val_iterator = iter(val_loader)
                while True:
                     if time_limit_sec is not None and (time.time() - training_start_time > time_limit_sec):
                         print(f"\nINFO: Time limit reached during validation epoch {epoch + 1}. Stopping.")
                         training_stopped_reason = "Time limit reached"
                         return model, actual_epochs_run, current_batch_size, best_model_state_cpu, saved_best_state_flag, training_stopped_reason
                     batch_data_val = None
                     try:
                         try:
                             batch_data_val = next(val_iterator); val_batch_idx += 1
                         except StopIteration: break
                         except (OSError, ConnectionResetError, TimeoutError, BrokenPipeError, EOFError) as e_load_batch_io_val:
                             print(f"Warning: I/O/Worker Error loading val batch {val_batch_idx}: {e_load_batch_io_val}. Skipping."); other_error_batches_val += 1; cleanup_memory(); continue
                         except Exception as e_load_batch_val:
                             print(f"Warning: Error loading val batch {val_batch_idx}: {e_load_batch_val}. Skipping."); other_error_batches_val += 1; cleanup_memory(); continue
                         if not isinstance(batch_data_val, (list, tuple)) or len(batch_data_val) < 2: continue
                         images_seq_val = batch_data_val[0]
                         if not isinstance(images_seq_val, torch.Tensor): continue
                         images_seq_val = images_seq_val.to(device, non_blocking=PIN_MEMORY)
                         current_val_batch_size = images_seq_val.size(0)
                         if current_val_batch_size == 0: continue
                         with torch.no_grad(), autocast(device_type=autocast_device_type, enabled=use_amp):
                             kb_logits_val, mouse_act_logits_val, mouse_pos_pred_val = model(images_seq_val)
                             loss_val = torch.tensor(0.0, device=device)
                             batch_loss_unweighted_val = torch.tensor(0.0, device=device)
                             loss_calculated_val = False
                             try:
                                 if model_name == 'Keyboard AI' and keyboard_criterion is not None and kb_logits_val is not None:
                                     if not isinstance(batch_data_val[1], torch.Tensor): continue
                                     targets_kb_val = batch_data_val[1].to(device, non_blocking=PIN_MEMORY)
                                     if kb_logits_val.shape == targets_kb_val.shape:
                                         loss_kb_val_raw = keyboard_criterion(kb_logits_val, targets_kb_val)
                                         if loss_kb_val_raw.ndim == 2: batch_loss_unweighted_val = loss_kb_val_raw.mean(dim=1); loss_calculated_val = True
                                 elif model_name == 'Mouse AI':
                                     if not isinstance(batch_data_val[1], (list, tuple)) or len(batch_data_val[1]) != 2: continue
                                     targets_mouse_act_val_raw, targets_mouse_pos_val_raw = batch_data_val[1]
                                     if not isinstance(targets_mouse_act_val_raw, torch.Tensor) or not isinstance(targets_mouse_pos_val_raw, torch.Tensor): continue
                                     targets_mouse_act_val = targets_mouse_act_val_raw.to(device, non_blocking=PIN_MEMORY)
                                     targets_mouse_pos_val = targets_mouse_pos_val_raw.to(device, non_blocking=PIN_MEMORY)
                                     loss_action_val, loss_pos_val = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                                     action_loss_ok_val, pos_loss_ok_val = False, False
                                     if mouse_action_criterion is not None and mouse_act_logits_val is not None and mouse_act_logits_val.shape == targets_mouse_act_val.shape:
                                         loss_action_val_raw = mouse_action_criterion(mouse_act_logits_val, targets_mouse_act_val)
                                         if loss_action_val_raw.ndim == 2: loss_action_val = loss_action_val_raw.mean(dim=1); action_loss_ok_val = True
                                     if mouse_pos_criterion is not None and mouse_pos_pred_val is not None and mouse_pos_pred_val.shape == targets_mouse_pos_val.shape:
                                         loss_pos_val_raw = mouse_pos_criterion(mouse_pos_pred_val, targets_mouse_pos_val)
                                         if loss_pos_val_raw.ndim == 2: loss_pos_val = loss_pos_val_raw.mean(dim=1); pos_loss_ok_val = True
                                     valid_mouse_losses_val = []
                                     if action_loss_ok_val: valid_mouse_losses_val.append(loss_action_val)
                                     if pos_loss_ok_val: valid_mouse_losses_val.append(loss_pos_val * 0.5)
                                     if valid_mouse_losses_val: batch_loss_unweighted_val = torch.stack(valid_mouse_losses_val).sum(dim=0); loss_calculated_val = True
                                 if loss_calculated_val: loss_val = batch_loss_unweighted_val.mean()
                                 else: loss_val = torch.tensor(float('nan'), device=device)
                             except Exception as e_loss_calc_val: print(f"ERROR calculating val loss batch {val_batch_idx}: {e_loss_calc_val}"); loss_val = torch.tensor(float('nan'), device=device)
                         if not torch.isfinite(loss_val).all():
                             print(f"Warning: NaN/Inf validation loss (val batch {val_batch_idx}). Loss: {loss_val.item()}. Skipping."); continue
                         val_loss_accum += loss_val.item() * current_val_batch_size
                         val_samples_processed += current_val_batch_size
                         del images_seq_val, loss_val, batch_data_val, kb_logits_val, mouse_act_logits_val, mouse_pos_pred_val
                         if 'targets_kb_val' in locals(): del targets_kb_val
                         if 'targets_mouse_act_val' in locals(): del targets_mouse_act_val, targets_mouse_pos_val, targets_mouse_act_val_raw, targets_mouse_pos_val_raw
                     except torch.cuda.OutOfMemoryError:
                         print(f"\nERROR: CUDA OOM during validation (epoch {epoch+1}, batch {val_batch_idx})!"); oom_during_val = True; cleanup_memory(); break
                     except Exception as e_batch_val:
                         print(f"ERROR during validation batch {val_batch_idx}: {type(e_batch_val).__name__} - {e_batch_val}"); traceback.print_exc(limit=1); other_error_batches_val += 1; cleanup_memory(); continue
                if oom_during_val: print(f"Epoch {epoch+1}: Validation interrupted by OOM.")
                if other_error_batches_val > 0: print(f"Epoch {epoch+1}: Validation had {other_error_batches_val} batch errors.")
                if val_samples_processed > 0:
                    avg_val_loss = val_loss_accum / val_samples_processed
                    if scheduler:
                        lr_before_step = optimizer.param_groups[0]['lr']
                        val_loss_for_scheduler = avg_val_loss if avg_val_loss != float('inf') else 1e10
                        scheduler.step(val_loss_for_scheduler)
                        lr_after_step = optimizer.param_groups[0]['lr']
                        if lr_after_step < lr_before_step: print(f"  Learning rate reduced by scheduler to {lr_after_step:.3e}")
                        current_lr = lr_after_step
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                        try:
                            original_device = next(model.parameters()).device
                            model.cpu()
                            best_model_state_cpu = {k: v.detach().clone() for k, v in model.state_dict().items()}
                            model.to(original_device)
                            saved_best_state_flag = True
                            print(f"  ** New best validation loss: {best_val_loss:.4f}. Model state captured. **")
                        except Exception as e_capture:
                            print(f"  Warning: Could not capture best model state: {e_capture}"); saved_best_state_flag = False; best_model_state_cpu = None
                    else:
                        epochs_no_improve += 1
                        best_loss_str = f"{best_val_loss:.4f}" if best_val_loss != float('inf') else 'inf'
                        print(f"  Validation loss ({avg_val_loss:.4f}) did not improve vs best ({best_loss_str}). Patience: {epochs_no_improve}/{PATIENCE_EARLY_STOPPING}")
                        if epochs_no_improve >= PATIENCE_EARLY_STOPPING:
                            print(f"\nEarly stopping triggered after {actual_epochs_run} epochs.")
                            training_stopped_reason = f"Early stopping (patience {PATIENCE_EARLY_STOPPING})"
                            return model, actual_epochs_run, current_batch_size, best_model_state_cpu, saved_best_state_flag, training_stopped_reason
                else:
                    print(f"  Warning: No validation samples processed successfully in epoch {epoch+1}.")
                    epochs_no_improve += 1
                    print(f"  Patience incremented due to failed validation: {epochs_no_improve}/{PATIENCE_EARLY_STOPPING}")
                    if epochs_no_improve >= PATIENCE_EARLY_STOPPING:
                         stop_msg = "validation failed repeatedly" if best_val_loss != float('inf') else "validation never succeeded"
                         print(f"\nStopping after {actual_epochs_run} epochs ({stop_msg}).")
                         training_stopped_reason = f"Stopping ({stop_msg} after {PATIENCE_EARLY_STOPPING} attempts)"
                         return model, actual_epochs_run, current_batch_size, best_model_state_cpu, saved_best_state_flag, training_stopped_reason
            epoch_duration = time.time() - epoch_start_time
            train_loss_str = f'{avg_train_loss:.4f}' if samples_processed_train > 0 and avg_train_loss != float('inf') else 'N/A'
            val_loss_str = f'{avg_val_loss:.4f}' if run_validation and val_samples_processed > 0 and avg_val_loss != float('inf') else ('Skipped' if not run_validation else 'N/A')
            oom_str = 'Yes' if oom_occurred_epoch_train or oom_during_val else 'No'
            skipped_batches_total = nan_skipped_batches + other_error_batches_train
            print(f"Epoch {epoch+1} Summary | Train Loss: {train_loss_str} | Val Loss: {val_loss_str} | LR: {current_lr:.3e} | Duration: {epoch_duration:.2f}s | Skipped Batches (Train): {skipped_batches_total} | OOM: {oom_str}")
            if time_limit_sec is not None and (time.time() - training_start_time > time_limit_sec):
                print(f"\nINFO: Time limit reached at the end of epoch {epoch + 1}. Stopping.")
                training_stopped_reason = "Time limit reached"
                return model, actual_epochs_run, current_batch_size, best_model_state_cpu, saved_best_state_flag, training_stopped_reason
        print(f"\nFinished training after reaching max {EPOCHS_MAX} epochs.")
        return model, actual_epochs_run, current_batch_size, best_model_state_cpu, saved_best_state_flag, training_stopped_reason
    except torch.cuda.OutOfMemoryError:
        print(f"\n--- CRITICAL ERROR: CUDA OOM during {model_name} training setup or loop! ---"); traceback.print_exc(limit=1)
        training_stopped_reason = "OOM Error during training/setup"
        return None, actual_epochs_run, current_batch_size, best_model_state_cpu, saved_best_state_flag, training_stopped_reason
    except Exception as e_train:
        print(f"\n--- CRITICAL ERROR during {model_name} training (Epoch {actual_epochs_run}) ---"); print(f"--- Error: {type(e_train).__name__}: {e_train}"); traceback.print_exc()
        training_stopped_reason = f"Crashed: {type(e_train).__name__}"
        return None, actual_epochs_run, current_batch_size, best_model_state_cpu, saved_best_state_flag, training_stopped_reason
    finally:
        cleanup_memory()
def save_model_checkpoint(model_state_dict_cpu, filename, old_filename, model_info):
    model_id_name = model_info.get('name', 'Unknown AI')
    epochs_run = model_info.get('epochs', 0)
    stop_reason = model_info.get('reason', 'Unknown')
    num_train_samples = model_info.get('train_samples', 0)
    num_val_samples = model_info.get('val_samples', 0)
    saved_best_flag = model_info.get('saved_best_flag', False)
    print(f"\nAttempting to save model: {model_id_name}")
    print(f"  Stop Reason: {stop_reason}, Epochs run: {epochs_run}")
    print(f"  Train/Val Sequences: {num_train_samples}/{num_val_samples}, Best state captured? {saved_best_flag}")
    allow_save = True
    if model_state_dict_cpu is None:
        print("  Decision: Cannot save - Model state dictionary is None."); allow_save = False
    elif ("Skipped" in stop_reason or "Error" in stop_reason or "Crash" in stop_reason or "OOM" in stop_reason or "Not run" in stop_reason) and not saved_best_flag and epochs_run <= 1:
         print("  Decision: Cannot save - Training failed/skipped early and no best validation state was captured."); allow_save = False
    elif "Insufficient sequences" in stop_reason or "No training data" in stop_reason or "No aligned samples" in stop_reason:
         print("  Decision: Cannot save - Training skipped due to insufficient/missing data."); allow_save = False
    if not allow_save:
        print(f"Skipping save operation for {model_id_name}. Old file (if exists) will be kept.")
        return False
    print(f"  Proceeding with save to {filename.name}...")
    if filename.exists():
        try:
            old_filename.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Backing up existing '{filename.name}' to '{old_filename.name}'...")
            shutil.move(str(filename), str(old_filename))
            print(f"  Backup successful.")
        except Exception as e_backup:
            print(f"  Warning: Could not backup existing model file '{filename.name}': {e_backup}.")
            print(f"  Attempting to overwrite '{filename.name}' without backup.")
            try: filename.unlink(missing_ok=True)
            except Exception as e_remove: print(f"  Warning: Failed to remove '{filename.name}' before overwrite: {e_remove}.")
    try:
        if not isinstance(model_state_dict_cpu, dict): raise TypeError("State dict is not a dictionary.")
        for k, v in model_state_dict_cpu.items():
            if not isinstance(v, torch.Tensor): raise TypeError(f"Item '{k}' is not a tensor (Type: {type(v)}).")
            if v.device.type != 'cpu': raise TypeError(f"CRITICAL: State dict tensor '{k}' is not on CPU ({v.device.type}). Aborting.")
        save_dict = {
            'state_dict': model_state_dict_cpu,
            'model_name': model_id_name, 'epochs_run': epochs_run,
            'batch_size': model_info.get('batch_size'), 'stop_reason': stop_reason,
            'saved_best_flag': saved_best_flag, 'timestamp': time.time(),
            'pytorch_version': torch.__version__,
            'timm_version': timm.__version__ if TIMM_AVAILABLE else None,
            'model_arch_name': model_info.get('model_arch'),
            'sequence_length': model_info.get('sequence_length'),
            'img_size': model_info.get('img_size'),
            'screen_resolution': model_info.get('screen_resolution'),
            'train_samples': num_train_samples, 'val_samples': num_val_samples,
            'transformer_d_model': model_info.get('transformer_d_model'),
            'transformer_nhead': model_info.get('transformer_nhead'),
            'transformer_num_layers': model_info.get('transformer_num_layers'),
            'transformer_dim_feedforward': model_info.get('transformer_dim_feedforward'),
            'transformer_dropout': model_info.get('transformer_dropout'),
        }
        if model_id_name == 'Keyboard AI':
            save_dict['key_to_idx'] = model_info.get('key_to_idx')
            save_dict['idx_to_key'] = model_info.get('idx_to_key')
            save_dict['num_classes'] = model_info.get('num_classes')
        elif model_id_name == 'Mouse AI':
            save_dict['mouse_action_to_idx'] = model_info.get('mouse_action_to_idx')
            save_dict['num_action_classes'] = model_info.get('num_action_classes')
        filename.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, filename, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved model '{model_id_name}' to: {filename.name}")
        return True
    except Exception as e_save:
        print(f"ERROR saving model '{filename.name}': {e_save}"); traceback.print_exc()
        if filename.exists():
            try: filename.unlink(); print(f"  Removed partially saved file '{filename.name}'.")
            except Exception as e_rem: print(f"  Warning: Failed to remove partially saved file: {e_rem}")
        return False
if __name__ == "__main__":
    if platform.system() != "Windows" and NUM_DATALOADER_WORKERS > 0:
         preferred_method = 'fork'
         if platform.system() == 'Darwin' and torch.cuda.is_available(): preferred_method = 'spawn'
         current_method = torch.multiprocessing.get_start_method(allow_none=True)
         if current_method != preferred_method:
             try:
                 torch.multiprocessing.set_start_method(preferred_method, force=True)
                 print(f"Set multiprocessing start method to '{preferred_method}'.")
             except RuntimeError as e_mp:
                 print(f"Warning: Could not set multiprocessing start method '{preferred_method}': {e_mp}.")
    print("\n" + "="*55); print("    Starting Transformer AI Model Training Script"); print("="*55)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch: {torch.__version__}, Platform: {platform.system()}, CPU Count: {os.cpu_count()}")
    if TIMM_AVAILABLE: print(f"Timm: {timm.__version__}, Vision Model: {VISION_MODEL_NAME}")
    else: print("Timm: Not Available (ERROR)")
    print(f"Architecture: ViT + Transformer (L:{TRANSFORMER_NUM_LAYERS}, H:{TRANSFORMER_NHEAD}, D:{TRANSFORMER_D_MODEL}, FFN:{TRANSFORMER_DIM_FEEDFORWARD}, Drp:{TRANSFORMER_DROPOUT})")
    print(f"Sequence Length: {SEQUENCE_LENGTH}, Batch Size: {BATCH_SIZE_INITIAL}, LR: {LEARNING_RATE:.1e}, W Decay: {WEIGHT_DECAY}")
    time_limit_str_main = str(timedelta(seconds=TRAINING_TIME_LIMIT_PER_MODEL_SEC)) if TRAINING_TIME_LIMIT_PER_MODEL_SEC else "None"
    print(f"Max Epochs: {EPOCHS_MAX}, Patience: {PATIENCE_EARLY_STOPPING}, Time Limit/Model: {time_limit_str_main}, Workers: {NUM_DATALOADER_WORKERS}")
    print(f"Min Valid Sequences Req: {MIN_VALID_SEQUENCES_FOR_TRAINING}")
    print("-"*55)
    main_start_time = time.time()
    trained_keyboard_model, trained_mouse_model = None, None
    keyboard_best_state_cpu, mouse_best_state_cpu = None, None
    kb_saved_best_flag, mouse_saved_best_flag = False, False
    keyboard_train_loader, keyboard_val_loader = None, None
    mouse_train_loader, mouse_val_loader = None, None
    key_to_idx, idx_to_key, mouse_action_to_idx = {}, {}, {}
    num_keyboard_classes, num_mouse_action_classes = 0, 0
    kb_train_sequences, kb_val_sequences = 0, 0
    mouse_train_sequences, mouse_val_sequences = 0, 0
    keyboard_epochs_run, mouse_epochs_run = 0, 0
    keyboard_batch_size_final, mouse_batch_size_final = BATCH_SIZE_INITIAL, BATCH_SIZE_INITIAL
    keyboard_stop_reason, mouse_stop_reason = "Not run", "Not run"
    df_aligned = None
    screenshot_list_full = None
    num_total_aligned_samples = 0
    kb_saved_final, mouse_saved_final = False, False
    preprocessing_success = False
    device = None
    current_screen_resolution = None
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            try:
                 gpu_mem_alloc_gb = torch.cuda.memory_allocated(device) / (1024**3)
                 gpu_mem_reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
                 print(f"Using GPU: {gpu_name} ({gpu_mem_total_gb:.2f} GB total, {gpu_mem_alloc_gb:.2f} GB allocated, {gpu_mem_reserved_gb:.2f} GB reserved)")
            except Exception as e_mem: print(f"Using GPU: {gpu_name} ({gpu_mem_total_gb:.2f} GB total). (Mem stats error: {e_mem})")
            cleanup_memory()
        else:
            device = torch.device("cpu"); print("Using CPU for training.")
        print("-" * 30)
        df_aligned, screenshot_list_full, key_to_idx, idx_to_key, mouse_action_to_idx, preprocessing_success = preprocess_data()
        current_screen_resolution = SCREEN_RESOLUTION

        if not preprocessing_success:
            print("Preprocessing failed. Cannot proceed with training.")
            if keyboard_stop_reason == "Not run": keyboard_stop_reason = "Skipped - Preprocessing failed"
            if mouse_stop_reason == "Not run": mouse_stop_reason = "Skipped - Preprocessing failed"
        elif df_aligned.empty:
            print("Preprocessing succeeded but returned no aligned data suitable for training.")
            num_total_aligned_samples = 0
            if keyboard_stop_reason == "Not run": keyboard_stop_reason = "Skipped - No aligned samples"
            if mouse_stop_reason == "Not run": mouse_stop_reason = "Skipped - No aligned samples"
        else:
            num_total_aligned_samples = len(df_aligned)
            num_keyboard_classes = len(key_to_idx)
            num_mouse_action_classes = len(mouse_action_to_idx)
            print("-" * 30); print("Preprocessing Complete:")
            print(f"  Total Aligned Samples (with actions): {num_total_aligned_samples}")
            print(f"  Keyboard Classes Found: {num_keyboard_classes} ({list(key_to_idx.keys())[:10]}...)")
            print(f"  Mouse Action Classes Found: {num_mouse_action_classes} ({list(mouse_action_to_idx.keys())})")
            print(f"  Screen Resolution Used: {current_screen_resolution}")
            print("-" * 30)
        if preprocessing_success and num_total_aligned_samples > 0:
            print("Creating Datasets and DataLoaders...")
            norm_mean, norm_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            image_transform = transforms.Compose([
                transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
            full_kb_dataset = None; kb_train_subset, kb_val_subset = None, None
            if num_keyboard_classes > 0:
                print("Initializing Keyboard Dataset...")
                try:
                    full_kb_dataset = ActionDatasetSequence(
                        aligned_data_df=df_aligned, screenshot_list_full=screenshot_list_full,
                        sequence_length=SEQUENCE_LENGTH, img_transform=image_transform,
                        target_type='keyboard', key_vocab_size=num_keyboard_classes
                    )
                    kb_total_sequences = len(full_kb_dataset)
                    print(f"Keyboard Dataset sequences found: {kb_total_sequences}")
                    if kb_total_sequences < MIN_VALID_SEQUENCES_FOR_TRAINING:
                         print(f"Warning: Insufficient KB sequences ({kb_total_sequences} < {MIN_VALID_SEQUENCES_FOR_TRAINING}). Skipping KB training."); keyboard_stop_reason = f"Skipped - Insufficient sequences ({kb_total_sequences})"
                         full_kb_dataset = None
                except Exception as e_ds_kb: print(f"ERROR creating KB Dataset: {e_ds_kb}"); keyboard_stop_reason = f"Dataset Error: {type(e_ds_kb).__name__}"; full_kb_dataset = None
            elif keyboard_stop_reason == "Not run": print("Skipping KB Dataset: No KB classes."); keyboard_stop_reason = "Skipped - No KB classes"
            full_mouse_dataset = None; mouse_train_subset, mouse_val_subset = None, None
            if num_mouse_action_classes > 1:
                print("Initializing Mouse Dataset...")
                try:
                    full_mouse_dataset = ActionDatasetSequence(
                        aligned_data_df=df_aligned, screenshot_list_full=screenshot_list_full,
                        sequence_length=SEQUENCE_LENGTH, img_transform=image_transform,
                        target_type='mouse', mouse_action_vocab_size=num_mouse_action_classes, mouse_action_to_idx=mouse_action_to_idx
                    )
                    mouse_total_sequences = len(full_mouse_dataset)
                    print(f"Mouse Dataset sequences found: {mouse_total_sequences}")
                    if mouse_total_sequences < MIN_VALID_SEQUENCES_FOR_TRAINING:
                        print(f"Warning: Insufficient Mouse sequences ({mouse_total_sequences} < {MIN_VALID_SEQUENCES_FOR_TRAINING}). Skipping Mouse training."); mouse_stop_reason = f"Skipped - Insufficient sequences ({mouse_total_sequences})"
                        full_mouse_dataset = None
                except Exception as e_ds_mouse: print(f"ERROR creating Mouse Dataset: {e_ds_mouse}"); mouse_stop_reason = f"Dataset Error: {type(e_ds_mouse).__name__}"; full_mouse_dataset = None
            elif mouse_stop_reason == "Not run": print("Skipping Mouse Dataset: Only 'no_action'."); mouse_stop_reason = "Skipped - Only 'no_action'"
            del df_aligned; cleanup_memory()
            loader_args_base = {
                'batch_size': BATCH_SIZE_INITIAL, 'num_workers': NUM_DATALOADER_WORKERS,
                'pin_memory': PIN_MEMORY, 'drop_last': True
            }
            if PERSISTENT_WORKERS: loader_args_base['persistent_workers'] = True
            if full_kb_dataset:
                kb_indices = list(range(kb_total_sequences)); random.shuffle(kb_indices)
                kb_split_idx = int(np.floor((1.0 - VAL_SPLIT) * kb_total_sequences))
                kb_train_indices, kb_val_indices = kb_indices[:kb_split_idx], kb_indices[kb_split_idx:]
                kb_train_sequences, kb_val_sequences = len(kb_train_indices), len(kb_val_indices)
                print(f"KB Split: {kb_train_sequences} train, {kb_val_sequences} val.")
                if kb_train_sequences >= BATCH_SIZE_INITIAL:
                    try:
                        kb_train_subset = Subset(full_kb_dataset, kb_train_indices)
                        keyboard_train_loader = DataLoader(kb_train_subset, shuffle=True, **loader_args_base)
                        print(f"KB Train Loader: {len(keyboard_train_loader)} batches")
                        if kb_val_sequences > 0:
                            val_bs_kb = max(1, min(BATCH_SIZE_INITIAL * 2, kb_val_sequences))
                            val_args_kb = {**loader_args_base, 'batch_size': val_bs_kb, 'shuffle': False, 'drop_last': False}
                            kb_val_subset = Subset(full_kb_dataset, kb_val_indices)
                            keyboard_val_loader = DataLoader(kb_val_subset, **val_args_kb)
                            print(f"KB Val Loader: {len(keyboard_val_loader)} batches (size {val_bs_kb})")
                    except Exception as e_dl_kb: print(f"ERROR creating KB DataLoaders: {e_dl_kb}"); keyboard_train_loader, keyboard_val_loader = None, None; keyboard_stop_reason = f"DataLoader Error: {type(e_dl_kb).__name__}"
                elif keyboard_stop_reason == "Not run": print(f"Skipping KB training: < {BATCH_SIZE_INITIAL} train sequences."); keyboard_stop_reason = f"Skipped - <{BATCH_SIZE_INITIAL} train seq"
            if full_mouse_dataset:
                mouse_indices = list(range(mouse_total_sequences)); random.shuffle(mouse_indices)
                mouse_split_idx = int(np.floor((1.0 - VAL_SPLIT) * mouse_total_sequences))
                mouse_train_indices, mouse_val_indices = mouse_indices[:mouse_split_idx], mouse_indices[mouse_split_idx:]
                mouse_train_sequences, mouse_val_sequences = len(mouse_train_indices), len(mouse_val_indices)
                print(f"Mouse Split: {mouse_train_sequences} train, {mouse_val_sequences} val.")
                if mouse_train_sequences >= BATCH_SIZE_INITIAL:
                    try:
                        mouse_train_subset = Subset(full_mouse_dataset, mouse_train_indices)
                        mouse_train_loader = DataLoader(mouse_train_subset, shuffle=True, **loader_args_base)
                        print(f"Mouse Train Loader: {len(mouse_train_loader)} batches")
                        if mouse_val_sequences > 0:
                            val_bs_mouse = max(1, min(BATCH_SIZE_INITIAL * 2, mouse_val_sequences))
                            val_args_mouse = {**loader_args_base, 'batch_size': val_bs_mouse, 'shuffle': False, 'drop_last': False}
                            mouse_val_subset = Subset(full_mouse_dataset, mouse_val_indices)
                            mouse_val_loader = DataLoader(mouse_val_subset, **val_args_mouse)
                            print(f"Mouse Val Loader: {len(mouse_val_loader)} batches (size {val_bs_mouse})")
                    except Exception as e_dl_mouse: print(f"ERROR creating Mouse DataLoaders: {e_dl_mouse}"); mouse_train_loader, mouse_val_loader = None, None; mouse_stop_reason = f"DataLoader Error: {type(e_dl_mouse).__name__}"
                elif mouse_stop_reason == "Not run": print(f"Skipping Mouse training: < {BATCH_SIZE_INITIAL} train sequences."); mouse_stop_reason = f"Skipped - <{BATCH_SIZE_INITIAL} train seq"

            del full_kb_dataset, full_mouse_dataset
            cleanup_memory(); print("-" * 30)
            keyboard_model_def = None
            if keyboard_train_loader is not None:
                print("Initializing Keyboard AI Model (ViT + Transformer)...")
                keyboard_criterion, keyboard_optimizer, keyboard_scheduler = None, None, None
                try:
                    keyboard_model_def = VisionSequenceModel(
                        vision_model_name=VISION_MODEL_NAME, pretrained_model_path=PRETRAINED_MODEL_DIR,
                        img_size=IMG_SIZE, sequence_length=SEQUENCE_LENGTH,
                        transformer_d_model=TRANSFORMER_D_MODEL, transformer_nhead=TRANSFORMER_NHEAD,
                        transformer_num_layers=TRANSFORMER_NUM_LAYERS, transformer_dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
                        transformer_dropout=TRANSFORMER_DROPOUT,
                        num_keyboard_classes=num_keyboard_classes, num_mouse_actions=0, freeze_vision=False
                    )
                    keyboard_criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)
                    keyboard_optimizer = optim.AdamW(keyboard_model_def.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                    keyboard_scheduler = ReduceLROnPlateau(keyboard_optimizer, mode='min', factor=0.2, patience=max(1, PATIENCE_EARLY_STOPPING // 2))

                    trained_keyboard_model, keyboard_epochs_run, keyboard_batch_size_final, keyboard_best_state_cpu, kb_saved_best_flag, keyboard_stop_reason = train_model(
                        model=keyboard_model_def, model_name='Keyboard AI', train_loader=keyboard_train_loader, val_loader=keyboard_val_loader,
                        optimizer=keyboard_optimizer, scheduler=keyboard_scheduler, keyboard_criterion=keyboard_criterion,
                        device=device, batch_size_start=BATCH_SIZE_INITIAL, time_limit_sec=TRAINING_TIME_LIMIT_PER_MODEL_SEC
                    )
                    print(f"Keyboard AI Training finished. Reason: {keyboard_stop_reason}")

                except Exception as e_kb_train_setup: print(f"ERROR setting up/running KB AI training: {e_kb_train_setup}"); traceback.print_exc(); keyboard_stop_reason = f"Setup/Train Crash: {type(e_kb_train_setup).__name__}"; trained_keyboard_model, keyboard_best_state_cpu, kb_saved_best_flag = None, None, False
                finally:
                    if 'keyboard_model_def' in locals() and keyboard_model_def: del keyboard_model_def
                    if keyboard_criterion: del keyboard_criterion
                    if keyboard_optimizer: del keyboard_optimizer
                    if keyboard_scheduler: del keyboard_scheduler
                    if keyboard_train_loader: del keyboard_train_loader
                    if keyboard_val_loader: del keyboard_val_loader
                    if kb_train_subset: del kb_train_subset
                    if kb_val_subset: del kb_val_subset
            elif keyboard_stop_reason == "Not run": print("Skipping KB AI training: No train loader."); keyboard_stop_reason = "Skipped - No train loader"
            cleanup_memory(); print("-" * 30)
            mouse_model_def = None
            if mouse_train_loader is not None:
                print("Initializing Mouse AI Model (ViT + Transformer)...")
                mouse_action_criterion, mouse_pos_criterion, mouse_optimizer, mouse_scheduler = None, None, None, None
                try:
                    mouse_model_def = VisionSequenceModel(
                        vision_model_name=VISION_MODEL_NAME, pretrained_model_path=PRETRAINED_MODEL_DIR,
                        img_size=IMG_SIZE, sequence_length=SEQUENCE_LENGTH,
                        transformer_d_model=TRANSFORMER_D_MODEL, transformer_nhead=TRANSFORMER_NHEAD,
                        transformer_num_layers=TRANSFORMER_NUM_LAYERS, transformer_dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
                        transformer_dropout=TRANSFORMER_DROPOUT,
                        num_keyboard_classes=0, num_mouse_actions=num_mouse_action_classes, freeze_vision=False
                    )
                    mouse_action_criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)
                    mouse_pos_criterion = nn.MSELoss(reduction='none').to(device)
                    mouse_optimizer = optim.AdamW(mouse_model_def.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                    mouse_scheduler = ReduceLROnPlateau(mouse_optimizer, mode='min', factor=0.2, patience=max(1, PATIENCE_EARLY_STOPPING // 2))

                    trained_mouse_model, mouse_epochs_run, mouse_batch_size_final, mouse_best_state_cpu, mouse_saved_best_flag, mouse_stop_reason = train_model(
                        model=mouse_model_def, model_name='Mouse AI', train_loader=mouse_train_loader, val_loader=mouse_val_loader,
                        optimizer=mouse_optimizer, scheduler=mouse_scheduler, mouse_action_criterion=mouse_action_criterion, mouse_pos_criterion=mouse_pos_criterion,
                        device=device, batch_size_start=BATCH_SIZE_INITIAL, time_limit_sec=TRAINING_TIME_LIMIT_PER_MODEL_SEC
                    )
                    print(f"Mouse AI Training finished. Reason: {mouse_stop_reason}")

                except Exception as e_mouse_train_setup: print(f"ERROR setting up/running Mouse AI training: {e_mouse_train_setup}"); traceback.print_exc(); mouse_stop_reason = f"Setup/Train Crash: {type(e_mouse_train_setup).__name__}"; trained_mouse_model, mouse_best_state_cpu, mouse_saved_best_flag = None, None, False
                finally:
                    if 'mouse_model_def' in locals() and mouse_model_def: del mouse_model_def
                    if mouse_action_criterion: del mouse_action_criterion
                    if mouse_pos_criterion: del mouse_pos_criterion
                    if mouse_optimizer: del mouse_optimizer
                    if mouse_scheduler: del mouse_scheduler
                    if mouse_train_loader: del mouse_train_loader
                    if mouse_val_loader: del mouse_val_loader
                    if mouse_train_subset: del mouse_train_subset
                    if mouse_val_subset: del mouse_val_subset
            elif mouse_stop_reason == "Not run": print("Skipping Mouse AI training: No train loader."); mouse_stop_reason = "Skipped - No train loader"
            cleanup_memory(); print("-" * 30)
            print("--- Saving Trained Models ---")
            state_to_save_kb = keyboard_best_state_cpu
            if state_to_save_kb is None and trained_keyboard_model is not None and \
               ("Crash" not in keyboard_stop_reason and "Skipped" not in keyboard_stop_reason and "Error" not in keyboard_stop_reason and "Not run" not in keyboard_stop_reason and "OOM" not in keyboard_stop_reason):
                 print("Keyboard AI: No 'best' state, attempting to save final state (may not be optimal).")
                 try:
                     original_kb_device = next(trained_keyboard_model.parameters()).device
                     state_to_save_kb = {k: v.detach().clone().cpu() for k, v in trained_keyboard_model.cpu().state_dict().items()}
                     trained_keyboard_model.to(original_kb_device)
                 except Exception as e: print(f"Warning: Could not get final KB state: {e}"); state_to_save_kb = None

            if state_to_save_kb:
                 kb_info = {
                     'name': 'Keyboard AI', 'epochs': keyboard_epochs_run, 'batch_size': keyboard_batch_size_final,
                     'reason': keyboard_stop_reason, 'saved_best_flag': kb_saved_best_flag, 'model_arch': VISION_MODEL_NAME,
                     'sequence_length': SEQUENCE_LENGTH, 'img_size': IMG_SIZE, 'screen_resolution': current_screen_resolution,
                     'train_samples': kb_train_sequences, 'val_samples': kb_val_sequences, 'transformer_d_model': TRANSFORMER_D_MODEL,
                     'transformer_nhead': TRANSFORMER_NHEAD, 'transformer_num_layers': TRANSFORMER_NUM_LAYERS,
                     'transformer_dim_feedforward': TRANSFORMER_DIM_FEEDFORWARD, 'transformer_dropout': TRANSFORMER_DROPOUT,
                     'key_to_idx': key_to_idx, 'idx_to_key': idx_to_key, 'num_classes': num_keyboard_classes
                 }
                 kb_saved_final = save_model_checkpoint(state_to_save_kb, KEYBOARD_MODEL_FILE, OLD_KEYBOARD_MODEL_FILE, kb_info)
            else: print("No valid keyboard model state obtained to save.")
            state_to_save_mouse = mouse_best_state_cpu
            if state_to_save_mouse is None and trained_mouse_model is not None and \
                ("Crash" not in mouse_stop_reason and "Skipped" not in mouse_stop_reason and "Error" not in mouse_stop_reason and "Not run" not in mouse_stop_reason and "OOM" not in mouse_stop_reason):
                 print("Mouse AI: No 'best' state, attempting to save final state (may not be optimal).")
                 try:
                     original_mouse_device = next(trained_mouse_model.parameters()).device
                     state_to_save_mouse = {k: v.detach().clone().cpu() for k, v in trained_mouse_model.cpu().state_dict().items()}
                     trained_mouse_model.to(original_mouse_device)
                 except Exception as e: print(f"Warning: Could not get final Mouse state: {e}"); state_to_save_mouse = None

            if state_to_save_mouse:
                 mouse_info = {
                     'name': 'Mouse AI', 'epochs': mouse_epochs_run, 'batch_size': mouse_batch_size_final,
                     'reason': mouse_stop_reason, 'saved_best_flag': mouse_saved_best_flag, 'model_arch': VISION_MODEL_NAME,
                     'sequence_length': SEQUENCE_LENGTH, 'img_size': IMG_SIZE, 'screen_resolution': current_screen_resolution,
                     'train_samples': mouse_train_sequences, 'val_samples': mouse_val_sequences, 'transformer_d_model': TRANSFORMER_D_MODEL,
                     'transformer_nhead': TRANSFORMER_NHEAD, 'transformer_num_layers': TRANSFORMER_NUM_LAYERS,
                     'transformer_dim_feedforward': TRANSFORMER_DIM_FEEDFORWARD, 'transformer_dropout': TRANSFORMER_DROPOUT,
                     'mouse_action_to_idx': mouse_action_to_idx, 'num_action_classes': num_mouse_action_classes
                 }
                 mouse_saved_final = save_model_checkpoint(state_to_save_mouse, MOUSE_MODEL_FILE, OLD_MOUSE_MODEL_FILE, mouse_info)
            else: print("No valid mouse model state obtained to save.")
        else:
             print("Training and saving steps skipped due to preprocessing failure or no aligned data.")
             if keyboard_stop_reason == "Not run": keyboard_stop_reason = "Skipped - Preprocessing failed or no data"
             if mouse_stop_reason == "Not run": mouse_stop_reason = "Skipped - Preprocessing failed or no data"
    except ValueError as ve: print(f"\n--- CONFIG/DATA ERROR ---\n{ve}"); traceback.print_exc(limit=1); error_msg = f"Config/Data Error: {str(ve)[:100]}"; kb_saved_final, mouse_saved_final = False, False; keyboard_stop_reason = error_msg if keyboard_stop_reason == "Not run" else keyboard_stop_reason; mouse_stop_reason = error_msg if mouse_stop_reason == "Not run" else mouse_stop_reason
    except FileNotFoundError as fnf: print(f"\n--- FILE NOT FOUND ERROR ---\n{fnf}"); traceback.print_exc(limit=1); error_msg = f"File Not Found: {str(fnf)[:100]}"; kb_saved_final, mouse_saved_final = False, False; keyboard_stop_reason = error_msg if keyboard_stop_reason == "Not run" else keyboard_stop_reason; mouse_stop_reason = error_msg if mouse_stop_reason == "Not run" else mouse_stop_reason
    except torch.cuda.OutOfMemoryError as oom: print(f"\n--- CUDA OOM ERROR (MAIN) ---\n{oom}\nReduce batch size or model complexity."); traceback.print_exc(limit=1); error_msg = "OOM Error (Main)"; kb_saved_final, mouse_saved_final = False, False; keyboard_stop_reason = error_msg if ("Not run" in keyboard_stop_reason or "Skipped" in keyboard_stop_reason) else keyboard_stop_reason; mouse_stop_reason = error_msg if ("Not run" in mouse_stop_reason or "Skipped" in mouse_stop_reason) else mouse_stop_reason
    except Exception as e: print(f"\n--- UNEXPECTED ERROR (MAIN) ---\n{type(e).__name__}: {e}"); traceback.print_exc(); error_msg = f"Unexpected Crash: {type(e).__name__}"; kb_saved_final, mouse_saved_final = False, False; keyboard_stop_reason = error_msg if ("Not run" in keyboard_stop_reason or "Skipped" in keyboard_stop_reason) else keyboard_stop_reason; mouse_stop_reason = error_msg if ("Not run" in mouse_stop_reason or "Skipped" in mouse_stop_reason) else mouse_stop_reason
    finally:
        print("\n" + "="*55); print("--- Training Run Summary ---"); print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Preprocessing Status: {'Success' if preprocessing_success else 'Failed or Skipped'}")
        if preprocessing_success: print(f"Total Aligned Samples Found: {num_total_aligned_samples}")
        else: print(f"Total Aligned Samples Found: N/A")
        if device: print(f"Device Used: {device.type}")
        else: print("Device Used: N/A (Setup Failed)")
        print(f"Vision Backbone: {VISION_MODEL_NAME}")
        print(f"SeqLen: {SEQUENCE_LENGTH}, Screen Res Used: {current_screen_resolution if current_screen_resolution else 'N/A'}")
        print(f"Architecture: ViT + Transformer (L:{TRANSFORMER_NUM_LAYERS}, H:{TRANSFORMER_NHEAD}, D:{TRANSFORMER_D_MODEL})")
        print("-" * 30)
        def get_bs_str(epochs, reason, final_bs):
            if epochs > 0 or ("Skipped" not in reason and "Not run" not in reason and "Error" not in reason and "Crash" not in reason): return str(final_bs)
            return 'N/A'
        print("Keyboard AI:")
        print(f"  Stop Reason    : {keyboard_stop_reason}")
        print(f"  Train/Val Seqs : {kb_train_sequences:<6} / {kb_val_sequences:<6} | Classes: {num_keyboard_classes}")
        kb_bs_str = get_bs_str(keyboard_epochs_run, keyboard_stop_reason, keyboard_batch_size_final)
        print(f"  Epochs Run     : {keyboard_epochs_run:<3} | Final Batch Size: {kb_bs_str:<3}")
        print(f"  Model Saved    : {'Yes' if kb_saved_final else 'No'} {'(Best State Used)' if kb_saved_best_flag and kb_saved_final else ''}")
        if kb_saved_final: print(f"  Saved To       : {KEYBOARD_MODEL_FILE.name}")
        print("-" * 30)
        print("Mouse AI:")
        print(f"  Stop Reason    : {mouse_stop_reason}")
        print(f"  Train/Val Seqs : {mouse_train_sequences:<6} / {mouse_val_sequences:<6} | Classes: {num_mouse_action_classes}")
        mouse_bs_str = get_bs_str(mouse_epochs_run, mouse_stop_reason, mouse_batch_size_final)
        print(f"  Epochs Run     : {mouse_epochs_run:<3} | Final Batch Size: {mouse_bs_str:<3}")
        print(f"  Model Saved    : {'Yes' if mouse_saved_final else 'No'} {'(Best State Used)' if mouse_saved_best_flag and mouse_saved_final else ''}")
        if mouse_saved_final: print(f"  Saved To       : {MOUSE_MODEL_FILE.name}")
        print("-" * 30)
        print("Final cleanup...")
        if 'trained_keyboard_model' in locals() and trained_keyboard_model is not None: del trained_keyboard_model
        if 'trained_mouse_model' in locals() and trained_mouse_model is not None: del trained_mouse_model
        if 'keyboard_best_state_cpu' in locals() and keyboard_best_state_cpu is not None: del keyboard_best_state_cpu
        if 'mouse_best_state_cpu' in locals() and mouse_best_state_cpu is not None: del mouse_best_state_cpu
        if 'screenshot_list_full' in locals() and screenshot_list_full is not None: del screenshot_list_full
        del key_to_idx, idx_to_key, mouse_action_to_idx
        cleanup_memory()
        total_script_time = time.time() - main_start_time
        print(f"\nTotal script execution time: {total_script_time:.2f} seconds ({timedelta(seconds=int(total_script_time))}).")
        print("="*55); print("    Transformer Training Script Finished"); print("="*55)