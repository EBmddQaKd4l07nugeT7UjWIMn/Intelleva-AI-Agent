import time
import threading
import os
from pathlib import Path
import json
import queue
import math
import platform
import traceback
from collections import deque
import random
import numpy as np
import sys
import contextlib
import warnings
import pickle
import gc
import fnmatch

warnings.filterwarnings("ignore", category=UserWarning, message=".*The given NumPy array is not writable.*")
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.*")
warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*encoder_layer.norm_first was True.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`torch.cuda.amp.GradScaler.*is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`torch.cuda.amp.autocast.*is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*The verbose parameter is deprecated.*")
warnings.filterwarnings("ignore", message=".*Detected call of `lr_scheduler.step()` before `optimizer.step()`.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Detected pickle protocol.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Weights only load failed.*Trying weights_only=False.*")

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image, ImageFile, UnidentifiedImageError
    ImageFile.LOAD_TRUNCATED_IMAGES = True # Allow loading truncated images
    Image.MAX_IMAGE_PIXELS = None # Allow loading large images if needed
    # mss is NOT needed for screen capture anymore, but keep pynput and tkinter
    from pynput import keyboard, mouse
    import tkinter as tk
    try:
        import timm
        TIMM_AVAILABLE = True
        timm_version = tuple(map(int, timm.__version__.split('.')[:2]))
        print(f"Found 'timm' library (version: {timm.__version__})")
    except ImportError:
        print("ERROR: 'timm' library not found. This script requires ViT models from timm.")
        print("Install with: pip install timm")
        sys.exit(1)
except ImportError as e:
    print(f"ERROR: Missing required library: {e.name}")
    print("Please install necessary libraries: pip install torch torchvision timm Pillow pynput")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during initial imports: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Model Definitions (Exactly as provided) ---
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
            # Dynamic PE adjustment if sequence length changes (though not expected here)
            if x.size(1) > self.pe.shape[1]:
                # Handle longer sequences if absolutely necessary, though usually indicates mismatch
                print(f"WARN: Input seq len ({x.size(1)}) > PE max_len ({self.pe.shape[1]}). Recalculating PE.")
                new_max_len = x.size(1)
                self.__init__(self.pe.shape[2], new_max_len, self.dropout.p) # Reinitialize
                self.pe = self.pe.to(x.device) # Ensure it's on the correct device
                # Re-register buffer? Might not be needed if __init__ handles it.

            pe_slice = self.pe[:, :x.size(1), :]
            x = x + pe_slice
            return self.dropout(x)
        except RuntimeError as e_pe:
            print(f"ERROR applying Positional Encoding: {e_pe}")
            print(f"  Input shape: {x.shape}, PE shape: {self.pe.shape}, Attempted PE slice shape: (1, {x.size(1)}, {self.pe.shape[2]})")
            raise

class VisionSequenceModel(nn.Module):
    def __init__(self, vision_model_name, img_size, sequence_length,
                 transformer_d_model, transformer_nhead, transformer_num_layers,
                 transformer_dim_feedforward, transformer_dropout=0.1,
                 num_keyboard_classes=0, num_mouse_actions=0, **kwargs): # Use kwargs to ignore unexpected params
        super().__init__()
        self.sequence_length = sequence_length
        self.transformer_d_model = transformer_d_model
        self.vision_model_name_used = vision_model_name
        self.num_keyboard_classes = num_keyboard_classes
        self.num_mouse_actions = num_mouse_actions
        self.vision_feature_dim = 0
        self.img_size = img_size # Store img_size used during training

        try:
            self.vision_backbone = timm.create_model(
                vision_model_name, pretrained=False, num_classes=0, global_pool='avg'
            )
            self.vision_feature_dim = self.vision_backbone.num_features
            if self.vision_feature_dim <= 0: raise ValueError(f"ViT feature dimension <= 0: {self.vision_feature_dim}")
            print(f"  Vision backbone '{vision_model_name}' created. Feature dim: {self.vision_feature_dim}")
        except Exception as e_timm:
             print(f"ERROR creating timm model '{vision_model_name}': {e_timm}")
             raise RuntimeError(f"Failed to create vision backbone '{vision_model_name}'") from e_timm

        if self.vision_feature_dim != self.transformer_d_model:
            self.input_proj = nn.Linear(self.vision_feature_dim, self.transformer_d_model)
            print(f"  Added input projection: {self.vision_feature_dim} -> {self.transformer_d_model}")
        else:
            self.input_proj = nn.Identity()
            print(f"  Using identity input projection (dims match: {self.transformer_d_model})")

        # Correct max_len logic derived from training script assumption
        correct_max_len = max(50, sequence_length * 2)
        self.pos_encoder = PositionalEncoding(self.transformer_d_model, correct_max_len, transformer_dropout)
        print(f"  Positional Encoding initialized (d_model={self.transformer_d_model}, max_len={correct_max_len})")

        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.transformer_d_model, nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout,
                activation="gelu", batch_first=True, norm_first=True # norm_first=True is crucial
            )
            encoder_norm = nn.LayerNorm(self.transformer_d_model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*norm_first was True")
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=transformer_num_layers, norm=encoder_norm,
                    enable_nested_tensor=False # Explicitly disable for safety/compatibility
                )
            print(f"  Transformer Encoder initialized ({transformer_num_layers} layers, nhead={transformer_nhead}, norm_first=True)")
        except Exception as e_transformer:
             print(f"ERROR initializing Transformer Encoder: {e_transformer}")
             raise RuntimeError("Failed to initialize Transformer Encoder") from e_transformer

        self.keyboard_head = None
        if num_keyboard_classes > 0:
             self.keyboard_head = nn.Linear(self.transformer_d_model, num_keyboard_classes)
             print(f"  Keyboard Head initialized (output classes: {num_keyboard_classes})")

        self.mouse_action_head = None
        self.mouse_pos_head = None
        if num_mouse_actions > 0:
             self.mouse_action_head = nn.Linear(self.transformer_d_model, num_mouse_actions)
             pos_head_intermediate = max(64, self.transformer_d_model // 4)
             # CRITICAL ASSUMPTION: Model outputs ONE coordinate pair (x, y) based on definition.
             # This will be treated as the target/end coordinate.
             self.mouse_pos_head = nn.Sequential(
                 nn.Linear(self.transformer_d_model, pos_head_intermediate),
                 nn.GELU(),
                 nn.Dropout(0.1),
                 nn.Linear(pos_head_intermediate, 2), # Output (x, y)
                 nn.Sigmoid() # Normalize to [0, 1] range
             )
             print(f"  Mouse Action Head initialized (output classes: {num_mouse_actions})")
             print(f"  Mouse Position Head initialized (intermediate: {pos_head_intermediate}, output: 2)")

    def forward(self, x_seq):
        if x_seq.ndim != 5: raise ValueError(f"Input shape error: Expected 5 dims (B, S, C, H, W), got {x_seq.ndim} ({x_seq.shape})")
        batch_size, seq_len, C, H, W = x_seq.shape
        x_flat = x_seq.view(batch_size * seq_len, C, H, W)
        vision_features_flat = None
        try:
            vision_features_flat = self.vision_backbone(x_flat)
        except Exception as e_vit_fwd: print(f"ERROR during Vision Backbone forward pass: {e_vit_fwd}"); raise
        try:
            vision_features_seq = vision_features_flat.view(batch_size, seq_len, self.vision_feature_dim)
        except RuntimeError as e_reshape: print(f"ERROR reshaping ViT features: {e_reshape}"); raise

        transformer_input = self.input_proj(vision_features_seq)
        final_features = None
        try:
            transformer_input_pe = self.pos_encoder(transformer_input)
            memory = self.transformer_encoder(transformer_input_pe)
            if memory is None or memory.ndim != 3 or memory.shape[0] != batch_size or memory.shape[1] != seq_len:
                raise ValueError(f"Transformer output 'memory' unexpected shape/None: {memory.shape if memory is not None else 'None'}")
            final_features = memory[:, -1, :] # Use features from the last token
        except Exception as e_transformer_fwd: print(f"ERROR during Transformer forward pass: {e_transformer_fwd}"); raise

        keyboard_logits, mouse_action_logits, mouse_pos_pred = None, None, None
        try:
            if final_features is None: raise ValueError("final_features is None before output heads.")
            if self.keyboard_head is not None: keyboard_logits = self.keyboard_head(final_features)
            if self.mouse_action_head is not None: mouse_action_logits = self.mouse_action_head(final_features)
            if self.mouse_pos_head is not None: mouse_pos_pred = self.mouse_pos_head(final_features)
        except Exception as e_head_fwd: print(f"ERROR during Output Head forward pass: {e_head_fwd}"); raise

        return keyboard_logits, mouse_action_logits, mouse_pos_pred
# --- End of Model Definition ---


# --- Constants and Paths ---
SCRIPT_DIR = Path(__file__).parent.resolve()
EXPERIENCE_POOL_DIR = SCRIPT_DIR / "experience_pool"
SCREENSHOT_DIR = EXPERIENCE_POOL_DIR / "screenshots"
KEYBOARD_LOG_FILE = EXPERIENCE_POOL_DIR / "keyboard_log.jsonl"
MOUSE_LOG_FILE = EXPERIENCE_POOL_DIR / "mouse_log.jsonl"
RESULTS_LOG_FILE = EXPERIENCE_POOL_DIR / "results_log.jsonl" # Kept for consistency, though not used for input
KEYBOARD_MODEL_FILE = SCRIPT_DIR / "keyboard_ai_model_transformer.pth"
MOUSE_MODEL_FILE = SCRIPT_DIR / "mouse_ai_model_transformer.pth"

# --- AI Agent Configuration (Set by loaded model metadata) ---
IMG_SIZE = None # Tuple (height, width) - SET BY MODEL LOAD
SEQUENCE_LENGTH = None # Integer - SET BY MODEL LOAD
screen_resolution = None # Tuple (width, height) - SET BY MODEL LOAD

# Agent behavior config
FOLDER_SCAN_INTERVAL_SEC = 0.1 # How often to check for new screenshots
MIN_INFERENCE_INTERVAL_SEC = 0.1 # Minimum time between inferences (prevents spamming if files arrive fast)
KEYBOARD_CONFIDENCE_THRESHOLD = 0.60 # Min confidence for KB action
MOUSE_ACTION_CONFIDENCE_THRESHOLD = 0.65 # Min confidence for non-'no_action' mouse action
MAX_PARALLEL_KEYS_OUTPUT = 3 # Max simultaneous keys (Requirement: 1-3)
AI_KEY_PRESS_DURATION_MS = (70, 150) # Rand range for AI key hold duration
AI_MOUSE_CLICK_DURATION_MS = (50, 120) # Rand range for AI click/long_press hold duration
# AI_MOUSE_DRAG_HOLD_MS = (30, 80) # Optional separate hold duration during drag (if needed)

# User Input Detection Thresholds
QP_EXIT_THRESHOLD_NS = 1_000_000_000 # Less than 1 second for Q+P combo exit

# --- Global Variables & Control Flags ---
agent_stop_event = threading.Event() # Signals all threads to terminate
log_lock = threading.Lock() # Ensures thread-safe writing to log files

# AI Model / Metadata Placeholders
keyboard_model = None
mouse_model = None
keyboard_metadata = {}
mouse_metadata = {}
device = None
idx_to_key = {} # Loaded from keyboard model metadata { "0": "w", ... }
idx_to_mouse_action = {} # Loaded from mouse model metadata { "0": "no_action", "1": "click_left", ... }
inference_queue = None # Queue for passing inference results (dict) to action workers

# Requirement: Keyboard AI should not output Ctrl or Alt.
# Use a comprehensive list of forbidden keys.
FORBIDDEN_AI_KEYS = {
    'ctrl', 'ctrl_l', 'ctrl_r', 'alt', 'alt_l', 'alt_r', 'alt_gr',
    'shift', 'shift_l', 'shift_r', 'caps_lock',
    'cmd', 'cmd_l', 'cmd_r', 'win', 'meta', 'windows', 'gui', 'gui_l', 'gui_r', # Command/Windows keys
    'apps', 'menu', 'compose', 'scroll_lock', 'num_lock', 'insert', #'delete', # Allowing delete
    'home', 'end', 'page_up', 'page_down', 'print_screen', 'pause', 'sleep', 'wakeup', 'esc', 'escape', # Function/Navigation
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', # Function keys
    'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24',
    'media_volume_mute', 'media_volume_down', 'media_volume_up', 'media_play_pause', # Media keys
    'media_previous', 'media_next', 'media_stop',
    'kana', 'kanji', 'convert', 'nonconvert', 'yen', 'hiragana', 'katakana', # IME keys
    'unknown', 'unidentified', None, '', ' ', 'none', #'spacebar', # Allowing spacebar
    #'tab', #'backspace', 'enter', 'return', # Allowing common typing keys
    'fn', # Function modifier key
    # Add any other system-level or potentially disruptive keys here
}
print(f"Forbidden AI Keys Count: {len(FORBIDDEN_AI_KEYS)}")


# --- Utility Functions ---
def write_log(filepath: Path, data: dict):
    """Appends a JSON line to the specified log file thread-safely, with validation."""
    try:
        with log_lock:
            log_type = "Unknown"
            required_fields = []
            if filepath == KEYBOARD_LOG_FILE:
                required_fields = ['timestamp_ns', 'press_time_ns', 'release_time_ns', 'key_name']
                log_type = "Keyboard"
            elif filepath == MOUSE_LOG_FILE:
                required_fields = ['timestamp_ns', 'press_time_ns', 'release_time_ns', 'action', 'button', 'start_pos', 'end_pos']
                log_type = "Mouse"
            elif filepath == RESULTS_LOG_FILE:
                required_fields = ['timestamp_ns', 'result']
                log_type = "Result"
            else:
                print(f"WARN: Unknown log file path {filepath}")
                return False

            # Validate essential fields
            missing_or_invalid = []
            for k in required_fields:
                val = data.get(k)
                is_valid = False
                if k.endswith('_ns'): # Check timestamps are positive integers
                    is_valid = isinstance(val, (int, np.int64)) and val > 0
                elif k in ['start_pos', 'end_pos']: # Check positions are lists of two ints
                    is_valid = isinstance(val, list) and len(val) == 2 and all(isinstance(v, int) for v in val)
                elif k == 'trajectory': # Check trajectory is list of lists/tuples (optional check, only for drag)
                     if data.get("action") == "drag":
                         is_valid = isinstance(val, list) and all(isinstance(p, list) and len(p) == 3 and isinstance(p[0],int) and isinstance(p[1],int) and isinstance(p[2],int) for p in val)
                     else:
                         is_valid = True # Trajectory not expected for non-drag
                elif k == 'action':
                    is_valid = isinstance(val, str) and val in ['click', 'long_press', 'drag']
                elif k == 'button':
                     is_valid = isinstance(val, str) and val in ['left', 'right']
                else: # Check other fields are non-empty strings (basic check)
                    is_valid = isinstance(val, str) and val

                if not is_valid:
                    missing_or_invalid.append(f"{k}: {val} (Type: {type(val).__name__})")

            # Ensure trajectory is only present for drag actions
            if data.get("action") != "drag" and "trajectory" in data:
                missing_or_invalid.append("trajectory present for non-drag action")
            elif data.get("action") == "drag" and "trajectory" not in data:
                 # Allow minimal trajectory to be absent, but log warning? For now, let it pass if other fields ok.
                 pass # print(f"WARN: Trajectory missing for drag action log to {filepath.name}")

            if missing_or_invalid:
                 print(f"WARN: Skip log to {filepath.name}, invalid/missing fields:")
                 for item in missing_or_invalid: print(f"  - {item}")
                 # print(f"  Full invalid data: {data}") # Optional: print full data for debug
                 return False

            if not filepath.parent.exists(): filepath.parent.mkdir(parents=True, exist_ok=True)
            if not filepath.exists(): filepath.touch(exist_ok=True)

            # Write the JSON line
            with filepath.open('a', encoding='utf-8') as f:
                # Ensure all values are JSON serializable (e.g., np.int64 -> int)
                serializable_data = json.loads(json.dumps(data, default=int))
                json.dump(serializable_data, f, separators=(',', ':')) # Compact JSON format
                f.write('\n')
            return True
    except Exception as e:
        print(f"ERROR writing log to {filepath.name} ({log_type}): {type(e).__name__} - {e}")
        traceback.print_exc(limit=1)
        return False

def cleanup_memory():
    """Performs garbage collection and empties CUDA cache if available."""
    try:
        collected = gc.collect()
        # print(f"GC collected {collected} objects.") # Optional debug
        if torch.cuda.is_available():
             torch.cuda.empty_cache()
             # print("CUDA cache emptied.") # Optional debug
    except Exception as e_cleanup: print(f"Warning: Memory cleanup failed: {e_cleanup}")

def ensure_experience_pool():
    """Checks and creates the necessary directories and empty log files."""
    print("Checking experience pool structure...")
    try:
        EXPERIENCE_POOL_DIR.mkdir(parents=True, exist_ok=True)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  Directory: {EXPERIENCE_POOL_DIR} - OK")
        print(f"  Directory: {SCREENSHOT_DIR} - OK")
        for fpath in [KEYBOARD_LOG_FILE, MOUSE_LOG_FILE, RESULTS_LOG_FILE]:
            if not fpath.exists():
                fpath.touch(exist_ok=True); print(f"  Created empty log file: {fpath.name}")
            elif fpath.is_dir():
                 raise OSError(f"Log path {fpath} is a directory, not a file!")
        print("Experience pool structure checked/created successfully.")
    except OSError as e: print(f"ERROR: Could not create/verify experience pool at {EXPERIENCE_POOL_DIR}: {e}"); raise

def _get_key_repr(key):
    """Gets a consistent, lowercase string representation for a pynput key event."""
    try:
        if hasattr(key, 'name') and key.name: return key.name.lower()
        elif hasattr(key, 'char') and key.char: return key.char.lower()
        elif hasattr(key, 'vk') and key.vk is not None: return f"vk_{key.vk}"
        else: return repr(key).lower().strip("'")
    except Exception: return repr(key).lower().strip("'")

def load_ai_model(model_path: Path, model_type: str, expected_device: torch.device):
    """Loads the AI model checkpoint and its metadata, performing validation."""
    global SEQUENCE_LENGTH, IMG_SIZE, screen_resolution, idx_to_key, idx_to_mouse_action

    print(f"\n--- Loading {model_type} AI Model ---")
    if not model_path.is_file(): print(f"ERROR: Model file not found at {model_path}"); return None, None
    model_instance, metadata = None, {}
    try:
        print(f"Loading checkpoint from: {model_path} (CPU map location)")
        checkpoint = None
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            print("  Checkpoint loaded successfully (weights_only=True).")
        except Exception as e_safe:
            print(f"  Warning: Failed loading with weights_only=True ({type(e_safe).__name__}). Trying weights_only=False (ensure trust)...")
            try:
                with warnings.catch_warnings():
                     warnings.filterwarnings("ignore", category=UserWarning, message=".*Trying weights_only=False.*")
                     checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                print("  Checkpoint loaded successfully (weights_only=False).")
            except (pickle.UnpicklingError, FileNotFoundError, EOFError, AttributeError, TypeError) as e_unsafe:
                 print(f"\nERROR: Critical failure loading {model_path.name} with weights_only=False. File corrupt/missing/unsafe?"); print(f"Details: {type(e_unsafe).__name__}: {e_unsafe}\n"); return None, None
            except Exception as e_generic_unsafe:
                 print(f"ERROR: Failed torch.load {model_path} (weights_only=False). Error: {type(e_generic_unsafe).__name__} - {e_generic_unsafe}"); traceback.print_exc(limit=1); return None, None

        if checkpoint is None or not isinstance(checkpoint, dict): print(f"ERROR: Loaded checkpoint invalid (None or not dict)."); return None, None

        def get_meta(key, default=None, is_tuple=False):
            val = checkpoint.get(key, default)
            if is_tuple and isinstance(val, list): return tuple(val)
            return val

        metadata = {
            'model_arch': get_meta('model_arch_name', get_meta('model_arch')),
            'sequence_length': get_meta('sequence_length'),
            'img_size': get_meta('img_size', is_tuple=True),
            'screen_resolution': get_meta('screen_resolution', is_tuple=True),
            'transformer_d_model': get_meta('transformer_d_model'),
            'transformer_nhead': get_meta('transformer_nhead'),
            'transformer_num_layers': get_meta('transformer_num_layers'),
            'transformer_dim_feedforward': get_meta('transformer_dim_feedforward'),
            'transformer_dropout': get_meta('transformer_dropout', 0.1),
        }
        if model_type == "Keyboard":
            metadata['key_to_idx'] = get_meta('key_to_idx', {})
            metadata['idx_to_key'] = get_meta('idx_to_key', {})
            metadata['num_classes'] = get_meta('num_classes', 0)
            if not metadata.get('idx_to_key') and isinstance(metadata.get('key_to_idx'), dict):
                 metadata['idx_to_key'] = {str(v): str(k) for k, v in metadata['key_to_idx'].items()}
            elif isinstance(metadata.get('idx_to_key'), dict):
                 metadata['idx_to_key'] = {str(k): str(v) for k,v in metadata['idx_to_key'].items()} # Ensure string keys/values
            else: metadata['idx_to_key'] = {} # Ensure it's a dict
        elif model_type == "Mouse":
            metadata['mouse_action_to_idx'] = get_meta('mouse_action_to_idx', {})
            metadata['idx_to_mouse_action'] = get_meta('idx_to_mouse_action', {})
            metadata['num_action_classes'] = get_meta('num_action_classes', 0)
            if not metadata.get('idx_to_mouse_action') and isinstance(metadata.get('mouse_action_to_idx'), dict):
                 metadata['idx_to_mouse_action'] = {str(v): str(k) for k, v in metadata['mouse_action_to_idx'].items()}
            elif isinstance(metadata.get('idx_to_mouse_action'), dict):
                 metadata['idx_to_mouse_action'] = {str(k): str(v) for k,v in metadata['idx_to_mouse_action'].items()} # Ensure string keys/values
            else: metadata['idx_to_mouse_action'] = {} # Ensure it's a dict

        print("Extracted Metadata:", {k: v if not isinstance(v, dict) or len(v)<10 else f"dict({len(v)} items)" for k, v in metadata.items()})

        required_meta = ['model_arch', 'sequence_length', 'img_size', 'screen_resolution', 'transformer_d_model', 'transformer_nhead', 'transformer_num_layers', 'transformer_dim_feedforward']
        if model_type == "Keyboard": required_meta.extend(['idx_to_key', 'num_classes'])
        if model_type == "Mouse": required_meta.extend(['idx_to_mouse_action', 'num_action_classes'])
        missing_meta = [k for k in required_meta if metadata.get(k) is None]
        if missing_meta: print(f"ERROR: Missing critical metadata: {missing_meta}"); return None, None

        errors = []
        if not isinstance(metadata['sequence_length'], int) or metadata['sequence_length'] <= 0: errors.append(f"Invalid sequence_length ({metadata['sequence_length']})")
        if not isinstance(metadata['img_size'], tuple) or len(metadata['img_size']) != 2 or not all(isinstance(d, int) and d > 0 for d in metadata['img_size']): errors.append(f"Invalid img_size ({metadata['img_size']})")
        if not isinstance(metadata['screen_resolution'], tuple) or len(metadata['screen_resolution']) != 2 or not all(isinstance(d, int) and d > 0 for d in metadata['screen_resolution']): errors.append(f"Invalid screen_resolution ({metadata['screen_resolution']})")
        if model_type == "Keyboard" and (not isinstance(metadata['idx_to_key'], dict) or not metadata['idx_to_key'] or metadata['num_classes'] <= 0): errors.append("Invalid KB vocab/class info")
        if model_type == "Mouse" and (not isinstance(metadata['idx_to_mouse_action'], dict) or not metadata['idx_to_mouse_action'] or metadata['num_action_classes'] <= 0): errors.append("Invalid Mouse vocab/class info")
        for k in ['transformer_d_model', 'transformer_nhead', 'transformer_num_layers', 'transformer_dim_feedforward']:
            if not isinstance(metadata[k], int) or metadata[k] <= 0: errors.append(f"Invalid transformer param: {k}={metadata[k]}")
        if errors: print(f"ERROR: Invalid metadata values: {errors}"); return None, None

        current_meta_seq_len, current_meta_img_size, current_meta_res = metadata['sequence_length'], metadata['img_size'], metadata['screen_resolution']
        if SEQUENCE_LENGTH is None: SEQUENCE_LENGTH = current_meta_seq_len; print(f"Set global SEQUENCE_LENGTH = {SEQUENCE_LENGTH}")
        elif current_meta_seq_len != SEQUENCE_LENGTH: print(f"ERROR: {model_type} seq len ({current_meta_seq_len}) mismatch with previous model's ({SEQUENCE_LENGTH})."); return None, None
        if IMG_SIZE is None: IMG_SIZE = current_meta_img_size; print(f"Set global IMG_SIZE = {IMG_SIZE}")
        elif current_meta_img_size != IMG_SIZE: print(f"ERROR: {model_type} img size ({current_meta_img_size}) mismatch with previous model's ({IMG_SIZE})."); return None, None
        if screen_resolution is None: screen_resolution = current_meta_res; print(f"Set global screen_resolution = {screen_resolution}")
        elif current_meta_res != screen_resolution: print(f"ERROR: {model_type} screen res ({current_meta_res}) mismatch with previous model's ({screen_resolution})."); return None, None
        if SEQUENCE_LENGTH is None or IMG_SIZE is None or screen_resolution is None: print("FATAL: Critical metadata (SeqLen/ImgSize/ScreenRes) missing after processing."); return None, None

        # Assign vocabularies globally AFTER consistency checks pass
        if model_type == "Keyboard" and metadata.get('idx_to_key'): idx_to_key = metadata['idx_to_key']
        if model_type == "Mouse" and metadata.get('idx_to_mouse_action'): idx_to_mouse_action = metadata['idx_to_mouse_action']

        print("Instantiating model architecture...")
        try:
            model_init_args = {
                'vision_model_name': metadata['model_arch'], 'img_size': IMG_SIZE, 'sequence_length': SEQUENCE_LENGTH,
                'transformer_d_model': metadata['transformer_d_model'], 'transformer_nhead': metadata['transformer_nhead'],
                'transformer_num_layers': metadata['transformer_num_layers'], 'transformer_dim_feedforward': metadata['transformer_dim_feedforward'],
                'transformer_dropout': metadata['transformer_dropout'],
                'num_keyboard_classes': metadata.get('num_classes', 0) if model_type == "Keyboard" else 0,
                'num_mouse_actions': metadata.get('num_action_classes', 0) if model_type == "Mouse" else 0,
            }
            model_instance = VisionSequenceModel(**model_init_args)
        except Exception as e_instantiate: print(f"ERROR instantiating model: {e_instantiate}"); traceback.print_exc(limit=2); return None, None

        state_dict = checkpoint.get('state_dict', checkpoint)
        if state_dict is None or not isinstance(state_dict, dict): print(f"ERROR: Invalid state_dict found in checkpoint."); return None, None
        needs_cleaning = any(k.startswith('module.') for k in state_dict.keys())
        if needs_cleaning:
            print("  Cleaning 'module.' prefix from state_dict keys...")
            state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

        print("Loading state_dict into model (strict=False)...")
        try:
             load_result = model_instance.load_state_dict(state_dict, strict=False)
             actual_missing = [k for k in load_result.missing_keys if 'num_batches_tracked' not in k]
             actual_unexpected = [k for k in load_result.unexpected_keys if 'optimizer' not in k and 'scheduler' not in k]
             if model_type == "Keyboard":
                 actual_missing = [k for k in actual_missing if not k.startswith('mouse_')]
                 actual_unexpected = [k for k in actual_unexpected if not k.startswith('mouse_')]
             elif model_type == "Mouse":
                 actual_missing = [k for k in actual_missing if not k.startswith('keyboard_')]
                 actual_unexpected = [k for k in actual_unexpected if not k.startswith('keyboard_')]
             if actual_missing: print(f"  WARNING: Missing Keys found in model state: {actual_missing}")
             if actual_unexpected: print(f"  WARNING: Unexpected Keys found in checkpoint file: {actual_unexpected}")

             # Positional Encoding Size Check
             loaded_pe_shape = state_dict.get('pos_encoder.pe', torch.zeros(0)).shape
             current_pe_shape = getattr(getattr(model_instance, 'pos_encoder', None), 'pe', torch.zeros(0)).shape
             # Check only the dimension size, allow flexibility in max_len if needed (use model's current)
             if len(loaded_pe_shape) < 3 or len(current_pe_shape) < 3 or loaded_pe_shape[0] != current_pe_shape[0] or loaded_pe_shape[2] != current_pe_shape[2]:
                 print(f"\nCRITICAL ERROR: Positional Encoding dimension mismatch or invalid!")
                 print(f"  Loaded PE shape: {loaded_pe_shape}, Current Model PE shape: {current_pe_shape}")
                 return None, None
             elif loaded_pe_shape[1] != current_pe_shape[1]:
                 print(f"  INFO: Loaded PE max_len ({loaded_pe_shape[1]}) differs from current ({current_pe_shape[1]}). Using current model's PE.")
             else: print(f"  Positional Encoding size check passed ({current_pe_shape}).")

        except Exception as e_load: print(f"ERROR loading state_dict: {e_load}"); traceback.print_exc(limit=2); return None, None

        model_instance.to(expected_device)
        model_instance.eval()
        print(f"{model_type} model loaded successfully to {expected_device}.")
        print("-" * 30)
        return model_instance, metadata

    except Exception as e: print(f"ERROR loading model {model_path}: {type(e).__name__} - {e}"); traceback.print_exc(); return None, None


# --- AI Agent Task Manager ---
class AIAgentTaskManager:
    """Manages background threads for AI inference and action based on monitored screenshots."""
    def __init__(self, root_tk, kb_model, mouse_model, kb_meta, mouse_meta, device_):
        global inference_queue, idx_to_key, idx_to_mouse_action, screen_resolution, IMG_SIZE, SEQUENCE_LENGTH
        self.root = root_tk
        self.keyboard_model = kb_model
        self.mouse_model = mouse_model
        self.keyboard_metadata = kb_meta if kb_meta else {}
        self.mouse_metadata = mouse_meta if mouse_meta else {}
        self.device = device_
        self.stop_event = agent_stop_event
        self.threads = {}

        # Validate critical parameters loaded from models
        if not isinstance(SEQUENCE_LENGTH, int) or SEQUENCE_LENGTH <= 0: raise ValueError(f"Invalid SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
        if not isinstance(IMG_SIZE, tuple) or len(IMG_SIZE) != 2 or not all(isinstance(d, int) and d>0 for d in IMG_SIZE): raise ValueError(f"Invalid IMG_SIZE: {IMG_SIZE}")
        if not isinstance(screen_resolution, tuple) or len(screen_resolution) != 2 or not all(isinstance(d, int) and d>0 for d in screen_resolution): raise ValueError(f"Invalid screen_resolution: {screen_resolution}")
        self.screen_width, self.screen_height = screen_resolution

        self.image_sequence = deque(maxlen=SEQUENCE_LENGTH) # Stores the sequence of image tensors
        self.last_inference_time = 0 # Track timing for inference interval
        self.last_processed_timestamp_ns = 0 # Track the timestamp of the last processed screenshot file

        try:
            self.kb_controller = keyboard.Controller(); self.mouse_controller = mouse.Controller()
            print("Pynput controllers initialized.")
        except Exception as e: raise RuntimeError(f"Failed to initialize pynput controllers: {e}") from e

        self._keyboard_listener_instance = None
        self._mouse_listener_instance = None
        self.last_q_press_time_ns = 0
        self.last_p_press_time_ns = 0

        inference_queue = queue.Queue(maxsize=10); self.inference_queue = inference_queue # Slightly larger queue

        # Define image transformation pipeline (must match training)
        norm_mean, norm_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5] # Standard normalization
        try:
            self.image_transform = transforms.Compose([
                transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
            print(f"Image transform created for size {IMG_SIZE}.")
        except Exception as e_tr: raise RuntimeError(f"Failed to create image transforms: {e_tr}") from e_tr

        # Vocabularies should be populated by load_ai_model
        if self.keyboard_model and not idx_to_key: raise ValueError("KB model loaded but idx_to_key mapping is empty.")
        if self.mouse_model and not idx_to_mouse_action: raise ValueError("Mouse model loaded but idx_to_mouse_action mapping is empty.")

        print(f"Agent Task Manager initialized. Screen: {self.screen_width}x{self.screen_height}, SeqLen: {SEQUENCE_LENGTH}, ImgSize: {IMG_SIZE}")

    def start_all(self):
        """Starts all necessary background threads."""
        if self.stop_event.is_set(): print("Start requested but stop event already set. Aborting."); return
        self.stop_event.clear()

        thread_targets = {
            "ScreenshotMonitorWorker": self.screenshot_monitor_worker, # Renamed worker
            "UserKeyboardListener": self.user_keyboard_listener_worker,
            "UserMouseListener": self.user_mouse_listener_worker # Kept for structure
        }
        if self.keyboard_model:
            thread_targets["KeyboardActionWorker"] = self.keyboard_action_worker
        if self.mouse_model:
            thread_targets["MouseActionWorker"] = self.mouse_action_worker

        self.threads = {}
        print("\n--- Starting Background Threads ---")
        for name, target in thread_targets.items():
            thread = threading.Thread(target=target, name=name, daemon=True)
            self.threads[name] = thread
            try:
                thread.start()
                print(f"  Thread '{name}' started.")
            except RuntimeError as e_start:
                print(f"FATAL ERROR starting thread '{name}': {e_start}. Initiating immediate shutdown.")
                self.stop_event.set()
                for t_name, t_obj in self.threads.items():
                    if t_obj is not thread and t_obj.is_alive():
                        try: t_obj.join(timeout=0.5)
                        except Exception: pass
                raise

    def stop_all(self):
        """Signals all threads to stop and waits for them to join."""
        print("\n--- Initiating AI Agent Task Shutdown ---")
        if self.stop_event.is_set(): print("Stop event already set."); return
        self.stop_event.set()

        print("Requesting listener stops...")
        if self._keyboard_listener_instance:
            try: self._keyboard_listener_instance.stop(); print("  KB listener stop requested.")
            except Exception as e: print(f"  Warn: Error stopping KB listener: {e}")
        if self._mouse_listener_instance:
            try: self._mouse_listener_instance.stop(); print("  Mouse listener stop requested.")
            except Exception as e: print(f"  Warn: Error stopping Mouse listener: {e}")
        time.sleep(0.2)

        print("Clearing inference queue and adding sentinel...")
        cleared_count = 0
        if self.inference_queue:
            while not self.inference_queue.empty():
                try: self.inference_queue.get_nowait(); cleared_count += 1
                except queue.Empty: break
            try: self.inference_queue.put_nowait(None) # Sentinel for Keyboard worker
            except queue.Full: print("  Warn: Inference queue full when adding KB sentinel.")
            try: self.inference_queue.put_nowait(None) # Sentinel for Mouse worker
            except queue.Full: print("  Warn: Inference queue full when adding Mouse sentinel.")
        print(f"  Cleared {cleared_count} items from inference queue.")

        print("Waiting for background threads to join (max ~2s each)...")
        join_timeout = 2.0
        threads_to_join = list(self.threads.items())
        for name, thread in threads_to_join:
            if thread and thread.is_alive():
                print(f"  Joining '{name}'...")
                thread.join(timeout=join_timeout)
                if thread.is_alive(): print(f"  WARNING: Thread '{name}' did not stop cleanly.")
                else: print(f"  Thread '{name}' joined successfully.")
        lingering = [t.name for t in self.threads.values() if t and t.is_alive()]
        if lingering: print(f"ERROR: The following threads are still alive after join attempt: {lingering}")
        else: print("All background threads joined.")

        self.threads = {}
        self._keyboard_listener_instance = None
        self._mouse_listener_instance = None
        cleanup_memory()
        print("Task manager resources cleaned.")

    # --- Worker Thread Methods ---

    def screenshot_monitor_worker(self):
        """Monitors screenshot folder, loads images, manages sequence, runs inference, queues results."""
        print("Screenshot Monitor worker started.");
        pin_memory_flag = (self.device.type == 'cuda')
        processed_files_in_session = set() # Keep track of files processed in this run

        # Initialize last processed timestamp by finding the latest existing file if any
        try:
            existing_files = [p for p in SCREENSHOT_DIR.glob("*.png") if p.stem.isdigit()]
            if existing_files:
                self.last_processed_timestamp_ns = max(int(p.stem) for p in existing_files)
                print(f"  Initialized last processed timestamp from existing files: {self.last_processed_timestamp_ns}")
            else:
                self.last_processed_timestamp_ns = 0
                print("  No existing screenshots found, starting timestamp tracking from 0.")
        except Exception as e_init_scan:
            print(f"WARN: Error during initial scan of screenshots folder: {e_init_scan}. Starting timestamp from 0.")
            self.last_processed_timestamp_ns = 0

        while not self.stop_event.is_set():
            loop_start_mono = time.monotonic()
            new_files_processed = 0
            try:
                # Find new image files based on timestamp in filename
                potential_files = []
                # Use scandir for potentially better performance on large directories
                with os.scandir(SCREENSHOT_DIR) as it:
                    for entry in it:
                        # Check if it's a file, ends with .png, and filename is digits only
                        if entry.is_file() and entry.name.lower().endswith('.png') and entry.name[:-4].isdigit():
                            try:
                                file_ts_ns = int(entry.name[:-4])
                                # Process only files newer than the last processed one
                                if file_ts_ns > self.last_processed_timestamp_ns:
                                    potential_files.append((file_ts_ns, SCREENSHOT_DIR / entry.name))
                            except ValueError:
                                continue # Ignore files with non-integer names

                # Sort new files chronologically
                potential_files.sort(key=lambda x: x[0])

                if not potential_files:
                    # No new files, wait before checking again
                    if self.stop_event.wait(FOLDER_SCAN_INTERVAL_SEC): break
                    continue # Go to next loop iteration

                # --- Process New Files ---
                for file_ts_ns, file_path in potential_files:
                    if self.stop_event.is_set(): break # Check stop event between files
                    if file_path in processed_files_in_session: continue # Skip if already processed in this run

                    # print(f"  Processing new screenshot: {file_path.name}") # Debug print
                    img_tensor = None
                    try:
                        # Load image using PIL
                        with Image.open(file_path) as img_pil:
                            # Ensure image is in RGB format
                            img_pil_rgb = img_pil.convert("RGB")
                        # Apply the image transformation pipeline
                        img_tensor = self.image_transform(img_pil_rgb)
                        new_files_processed += 1
                        processed_files_in_session.add(file_path) # Mark as processed for this session
                        # Add the processed tensor to the sequence deque
                        self.image_sequence.append(img_tensor)
                        # Update the last processed timestamp *only after successful processing*
                        self.last_processed_timestamp_ns = file_ts_ns

                    except (UnidentifiedImageError, FileNotFoundError, OSError, ValueError) as e_load:
                        print(f"Warn: Failed to load/process screenshot {file_path.name}: {type(e_load).__name__} - {e_load}")
                        img_tensor = None # Ensure tensor is None on failure
                    except Exception as e_proc:
                         print(f"ERROR processing screenshot {file_path.name}: {type(e_proc).__name__} - {e_proc}")
                         traceback.print_exc(limit=1)
                         img_tensor = None

                    # --- Inference Check ---
                    sequence_ready = (len(self.image_sequence) == SEQUENCE_LENGTH)
                    time_since_last_inference = loop_start_mono - self.last_inference_time
                    min_interval_passed = time_since_last_inference >= MIN_INFERENCE_INTERVAL_SEC

                    # Run inference ONLY if sequence is full AND min interval has passed
                    if img_tensor is not None and sequence_ready and min_interval_passed:
                        self.last_inference_time = loop_start_mono # Update last inference time
                        # print(f"  Sequence full ({SEQUENCE_LENGTH}), running inference (last image: {file_path.name})") # Debug
                        try:
                            with torch.no_grad():
                                input_list = list(self.image_sequence)
                                if not all(isinstance(t, torch.Tensor) for t in input_list):
                                    print("Warn: Non-tensor found in image sequence. Clearing queue."); self.image_sequence.clear(); continue
                                try:
                                    input_tensor = torch.stack(input_list, dim=0).unsqueeze(0).to(self.device, non_blocking=pin_memory_flag)
                                except RuntimeError as e_stack: print(f"ERROR stacking tensors for inference: {e_stack}"); self.image_sequence.clear(); continue

                                results = {'timestamp_ns': file_ts_ns} # Use timestamp of the *last* image in sequence
                                kb_logits, mouse_act_logits, mouse_pos_pred = None, None, None

                                if self.keyboard_model:
                                     try:
                                         outputs = self.keyboard_model(input_tensor)
                                         kb_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                                         if kb_logits is not None: results['keyboard'] = kb_logits.cpu()
                                     except Exception as e_kb_inf: print(f"ERROR during Keyboard inference: {type(e_kb_inf).__name__}")

                                if self.mouse_model:
                                     try:
                                         outputs = self.mouse_model(input_tensor)
                                         if isinstance(outputs, tuple) and len(outputs) >= 3:
                                             mouse_act_logits, mouse_pos_pred = outputs[1], outputs[2]
                                         else: print(f"Warn: Unexpected Mouse model output format: {type(outputs)}")
                                         if mouse_act_logits is not None: results['mouse_action'] = mouse_act_logits.cpu()
                                         if mouse_pos_pred is not None: results['mouse_position'] = mouse_pos_pred.cpu() # This contains ONE coordinate pair
                                     except Exception as e_mouse_inf: print(f"ERROR during Mouse inference: {type(e_mouse_inf).__name__}")

                                if len(results) > 1:
                                   try:
                                       # Add results to queue for both workers
                                       self.inference_queue.put_nowait(results.copy()) # Give copy to Keyboard
                                       self.inference_queue.put_nowait(results.copy()) # Give copy to Mouse
                                   except queue.Full:
                                       print("Warn: Inference queue full, skipped putting new result.")
                                       # Optional: Clear queue or drop oldest if needed
                                       # try: self.inference_queue.get_nowait(); self.inference_queue.get_nowait() except queue.Empty: pass

                            # Clean up tensors
                            del input_list, input_tensor, results
                            if 'kb_logits' in locals(): del kb_logits
                            if 'mouse_act_logits' in locals(): del mouse_act_logits, mouse_pos_pred
                            cleanup_memory() # Clean GPU memory after inference
                        except Exception as e_inf_outer: print(f"ERROR in inference block: {type(e_inf_outer).__name__}"); traceback.print_exc(limit=1); cleanup_memory()
                    elif img_tensor is not None and not sequence_ready:
                        # print(f"  Sequence length {len(self.image_sequence)}/{SEQUENCE_LENGTH}") # Debug
                        pass
                    elif img_tensor is not None and not min_interval_passed:
                         # print(f"  Inference interval too short ({time_since_last_inference:.3f}s < {MIN_INFERENCE_INTERVAL_SEC:.3f}s)") # Debug
                         pass

                # End of processing loop for this batch of files
                # Add a small sleep if many files were processed quickly to yield CPU
                if new_files_processed > 5: # Arbitrary number to indicate rapid processing
                    if self.stop_event.wait(0.01): break

            except FileNotFoundError:
                 print(f"ERROR: Screenshots directory not found: {SCREENSHOT_DIR}. Stopping worker.")
                 self.stop_event.set()
            except Exception as e_worker_main:
                print(f"FATAL error in screenshot monitor worker: {type(e_worker_main).__name__} - {e_worker_main}");
                traceback.print_exc();
                self.stop_event.set() # Signal shutdown on critical error

            # Minimal sleep at the end of the outer loop if no files were found
            # This is handled by the wait() inside the loop now.

        cleanup_memory();
        print("Screenshot Monitor worker finished.")


    def keyboard_action_worker(self):
        """Processes keyboard inference results and executes/logs actions with detailed timing."""
        print("Keyboard Action worker started.")
        global idx_to_key
        if not idx_to_key: print("KB Worker: ERROR - idx_to_key mapping empty. Exiting."); return
        if not self.keyboard_model: print("KB Worker: ERROR - Keyboard model not loaded. Exiting."); return

        while not self.stop_event.is_set():
            results = None
            try:
                # Wait for inference results targeting keyboard
                results = self.inference_queue.get(timeout=0.5)
                if results is None: print("KB Worker: Received shutdown signal."); break
                if 'keyboard' not in results: continue # Should have keyboard key if intended for this worker

                kb_logits = results.get('keyboard')
                if kb_logits is None: continue
                screenshot_time_ns = results.get('timestamp_ns', time.time_ns()) # Timestamp from screenshot file

                try:
                    kb_probs = torch.sigmoid(kb_logits).squeeze(0)
                    if kb_probs.ndim != 1: print(f"Warn: Invalid KB prob shape {kb_probs.shape}, expected 1D."); continue

                    eligible_indices = torch.where(kb_probs > KEYBOARD_CONFIDENCE_THRESHOLD)[0]
                    if eligible_indices.numel() == 0: continue

                    eligible_probs = kb_probs[eligible_indices]
                    sorted_conf_indices_rel = torch.argsort(eligible_probs, descending=True)

                    selected_keys_info = []
                    num_keys_to_select = min(eligible_indices.numel(), MAX_PARALLEL_KEYS_OUTPUT)

                    print(f"\n⌨️ AI KB Action Analysis (Src Ts: {screenshot_time_ns})")
                    print(f"  Eligible Keys ({eligible_indices.numel()} > {KEYBOARD_CONFIDENCE_THRESHOLD:.2f}):")

                    keys_to_press_this_cycle = []
                    for i in range(eligible_indices.numel()): # Iterate all eligible to show confidence
                        original_index = eligible_indices[i].item()
                        key_index_str = str(original_index)
                        key_name = idx_to_key.get(key_index_str)
                        confidence = kb_probs[original_index].item() # Get confidence of this specific index

                        if not key_name:
                            print(f"    - Idx {key_index_str}: ??? (Not in vocab) - Conf: {confidence:.3f}")
                            continue

                        key_name_lower = key_name.lower()
                        is_forbidden = key_name_lower in FORBIDDEN_AI_KEYS
                        is_selected = i < num_keys_to_select # Check if this key ranks high enough

                        print(f"    - Idx {key_index_str}: '{key_name}' - Conf: {confidence:.3f} {'[FORBIDDEN]' if is_forbidden else ''} {'[SELECTED]' if is_selected and not is_forbidden else ''}")

                        if is_selected and not is_forbidden:
                            keys_to_press_this_cycle.append({'name': key_name, 'confidence': confidence, 'idx': key_index_str})


                    # --- Execute Actions ---
                    if keys_to_press_this_cycle:
                        num_selected = len(keys_to_press_this_cycle)
                        print(f"  Executing {num_selected} parallel key press(es):")

                        pressed_key_details = [] # List to store details of successfully pressed keys

                        # --- Press Phase ---
                        action_start_ns = time.time_ns() # Overall start time for this batch of presses
                        for key_info in keys_to_press_this_cycle:
                            k_name = key_info['name']
                            k_conf = key_info['confidence']
                            pynput_key = None
                            try:
                                # Map key name to pynput key object
                                k_lower = k_name.lower()
                                special_key = getattr(keyboard.Key, k_lower, None)
                                if special_key: pynput_key = special_key
                                elif len(k_name) == 1: pynput_key = keyboard.KeyCode.from_char(k_name) # Use from_char for single chars
                                elif k_lower.startswith("vk_") and k_lower.split('_')[1].isdigit(): pynput_key = keyboard.KeyCode(vk=int(k_lower.split('_')[1]))
                                else: print(f"    -> Warn: Cannot map key name '{k_name}' to pynput key. Skipping press."); continue

                                if self.stop_event.is_set(): print("    -> Skip press: stop event set."); break

                                # Press the key and record the exact press time
                                press_ns = time.time_ns()
                                self.kb_controller.press(pynput_key)
                                pressed_key_details.append({
                                    'pynput': pynput_key,
                                    'name': k_name,
                                    'confidence': k_conf,
                                    'press_time_ns': press_ns,
                                    'release_time_ns': 0 # Placeholder
                                })
                                print(f"    + Press: '{k_name}' (Conf: {k_conf:.3f}) at {press_ns}")

                            except Exception as e_press: print(f"    -> ERROR pressing '{k_name}': {type(e_press).__name__}")

                        # --- Hold Phase ---
                        if pressed_key_details:
                            duration_ms = random.randint(AI_KEY_PRESS_DURATION_MS[0], AI_KEY_PRESS_DURATION_MS[1])
                            hold_end_mono = time.monotonic() + (duration_ms / 1000.0)
                            # print(f"  Holding for ~{duration_ms}ms...") # Verbose

                            interrupted_hold = False
                            while time.monotonic() < hold_end_mono:
                                if self.stop_event.is_set(): interrupted_hold = True; break
                                if agent_stop_event.wait(0.005): interrupted_hold = True; break # Check stop event during wait

                            # --- Release Phase ---
                            release_start_ns = time.time_ns() # Overall release time for this batch
                            print(f"  Releasing keys{' (Interrupted)' if interrupted_hold else ''} at {release_start_ns}:")
                            release_count = 0
                            for item in pressed_key_details:
                                try:
                                    self.kb_controller.release(item['pynput'])
                                    item['release_time_ns'] = time.time_ns() # Record individual release time
                                    release_count += 1
                                    print(f"    - Release: '{item['name']}' at {item['release_time_ns']}")

                                    # Log the action if it completed fully (press and release occurred without interruption)
                                    if not interrupted_hold and item['press_time_ns'] > 0 and item['release_time_ns'] > 0:
                                        log_data = {
                                            "timestamp_ns": screenshot_time_ns, # Use screenshot time as the event time
                                            "press_time_ns": item['press_time_ns'],
                                            "release_time_ns": item['release_time_ns'],
                                            "key_name": item['name']
                                            # Optional: Add confidence here if needed: "confidence": item['confidence']
                                        }
                                        if not write_log(KEYBOARD_LOG_FILE, log_data):
                                             print(f"      -> Failed to log release for '{item['name']}'")

                                except Exception as e_release: print(f"    -> ERROR releasing '{item['name']}': {type(e_release).__name__}")

                            print(f"  --- End AI KB Action (Released {release_count}) ---")
                        else:
                             print(f"  No keys were successfully pressed.")

                except Exception as e_proc_kb: print(f"ERROR processing KB logits/executing action: {type(e_proc_kb).__name__} - {e_proc_kb}"); traceback.print_exc(limit=1)
                finally:
                    del results, kb_logits, kb_probs # Clean up tensors
                    cleanup_memory()

            except queue.Empty: continue # Normal timeout
            except Exception as e_outer: print(f"ERROR in KB Action worker main loop: {type(e_outer).__name__} - {e_outer}"); traceback.print_exc(limit=1); self.stop_event.wait(0.5)

        print("Keyboard Action worker finished.")


    def mouse_action_worker(self):
        """Processes mouse inference results and executes/logs actions with detailed timing and coords."""
        print("Mouse Action worker started.")
        global idx_to_mouse_action, screen_resolution
        if not idx_to_mouse_action: print("Mouse Worker: ERROR - idx_to_mouse_action mapping empty. Exiting."); return
        if not screen_resolution: print("Mouse Worker: ERROR - Screen resolution not set. Exiting."); return
        if not self.mouse_model: print("Mouse Worker: ERROR - Mouse model not loaded. Exiting."); return
        screen_w, screen_h = self.screen_width, self.screen_height
        print(f"Mouse worker using screen resolution: {screen_w}x{screen_h}")

        while not self.stop_event.is_set():
            results = None
            try:
                # Wait for inference results targeting mouse
                results = self.inference_queue.get(timeout=0.5)
                if results is None: print("Mouse Worker: Received shutdown signal."); break
                if 'mouse_action' not in results or 'mouse_position' not in results: continue

                mouse_act_logits = results.get('mouse_action')
                mouse_pos_pred = results.get('mouse_position') # Shape [1, 2] with (x_norm, y_norm)
                if mouse_act_logits is None or mouse_pos_pred is None: continue
                screenshot_time_ns = results.get('timestamp_ns', time.time_ns()) # Timestamp from screenshot file

                try:
                    mouse_act_probs = torch.softmax(mouse_act_logits, dim=-1).squeeze(0)
                    mouse_pos_norm = mouse_pos_pred.squeeze(0) # Shape [2]
                    if mouse_act_probs.ndim != 1 or mouse_pos_norm.shape != torch.Size([2]): print(f"Warn: Invalid mouse output shape. Action Probs: {mouse_act_probs.shape}, Pos Norm: {mouse_pos_norm.shape}"); continue

                    best_action_idx = torch.argmax(mouse_act_probs).item()
                    best_action_prob = mouse_act_probs[best_action_idx].item()
                    action_index_str = str(best_action_idx)
                    predicted_action_name = idx_to_mouse_action.get(action_index_str)

                    if not predicted_action_name or predicted_action_name == 'no_action' or best_action_prob < MOUSE_ACTION_CONFIDENCE_THRESHOLD: continue

                    # --- Parse Action ---
                    action_parts = predicted_action_name.lower().split('_'); pynput_button = None; log_action_type = None; log_button_name = None
                    # Expected formats: "click_left", "long_press_right", "drag_left"
                    if len(action_parts) >= 2:
                        type_part, button_part = action_parts[0], action_parts[-1]
                        # Validate action type
                        if type_part in ['click', 'long_press', 'drag']: log_action_type = type_part
                        # Validate and map button part
                        if button_part == 'left': log_button_name = 'left'; pynput_button = mouse.Button.left
                        elif button_part == 'right': log_button_name = 'right'; pynput_button = mouse.Button.right

                    if log_action_type is None or pynput_button is None: print(f"Warn: Cannot parse mouse action name '{predicted_action_name}'. Skipping."); continue

                    # --- Calculate Coordinates ---
                    # Model predicts ONE point. Treat it as the target/end point.
                    x_norm, y_norm = mouse_pos_norm[0].item(), mouse_pos_norm[1].item()
                    target_x = max(0, min(screen_w - 1, int(x_norm * screen_w)))
                    target_y = max(0, min(screen_h - 1, int(y_norm * screen_h)))
                    target_pos_pixels = (target_x, target_y) # This is the END point for drag, target for click/long_press

                    # --- Determine Start Position ---
                    start_pos_pixels = None
                    if log_action_type == "drag":
                         # For drag, start position is the current mouse position *before* the action starts
                         try: start_pos_pixels = tuple(self.mouse_controller.position)
                         except Exception as e_getpos: print(f"Warn: Could not get current mouse pos for drag start: {e_getpos}"); start_pos_pixels = target_pos_pixels # Fallback
                    else: # For click/long_press, start and end are the same target position
                        start_pos_pixels = target_pos_pixels

                    # --- Execute Action ---
                    action_start_overall_ns = time.time_ns() # Timestamp when AI decides to act
                    print(f"\n🖱️ AI Mouse Action @ {action_start_overall_ns} (Src Ts: {screenshot_time_ns}, Conf: {best_action_prob:.3f})")
                    print(f"  Action: {log_action_type.upper()} {log_button_name.upper()}")
                    print(f"  Start Pos: {start_pos_pixels}")
                    print(f"  End/Target Pos: {target_pos_pixels} (Pred: '{predicted_action_name}')")


                    log_entry = {
                        "timestamp_ns": screenshot_time_ns, # Use screenshot time as the event time
                        "press_time_ns": 0,
                        "release_time_ns": 0,
                        "action": log_action_type,
                        "button": log_button_name,
                        "start_pos": list(start_pos_pixels),
                        "end_pos": list(target_pos_pixels),
                        # Trajectory added later only for drag
                    }
                    action_completed = False
                    interrupted_action = False
                    press_actual_ns = 0
                    release_actual_ns = 0

                    try:
                        if self.stop_event.is_set(): print("    -> Skip action: stop event set."); continue

                        # 1. Move mouse to the START position (relevant for click/long_press, drag starts from current)
                        if log_action_type in ["click", "long_press"]:
                            self.mouse_controller.position = start_pos_pixels
                            if self.stop_event.wait(0.02): raise InterruptedError("Stop event during move") # Brief pause/check

                        # 2. Press the mouse button (at start_pos for click/long, current pos for drag start)
                        current_pos_before_press = tuple(self.mouse_controller.position) # Record pos right before press
                        if log_action_type == "drag":
                            log_entry["start_pos"] = list(current_pos_before_press) # Update drag start pos to actual

                        press_actual_ns = time.time_ns()
                        self.mouse_controller.press(pynput_button)
                        log_entry["press_time_ns"] = press_actual_ns
                        print(f"    + Press at {current_pos_before_press} at {press_actual_ns}")

                        # 3. Handle Hold Duration / Drag Movement
                        duration_ms = random.randint(AI_MOUSE_CLICK_DURATION_MS[0], AI_MOUSE_CLICK_DURATION_MS[1])
                        hold_end_mono = time.monotonic() + (duration_ms / 1000.0)

                        if log_action_type == "drag":
                            # If dragging, move to the target position *during* the hold time
                            # Simple linear interpolation for movement (can be improved)
                            move_start_mono = time.monotonic()
                            print(f"    Dragging to {target_pos_pixels} over ~{duration_ms}ms...")
                            while time.monotonic() < hold_end_mono:
                                if self.stop_event.is_set(): interrupted_action = True; break
                                # Optional: Interpolate position here if smoother drag needed
                                # For simplicity, just move directly to target near the end or immediately
                                time_elapsed = time.monotonic() - move_start_mono
                                move_fraction = min(1.0, time_elapsed / (duration_ms / 1000.0))
                                if move_fraction >= 0.1: # Start moving after a short delay
                                    self.mouse_controller.position = target_pos_pixels # Move to final position

                                if agent_stop_event.wait(0.01): interrupted_action = True; break # Check stop frequently
                            # Ensure final position is set if loop finished early
                            if not interrupted_action: self.mouse_controller.position = target_pos_pixels

                        else: # Click or Long Press - Just hold
                            print(f"    Holding for ~{duration_ms}ms...")
                            while time.monotonic() < hold_end_mono:
                                if self.stop_event.is_set(): interrupted_action = True; break
                                if agent_stop_event.wait(0.01): interrupted_action = True; break

                        # 4. Release the mouse button
                        release_pos_pixels = tuple(self.mouse_controller.position) # Get position at release time
                        release_actual_ns = time.time_ns()
                        self.mouse_controller.release(pynput_button)
                        log_entry["release_time_ns"] = release_actual_ns
                        log_entry["end_pos"] = list(release_pos_pixels) # Update end_pos to actual release location
                        action_completed = True
                        print(f"    - Release at {release_pos_pixels} at {release_actual_ns}")


                        # 5. Log the action
                        if action_completed and not interrupted_action:
                            if log_entry["action"] == "drag":
                                # Create minimal trajectory: [start_x, start_y, press_ns], [end_x, end_y, release_ns]
                                start_pos_log = log_entry["start_pos"]
                                end_pos_log = log_entry["end_pos"]
                                press_ns_log = log_entry["press_time_ns"]
                                release_ns_log = log_entry["release_time_ns"]
                                log_entry["trajectory"] = [
                                    [start_pos_log[0], start_pos_log[1], press_ns_log],
                                    [end_pos_log[0], end_pos_log[1], release_ns_log]
                                ]

                            if write_log(MOUSE_LOG_FILE, log_entry): print(f"    Action logged successfully.")
                            else: print(f"    -> Failed to log mouse action.")
                        elif not action_completed: print("    Action failed before release.")
                        else: print(f"    Action interrupted. Log skipped.")

                    except InterruptedError:
                         print("    Action interrupted by stop event.")
                         interrupted_action = True # Ensure log is skipped
                    except Exception as e_exec: print(f"    ERROR executing mouse action: {type(e_exec).__name__} - {e_exec}"); traceback.print_exc(limit=1)
                    finally:
                        # Ensure button is released if press happened but release failed/was interrupted
                        if press_actual_ns > 0 and not release_actual_ns > 0:
                             print("    Ensuring mouse button is released due to potential error/interruption.")
                             with contextlib.suppress(Exception): self.mouse_controller.release(pynput_button)
                        print(f"  --- End AI Mouse Action ---")

                except Exception as e_proc_mouse: print(f"ERROR processing Mouse outputs/executing action: {type(e_proc_mouse).__name__} - {e_proc_mouse}"); traceback.print_exc(limit=1)
                finally:
                    del results, mouse_act_logits, mouse_pos_pred # Clean up tensors
                    cleanup_memory()

            except queue.Empty: continue # Normal timeout
            except Exception as e_outer: print(f"ERROR in Mouse Action worker main loop: {type(e_outer).__name__} - {e_outer}"); traceback.print_exc(limit=1); self.stop_event.wait(0.5)

        print("Mouse Action worker finished.")


    # --- User Input Listeners (Only for Q+P Exit Combo) ---
    def _on_press(self, key):
        """Handles key presses ONLY to detect the Q+P exit combo."""
        press_ns = time.time_ns(); key_repr = _get_key_repr(key)
        try:
            is_q = (key_repr == 'q'); is_p = (key_repr == 'p')
            exit_triggered = False

            if is_p and (0 < self.last_q_press_time_ns < press_ns) and (press_ns - self.last_q_press_time_ns < QP_EXIT_THRESHOLD_NS):
                exit_triggered = True; print(f"\n>>> Q -> P combo detected! Requesting SHUTDOWN.")
            elif is_q and (0 < self.last_p_press_time_ns < press_ns) and (press_ns - self.last_p_press_time_ns < QP_EXIT_THRESHOLD_NS):
                exit_triggered = True; print(f"\n>>> P -> Q combo detected! Requesting SHUTDOWN.")

            if is_q: self.last_q_press_time_ns = press_ns; self.last_p_press_time_ns = 0
            if is_p: self.last_p_press_time_ns = press_ns; self.last_q_press_time_ns = 0

            if exit_triggered:
                self.last_q_press_time_ns = self.last_p_press_time_ns = 0
                self._request_shutdown()

            # Cleanup stale combo times
            cleanup_threshold = QP_EXIT_THRESHOLD_NS * 2 # Allow 2x interval before reset
            now_ns = press_ns
            if self.last_q_press_time_ns > 0 and (now_ns - self.last_q_press_time_ns) >= cleanup_threshold: self.last_q_press_time_ns = 0
            if self.last_p_press_time_ns > 0 and (now_ns - self.last_p_press_time_ns) >= cleanup_threshold: self.last_p_press_time_ns = 0

        except Exception as e: print(f"ERROR in _on_press (keyboard listener): {type(e).__name__}")

    def _on_release(self, key):
        """Resets Q/P state if needed on release."""
        key_repr = _get_key_repr(key)
        try:
            # Optional: Resetting on release makes the combo timing stricter (presses must overlap)
            # If commented out, just pressing Q then P within 1s works, even if Q is released first.
            # Let's keep it simple and reset only in _on_press when the *other* key is pressed.
            pass
            # if key_repr == 'q': self.last_q_press_time_ns = 0
            # if key_repr == 'p': self.last_p_press_time_ns = 0
        except Exception as e: print(f"ERROR in _on_release: {type(e).__name__}")

    def user_keyboard_listener_worker(self):
        """Listens for user keyboard input specifically for the Q+P exit combo."""
        print("User Keyboard listener worker started (for exit combo)."); listener = None
        self.last_q_press_time_ns = 0; self.last_p_press_time_ns = 0;
        try:
            listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._keyboard_listener_instance = listener; print("Keyboard listener running...")
            listener.run() # Blocks until listener.stop() is called or error
        except Exception as e: print(f"FATAL ERROR in Keyboard listener: {type(e).__name__}. Requesting shutdown."); self._request_shutdown()
        finally: print("User Keyboard listener worker finished."); self._keyboard_listener_instance = None

    def user_mouse_listener_worker(self):
        """Runs a mouse listener that performs NO actions or logging."""
        print("User Mouse listener worker started (no actions performed)."); listener = None
        try:
            def on_move(x,y): pass
            def on_click(x, y, button, pressed): pass
            def on_scroll(x, y, dx, dy): pass
            listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
            self._mouse_listener_instance = listener; print("Mouse listener running (idle)...")
            listener.run()
        except Exception as e: print(f"FATAL ERROR in Mouse listener: {type(e).__name__}. Requesting shutdown."); self._request_shutdown()
        finally: print("User Mouse listener worker finished."); self._mouse_listener_instance = None

    def _request_shutdown(self):
        """Safely requests the application to shut down via the main event loop."""
        if not self.stop_event.is_set():
            print("Shutdown requested via combo/error!");
            self.stop_event.set()
            # Schedule the closing function to run in the main Tkinter thread
            if self.root and self.root.winfo_exists():
                 # Schedule with a short delay to allow listener context to unwind potentially
                self.root.after(50, _trigger_main_thread_shutdown)
            else:
                print("Tkinter root unavailable. Shutdown might be less clean.")
                # Attempt direct call if no root, but might cause issues if called from wrong thread
                _trigger_main_thread_shutdown()


# --- Global Shutdown Trigger ---
_shutdown_in_progress = threading.Event()
root = None # Make root global for access in shutdown trigger
task_manager = None # Make task_manager global

def _trigger_main_thread_shutdown():
    """Function called by root.after to run on_closing in the main thread."""
    if not _shutdown_in_progress.is_set():
        print("Main thread executing on_closing().")
        on_closing()
    else:
         print("Main thread shutdown already in progress.")

def on_closing():
    """Handles the application shutdown sequence. Should run in the main thread."""
    if _shutdown_in_progress.is_set(): return
    _shutdown_in_progress.set(); print("\n>>> Shutdown Sequence Initiated <<<")
    agent_stop_event.set() # Ensure stop event is set for all threads

    if task_manager: print("Stopping AI agent tasks..."); task_manager.stop_all()
    else: print("Task manager not initialized or already stopped.")

    print("Destroying Tkinter root (if exists)...")
    if root and root.winfo_exists():
        try: root.destroy(); print("  Tkinter root destroyed.")
        except Exception as e: print(f"  ERROR destroying Tkinter root: {type(e).__name__}")

    cleanup_memory(); print(">>> Shutdown sequence complete. <<<")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("="*60 + "\n Starting AI Agent Application (play_game.py) \n" + "="*60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}, OS: {platform.system()}, Python: {platform.python_version()}")
    try: print(f"PyTorch: {torch.__version__}")
    except NameError: print("ERROR: PyTorch not found."); sys.exit(1)
    if TIMM_AVAILABLE: print(f"Timm: {timm.__version__}")
    else: print("ERROR: Timm library not available."); sys.exit(1)
    print("-" * 60)

    try:
        ensure_experience_pool()

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"\nDevice: GPU ({torch.cuda.get_device_name(0)})")
            cleanup_memory()
        else:
            device = torch.device("cpu"); print("\nDevice: CPU")

        print("\n--- Loading AI Models ---")
        SEQUENCE_LENGTH, IMG_SIZE, screen_resolution = None, None, None
        idx_to_key, idx_to_mouse_action = {}, {} # Reset vocabs
        keyboard_model, keyboard_metadata = load_ai_model(KEYBOARD_MODEL_FILE, "Keyboard", device)
        mouse_model, mouse_metadata = load_ai_model(MOUSE_MODEL_FILE, "Mouse", device)

        if keyboard_model is None and mouse_model is None:
            message = "FATAL ERROR: Neither Keyboard nor Mouse AI model could be loaded. Check model files and paths."
            print(f"\n{message}"); sys.exit(1)
        if keyboard_model is None: print("\nWARNING: Keyboard AI disabled (model load failed).")
        if mouse_model is None: print("\nWARNING: Mouse AI disabled (model load failed).")
        if SEQUENCE_LENGTH is None or IMG_SIZE is None or screen_resolution is None:
            message = "FATAL ERROR: Critical metadata (SeqLen, ImgSize, ScreenRes) missing after model loading attempts. Models may be incompatible or corrupt."
            print(f"\n{message}"); sys.exit(1)

        print("-" * 60 + "\n--- Final Agent Configuration ---")
        print(f"  Sequence Length: {SEQUENCE_LENGTH}, Image Size: {IMG_SIZE}, Screen Resolution: {screen_resolution}")
        print(f"  Intervals (ms): FolderScan={FOLDER_SCAN_INTERVAL_SEC*1000:.0f}, MinInference={MIN_INFERENCE_INTERVAL_SEC*1000:.0f}")
        if keyboard_model: print(f"  KB AI: Enabled (Conf: {KEYBOARD_CONFIDENCE_THRESHOLD:.2f}, MaxKeys: {MAX_PARALLEL_KEYS_OUTPUT}, Classes: {len(idx_to_key)}, Forbidden: {len(FORBIDDEN_AI_KEYS)})")
        else: print("  KB AI: Disabled")
        if mouse_model: print(f"  Mouse AI: Enabled (Conf: {MOUSE_ACTION_CONFIDENCE_THRESHOLD:.2f}, Classes: {len(idx_to_mouse_action)})")
        else: print("  Mouse AI: Disabled")
        print("-" * 60)

        print("Initializing Tkinter and Task Manager...")
        root = tk.Tk(); root.withdraw();
        root.protocol("WM_DELETE_WINDOW", on_closing)
        task_manager = AIAgentTaskManager(root, keyboard_model, mouse_model, keyboard_metadata, mouse_metadata, device)
        print("Task Manager initialized.")

        task_manager.start_all()

        print("=" * 60 + "\n>>> AI Agent is RUNNING <<<\n" + "-" * 60)
        print(f" Monitoring folder: {SCREENSHOT_DIR}")
        print(f" Control:\n  - Press Q then P (or P then Q) within {QP_EXIT_THRESHOLD_NS/1e9:.1f} sec to QUIT Application\n" + "=" * 60)

        # Main Application Loop (Keep main thread alive for Tkinter events and clean shutdown)
        while not agent_stop_event.is_set():
             try:
                 # Process Tkinter events required for root.after() and WM_DELETE_WINDOW
                 root.update_idletasks()
                 root.update()
             except tk.TclError as e:
                 if "application has been destroyed" in str(e).lower():
                     print("Tkinter root destroyed unexpectedly. Ensuring shutdown.")
                 else: print(f"Tkinter loop error: {e}")
                 if not agent_stop_event.is_set(): agent_stop_event.set()
                 if not _shutdown_in_progress.is_set(): on_closing() # Attempt cleanup if not already started
                 break # Exit loop

             # Use wait on the stop event for sleeping, makes shutdown faster
             if agent_stop_event.wait(timeout=0.05): # Check ~20 times per second
                 break

        print("Main application loop finished.")
        if not _shutdown_in_progress.is_set():
            print("Main loop exited, ensuring final shutdown...")
            on_closing()

    except (ValueError, RuntimeError, OSError, FileNotFoundError) as e_init:
        print(f"\n--- INITIALIZATION ERROR ---")
        print(f"{type(e_init).__name__}: {e_init}")
        print(f"{'-'*40}"); traceback.print_exc(limit=3)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected during setup/run. Initiating shutdown...")
        if not _shutdown_in_progress.is_set(): on_closing()
    except Exception as e_main:
        print(f"\n{'='*40}\n UNEXPECTED ERROR during Setup / Main Execution \n{'='*40}")
        print(f" Type: {type(e_main).__name__}\n Error: {e_main}\n{'-'*40}")
        traceback.print_exc()
        if not _shutdown_in_progress.is_set(): on_closing()
        sys.exit(1)
    finally:
        # Final check to ensure shutdown runs if it hasn't already
        if not _shutdown_in_progress.is_set():
             print("Executing final cleanup...")
             on_closing()
        print("\nAI Agent Application finished.")
        sys.exit(0)