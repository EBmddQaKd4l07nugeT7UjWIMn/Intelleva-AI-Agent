# -*- coding: utf-8 -*-
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
import queue  # For thread-safe communication
import math # For distance calculation
import platform # For potential platform-specific tweaks
import traceback # For detailed error logging

# --- Optional: Try importing GPU libraries ---
GPU_LIB = None
try:
    # Prioritize pynvml for more detailed NVIDIA info if available
    import pynvml
    GPU_LIB = "pynvml"
    try:
        print("Initializing pynvml...")
        pynvml.nvmlInit()
        print("pynvml initialized successfully.")
        # Check if devices exist
        try:
             device_count = pynvml.nvmlDeviceGetCount()
             if device_count == 0:
                 print("WARN: pynvml initialized, but no NVIDIA devices found.")
                 GPU_LIB = None # No devices to monitor
             else:
                 print(f"Found {device_count} NVIDIA device(s). Using pynvml for GPU monitoring.")
        except pynvml.NVMLError_LibraryNotFound:
             print("WARN: NVML Library found but potentially not functional (driver issue?). Disabling pynvml.")
             GPU_LIB = None
             try: pynvml.nvmlShutdown() # Attempt shutdown if init partially worked
             except: pass
        except pynvml.NVMLError as e:
             print(f"WARN: NVML Initialization check failed: {e}. Disabling pynvml.")
             GPU_LIB = None
             try: pynvml.nvmlShutdown()
             except: pass
        except Exception as e: # Catch unexpected init errors
             print(f"WARN: Unexpected error during NVML init check: {e}. Disabling pynvml.")
             GPU_LIB = None
             try: pynvml.nvmlShutdown()
             except: pass
    except pynvml.NVMLError as e:
        GPU_LIB = None
        print(f"WARNING: Error initializing pynvml: {e}. GPU information will be unavailable.")
    except ImportError:
        GPU_LIB = None # pynvml not installed
        print("pynvml not found, checking for gpustat...")
    except Exception as e:
        GPU_LIB = None
        print(f"WARNING: An unexpected error occurred during pynvml import/init: {e}. GPU info unavailable.")

    # Fallback to gpustat if pynvml failed or wasn't imported/initialized
    if GPU_LIB is None:
        print("Attempting to use gpustat as fallback...")
        try:
            import gpustat
            # Basic check to see if gpustat can query
            gpustat.new_query() # Throws exception if fails
            GPU_LIB = "gpustat"
            print("Using gpustat for GPU monitoring.")
        except ImportError:
            GPU_LIB = None
            print("WARNING: Cannot import gpustat either. GPU information will be unavailable.")
        except Exception as e:
            GPU_LIB = None
            print(f"WARNING: gpustat found but failed during initial query: {e}. GPU info unavailable.")

except ImportError:
    # This outer ImportError means neither pynvml nor gpustat could be initially imported
    GPU_LIB = None
    print("WARNING: Cannot import pynvml or gpustat. GPU information will be unavailable.")
except Exception as e:
    # Catch any other unexpected errors during the import/check phase
    GPU_LIB = None
    print(f"WARNING: An unexpected error occurred during GPU library check: {e}. GPU info unavailable.")


# --- Global Constants and Variables ---
SCRIPT_DIR = Path(__file__).parent.resolve() # Use resolve() for absolute path
EXPERIENCE_POOL_DIR = SCRIPT_DIR / "experience_pool"
SCREENSHOT_DIR = EXPERIENCE_POOL_DIR / "screenshots"
KEYBOARD_LOG_FILE = EXPERIENCE_POOL_DIR / "keyboard_log.jsonl"
MOUSE_LOG_FILE = EXPERIENCE_POOL_DIR / "mouse_log.jsonl"
RESULTS_LOG_FILE = EXPERIENCE_POOL_DIR / "results_log.jsonl"

# Thread-safe queue for GUI updates
gui_update_queue = queue.Queue()

# Event to control data collection pause state (thread-safe)
data_collection_paused = threading.Event()
data_collection_paused.clear() # Initial state: not paused

# --- Helper Functions ---

def get_gpu_usage():
    """Try to get GPU and VRAM usage percentage. Returns ('N/A', 'N/A') on failure."""
    gpu_percent = "N/A"
    vram_percent = "N/A"
    global GPU_LIB # Allow modification if NVML fails runtime

    if not GPU_LIB:
        return gpu_percent, vram_percent

    try:
        if GPU_LIB == "gpustat":
            # Consider adding a try-except around new_query specificially
            stats = gpustat.new_query()
            if stats:
                gpu_info = stats[0] # Assume first GPU
                gpu_percent = gpu_info.utilization if gpu_info.utilization is not None else 0.0
                vram_total = gpu_info.memory_total
                vram_used = gpu_info.memory_used
                if vram_total and vram_total > 0:
                    vram_percent = round((vram_used / vram_total) * 100, 1)
                else:
                    vram_percent = 0.0 # Report 0 if total is unknown/zero
            else:
                gpu_percent = "No Device"
                vram_percent = "No Device"

        elif GPU_LIB == "pynvml":
            # NVML might have been initialized but fail later (e.g., driver update)
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assume first GPU
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_percent = float(utilization.gpu) # Ensure float
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if mem_info.total > 0:
                    vram_percent = round((mem_info.used / mem_info.total) * 100, 1)
                else:
                    vram_percent = 0.0
            else:
                 gpu_percent = "No Device"
                 vram_percent = "No Device"

    except pynvml.NVMLError as e:
         # Handle NVML errors more specifically if needed
        if not getattr(get_gpu_usage, "nvml_error_logged", False):
             print(f"ERROR getting GPU info via pynvml: {e}. Check NVIDIA drivers/runtime. Disabling NVML for this session.")
             get_gpu_usage.nvml_error_logged = True # Log only once per run
             GPU_LIB = None # Disable NVML if it fails during runtime
             try: pynvml.nvmlShutdown() # Try to clean up
             except: pass
        gpu_percent = "Error NVML"
        vram_percent = "Error NVML"
    except Exception as e:
        # Catch other potential errors (e.g., gpustat issues)
        library_name = GPU_LIB or "unavailable library"
        if not getattr(get_gpu_usage, f"{library_name}_error_logged", False):
            print(f"ERROR getting GPU info (using {library_name}): {e}. Will not log further generic GPU errors for this library.")
            setattr(get_gpu_usage, f"{library_name}_error_logged", True) # Log only once per run
        gpu_percent = "Error"
        vram_percent = "Error"

    return gpu_percent, vram_percent

def ensure_experience_pool():
    """Check and create necessary experience pool files and directories."""
    print("Checking experience pool structure...")
    try:
        if not EXPERIENCE_POOL_DIR.exists():
            print(f"Creating directory: {EXPERIENCE_POOL_DIR}")
            EXPERIENCE_POOL_DIR.mkdir(parents=True, exist_ok=True)
        if not SCREENSHOT_DIR.exists():
            print(f"Creating directory: {SCREENSHOT_DIR}")
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        # Create empty files if they don't exist
        for fpath in [KEYBOARD_LOG_FILE, MOUSE_LOG_FILE, RESULTS_LOG_FILE]:
            if not fpath.exists():
                print(f"Creating file: {fpath}")
                fpath.touch()
        print("Experience pool structure checked/created successfully.")
    except OSError as e:
        print(f"ERROR: Could not create experience pool directories/files: {e}")
        # Consider exiting or raising the exception if essential
        raise # Re-raise the error as this is critical

def count_log_lines(filepath):
    """Count lines (data entries) in a JSONL file."""
    count = 0
    if not filepath.exists(): return 0 # Avoid FileNotFoundError
    try:
        with filepath.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): # Count only non-empty lines
                    count += 1
    except Exception as e:
        print(f"ERROR reading line count for {filepath}: {e}")
        # Optionally return -1 or some indicator of error? Returning current count is okay.
    return count

def count_screenshots():
    """Count screenshot PNG files."""
    count = 0
    if not SCREENSHOT_DIR.exists(): return 0
    try:
        # Use generator expression for memory efficiency
        count = sum(1 for f in SCREENSHOT_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.png')
    except Exception as e:
        print(f"ERROR counting screenshots in {SCREENSHOT_DIR}: {e}")
    return count

# --- Data Logging Function ---
log_lock = threading.Lock() # Lock for writing to log files

def write_log(filepath, data):
    """Append data as a JSON line to the specified file (thread-safe)."""
    try:
        with log_lock: # Ensure only one thread writes at a time to avoid garbled files
            with filepath.open('a', encoding='utf-8') as f:
                json.dump(data, f, separators=(',', ':')) # Use compact separators
                f.write('\n')
        return True
    except Exception as e:
        print(f"ERROR writing log to {filepath}: {e}")
        traceback.print_exc() # Print full traceback for logging errors
        return False

# --- Window A: Status Monitoring ---

class MonitoringWindow:
    # Constants for resizing
    BASE_WIDTH = 450
    BASE_HEIGHT = 350
    BASE_FONT_SIZE = 10
    MIN_FONT_SIZE = 7
    MAX_FONT_SIZE = 20

    def __init__(self, root):
        self.root = root
        self.root.title("System Status & Data Monitor")

        # Intelligent Initial Sizing
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        # Start with base size, but allow it to grow up to a screen percentage/max size
        win_width = max(self.BASE_WIDTH, min(int(screen_width * 0.25), 700))
        win_height = max(self.BASE_HEIGHT, min(int(screen_height * 0.35), 600))
        # Position near top-right corner
        win_x = screen_width - win_width - 30
        win_y = 30
        self.root.geometry(f"{win_width}x{win_height}+{win_x}+{win_y}")
        self.root.minsize(380, 300) # Reasonable minimum size

        self.current_width = win_width
        self.current_height = win_height

        # Main Frame with Padding - Use ttk for consistency
        self.frame = ttk.Frame(root, padding="10")
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Labels for Displaying Stats
        self.stat_labels = {}
        self.title_labels = [] # Track all labels for font resizing

        stats_to_display = [
            ("CPU Usage:", "cpu_usage"),
            ("RAM Usage:", "ram_usage"),
            ("GPU Usage:", "gpu_usage"),
            ("VRAM Usage:", "vram_usage"),
            ("Screenshot Freq (Hz):", "screenshot_freq"),
            ("--- Data Counts ---", None), # Visual separator
            ("Screenshots:", "screenshot_count"),
            ("Keyboard Events:", "keyboard_count"),
            ("Mouse Events:", "mouse_count"),
            ("Game Results:", "result_count")
        ]

        # Grid Layout Configuration - Make columns/rows resize proportionally
        self.frame.grid_columnconfigure(0, weight=1, uniform="labelcol") # Label column
        self.frame.grid_columnconfigure(1, weight=3, uniform="valuecol") # Value column (more space)

        # Calculate initial font based on initial size (optional, could just start with BASE)
        initial_scale = min(win_width / self.BASE_WIDTH, win_height / self.BASE_HEIGHT)
        initial_font_size = max(self.MIN_FONT_SIZE, min(self.MAX_FONT_SIZE, int(self.BASE_FONT_SIZE * initial_scale)))

        self.initial_font = ('Segoe UI', initial_font_size) # Use a common modern font if available
        self.initial_title_font = ('Segoe UI', initial_font_size, 'bold')
        self._last_applied_font_size = initial_font_size # Track applied size


        for i, (text, key) in enumerate(stats_to_display):
            self.frame.grid_rowconfigure(i, weight=1) # Allow rows to expand vertically

            if key:
                # Stat row
                label_text = ttk.Label(self.frame, text=text, anchor="w", font=self.initial_font)
                label_text.grid(row=i, column=0, sticky="nsew", padx=(0, 5), pady=2)
                label_value = ttk.Label(self.frame, text="N/A", anchor="w", relief=tk.SUNKEN, padding=(5,2), font=self.initial_font)
                label_value.grid(row=i, column=1, sticky="nsew", padx=(5, 0), pady=2)
                self.stat_labels[key] = label_value
                self.title_labels.append(label_text) # Add both labels for resizing
                self.title_labels.append(label_value)
            else:
                # Separator row
                sep_label = ttk.Label(self.frame, text=text, anchor="w", font=self.initial_title_font)
                # Span across columns, center it horizontally? Use 'ew' sticky.
                sep_label.grid(row=i, column=0, columnspan=2, sticky="ew", padx=0, pady=(10, 2))
                self.title_labels.append(sep_label)

        # Bind resize event to the frame
        self.frame.bind("<Configure>", self._on_resize)

        # Start checking the update queue
        self.check_queue()

    def update_stats(self, stats_dict):
        """Update GUI labels with data from stats_dict."""
        for key, value in stats_dict.items():
            if key in self.stat_labels:
                formatted_value = "N/A" # Default
                try:
                    if isinstance(value, (int, float)):
                        if key.endswith("_usage") or key == "vram_usage":
                            formatted_value = f"{value:.1f}%"
                        elif key.endswith("_freq"):
                             formatted_value = f"{value:.1f}" # Already Hz
                        elif key.endswith("_count"):
                            formatted_value = f"{int(value):,}" # Format counts with commas
                        else:
                            formatted_value = str(value)
                    elif isinstance(value, str) and (value.startswith("Error") or value == "No Device"):
                        formatted_value = value # Show error strings directly
                    elif value is None:
                         formatted_value = "N/A"
                    else:
                        formatted_value = str(value) # Fallback
                except (ValueError, TypeError) as e:
                     print(f"Warning: Could not format value for {key}: {value} ({type(value)}) - Error: {e}")
                     formatted_value = "Error fmt"

                self.stat_labels[key].config(text=formatted_value)

    def check_queue(self):
        """Check the GUI update queue and process messages."""
        try:
            while True: # Process all pending updates
                update_data = gui_update_queue.get_nowait()
                self.update_stats(update_data)
        except queue.Empty:
            pass # No more updates for now
        except Exception as e:
            print(f"Error processing GUI update queue: {e}")
            traceback.print_exc()
        finally:
            # Schedule the next check safely using root
            self.root.after(150, self.check_queue) # Check slightly less often

    def _on_resize(self, event):
        """Handles window resize events to adjust font size."""
        # Use the frame's dimensions as it triggers the event
        new_width = event.width
        new_height = event.height

        # Calculate scaling factor based on the smaller dimension change ratio
        width_scale = new_width / self.BASE_WIDTH
        height_scale = new_height / self.BASE_HEIGHT
        scale = min(width_scale, height_scale) # Use the limiting dimension

        # Calculate new font size, clamping it within limits
        new_size = max(self.MIN_FONT_SIZE, min(self.MAX_FONT_SIZE, int(self.BASE_FONT_SIZE * scale)))

        # Avoid unnecessary font updates if size hasn't changed
        if new_size == self._last_applied_font_size:
            return

        # print(f"Resizing window: {new_width}x{new_height}, Scale: {scale:.2f}, New Font Size: {new_size}") # Debug

        # Create new font objects (Tkinter typically handles caching)
        new_font_tuple = ('Segoe UI', new_size)
        new_bold_font_tuple = ('Segoe UI', new_size, 'bold')

        # Update all labels stored in title_labels
        for label in self.title_labels:
            try:
                current_font_config = label.cget("font") # Get current font configuration
                # Check if the current font is bold (heuristic check)
                is_bold = 'bold' in str(current_font_config).lower()

                if is_bold:
                    label.config(font=new_bold_font_tuple)
                else:
                    label.config(font=new_font_tuple)
            except tk.TclError:
                # Widget might have been destroyed during rapid resize/close
                pass
            except Exception as e:
                 print(f"Error updating font for label {label}: {e}")


        self._last_applied_font_size = new_size # Store the applied size

        # Optional: Update current dimensions if needed elsewhere
        self.current_width = new_width
        self.current_height = new_height

# --- Window B: Game Result Selection ---

class ResultWindow:
    _instance = None # Class variable to track the single instance

    def __new__(cls, parent_root):
        # Prevent multiple instances
        if cls._instance is None or not cls._instance.window.winfo_exists():
            cls._instance = super(ResultWindow, cls).__new__(cls)
            return cls._instance
        else:
            # If instance exists, bring it to front and focus
            print("Result window already exists. Bringing to front.")
            try:
                cls._instance.window.lift()
                cls._instance.window.focus_force()
                cls._instance.window.grab_set() # Re-grab focus
            except tk.TclError as e:
                 print(f"Error focusing existing result window (may be closing): {e}")
                 cls._instance = None # Clear potentially dead instance
                 # Allow creation of a new one by returning None or raising error?
                 # Returning None might lead to the caller creating a new one anyway.
                 # Let's return the (potentially broken) instance for now, caller handles None check.
            return cls._instance


    def __init__(self, parent_root):
        # Check if this instance was already initialized (due to __new__ logic)
        if hasattr(self, 'initialized') and self.initialized:
             return

        self.parent_root = parent_root
        self.result = None
        self.selected_button_var = tk.StringVar(value="") # Tracks Lose/Draw/Win selection

        # --- Pause Data Collection ---
        if not data_collection_paused.is_set():
             print("Pausing data collection for result selection...")
             data_collection_paused.set()

        self.window = tk.Toplevel(parent_root)
        self.window.title("Select Game Result")
        self.window.transient(parent_root)
        self.window.protocol("WM_DELETE_WINDOW", self._handle_close_attempt)
        self.window.resizable(False, False)

        # --- Intelligent Initial Sizing & Centering ---
        # Estimates (adjust as needed)
        button_width_chars = 10 # tk uses char width estimates
        button_internal_padx = 15
        button_internal_pady = 5
        button_external_padx = 10 # Padding between buttons
        frame_padx = 20 # Padding around frames
        frame_pady = 15
        button_font_size = 11 # Used for height estimation maybe

        # Calculate width based on widest row (3 result buttons)
        est_btn_width_pixels = button_width_chars * button_font_size + 2 * button_internal_padx # Rough guess
        win_width = (est_btn_width_pixels * 3) + (button_external_padx * 2) + (frame_padx * 2)
        win_width = max(win_width, 450) # Ensure a minimum reasonable width

        # Calculate height based on two rows of buttons + padding
        # Height estimation is trickier, use fixed value or font metrics if possible
        est_btn_height_pixels = 2 * button_internal_pady + button_font_size * 2 # Very rough guess
        win_height = (est_btn_height_pixels * 2) + frame_pady * 3 # Two rows, 3 vertical padding gaps
        win_height = max(win_height, 180) # Ensure minimum height

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        # Constrain by screen percentage
        win_width = min(win_width, int(screen_width * 0.6))
        win_height = min(win_height, int(screen_height * 0.3))

        win_x = (screen_width // 2) - (win_width // 2)
        win_y = (screen_height // 2) - (win_height // 2)
        self.window.geometry(f"{win_width}x{win_height}+{win_x}+{win_y}")

        # --- Frames for Layout (Using standard tk Frame here is fine) ---
        top_frame = tk.Frame(self.window)
        # Apply padding via pack - this was the original suspicion for the TclError source
        top_frame.pack(expand=True, fill=tk.X, padx=frame_padx, pady=(frame_pady, frame_pady // 2))

        bottom_frame = tk.Frame(self.window)
        bottom_frame.pack(expand=True, fill=tk.X, padx=frame_padx, pady=(frame_pady // 2, frame_pady))

        # Configure column weights within frames for button expansion using grid
        top_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="top_buttons")
        bottom_frame.grid_columnconfigure((0, 1), weight=1, uniform="bottom_buttons")
        # Configure row weights if needed (only one row here)
        top_frame.grid_rowconfigure(0, weight=1)
        bottom_frame.grid_rowconfigure(0, weight=1)

        # --- Button Styling ---
        button_font = ('Segoe UI', button_font_size, 'bold')
        radio_style = ttk.Style()
        # Define a custom style for Radiobuttons that look like buttons
        radio_style_name = 'Result.TRadiobutton'
        radio_style.configure(radio_style_name,
                              font=button_font,
                              padding=(button_internal_padx, button_internal_pady),
                              anchor=tk.CENTER, # Center text
                              indicatoron=0, # Crucial: Hide the radio indicator
                              relief=tk.RAISED,
                              borderwidth=2)
        # Map states (e.g., selected, active) - This is tricky for background color
        # Tkinter/TTK makes it hard to change background color of selected radiobutton reliably cross-platform
        # We will use a command callback to visually change relief instead.

        # --- Result Buttons (Top Row - ttk.Radiobutton) ---
        results = [("LOSE", "lose", "red"), ("DRAW", "draw", "green"), ("WIN", "win", "blue")]
        self.result_buttons = {}

        for i, (text, value, color) in enumerate(results):
            # Create the radiobutton with the custom style
            rb = ttk.Radiobutton(top_frame, text=text, variable=self.selected_button_var, value=value,
                                 style=radio_style_name, # Apply custom style
                                 command=self._on_result_select) # Callback on selection change

            # Grid placement within top_frame
            rb.grid(row=0, column=i, sticky="nsew", padx=button_external_padx // 2, pady=5)
            self.result_buttons[value] = rb

            # --- HACK: Apply background/foreground colors directly AFTER creation ---
            # TTK styling for background/foreground on Radiobuttons (especially selected state)
            # is notoriously difficult and platform-dependent. Direct config might work better sometimes.
            try:
                s = ttk.Style()
                # Create a dynamic style for this specific button's colors
                dynamic_style_name = f'{text}.Result.TRadiobutton'
                s.configure(dynamic_style_name,
                            background=color,
                            foreground='white',
                            font=button_font,
                            padding=(button_internal_padx, button_internal_pady),
                            anchor=tk.CENTER,
                            relief=tk.RAISED,
                            borderwidth=2)
                # Map states to keep color consistent (might not fully work on all OS)
                s.map(dynamic_style_name,
                      background=[('active', color), ('selected', color), ('!selected', color)],
                      foreground=[('active', 'white'), ('selected', 'white'), ('!selected', 'white')])
                rb.config(style=dynamic_style_name)
            except Exception as style_e:
                print(f"Warning: Could not apply custom style colors for {text} button: {style_e}")
                # Fallback attempt using basic config (less likely to work well with ttk)
                try:
                     rb.config(background=color, foreground='white')
                except:
                     pass # Ignore if basic config fails too


        # --- Control Buttons (Bottom Row - tk.Button for direct color control) ---
        common_options = {
            "font": button_font,
            "relief": tk.RAISED,
            "borderwidth": 2,
            "width": button_width_chars, # Use char width estimate
            "padx": button_internal_padx,
            "pady": button_internal_pady,
        }

        # Skip Button
        skip_btn = tk.Button(bottom_frame, text="SKIP", command=self.skip,
                             bg="black", fg="white",
                             activebackground="gray20", activeforeground="white",
                             **common_options)
        skip_btn.grid(row=0, column=0, sticky="nsew", padx=button_external_padx // 2, pady=5)

        # Confirm Button
        confirm_btn = tk.Button(bottom_frame, text="CONFIRM", command=self.confirm,
                                bg="white", fg="black",
                                activebackground="gray90", activeforeground="black",
                                **common_options)
        confirm_btn.grid(row=0, column=1, sticky="nsew", padx=button_external_padx // 2, pady=5)

        # Final setup
        self.window.lift()
        self.window.focus_force()
        self.window.grab_set() # Make modal
        self.initialized = True # Mark as initialized

    def _on_result_select(self):
        """Callback when a result radio button is selected."""
        self.result = self.selected_button_var.get()
        print(f"Result selected: {self.result}")
        # Provide visual feedback by changing relief (since color is hard)
        for value, button in self.result_buttons.items():
             try:
                 if value == self.result:
                     button.state(['pressed']) # TTK state for pressed look (might change relief)
                     # Or direct relief change if state doesn't work visually:
                     # button.config(relief=tk.SUNKEN)
                 else:
                     button.state(['!pressed'])
                     # button.config(relief=tk.RAISED)
             except tk.TclError: pass # Ignore if button disappears

    def skip(self):
        """Handle 'SKIP' button click."""
        print("User chose to SKIP result logging.")
        self.result = "skip"
        self.close_window()

    def confirm(self):
        """Handle 'CONFIRM' button click."""
        selected_result = self.selected_button_var.get()
        if selected_result and selected_result in ["lose", "draw", "win"]:
            confirm_time_ns = time.time_ns()
            print(f"User CONFIRMED result: {selected_result} at {confirm_time_ns}")
            data = {
                "timestamp_ns": confirm_time_ns, # Use confirm time as the key
                "result": selected_result
            }
            if write_log(RESULTS_LOG_FILE, data):
                # Update GUI count after successful write
                gui_update_queue.put({"result_count": count_log_lines(RESULTS_LOG_FILE)})
            self.close_window()
        else:
            messagebox.showwarning("Selection Required",
                                   "Please select 'LOSE', 'DRAW', or 'WIN' before confirming.",
                                   parent=self.window) # Ensure messagebox is parented
            print("Confirm clicked without a valid result selection.")
            # Re-grab focus if lost
            self.window.grab_set()
            self.window.focus_force()


    def _handle_close_attempt(self):
        """Handle user trying to close the window (e.g., Alt+F4, close button). Treat as skip."""
        print("Result window close attempted - interpreting as SKIP.")
        self.skip()

    def close_window(self):
        """Clean up and close the result window."""
        if data_collection_paused.is_set():
            print("Resuming data collection.")
            data_collection_paused.clear() # Resume data collection

        # Destroy window and clean up instance tracker
        if self.window and self.window.winfo_exists():
            self.window.grab_release()
            self.window.destroy()

        # Important: Reset the class instance variable so a new one can be created
        ResultWindow._instance = None
        self.initialized = False # Mark as not initialized


# --- Background Task Manager ---

class BackgroundTaskManager:
    # Class variable to track the ResultWindow instance (moved from inside ResultWindow for clarity)
    # _result_window_instance = None # Using ResultWindow._instance instead

    def __init__(self, root_tk):
        self.root = root_tk # Main Tkinter instance for scheduling GUI tasks
        self.stop_event = threading.Event()
        self.threads = []

        # Screenshot settings
        self.base_screenshot_delay_sec = 1.0 # Target 1 Hz base frequency
        self.current_screenshot_delay_sec = self.base_screenshot_delay_sec
        self.min_screenshot_delay_sec = 0.2 # Max ~5 Hz
        self.max_screenshot_delay_sec = 5.0 # Min ~0.2 Hz

        # Keyboard listener state
        self.key_press_times = {} # {key_repr: press_time_ns}
        self.last_ctrl_press_time_ns = 0
        self.last_alt_press_time_ns = 0
        self.ctrl_alt_threshold_ns = 950_000_000 # 0.95 seconds

        # Mouse listener state
        self.mouse_press_info = {} # {button_name: {'press_time_ns': ns, 'start_pos': (x,y), 'trajectory':[]}}
        self.long_press_threshold_ns = 500_000_000 # 500 ms
        self.drag_threshold_pixels = 10 # Drag if moved more than 10 pixels

        # Pynput listeners (will be created in their threads)
        self.keyboard_listener = None
        self.mouse_listener = None

        # mss instance (created in screenshot thread)
        self.sct_instance = None

    def start_all(self):
        """Start all background threads."""
        self.stop_event.clear()
        data_collection_paused.clear() # Ensure not paused on start
        # Reset GPU error flags on start
        if hasattr(get_gpu_usage, "nvml_error_logged"): delattr(get_gpu_usage, "nvml_error_logged")
        # Reset flags for other potential library errors
        if GPU_LIB and hasattr(get_gpu_usage, f"{GPU_LIB}_error_logged"):
             delattr(get_gpu_usage, f"{GPU_LIB}_error_logged")


        thread_targets = {
            "SystemMonitor": self.monitor_system,
            "ScreenshotWorker": self.screenshot_worker,
            "KeyboardListener": self.keyboard_listener_worker,
            "MouseListener": self.mouse_listener_worker
        }
        self.threads = [] # Ensure list is clear before starting
        for name, target in thread_targets.items():
            thread = threading.Thread(target=target, name=name, daemon=True)
            self.threads.append(thread)
            thread.start()
        print(f"Started {len(self.threads)} background tasks.")

    def stop_all(self):
        """Signal all background threads to stop and wait for them."""
        print("Stopping background tasks...")
        self.stop_event.set() # Signal threads using the event

        # Stop pynput listeners safely
        # These stop() methods are designed to be called from other threads
        if self.keyboard_listener and self.keyboard_listener.is_alive():
             print("Requesting keyboard listener stop...")
             try:
                 # pynput listeners stop themselves when their controlling thread ends,
                 # but calling stop() explicitly can sometimes speed it up or unblock it.
                 # However, it can also cause issues if called at the wrong time.
                 # Relying on the thread checking self.stop_event is safer.
                 # keyboard.Listener.stop(self.keyboard_listener) # Alternative way
                 pass # Let the thread stop itself via stop_event check
             except Exception as e: print(f" Minor error requesting kbd listener stop: {e}")

        if self.mouse_listener and self.mouse_listener.is_alive():
             print("Requesting mouse listener stop...")
             try:
                 # mouse.Listener.stop(self.mouse_listener)
                 pass # Let the thread stop itself via stop_event check
             except Exception as e: print(f" Minor error requesting mouse listener stop: {e}")

        # Wait for threads to finish
        print("Waiting for threads to join...")
        active_threads_before_join = [t.name for t in self.threads if t.is_alive()]
        if active_threads_before_join:
             print(f"  Active threads: {', '.join(active_threads_before_join)}")

        for thread in self.threads:
            if thread.is_alive():
                # print(f"  Joining {thread.name}...") # Debug
                thread.join(timeout=3.0) # Increased timeout
                if thread.is_alive():
                      print(f"  WARNING: Thread {thread.name} did not stop gracefully after timeout.")
                # else:
                      # print(f"  Thread {thread.name} finished.") # Debug
            # else:
                 # print(f" Thread {thread.name} was already finished.") # Debug

        self.threads = [] # Clear thread list
        print("All background threads processed.")

        # Clean up GPU library
        if GPU_LIB == "pynvml":
            try:
                # Check if NVML is still initialized before shutting down
                # This requires storing the init state or just trying shutdown
                print("Attempting to shut down pynvml...")
                pynvml.nvmlShutdown()
                print("pynvml shutdown successful.")
            except pynvml.NVMLError as e:
                 # Don't warn if it was never initialized or already shut down
                 if e.value != pynvml.NVML_ERROR_UNINITIALIZED and e.value != pynvml.NVML_ERROR_NOT_FOUND:
                      print(f"WARN: Error shutting down pynvml: {e}")
            except Exception as e:
                 print(f"WARN: Unexpected error during pynvml shutdown: {e}")

        print("Background tasks stopped.")

    # --- System Monitoring Thread ---
    def monitor_system(self):
        """Periodically monitor system resources and update frequency/GUI."""
        print("System monitor thread started.")
        while not self.stop_event.is_set():
            loop_start_time = time.monotonic()
            try:
                # Check pause state first (although this thread doesn't collect user data directly)
                # if data_collection_paused.is_set():
                #     self.stop_event.wait(0.5) # Wait longer if paused
                #     continue

                cpu_usage = psutil.cpu_percent()
                ram = psutil.virtual_memory()
                ram_usage = ram.percent
                gpu_usage, vram_usage = get_gpu_usage()

                # --- Adaptive Screenshot Frequency Logic ---
                load_factor = 0.0
                # Add to factor if usage exceeds threshold, normalized contribution capped at 1.0 each
                if isinstance(cpu_usage, (int, float)):
                    load_factor += max(0.0, min(1.0, (cpu_usage - 70) / 30.0)) # 70-100% range -> 0-1 factor
                if isinstance(ram_usage, (int, float)):
                    load_factor += max(0.0, min(1.0, (ram_usage - 75) / 25.0)) # 75-100% range -> 0-1 factor
                if isinstance(gpu_usage, (int, float)):
                    load_factor += max(0.0, min(1.0, (gpu_usage - 75) / 25.0)) * 1.2 # GPU weighted slightly higher
                if isinstance(vram_usage, (int, float)):
                    load_factor += max(0.0, min(1.0, (vram_usage - 80) / 20.0)) * 0.8 # VRAM weighted slightly lower

                # Calculate delay multiplier: 1.0 (no load) up to (1 + max_load_contribution)
                # Max possible load_factor = 1.0 + 1.0 + 1.2 + 0.8 = 4.0
                # Let multiplier go from 1x up to 4x delay (controlled by max_screenshot_delay_sec)
                delay_multiplier = 1.0 + load_factor # Range 1.0 to 5.0
                target_delay = self.base_screenshot_delay_sec * delay_multiplier

                # Clamp delay within min/max bounds
                self.current_screenshot_delay_sec = max(self.min_screenshot_delay_sec, min(self.max_screenshot_delay_sec, target_delay))

                current_freq = 1.0 / self.current_screenshot_delay_sec if self.current_screenshot_delay_sec > 0 else float('inf')

                # --- Queue GUI Update ---
                stats_update = {
                    "cpu_usage": cpu_usage,
                    "ram_usage": ram_usage,
                    "gpu_usage": gpu_usage,
                    "vram_usage": vram_usage,
                    "screenshot_freq": current_freq,
                    # Data counts are updated by their respective workers upon successful write
                }
                gui_update_queue.put(stats_update)

            except Exception as e:
                print(f"ERROR in monitor_system loop: {e}")
                traceback.print_exc()
                # Avoid busy-looping on error
                wait_time = 5.0
            else:
                 # Calculate time spent and wait remaining part of the ~1 second interval
                 elapsed = time.monotonic() - loop_start_time
                 wait_time = max(0.05, 1.0 - elapsed) # Ensure minimum wait, target 1s loop

            # Wait for the next cycle, interruptible by stop_event
            self.stop_event.wait(wait_time)
        print("System monitor thread finished.")


    # --- Screenshotting Thread ---
    def screenshot_worker(self):
        """Continuously capture screenshots and save them."""
        print("Screenshot worker thread started.")
        self.sct_instance = None # Ensure it's None initially
        monitor_definition = None

        try:
            self.sct_instance = mss()
            print("mss instance created for screenshots.")

            # Determine primary monitor once (usually better performance than checking each time)
            # Monitor 0 is typically the combined virtual screen, Monitor 1 is the primary physical.
            if len(self.sct_instance.monitors) > 1:
                monitor_definition = self.sct_instance.monitors[1]
                print(f"Using monitor 1 (Primary physical): {monitor_definition}")
            elif len(self.sct_instance.monitors) == 1:
                monitor_definition = self.sct_instance.monitors[0]
                print(f"Using monitor 0 (Only one found - Virtual?): {monitor_definition}")
            else:
                print("ERROR: mss found no monitors!")
                # Optionally try default monitor dict?
                # monitor_definition = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080} # Example fallback
                raise RuntimeError("No monitors detected by mss")

            while not self.stop_event.is_set():
                loop_start_time = time.monotonic()
                current_delay = self.current_screenshot_delay_sec # Capture current delay for this iteration

                # Check pause state *before* potentially long operation
                if data_collection_paused.is_set():
                    self.stop_event.wait(0.2) # Check pause/stop status frequently when paused
                    continue # Skip the rest of the loop

                try:
                    timestamp_ns = time.time_ns()
                    filename = SCREENSHOT_DIR / f"{timestamp_ns}.png"

                    # Capture the screen using the pre-determined monitor definition
                    sct_img = self.sct_instance.grab(monitor_definition)

                    # Save the image to PNG file using mss helper
                    # Consider running this save in a separate thread pool if it becomes a bottleneck
                    to_png(sct_img.rgb, sct_img.size, output=str(filename))

                    # Update screenshot count in GUI (consider throttling this update if high frequency)
                    # Only update if file saving seemed successful (no exception)
                    if (timestamp_ns // 1_000_000_000) % 2 == 0: # Update count every ~2 seconds
                          gui_update_queue.put({"screenshot_count": count_screenshots()})

                except Exception as e:
                    print(f"ERROR during screenshot capture/save: {e}")
                    traceback.print_exc()
                    # Wait a bit longer after an error before retrying
                    wait_time = max(current_delay, 3.0) # Use current delay or 3s, whichever is longer
                    self.stop_event.wait(wait_time)
                    continue # Skip the normal delay wait at the end

                # Wait for the calculated delay, adjusted for processing time
                elapsed = time.monotonic() - loop_start_time
                wait_time = max(0.01, current_delay - elapsed) # Ensure a tiny minimum wait
                self.stop_event.wait(wait_time)

        except Exception as e:
             print(f"FATAL ERROR initializing mss or in screenshot loop: {e}")
             traceback.print_exc()
        finally:
            # Clean up mss instance if it was created
            if hasattr(self.sct_instance, 'close') and callable(self.sct_instance.close):
                 try:
                     self.sct_instance.close()
                     print("mss instance closed.")
                 except Exception as e_close:
                      print(f"Error closing mss instance: {e_close}")
            self.sct_instance = None
            print("Screenshot worker finished.")


    # --- Keyboard Listener Callbacks & Thread ---
    def _get_key_repr(self, key):
        """Get a consistent string representation for a key."""
        try:
            if isinstance(key, keyboard.Key):
                return key.name
            elif isinstance(key, keyboard.KeyCode):
                # Use char if available and printable, otherwise use vk or name if possible
                if key.char and key.char.isprintable():
                    return key.char
                # Fallback for non-printable or None char
                # Try name if available (pynput >= 1.7.0)
                if hasattr(key, 'name'): return key.name
                # Try vk if available
                if hasattr(key, 'vk'): return f"vk_{key.vk}"
                # Final fallback
                return str(key)
            else:
                return str(key)
        except Exception:
             # Fallback for unexpected key types or errors
             return repr(key)

    def _on_press(self, key):
        press_time_ns = time.time_ns() # Capture time immediately

        # Check pause state *after* capturing time but *before* processing
        if data_collection_paused.is_set(): return

        try:
            key_repr = self._get_key_repr(key)

            # Store press time, overwriting if key is pressed again before release
            self.key_press_times[key_repr] = press_time_ns

            # --- Ctrl + Alt Combination Check ---
            # Ignore Shift, CapsLock etc for the combo trigger itself
            if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
                       keyboard.Key.caps_lock, keyboard.Key.cmd, keyboard.Key.cmd_l,
                       keyboard.Key.cmd_r, keyboard.Key.scroll_lock, keyboard.Key.num_lock):
                return

            is_ctrl = key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)
            # Include AltGr as Alt
            is_alt = key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr)

            combo_detected = False
            if is_ctrl:
                current_ctrl_time = press_time_ns
                # Check if Alt was pressed recently *before* this Ctrl
                if self.last_alt_press_time_ns > 0 and (current_ctrl_time - self.last_alt_press_time_ns) < self.ctrl_alt_threshold_ns:
                    print(f"Ctrl+Alt detected (Alt then Ctrl)! Diff: {(current_ctrl_time - self.last_alt_press_time_ns)/1e6:.1f} ms")
                    combo_detected = True
                # Always update last_ctrl_press_time, even if part of combo, to handle rapid presses
                self.last_ctrl_press_time_ns = current_ctrl_time

            elif is_alt:
                current_alt_time = press_time_ns
                # Check if Ctrl was pressed recently *before* this Alt
                if self.last_ctrl_press_time_ns > 0 and (current_alt_time - self.last_ctrl_press_time_ns) < self.ctrl_alt_threshold_ns:
                    print(f"Alt+Ctrl detected (Ctrl then Alt)! Diff: {(current_alt_time - self.last_ctrl_press_time_ns)/1e6:.1f} ms")
                    combo_detected = True
                # Always update last_alt_press_time
                self.last_alt_press_time_ns = current_alt_time

            # Trigger window creation if combo detected
            if combo_detected:
                # Schedule GUI action in main thread
                self.root.after(0, self._trigger_result_window)
                # Reset timers immediately AFTER scheduling to prevent re-trigger on release/re-press
                self.last_alt_press_time_ns = 0
                self.last_ctrl_press_time_ns = 0

            # --- Stale Modifier Cleanup (Optional but good practice) ---
            # If a modifier was pressed > threshold ago, reset its time
            cleanup_threshold_ns = self.ctrl_alt_threshold_ns + 100_000_000 # Slightly > 1s
            if self.last_ctrl_press_time_ns > 0 and (press_time_ns - self.last_ctrl_press_time_ns) > cleanup_threshold_ns:
                 # print("Cleaning up stale Ctrl time") # Debug
                 self.last_ctrl_press_time_ns = 0
            if self.last_alt_press_time_ns > 0 and (press_time_ns - self.last_alt_press_time_ns) > cleanup_threshold_ns:
                 # print("Cleaning up stale Alt time") # Debug
                 self.last_alt_press_time_ns = 0

        except Exception as e:
            print(f"ERROR in _on_press callback for key {key}: {e}")
            traceback.print_exc()


    def _on_release(self, key):
        release_time_ns = time.time_ns() # Capture time immediately

        # Check pause state
        if data_collection_paused.is_set(): return

        try:
            key_repr = self._get_key_repr(key)
            key_name = key_repr # Use the same representation for logging name for simplicity now

            # Check if we have a press time recorded for this key representation
            if key_repr in self.key_press_times:
                press_time_ns = self.key_press_times.pop(key_repr) # Get and remove press time

                # Sanity check timestamps
                if release_time_ns < press_time_ns:
                    print(f"WARN: Key '{key_name}' release time ({release_time_ns}) is before press time ({press_time_ns}). Skipping log.")
                    # Maybe still reset modifier state below?
                else:
                    # Prepare data payload
                    key_data = {
                        "timestamp_ns": press_time_ns, # Use press time as the main identifier
                        "press_time_ns": press_time_ns,
                        "release_time_ns": release_time_ns,
                        "key_name": key_name
                    }

                    # Write to log file and update GUI count on success
                    if write_log(KEYBOARD_LOG_FILE, key_data):
                        gui_update_queue.put({"keyboard_count": count_log_lines(KEYBOARD_LOG_FILE)})

            # else: # Release event without a corresponding press in our dict
                # This is normal if logging started while a key was held down.
                # print(f"Debug: Release event for '{key_name}' without tracked press time.")
                pass

            # --- Reset Ctrl/Alt state on release of the specific modifier ---
            # Prevents combo if Ctrl held, Alt tapped, Ctrl released later.
            if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self.last_ctrl_press_time_ns = 0
            elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
                self.last_alt_press_time_ns = 0

        except Exception as e:
             print(f"ERROR in _on_release callback for key {key}: {e}")
             traceback.print_exc()

    def keyboard_listener_worker(self):
        """Manages the pynput keyboard listener."""
        print("Keyboard listener thread started.")
        listener = None
        try:
            listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self.keyboard_listener = listener # Store reference for stop_all
            print("Keyboard listener created and starting...")
            # listener.start() # This starts a *new* thread internally
            # listener.join() # This would wait for that internal thread

            # Instead, run the listener directly in *this* worker thread
            # This makes stopping via self.stop_event more direct.
            # We need a way to check self.stop_event periodically.
            # The 'run' method blocks, so we use 'start' and manage the loop.
            listener.start()
            print("Keyboard listener running.")
            # Keep this thread alive while the listener is running and stop_event is not set
            while listener.is_alive() and not self.stop_event.is_set():
                 # Check periodically - needed if listener blocks internally somehow
                 self.stop_event.wait(0.5) # Check stop event every 500ms

            print("Keyboard listener stop condition met.")
            # If the loop exited because stop_event was set, explicitly stop the listener
            if listener.is_alive():
                print("Requesting explicit listener stop...")
                listener.stop()

            # Wait for the listener's internal thread to actually finish
            print("Waiting for listener join...")
            listener.join()
            print("Listener joined.")

        except Exception as e:
             print(f"FATAL ERROR in keyboard listener worker: {e}")
             traceback.print_exc()
        finally:
             print("Keyboard listener worker finished.")
             self.keyboard_listener = None # Clear reference


    # --- Mouse Listener Callbacks & Thread ---
    def _get_mouse_button_name(self, button):
        """Get a string name for a mouse button."""
        try:
             return button.name
        except AttributeError:
             # Handle special mouse buttons (e.g., XButton1/2)
             return str(button)

    def _on_move(self, x, y):
        # Check pause state early
        if data_collection_paused.is_set(): return

        current_time_ns = time.time_ns()
        # Append movement to trajectory ONLY if a button is currently pressed
        # Iterate over a copy of keys as the dictionary might change during iteration
        for button_name in list(self.mouse_press_info.keys()):
            if button_name in self.mouse_press_info: # Check again if still exists
                 info = self.mouse_press_info[button_name]
                 # Only record move if needed (e.g., if drag potential exists)
                 # Let's always record for simplicity, filter later if needed.
                 info['trajectory'].append((x, y, current_time_ns))
                 # Optional: Limit trajectory size for memory
                 MAX_TRAJ_POINTS = 2000
                 if len(info['trajectory']) > MAX_TRAJ_POINTS:
                     # Keep the start and the last N points? Or just prune oldest?
                     info['trajectory'].pop(1) # Remove second oldest point (keep start)


    def _on_click(self, x, y, button, pressed):
        current_time_ns = time.time_ns() # Capture time immediately

        # Check pause state
        if data_collection_paused.is_set(): return

        try:
            button_name = self._get_mouse_button_name(button)
            # We only care about left and right clicks as per requirement? Assume yes.
            if button_name not in ['left', 'right']:
                 return # Ignore middle, scroll wheel clicks, X buttons etc.

            if pressed:
                # Record press info, overwriting if clicked again before release
                self.mouse_press_info[button_name] = {
                    'press_time_ns': current_time_ns,
                    'start_pos': (int(x), int(y)), # Ensure integer coords
                    'trajectory': [(int(x), int(y), current_time_ns)] # Start trajectory
                }
            else: # Button released
                # Process the event only if we have corresponding press info
                if button_name in self.mouse_press_info:
                    info = self.mouse_press_info.pop(button_name) # Get and remove data
                    press_time_ns = info['press_time_ns']
                    start_pos = info['start_pos']
                    end_pos = (int(x), int(y)) # Release position
                    trajectory = info['trajectory']
                    release_time_ns = current_time_ns # Use current time as release time

                    # Sanity check timestamps
                    if release_time_ns < press_time_ns:
                        print(f"WARN: Mouse '{button_name}' release time ({release_time_ns}) is before press time ({press_time_ns}). Skipping log.")
                        return

                    duration_ns = release_time_ns - press_time_ns

                    # Determine action type: click, long_press, or drag
                    # Calculate distance between start and end points
                    dx = end_pos[0] - start_pos[0]
                    dy = end_pos[1] - start_pos[1]
                    distance = math.hypot(dx, dy) # Use hypot for Euclidean distance

                    action_type = "click" # Default
                    if distance >= self.drag_threshold_pixels:
                        action_type = "drag"
                    elif duration_ns >= self.long_press_threshold_ns:
                        action_type = "long_press"

                    # Prepare data payload
                    mouse_data = {
                        "timestamp_ns": press_time_ns, # Use press time as main identifier
                        "press_time_ns": press_time_ns,
                        "release_time_ns": release_time_ns,
                        "action": action_type,
                        "button": button_name,
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                    }
                    # Only include trajectory for drags
                    if action_type == "drag":
                        # Ensure trajectory has start and end points (should be covered)
                        # Add end_pos to trajectory if not captured by _on_move (unlikely but safe)
                        if not trajectory or trajectory[-1][:2] != end_pos:
                             trajectory.append((end_pos[0], end_pos[1], release_time_ns))
                        mouse_data["trajectory"] = trajectory

                    # Write log and update GUI
                    if write_log(MOUSE_LOG_FILE, mouse_data):
                        gui_update_queue.put({"mouse_count": count_log_lines(MOUSE_LOG_FILE)})
                # else: # Release without tracked press
                    # print(f"Debug: Release event for mouse button '{button_name}' without tracked press.")
                    pass

        except Exception as e:
             print(f"ERROR in _on_click callback for button {button} ({pressed}): {e}")
             traceback.print_exc()

    def _on_scroll(self, x, y, dx, dy):
        # Requirement doesn't explicitly mention scrolling, so ignore.
        if data_collection_paused.is_set(): return
        pass

    def mouse_listener_worker(self):
        """Manages the pynput mouse listener."""
        print("Mouse listener thread started.")
        listener = None
        try:
            listener = mouse.Listener(on_move=self._on_move,
                                      on_click=self._on_click,
                                      on_scroll=self._on_scroll)
            self.mouse_listener = listener # Store reference
            print("Mouse listener created and starting...")
            # Run listener directly in this thread for better stop_event handling
            listener.start()
            print("Mouse listener running.")

            while listener.is_alive() and not self.stop_event.is_set():
                self.stop_event.wait(0.5) # Check stop event periodically

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
             self.mouse_listener = None # Clear reference


    # --- Trigger for Result Window (runs in main GUI thread via root.after) ---
    def _trigger_result_window(self):
        """Creates or focuses the result window. Runs in main GUI thread."""
        # Use the ResultWindow's class-level instance checking
        print("Attempting to trigger result window...")
        try:
            # This will either create a new window or focus the existing one
            # The __new__ method in ResultWindow handles the logic.
             instance = ResultWindow(self.root)
             # If __new__ returned an existing but broken window (e.g., TclError on lift),
             # instance might be None or unusable. A check could be added here.
             if instance is None or not instance.window.winfo_exists():
                  print("Failed to create or focus ResultWindow instance.")
             else:
                  print("ResultWindow triggered/focused successfully.")
        except Exception as e:
            print(f"ERROR trying to create/focus ResultWindow: {e}")
            traceback.print_exc()
            # Ensure data collection resumes if window creation failed badly
            if data_collection_paused.is_set():
                 print("Resuming data collection due to ResultWindow trigger error.")
                 data_collection_paused.clear()


# --- Main Application Entry Point ---
if __name__ == "__main__":
    print("========================================")
    print(" Starting Data Recorder Application")
    print("========================================")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version: {platform.python_version()}")

    # 1. Ensure Experience Pool Structure Exists
    try:
        ensure_experience_pool()
    except Exception as e:
        message = f"CRITICAL: Failed to set up experience pool at\n{EXPERIENCE_POOL_DIR}\n\nError: {e}\n\nApplication cannot continue."
        print(message)
        # Try showing a simple Tk messagebox if possible, even without full app setup
        try:
            root_err = tk.Tk()
            root_err.withdraw()
            messagebox.showerror("Startup Error", message, parent=None)
            root_err.destroy()
        except Exception: # If Tkinter itself fails early
            pass # Message already printed to console
        exit(1)

    root = None
    task_manager = None

    try:
        # 2. Initialize Tkinter Root Window
        root = tk.Tk()
        root.withdraw() # Hide root initially while setting up

        # 3. Create Monitoring Window
        # Should show the window implicitly
        app = MonitoringWindow(root)
        root.deiconify() # Make window visible now

        # 4. Get Initial Data Counts and Update GUI
        print("Getting initial data counts...")
        initial_counts = {
            "screenshot_count": count_screenshots(),
            "keyboard_count": count_log_lines(KEYBOARD_LOG_FILE),
            "mouse_count": count_log_lines(MOUSE_LOG_FILE),
            "result_count": count_log_lines(RESULTS_LOG_FILE)
        }
        # Update GUI directly for initial state
        app.update_stats(initial_counts)
        # Process pending Tkinter events to ensure initial display is correct
        root.update_idletasks()
        print("Initial counts displayed.")

        # 5. Create Background Task Manager
        task_manager = BackgroundTaskManager(root)

        # 6. Setup Graceful Shutdown
        def on_closing():
            print("\nShutdown requested...")
            # Optional: Confirmation dialog
            # if messagebox.askokcancel("Quit", "Stop monitoring and recording?", parent=root):
            print("Proceeding with shutdown sequence...")
            if task_manager:
                task_manager.stop_all() # Stop background threads first
            if root:
                print("Destroying Tkinter root window...")
                root.destroy() # Close the Tkinter window
                print("Tkinter root window destroyed.")
            print("Application shutdown complete.")
            # else:
            #     print("Shutdown cancelled by user.")

        root.protocol("WM_DELETE_WINDOW", on_closing) # Hook the window close button

        # 7. Start Background Tasks
        print("Starting background tasks...")
        task_manager.start_all()

        # 8. Start Tkinter Main Loop
        print("========================================")
        print(" Application setup complete. Running...")
        print(" Press Ctrl+Alt (<1s) to log results.")
        print(" Close the Status window to quit.")
        print("========================================")
        root.mainloop() # Blocks here until root window is closed

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Initiating graceful shutdown...")
        on_closing() # Use the same shutdown sequence

    except Exception as e:
        print("\n" + "="*40)
        print(f" UNEXPECTED ERROR in main execution block: {e}")
        print(" Traceback:")
        traceback.print_exc()
        print("="*40 + "\n")
        # Try to show an error message box
        try:
            if root and root.winfo_exists():
                 messagebox.showerror("Fatal Runtime Error", f"An unexpected error occurred:\n\n{e}\n\nPlease check the console output.\nAttempting shutdown.", parent=root)
            else:
                 # If root window doesn't exist, try a temporary one
                 root_err = tk.Tk()
                 root_err.withdraw()
                 messagebox.showerror("Fatal Runtime Error", f"An unexpected error occurred:\n\n{e}\n\nPlease check the console output.\nAttempting shutdown.", parent=None)
                 root_err.destroy()

        except Exception as e_msg:
             print(f"(Could not display error messagebox: {e_msg})")

        # Attempt graceful shutdown even after unexpected error
        print("Attempting shutdown after error...")
        try:
            if task_manager:
                task_manager.stop_all()
        except Exception as shutdown_e:
            print(f"Error during shutdown after exception: {shutdown_e}")
        finally:
            # Try destroying root window again if it might still exist
            try:
                if root and root.winfo_exists():
                    root.destroy()
            except Exception:
                pass
        print("Shutdown attempted.")

    finally:
        # This block executes whether there was an exception or not (after mainloop/error handling)
        print("Application final cleanup.")
        # Ensure threads are definitely stopped if shutdown wasn't fully clean
        if task_manager and any(t.is_alive() for t in task_manager.threads):
             print("Forcing stop on any remaining threads...")
             task_manager.stop_event.set() # Make sure event is set
             # Give a very short extra time
             time.sleep(0.1)

    print("Application finished.")