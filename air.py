import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
import time
import threading
import json
from collections import deque
import tkinter as tk
from tkinter import filedialog, colorchooser
from PIL import Image, ImageTk
import sys

class AirCanvas:
    def __init__(self):
        # Initialize MediaPipe components with improved settings
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Canvas properties with better memory management
        self.canvas = None
        self.canvas_history = deque(maxlen=20)  # Store multiple history states for undo/redo
        self.history_position = 0
        self.h, self.w = 720, 1280  # Default size, will be updated
        
        # Multi-layer support
        self.layers = []
        self.active_layer = 0
        self.layer_visibility = []
        
        # Drawing properties with expanded options
        self.drawing_mode = "freehand"  # Default mode
        self.shape_mode = None
        self.brush_thickness = 5
        self.current_color = (0, 0, 255)  # Default red
        self.brush_style = "solid"  # solid, dotted, dashed
        self.opacity = 1.0  # Full opacity
        
        # Points tracking with better memory management
        self.start_point = None
        self.end_point = None
        self.drawing_points = deque(maxlen=200)
        
        # Selection and transformation
        self.selection_area = None
        self.selected_content = None
        self.transform_mode = None  # move, resize, rotate
        
        # Extended UI properties with more options
        self.colors = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "orange": (0, 165, 255),
            "purple": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        
        self.tools = {
            "freehand": {"icon": "F", "description": "Freehand Drawing"},
            "line": {"icon": "L", "description": "Line Drawing"},
            "rectangle": {"icon": "R", "description": "Rectangle"},
            "circle": {"icon": "C", "description": "Circle"},
            "square": {"icon": "S", "description": "Square"},
            "triangle": {"icon": "T", "description": "Triangle"},
            "text": {"icon": "A", "description": "Text Tool"},
            "select": {"icon": "S+", "description": "Select Tool"},
            "fill": {"icon": "FL", "description": "Fill Tool"},
            "eraser": {"icon": "E", "description": "Eraser"}
        }
        
        # Brush styles
        self.brush_styles = ["solid", "dotted", "dashed", "airbrush"]
        
        # Status indicators with improved tracking
        self.is_drawing = False
        self.is_selecting = False
        self.selection_cooldown = 0
        self.show_help = True
        self.help_timeout = time.time() + 10  # Show help for 10 seconds initially
        self.show_floating_menu = False
        self.floating_menu_position = (0, 0)
        
        # Text tool variables
        self.text_input = ""
        self.text_position = None
        self.text_size = 1
        self.text_color = (255, 255, 255)
        self.text_input_active = False
        
        # Performance tracking
        self.fps = 0
        self.last_frame_time = 0
        self.frame_times = deque(maxlen=30)
        
        # Settings
        self.settings = {
            "accessibility": {
                "color_blind_mode": False,
                "high_contrast": False,
                "gesture_sensitivity": 5,  # 1-10 scale
                "audio_feedback": False
            },
            "performance": {
                "hand_detection_quality": "balanced",  # low, balanced, high
                "max_hands": 2,
                "enable_gpu": False
            },
            "ui": {
                "toolbar_position": "left",  # left, right, bottom
                "ui_size": "medium",  # small, medium, large
                "show_tooltips": True
            }
        }
        
        # Initialize directories for saving drawings
        self.setup_directories()
        
        # Initialize threading components for better performance
        self.lock = threading.Lock()
        self.processing_frame = None
        self.processed_results = None
        self.processing_thread_active = False
        self.processing_thread = None
        
        # Voice command support
        self.voice_commands_enabled = False
        
        # Initialize the first layer
        self.add_new_layer()
        
        # Initialize layers after setting canvas dimensions
        self.initialize_layers()

    def setup_directories(self):
        """Create necessary directories for saving drawings and settings"""
        directories = ["saved_drawings", "saved_projects", "exports", "settings"]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Load settings if available
        self.load_settings()

    def load_settings(self):
        """Load user settings from file if available"""
        try:
            if os.path.exists("settings/user_settings.json"):
                with open("settings/user_settings.json", "r") as f:
                    loaded_settings = json.load(f)
                    # Update only existing settings to prevent errors with new settings
                    for category in loaded_settings:
                        if category in self.settings:
                            for setting in loaded_settings[category]:
                                if setting in self.settings[category]:
                                    self.settings[category][setting] = loaded_settings[category][setting]
        except Exception as e:
            print(f"Error loading settings: {e}")

    def save_settings(self):
        """Save current settings to file"""
        try:
            with open("settings/user_settings.json", "w") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def add_new_layer(self):
        """Add a new transparent layer to the canvas"""
        if self.canvas is not None:
            # Create a new transparent layer
            new_layer = np.zeros((self.h, self.w, 4), dtype=np.uint8)
            self.layers.append(new_layer)
            self.layer_visibility.append(True)
            self.active_layer = len(self.layers) - 1
        else:
            # First initialization will happen when we get frame dimensions
            pass

    def initialize_layers(self):
        """Initialize the layer system once we know dimensions"""
        # Clear any existing layers
        self.layers = []
        self.layer_visibility = []
        
        # Create the background layer (white)
        background = np.ones((self.h, self.w, 4), dtype=np.uint8) * 255
        background[:,:,3] = 255  # Fully opaque
        self.layers.append(background)
        self.layer_visibility.append(True)
        
        # Create the first drawing layer
        drawing_layer = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        self.layers.append(drawing_layer)
        self.layer_visibility.append(True)
        
        # Set active layer to the drawing layer
        self.active_layer = 1
        
        # Initialize the canvas for display
        self.update_composite_canvas()

    def update_composite_canvas(self):
        """Combine all visible layers into one display canvas"""
        # Start with a transparent canvas
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # Combine visible layers
        for i, layer in enumerate(self.layers):
            if self.layer_visibility[i]:
                # For each pixel where the layer is not transparent
                mask = layer[:,:,3] > 0
                if np.any(mask):
                    # Only copy RGB channels
                    alpha_factor = layer[:,:,3:4].astype(np.float32) / 255.0
                    self.canvas[mask] = (
                        (1 - alpha_factor[mask]) * self.canvas[mask].astype(np.float32) + 
                        alpha_factor[mask] * layer[mask, :3].astype(np.float32)
                    ).astype(np.uint8)
        
        # Save to history if this is a meaningful update
        self.save_to_history()

    def save_to_history(self):
        """Save current state to history for undo/redo"""
        # Only save if we've made changes and we're not in the middle of an operation
        if not self.is_drawing and not self.is_selecting:
            # Create a deep copy of all layers
            history_entry = []
            for layer in self.layers:
                history_entry.append(layer.copy())
            
            # If we're not at the end of the history, trim it
            if self.history_position < len(self.canvas_history):
                # Remove everything after the current position
                while self.history_position < len(self.canvas_history):
                    self.canvas_history.pop()
            
            # Add the new state
            self.canvas_history.append(history_entry)
            self.history_position = len(self.canvas_history) - 1

    def undo(self):
        """Undo the last action"""
        if self.history_position > 0:
            self.history_position -= 1
            # Restore from history
            self.layers = []
            for layer in self.canvas_history[self.history_position]:
                self.layers.append(layer.copy())
            # Update layer visibility if needed
            if len(self.layer_visibility) != len(self.layers):
                self.layer_visibility = [True] * len(self.layers)
            self.update_composite_canvas()

    def redo(self):
        """Redo the previously undone action"""
        if self.history_position < len(self.canvas_history) - 1:
            self.history_position += 1
            # Restore from history
            self.layers = []
            for layer in self.canvas_history[self.history_position]:
                self.layers.append(layer.copy())
            # Update layer visibility if needed
            if len(self.layer_visibility) != len(self.layers):
                self.layer_visibility = [True] * len(self.layers)
            self.update_composite_canvas()

    def detect_hand_landmarks(self, image, multi_hand_landmarks):
        """Improved hand landmark detection with support for multiple hands"""
        finger_tips = []
        hand_gestures = []
        
        if not multi_hand_landmarks:
            return finger_tips, hand_gestures
        
        # Process each detected hand
        for hand_idx, handLms in enumerate(multi_hand_landmarks):
            hand_info = {}
            
            # Get key finger landmarks
            hand_info["index"] = (
                int(handLms.landmark[8].x * self.w),
                int(handLms.landmark[8].y * self.h)
            )
            hand_info["thumb"] = (
                int(handLms.landmark[4].x * self.w),
                int(handLms.landmark[4].y * self.h)
            )
            hand_info["middle"] = (
                int(handLms.landmark[12].x * self.w),
                int(handLms.landmark[12].y * self.h)
            )
            hand_info["ring"] = (
                int(handLms.landmark[16].x * self.w),
                int(handLms.landmark[16].y * self.h)
            )
            hand_info["pinky"] = (
                int(handLms.landmark[20].x * self.w),
                int(handLms.landmark[20].y * self.h)
            )
            hand_info["wrist"] = (
                int(handLms.landmark[0].x * self.w),
                int(handLms.landmark[0].y * self.h)
            )
            
            # Determine handedness (left/right)
            # This is a simplistic approach - mediapipe actually provides handedness
            hand_info["handedness"] = "right" if hand_idx == 0 else "left"
            
            # Visualize the index finger tip
            cv2.circle(image, hand_info["index"], 8, (255, 0, 255), cv2.FILLED)
            
            # Calculate distances for gesture detection
            thumb_index_distance = np.sqrt(
                (hand_info["thumb"][0] - hand_info["index"][0]) ** 2 +
                (hand_info["thumb"][1] - hand_info["index"][1]) ** 2
            )
            
            index_middle_distance = np.sqrt(
                (hand_info["index"][0] - hand_info["middle"][0]) ** 2 +
                (hand_info["index"][1] - hand_info["middle"][1]) ** 2
            )
            
            # Detect gestures
            gesture = {}
            
            # Pinch gesture (thumb and index close)
            if thumb_index_distance < 40:
                gesture["pinch"] = True
                # Draw a line between thumb and index to show the pinch
                cv2.line(image, hand_info["thumb"], hand_info["index"], (0, 255, 0), 2)
            else:
                gesture["pinch"] = False
            
            # Open palm gesture (all fingers extended)
            if (thumb_index_distance > 40 and 
                index_middle_distance > 40):
                gesture["open_palm"] = True
            else:
                gesture["open_palm"] = False
            
            # Point gesture (index extended, others curled)
            if (hand_info["index"][1] < hand_info["middle"][1] - 20 and
                hand_info["index"][1] < hand_info["ring"][1] - 20 and
                hand_info["index"][1] < hand_info["pinky"][1] - 20):
                gesture["point"] = True
            else:
                gesture["point"] = False
                
            # Add to our result lists
            finger_tips.append(hand_info)
            hand_gestures.append(gesture)
            
        return finger_tips, hand_gestures

    def draw_ui_elements(self, frame):
        """Draw enhanced UI elements with accessibility options"""
        ui_scale = 1.0
        if self.settings["ui"]["ui_size"] == "small":
            ui_scale = 0.8
        elif self.settings["ui"]["ui_size"] == "large":
            ui_scale = 1.2
            
        # Apply high contrast if enabled
        bg_color = (50, 50, 50)
        text_color = (255, 255, 255)
        if self.settings["accessibility"]["high_contrast"]:
            bg_color = (0, 0, 0)
            text_color = (255, 255, 0)
        
        # Draw top bar with info and color palette
        cv2.rectangle(frame, (10, 10), (self.w - 10, 60), bg_color, cv2.FILLED)
        
        # Draw FPS counter
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (self.w - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5 * ui_scale, text_color, 1)
        
        # Draw active layer indicator
        layer_text = f"Layer: {self.active_layer + 1}/{len(self.layers)}"
        cv2.putText(frame, layer_text, (self.w - 220, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5 * ui_scale, text_color, 1)
        
        # Draw color palette
        x_offset = 20
        for color_name, color_value in self.colors.items():
            # Apply color blind mode if enabled
            display_color = color_value
            if self.settings["accessibility"]["color_blind_mode"]:
                # Simple conversion to more distinguishable colors
                if color_name == "red":
                    display_color = (0, 0, 150)  # Darker red
                elif color_name == "green":
                    display_color = (0, 150, 150)  # Teal instead of green
                    
            cv2.circle(frame, (x_offset, 35), int(15 * ui_scale), display_color, cv2.FILLED)
            if self.current_color == color_value:
                cv2.circle(frame, (x_offset, 35), int(18 * ui_scale), text_color, 2)
            
            # Add color name text for high contrast mode
            if self.settings["accessibility"]["high_contrast"]:
                cv2.putText(frame, color_name[0].upper(), (x_offset-5, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ui_scale, (0, 0, 0), 1)
                
            x_offset += int(40 * ui_scale)
        
        # Custom color picker button
        cv2.circle(frame, (x_offset, 35), int(15 * ui_scale), (127, 127, 127), cv2.FILLED)
        cv2.putText(frame, "+", (x_offset-5, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5 * ui_scale, text_color, 1)
        x_offset += int(40 * ui_scale)
        
        # Draw toolbar based on position setting
        if self.settings["ui"]["toolbar_position"] == "left":
            self.draw_left_toolbar(frame, bg_color, text_color, ui_scale)
        elif self.settings["ui"]["toolbar_position"] == "right":
            self.draw_right_toolbar(frame, bg_color, text_color, ui_scale)
        else:  # bottom
            self.draw_bottom_toolbar(frame, bg_color, text_color, ui_scale)
            
        # Draw layer controls
        self.draw_layer_controls(frame, bg_color, text_color, ui_scale)
        
        # Draw brush property controls
        self.draw_brush_controls(frame, bg_color, text_color, ui_scale)
            
        # Draw bottom controls
        self.draw_bottom_controls(frame, bg_color, text_color, ui_scale)
        
        # Show help overlay if enabled
        if self.show_help and time.time() < self.help_timeout:
            self.draw_help_overlay(frame, text_color)
            
        # Draw floating menu if enabled
        if self.show_floating_menu:
            self.draw_floating_menu(frame, bg_color, text_color, ui_scale)
            
        # Show current mode and status
        status_text = f"Mode: {self.drawing_mode.upper()}"
        cv2.putText(frame, status_text, (self.w - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7 * ui_scale, text_color, 1)
                   
        # If text input is active, draw the text input interface
        if self.text_input_active:
            self.draw_text_input_interface(frame, bg_color, text_color, ui_scale)

    def draw_left_toolbar(self, frame, bg_color, text_color, ui_scale):
        """Draw toolbar on the left side"""
        # Tool selection
        y_offset = 100
        for tool_name, tool_info in self.tools.items():
            cv2.rectangle(frame, 
                         (10, int(y_offset - 20 * ui_scale)), 
                         (int(60 * ui_scale), int(y_offset + 20 * ui_scale)), 
                         bg_color, cv2.FILLED)
            cv2.putText(frame, tool_info["icon"], 
                       (int(25 * ui_scale), int(y_offset + 5 * ui_scale)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8 * ui_scale, text_color, 2)
            
            if self.drawing_mode == tool_name:
                cv2.rectangle(frame, 
                             (10, int(y_offset - 20 * ui_scale)), 
                             (int(60 * ui_scale), int(y_offset + 20 * ui_scale)), 
                             (0, 255, 0), 2)
                             
            # Show tooltip if enabled
            if self.settings["ui"]["show_tooltips"]:
                cv2.putText(frame, tool_info["description"], 
                           (int(65 * ui_scale), int(y_offset * ui_scale)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ui_scale, text_color, 1)
                
            y_offset += int(50 * ui_scale)

    def draw_right_toolbar(self, frame, bg_color, text_color, ui_scale):
        """Draw toolbar on the right side"""
        # Similar to left toolbar but positioned on right
        y_offset = 100
        for tool_name, tool_info in self.tools.items():
            right_edge = self.w - 10
            left_edge = right_edge - int(50 * ui_scale)
            
            cv2.rectangle(frame, 
                         (left_edge, int(y_offset - 20 * ui_scale)), 
                         (right_edge, int(y_offset + 20 * ui_scale)), 
                         bg_color, cv2.FILLED)
            cv2.putText(frame, tool_info["icon"], 
                       (left_edge + int(15 * ui_scale), int(y_offset + 5 * ui_scale)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8 * ui_scale, text_color, 2)
            
            if self.drawing_mode == tool_name:
                cv2.rectangle(frame, 
                             (left_edge, int(y_offset - 20 * ui_scale)), 
                             (right_edge, int(y_offset + 20 * ui_scale)), 
                             (0, 255, 0), 2)
                             
            # Show tooltip if enabled
            if self.settings["ui"]["show_tooltips"]:
                tooltip_width = len(tool_info["description"]) * 8
                cv2.putText(frame, tool_info["description"], 
                           (left_edge - tooltip_width, int(y_offset * ui_scale)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ui_scale, text_color, 1)
                
            y_offset += int(50 * ui_scale)

    def draw_bottom_toolbar(self, frame, bg_color, text_color, ui_scale):
        """Draw toolbar at the bottom"""
        # Tool selection in a horizontal layout
        x_offset = 70
        bottom_y = self.h - 40
        
        for tool_name, tool_info in self.tools.items():
            cv2.rectangle(frame, 
                         (x_offset, int(bottom_y - 20 * ui_scale)), 
                         (int(x_offset + 40 * ui_scale), int(bottom_y + 20 * ui_scale)), 
                         bg_color, cv2.FILLED)
            cv2.putText(frame, tool_info["icon"], 
                       (int(x_offset + 10 * ui_scale), int(bottom_y + 5 * ui_scale)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8 * ui_scale, text_color, 2)
            
            if self.drawing_mode == tool_name:
                cv2.rectangle(frame, 
                             (x_offset, int(bottom_y - 20 * ui_scale)), 
                             (int(x_offset + 40 * ui_scale), int(bottom_y + 20 * ui_scale)), 
                             (0, 255, 0), 2)
                
            # Show tooltip above if enabled
            if self.settings["ui"]["show_tooltips"]:
                cv2.putText(frame, tool_info["description"], 
                           (x_offset, int(bottom_y - 25 * ui_scale)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ui_scale, text_color, 1)
                
            x_offset += int(50 * ui_scale)

    def draw_layer_controls(self, frame, bg_color, text_color, ui_scale):
        """Draw layer management controls"""
        # Position on the right side
        right_edge = self.w - 70
        y_offset = 100
        
        # Layer control title
        cv2.putText(frame, "LAYERS", 
                   (right_edge, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui_scale, text_color, 1)
        y_offset += 30
        
        # Layer list
        for i, _ in enumerate(self.layers):
            # Layer background
            layer_color = (0, 200, 0) if i == self.active_layer else bg_color
            cv2.rectangle(frame, 
                         (right_edge - 100, y_offset - 15), 
                         (right_edge, y_offset + 15), 
                         layer_color, cv2.FILLED)
            
            # Layer name/number
            cv2.putText(frame, f"Layer {i+1}", 
                       (right_edge - 90, y_offset + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ui_scale, text_color, 1)
            
            # Visibility toggle
            vis_color = (0, 255, 0) if self.layer_visibility[i] else (150, 150, 150)
            cv2.circle(frame, (right_edge - 20, y_offset), 7, vis_color, cv2.FILLED)
            
            y_offset += 35
        
        # Add layer button
        cv2.rectangle(frame, 
                     (right_edge - 100, y_offset - 15), 
                     (right_edge, y_offset + 15), 
                     bg_color, cv2.FILLED)
        cv2.putText(frame, "+ Add Layer", 
                   (right_edge - 90, y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ui_scale, text_color, 1)

    def draw_brush_controls(self, frame, bg_color, text_color, ui_scale):
        """Draw brush property controls"""
        # Position near the bottom left
        x_offset = 20
        y_offset = self.h - 120
        
        # Brush size
        cv2.rectangle(frame, (x_offset, y_offset - 50), (x_offset + 100, y_offset), bg_color, cv2.FILLED)
        cv2.putText(frame, "Size: " + str(self.brush_thickness), 
                   (x_offset + 10, y_offset - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ui_scale, text_color, 1)
        
        # Size slider
        slider_x = x_offset + 10
        slider_width = 80
        slider_pos = slider_x + (self.brush_thickness / 20.0) * slider_width
        
        cv2.line(frame, (slider_x, y_offset - 30), (slider_x + slider_width, y_offset - 30), 
                (150, 150, 150), 2)
        cv2.circle(frame, (int(slider_pos), y_offset - 30), 5, (0, 255, 0), cv2.FILLED)
        
        # Opacity
        cv2.rectangle(frame, (x_offset + 120, y_offset - 50), (x_offset + 220, y_offset), bg_color, cv2.FILLED)
        cv2.putText(frame, "Opacity: " + str(int(self.opacity * 100)) + "%", 
                   (x_offset + 130, y_offset - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ui_scale, text_color, 1)
        
        # Opacity slider
        slider_x = x_offset + 130
        slider_pos = slider_x + self.opacity * slider_width
        
        cv2.line(frame, (slider_x, y_offset - 30), (slider_x + slider_width, y_offset - 30), 
                (150, 150, 150), 2)
        cv2.circle(frame, (int(slider_pos), y_offset - 30), 5, (0, 255, 0), cv2.FILLED)
        
        # Brush style
        cv2.rectangle(frame, (x_offset + 240, y_offset - 50), (x_offset + 340, y_offset), bg_color, cv2.FILLED)
        cv2.putText(frame, "Style: " + self.brush_style, 
                   (x_offset + 250, y_offset - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ui_scale, text_color, 1)

    def draw_bottom_controls(self, frame, bg_color, text_color, ui_scale):
        """Draw bottom control buttons"""
        bottom_y = self.h - 50
        
        # Clear button
        cv2.rectangle(frame, (20, bottom_y - 20), (80, bottom_y + 20), bg_color, cv2.FILLED)
        cv2.putText(frame, "Clear", (25, bottom_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6 * ui_scale, text_color, 1)
        
        # Undo button
        cv2.rectangle(frame, (100, bottom_y - 20), (160, bottom_y + 20), bg_color, cv2.FILLED)
        cv2.putText(frame, "Undo", (105, bottom_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6 * ui_scale, text_color, 1)
        
        # Redo button
        cv2.rectangle(frame, (180, bottom_y - 20), (240, bottom_y + 20), bg_color, cv2.FILLED)
        cv2.putText(frame, "Redo", (185, bottom_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6 * ui_scale, text_color, 1)
        
        # Save button
        cv2.rectangle(frame, (260, bottom_y - 20), (320, bottom_y + 20), bg_color, cv2.FILLED)  # Save button
        cv2.rectangle(frame, (260, bottom_y - 20), (320, bottom_y + 20), bg_color, cv2.FILLED)
        cv2.putText(frame, "Save", (265, bottom_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6 * ui_scale, text_color, 1)
        
        # Settings button
        cv2.rectangle(frame, (340, bottom_y - 20), (400, bottom_y + 20), bg_color, cv2.FILLED)
        cv2.putText(frame, "Settings", (345, bottom_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6 * ui_scale, text_color, 1)
        
        # Help button
        cv2.rectangle(frame, (self.w - 100, bottom_y - 20), (self.w - 20, bottom_y + 20), bg_color, cv2.FILLED)
        cv2.putText(frame, "Help", (self.w - 90, bottom_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6 * ui_scale, text_color, 1)

    def draw_help_overlay(self, frame, text_color):
        """Draw help overlay with instructions"""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 0), cv2.FILLED)
        frame_with_overlay = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Title
        cv2.putText(frame_with_overlay, "AIR CANVAS - QUICK HELP", 
                   (self.w // 2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # Gesture instructions
        instructions = [
            "GESTURES:",
            "• Point with index finger to move cursor",
            "• Pinch (thumb + index) to draw or select",
            "• Open palm to access floating menu",
            "",
            "KEYBOARD SHORTCUTS:",
            "• Ctrl+Z: Undo",
            "• Ctrl+Y: Redo",
            "• Ctrl+S: Save",
            "• Esc: Cancel current operation",
            "• H: Toggle help",
            "",
            "Press any key to close this help"
        ]
        
        y_pos = 100
        for line in instructions:
            cv2.putText(frame_with_overlay, line, 
                       (self.w // 2 - 250, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, text_color, 1)
            y_pos += 30
            
        # Copy back to the frame
        frame[:] = frame_with_overlay[:]

    def draw_floating_menu(self, frame, bg_color, text_color, ui_scale):
        """Draw context-sensitive floating menu near cursor"""
        x, y = self.floating_menu_position
        
        # Menu background
        menu_width = int(150 * ui_scale)
        menu_height = int(220 * ui_scale)
        cv2.rectangle(frame, 
                     (x, y), 
                     (x + menu_width, y + menu_height), 
                     bg_color, cv2.FILLED)
        cv2.rectangle(frame, 
                     (x, y), 
                     (x + menu_width, y + menu_height), 
                     (255, 255, 255), 1)
        
        # Menu title
        cv2.putText(frame, "Quick Actions", 
                   (x + 10, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui_scale, text_color, 1)
        
        # Menu items depend on current mode
        menu_items = []
        if self.drawing_mode == "freehand":
            menu_items = [
                ("Change Color", 50),
                ("Change Size", 90),
                ("Change Style", 130),
                ("Switch Tool", 170)
            ]
        elif self.drawing_mode in ["rectangle", "circle", "line"]:
            menu_items = [
                ("Change Color", 50),
                ("Fill Shape", 90),
                ("Outline Only", 130),
                ("Switch Tool", 170)
            ]
        elif self.drawing_mode == "select":
            menu_items = [
                ("Cut", 50),
                ("Copy", 90),
                ("Paste", 130),
                ("Delete", 170)
            ]
        else:
            menu_items = [
                ("Change Tool", 50),
                ("Clear Canvas", 90),
                ("Save Drawing", 130),
                ("Settings", 170)
            ]
        
        # Draw menu items
        for item_text, item_y in menu_items:
            cv2.putText(frame, item_text, 
                       (x + 20, y + item_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ui_scale, text_color, 1)

    def draw_text_input_interface(self, frame, bg_color, text_color, ui_scale):
        """Draw text input interface for the text tool"""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 0), cv2.FILLED)
        frame_with_overlay = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Text input box
        box_width = 400
        box_height = 200
        box_x = self.w // 2 - box_width // 2
        box_y = self.h // 2 - box_height // 2
        
        cv2.rectangle(frame_with_overlay, 
                     (box_x, box_y), 
                     (box_x + box_width, box_y + box_height), 
                     bg_color, cv2.FILLED)
        cv2.rectangle(frame_with_overlay, 
                     (box_x, box_y), 
                     (box_x + box_width, box_y + box_height), 
                     (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame_with_overlay, "Text Input", 
                   (box_x + 10, box_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8 * ui_scale, text_color, 1)
        
        # Text entry area
        cv2.rectangle(frame_with_overlay, 
                     (box_x + 10, box_y + 50), 
                     (box_x + box_width - 10, box_y + 100), 
                     (255, 255, 255), cv2.FILLED)
        
        # Show current text
        cv2.putText(frame_with_overlay, self.text_input if self.text_input else "Type here...", 
                   (box_x + 15, box_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7 * ui_scale, 
                   (0, 0, 0) if self.text_input else (150, 150, 150), 1)
        
        # Buttons
        # OK Button
        cv2.rectangle(frame_with_overlay, 
                     (box_x + 10, box_y + 120), 
                     (box_x + 100, box_y + 160), 
                     (0, 200, 0), cv2.FILLED)
        cv2.putText(frame_with_overlay, "OK", 
                   (box_x + 40, box_y + 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7 * ui_scale, (255, 255, 255), 1)
        
        # Cancel Button
        cv2.rectangle(frame_with_overlay, 
                     (box_x + 120, box_y + 120), 
                     (box_x + 210, box_y + 160), 
                     (0, 0, 200), cv2.FILLED)
        cv2.putText(frame_with_overlay, "Cancel", 
                   (box_x + 135, box_y + 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7 * ui_scale, (255, 255, 255), 1)
        
        # Copy back to the frame
        frame[:] = frame_with_overlay[:]

    def process_hand_gestures(self, finger_tips, hand_gestures):
        """Process detected hand gestures and update drawing state"""
        # Only proceed if we have detected hands
        if not finger_tips or not hand_gestures:
            # Reset drawing state if no hands are detected
            self.is_drawing = False
            return
        
        # Get primary hand (we'll use the first detected hand as primary)
        primary_tip = finger_tips[0]
        primary_gesture = hand_gestures[0]
        
        # Track index finger position for cursor
        index_position = primary_tip["index"]
        
        # Check for UI interactions first (priority over drawing)
        if self.check_ui_interaction(index_position, primary_gesture):
            return
        
        # Handle gestures based on current mode
        if self.drawing_mode == "freehand":
            self.handle_freehand_drawing(index_position, primary_gesture)
        elif self.drawing_mode == "line":
            self.handle_line_drawing(index_position, primary_gesture)
        elif self.drawing_mode in ["rectangle", "circle", "square", "triangle"]:
            self.handle_shape_drawing(index_position, primary_gesture)
        elif self.drawing_mode == "text":
            self.handle_text_tool(index_position, primary_gesture)
        elif self.drawing_mode == "select":
            self.handle_selection_tool(index_position, primary_gesture)
        elif self.drawing_mode == "eraser":
            self.handle_eraser_tool(index_position, primary_gesture)
        elif self.drawing_mode == "fill":
            self.handle_fill_tool(index_position, primary_gesture)

    def check_ui_interaction(self, index_position, gesture):
        """Check for UI element interactions and handle them
        Returns True if an interaction was handled, False otherwise"""
        x, y = index_position
        
        # Check for toolbar interactions
        if self.settings["ui"]["toolbar_position"] == "left" and x < 60:
            y_offset = 100
            for tool_name in self.tools:
                if y_offset - 20 <= y <= y_offset + 20:
                    if gesture["pinch"]:
                        self.drawing_mode = tool_name
                        return True
                y_offset += 50
                
        # Check bottom controls
        bottom_y = self.h - 50
        if bottom_y - 20 <= y <= bottom_y + 20:
            # Clear button
            if 20 <= x <= 80 and gesture["pinch"]:
                self.clear_canvas()
                return True
            # Undo button
            elif 100 <= x <= 160 and gesture["pinch"]:
                self.undo()
                return True
            # Redo button
            elif 180 <= x <= 240 and gesture["pinch"]:
                self.redo()
                return True
            # Save button
            elif 260 <= x <= 320 and gesture["pinch"]:
                self.save_drawing()
                return True
                
        # Color palette
        y_top_bar = 35
        if abs(y - y_top_bar) < 20:
            x_offset = 20
            for color_name, color_value in self.colors.items():
                if abs(x - x_offset) < 15 and gesture["pinch"]:
                    self.current_color = color_value
                    return True
                x_offset += 40
                
        # Check layer controls
        right_edge = self.w - 70
        y_offset = 130
        for i in range(len(self.layers)):
            if right_edge - 100 <= x <= right_edge and y_offset - 15 <= y <= y_offset + 15:
                if gesture["pinch"]:
                    self.active_layer = i
                    return True
            
            # Visibility toggle
            if abs(x - (right_edge - 20)) < 10 and abs(y - y_offset) < 10:
                if gesture["pinch"]:
                    self.layer_visibility[i] = not self.layer_visibility[i]
                    self.update_composite_canvas()
                    return True
                    
            y_offset += 35
            
        # Check for add layer button
        if right_edge - 100 <= x <= right_edge and y_offset - 15 <= y <= y_offset + 15:
            if gesture["pinch"]:
                self.add_new_layer()
                return True
                
        # Check for open palm gesture to show floating menu
        if gesture["open_palm"] and not self.show_floating_menu:
            self.show_floating_menu = True
            self.floating_menu_position = (x, y)
            return True
        
        return False

    def handle_freehand_drawing(self, index_position, gesture):
        """Handle freehand drawing mode"""
        if gesture["pinch"]:
            # Start or continue drawing
            if not self.is_drawing:
                self.is_drawing = True
                self.drawing_points.clear()
                
            # Add point to drawing path
            self.drawing_points.append(index_position)
            
            # Draw on active layer
            if len(self.drawing_points) >= 2:
                # Get current layer
                layer = self.layers[self.active_layer]
                
                # Get last two points
                p1 = self.drawing_points[-2]
                p2 = self.drawing_points[-1]
                
                # Create a separate layer for drawing to handle transparency
                drawing_overlay = np.zeros_like(layer)
                
                # Apply different brush styles
                if self.brush_style == "solid":
                    cv2.line(drawing_overlay, p1, p2, 
                            (*self.current_color, int(255 * self.opacity)), 
                            self.brush_thickness)
                elif self.brush_style == "dotted":
                    # Draw dotted line
                    dist = np.sqrt((p2[0] - p1[0])*2 + (p2[1] - p1[1])*2)
                    if dist > 0:
                        dots = max(2, int(dist / 10))
                        for i in range(dots):
                            t = i / (dots - 1)
                            x = int((1 - t) * p1[0] + t * p2[0])
                            y = int((1 - t) * p1[1] + t * p2[1])
                            cv2.circle(drawing_overlay, (x, y), 
                                      self.brush_thickness // 2, 
                                      (*self.current_color, int(255 * self.opacity)), 
                                      cv2.FILLED)
                elif self.brush_style == "dashed":
                    # Draw dashed line
                    dist = np.sqrt((p2[0] - p1[0])*2 + (p2[1] - p1[1])*2)
                    if dist > 0:
                        segments = max(2, int(dist / 20))
                        for i in range(segments):
                            if i % 2 == 0:  # Only draw even segments
                                t1 = i / segments
                                t2 = (i + 0.5) / segments
                                x1 = int((1 - t1) * p1[0] + t1 * p2[0])
                                y1 = int((1 - t1) * p1[1] + t1 * p2[1])
                                x2 = int((1 - t2) * p1[0] + t2 * p2[0])
                                y2 = int((1 - t2) * p1[1] + t2 * p2[1])
                                cv2.line(drawing_overlay, (x1, y1), (x2, y2), 
                                        (*self.current_color, int(255 * self.opacity)), 
                                        self.brush_thickness)
                elif self.brush_style == "airbrush":
                    # Airbrush effect
                    cv2.line(drawing_overlay, p1, p2, 
                            (*self.current_color, int(100 * self.opacity)), 
                            self.brush_thickness * 2)
                    cv2.line(drawing_overlay, p1, p2, 
                            (*self.current_color, int(200 * self.opacity)), 
                            self.brush_thickness)
                
                # Blend with layer
                mask = drawing_overlay[:,:,3] > 0
                if np.any(mask):
                    layer[mask] = drawing_overlay[mask]
                
                # Update composite canvas
                self.update_composite_canvas()
        else:
            # End drawing if the pinch is released
            if self.is_drawing:
                self.is_drawing = False
                self.save_to_history()

    def handle_line_drawing(self, index_position, gesture):
        """Handle line drawing mode"""
        if gesture["pinch"]:
            # Start line or update end point
            if self.start_point is None:
                self.start_point = index_position
            else:
                self.end_point = index_position
                
            # Create a preview copy
            preview = self.canvas.copy()
            
            # Draw line preview
            if self.start_point and self.end_point:
                cv2.line(preview, self.start_point, self.end_point, 
                        self.current_color, self.brush_thickness)
                
                # Replace canvas with preview for display
                self.canvas = preview
        else:
            # Finalize line when pinch is released
            if self.start_point and self.end_point:
                # Get current layer
                layer = self.layers[self.active_layer]
                
                # Create a separate layer for drawing to handle transparency
                drawing_overlay = np.zeros_like(layer)
                
                # Draw final line
                cv2.line(drawing_overlay, self.start_point, self.end_point, 
                        (*self.current_color, int(255 * self.opacity)), 
                        self.brush_thickness)
                
                # Blend with layer
                mask = drawing_overlay[:,:,3] > 0
                if np.any(mask):
                    layer[mask] = drawing_overlay[mask]
                
                # Update composite canvas
                self.update_composite_canvas()
                
                # Reset points
                self.start_point = None
                self.end_point = None
                
                # Save to history
                self.save_to_history()

    def handle_shape_drawing(self, index_position, gesture):
        """Handle shape drawing mode (rectangle, circle, square, triangle)"""
        if gesture["pinch"]:
            # Start shape or update end point
            if self.start_point is None:
                self.start_point = index_position
            else:
                self.end_point = index_position
                
            # Create a preview copy
            preview = self.canvas.copy()
            
            # Draw shape preview
            if self.start_point and self.end_point:
                if self.drawing_mode == "rectangle":
                    cv2.rectangle(preview, self.start_point, self.end_point, 
                                 self.current_color, self.brush_thickness)
                elif self.drawing_mode == "square":
                    # Calculate square dimensions
                    side_length = max(abs(self.end_point[0] - self.start_point[0]), 
                                      abs(self.end_point[1] - self.start_point[1]))
                    end_x = self.start_point[0] + side_length * (1 if self.end_point[0] > self.start_point[0] else -1)
                    end_y = self.start_point[1] + side_length * (1 if self.end_point[1] > self.start_point[1] else -1)
                    cv2.rectangle(preview, self.start_point, (end_x, end_y), 
                                 self.current_color, self.brush_thickness)
                elif self.drawing_mode == "circle":
                    # Calculate radius
                    radius = int(np.sqrt((self.end_point[0] - self.start_point[0]) ** 2 + 
                                       (self.end_point[1] - self.start_point[1]) ** 2))
                    cv2.circle(preview, self.start_point, radius, 
                              self.current_color, self.brush_thickness)
                elif self.drawing_mode == "triangle":
                    # Draw triangle using start point as apex
                    # and end point to determine the base
                    width = self.end_point[0] - self.start_point[0]
                    height = self.end_point[1] - self.start_point[1]
                    
                    pt1 = self.start_point
                    pt2 = (self.start_point[0] - width // 2, self.start_point[1] + height)
                    pt3 = (self.start_point[0] + width // 2, self.start_point[1] + height)
                    
                    triangle_pts = np.array([pt1, pt2, pt3])
                    cv2.polylines(preview, [triangle_pts], True, 
                                 self.current_color, self.brush_thickness)
                
                # Replace canvas with preview for display
                self.canvas = preview
        else:
            # Finalize shape when pinch is released
            if self.start_point and self.end_point:
                # Get current layer
                layer = self.layers[self.active_layer]
                
                # Create a separate layer for drawing to handle transparency
                drawing_overlay = np.zeros_like(layer)
                
                # Draw final shape
                if self.drawing_mode == "rectangle":
                    cv2.rectangle(drawing_overlay, self.start_point, self.end_point, 
                                 (*self.current_color, int(255 * self.opacity)), 
                                 self.brush_thickness)
                elif self.drawing_mode == "square":
                    # Calculate square dimensions
                    side_length = max(abs(self.end_point[0] - self.start_point[0]), 
                                      abs(self.end_point[1] - self.start_point[1]))
                    end_x = self.start_point[0] + side_length * (1 if self.end_point[0] > self.start_point[0] else -1)
                    end_y = self.start_point[1] + side_length * (1 if self.end_point[1] > self.start_point[1] else -1)
                    cv2.rectangle(drawing_overlay, self.start_point, (end_x, end_y), 
                                 (*self.current_color, int(255 * self.opacity)), 
                                 self.brush_thickness)
                elif self.drawing_mode == "circle":
                    # Calculate radius
                    radius = int(np.sqrt((self.end_point[0] - self.start_point[0]) ** 2 + 
                                       (self.end_point[1] - self.start_point[1]) ** 2))
                    cv2.circle(drawing_overlay, self.start_point, radius, 
                              (*self.current_color, int(255 * self.opacity)), 
                              self.brush_thickness)
                elif self.drawing_mode == "triangle":
                    # Draw triangle using start point as apex
                    # and end point to determine the base
                    width = self.end_point[0] - self.start_point[0]
                    height = self.end_point[1] - self.start_point[1]
                    
                    pt1 = self.start_point
                    pt2 = (self.start_point[0] - width // 2, self.start_point[1] + height)
                    pt3 = (self.start_point[0] + width // 2, self.start_point[1] + height)
                    
                    triangle_pts = np.array([pt1, pt2, pt3])
                    cv2.drawContours(drawing_overlay, [triangle_pts], 0, 
                                    (*self.current_color, int(255 * self.opacity)), 
                                    self.brush_thickness)
                
                # Blend with layer
                mask = drawing_overlay[:,:,3] > 0
                if np.any(mask):
                    layer[mask] = drawing_overlay[mask]
                
                # Update composite canvas
                self.update_composite_canvas()
                
                # Reset points
                self.start_point = None
                self.end_point = None
                
                # Save to history
                self.save_to_history()

    def handle_text_tool(self, index_position, gesture):
        """Handle text tool interactions"""
        if gesture["pinch"] and self.selection_cooldown <= 0:
            # Set text position and activate input
            if not self.text_input_active:
                self.text_position = index_position
                self.text_input_active = True
                self.text_input = ""
                # Reset cooldown
                self.selection_cooldown = 10
        
        # Note: Actual text input happens through keyboard events
        # Text drawing occurs when text input is confirmed

    def handle_selection_tool(self, index_position, gesture):
        """Handle selection tool interactions"""
        if gesture["pinch"]:
            # Start selection or update end point
            if not self.is_selecting:
                self.is_selecting = True
                self.start_point = index_position
                self.end_point = None
                self.selected_content = None
            else:
                self.end_point = index_position
                
            # Create a preview copy
            preview = self.canvas.copy()
            
            # Draw selection rectangle
            if self.start_point and self.end_point:
                cv2.rectangle(preview, self.start_point, self.end_point, 
                             (0, 255, 0), 2)
            
            # Replace canvas with preview for display
            self.canvas = preview
        else:
            # Finalize selection when pinch is released
            if self.is_selecting and self.start_point and self.end_point:
                # Calculate selection rectangle
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                
                # Store selection area
                self.selection_area = (x1, y1, x2, y2)
                
                # Extract selected content from active layer
                layer = self.layers[self.active_layer]
                self.selected_content = layer[y1:y2, x1:x2].copy()
                
                # Reset selection state
                self.is_selecting = False
                
                # Reset cooldown
                self.selection_cooldown = 10

    def handle_eraser_tool(self, index_position, gesture):
        """Handle eraser tool interactions"""
        if gesture["pinch"]:
            # Start or continue erasing
            if not self.is_drawing:
                self.is_drawing = True
                self.drawing_points.clear()
                
            # Add point to eraser path
            self.drawing_points.append(index_position)
            
            # Erase on active layer
            if len(self.drawing_points) >= 2:
                # Get current layer
                layer = self.layers[self.active_layer]
                
                # Get last two points
                p1 = self.drawing_points[-2]
                p2 = self.drawing_points[-1]
                
                # Create erase mask
                temp = np.zeros_like(layer)
                cv2.line(temp, p1, p2, (255, 255, 255, 255), self.brush_thickness * 2)
                mask = temp[:,:,0] > 0
                
                # Clear alpha channel where mask is True
                if np.any(mask):
                    layer[mask, 3] = 0
                
                # Update composite canvas
                self.update_composite_canvas()
        else:
            # End erasing if the pinch is released
            if self.is_drawing:
                self.is_drawing = False
                self.save_to_history()

    def handle_fill_tool(self, index_position, gesture):
        """Handle fill tool (bucket) interactions"""
        if gesture["pinch"] and self.selection_cooldown <= 0:
            # Get current layer
            layer = self.layers[self.active_layer]
            
            # Get index position as integers
            x, y = index_position
            x, y = int(x), int(y)
            
            # Check if within bounds
            if 0 <= x < self.w and 0 <= y < self.h:
                # Create a mask for the flood fill
                mask = np.zeros((self.h + 2, self.w + 2), np.uint8)
                
                # Get current color at that position
                target_color = layer[y, x].copy()
                
                # Only fill areas with similar color or transparent areas
                if target_color[3] < 128:  # If mostly transparent
                    # Use flood fill on alpha channel
                    cv2.floodFill(layer, mask, (x, y), 
                                 (*self.current_color, int(255 * self.opacity)), 
                                 (10, 10, 10, 10), (10, 10, 10, 10), 
                                 cv2.FLOODFILL_FIXED_RANGE)
                else:
                    # Use flood fill on color channels
                    cv2.floodFill(layer, mask, (x, y), 
                                 (*self.current_color, int(255 * self.opacity)), 
                                 (30, 30, 30, 0), (30, 30, 30, 0), 
                                 cv2.FLOODFILL_FIXED_RANGE)
                
                # Update composite canvas
                self.update_composite_canvas()
                
                # Save to history
                self.save_to_history()
                
                # Reset cooldown
                self.selection_cooldown = 10

    def clear_canvas(self):
        """Clear the active layer"""
        # Clear the active layer
        self.layers[self.active_layer] = np.zeros_like(self.layers[self.active_layer])
        
        # Update composite canvas
        self.update_composite_canvas()
        
        # Save to history
        self.save_to_history()

    def save_drawing(self):
        """Save the current drawing to file"""
        try:
            # Create a file name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join("saved_drawings", f"drawing_{timestamp}.png")
            
            # Save the composite canvas
            cv2.imwrite(file_path, self.canvas)
            print(f"Drawing saved to {file_path}")
        except Exception as e:
            print(f"Error saving drawing: {e}")

    def run(self):
        """Main loop to run the AirCanvas application"""
        cap = cv2.VideoCapture(0)
        with self.mp_hands.Hands(
            max_num_hands=self.settings["performance"]["max_hands"],
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                
                # Flip the frame horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image and detect hands
                results = hands.process(image)
                
                # Draw the hand annotations on the image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                
                # Detect hand landmarks and gestures
                finger_tips, hand_gestures = self.detect_hand_landmarks(image, results.multi_hand_landmarks)
                
                # Process hand gestures
                self.process_hand_gestures(finger_tips, hand_gestures)
                
                # Draw UI elements
                self.draw_ui_elements(image)
                
                # Calculate FPS
                current_time = time.time()
                if self.last_frame_time != 0:
                    self.frame_times.append(current_time - self.last_frame_time)
                    if len(self.frame_times) > 0:
                        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                self.last_frame_time = current_time
                
                # Display the resulting frame
                cv2.imshow('AirCanvas', image)
                
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    air_canvas = AirCanvas()
    air_canvas.run()