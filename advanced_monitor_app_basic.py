import sys
import os
import serial
import serial.tools.list_ports
import cv2
import numpy as np
import csv
import json
from datetime import datetime
import time
import platform # To check OS for VideoCapture API preference

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QTextEdit, QLineEdit, QLabel, QStatusBar,
    QGridLayout, QGroupBox, QFileDialog, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QCheckBox, QSpinBox, QMessageBox, QSizePolicy,
    #QTextCursor # Added back for QTextEdit cursor manipulation
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QSettings
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor # QPalette removed

# --- Constants ---
APP_NAME = "Advanced Serial & Camera Controller"
ORG_NAME = "MyCompany"
DEFAULT_BAUDRATE = 9600
DEFAULT_LOG_FILE = "command_log.txt"
# DEFAULT_ALERT_SOUND = "alert.wav" # Removed
DEFAULT_VIDEO_DIR = os.path.join(os.path.expanduser("~"), "Videos", "SerialCamRecordings")
DEFAULT_VIDEO_FORMAT = ".mp4"
# Common video codecs, choose based on format and OS compatibility
VIDEO_CODECS = {
    ".mp4": 'mp4v', # More common/compatible than X264 sometimes
    ".avi": 'XVID',
    ".mkv": 'mp4v'  # MKV is a container, use a common codec
}
CAMERA_API_PREFERENCE = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

# --- Helper Functions ---
def get_timestamp(for_filename=False):
    """Returns the current timestamp in a readable or filename-safe format."""
    now = datetime.now()
    if for_filename:
        return now.strftime("%Y%m%d_%H%M%S")
    else:
        return now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def validate_serial_settings(baudrate_str, data_bits_str, parity_str, stop_bits_str):
    # (Same as previous version)
    try:
        baudrate = int(baudrate_str)
        data_bits_map = {"5": serial.FIVEBITS, "6": serial.SIXBITS, "7": serial.SEVENBITS, "8": serial.EIGHTBITS}
        parity_map = {"None": serial.PARITY_NONE, "Even": serial.PARITY_EVEN, "Odd": serial.PARITY_ODD, "Mark": serial.PARITY_MARK, "Space": serial.PARITY_SPACE}
        stop_bits_map = {"1": serial.STOPBITS_ONE, "1.5": serial.STOPBITS_ONE_POINT_FIVE, "2": serial.STOPBITS_TWO}
        data_bits = data_bits_map.get(data_bits_str)
        parity = parity_map.get(parity_str)
        stop_bits = stop_bits_map.get(stop_bits_str)
        if not all([data_bits is not None, parity is not None, stop_bits is not None]): # Check for None
            raise ValueError("Invalid serial setting value.")
        return baudrate, data_bits, parity, stop_bits
    except (ValueError, TypeError) as e:
        print(f"Error validating settings: {e}")
        return None, None, None, None

# --- Serial Communication Thread ---
class SerialWorker(QThread):
    # (Mostly same as previous version)
    data_received = pyqtSignal(bytes)
    connection_status = pyqtSignal(bool, str) # Signal: connected(True/False), message/port
    connection_error = pyqtSignal(str)

    def __init__(self, port, baudrate, bytesize, parity, stopbits, read_interval_ms=50):
        super().__init__()
        self.serial_port = None
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.read_interval_ms = max(10, read_interval_ms) # Minimum read interval
        self._is_running = False

    def run(self):
        self._is_running = True
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits,
                timeout=0.1 # Short timeout for non-blocking read attempt
            )
            self.connection_status.emit(True, self.port) # Signal successful connection
            print(f"SerialWorker: Connected to {self.port}")

            while self._is_running and self.serial_port and self.serial_port.is_open:
                bytes_available = 0
                try:
                    bytes_available = self.serial_port.in_waiting
                except Exception as wait_err: # Catch errors like port disappearing
                    print(f"SerialWorker: Error checking in_waiting on {self.port}: {wait_err}")
                    self.connection_error.emit(f"Lost connection to {self.port}: {wait_err}")
                    self._is_running = False # Stop the loop
                    break # Exit while loop

                if bytes_available > 0:
                    try:
                        data = self.serial_port.read(bytes_available)
                        if data:
                            self.data_received.emit(data)
                    except serial.SerialException as read_err:
                        print(f"SerialWorker: Read error on {self.port}: {read_err}")
                        self.msleep(100) # Avoid busy-looping on error
                    except Exception as general_read_err:
                        print(f"SerialWorker: Unexpected read error: {general_read_err}")
                        self.msleep(100)
                else:
                    self.msleep(self.read_interval_ms)

        except serial.SerialException as e:
            print(f"SerialWorker: Failed to open port {self.port}: {e}")
            self.connection_error.emit(f"Failed to connect: {e}")
        except Exception as ex:
            print(f"SerialWorker: Unexpected error: {ex}")
            self.connection_error.emit(f"Unexpected error: {ex}")
        finally:
            is_open = False
            if self.serial_port:
                 try:
                      is_open = self.serial_port.is_open
                      if is_open:
                          self.serial_port.close()
                          print(f"SerialWorker: Port {self.port} closed.")
                 except Exception as close_err:
                      print(f"SerialWorker: Error closing port {self.port}: {close_err}")

            self._is_running = False
            if 'e' not in locals() and 'ex' not in locals() or not is_open :
                self.connection_status.emit(False, "Disconnected")
            print("SerialWorker: Thread finished.")

    def stop(self):
        self._is_running = False
        print("SerialWorker: Stop requested.")

    def set_read_interval(self, interval_ms):
        self.read_interval_ms = max(10, interval_ms)
        print(f"SerialWorker: Read interval set to {self.read_interval_ms} ms")


# --- Camera Handling Thread (Enhanced for Recording) ---
class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    camera_status = pyqtSignal(bool, str) # Signal: is_running, message
    recording_status = pyqtSignal(str) # Signal: "Idle", "REC", "Paused", "Error:...", "Saved: <filename>"
    camera_error = pyqtSignal(str)

    def __init__(self, camera_source):
        super().__init__()
        self.camera_source = camera_source
        self.cap = None
        self._is_running = False
        self.do_grayscale = False
        self.do_motion_detection = False
        self.show_overlay = True
        self.last_frame = None
        self.motion_threshold = 500

        # Recording state
        self.video_writer = None
        self.is_recording = False
        self.is_paused = False
        self.recording_start_time = None
        self.output_directory = None
        self.video_filename_base = "rec"
        self.video_format = DEFAULT_VIDEO_FORMAT
        self.video_filename = None # Full path to current video file
        self.target_fps = 20 # Target FPS for recording, adjust as needed

    def run(self):
        self._is_running = True
        frame_count = 0
        start_time = time.time()
        actual_fps = self.target_fps # Initialize with target

        print(f"CameraWorker: Trying to open camera: {self.camera_source} with API {CAMERA_API_PREFERENCE}")
        try:
            # Attempt to convert to int if possible (for USB cameras)
            try:
                source = int(self.camera_source)
            except ValueError:
                source = self.camera_source # Assume it's a URL/path

            self.cap = cv2.VideoCapture(source, CAMERA_API_PREFERENCE)

            if not self.cap or not self.cap.isOpened():
                raise IOError(f"Cannot open camera: {self.camera_source}")

            self.camera_status.emit(True, f"Camera {self.camera_source} running")
            print(f"CameraWorker: Camera {self.camera_source} opened successfully.")

            last_frame_time = time.time()

            while self._is_running and self.cap.isOpened():
                current_time = time.time()
                expected_delay = 1.0 / self.target_fps
                time_since_last_frame = current_time - last_frame_time

                if time_since_last_frame < expected_delay:
                    sleep_duration_ms = int((expected_delay - time_since_last_frame) * 1000) -1
                    if sleep_duration_ms > 0:
                        self.msleep(sleep_duration_ms)

                ret, frame = self.cap.read()
                last_frame_time = time.time()

                if not ret:
                    print("CameraWorker: Failed to grab frame or stream ended.")
                    self.msleep(100)
                    continue

                processed_frame = frame.copy()
                overlay_texts = []

                motion_detected_this_frame = False
                if self.do_motion_detection:
                    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0)
                    if self.last_frame is not None:
                        frame_delta = cv2.absdiff(self.last_frame, gray)
                        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                        thresh = cv2.dilate(thresh, None, iterations=2)
                        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if cv2.contourArea(contour) >= self.motion_threshold:
                                motion_detected_this_frame = True
                                (x, y, w, h) = cv2.boundingRect(contour)
                                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        if motion_detected_this_frame:
                             overlay_texts.append(("Motion Detected", (10, 70), 0.7, (0, 0, 255)))
                    self.last_frame = gray

                processed_display_frame = processed_frame
                processed_write_frame = processed_frame

                if self.do_grayscale:
                    if len(processed_display_frame.shape) == 3:
                         processed_display_frame = cv2.cvtColor(processed_display_frame, cv2.COLOR_BGR2GRAY)

                if self.is_recording and not self.is_paused:
                    if self.video_writer is not None:
                        try:
                            write_frame = processed_write_frame
                            if len(write_frame.shape) == 2:
                                write_frame = cv2.cvtColor(write_frame, cv2.COLOR_GRAY2BGR)
                            if write_frame.shape[0] > 0 and write_frame.shape[1] > 0:
                                 self.video_writer.write(write_frame)
                        except cv2.error as write_err:
                            error_msg = f"Video write error: {write_err}"
                            print(f"CameraWorker: {error_msg}")
                            self.recording_status.emit(f"Error: {error_msg}")
                            self.stop_recording(emit_saved=False)
                        except Exception as generic_write_err:
                            error_msg = f"Unexpected video write error: {generic_write_err}"
                            print(f"CameraWorker: {error_msg}")
                            self.recording_status.emit(f"Error: {error_msg}")
                            self.stop_recording(emit_saved=False)

                if self.show_overlay:
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 1.0 :
                        actual_fps = frame_count / elapsed_time
                        frame_count = 0
                        start_time = time.time()

                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    overlay_texts.append((f"CAM {self.camera_source} {timestamp_str} FPS:{actual_fps:.1f}", (10, 20), 0.5, (0, 255, 0)))

                    status_text = "LIVE"
                    status_color = (0, 255, 0) # Green

                    if self.is_recording:
                        if self.is_paused:
                            status_text = "PAUSED"
                            status_color = (0, 255, 255) # Yellow
                        else:
                            rec_time = time.time() - self.recording_start_time
                            status_text = f"REC {int(rec_time // 60):02d}:{int(rec_time % 60):02d}"
                            status_color = (0, 0, 255) # Red

                    if self.do_grayscale: status_text += " GRAY"
                    overlay_texts.append((status_text, (processed_display_frame.shape[1] - 150, 20), 0.6, status_color))

                    frame_to_draw_on = processed_display_frame
                    is_color_display = len(frame_to_draw_on.shape) == 3

                    for text, pos, scale, color in overlay_texts:
                        if not is_color_display:
                             draw_color = 255
                        else:
                            draw_color = color
                        try:
                            cv2.putText(frame_to_draw_on, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                                        scale, draw_color, 1 if scale <= 0.5 else 2, cv2.LINE_AA)
                        except Exception as puttext_err:
                            print(f"Error adding overlay text '{text}': {puttext_err}")

                self.frame_ready.emit(processed_display_frame)

        except (IOError, cv2.error) as e:
            error_msg = f"Camera {self.camera_source} error: {e}"
            print(f"CameraWorker: {error_msg}")
            self.camera_error.emit(error_msg)
        except Exception as ex:
            error_msg = f"Unexpected camera {self.camera_source} error: {ex}"
            print(f"CameraWorker: {error_msg}")
            self.camera_error.emit(error_msg)
        finally:
            print(f"CameraWorker: Cleaning up for camera {self.camera_source}...")
            self.stop_recording(emit_saved=True)
            if self.cap:
                try:
                    self.cap.release()
                    print(f"CameraWorker: Camera {self.camera_source} released.")
                except Exception as release_err:
                     print(f"CameraWorker: Error releasing camera: {release_err}")
            self._is_running = False
            self.last_frame = None
            self.camera_status.emit(False, f"Camera {self.camera_source} stopped")
            if not self.is_recording:
                 self.recording_status.emit("Idle")
            print("CameraWorker: Thread finished.")

    def stop(self):
        self._is_running = False
        print(f"CameraWorker: Stop requested for camera {self.camera_source}.")

    def start_recording(self, output_dir, filename_base="rec", format=".mp4", fps=None):
        if not self._is_running or not self.cap or not self.cap.isOpened():
            self.recording_status.emit("Error: Camera not running")
            return

        if self.is_recording and not self.is_paused:
            print("CameraWorker: Already recording.")
            return

        if self.is_paused:
            print("CameraWorker: Resuming recording.")
            self.is_paused = False
            self.recording_status.emit("REC")
            return

        self.output_directory = output_dir
        self.video_filename_base = filename_base
        self.video_format = format.lower()
        effective_fps = float(fps) if fps else float(self.target_fps)

        if not self.output_directory or not os.path.exists(self.output_directory):
            error_msg = f"Output directory invalid or does not exist: {self.output_directory}"
            print(f"CameraWorker: {error_msg}")
            self.recording_status.emit(f"Error: {error_msg}")
            return

        try:
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if frame_width <= 0 or frame_height <= 0:
                raise ValueError("Invalid frame dimensions obtained from camera.")

            codec_tag = VIDEO_CODECS.get(self.video_format, 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*codec_tag)

            timestamp = get_timestamp(for_filename=True)
            self.video_filename = os.path.join(self.output_directory, f"{self.video_filename_base}_{timestamp}{self.video_format}")

            print(f"CameraWorker: Starting recording to {self.video_filename} "
                  f"({frame_width}x{frame_height} @ {effective_fps:.2f} FPS, Codec: {codec_tag})")

            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, effective_fps, (frame_width, frame_height))

            if not self.video_writer.isOpened():
                 if self.video_format == ".mp4" and codec_tag == 'mp4v':
                     print("mp4v failed, trying avc1...")
                     codec_tag = 'avc1'
                     fourcc = cv2.VideoWriter_fourcc(*codec_tag)
                     self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, effective_fps, (frame_width, frame_height))

                 if not self.video_writer.isOpened():
                    if self.video_format == ".mp4" and codec_tag != 'XVID':
                        print("avc1/mp4v failed, trying XVID for mp4 container...")
                        codec_tag = 'XVID'
                        fourcc = cv2.VideoWriter_fourcc(*codec_tag)
                        self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, effective_fps, (frame_width, frame_height))

                 if not self.video_writer.isOpened():
                    raise IOError(f"Could not open VideoWriter for {self.video_filename} with codec {codec_tag}. Check codec availability.")


            self.is_recording = True
            self.is_paused = False
            self.recording_start_time = time.time()
            self.recording_status.emit("REC")

        except (cv2.error, IOError, ValueError, Exception) as e:
            error_msg = f"Failed to start recording: {e}"
            print(f"CameraWorker: {error_msg}")
            self.recording_status.emit(f"Error: {error_msg}")
            self.is_recording = False
            self.is_paused = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            if self.video_filename and os.path.exists(self.video_filename) and os.path.getsize(self.video_filename) == 0:
                try:
                    os.remove(self.video_filename)
                except OSError:
                    pass
            self.video_filename = None

    def pause_recording(self):
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            print("CameraWorker: Recording paused.")
            self.recording_status.emit("Paused")

    def stop_recording(self, emit_saved=True):
        saved_file = None
        if self.is_recording:
            print(f"CameraWorker: Stopping recording for {self.video_filename}...")
            self.is_recording = False
            self.is_paused = False
            if self.video_writer:
                try:
                    self.video_writer.release()
                    print(f"CameraWorker: VideoWriter released.")
                    saved_file = self.video_filename
                except Exception as e:
                     print(f"CameraWorker: Error releasing video writer: {e}")
                self.video_writer = None
                self.video_filename = None # Reset filename after stopping

            if saved_file and emit_saved:
                self.recording_status.emit(f"Saved: {os.path.basename(saved_file)}")
            elif emit_saved: # If recording was true but no file was saved (e.g. error)
                 self.recording_status.emit("Idle")

        elif emit_saved: # Ensure Idle state is emitted if stop is called when not recording
             self.recording_status.emit("Idle")

        return saved_file # Return the path in case needed


    # --- Setters for processing ---
    def set_grayscale(self, enabled):
        self.do_grayscale = enabled

    def set_motion_detection(self, enabled):
        self.do_motion_detection = enabled
        if not enabled:
            self.last_frame = None

    def set_overlay(self, enabled):
        self.show_overlay = enabled

# --- Main Application Window ---
class SerialMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, 1300, 850) # Wider for clarity

        # --- Member Variables ---
        self.serial_worker = None
        self.camera_worker = None
        self.is_serial_connected = False
        self.is_camera_running = False
        self.is_camera_recording = False # Simplified state for UI logic
        self.log_file_path = DEFAULT_LOG_FILE
        self.command_history = []
        self.settings = QSettings(ORG_NAME, APP_NAME)
        # self.alert_sound = QSoundEffect(self) # Removed
        self.read_interval_timer = QTimer(self) # For updating read interval setting

        # --- Load Settings ---
        self.video_output_dir = self.settings.value("videoOutputDir", DEFAULT_VIDEO_DIR)
        # Ensure default video dir exists
        os.makedirs(self.video_output_dir, exist_ok=True)
        # self.alert_sound_path = self.settings.value("alertSoundPath", DEFAULT_ALERT_SOUND) # Removed

        # --- Initialize Sound ---
        # self.setup_sound() # Removed

        # --- Build UI ---
        self.initUI()

        # --- Final Touches ---
        self.populate_ports()
        self.populate_cameras()
        self.load_ui_settings() # Load checkbox states etc AFTER UI exists
        self.update_ui_state() # Set initial enable/disable states

    def initUI(self):
        """Creates all UI elements."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        grid_layout = QGridLayout()
        self.main_layout.addLayout(grid_layout)

        # --- Section 1: Serial Config & Control ---
        serial_group = self.create_serial_group()
        grid_layout.addWidget(serial_group, 0, 0)

        # --- Section 2: Serial Data RX ---
        rx_data_group = self.create_rx_data_group()
        grid_layout.addWidget(rx_data_group, 1, 0)

        # --- Section 3: Serial Command TX & Simulation ---
        tx_command_group = self.create_tx_command_group()
        grid_layout.addWidget(tx_command_group, 2, 0)

        # --- Section 4: Camera Control & View ---
        camera_group = self.create_camera_group()
        grid_layout.addWidget(camera_group, 0, 1, 2, 1) # Span 2 rows

        # --- Section 5: Command Log ---
        log_group = self.create_log_group()
        grid_layout.addWidget(log_group, 2, 1, 2, 1) # Span 2 rows

        # --- Section 6: Settings ---
        settings_group = self.create_settings_group()
        grid_layout.addWidget(settings_group, 3, 0, 1, 2) # Span 2 cols at bottom

        # Make columns roughly equal width, let rows adjust
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.com_status_label = QLabel("COM: Disconnected")
        self.cam_status_label = QLabel("CAM: Idle")
        self.rec_status_label = QLabel("REC: Idle")
        self.status_bar.addPermanentWidget(self.com_status_label)
        self.status_bar.addPermanentWidget(self.cam_status_label)
        self.status_bar.addPermanentWidget(self.rec_status_label)
        self.status_bar.showMessage("Ready", 5000)

    # --- UI Creation Helper Methods ---
    def create_serial_group(self):
        group = QGroupBox("Serial Communication (COM)")
        layout = QGridLayout()
        group.setLayout(layout)

        layout.addWidget(QLabel("Port:"), 0, 0)
        self.combo_ports = QComboBox()
        self.combo_ports.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.combo_ports, 0, 1)
        self.btn_refresh_ports = QPushButton("Refresh")
        self.btn_refresh_ports.clicked.connect(self.populate_ports)
        layout.addWidget(self.btn_refresh_ports, 0, 2)

        layout.addWidget(QLabel("Baud Rate:"), 1, 0)
        self.combo_baud = QComboBox()
        self.combo_baud.addItems(['9600', '19200', '38400', '57600', '115200', '230400', '460800', '921600'])
        last_baud = self.settings.value("lastBaudRate", str(DEFAULT_BAUDRATE))
        self.combo_baud.setCurrentText(last_baud)
        layout.addWidget(self.combo_baud, 1, 1, 1, 2)

        layout.addWidget(QLabel("Data Bits:"), 2, 0)
        self.combo_databits = QComboBox(); self.combo_databits.addItems(['8', '7', '6', '5']); self.combo_databits.setCurrentText('8')
        layout.addWidget(self.combo_databits, 2, 1)
        layout.addWidget(QLabel("Parity:"), 2, 2)
        self.combo_parity = QComboBox(); self.combo_parity.addItems(['None', 'Even', 'Odd', 'Mark', 'Space'])
        layout.addWidget(self.combo_parity, 2, 3)
        layout.addWidget(QLabel("Stop Bits:"), 3, 0)
        self.combo_stopbits = QComboBox(); self.combo_stopbits.addItems(['1', '1.5', '2'])
        layout.addWidget(self.combo_stopbits, 3, 1)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self.toggle_serial_connection)
        layout.addWidget(self.btn_connect, 4, 0, 1, 4)

        return group

    def create_rx_data_group(self):
        group = QGroupBox("Received Serial Data")
        layout = QVBoxLayout()
        group.setLayout(layout)
        self.text_serial_output = QTextEdit()
        self.text_serial_output.setReadOnly(True)
        self.text_serial_output.setFontFamily("Consolas")
        layout.addWidget(self.text_serial_output)
        self.btn_clear_rx = QPushButton("Clear Output")
        self.btn_clear_rx.clicked.connect(lambda: self.text_serial_output.clear())
        layout.addWidget(self.btn_clear_rx)
        return group

    def create_tx_command_group(self):
        group = QGroupBox("Send & Simulate Serial Commands")
        layout = QGridLayout()
        group.setLayout(layout)

        layout.addWidget(QLabel("Manual:"), 0, 0)
        self.entry_command = QLineEdit(); self.entry_command.setPlaceholderText("Type command")
        layout.addWidget(self.entry_command, 0, 1)
        self.btn_send_manual = QPushButton("Send Manual")
        self.btn_send_manual.clicked.connect(self.send_manual_command)
        layout.addWidget(self.btn_send_manual, 0, 2)

        layout.addWidget(QLabel("Simulate:"), 1, 0)
        sim_layout = QHBoxLayout()
        self.btn_sim_quay = QPushButton("Sim QUAY")
        self.btn_sim_tamdung = QPushButton("Sim TAM DUNG")
        self.btn_sim_luu = QPushButton("Sim LUU")
        self.btn_sim_quay.clicked.connect(lambda: self.handle_command("quay", source="SIM"))
        self.btn_sim_tamdung.clicked.connect(lambda: self.handle_command("tam dung", source="SIM"))
        self.btn_sim_luu.clicked.connect(lambda: self.handle_command("luu", source="SIM"))
        sim_layout.addWidget(self.btn_sim_quay)
        sim_layout.addWidget(self.btn_sim_tamdung)
        sim_layout.addWidget(self.btn_sim_luu)
        layout.addLayout(sim_layout, 1, 1, 1, 2)

        return group

    def create_camera_group(self):
        group = QGroupBox("Camera Control")
        layout = QVBoxLayout()
        group.setLayout(layout)

        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Select Cam:"))
        self.combo_cameras = QComboBox()
        select_layout.addWidget(self.combo_cameras)
        self.btn_refresh_cameras = QPushButton("Refresh")
        self.btn_refresh_cameras.clicked.connect(self.populate_cameras)
        select_layout.addWidget(self.btn_refresh_cameras)
        layout.addLayout(select_layout)

        self.label_camera_feed = QLabel("Camera Off")
        self.label_camera_feed.setAlignment(Qt.AlignCenter)
        self.label_camera_feed.setMinimumSize(480, 360)
        self.label_camera_feed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_camera_feed.setStyleSheet("background-color: #DDD; color: black; border: 1px solid gray;") # Default light bg
        layout.addWidget(self.label_camera_feed, 1)

        man_ctrl_layout = QGridLayout()
        self.btn_toggle_camera = QPushButton("Start Camera")
        self.btn_toggle_camera.clicked.connect(self.toggle_camera_feed)
        man_ctrl_layout.addWidget(self.btn_toggle_camera, 0, 0)

        self.btn_manual_rec = QPushButton("Record")
        self.btn_manual_rec.setCheckable(True)
        self.btn_manual_rec.clicked.connect(self.toggle_manual_recording)
        man_ctrl_layout.addWidget(self.btn_manual_rec, 0, 1)

        self.btn_capture_image = QPushButton("Capture Image")
        self.btn_capture_image.clicked.connect(self.capture_image)
        man_ctrl_layout.addWidget(self.btn_capture_image, 0, 2)
        layout.addLayout(man_ctrl_layout)

        proc_layout = QHBoxLayout()
        self.check_grayscale = QCheckBox("Grayscale")
        self.check_grayscale.stateChanged.connect(self.update_camera_processing_options)
        proc_layout.addWidget(self.check_grayscale)
        self.check_motion = QCheckBox("Motion Detect")
        self.check_motion.stateChanged.connect(self.update_camera_processing_options)
        proc_layout.addWidget(self.check_motion)
        self.check_overlay = QCheckBox("Show Overlay")
        self.check_overlay.setChecked(True)
        self.check_overlay.stateChanged.connect(self.update_camera_processing_options)
        proc_layout.addWidget(self.check_overlay)
        layout.addLayout(proc_layout)

        vid_set_layout = QHBoxLayout()
        vid_set_layout.addWidget(QLabel("Save Dir:"))
        self.label_video_dir = QLineEdit(self.video_output_dir)
        self.label_video_dir.setReadOnly(True)
        vid_set_layout.addWidget(self.label_video_dir, 1)
        self.btn_select_video_dir = QPushButton("...")
        self.btn_select_video_dir.setToolTip("Select Video Save Directory")
        self.btn_select_video_dir.clicked.connect(self.select_video_directory)
        vid_set_layout.addWidget(self.btn_select_video_dir)
        vid_set_layout.addWidget(QLabel("Format:"))
        self.combo_video_format = QComboBox()
        self.combo_video_format.addItems(VIDEO_CODECS.keys())
        last_format = self.settings.value("videoFormat", DEFAULT_VIDEO_FORMAT)
        self.combo_video_format.setCurrentText(last_format)
        vid_set_layout.addWidget(self.combo_video_format)
        layout.addLayout(vid_set_layout)

        return group

    def create_log_group(self):
        group = QGroupBox("Command Log")
        layout = QVBoxLayout()
        group.setLayout(layout)

        self.table_log = QTableWidget()
        self.table_log.setColumnCount(4)
        self.table_log.setHorizontalHeaderLabels(["Timestamp", "Source", "Command", "Status"])
        self.table_log.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_log.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_log.verticalHeader().setVisible(False)
        header = self.table_log.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents) # Source
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self.table_log, 1)

        log_btn_layout = QHBoxLayout()
        self.btn_export_log = QPushButton("Export Log")
        self.btn_export_log.clicked.connect(self.export_log)
        log_btn_layout.addWidget(self.btn_export_log)
        self.btn_clear_log_table = QPushButton("Clear Table")
        self.btn_clear_log_table.clicked.connect(self.clear_log_table)
        log_btn_layout.addWidget(self.btn_clear_log_table)
        layout.addLayout(log_btn_layout)
        return group

    def create_settings_group(self):
        group = QGroupBox("Settings")
        layout = QGridLayout()
        group.setLayout(layout)

        # Row 0: Serial Read Interval (moved to top-left)
        layout.addWidget(QLabel("Serial Read Interval (ms):"), 0, 0, Qt.AlignRight) # Align label right
        self.spin_read_interval = QSpinBox()
        self.spin_read_interval.setRange(10, 1000) # 10ms to 1s
        self.spin_read_interval.setValue(self.settings.value("readInterval", 50, type=int))
        self.spin_read_interval.setSingleStep(10)
        self.spin_read_interval.valueChanged.connect(self.schedule_read_interval_update) # Update after short delay
        layout.addWidget(self.spin_read_interval, 0, 1) # Put spinbox next to label

        # Row 1: Debug Mode
        self.check_debug_mode = QCheckBox("Debug Mode (Verbose Logging)") # Placeholder
        # self.check_debug_mode.stateChanged.connect(self.set_debug_mode)
        layout.addWidget(self.check_debug_mode, 1, 0) # Moved to row 1, column 0

        # Dark Mode Checkbox Removed
        # Sound Alert Checkbox & Button Removed

        # Add stretch to push items to the left/top if desired, or manage columns
        layout.setColumnStretch(2, 1) # Add stretch to the right

        return group

    # --- Settings & State Loading/Saving ---

    # setup_sound removed

    def load_ui_settings(self):
        """Load settings that affect UI state after UI is built."""
        # Theme related code removed
        # Checkbox for sound alert removed

        self.spin_read_interval.setValue(self.settings.value("readInterval", 50, type=int))

    def save_basic_settings(self):
        """Save simple settings immediately when they change."""
        # Dark mode setting removed
        # Sound settings removed
        self.settings.setValue("readInterval", self.spin_read_interval.value())
        self.settings.setValue("videoOutputDir", self.video_output_dir)
        self.settings.setValue("videoFormat", self.combo_video_format.currentText())

        if self.is_serial_connected:
            current_port_text = self.combo_ports.currentText()
            if current_port_text and "No ports" not in current_port_text :
                self.settings.setValue("lastSerialPort", current_port_text)
            self.settings.setValue("lastBaudRate", self.combo_baud.currentText())

    def schedule_read_interval_update(self):
        """Update serial read interval after a short delay to avoid rapid changes."""
        self.read_interval_timer.stop()
        self.read_interval_timer.setSingleShot(True)
        self.read_interval_timer.timeout.connect(self.update_serial_read_interval)
        self.read_interval_timer.start(500)

    def update_serial_read_interval(self):
        """Applies the new read interval to the worker thread."""
        new_interval = self.spin_read_interval.value()
        if self.serial_worker and self.serial_worker.isRunning():
            self.serial_worker.set_read_interval(new_interval)
        self.settings.setValue("readInterval", new_interval) # Save the final value
        print(f"Applied serial read interval: {new_interval} ms")

    # --- UI State Management ---
    def update_ui_state(self):
        """Enable/disable UI elements based on connection/running states."""
        serial_options_enabled = not self.is_serial_connected
        self.combo_ports.setEnabled(serial_options_enabled)
        self.btn_refresh_ports.setEnabled(serial_options_enabled)
        self.combo_baud.setEnabled(serial_options_enabled)
        self.combo_databits.setEnabled(serial_options_enabled)
        self.combo_parity.setEnabled(serial_options_enabled)
        self.combo_stopbits.setEnabled(serial_options_enabled)
        self.spin_read_interval.setEnabled(True) # Always enabled
        self.btn_connect.setText("Disconnect" if self.is_serial_connected else "Connect")
        self.entry_command.setEnabled(self.is_serial_connected)
        self.btn_send_manual.setEnabled(self.is_serial_connected)

        camera_select_enabled = not self.is_camera_running
        camera_running = self.is_camera_running
        self.combo_cameras.setEnabled(camera_select_enabled)
        self.btn_refresh_cameras.setEnabled(camera_select_enabled)
        self.btn_toggle_camera.setText("Stop Camera" if camera_running else "Start Camera")

        self.check_grayscale.setEnabled(camera_running)
        self.check_motion.setEnabled(camera_running)
        self.check_overlay.setEnabled(camera_running)
        self.btn_capture_image.setEnabled(camera_running)
        self.btn_manual_rec.setEnabled(camera_running)


    # --- Serial Port Methods ---
    # populate_ports, toggle_serial_connection, connect_serial, disconnect_serial
    # handle_serial_connection_status, handle_serial_error
    # are mostly unchanged
    def populate_ports(self):
        self.combo_ports.clear()
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.combo_ports.addItem("No ports found")
            self.combo_ports.setEnabled(False)
            self.btn_connect.setEnabled(False)
        else:
            last_port = self.settings.value("lastSerialPort", "")
            selected_index = -1
            current_index = 0
            for port in sorted(ports):
                port_str = f"{port.device} - {port.description}"
                self.combo_ports.addItem(port_str)
                if last_port and last_port in port_str:
                    selected_index = current_index
                current_index += 1

            if selected_index != -1:
                 self.combo_ports.setCurrentIndex(selected_index)

            self.combo_ports.setEnabled(True)
            self.btn_connect.setEnabled(True)

    def toggle_serial_connection(self):
        if self.is_serial_connected:
            self.disconnect_serial()
        else:
            self.connect_serial()

    def connect_serial(self):
        selected_port_text = self.combo_ports.currentText()
        if not selected_port_text or "No ports found" in selected_port_text:
            self.show_error_message("No serial port selected.")
            return

        port = selected_port_text.split(" - ")[0]
        baudrate, bytesize, parity, stopbits = validate_serial_settings(
            self.combo_baud.currentText(),
            self.combo_databits.currentText(),
            self.combo_parity.currentText(),
            self.combo_stopbits.currentText()
        )

        if baudrate is None:
            self.show_error_message("Invalid serial port settings.")
            return

        self.status_bar.showMessage(f"Connecting to {port}...", 5000)
        self.com_status_label.setText("COM: Connecting...")
        self.com_status_label.setStyleSheet("color: orange;")

        read_interval = self.spin_read_interval.value()
        self.serial_worker = SerialWorker(port, baudrate, bytesize, parity, stopbits, read_interval)
        self.serial_worker.data_received.connect(self.handle_serial_data)
        self.serial_worker.connection_status.connect(self.handle_serial_connection_status)
        self.serial_worker.connection_error.connect(self.handle_serial_error)
        self.serial_worker.finished.connect(self.on_serial_worker_finished)
        self.serial_worker.start()

        self.btn_connect.setEnabled(False)

    def disconnect_serial(self):
        if self.serial_worker and self.serial_worker.isRunning():
            self.status_bar.showMessage("Disconnecting serial port...", 3000)
            self.serial_worker.stop()
        else:
             if self.is_serial_connected:
                  self.handle_serial_connection_status(False, "Forcibly Disconnected")

    def handle_serial_connection_status(self, connected, message):
        self.is_serial_connected = connected
        self.btn_connect.setEnabled(True)

        if connected:
            port_name = message
            self.status_bar.showMessage(f"Serial connected: {port_name}", 5000)
            self.com_status_label.setText(f"COM: Connected ({os.path.basename(port_name)})")
            self.com_status_label.setStyleSheet("color: green;")
            self.save_basic_settings() # Save successful connection settings
        else:
            self.status_bar.showMessage(f"Serial disconnected: {message}", 5000)
            self.com_status_label.setText("COM: Disconnected")
            self.com_status_label.setStyleSheet("color: red;")
            if "error" in message.lower():
                self.com_status_label.setStyleSheet("color: red; font-weight: bold;")

        self.update_ui_state()

    def handle_serial_error(self, error_message):
        self.is_serial_connected = False
        self.show_error_message(f"Serial Error: {error_message}")
        self.status_bar.showMessage(f"Serial Error: {error_message}", 10000)
        self.com_status_label.setText("COM: Error")
        self.com_status_label.setStyleSheet("color: red; font-weight: bold;")
        self.btn_connect.setEnabled(True)
        self.update_ui_state()
        self.serial_worker = None


    def handle_serial_data(self, data):
        try:
            # Try decoding common encodings, fallback to replacing errors
            try:
                 text = data.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = data.decode('latin-1')
                except Exception:
                    text = data.decode('utf-8', errors='replace')

        except Exception as e:
            text = f"[Decode Error: {e}] {repr(data)}"

        timestamp = get_timestamp()
        cursor = self.text_serial_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_serial_output.setTextCursor(cursor)
        self.text_serial_output.insertPlainText(f"[{timestamp}] RX: {text}\n") # Ensure newline if needed
        # Scroll to bottom only if near the bottom already (optional optimization)
        # sb = self.text_serial_output.verticalScrollBar()
        # if sb.value() >= sb.maximum() - 15: # Adjust threshold as needed
        #     sb.setValue(sb.maximum())
        self.text_serial_output.verticalScrollBar().setValue(self.text_serial_output.verticalScrollBar().maximum())


        command = text.strip().lower()
        self.handle_command(command, source="RX")


    def handle_command(self, command, source="Unknown"):
        """Central handler for commands from Serial or Simulation."""
        self.log_command(source, command, "Received")

        if command == "quay":
            self.status_bar.showMessage("Command: QUAY received.", 3000)
            # self.play_alert_sound() # Removed
            self.trigger_recording_start(log_source=source) # Pass source
        elif command == "tam dung" or command == "tạm dừng":
             self.status_bar.showMessage("Command: TAM DUNG received.", 3000)
             # self.play_alert_sound() # Removed
             self.trigger_recording_pause(log_source=source) # Pass source
        elif command == "luu" or command == "lưu":
             self.status_bar.showMessage("Command: LUU received.", 3000)
             self.trigger_recording_stop(log_source=source) # Pass source

    def on_serial_worker_finished(self):
        print("Serial worker thread finished.")
        if self.is_serial_connected:
            print("Warning: Serial worker finished unexpectedly while connected state was True.")
            self.handle_serial_connection_status(False, "Worker Finished Unexpectedly")
        self.serial_worker = None


    def send_serial_command(self, command_str, source="TX"):
        if not self.is_serial_connected or not self.serial_worker or not self.serial_worker.serial_port:
            self.log_command(source, command_str, "Error: Not Connected")
            self.show_error_message("Not connected to serial port.")
            return False
        if not command_str: return False

        try:
            # Determine line ending (simple heuristic, could be a setting)
            line_ending = b'\n' if '\n' not in command_str else b''
            # Send command + line ending
            command_to_send = command_str.encode('utf-8') + line_ending
            bytes_sent = self.serial_worker.serial_port.write(command_to_send)

            timestamp = get_timestamp()
            # Ensure the QTextCursor class is imported
            cursor = self.text_serial_output.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.text_serial_output.setTextCursor(cursor)
            # Display sent command with any auto-added newline for clarity
            display_sent = command_str + ('\\n' if line_ending == b'\n' else '')
            self.text_serial_output.insertPlainText(f"[{timestamp}] {source}: {display_sent}\n")
            cursor.movePosition(QTextCursor.End)
            self.text_serial_output.setTextCursor(cursor)


            self.log_command(source, command_str, f"Success ({bytes_sent} bytes)")
            self.status_bar.showMessage(f"{source}: Sent '{command_str}'", 2000)
            return True
        except (serial.SerialException, serial.SerialTimeoutException) as e:
            error_msg = f"Error sending command '{command_str}': {e}"
            self.show_error_message(error_msg)
            self.log_command(source, command_str, f"Error: {e}")
            self.status_bar.showMessage(error_msg, 7000)
            return False
        except Exception as ex:
            error_msg = f"Unexpected error sending command '{command_str}': {ex}"
            self.show_error_message(error_msg)
            self.log_command(source, command_str, f"Error: {ex}")
            self.status_bar.showMessage(error_msg, 7000)
            return False

    def send_manual_command(self):
        command = self.entry_command.text()
        if self.send_serial_command(command, source="TX"):
            self.entry_command.clear()

    # --- Camera Methods ---
    # populate_cameras, toggle_camera_feed, start_camera_feed, stop_camera_feed
    # handle_camera_status, handle_camera_error, on_camera_worker_finished
    # update_camera_frame, update_camera_processing_options, capture_image
    # are mostly unchanged
    def populate_cameras(self):
        self.combo_cameras.clear()
        available_cameras = []
        print("Scanning for cameras...")
        for i in range(5):
            try:
                cap_test = cv2.VideoCapture(i, CAMERA_API_PREFERENCE)
                if cap_test is not None and cap_test.isOpened():
                    available_cameras.append((i, f"Camera {i}"))
                    cap_test.release()
                    print(f"Found Camera {i}")
                else:
                    if i == 0: break
            except Exception as e:
                 print(f"Error checking camera index {i}: {e}")
                 if i == 0 : break # Stop if index 0 causes an error

        if not available_cameras:
            self.combo_cameras.addItem("No cameras detected")
            self.combo_cameras.setEnabled(False)
            self.btn_toggle_camera.setEnabled(False)
        else:
            for index, name in available_cameras:
                self.combo_cameras.addItem(name, userData=index)
            self.combo_cameras.setEnabled(True)
            self.btn_toggle_camera.setEnabled(True)

    def toggle_camera_feed(self):
        if self.is_camera_running:
            self.stop_camera_feed()
        else:
            self.start_camera_feed()

    def start_camera_feed(self):
        if self.is_camera_running: return # Already running

        selected_index = self.combo_cameras.currentIndex()
        if selected_index < 0:
            self.show_error_message("No camera selected.")
            return
        camera_source = self.combo_cameras.itemData(selected_index)
        if camera_source is None:
             self.show_error_message("Invalid camera selection.")
             return

        if self.btn_manual_rec.isChecked():
             print("Stopping manual recording before starting camera feed...")
             self.toggle_manual_recording(False)

        print(f"Starting camera: {camera_source}")
        self.status_bar.showMessage(f"Starting camera {camera_source}...")
        self.label_camera_feed.setText(f"Starting Camera {camera_source}...")
        self.label_camera_feed.setStyleSheet("background-color: #CCC; color: black;") # Lighter gray bg

        self.camera_worker = CameraWorker(camera_source)
        self.camera_worker.frame_ready.connect(self.update_camera_frame)
        self.camera_worker.camera_status.connect(self.handle_camera_status)
        self.camera_worker.recording_status.connect(self.handle_recording_status)
        self.camera_worker.camera_error.connect(self.handle_camera_error)
        self.camera_worker.finished.connect(self.on_camera_worker_finished)

        self.update_camera_processing_options()
        self.camera_worker.start()

        self.btn_toggle_camera.setEnabled(False)


    def stop_camera_feed(self):
        if self.camera_worker and self.camera_worker.is_recording:
             print("Stopping recording before stopping camera feed...")
             self.trigger_recording_stop(log_source="APP_CLOSE_FEED")

        if self.camera_worker and self.camera_worker.isRunning():
            self.status_bar.showMessage("Stopping camera...", 3000)
            self.camera_worker.stop()
        else:
             self.handle_camera_status(False, "Camera Forcibly Stopped")


    def handle_camera_status(self, running, message):
        self.is_camera_running = running
        self.btn_toggle_camera.setEnabled(True)

        if running:
             self.status_bar.showMessage(message, 5000)
             self.cam_status_label.setText(f"CAM: Running ({self.camera_worker.camera_source})")
             self.cam_status_label.setStyleSheet("color: green;")
             # No specific label style needed here, frame update handles it
        else:
             self.status_bar.showMessage(message, 5000)
             self.cam_status_label.setText("CAM: Idle")
             self.cam_status_label.setStyleSheet("color: gray;") # Or leave default
             self.label_camera_feed.setText("Camera Off")
             self.label_camera_feed.setStyleSheet("background-color: #DDD; color: black; border: 1px solid gray;") # Default light bg
             self.handle_recording_status("Idle")
             self.btn_manual_rec.setChecked(False)

        self.update_ui_state()

    def handle_camera_error(self, error_message):
        self.is_camera_running = False
        self.show_error_message(f"Camera Error: {error_message}")
        self.status_bar.showMessage(f"Camera Error: {error_message}", 10000)
        self.cam_status_label.setText("CAM: Error")
        self.cam_status_label.setStyleSheet("color: red; font-weight: bold;")
        self.label_camera_feed.setText(f"Camera Error:\n{error_message}")
        self.label_camera_feed.setStyleSheet("background-color: #EEE; color: red; border: 1px solid red;") # Light bg, red text
        self.btn_toggle_camera.setEnabled(True)
        self.handle_recording_status("Idle")
        self.btn_manual_rec.setChecked(False)
        self.update_ui_state()
        self.camera_worker = None


    def on_camera_worker_finished(self):
        print("Camera worker thread finished.")
        if self.is_camera_running:
            print("Warning: Camera worker finished unexpectedly while running state was True.")
            self.handle_camera_status(False, "Worker Finished Unexpectedly")
        self.camera_worker = None


    def update_camera_frame(self, frame_np):
        if not self.is_camera_running: return

        try:
            if frame_np is None or frame_np.size == 0: return

            height, width = frame_np.shape[:2]
            bytes_per_line = frame_np.strides[0]

            # Convert color format for QImage
            if len(frame_np.shape) == 3:
                 # OpenCV uses BGR, QImage needs RGB
                 frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                 image_format = QImage.Format_RGB888
                 q_image = QImage(frame_rgb.data, width, height, bytes_per_line, image_format)
            elif len(frame_np.shape) == 2:
                 image_format = QImage.Format_Grayscale8
                 q_image = QImage(frame_np.data, width, height, bytes_per_line, image_format)
            else:
                return # Unsupported format

            pixmap = QPixmap.fromImage(q_image)
            target_size = self.label_camera_feed.size()
            if target_size.width() > 10 and target_size.height() > 10:
                 scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 self.label_camera_feed.setPixmap(scaled_pixmap)
            else:
                 self.label_camera_feed.setPixmap(pixmap.scaled(QSize(320,240), Qt.KeepAspectRatio, Qt.SmoothTransformation))


        except Exception as e:
            print(f"Error updating camera frame display: {e}")
            # Revert to default style on error
            self.label_camera_feed.setText(f"Frame Display Error: {e}")
            self.label_camera_feed.setStyleSheet("background-color: #DDD; color: red; border: 1px solid gray;")


    def update_camera_processing_options(self):
        """Sends current checkbox states to the camera worker."""
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.set_grayscale(self.check_grayscale.isChecked())
            self.camera_worker.set_motion_detection(self.check_motion.isChecked())
            self.camera_worker.set_overlay(self.check_overlay.isChecked())

    def capture_image(self):
        if not self.is_camera_running or not self.camera_worker:
            self.show_error_message("Camera is not running.")
            return

        current_pixmap = self.label_camera_feed.pixmap()
        if not current_pixmap or current_pixmap.isNull():
             self.show_error_message("No image frame available to capture.")
             return

        save_dir = self.video_output_dir if os.path.exists(self.video_output_dir) else os.path.join(os.path.expanduser("~"), "Pictures")
        os.makedirs(save_dir, exist_ok=True)

        timestamp = get_timestamp(for_filename=True)
        default_filename = os.path.join(save_dir, f"capture_{timestamp}.png")

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Image", default_filename,
                                                  "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)

        if fileName:
            base, ext = os.path.splitext(fileName)
            if not ext:
                 if _ and ("*.jpg" in _ or "*.jpeg" in _): ext = ".jpg"
                 elif _ and "*.bmp" in _: ext = ".bmp"
                 else: ext = ".png"
                 fileName = base + ext

            if not current_pixmap.save(fileName):
                self.show_error_message(f"Failed to save image to {fileName}")
                self.log_command("APP", "Capture Image", f"Error: Save failed {fileName}")
            else:
                self.status_bar.showMessage(f"Image saved: {os.path.basename(fileName)}", 4000)
                self.log_command("APP", "Capture Image", f"Success: {fileName}")

    # --- Video Directory Selection ---
    def select_video_directory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        directory = QFileDialog.getExistingDirectory(self,
                        "Select Video Save Directory",
                        self.video_output_dir,
                        options=options)
        if directory:
            if os.path.isdir(directory):
                self.video_output_dir = directory
                self.label_video_dir.setText(directory)
                self.settings.setValue("videoOutputDir", directory)
                self.status_bar.showMessage(f"Video save directory set to: {directory}", 4000)
            else:
                 self.show_error_message(f"Invalid directory selected: {directory}")

    # --- Recording Logic ---
    # trigger_recording_start, trigger_recording_pause, trigger_recording_stop
    # handle_recording_status, toggle_manual_recording
    # are mostly unchanged except for passing log_source
    def trigger_recording_start(self, log_source="Unknown"):
        if not self.is_camera_running:
            self.log_command(log_source, "QUAY", "Error: Camera not running")
            self.show_error_message("Cannot start recording: Camera is not running. Please start camera first.")
            return

        if not self.camera_worker:
             self.log_command(log_source, "QUAY", "Error: Camera worker not ready")
             return

        self.log_command(log_source, "QUAY", "Executing")

        if not self.video_output_dir or not os.path.isdir(self.video_output_dir):
             self.log_command(log_source, "QUAY", f"Error: Invalid Save Directory '{self.video_output_dir}'")
             self.show_error_message(f"Cannot start recording: Invalid save directory selected.\n'{self.video_output_dir}'\nPlease select a valid directory in Camera Controls.")
             return

        # Programmatically ensure button is checked only *after* starting worker method successfully
        # But first, call the worker method which might emit errors
        selected_format = self.combo_video_format.currentText()
        self.camera_worker.start_recording(self.video_output_dir, filename_base="rec", format=selected_format)
        # Don't set self.is_camera_recording=True here. Let handle_recording_status do it on success.
        # Set button checked state based on internal recording flag, handled by status update.

    def trigger_recording_pause(self, log_source="Unknown"):
        if not self.is_camera_recording or not self.camera_worker:
            self.log_command(log_source, "TAM DUNG", "Ignored: Not recording or paused")
            return

        self.log_command(log_source, "TAM DUNG", "Executing")
        self.camera_worker.pause_recording() # Worker emits "Paused" status

    def trigger_recording_stop(self, log_source="Unknown"):
        # Only stop if actively recording or paused
        if not self.is_camera_recording or not self.camera_worker: # Covers both REC and PAUSED state
            self.log_command(log_source, "LUU", "Ignored: Not recording or paused")
            return

        self.log_command(log_source, "LUU", "Executing")
        saved_file_path = self.camera_worker.stop_recording(emit_saved=True)

        # Don't update flags here, let handle_recording_status do it based on emitted signal.
        # It will emit "Saved: ..." or "Idle"
        if saved_file_path:
            self.log_command(log_source, "LUU", f"Success: Saved {os.path.basename(saved_file_path)}")
        else:
             self.log_command(log_source, "LUU", "Stopped (Check Status/Logs)")


    def handle_recording_status(self, status_message):
        """Slot for recording_status signal from CameraWorker."""
        self.rec_status_label.setText(f"REC: {status_message}")
        current_stylesheet = self.rec_status_label.styleSheet() # Preserve existing font etc.
        new_style = "" # Start fresh for color/weight

        if status_message.startswith("Error"):
            new_style = "color: red; font-weight: bold;"
            self.is_camera_recording = False # Error means stopped
            self.btn_manual_rec.setChecked(False)
            self.log_command("CAMERA", "Recording", f"Status: {status_message}")
        elif status_message == "REC":
            new_style = "color: red;"
            self.is_camera_recording = True
            if not self.btn_manual_rec.isChecked(): self.btn_manual_rec.setChecked(True) # Ensure button matches
        elif status_message == "Paused":
            new_style = "color: orange;"
            self.is_camera_recording = True # Still considered 'recording' conceptually
            if not self.btn_manual_rec.isChecked(): self.btn_manual_rec.setChecked(True) # Button stays checked
        elif status_message.startswith("Saved"):
             new_style = "color: green;"
             self.is_camera_recording = False
             self.btn_manual_rec.setChecked(False) # Uncheck after save
        else: # Idle or other
             new_style = "" # Default color from system/stylesheet
             self.is_camera_recording = False
             self.btn_manual_rec.setChecked(False)

        self.rec_status_label.setStyleSheet(new_style) # Apply only color/weight changes


    def toggle_manual_recording(self, checked):
         # This function now acts based on the *intended* state from the button click.
         # The actual state is managed via handle_recording_status.
         if checked:
             # User wants to START or RESUME recording
             # Trigger start only if camera is running
             if self.is_camera_running:
                  self.trigger_recording_start(log_source="MANUAL")
             else:
                  # Prevent check if camera isn't running
                  self.btn_manual_rec.setChecked(False)
                  self.show_info_message("Please start the camera first.")
         else:
             # User wants to STOP recording
             # Only trigger stop if the internal state shows we ARE recording/paused
             if self.is_camera_recording: # Checks if REC or PAUSED
                  self.trigger_recording_stop(log_source="MANUAL")
             # If not recording, unchecking does nothing further


    # --- Logging Methods ---
    # log_command, export_log, clear_log_table
    # are mostly unchanged
    def log_command(self, source, command, status=""):
        timestamp = get_timestamp()
        log_entry = {"ts": timestamp, "src": source, "cmd": command, "status": status}
        self.command_history.append(log_entry)

        row_position = self.table_log.rowCount()
        self.table_log.insertRow(row_position)
        self.table_log.setItem(row_position, 0, QTableWidgetItem(timestamp))
        self.table_log.setItem(row_position, 1, QTableWidgetItem(source))
        display_cmd = command if len(command) < 150 else command[:147] + "..."
        self.table_log.setItem(row_position, 2, QTableWidgetItem(display_cmd))
        self.table_log.setItem(row_position, 3, QTableWidgetItem(status))
        self.table_log.scrollToBottom()

        try:
            with open(self.log_file_path, 'a', encoding='utf-8', newline='') as f:
                 f.write(f"{timestamp}|{source}|{command}|{status}\n")
        except Exception as e:
            print(f"Error writing to log file '{self.log_file_path}': {e}")

    def export_log(self):
        if self.table_log.rowCount() == 0: return
        options = QFileDialog.Options()
        fileName, selected_filter = QFileDialog.getSaveFileName(self, "Export Log", "command_log_export",
                                                  "CSV Files (*.csv);;Text Files (*.txt);;JSON Files (*.json);;All Files (*)", options=options)
        if not fileName: return

        try:
            file_ext = os.path.splitext(fileName)[1].lower()
            if not file_ext:
                 if selected_filter.startswith("CSV"): file_ext = ".csv"; fileName += file_ext
                 elif selected_filter.startswith("JSON"): file_ext = ".json"; fileName += file_ext
                 else: file_ext = ".txt"; fileName += file_ext

            headers = [self.table_log.horizontalHeaderItem(i).text() for i in range(self.table_log.columnCount())]
            data_rows = []
            for row in range(self.table_log.rowCount()):
                 rowData = [self.table_log.item(row, col).text() if self.table_log.item(row, col) else ""
                            for col in range(self.table_log.columnCount())]
                 data_rows.append(rowData)


            with open(fileName, 'w', encoding='utf-8', newline='') as outfile:
                if file_ext == ".csv":
                    writer = csv.writer(outfile)
                    writer.writerow(headers)
                    writer.writerows(data_rows)
                elif file_ext == ".json":
                    json_output = [dict(zip(headers, row)) for row in data_rows]
                    json.dump(json_output, outfile, indent=4)
                else: # TXT
                    outfile.write("\t".join(headers) + "\n")
                    for row in data_rows:
                         outfile.write("\t".join(row) + "\n")
            self.status_bar.showMessage(f"Log exported to {os.path.basename(fileName)}", 4000)
            self.log_command("APP", "Export Log", f"Success: {fileName}")
        except Exception as e:
            self.show_error_message(f"Error exporting log: {e}")
            self.log_command("APP", "Export Log", f"Error: {e}")

    def clear_log_table(self):
        self.table_log.setRowCount(0)
        self.status_bar.showMessage("Log table cleared.", 2000)

    # --- Settings & Appearance ---
    # toggle_dark_mode removed
    # apply_theme removed
    # select_alert_sound removed
    # play_alert_sound removed

    # --- Utility Methods ---
    def show_error_message(self, message):
        QMessageBox.critical(self, "Error", message)

    def show_info_message(self, message):
        QMessageBox.information(self, "Information", message)

    # --- Window Closing ---
    def closeEvent(self, event):
        print("Close event triggered")

        reply = QMessageBox.question(self, 'Confirm Exit',
                                     "Are you sure you want to exit? Any active recording will be stopped and saved.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            print("Stopping threads...")
            if self.is_camera_running:
                self.stop_camera_feed() # Handles recording stop too
                if self.camera_worker: self.camera_worker.wait(1500)

            if self.is_serial_connected:
                self.disconnect_serial()
                if self.serial_worker: self.serial_worker.wait(1000)

            self.save_basic_settings()
            print("Settings saved.")
            event.accept()
            print("Exiting application.")
        else:
            event.ignore()

# --- Main Execution ---
if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) # Re-enable High DPI if desired
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationName(APP_NAME)

    # Initial theme application removed

    mainWin = SerialMonitorApp()
    mainWin.show()
    sys.exit(app.exec_())