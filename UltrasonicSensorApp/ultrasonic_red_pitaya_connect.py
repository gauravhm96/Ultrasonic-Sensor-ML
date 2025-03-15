import datetime
import os
import socket
import struct
import time
import tempfile, os
import matplotlib.pyplot as plt
import numpy as np
import paramiko
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from ping3 import ping
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)
from object_differentiation_predict import Predict_FFT
# Global variables for UDP client management:
global udp_thread
udp_thread = None
udp_socket = None
client_active = False
udp_thread_running = False
fft_running = False
sensor_active = False

# Global variables for plotting
fft_canvas = None
XValues = None
FFT_MIN_FREQUENCY = 35000  # from FFTConfig.MinFrequency


# Global variables for Saving FFT Data
global SAVE_HEADER
SAVE_FILE = "fft_log.txt"  # default log file
logging_active = False
save_streams_count = 0

# Global variables for ML
MODEL_FILE = None
ML_running = False
RealTimePredict = False
predict_buffer = [] 
MAX_PREDICT_MEASUREMENTS = 200
display_distance = None

class UDPClientThread(QThread):
    # Signal to emit incoming UDP data (as bytes)
    data_received = pyqtSignal(bytes)

    def __init__(self, sock, parent=None):
        super().__init__(parent)
        self.sock = sock
        self.running = True

    def run(self):
        self.sock.settimeout(1.0)  # Check every second if we should exit
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                self.data_received.emit(data)
            except socket.timeout:
                continue
            except Exception as e:
                print("UDP error:", e)
                break

    def stop(self):
        self.running = False
        self.wait()


def connect_to_red_pitaya(layout, output_box):
    global udp_thread

    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setLineWidth(2)
    layout.addWidget(line)

    # Main horizontal layout for the left config panel and right data display
    main_h_layout = QHBoxLayout()
    main_h_layout.setSpacing(10)
    main_h_layout.setContentsMargins(10, 10, 10, 10)

    # ------------------------- LEFT PANEL -------------------------
    left_v_layout = QVBoxLayout()

    # 1) UDP CLIENT CONFIG GROUP
    udp_group = QGroupBox("UDP client config")
    udp_group.setStyleSheet("font-size: 18px;")
    udp_layout = QGridLayout()
    udp_layout.setSpacing(5)
    udp_layout.setContentsMargins(5, 5, 5, 5)
    udp_layout.setColumnStretch(0, 1)
    udp_layout.setColumnStretch(1, 2)

    sensor_ip_label_1 = QLabel("Sensor IP address:")
    sensor_ip_label_1.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    sensor_ip_input_1 = QLineEdit("192.168.128.1")

    sensor_ip_port_label = QLabel("Sensor IP port:")
    sensor_ip_port_label.setStyleSheet(
        "font-size: 18px; font-weight: bold; padding: 5px;"
    )
    sensor_ip_port_input = QSpinBox()
    sensor_ip_port_input.setRange(1, 65535)
    sensor_ip_port_input.setValue(61231)

    local_ip_label_1 = QLabel("Local IP address:")
    local_ip_label_1.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    local_ip_input_1 = QLineEdit("127.0.0.1")

    local_ip_port_label = QLabel("Local IP port:")
    local_ip_port_label.setStyleSheet(
        "font-size: 18px; font-weight: bold; padding: 5px;"
    )
    local_ip_port_input = QSpinBox()
    local_ip_port_input.setRange(1, 65535)
    local_ip_port_input.setValue(61231)

    check_connection_button = QPushButton("check sensor connection")
    check_connection_button.setStyleSheet("font-size: 18px; padding: 5px;")
    check_connection_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    start_sensor_button = QPushButton("Start Sensor")
    start_sensor_button.setStyleSheet("font-size: 18px; padding: 5px;")
    start_sensor_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    start_client_button = QPushButton("Start Client")
    start_client_button.setCheckable(True)
    start_client_button.setStyleSheet("font-size: 18px; padding: 5px;")
    start_client_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    # Create a horizontal layout for the three buttons
    button_h_layout = QHBoxLayout()
    button_h_layout.setSpacing(10)
    button_h_layout.setContentsMargins(0, 0, 0, 0)
    button_h_layout.setAlignment(Qt.AlignLeft)
    button_h_layout.addWidget(check_connection_button)
    button_h_layout.addWidget(start_sensor_button)
    button_h_layout.addWidget(start_client_button)

    udp_layout.addWidget(sensor_ip_label_1, 0, 0)
    udp_layout.addWidget(sensor_ip_input_1, 0, 1)
    udp_layout.addWidget(sensor_ip_port_label, 1, 0)
    udp_layout.addWidget(sensor_ip_port_input, 1, 1)
    udp_layout.addWidget(local_ip_label_1, 2, 0)
    udp_layout.addWidget(local_ip_input_1, 2, 1)
    udp_layout.addWidget(local_ip_port_label, 3, 0)
    udp_layout.addWidget(local_ip_port_input, 3, 1)
    udp_layout.addLayout(button_h_layout, 4, 0, 1, 2)

    udp_group.setLayout(udp_layout)
    left_v_layout.addWidget(udp_group)

    # 2) SEND / RECEIVE GROUP
    sr_group = QGroupBox("Send / Receive")
    sr_group.setStyleSheet("font-size: 18px;")
    sr_layout = QGridLayout()
    sr_layout.setSpacing(5)
    sr_layout.setContentsMargins(5, 5, 5, 5)
    sr_layout.setColumnStretch(0, 1)
    sr_layout.setColumnStretch(1, 2)

    # Row 0: FFT Window controls
    fftwindow_label = QLabel("FFT Window Width:")
    fftwindow_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    fftwindow_input = QComboBox()
    fftwindow_input.setStyleSheet("font-size: 18px;")
    fft_values = ["1024", "2048", "4096", "8192"]
    fftwindow_input.addItems(fft_values)
    fftwindow_input.setCurrentText("8192")
    sr_layout.addWidget(fftwindow_label, 0, 0)
    sr_layout.addWidget(fftwindow_input, 0, 1)

    # Row 1: Measurements and Buttons
    row1_layout = QHBoxLayout()
    row1_layout.setSpacing(10)
    row1_layout.setContentsMargins(0, 0, 0, 0)
    row1_layout.setAlignment(Qt.AlignLeft)

    measurements_label = QLabel("Readings:")
    measurements_label.setStyleSheet(
        "font-size: 18px; font-weight: bold; padding: 5px;"
    )
    measurements_input = QSpinBox()
    measurements_input.setMinimumWidth(80)
    measurements_input.setRange(1, 9999)
    measurements_input.setValue(100)
    measurements_input.setStyleSheet("font-size: 18px;")

    start_logging_button = QPushButton("start logging")
    start_logging_button.setStyleSheet("font-size: 18px; padding: 5px;")
    start_logging_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    start_fft_button = QPushButton("start FFT")
    start_fft_button.setCheckable(True)
    start_fft_button.setStyleSheet("font-size: 18px; padding: 5px;")
    start_fft_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    row1_layout.addWidget(measurements_label)
    row1_layout.addWidget(measurements_input)
    row1_layout.addWidget(start_logging_button)
    row1_layout.addWidget(start_fft_button)

    sr_layout.addWidget(fftwindow_label, 0, 0)
    sr_layout.addWidget(fftwindow_input, 0, 1)
    sr_layout.addLayout(row1_layout, 1, 0, 1, 2)

    sr_group.setLayout(sr_layout)
    left_v_layout.addWidget(sr_group)

    # ------------------------- MACHINE LEARNING GROUP -------------------------
    ml_group = QGroupBox("Machine Learning")
    ml_group.setStyleSheet("font-size: 18px;")
    ml_layout = QGridLayout()
    ml_layout.setSpacing(5)
    ml_layout.setContentsMargins(5, 5, 5, 5)
    ml_layout.setColumnStretch(0, 1)
    ml_layout.setColumnStretch(1, 2)

    select_model_button = QPushButton("Select Model")
    select_model_button.setStyleSheet("font-size: 18px; padding: 5px;")
    select_model_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    start_predict_button = QPushButton("Start Predict")
    start_predict_button.setCheckable(True)
    start_predict_button.setStyleSheet("font-size: 18px; padding: 5px;")
    start_predict_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    ml_layout.addWidget(select_model_button, 0, 0)
    ml_layout.addWidget(start_predict_button, 0, 1)
    ml_group.setLayout(ml_layout)
    left_v_layout.addWidget(ml_group)

    # ------------------------- BOTTOM LEFT SENSOR STATUS (VERTICAL) -------------------------
    sensor_status_group = QGroupBox("Sensor Status")
    sensor_status_group.setStyleSheet("font-size: 18px;")
    sensor_status_layout = QVBoxLayout()
    sensor_status_layout.setSpacing(5)
    sensor_status_layout.setContentsMargins(5, 5, 5, 5)

    sensor1_h_layout = QHBoxLayout()
    sensor1_h_layout.setSpacing(10)
    sensor1_h_layout.setContentsMargins(0, 0, 0, 0)
    sensor1_h_layout.setAlignment(Qt.AlignLeft)

    sensor1_label = QLabel("Sensor Status:")
    sensor1_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    sensor1_indicator = QFrame()
    sensor1_indicator.setFixedSize(20, 20)
    sensor1_indicator.setStyleSheet("background-color: red; border: 1px solid black;")
    
    status_display = QLabel()
    status_display.setStyleSheet("font-size: 18px; border: 1px solid gray; padding: 5px;")
    status_display.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    status_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    sensor1_h_layout.addWidget(sensor1_label)
    sensor1_h_layout.addWidget(sensor1_indicator)
    sensor1_h_layout.addWidget(status_display)
    sensor_status_layout.addLayout(sensor1_h_layout)
    sensor_status_group.setLayout(sensor_status_layout)
    left_v_layout.addWidget(sensor_status_group)

    left_v_layout.addStretch()

    # ------------------------- RIGHT PANEL (Data Display) -------------------------
    data_group = QGroupBox("FFT Data")
    data_group.setStyleSheet("font-size: 18px;")
    data_layout = QVBoxLayout()
    data_layout.setSpacing(5)
    data_layout.setContentsMargins(5, 5, 5, 5)

    data_display = QTextEdit()
    data_display.setReadOnly(True)
    data_display.setStyleSheet("background-color: #ffffff; font-size: 14px;")
    data_layout.addWidget(data_display)
    data_group.setLayout(data_layout)

    # Assemble the panels with stretch factors (left:1, right:2)
    main_h_layout.addLayout(left_v_layout, 1)
    main_h_layout.addWidget(data_group, 2)
    layout.addLayout(main_h_layout)

    def UDPSendData(cmd):
        """
        Sends a UDP command to the sensor.
        Retrieves the sensor IP and port from the GUI widgets.
        """
        ip = sensor_ip_input_1.text().strip()
        port = sensor_ip_port_input.value()
        global udp_socket
        if udp_socket is None:
            output_box.append("UDP socket not initialized!")
            return
        try:
            # Send command using the existing socket
            udp_socket.sendto(cmd.encode("utf-8"), (ip, port))
            #output_box.append(f"Sent UDP command: {cmd}")
        except Exception as e:
            output_box.append(f"Error sending UDP command: {e}")

    # --- Define a slot to process incoming UDP data ---
    def process_udp_message(data):
        if len(data) >= 8:  # At least two floats for header and data length
            header_length, data_length = struct.unpack("ff", data[:8])
            header_length = int(header_length)
            data_length = int(data_length)
            if len(data) >= header_length + data_length:
                header_buffer = data[:header_length]
                data_buffer = data[header_length : header_length + data_length]
                return header_buffer, data_buffer
        return None, None
    
    def log_fft_data(header_buf, data_buf, FFT_MinFrequencyIndex, FFT_MaxFrequencyIndex, FFT_FrequencyFactor):
        global SAVE_FILE, SAVE_HEADER, save_streams_count
        if not os.path.exists(SAVE_FILE):
            freq_line = ""
            for ix in range(FFT_MinFrequencyIndex, FFT_MaxFrequencyIndex + 1):
                freq_line += f"{ix * FFT_FrequencyFactor:.0f}\t"
            freq_line = freq_line.rstrip("\t") + "\n"

            if len(header_buf) >= 64:
                header_vals = struct.unpack("16f", header_buf[:64])
                header_line = "\t".join(f"{val:.2f}" for val in header_vals)
            else:
                header_line = "Header Error\n"
            SAVE_HEADER = header_line
            try:
                with open(SAVE_FILE, "w") as f:
                    f.write(freq_line)
                    #output_box.append(f"Header written to {SAVE_FILE}")
            except Exception as e:
                    output_box.append(f"Error writing header: {e}")
        # Convert the data buffer to an array of int16 values (Y values)
        YValues = np.frombuffer(data_buf, dtype=np.int16)
        line = "\t".join(str(val) for val in YValues)
        full_line = SAVE_HEADER + "\t" + line + "\n"
        try:
           with open(SAVE_FILE, "a") as f:
                f.write(full_line)
        except Exception as e:
                output_box.append(f"Error logging data: {e}")

        save_streams_count -= 1
        logged = start_logging_button.initial_count - save_streams_count
        status_display.setText(f"Measurements: {logged} / {start_logging_button.initial_count}")
        if save_streams_count <= 0:
            logging_active = False
            start_logging_button.setEnabled(True)
            output_box.append("Logging completed...!!")
            UDPSendData("-f 0")
            start_fft_button.setText("start FFT")
            output_box.append("Stopping FFT process...")
    
    def real_time_predict(header_buf, data_buf, FFT_MinFrequencyIndex, FFT_MaxFrequencyIndex, FFT_FrequencyFactor):
        global predict_buffer,MODEL_FILE,SAVE_HEADER,display_distance
        Predict = Predict_FFT()

        if not predict_buffer:
            freq_line = ""
            for ix in range(FFT_MinFrequencyIndex, FFT_MaxFrequencyIndex + 1):
                freq_line += f"{ix * FFT_FrequencyFactor:.0f}\t"
            freq_line = freq_line.rstrip("\t")

            if len(header_buf) >= 64:
                header_vals = struct.unpack("16f", header_buf[:64])
                header_line = "\t".join(f"{val:.2f}" for val in header_vals)
                display_distance = header_vals[10]
            else:
                header_line = "Header Error"
            SAVE_HEADER = header_line
            predict_buffer.append(freq_line)
        
        YValues = np.frombuffer(data_buf, dtype=np.int16)
        measurement_line = "\t".join(str(val) for val in YValues)

        full_line = SAVE_HEADER + "\t" + measurement_line + "\n"
        predict_buffer.append(full_line)

        if len(predict_buffer) - 1 >= MAX_PREDICT_MEASUREMENTS:
            temp_file_content = "\n".join(predict_buffer)

            with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as temp_file:
                temp_file.write(temp_file_content)
                temp_file_path = temp_file.name
                print(temp_file_path)
                PredictPCA = Predict.loadpredictfile(temp_file_path)
                UpdatedPredictPCA = PredictPCA.reshape(PredictPCA.shape[0], -1)
                Loadcnnmodel = Predict.loadCNNModel(MODEL_FILE)
                predictions = Loadcnnmodel.predict(UpdatedPredictPCA)
                predicted_labels = (predictions > 0.5).astype(int)

                for label in predicted_labels:
                    if label == 0:
                        message = "Wear Seat Belt..!!"
                        print(message)
                    else:
                        message = "No Presence Detected.."
                        print(message)
                    status_display.setText(f"<span style='color:red;'>{message}</span> <span style='color:black;'>Dist: {display_distance:.2f} m</span>")
            os.remove(temp_file_path)

            if MODEL_FILE:
                Loadcnnmodel = Predict.loadCNNModel(MODEL_FILE)
            else:
                output_box.append("Invalid Model or File..!!")

            predict_buffer.clear()
            output_box.append("Prediction completed; buffer cleared.")


    def process_udp_data(data):
        start_time = time.perf_counter()
        global logging_active, save_streams_count, SAVE_FILE, SAVE_HEADER, RealTimePredict,display_distance
        header_buf, data_buf = process_udp_message(data)
        FFT_MinFrequencyIndex = 293
        FFT_MaxFrequencyIndex = 377
        FFT_FrequencyFactor = (125e6 / 64) / (2**14)  # Approximately 119.2093

        if header_buf is not None:
            create_series(data_buf, header_buf)
            
            if not RealTimePredict:
                if len(header_buf) >= 64:
                    header_vals = struct.unpack("16f", header_buf[:64])
                    display_distance = header_vals[10]
                    status_display.setText(f"Dist: {display_distance:.2f} m")

            if logging_active:
                log_fft_data(header_buf, data_buf, FFT_MinFrequencyIndex, FFT_MaxFrequencyIndex, FFT_FrequencyFactor)
            
            if RealTimePredict:
                real_time_predict(header_buf, data_buf, FFT_MinFrequencyIndex, FFT_MaxFrequencyIndex, FFT_FrequencyFactor)
    
        else:
            output_box.append("Received UDP data (unparsed): " + str(data))
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # print(f"process_udp_data execution time: {elapsed_time:.6f} seconds")
    # ------------- Change FFT Window -------------
    def fft_window_width(index):
        # Check if the UDP client is active (for example, if udp_socket is not None)
        if udp_socket is not None:
            cmd = "-w " + str(index)
            UDPSendData(cmd)
            output_box.append(f"FFT window width changed; sent command: {cmd}")

    # ------------- Plot FFT Data -------------
    def Plot_fft():
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        canvas.ax = ax  # Store axis reference for later updates
        return canvas

    def create_series(data_buf, header_buf):
        """
        - DataBuffer is interpreted as Int16 samples.
        - The header is expected to contain 16 floats (64 bytes), and we assume the 5th float (index 4) is the frequency factor (Header.X_Interval).
        - X values are computed starting at FFT_MIN_FREQUENCY, using the frequency factor as increment.
        - Returns a matplotlib FigureCanvas with the plotted data.
        """
        global XValues, fft_canvas
        FFT_MIN_FREQUENCY = 35000
        FFT_MAX_FREQUENCY = 45000
        Y_MIN = 0
        Y_MAX = 1200
        # Compute the number of samples (each sample is 2 bytes)
        DataLength = len(data_buf) // 2

        # Convert the raw data buffer into an array of int16 values (Y values)
        YValues = np.frombuffer(data_buf, dtype=np.int16)

        # Parse header to extract frequency factor (Header.X_Interval)
        if len(header_buf) >= 64:
            header_vals = struct.unpack("16f", header_buf[:64])
            # Assume header_vals[4] holds the frequency factor.
            x_interval = header_vals[4]
        else:
            x_interval = 1.0  # fallback in case header is incomplete

        # Create or update XValues if needed
        if XValues is None or len(XValues) != DataLength:
            XValues = np.array(
                [FFT_MIN_FREQUENCY + i * x_interval for i in range(DataLength)],
                dtype=np.uint16,
            )

        if fft_canvas is None:
            fft_canvas = Plot_fft()
            data_layout.addWidget(fft_canvas)
            data_layout.removeWidget(data_display)
            data_display.deleteLater()
        else:
            fft_canvas.ax.clear()

        fft_canvas.ax.plot(XValues, YValues, linestyle="-", marker="o")
        fft_canvas.ax.set_xlabel("Frequency (Hz)")
        fft_canvas.ax.set_ylabel("Amplitude")
        fft_canvas.ax.grid(True)
        fft_canvas.ax.set_xlim(FFT_MIN_FREQUENCY, FFT_MAX_FREQUENCY)
        fft_canvas.ax.set_ylim(Y_MIN, Y_MAX)
        fft_canvas.draw()

    # ------------- UDP PING HELPER FUNCTION -------------
    def ping_sensor_icmp(ip, timeout=0.05):
        """Send an ICMP ping with a 50ms timeout (0.05 sec) and return True if the sensor responds."""
        response = ping(ip, timeout=timeout)
        return response is not None

    # ------------------------- DStart Sensor -------------------------
    def check_sensor_connection():
        global sensor_active
        ip = sensor_ip_input_1.text().strip()
        output_box.append("Checking sensor connection...")
        if ping_sensor_icmp(ip):
            sensor1_indicator.setStyleSheet(
                "background-color: green; border: 1px solid black;"
            )
            output_box.append("Sensor Active..!!")
            sensor_active = True
        else:
            sensor1_indicator.setStyleSheet(
                "background-color: red; border: 1px solid black;"
            )
            output_box.append("Cannot Find Any Sensor..!!")
            sensor_active = False

    def ssh_send_data(
        ip, command, username="root", password="root", port=22, remote_dir="/root/iic"
    ):
        """
        Connects to the given IP address via SSH and runs a command in the specified remote directory.
        Returns a tuple (output, errors).
        """
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(ip, port=port, username=username, password=password)
            # Change to the remote directory and execute the command
            full_command = f"(cd {remote_dir}; {command})"
            stdin, stdout, stderr = client.exec_command(full_command)
            output = stdout.read().decode()
            errors = stderr.read().decode()
            client.close()
            return output, errors
        except Exception as e:
            return None, str(e)

    def start_sensor():
        global sensor_active
        if sensor_active:
            start_sensor_button.setText("stop Sensor")
            ip = sensor_ip_input_1.text().strip()
            out, err = ssh_send_data(ip, "pkill iic")
            if err:
                output_box.append(f"Error stopping sensor: {err}")
            else:
                output_box.append("Initializing...")

            output_box.append("Starting sensor...")

            out, err = ssh_send_data(ip, "./iic")
            if err:
                output_box.append(f"Error starting sensor: {err}")
            else:
                output_box.append("Sensor started successfully.")
        else:
            start_sensor_button.setText("start Sensor")
            output_box.append("Sensor Not Active")

    def start_udp_thread(start):
        global udp_thread_running, udp_thread
        udp_thread_running = start
        if start:
            output_box.append("UDP client started.")
            udp_thread.data_received.connect(process_udp_data)
            udp_thread.start()
        else:
            output_box.append("UDP client stopped.")

    def start_client():
        global udp_thread, udp_socket, client_active, sensor_active

        if sensor_active:
            # Query the button's state explicitly:
            if start_client_button.isChecked():
                client_active = True
                start_client_button.setText("stop client")
                output_box.append("Starting UDP client...")
                # Get the local port from the widget
                local_port = local_ip_port_input.value()
                udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                udp_socket.bind(("0.0.0.0", local_port))
                # Create and start the UDP thread
                udp_thread = UDPClientThread(udp_socket)
                udp_thread.data_received.connect(process_udp_data)
                start_udp_thread(True)
            else:
                start_client_button.setText("start client")
                output_box.append("Stopping UDP client...")
                client_active = False
                if udp_socket is not None:
                    udp_socket.close()
                    udp_socket = None
                start_udp_thread(False)
        else:
            output_box.append("Sensor Not Active")

    def start_fft(state):
        global fft_running, client_active, sensor_active

        if sensor_active:
            fft_running = state
            if client_active:
                if fft_running:
                    start_fft_button.setText("stop FFT")
                    UDPSendData("-f 1")
                    output_box.append("Started FFT process...")
                else:
                    start_fft_button.setText("start FFT")
                    output_box.append("Stopping FFT process...")
                    UDPSendData("-f 0")
            else:
                output_box.append("UDP Client not Started..!!")
        else:
            output_box.append("Sensor Not Active")

    def start_logging():
        global sensor_active, logging_active, save_streams_count, SAVE_FILE
        if sensor_active:
            if not logging_active:
                logging_active = True

                cwd = os.getcwd()
                logs_folder = os.path.join(cwd, "SensorUIApp - Logs")
                if not os.path.exists(logs_folder):
                    os.makedirs(logs_folder)
                now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"FFT_readings_{now_str}.txt"

                SAVE_FILE = os.path.join(logs_folder, filename)

                # Disable the button so the user cannot press it again until logging is done.
                start_logging_button.setEnabled(False)
                # Use the value from measurements_input as the counter
                save_streams_count = measurements_input.value()
                start_logging_button.initial_count = measurements_input.value()
                output_box.append(
                    f"Logging started. Will log {save_streams_count} data sets."
                )
            else:
                output_box.append("Logging is already in progress.")
        else:
            output_box.append("Sensor Not Active")

    def select_model():
        global MODEL_FILE
        MODEL_FILE, _ = QFileDialog.getOpenFileName(
            None, "Select Model File", os.getcwd(), "H5 Files (*.h5)"
        )
        if MODEL_FILE:
            output_box.append(f"Model selected: {MODEL_FILE}")
        else:
            output_box.append("No model file selected.")

    def start_predict(state):
        global MODEL_FILE, client_active, ML_running, sensor_active, RealTimePredict

        if sensor_active:
            ML_running = state

            if client_active:
                if ML_running:
                    if MODEL_FILE:
                        output_box.append(f"Model selected: {MODEL_FILE}")
                        start_predict_button.setText("Stop Predict")
                        UDPSendData("-f 1")
                        output_box.append("Started Predict process...")
                        RealTimePredict = True
                    else:
                        start_predict_button.setText("Start Predict")
                        output_box.append("No Predicting Model Selected...")
                        UDPSendData("-f 0")
                        RealTimePredict = False
                else:
                    start_predict_button.setText("Start Predict")
                    output_box.append("Stopping Predict process...")
                    UDPSendData("-f 0")
                    RealTimePredict = False
            else:
                output_box.append("UDP Client not Started..!!")
        else:
            output_box.append("Sensor Not Active")

    check_connection_button.clicked.connect(check_sensor_connection)
    start_sensor_button.clicked.connect(start_sensor)
    start_client_button.setCheckable(True)
    start_client_button.toggled.connect(lambda state: start_client())
    start_fft_button.setCheckable(True)
    start_fft_button.toggled.connect(lambda state: start_fft(state))
    fftwindow_input.currentIndexChanged.connect(fft_window_width)
    start_logging_button.clicked.connect(start_logging)
    select_model_button.clicked.connect(select_model)
    start_predict_button.setCheckable(True)
    start_predict_button.toggled.connect(lambda state: start_predict(state))
