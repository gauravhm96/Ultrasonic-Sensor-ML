from PyQt5.QtWidgets import (
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QTextEdit,
    QFrame,
    QSizePolicy
)
from PyQt5.QtCore import Qt,QThread, pyqtSignal
import socket
from ping3 import ping 
import paramiko


# Global variables for UDP client management:
global udp_thread
udp_thread = None
udp_socket = None
client_active = False
udp_thread_running = False

class UDPClientThread(QThread):
    # Signal to emit incoming UDP data (as bytes)
    data_received = pyqtSignal(bytes)

    def __init__(self, local_port, parent=None):
        super().__init__(parent)
        self.local_port = local_port
        self.running = True

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.local_port))
        sock.settimeout(1.0)  # Check every second if we should exit
        while self.running:
            try:
                data, addr = sock.recvfrom(4096)
                self.data_received.emit(data)
            except socket.timeout:
                continue
            except Exception as e:
                print("UDP error:", e)
                break
        sock.close()

    def stop(self):
        self.running = False
        self.wait()




def connect_to_red_pitaya(layout, output_box):
    """
    Sets up a GUI layout for connecting to and interacting with a Red Pitaya sensor.
    """

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
    # Set column stretch so that column 1 (inputs) gets more space
    udp_layout.setColumnStretch(0, 1)
    udp_layout.setColumnStretch(1, 2)

    sensor_ip_label_1 = QLabel("Sensor IP address 1:")
    sensor_ip_label_1.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    sensor_ip_input_1 = QLineEdit("192.168.128.1")

    sensor_ip_port_label = QLabel("Sensor IP port:")
    sensor_ip_port_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    sensor_ip_port_input = QSpinBox()
    sensor_ip_port_input.setRange(1, 65535)
    sensor_ip_port_input.setValue(61231)

    local_ip_label_1 = QLabel("Local IP address 1:")
    local_ip_label_1.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    local_ip_input_1 = QLineEdit("127.0.0.1")

    local_ip_port_label = QLabel("Local IP port:")
    local_ip_port_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
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

    wave_label = QLabel("set wave:")
    wave_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    wave_input = QSpinBox()
    wave_input.setRange(1, 99999)
    wave_input.setValue(8192)

    measurements_label = QLabel("measurements:")
    measurements_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    measurements_input = QSpinBox()
    measurements_input.setValue(5)

    measure_count_label = QLabel("measure count:")
    measure_count_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    measure_count_value = QLabel("0")

    # Buttons in a horizontal layout (all in one row)
    button_h_layout = QHBoxLayout()
    button_h_layout.setSpacing(10)
    button_h_layout.setContentsMargins(0, 0, 0, 0)
    button_h_layout.setAlignment(Qt.AlignLeft)

    start_fft_button = QPushButton("start FFT")
    start_fft_button.setStyleSheet("font-size: 18px; padding: 5px;")
    start_fft_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    start_logging_button = QPushButton("start logging")
    start_logging_button.setStyleSheet("font-size: 18px; padding: 5px;")
    start_logging_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    clear_rx_button = QPushButton("clear RX")
    clear_rx_button.setStyleSheet("font-size: 18px; padding: 5px;")
    clear_rx_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    button_h_layout.addWidget(start_fft_button)
    button_h_layout.addWidget(start_logging_button)
    button_h_layout.addWidget(clear_rx_button)

    sr_layout.addWidget(wave_label, 0, 0)
    sr_layout.addWidget(wave_input, 0, 1)
    sr_layout.addLayout(button_h_layout, 1, 0, 1, 2)
    sr_layout.addWidget(measurements_label, 2, 0)
    sr_layout.addWidget(measurements_input, 2, 1)
    sr_layout.addWidget(measure_count_label, 3, 0)
    sr_layout.addWidget(measure_count_value, 3, 1)

    sr_group.setLayout(sr_layout)
    left_v_layout.addWidget(sr_group)

    # ------------------------- BOTTOM LEFT SENSOR STATUS (VERTICAL) -------------------------
    sensor_status_group = QGroupBox("Sensor Status")
    sensor_status_group.setStyleSheet("font-size: 18px;")
    sensor_status_layout = QVBoxLayout()
    sensor_status_layout.setSpacing(5)
    sensor_status_layout.setContentsMargins(5, 5, 5, 5)

    # Only Sensor 1 is displayed
    sensor1_h_layout = QHBoxLayout()
    sensor1_h_layout.setSpacing(10)
    sensor1_h_layout.setContentsMargins(0, 0, 0, 0)
    sensor1_h_layout.setAlignment(Qt.AlignLeft)

    sensor1_label = QLabel("Sensor Status:")
    sensor1_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    sensor1_indicator = QFrame()
    sensor1_indicator.setFixedSize(20, 20)
    sensor1_indicator.setStyleSheet("background-color: red; border: 1px solid black;")

    sensor1_h_layout.addWidget(sensor1_label)
    sensor1_h_layout.addWidget(sensor1_indicator)
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

    # --- Define a slot to process incoming UDP data ---
    def process_udp_data(data):
        # Update your GUI or log the received data
        output_box.append("Received UDP data: " + str(data))




    # ------------- UDP PING HELPER FUNCTION -------------
    def ping_sensor_icmp(ip, timeout=0.05):
        """Send an ICMP ping with a 50ms timeout (0.05 sec) and return True if the sensor responds."""
        response = ping(ip, timeout=timeout)
        return response is not None
    # ------------------------- DStart Sensor -------------------------
    def check_sensor_connection():
        ip = sensor_ip_input_1.text().strip()
        output_box.append("Checking sensor connection...")
        if ping_sensor_icmp(ip):
            sensor1_indicator.setStyleSheet("background-color: green; border: 1px solid black;")
            output_box.append("Sensor Active..!!")
        else:
            sensor1_indicator.setStyleSheet("background-color: red; border: 1px solid black;")
            output_box.append("Cannot Find Any Sensor..!!")

    def ssh_send_data(ip, command, username="root", password="root", port=22, remote_dir="/root/iic"):
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

    def start_udp_thread(start=True):
        """
        Starts or stops the UDP listening thread.
        Here you would start a QThread or similar that listens on udp_socket.
        For now, we simply log the state.
        """
        global udp_thread_running
        udp_thread_running = start
        if start:
            output_box.append("UDP client started.")
            # TODO: Start your UDP listening thread here.
        else:
            output_box.append("UDP client stopped.")

    def start_client():
        global udp_socket, client_active
        state = start_client_button.isChecked()
        client_active = state
        # Retrieve IP and port values from the input widgets:
        ip = sensor_ip_input_1.text().strip()
        port = sensor_ip_port_input.value()
        local_port = local_ip_port_input.value()
        local_endpoint = ('0.0.0.0', local_port)
        
        # If there's an existing socket, close it:
        try:
            if udp_socket is not None:
                udp_socket.close()
        except Exception:
            pass

        # Create a new UDP socket bound to the local endpoint:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.bind(local_endpoint)
        
        # Check the state of the checkable button:
        client_active = start_client_button.isChecked()
        
        if client_active:
            start_client_button.setText("stop client")
            output_box.append("Starting UDP client...")
            start_udp_thread(True)
        else:
            start_client_button.setText("start client")
            output_box.append("Stopping UDP client...")
            start_udp_thread(False)

    def start_fft():
        output_box.append("Starting FFT process (dummy).")

    def start_logging():
        output_box.append("Starting logging (dummy).")

    def clear_rx():
        output_box.append("Clearing RX buffer (dummy).")

    check_connection_button.clicked.connect(check_sensor_connection)
    start_sensor_button.clicked.connect(start_sensor)
    start_client_button.setCheckable(True)
    start_client_button.toggled.connect(lambda state: start_client())
    start_fft_button.clicked.connect(start_fft)
    start_logging_button.clicked.connect(start_logging)
    clear_rx_button.clicked.connect(clear_rx)
