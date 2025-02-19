from PyQt5.QtWidgets import (
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QCheckBox,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QTextEdit,
    QFrame,
    QSizePolicy
)
from PyQt5.QtCore import Qt

def connect_to_red_pitaya(layout, output_box):
    """
    Sets up a GUI layout for connecting to and interacting with a Red Pitaya sensor.
    """

    # -- Main horizontal layout to hold the left config panel and right data display --
    main_h_layout = QHBoxLayout()
    main_h_layout.setSpacing(10)
    main_h_layout.setContentsMargins(0, 0, 0, 0)
    main_h_layout.setAlignment(Qt.AlignLeft)

    # ------------------------- LEFT PANEL -------------------------
    left_v_layout = QVBoxLayout()

    #
    # 1) UDP CLIENT CONFIG GROUP
    #
    udp_group = QGroupBox("UDP client config")
    udp_group.setStyleSheet("font-size: 18px; font-weight: normal;")
    udp_layout = QGridLayout()

    sensor_ip_label_1 = QLabel("Sensor IP address 1:")
    sensor_ip_label_1.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    sensor_ip_input_1 = QLineEdit("192.168.128.1")

    sensor_ip_label_2 = QLabel("Sensor IP address 2:")
    sensor_ip_label_2.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    sensor_ip_input_2 = QLineEdit("192.168.128.2")

    sensor_ip_port_label = QLabel("Sensor IP port:")
    sensor_ip_port_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    
    sensor_ip_port_input = QSpinBox()
    sensor_ip_port_input.setRange(1, 65535)
    sensor_ip_port_input.setValue(5001)

    local_ip_label_1 = QLabel("Local IP address 1:")
    local_ip_label_1.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    local_ip_input_1 = QLineEdit("127.0.0.1")

    local_ip_label_2 = QLabel("Local IP address 2:")
    local_ip_label_2.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    local_ip_input_2 = QLineEdit("127.0.0.2")

    local_ip_port_label = QLabel("Local IP port:")
    local_ip_port_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    local_ip_port_input = QSpinBox()
    local_ip_port_input.setRange(1, 65535)
    local_ip_port_input.setValue(5002)

    check_connection_button = QPushButton("check sensor connection")
    check_connection_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    check_connection_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    # Place these widgets in a grid
    udp_layout.addWidget(sensor_ip_label_1,     0, 0)
    udp_layout.addWidget(sensor_ip_input_1,     0, 1)
    udp_layout.addWidget(sensor_ip_label_2,     1, 0)
    udp_layout.addWidget(sensor_ip_input_2,     1, 1)
    udp_layout.addWidget(sensor_ip_port_label,  2, 0)
    udp_layout.addWidget(sensor_ip_port_input,  2, 1)
    udp_layout.addWidget(local_ip_label_1,      3, 0)
    udp_layout.addWidget(local_ip_input_1,      3, 1)
    udp_layout.addWidget(local_ip_label_2,      4, 0)
    udp_layout.addWidget(local_ip_input_2,      4, 1)
    udp_layout.addWidget(local_ip_port_label,   5, 0)
    udp_layout.addWidget(local_ip_port_input,   5, 1)
    udp_layout.addWidget(check_connection_button, 6, 0, 1, 2)

    udp_group.setLayout(udp_layout)
    left_v_layout.addWidget(udp_group)

    #
    # 2) SEND / RECEIVE GROUP
    #
    sr_group = QGroupBox("Send / Receive")
    sr_group.setStyleSheet("font-size: 18px; font-weight: normal;")
    sr_layout = QGridLayout()

    sensor_sw_label = QLabel("Sensor SW Version:")
    sensor_sw_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    sensor_sw_input = QLabel("1.10")  # Could be a QLineEdit if you want to edit it

    wave_label = QLabel("set wave:")
    wave_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    wave_input = QSpinBox()
    wave_input.setRange(1, 99999)
    wave_input.setValue(8192)

    save_config_checkbox = QCheckBox("save config")
    save_header_checkbox = QCheckBox("save header infos")
    save_data_checkbox = QCheckBox("save data")

    measurements_label = QLabel("measurements:")
    measurements_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    measurements_input = QSpinBox()
    measurements_input.setValue(5)

    measure_count_label = QLabel("measure count:")
    measure_count_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    measure_count_value = QLabel("0")  # Could be updated dynamically
    

    # Buttons in a single row
    button_h_layout = QHBoxLayout()
    button_h_layout.setSpacing(10)
    button_h_layout.setContentsMargins(0, 0, 0, 0)
    button_h_layout.setAlignment(Qt.AlignLeft)
    
    send_cmd_button = QPushButton("send cmd")
    send_cmd_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    send_cmd_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    start_fft_button = QPushButton("start FFT")
    start_fft_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    start_fft_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    start_logging_button = QPushButton("start logging")
    start_logging_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    start_logging_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    
    clear_rx_button = QPushButton("clear RX")
    clear_rx_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    clear_rx_button.setFixedWidth(250)  # Set a fixed width for the buttons
    clear_rx_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    button_h_layout.addWidget(send_cmd_button)
    button_h_layout.addWidget(start_fft_button)
    button_h_layout.addWidget(start_logging_button)
    button_h_layout.addWidget(clear_rx_button)

    # Row 0: Sensor SW Version / wave
    sr_layout.addWidget(sensor_sw_label, 0, 0)
    sr_layout.addWidget(sensor_sw_input, 0, 1)
    sr_layout.addWidget(wave_label,      1, 0)
    sr_layout.addWidget(wave_input,      1, 1)

    # Row 2: The horizontal layout with the buttons (spanning 2 columns)
    sr_layout.addLayout(button_h_layout, 2, 0, 1, 2)

    # Row 3: The checkboxes (all in one row if you prefer)
    checkbox_h_layout = QHBoxLayout()
    checkbox_h_layout.setSpacing(10)
    checkbox_h_layout.setContentsMargins(0, 0, 0, 0)
    checkbox_h_layout.setAlignment(Qt.AlignLeft)
    
    checkbox_h_layout.addWidget(save_config_checkbox)
    checkbox_h_layout.addWidget(save_header_checkbox)
    checkbox_h_layout.addWidget(save_data_checkbox)
    sr_layout.addLayout(checkbox_h_layout, 3, 0, 1, 2)

    # Row 4: measurements label + spinbox
    sr_layout.addWidget(measurements_label, 4, 0)
    sr_layout.addWidget(measurements_input, 4, 1)

    # Row 5: measure count label + value
    sr_layout.addWidget(measure_count_label, 5, 0)
    sr_layout.addWidget(measure_count_value, 5, 1)

    sr_group.setLayout(sr_layout)
    left_v_layout.addWidget(sr_group)

    # ------------------------- BOTTOM LEFT SENSOR STATUS (VERTICAL) -------------------------
    sensor_status_group = QGroupBox("Sensor Status")
    sensor_status_group.setStyleSheet("font-size: 18px;font-weight: normal;")
    sensor_status_layout = QVBoxLayout()

    # Sensor 1
    sensor1_h_layout = QHBoxLayout()
    sensor1_h_layout.setSpacing(10)
    sensor1_h_layout.setContentsMargins(0, 0, 0, 0)
    sensor1_h_layout.setAlignment(Qt.AlignLeft)
    
    sensor1_label = QLabel("Sensor 1:")
    sensor1_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    sensor1_indicator = QFrame()
    sensor1_indicator.setFixedSize(20, 20)
    sensor1_indicator.setStyleSheet("background-color: red; border: 1px solid black;")
    sensor1_h_layout.addWidget(sensor1_label)
    sensor1_h_layout.addWidget(sensor1_indicator)

    # Sensor 2
    sensor2_h_layout = QHBoxLayout()
    sensor2_h_layout.setSpacing(10)
    sensor2_h_layout.setContentsMargins(0, 0, 0, 0)
    sensor2_h_layout.setAlignment(Qt.AlignLeft)
    
    sensor2_label = QLabel("Sensor 2:")
    sensor2_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    sensor2_indicator = QFrame()
    sensor2_indicator.setFixedSize(20, 20)
    sensor2_indicator.setStyleSheet("background-color: red; border: 1px solid black;")
    sensor2_h_layout.addWidget(sensor2_label)
    sensor2_h_layout.addWidget(sensor2_indicator)

    sensor_status_layout.addLayout(sensor1_h_layout)
    sensor_status_layout.addLayout(sensor2_h_layout)

    sensor_status_group.setLayout(sensor_status_layout)
    left_v_layout.addWidget(sensor_status_group)

    # Add any final stretch or spacing on the left panel
    left_v_layout.addStretch()

    # ------------------------- RIGHT PANEL (Data Display) -------------------------
    data_group = QGroupBox("FFT Data")
    data_group.setStyleSheet("font-size: 18px;font-weight: normal;")
    data_layout = QVBoxLayout()

    # Placeholder for the data display area
    data_display = QTextEdit()
    data_display.setReadOnly(True)
    data_display.setStyleSheet("background-color: #ffffff; font-size: 14px;")

    data_layout.addWidget(data_display)
    data_group.setLayout(data_layout)

    # ------------------------- ASSEMBLE THE PANELS -------------------------
    main_h_layout.addLayout(left_v_layout)
    main_h_layout.addWidget(data_group)
    layout.addLayout(main_h_layout)

    # ------------------------- DUMMY FUNCTIONS / SIGNALS -------------------------
    def check_sensor_connection():
        # In a real application, you’d attempt to connect to the Red Pitaya here.
        output_box.append("Checking sensor connection... (dummy)")

        # For demonstration, let's say we successfully connect to both sensors:
        sensor1_indicator.setStyleSheet("background-color: green; border: 1px solid black;")
        sensor2_indicator.setStyleSheet("background-color: green; border: 1px solid black;")

        output_box.append("Sensor connection check complete (dummy).")

    def send_cmd():
        output_box.append("Sending command to Red Pitaya (dummy).")

    def start_fft():
        output_box.append("Starting FFT process (dummy).")

    def start_logging():
        output_box.append("Starting logging (dummy).")

    def clear_rx():
        output_box.append("Clearing RX buffer (dummy).")

    # Connect the buttons to their respective dummy functions
    check_connection_button.clicked.connect(check_sensor_connection)
    send_cmd_button.clicked.connect(send_cmd)
    start_fft_button.clicked.connect(start_fft)
    start_logging_button.clicked.connect(start_logging)
    clear_rx_button.clicked.connect(clear_rx)
