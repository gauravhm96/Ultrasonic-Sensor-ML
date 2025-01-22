# signal_processor.py

import pandas as pd
import numpy as np
from SignalProcess import SignalProcessor



if __name__ == "__main__":
    file_path = 'C:\\@DevDocs\\Projects\\Mine\\New folder\\Ultrasonic-Sensor-ML\\UltrasonicSensorApp\\Raw_Data\\adc_38.txt'

    ProcessSignal = SignalProcessor()
    ProcessSignal.set_file_path(file_path)
    data = ProcessSignal.load_signal_data()
    ProcessSignal.analyze_raw_signals()


    
    