# In-Vehicle Passenger and Object Detection Using Ultrasonic Sensors with Machine Learning Algorithms

**About Ultrasonic Application**

- Complex Signal Analysis For Object Differentiation
- Complex Signal Analysis For Object Detection
- Create Machine Learning Models for Object Differentiation & Object Detection using Large Datasets
- Communicate with Red Pitaya Sensor
  - Visualize Real-Time FFT data from Ultrasonic Sensor
  - Log Measurements
  - Use Machine Learning Models & Predict Object Class from the Ultrasonic Sensor
  - Predict Object Distance

<div align="center">
  <img width="250" alt="UltrasonicApp" src="https://github.com/user-attachments/assets/7c089a4a-d250-4524-804e-f24797389629" />
</div>

#### [Link to the Repository --> Ultrasonic Application](https://github.com/gauravhm96/Ultrasonic-Sensor-ML/tree/main/UltrasonicSensorApp)

## Table of Contents 
- [Overview](https://github.com/gauravhm96/Red-Pitaya-Ultrasonic-Sensor-Detection-DIfferentiation/blob/main/README.md#overview)
- [Requirement Analysis](https://github.com/gauravhm96/Red-Pitaya-Ultrasonic-Sensor-Detection-DIfferentiation/blob/main/README.md#requirement-analysis)
- [Use Cases of the Prototype](https://github.com/gauravhm96/Red-Pitaya-Ultrasonic-Sensor-Detection-DIfferentiation/blob/main/README.md#use-cases-of-the-prototype)
- [Implementation](https://github.com/gauravhm96/Red-Pitaya-Ultrasonic-Sensor-Detection-DIfferentiation/edit/main/README.md#implementation---environment-setup)

 
## Overview

This Project describes the design and development of an advanced machine learning–based application that enhances the performance of ultrasonic sensors for integration into vehicle passive safety systems. At the core of this project, a robust framework was implemented to process complex ultrasonic signals in real-time. 

The application integrates advanced signal processing techniques—including frequency and time domain analyses—and predictive analytics powered by neural networks and CNNs to enable real-time communication with the sensor system and comprehensive data
evaluation. It actively captures sensor data, performs complex signal computations, and applies these predictive models to accurately assess the vehicle cabin’s status, identify objects, classify occupants, and determine object distances.

Furthermore, the application captures raw signals from the sensor system, which can be used to study signal properties and characteristics further by employing complex signal analysis algorithms. The system also collects large datasets that can be processed to build predictive models. These models, developed under various conditions and use cases, enable the user to perform real-time predictions based on their performance from the ultrasonic sensor data.

This work addresses critical challenges related to occupant safety and object detection,contributing to reduced accident risks and improved overall vehicle safety. By integrating machine learning, we enhance the functionality of ultrasonic sensors and pave the way for future innovations in automotive safety systems. This project showcases how artificial
intelligence can transform traditional sensor technologies into proactive safety solutions that support real-time decision-making in complex driving environments. Additionally, the application was developed by extensively studying current technologies in in-cabin safety systems and the automotive safety standards established by Euro NCAP, highlighting its relevance and potential impact in this field.


## Requirement Analysis

This application serves as both a proof of concept and a practical solution, demonstrating how advanced machine learning can enhance in-cabin safety and vehicle intelligence. Using ultrasonic sensor data, it identifies and differentiates objects and passenger presence, aligning with current trends in automotive safety.

By leveraging models like _**CNNs**_ and _**Logistic Regression**_, it enables accurate detection of passenger and object status, supporting the development of proactive safety features.

Furthermore, the approach offers a foundation for integration within _**Vehicle-to-Everything (V2X) technology**_, where critical vehicle and passenger information is transmitted to _**Road Side Units (RSUs)**_ or _**Intelligent Transport Systems (ITS)**_. This enables real-time decisionmaking that supports safety, traffic management, and autonomous driving, further advancing the goals of intelligent transportation networks and enhanced ADAS capabilities.

## Use Cases of the Prototype

The figure provides an overview of the system architecture for the developed application. The outlined use cases represent scenarios that are implemented and tested to verify the system’s capabilities.


<div align="center">
  <img width="800" alt="Use case diagram" src="https://github.com/gauravhm96/Latex-Report/blob/main/Figures/Use%20case%20diagram.jpeg" />
</div>


## Implementation - Environment Setup

The application primarily involves building machine learning algorithms with a focus on object differentiation and detection, real-time communication with the sensor system, and performing predictions using the developed model.The application is driven by a user interface that handles various tasks efficiently. It includes several stages or milestones, each of which plays a critical role in ensuring the success of the subsequent stages. These stages enhance the application reliability, scalability, and overall performance.


<div align="center">
  <img width="650" alt="Block Diagram for Implementation" src="https://github.com/gauravhm96/Latex-Report/blob/main/Figures/MLModel.png" />
</div>


### Hardware Requirements​

- Red Pitaya STEMlab 125-14​
- SRF02 Ultrasonic Range Finder​
- Dashboard Mount ​

![image](https://github.com/user-attachments/assets/ca7e8b7c-eebf-4fda-9f26-4dada5fd76c8)
![image](https://github.com/user-attachments/assets/7e4155cd-6b29-4748-b428-c0adcec17e20)
![image](https://github.com/user-attachments/assets/1806613c-17c5-48e5-9e06-ee3a034364b1)




### Software Requirements​

- Visual Studio 2022​
- Visual Studio Code (VS Code)​
- Python 3.12.1

![image](https://github.com/user-attachments/assets/0fc22e9f-8506-46b2-a8de-fc8a9233c121)

![image](https://github.com/user-attachments/assets/69d23767-e7e6-483e-946e-5ddfc98a9420)


### Ultrasonic Sensor App (GUI)

#### Object Differentiation 
  - Signal Analysis (Feature Extraction)
  - Train Models
  - Model Prediction on Data

<div align="center">
  <img width="850" alt="GUI Object Differentiation" src="https://github.com/gauravhm96/Latex-Report/blob/main/Figures/GUI_ObjDif.png" />
</div>


The Message sequence diagram for Object differentiation algorithm to build ML model is as shown in the figure

<div align="center">
  <img width="850" alt="GUI Object Differentiation" src="https://github.com/gauravhm96/Latex-Report/blob/main/Figures/MessageSequenceObjectDifferentiation.png" />
</div>

#### Object Detection 

- Signal Analysis (Feature Extraction)
- Train Models
- Model Prediction on Data
- Find ToF, by prediction of real first echo & find the distance of the object


<div align="center">
  <img width="850" alt="GUI Object Detection" src="https://github.com/gauravhm96/Latex-Report/blob/main/Figures/GUI_ObjDet.png" />
</div>


The Message sequence diagram for Object detection algorithm to build ML model is as shown in the figure


<div align="center">
  <img width="850" alt="Msg Sequence Diagram Object Detection" src="https://github.com/gauravhm96/Latex-Report/blob/main/Figures/MessageSequenceObjectDetection.png" />
</div>

#### Real Time Sensor Communication 

 - Real-Time Prediction using ML Model
 - Selection of Different ML Models and Analyze the Model Performance
 - Logging

<div align="center">
  <img width="850" alt="GUI Red Pitaya Communication" src="https://github.com/gauravhm96/Latex-Report/blob/main/Figures/GUIConnectCommunication.png" />
</div>

The Message sequence diagram for Real-Time sensor Communication and Real time prediction is as shown in the figure

<div align="center">
  <img width="850" alt="Msg Seq Red Pitaya Communication" src="https://github.com/gauravhm96/Latex-Report/blob/main/Figures/MessageSequenceRedPitaya.png" />
</div>

- Check sensor availability via ICMP Request
- Request ssh to access remotely into Red Pitaya
- Kill all process and initiate I2C communication
- Start UDP thread in Ultrasoninc App
- Start sending command
   - UDPSendData("-f 1")
   - UDPSendData("-f 0")


<div align="center">
  <img width="850" alt="Msg Seq Red Pitaya Communication(1)" src="https://github.com/gauravhm96/Latex-Report/blob/main/Figures/MessageSequencesensorconnect.png" />
</div>


The Flow chart for Establishing connection with Red-Pitaya Sensor 

<div align="center">
  <img width="300" alt="Flow-Chart Red Pitaya Communication" src="https://github.com/gauravhm96/Latex-Report/blob/main/Figures/Flowchartsensorconnect.png" />
</div>

#### About​
 -	This Section is to know the features of the application, and some information about version source code.

<div align="center">
  <img width="850" alt="Flow-Chart Red Pitaya Communication" src="https://github.com/gauravhm96/Latex-Report/blob/main/Figures/GUI_About.png" />
</div>













