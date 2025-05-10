# TORCS Car Racing Controller

## Overview

This project involves creating a car racing controller for the **TORCS** (The Open Racing Car Simulator) framework. The controller is designed to race on different tracks and compete against other cars, focusing on performance metrics like speed, obstacle avoidance, and track following. The controller perceives the environment through **telemetry** data provided by sensors in the TORCS simulation. The objective is to design an autonomous controller capable of racing successfully using the provided telemetry.

The client is built in **Python** and uses a telemetry model to receive sensor data and act accordingly. The project follows the guidelines from the TORCS documentation for creating a client-server architecture for the car racing simulation.

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Telemetry Implementation](#telemetry-implementation)
- [Training Methodology](#training-methodology)
- [Submission Details](#submission-details)
- [License](#license)

---

## Installation

### Prerequisites

- **Python 3.x** (Recommended version: 3.6 or later)
- **TORCS** (The Open Racing Car Simulator): [Download TORCS here](http://cs.adelaide.edu.au/~optlog/SCR2015/index.html)
- Required libraries:
  - `numpy`
  - `torch` (for machine learning model)
  - `matplotlib`
  - `pandas`
  
### Steps to Install

1. **Install Python dependencies**:

   You can install the required Python libraries by running:

   ```bash
   pip install -r requirements.txt
   
### Download TORCS:
Follow the installation instructions on the official TORCS site.

### Configure TORCS server:
Follow the instructions in the TORCS documentation to configure the src_server file.

├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── telemetry_client.py      # The Python client to control the car
├── telemetry.py             # Telemetry data processing
├── controller.py            # Logic for car control
└── utils.py                 # Helper functions and utilities


### Telemetry Implementation
The telemetry system in this project handles the data sent from the TORCS server. The sensors in the game provide data on various aspects like:

Track position (e.g., distance to track limits)
Car state (e.g., fuel level, engine RPM)
Game state (e.g., lap time, number of laps)

The Python client receives this data, processes it, and makes decisions on how to steer, accelerate, or brake based on predefined strategies.

## Sensor Data Example

{
    'track_position': 0.3,
    'speed': 120,
    'fuel': 100,
    'lap_time': 15.5,
    'engine_rpm': 3500,
    'current_gear': 3
}


### Training Methodology
For the car controller, we have implemented a machine learning-based approach using reinforcement learning techniques. The model was trained using telemetry data to learn the best actions (steer, accelerate, brake) to take under different circumstances on various tracks. The training process was based on the following key aspects:

## Speed: Maximizing speed while maintaining control.

Obstacle avoidance: Ensuring the car avoids collisions with other cars and obstacles on the track.
Track following: Keeping the car within track boundaries to avoid penalties.
The model uses deep learning techniques (e.g., neural networks) to map telemetry data to the correct control actions.





