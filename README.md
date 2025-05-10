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


