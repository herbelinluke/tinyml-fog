# TinyML Fog Computing Demo

A distributed anomaly detection system demonstrating fog computing concepts with edge devices and a fog aggregation layer.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Edge Node 1   │    │   Edge Node 2   │    │   Edge Node N   │
│  (Arduino/Pico) │    │  (Arduino/Pico) │    │  (Simulated)    │
│                 │    │                 │    │                 │
│  HC-SR04 Sensor │    │  HC-SR04 Sensor │    │  Synthetic Data │
│        ↓        │    │        ↓        │    │        ↓        │
│  Rolling Avg    │    │  Rolling Avg    │    │  Rolling Avg    │
│  Anomaly Detect │    │  Anomaly Detect │    │  Anomaly Detect │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │ Serial              │ Serial              │ UDP
         └──────────────┬──────┴──────────────┬──────┘
                        ↓                     ↓
              ┌─────────────────────────────────────┐
              │            Fog Node                 │
              │         (Laptop/RPi)                │
              │                                     │
              │  • Aggregates multi-node data       │
              │  • Sliding window anomaly count     │
              │  • System-level alert when          │
              │    threshold exceeded               │
              └─────────────────────────────────────┘
```

## Hardware Requirements

- Arduino Nano (or compatible) with HC-SR04 ultrasonic sensor
- USB cable for serial communication
- Optional: Additional Arduino/Pico boards for multi-node setup

## Software Requirements

```bash
pip install pyserial
```

## Quick Start

### 1. Flash Arduino

Upload `sketchs/ex_HC-SRO4_reading/ex_HC-SRO4_reading.ino` to your Arduino Nano.

The sketch:
- Reads distance from HC-SR04 sensor at 10Hz
- Computes rolling average of last 10 readings
- Detects anomalies when reading deviates >30% from average
- Outputs: `distance,anomaly_flag` (e.g., `150,0` or `45,1`)

### 2. Run Fog Node with Real Hardware

```bash
# Single Arduino node
python fog_node.py --ports /dev/ttyUSB0

# Multiple Arduino nodes
python fog_node.py --ports /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyACM0
```

### 3. Run Fog Node with Simulated Nodes (No Hardware Required)

Terminal 1 - Start fog node in simulated mode:
```bash
python fog_node.py --simulated --alert-threshold 2
```

Terminal 2 - Run multi-node simulation:
```bash
python simulate_node.py --demo
```

## Scripts

### `fog_node.py` - Fog Aggregation Layer

Aggregates data from multiple edge nodes and raises system-level alerts.

```bash
python fog_node.py [OPTIONS]

Options:
  --ports           Serial ports to connect (default: /dev/ttyUSB0)
  --baud            Baud rate (default: 9600)
  --alert-threshold Number of anomalous nodes to trigger alert (default: 2)
  --time-window     Seconds to consider for anomaly aggregation (default: 5.0)
  --status-interval Dashboard update interval (default: 3.0)
  --simulated       Use UDP input from simulate_node.py
  --udp-port        UDP port for simulated mode (default: 5000)
```

### `simulate_node.py` - Virtual Sensor Nodes

Generate synthetic sensor data for testing without hardware.

```bash
# Single node output to stdout
python simulate_node.py --node-id sensor_1

# Multi-node demo with coordinated anomalies
python simulate_node.py --demo

# Custom parameters
python simulate_node.py --base-distance 150 --anomaly-rate 0.1
```

### `collect.py` - Data Collection

Collect sensor data to CSV for analysis or model training.

```bash
python collect.py --port /dev/ttyUSB0 --node-id node_1 --output data.csv --samples 1000
```

### `collect_control.py` - Labeled Data Collection

Collect labeled data for training. Press 'a' + Enter to mark next 3 seconds as anomaly.

## Anomaly Detection Logic

### Edge Level (Arduino)
1. Maintain rolling average of last 10 valid readings
2. Compare current reading against rolling average
3. Flag as anomaly if deviation > 30% or outside valid range (2-400cm)
4. Only update rolling average with non-anomalous readings

### Fog Level
1. Receive readings from all connected nodes
2. Track anomaly events in sliding time window
3. Count unique nodes reporting anomalies
4. Raise system alert when count >= threshold

## Example Output

```
[FOG] Starting Fog Node
[FOG] Alert threshold: 2 nodes in 5.0s window
[FOG] Running in simulated mode on UDP port 5000

==================================================
[FOG] Node Status Dashboard - 14:32:15
==================================================
  🟢 node_1: dist=102cm ✓ | anomalies(recent/total): 0/3
  🟢 node_2: dist=148cm ✓ | anomalies(recent/total): 0/2
  🟢 node_3: dist=195cm ⚠ | anomalies(recent/total): 1/5

  System: ✓ Normal
==================================================

============================================================
[FOG] ⚠ SYSTEM ALERT - Multiple nodes reporting anomalies!
[FOG] Affected nodes: node_1, node_2
[FOG] Threshold: 2/2 nodes in 5.0s window
============================================================
```

## Project Structure

```
tinyml-fog/
├── sketchs/
│   └── ex_HC-SRO4_reading/
│       └── ex_HC-SRO4_reading.ino  # Arduino sketch with anomaly detection
├── fog_node.py          # Fog aggregation layer
├── simulate_node.py     # Virtual node simulator
├── collect.py           # Data collection script
├── collect_control.py   # Labeled data collection
├── arduino_data.csv     # Collected sensor data
├── anomoly_data.csv     # Labeled anomaly data
└── README.md
```

## Stretch Goals

### TFLite Micro on Pico 2
- Train model using `anomoly_data.csv`
- Convert to TFLite format
- Deploy on Raspberry Pi Pico 2

### BLE Communication
- Implement BLE peripheral on Nordic nRF52840
- Update fog node to receive BLE advertisements
