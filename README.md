# TinyML Fog Computing Demo

A distributed anomaly detection system with edge ML on Raspberry Pi Pico 2 and fog-level aggregation.

## Architecture

```
HC-SR04 Sensor → Arduino Nano → Pico 2 (ML) → Computer (Fog Node)
                    UART           USB
```

- **Arduino**: Reads ultrasonic sensor, sends raw distance
- **Pico 2**: Runs velocity-based ML model, detects fast approaches
- **Fog Node**: Aggregates anomalies, triggers alerts on sustained events

## Quick Start (Simulated - No Hardware)

```bash
pip install pyserial

# Terminal 1: Fog node
python fog_node.py --simulated --alert-threshold 2

# Terminal 2: Simulated sensors
python simulate_node.py --demo
```

## Hardware Setup

### Wiring

```
HC-SR04 VCC  → Arduino 5V
HC-SR04 GND  → Arduino GND
HC-SR04 TRIG → Arduino D8
HC-SR04 ECHO → Arduino D9
Arduino TX   → Pico GP1 (pin 2)
Arduino GND  → Pico GND (pin 38)
Pico USB     → Computer
```

### Flash Arduino

Upload `sketchs/arduino_sensor_node/arduino_sensor_node.ino` to Arduino Nano.

### Build & Flash Pico 2

```bash
# Install Pico SDK (one time)
sudo apt install cmake gcc-arm-none-eabi libnewlib-arm-none-eabi build-essential
git clone https://github.com/raspberrypi/pico-sdk.git ~/pico-sdk
cd ~/pico-sdk && git submodule update --init
export PICO_SDK_PATH=~/pico-sdk

# Build
cd pico_ml_node
mkdir build && cd build
cmake .. -DPICO_BOARD=pico2
make -j4

# Flash: Hold BOOTSEL, plug in Pico, copy .uf2 to drive
cp pico_ml_node.uf2 /media/$USER/RP2350/
```

### Run Fog Node

```bash
python fog_node.py --ports /dev/ttyACM1 --min-anomalies 3
```

## Training Your Own Model

### 1. Collect Data

```bash
python collect_training_data.py
```
- Press `n` → collect NORMAL data (walking around)
- Press `a` → collect ANOMALY data (rushing toward sensor)
- Press `q` → save and quit

### 2. Train Model

```bash
pip install tensorflow numpy pandas scikit-learn
python train_velocity_model.py
```

Outputs `velocity_model.h` with weights for Pico.

### 3. Deploy

1. Copy weights from `velocity_model.h` to `pico_ml_node/ml_model.h`
2. Rebuild: `cd pico_ml_node/build && make`
3. Flash new `.uf2` file

## ML Model

Uses 6 velocity-based features computed from a 10-reading buffer:

| Feature | Description |
|---------|-------------|
| distance | Current reading (normalized) |
| velocity | Rate of change |
| acceleration | Rate of velocity change |
| variance | Stability over window |
| min | Minimum in window |
| max | Maximum in window |

Architecture: `Input(6) → Dense(12, ReLU) → Dense(6, ReLU) → Dense(1, Sigmoid)`

Size: ~170 parameters, <1KB on device

### Tuning

Adjust sensitivity in `pico_ml_node/ml_model.h`:
```c
#define ANOMALY_THRESHOLD 0.7f  // Higher = less sensitive
```

Adjust fog alerts:
```bash
python fog_node.py --min-anomalies 3 --time-window 5.0
```

## Project Structure

```
├── pico_ml_node/              # Pico 2 firmware
│   ├── main.cpp               # UART receive + USB output
│   ├── ml_model.cpp           # TFLite inference
│   ├── ml_model.h             # Model weights + config
│   └── CMakeLists.txt
├── sketchs/
│   └── arduino_sensor_node/   # Arduino sensor sketch
├── fog_node.py                # Fog aggregation
├── simulate_node.py           # Virtual sensors for testing
├── train_velocity_model.py    # Model training
├── collect_training_data.py   # Labeled data collection
└── training_data.csv          # Your collected data
```
