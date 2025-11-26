#!/usr/bin/env python3
"""
Collect labeled training data for anomaly detection.

Controls:
- Press 'n' + Enter: Start NORMAL collection (with countdown)
- Press 'a' + Enter: Start ANOMALY collection (with countdown)
- Press 's' + Enter: Stop current collection
- Press 'q' + Enter: Quit and save

Workflow:
1. Press 'n' → 3 second countdown → collects NORMAL for 10 seconds → stops
2. Press 'a' → 3 second countdown → collects ANOMALY for 5 seconds → stops
3. Repeat as needed
4. Press 'q' to save and exit
"""

import serial
import time
import csv
import threading
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

ARDUINO_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
OUTPUT_FILE = 'training_data.csv'

COUNTDOWN_SECONDS = 3       # Countdown before collection starts
NORMAL_DURATION = 10        # Seconds to collect normal data
ANOMALY_DURATION = 1        # Seconds to collect anomaly data

# ============================================================================
# GLOBAL STATE
# ============================================================================

class CollectorState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.collecting = False
        self.current_label = None  # None = not collecting, 0 = normal, 1 = anomaly
        self.collection_end_time = 0
        self.countdown_active = False

state = CollectorState()

# ============================================================================
# COUNTDOWN AND COLLECTION CONTROL
# ============================================================================

def start_collection(label, duration):
    """Start a timed collection session with countdown."""
    label_name = "ANOMALY" if label == 1 else "NORMAL"
    
    print(f"\n{'='*50}")
    print(f"  Preparing to collect: {label_name}")
    print(f"  Duration: {duration} seconds")
    print(f"{'='*50}")
    
    # Countdown
    state.countdown_active = True
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        if not state.running:
            return
        print(f"  >>> Starting in {i}... <<<")
        time.sleep(1)
    
    state.countdown_active = False
    
    # Start collection
    with state.lock:
        state.collecting = True
        state.current_label = label
        state.collection_end_time = time.time() + duration
    
    print(f"\n  🔴 COLLECTING {label_name} DATA NOW! 🔴\n")

def stop_collection():
    """Stop current collection."""
    with state.lock:
        if state.collecting:
            state.collecting = False
            state.current_label = None
            print("\n  ⏹️  Collection stopped. Waiting for next command...\n")

def check_collection_timeout():
    """Check if collection timer has expired."""
    with state.lock:
        if state.collecting and time.time() >= state.collection_end_time:
            state.collecting = False
            label_name = "ANOMALY" if state.current_label == 1 else "NORMAL"
            state.current_label = None
            print(f"\n  ✅ {label_name} collection complete!")
            print("  Waiting for next command (n=normal, a=anomaly, q=quit)...\n")

# ============================================================================
# INPUT HANDLER
# ============================================================================

def input_thread():
    """Handle user input commands."""
    print("\n" + "="*60)
    print("  TRAINING DATA COLLECTION")
    print("="*60)
    print("\n  Commands:")
    print("    'n' + Enter = Collect NORMAL data (walking around)")
    print("    'a' + Enter = Collect ANOMALY data (fast approach)")
    print("    's' + Enter = Stop current collection")
    print("    'q' + Enter = Quit and save")
    print("\n  Waiting for command...\n")
    
    while state.running:
        try:
            cmd = input().strip().lower()
            
            if cmd == 'q':
                state.running = False
                print("\n>>> Quitting... <<<\n")
                break
            
            elif cmd == 's':
                stop_collection()
            
            elif cmd == 'n':
                if state.collecting or state.countdown_active:
                    print("  ⚠️  Already collecting! Press 's' to stop first.")
                else:
                    # Start in a separate thread so we don't block input
                    t = threading.Thread(target=start_collection, args=(0, NORMAL_DURATION))
                    t.daemon = True
                    t.start()
            
            elif cmd == 'a':
                if state.collecting or state.countdown_active:
                    print("  ⚠️  Already collecting! Press 's' to stop first.")
                else:
                    t = threading.Thread(target=start_collection, args=(1, ANOMALY_DURATION))
                    t.daemon = True
                    t.start()
            
        except EOFError:
            break
        except Exception as e:
            print(f"Input error: {e}")

# ============================================================================
# MAIN DATA COLLECTION
# ============================================================================

def main():
    # Start input handler
    input_handler = threading.Thread(target=input_thread, daemon=True)
    input_handler.start()
    
    # Open serial connection
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        print(f"  ✅ Connected to {ARDUINO_PORT}")
    except serial.SerialException as e:
        print(f"  ❌ Error: {e}")
        print(f"     Make sure Arduino is connected to {ARDUINO_PORT}")
        return
    
    # Stats
    sample_count = 0
    normal_count = 0
    anomaly_count = 0
    
    # Open CSV file
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Distance', 'Label', 'LabelName'])
        
        print(f"  📁 Saving to {OUTPUT_FILE}\n")
        
        while state.running:
            try:
                # Check for collection timeout
                check_collection_timeout()
                
                # Read from Arduino
                line = ser.readline().decode('utf-8').strip()
                
                if not line:
                    continue
                
                # Parse distance
                parts = line.split(',')
                try:
                    distance = int(parts[0])
                except ValueError:
                    continue
                
                # Only record if actively collecting
                with state.lock:
                    if not state.collecting or state.current_label is None:
                        # Show waiting indicator
                        sys.stdout.write(f"\r  ⏸️  Waiting... (sensor: {distance}cm)    ")
                        sys.stdout.flush()
                        continue
                    
                    label = state.current_label
                    remaining = max(0, state.collection_end_time - time.time())
                
                # Record data
                timestamp = time.time()
                label_name = 'anomaly' if label == 1 else 'normal'
                
                writer.writerow([timestamp, distance, label, label_name])
                f.flush()
                
                sample_count += 1
                if label == 0:
                    normal_count += 1
                else:
                    anomaly_count += 1
                
                # Status indicator
                icon = "🔴" if label == 1 else "🟢"
                label_str = "ANOMALY" if label == 1 else "NORMAL "
                print(f"  {icon} {label_str}: {distance:3d}cm  |  ⏱️ {remaining:.1f}s left  |  total: {sample_count}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    ser.close()
    
    # Final summary
    print("\n" + "="*60)
    print("  COLLECTION COMPLETE")
    print("="*60)
    print(f"  Total samples: {sample_count}")
    print(f"  Normal:  {normal_count}")
    print(f"  Anomaly: {anomaly_count}")
    print(f"  Saved to: {OUTPUT_FILE}")
    print("="*60 + "\n")
    
    if anomaly_count < 50:
        print("  ⚠️  Tip: Try to collect at least 50-100 anomaly samples")
        print("      for good model training.\n")

if __name__ == '__main__':
    main()
