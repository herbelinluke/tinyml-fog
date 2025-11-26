import serial
import time
import csv
import argparse

def main():
    parser = argparse.ArgumentParser(description='Collect sensor data from Arduino node')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=9600, help='Baud rate (default: 9600)')
    parser.add_argument('--node-id', default='node_1', help='Node identifier (default: node_1)')
    parser.add_argument('--output', default='sensor_data.csv', help='Output CSV file (default: sensor_data.csv)')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to collect (default: 1000)')
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"[{args.node_id}] Connected to Arduino on {args.port}")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        exit(1)

    try:
        with open(args.output, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Timestamp', 'NodeID', 'Distance', 'Anomaly'])

            print(f"[{args.node_id}] Collecting data to {args.output}. Press Ctrl+C to stop.")
            sample_count = 0
            
            while sample_count < args.samples:
                try:
                    line = ser.readline().decode('utf-8').strip()

                    if line:
                        try:
                            # Parse new format: distance,anomaly
                            parts = line.split(',')
                            if len(parts) == 2:
                                distance = int(parts[0])
                                anomaly = int(parts[1])
                            else:
                                # Fallback for old format (just distance)
                                distance = int(line)
                                anomaly = 0
                            
                            timestamp = time.time()
                            csv_writer.writerow([timestamp, args.node_id, distance, anomaly])
                            
                            anomaly_str = " [ANOMALY]" if anomaly else ""
                            print(f"[{args.node_id}] Distance: {distance} cm{anomaly_str}")
                            
                            sample_count += 1

                        except ValueError:
                            print(f"[{args.node_id}] Could not parse data: {line}")

                except KeyboardInterrupt:
                    print(f"\n[{args.node_id}] Data collection stopped by user.")
                    break
                except serial.SerialException as e:
                    print(f"[{args.node_id}] Serial communication error: {e}")
                    break

    except IOError as e:
        print(f"Error opening/writing to file: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print(f"[{args.node_id}] Serial port closed. Collected {sample_count} samples.")

if __name__ == '__main__':
    main()
