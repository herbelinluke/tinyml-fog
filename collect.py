import serial
import time
import csv

arduino_port = '/dev/ttyUSB0'
baud_rate = 9600

try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    print(f"Connected to Arduino on {arduino_port}")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit()

filename = 'arduino_data.csv'
try:
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Timestamp', 'SensorValue']) # Write header row

        print(f"Collecting data to {filename}. Press Ctrl+C to stop.")
        while True:
            try:
                # Read a line from the serial port
                line = ser.readline().decode('utf-8').strip()

                if line:  # Check if a non-empty line was received
                    try:
                        sensor_value = int(line)  # Convert the received string to an integer
                        timestamp = time.time()  # Get current timestamp

                        # Write data to CSV
                        csv_writer.writerow([timestamp, sensor_value])
                        print(f"Timestamp: {timestamp}, Sensor Value: {sensor_value}")

                    except ValueError:
                        print(f"Could not convert data to int: {line}")

            except KeyboardInterrupt:
                print("\nData collection stopped by user.")
                break
            except serial.SerialException as e:
                print(f"Serial communication error: {e}")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break

except IOError as e:
    print(f"Error opening/writing to file: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")
