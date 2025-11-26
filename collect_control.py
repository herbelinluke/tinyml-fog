import serial
import time
import csv
import threading

arduino_port = '/dev/ttyUSB0'
baud_rate = 9600

try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    print(f"Connected to Arduino on {arduino_port}")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit()

filename = 'anomoly_data.csv'

flag = False
lock = threading.Lock()

def input_thread():
    global flag
    while True:
        try:
            s = input()
        except EOFError:
            break
        if s.strip().lower() == 'a':
            with lock:
                flag = True
            print("Next 3 seconds marked as ANOMALY.")
            time.sleep(3)
            with lock:
                flag = False

t = threading.Thread(target=input_thread, daemon=True)
t.start()


try:
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Timestamp', 'SensorValue', 'Label'])

        print(f"Collecting data to {filename}. -- press 'a' + Enter to mark anomaly for next 3 seconds. Press Ctrl+C to stop.")
        for _ in range(50):
            try:
                line = ser.readline().decode('utf-8').strip()

                if line:
                    try:
                        with lock:
                            lab = 1 if flag else 0
                        
                        sensor_value = int(line)
                        timestamp = time.time()

                        csv_writer.writerow([timestamp, sensor_value, lab])
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
