#!/usr/bin/env python3
"""
Simulated Edge Node - Generates synthetic sensor data for testing fog aggregation.

This script creates virtual sensor nodes that output data in the same format as
the Arduino sketch, allowing testing of the fog node without physical hardware.

Can run standalone (prints to stdout) or pipe to fog_node.py
"""

import time
import random
import argparse
import socket
import sys

class SimulatedSensor:
    def __init__(self, node_id: str, base_distance: int = 100, 
                 noise_std: float = 5.0, anomaly_probability: float = 0.05):
        self.node_id = node_id
        self.base_distance = base_distance
        self.noise_std = noise_std
        self.anomaly_probability = anomaly_probability
        
        # Rolling average state (mirrors Arduino logic)
        self.window_size = 10
        self.readings = []
        self.deviation_threshold = 0.30
    
    def get_rolling_average(self) -> float:
        if not self.readings:
            return 0
        return sum(self.readings) / len(self.readings)
    
    def update_rolling_average(self, reading: int) -> None:
        self.readings.append(reading)
        if len(self.readings) > self.window_size:
            self.readings.pop(0)
    
    def detect_anomaly(self, distance: int) -> bool:
        if len(self.readings) < 3:
            return False
        
        avg = self.get_rolling_average()
        if avg > 0:
            deviation = abs(distance - avg) / avg
            return deviation > self.deviation_threshold
        return False
    
    def generate_reading(self) -> tuple:
        """Generate a sensor reading, optionally injecting anomalies."""
        
        # Decide if this should be an anomaly
        is_injected_anomaly = random.random() < self.anomaly_probability
        
        if is_injected_anomaly:
            # Generate anomalous reading (sudden jump or drop)
            if random.random() < 0.5:
                distance = int(self.base_distance * random.uniform(0.3, 0.5))  # Sudden close
            else:
                distance = int(self.base_distance * random.uniform(2.0, 3.0))  # Sudden far
        else:
            # Normal reading with noise
            distance = int(self.base_distance + random.gauss(0, self.noise_std))
            distance = max(2, min(400, distance))  # Clamp to valid range
        
        # Detect anomaly using rolling average (same as Arduino)
        anomaly = self.detect_anomaly(distance)
        
        # Only update rolling average with non-anomalous readings
        if not anomaly:
            self.update_rolling_average(distance)
        
        return distance, 1 if anomaly else 0


class MultiNodeSimulator:
    """Simulates multiple nodes and communicates with fog node via UDP."""
    
    def __init__(self, fog_host: str = 'localhost', fog_port: int = 5000):
        self.fog_host = fog_host
        self.fog_port = fog_port
        self.nodes: dict = {}
    
    def add_node(self, node_id: str, **kwargs) -> None:
        self.nodes[node_id] = SimulatedSensor(node_id, **kwargs)
    
    def run(self, sample_rate: float = 0.1, duration: float = None) -> None:
        """Run simulation, sending data to fog node."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        start_time = time.time()
        print(f"[SIM] Starting simulation with {len(self.nodes)} nodes")
        print(f"[SIM] Sending to {self.fog_host}:{self.fog_port}")
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                
                for node_id, sensor in self.nodes.items():
                    distance, anomaly = sensor.generate_reading()
                    
                    # Send to fog node via UDP
                    message = f"{node_id},{distance},{anomaly}"
                    sock.sendto(message.encode(), (self.fog_host, self.fog_port))
                    
                    anomaly_str = " [ANOMALY]" if anomaly else ""
                    print(f"[{node_id}] {distance},{anomaly}{anomaly_str}")
                
                time.sleep(sample_rate)
                
        except KeyboardInterrupt:
            print("\n[SIM] Simulation stopped")
        finally:
            sock.close()


def run_standalone(args):
    """Run a single simulated node, outputting to stdout or UDP."""
    sensor = SimulatedSensor(
        node_id=args.node_id,
        base_distance=args.base_distance,
        noise_std=args.noise,
        anomaly_probability=args.anomaly_rate
    )
    
    # Setup UDP socket if needed
    sock = None
    if args.udp:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"# Sending to {args.udp_host}:{args.udp_port} via UDP", file=sys.stderr)
    
    print(f"# Simulated node: {args.node_id}", file=sys.stderr)
    print(f"# Base distance: {args.base_distance}cm, Noise: {args.noise}, Anomaly rate: {args.anomaly_rate}", file=sys.stderr)
    print(f"# Output format: distance,anomaly", file=sys.stderr)
    
    sample_count = 0
    try:
        while args.samples == 0 or sample_count < args.samples:
            distance, anomaly = sensor.generate_reading()
            
            if args.udp and sock:
                # Send via UDP: node_id,distance,anomaly
                message = f"{args.node_id},{distance},{anomaly}"
                sock.sendto(message.encode(), (args.udp_host, args.udp_port))
            
            anomaly_str = " [ANOMALY]" if anomaly else ""
            print(f"{distance},{anomaly}{anomaly_str}")
            sys.stdout.flush()
            
            sample_count += 1
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print(f"\n# Stopped after {sample_count} samples", file=sys.stderr)
    finally:
        if sock:
            sock.close()


def run_multi_node_demo(args):
    """Run a multi-node demo with coordinated anomaly injection."""
    print("[DEMO] Multi-node demonstration mode")
    print("[DEMO] Simulating 3 nodes with occasional coordinated anomalies")
    
    # Setup UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[DEMO] Sending to {args.udp_host}:{args.udp_port} via UDP")
    
    nodes = {
        'node_1': SimulatedSensor('node_1', base_distance=100, anomaly_probability=0.02),
        'node_2': SimulatedSensor('node_2', base_distance=150, anomaly_probability=0.02),
        'node_3': SimulatedSensor('node_3', base_distance=200, anomaly_probability=0.02),
    }
    
    coordinated_anomaly_prob = 0.03  # Chance of all nodes having anomaly at once
    
    try:
        while True:
            # Check for coordinated anomaly event
            if random.random() < coordinated_anomaly_prob:
                print("\n[DEMO] >>> Injecting coordinated anomaly event <<<")
                # Force anomalies on multiple nodes
                for node_id, sensor in nodes.items():
                    distance = int(sensor.base_distance * random.uniform(0.3, 0.5))
                    message = f"{node_id},{distance},1"
                    sock.sendto(message.encode(), (args.udp_host, args.udp_port))
                    print(f"{node_id}: {distance},1 [COORDINATED ANOMALY]")
            else:
                # Normal operation
                for node_id, sensor in nodes.items():
                    distance, anomaly = sensor.generate_reading()
                    message = f"{node_id},{distance},{anomaly}"
                    sock.sendto(message.encode(), (args.udp_host, args.udp_port))
                    anomaly_str = " [ANOMALY]" if anomaly else ""
                    print(f"{node_id}: {distance},{anomaly}{anomaly_str}")
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n[DEMO] Demonstration stopped")
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(
        description='Simulate edge sensor nodes for fog computing demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single node to stdout (can pipe to fog_node or save to file)
  python simulate_node.py --node-id sensor_1

  # Multi-node demo with coordinated anomalies  
  python simulate_node.py --demo

  # Custom parameters
  python simulate_node.py --node-id sensor_1 --base-distance 150 --anomaly-rate 0.1
        """
    )
    
    parser.add_argument('--node-id', default='sim_node_1', 
                        help='Node identifier (default: sim_node_1)')
    parser.add_argument('--base-distance', type=int, default=100,
                        help='Base distance in cm (default: 100)')
    parser.add_argument('--noise', type=float, default=5.0,
                        help='Noise standard deviation (default: 5.0)')
    parser.add_argument('--anomaly-rate', type=float, default=0.05,
                        help='Probability of anomaly injection (default: 0.05)')
    parser.add_argument('--interval', type=float, default=0.1,
                        help='Sample interval in seconds (default: 0.1)')
    parser.add_argument('--samples', type=int, default=0,
                        help='Number of samples (0 = infinite, default: 0)')
    parser.add_argument('--demo', action='store_true',
                        help='Run multi-node demonstration')
    parser.add_argument('--udp-host', default='localhost',
                        help='Fog node host for UDP mode (default: localhost)')
    parser.add_argument('--udp-port', type=int, default=5000,
                        help='Fog node UDP port (default: 5000)')
    parser.add_argument('--udp', action='store_true',
                        help='Send data via UDP to fog node')
    
    args = parser.parse_args()
    
    if args.demo:
        run_multi_node_demo(args)
    else:
        run_standalone(args)


if __name__ == '__main__':
    main()

