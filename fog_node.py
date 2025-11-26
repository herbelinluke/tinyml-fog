#!/usr/bin/env python3
"""
Fog Node - Aggregates sensor data from multiple edge nodes and raises system alerts.

This script acts as the fog layer in a distributed anomaly detection system.
It connects to multiple edge nodes (via serial or simulated), aggregates their
anomaly reports, and raises system-level alerts when multiple nodes report issues.
"""

import serial
import time
import threading
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional
import sys

@dataclass
class NodeReading:
    timestamp: float
    distance: int
    anomaly: bool

@dataclass 
class NodeState:
    node_id: str
    last_reading: Optional[NodeReading] = None
    recent_anomalies: int = 0
    total_readings: int = 0
    total_anomalies: int = 0
    connected: bool = True

class FogNode:
    def __init__(self, alert_threshold: int = 2, time_window: float = 5.0):
        self.nodes: Dict[str, NodeState] = {}
        self.anomaly_events: deque = deque()  # (timestamp, node_id) tuples
        self.alert_threshold = alert_threshold  # Number of nodes that must report anomaly
        self.time_window = time_window  # Seconds to consider for alert
        self.lock = threading.Lock()
        self.alert_active = False
        self.running = True
        
    def register_node(self, node_id: str) -> None:
        with self.lock:
            if node_id not in self.nodes:
                self.nodes[node_id] = NodeState(node_id=node_id)
                print(f"[FOG] Registered node: {node_id}")
    
    def report_reading(self, node_id: str, distance: int, anomaly: bool) -> None:
        timestamp = time.time()
        reading = NodeReading(timestamp=timestamp, distance=distance, anomaly=anomaly)
        
        with self.lock:
            if node_id not in self.nodes:
                self.register_node(node_id)
            
            node = self.nodes[node_id]
            node.last_reading = reading
            node.total_readings += 1
            
            if anomaly:
                node.total_anomalies += 1
                self.anomaly_events.append((timestamp, node_id))
            
            # Clean old events outside time window
            cutoff = timestamp - self.time_window
            while self.anomaly_events and self.anomaly_events[0][0] < cutoff:
                self.anomaly_events.popleft()
            
            # Check for system-level alert
            self._check_alert()
    
    def _check_alert(self) -> None:
        """Check if enough nodes have reported anomalies within the time window."""
        now = time.time()
        cutoff = now - self.time_window
        
        # Get unique nodes that reported anomalies in the window
        nodes_with_anomalies = set()
        for ts, node_id in self.anomaly_events:
            if ts >= cutoff:
                nodes_with_anomalies.add(node_id)
        
        # Update per-node recent anomaly counts
        for node_id, node in self.nodes.items():
            node.recent_anomalies = sum(1 for ts, nid in self.anomaly_events 
                                        if nid == node_id and ts >= cutoff)
        
        if len(nodes_with_anomalies) >= self.alert_threshold:
            if not self.alert_active:
                self.alert_active = True
                self._raise_alert(nodes_with_anomalies)
        else:
            if self.alert_active:
                self.alert_active = False
                print(f"\n[FOG] ✓ Alert cleared - system normal")
    
    def _raise_alert(self, affected_nodes: set) -> None:
        """Raise a system-level alert."""
        print(f"\n{'='*60}")
        print(f"[FOG] ⚠ SYSTEM ALERT - Multiple nodes reporting anomalies!")
        print(f"[FOG] Affected nodes: {', '.join(sorted(affected_nodes))}")
        print(f"[FOG] Threshold: {len(affected_nodes)}/{self.alert_threshold} nodes in {self.time_window}s window")
        print(f"{'='*60}\n")
    
    def get_status(self) -> str:
        """Get current status of all nodes."""
        with self.lock:
            lines = ["\n" + "="*50]
            lines.append(f"[FOG] Node Status Dashboard - {time.strftime('%H:%M:%S')}")
            lines.append("="*50)
            
            if not self.nodes:
                lines.append("  No nodes connected")
            else:
                for node_id, node in sorted(self.nodes.items()):
                    status = "🟢" if node.connected else "🔴"
                    if node.last_reading:
                        dist = node.last_reading.distance
                        anom = "⚠" if node.last_reading.anomaly else "✓"
                    else:
                        dist = "N/A"
                        anom = "-"
                    
                    lines.append(f"  {status} {node_id}: dist={dist}cm {anom} | "
                               f"anomalies(recent/total): {node.recent_anomalies}/{node.total_anomalies}")
            
            alert_status = "⚠ ALERT ACTIVE" if self.alert_active else "✓ Normal"
            lines.append(f"\n  System: {alert_status}")
            lines.append("="*50)
            
            return "\n".join(lines)


def serial_reader(fog: FogNode, port: str, baud: int, node_id: str) -> None:
    """Read from a serial port and report to fog node."""
    fog.register_node(node_id)
    
    while fog.running:
        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                print(f"[{node_id}] Connected to {port}")
                while fog.running:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        try:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                distance = int(parts[0])
                                anomaly = int(parts[1]) == 1
                                fog.report_reading(node_id, distance, anomaly)
                        except ValueError:
                            pass  # Ignore malformed lines
        except serial.SerialException as e:
            print(f"[{node_id}] Serial error: {e}. Retrying in 5s...")
            with fog.lock:
                if node_id in fog.nodes:
                    fog.nodes[node_id].connected = False
            time.sleep(5)


def status_printer(fog: FogNode, interval: float = 2.0) -> None:
    """Periodically print the fog node status."""
    while fog.running:
        print(fog.get_status())
        time.sleep(interval)


def udp_receiver(fog: FogNode, port: int) -> None:
    """Receive data from simulated nodes via UDP."""
    import socket
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', port))
    sock.settimeout(1.0)
    
    print(f"[FOG] UDP receiver listening on port {port}")
    
    while fog.running:
        try:
            data, addr = sock.recvfrom(1024)
            message = data.decode('utf-8').strip()
            
            # Parse: node_id,distance,anomaly
            parts = message.split(',')
            if len(parts) >= 3:
                node_id = parts[0]
                distance = int(parts[1])
                anomaly = int(parts[2]) == 1
                fog.report_reading(node_id, distance, anomaly)
        except socket.timeout:
            continue
        except Exception as e:
            if fog.running:
                print(f"[FOG] UDP error: {e}")
    
    sock.close()


def main():
    parser = argparse.ArgumentParser(description='Fog Node - Aggregates data from edge nodes')
    parser.add_argument('--ports', nargs='+', default=['/dev/ttyUSB0'],
                        help='Serial ports to connect to (default: /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=9600, help='Baud rate (default: 9600)')
    parser.add_argument('--alert-threshold', type=int, default=2,
                        help='Number of nodes with anomalies to trigger alert (default: 2)')
    parser.add_argument('--time-window', type=float, default=5.0,
                        help='Time window in seconds for anomaly aggregation (default: 5.0)')
    parser.add_argument('--status-interval', type=float, default=3.0,
                        help='Status update interval in seconds (default: 3.0)')
    parser.add_argument('--simulated', action='store_true',
                        help='Run with simulated nodes via UDP instead of serial ports')
    parser.add_argument('--udp-port', type=int, default=5000,
                        help='UDP port for simulated nodes (default: 5000)')
    args = parser.parse_args()

    fog = FogNode(alert_threshold=args.alert_threshold, time_window=args.time_window)
    threads = []

    print(f"[FOG] Starting Fog Node")
    print(f"[FOG] Alert threshold: {args.alert_threshold} nodes in {args.time_window}s window")
    
    if args.simulated:
        print(f"[FOG] Running in simulated mode on UDP port {args.udp_port}")
        t = threading.Thread(target=udp_receiver, args=(fog, args.udp_port), daemon=True)
        t.start()
        threads.append(t)
    else:
        # Start serial readers for each port
        for i, port in enumerate(args.ports):
            node_id = f"node_{i+1}"
            t = threading.Thread(target=serial_reader, args=(fog, port, args.baud, node_id), 
                               daemon=True)
            t.start()
            threads.append(t)

    # Start status printer
    status_thread = threading.Thread(target=status_printer, 
                                    args=(fog, args.status_interval), daemon=True)
    status_thread.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[FOG] Shutting down...")
        fog.running = False


if __name__ == '__main__':
    main()

