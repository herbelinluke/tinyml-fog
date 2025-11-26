#!/usr/bin/env python3
"""
Fog Node - Distributed ML Anomaly Aggregation

This fog node aggregates anomaly reports from edge ML devices (like the Pico 2)
and triggers system-level alerts when sustained anomalies are detected.

Key Features:
- Requires MULTIPLE anomalies from a node before considering it "in alert"
- Time-windowed analysis to filter out brief spikes
- Multi-node correlation for system-wide threat detection
"""

import serial
import time
import threading
import argparse
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import sys

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Reading:
    timestamp: float
    distance: int
    anomaly: bool
    confidence: float = 0.0

@dataclass 
class NodeState:
    node_id: str
    last_reading: Optional[Reading] = None
    readings_in_window: int = 0
    anomalies_in_window: int = 0
    total_readings: int = 0
    total_anomalies: int = 0
    connected: bool = True
    in_alert: bool = False  # True if this node has sustained anomalies

# ============================================================================
# FOG NODE
# ============================================================================

class FogNode:
    def __init__(self, 
                 min_anomalies_per_node: int = 3,
                 node_alert_threshold: int = 1,
                 time_window: float = 5.0):
        """
        Args:
            min_anomalies_per_node: How many anomalies a node must report 
                                   within the time window to be "in alert"
            node_alert_threshold: How many nodes must be "in alert" to 
                                 trigger a system alert
            time_window: Seconds to look back for anomaly counting
        """
        self.nodes: Dict[str, NodeState] = {}
        self.events: deque = deque()  # (timestamp, node_id, anomaly, confidence)
        
        self.min_anomalies_per_node = min_anomalies_per_node
        self.node_alert_threshold = node_alert_threshold
        self.time_window = time_window
        
        self.lock = threading.RLock()
        self.system_alert_active = False
        self.running = True
        
    def register_node(self, node_id: str) -> None:
        with self.lock:
            if node_id not in self.nodes:
                self.nodes[node_id] = NodeState(node_id=node_id)
                print(f"\n[FOG] 📡 New node registered: {node_id}", flush=True)
    
    def report_reading(self, node_id: str, distance: int, anomaly: bool, 
                       confidence: float = 0.0) -> None:
        timestamp = time.time()
        reading = Reading(timestamp=timestamp, distance=distance, 
                         anomaly=anomaly, confidence=confidence)
        
        with self.lock:
            if node_id not in self.nodes:
                self.register_node(node_id)
            
            node = self.nodes[node_id]
            node.last_reading = reading
            node.total_readings += 1
            
            if anomaly:
                node.total_anomalies += 1
            
            # Store event for time-window analysis
            self.events.append((timestamp, node_id, anomaly, confidence))
            
            # Clean old events
            self._cleanup_old_events()
            
            # Update node statistics
            self._update_node_stats()
            
            # Check for system-level alert
            self._check_system_alert()
    
    def _cleanup_old_events(self) -> None:
        """Remove events outside the time window."""
        cutoff = time.time() - self.time_window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()
    
    def _update_node_stats(self) -> None:
        """Update per-node statistics from recent events."""
        now = time.time()
        cutoff = now - self.time_window
        
        # Reset counts
        for node in self.nodes.values():
            node.readings_in_window = 0
            node.anomalies_in_window = 0
        
        # Count events per node
        for ts, node_id, anomaly, conf in self.events:
            if ts >= cutoff and node_id in self.nodes:
                self.nodes[node_id].readings_in_window += 1
                if anomaly:
                    self.nodes[node_id].anomalies_in_window += 1
        
        # Update node alert status
        for node in self.nodes.values():
            was_in_alert = node.in_alert
            node.in_alert = node.anomalies_in_window >= self.min_anomalies_per_node
            
            if node.in_alert and not was_in_alert:
                print(f"\n[FOG] ⚠️  Node '{node.node_id}' entered alert state "
                      f"({node.anomalies_in_window} anomalies in {self.time_window}s)", 
                      flush=True)
            elif not node.in_alert and was_in_alert:
                print(f"\n[FOG] ✓ Node '{node.node_id}' returned to normal", flush=True)
    
    def _check_system_alert(self) -> None:
        """Check if enough nodes are in alert to trigger system alert."""
        nodes_in_alert = [n for n in self.nodes.values() if n.in_alert]
        
        if len(nodes_in_alert) >= self.node_alert_threshold:
            if not self.system_alert_active:
                self.system_alert_active = True
                self._raise_system_alert(nodes_in_alert)
        else:
            if self.system_alert_active:
                self.system_alert_active = False
                print(f"\n{'='*60}", flush=True)
                print(f"[FOG] ✅ SYSTEM ALERT CLEARED", flush=True)
                print(f"{'='*60}\n", flush=True)
    
    def _raise_system_alert(self, nodes_in_alert: List[NodeState]) -> None:
        """Raise a system-level alert."""
        print(f"\n{'='*60}", flush=True)
        print(f"[FOG] 🚨 SYSTEM ALERT - SUSTAINED ANOMALY DETECTED!", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Nodes in alert: {len(nodes_in_alert)}/{self.node_alert_threshold} required", flush=True)
        for node in nodes_in_alert:
            print(f"    - {node.node_id}: {node.anomalies_in_window} anomalies "
                  f"in last {self.time_window}s", flush=True)
        print(f"{'='*60}\n", flush=True)
    
    def get_status(self) -> str:
        """Get current status dashboard."""
        with self.lock:
            lines = ["\n" + "="*55]
            lines.append(f"  FOG NODE STATUS - {time.strftime('%H:%M:%S')}")
            lines.append("="*55)
            
            if not self.nodes:
                lines.append("  No nodes connected")
            else:
                lines.append(f"  {'Node':<12} {'Dist':>6} {'Anomalies':>12} {'Status':>10}")
                lines.append("  " + "-"*45)
                
                for node_id, node in sorted(self.nodes.items()):
                    if node.last_reading:
                        dist = f"{node.last_reading.distance}cm"
                        conf = f"{node.last_reading.confidence:.0%}" if node.last_reading.confidence else ""
                    else:
                        dist = "N/A"
                        conf = ""
                    
                    anomaly_str = f"{node.anomalies_in_window}/{self.min_anomalies_per_node}"
                    
                    if node.in_alert:
                        status = "🔴 ALERT"
                    elif node.anomalies_in_window > 0:
                        status = "🟡 WARN"
                    else:
                        status = "🟢 OK"
                    
                    lines.append(f"  {node_id:<12} {dist:>6} {anomaly_str:>12} {status:>10}")
            
            lines.append("  " + "-"*45)
            
            if self.system_alert_active:
                lines.append("  🚨 SYSTEM: ALERT ACTIVE")
            else:
                lines.append("  ✅ SYSTEM: Normal")
            
            lines.append(f"  Config: {self.min_anomalies_per_node} anomalies needed, "
                        f"{self.time_window}s window")
            lines.append("="*55)
            
            return "\n".join(lines)


# ============================================================================
# SERIAL READER
# ============================================================================

def serial_reader(fog: FogNode, port: str, baud: int, node_id: str) -> None:
    """Read from a serial port and report to fog node."""
    fog.register_node(node_id)
    
    while fog.running:
        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                print(f"[{node_id}] Connected to {port}", flush=True)
                while fog.running:
                    line = ser.readline().decode('utf-8').strip()
                    if line and not line.startswith('#'):
                        try:
                            parts = line.split(',')
                            
                            # Pico ML format: node_id,distance,anomaly,confidence
                            if len(parts) >= 4 and parts[0].startswith('pico'):
                                actual_node_id = parts[0]
                                distance = int(parts[1])
                                anomaly = int(parts[2]) == 1
                                confidence = float(parts[3])
                                fog.report_reading(actual_node_id, distance, 
                                                 anomaly, confidence)
                            
                            # Arduino format: distance,anomaly
                            elif len(parts) >= 2:
                                distance = int(parts[0])
                                anomaly = int(parts[1]) == 1
                                fog.report_reading(node_id, distance, anomaly)
                                
                        except ValueError:
                            pass
        except serial.SerialException as e:
            print(f"[{node_id}] Serial error: {e}. Retrying in 5s...", flush=True)
            with fog.lock:
                if node_id in fog.nodes:
                    fog.nodes[node_id].connected = False
            time.sleep(5)


def status_printer(fog: FogNode, interval: float = 3.0) -> None:
    """Periodically print the fog node status."""
    while fog.running:
        print(fog.get_status(), flush=True)
        time.sleep(interval)


def udp_receiver(fog: FogNode, port: int) -> None:
    """Receive data from simulated nodes via UDP."""
    import socket
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', port))
    sock.settimeout(1.0)
    
    print(f"[FOG] UDP receiver listening on port {port}", flush=True)
    
    while fog.running:
        try:
            data, addr = sock.recvfrom(1024)
            message = data.decode('utf-8').strip()
            
            parts = message.split(',')
            if len(parts) >= 3:
                node_id = parts[0]
                distance = int(parts[1])
                anomaly = int(parts[2]) == 1
                confidence = float(parts[3]) if len(parts) > 3 else 0.0
                fog.report_reading(node_id, distance, anomaly, confidence)
        except socket.timeout:
            continue
        except Exception as e:
            if fog.running:
                print(f"[FOG] UDP error: {e}")
    
    sock.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fog Node - Distributed ML Anomaly Aggregation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single Pico ML node (requires 3 anomalies in 5s to alert)
  python fog_node.py --ports /dev/ttyACM1 --min-anomalies 3

  # More sensitive (2 anomalies triggers alert)
  python fog_node.py --ports /dev/ttyACM1 --min-anomalies 2 --time-window 3

  # Multiple nodes (alert when any 1 node has sustained anomalies)
  python fog_node.py --ports /dev/ttyACM1 /dev/ttyUSB0 --node-threshold 1
        """
    )
    parser.add_argument('--ports', nargs='+', default=['/dev/ttyACM1'],
                        help='Serial ports to connect to')
    parser.add_argument('--baud', type=int, default=9600, 
                        help='Baud rate (default: 9600)')
    parser.add_argument('--min-anomalies', type=int, default=3,
                        help='Anomalies needed per node to trigger (default: 3)')
    parser.add_argument('--node-threshold', type=int, default=1,
                        help='Nodes that must be in alert for system alert (default: 1)')
    parser.add_argument('--time-window', type=float, default=5.0,
                        help='Time window in seconds (default: 5.0)')
    parser.add_argument('--status-interval', type=float, default=5.0,
                        help='Status update interval (default: 5.0)')
    parser.add_argument('--simulated', action='store_true',
                        help='Use UDP input from simulate_node.py')
    parser.add_argument('--udp-port', type=int, default=5000,
                        help='UDP port for simulated mode (default: 5000)')
    args = parser.parse_args()

    fog = FogNode(
        min_anomalies_per_node=args.min_anomalies,
        node_alert_threshold=args.node_threshold,
        time_window=args.time_window
    )
    
    print("\n" + "="*55)
    print("  FOG NODE - Distributed ML Anomaly Aggregation")
    print("="*55)
    print(f"  Min anomalies per node: {args.min_anomalies}")
    print(f"  Time window: {args.time_window}s")
    print(f"  Node threshold for system alert: {args.node_threshold}")
    print("="*55 + "\n")

    threads = []
    
    if args.simulated:
        print(f"[FOG] Running in simulated mode on UDP port {args.udp_port}", flush=True)
        t = threading.Thread(target=udp_receiver, args=(fog, args.udp_port), daemon=True)
        t.start()
        threads.append(t)
    else:
        for i, port in enumerate(args.ports):
            node_id = f"node_{i+1}"
            t = threading.Thread(target=serial_reader, 
                               args=(fog, port, args.baud, node_id), daemon=True)
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
