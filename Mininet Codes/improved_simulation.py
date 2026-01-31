#!/usr/bin/env python3
import random
import csv
import time

def simulate_metrics(samples=6000, output_file='dc_metrics_balanced.csv'):
    fieldnames = ['src', 'dst', 'link_down', 'congestion', 'packet_size',
                  'latency_ms', 'jitter_ms', 'packet_loss', 'throughput_mbps',
                  'network_load', 'bandwidth_usage_mbps', 'routing_failure']
    
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(samples):
            src_id = random.randint(1, 16)
            dst_id = random.choice([x for x in range(1, 17) if x != src_id])
            
            link_down = 1 if random.random() < 0.15 else 0
            congestion = 1 if random.random() < 0.25 else 0
            packet_size = random.choice([64, 512, 1400])
            
            # Base metrics
            latency = random.uniform(2, 10)
            jitter = random.uniform(0.1, 2)
            packet_loss = 0.0
            throughput = random.uniform(80, 100)
            
            if link_down:
                # Link down causes high loss or increased latency if rerouted
                if random.random() < 0.7:
                    packet_loss = random.uniform(80, 100)
                    throughput = random.uniform(0, 10)
                    latency = 0.0
                    jitter = 0.0
                else:
                    # Rerouted
                    packet_loss = random.uniform(5, 20)
                    latency = random.uniform(20, 50)
                    jitter = random.uniform(5, 15)
                    throughput = random.uniform(30, 60)
            
            if congestion:
                # Congestion increases latency and jitter, and causes some loss
                latency += random.uniform(10, 100)
                jitter += random.uniform(5, 30)
                packet_loss += random.uniform(1, 15)
                throughput *= random.uniform(0.2, 0.7)
            
            # Cap packet loss at 100%
            packet_loss = min(100.0, packet_loss)
            
            network_load = random.uniform(10, 95)
            bandwidth_usage = (throughput * (network_load/100.0)) + random.uniform(0, 5)
            routing_failure = 1 if (link_down and packet_loss > 90) else 0

            writer.writerow({
                'src': f'h{src_id}', 'dst': f'h{dst_id}', 'link_down': link_down,
                'congestion': congestion, 'packet_size': packet_size,
                'latency_ms': round(latency, 2), 'jitter_ms': round(jitter, 2),
                'packet_loss': round(packet_loss, 2), 'throughput_mbps': round(throughput, 2),
                'network_load': round(network_load, 2), 'bandwidth_usage_mbps': round(bandwidth_usage, 2),
                'routing_failure': routing_failure
            })
            
            if i % 500 == 0:
                print(f"Generated {i}/{samples} samples...")
    
    print(f"Simulation completed. Balanced dataset saved to {output_file}")

if __name__ == '__main__':
    simulate_metrics(samples=6000)
