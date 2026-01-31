#!/usr/bin/python3

from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.clean import cleanup

import time
import csv
import random
import re
import os

# ================= CONFIG =================
TOTAL_SAMPLES = 15000
CSV_FILE = "dc_15k_dataset.csv"

BASE_BW = 100      # Mbps
MAX_DELAY = 30     # ms
MAX_LOSS = 5       # %
IPERF_RATE = "50M"
TEST_DURATION = 5
# ==========================================


def create_topology():
    net = Mininet(
        controller=Controller,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )

    net.addController('c0')

    core1 = net.addSwitch('c1')
    core2 = net.addSwitch('c2')

    aggs = [net.addSwitch(f'a{i+1}') for i in range(8)]
    hosts = [net.addHost(f'h{i+1}') for i in range(64)]

    host_links = []
    core_links = []

    # Host -> Aggregation
    for i, host in enumerate(hosts):
        agg = aggs[i // 8]
        link = net.addLink(host, agg, bw=BASE_BW, delay='2ms', use_htb=True)
        host_links.append(link)

    # Aggregation -> Core
    for agg in aggs:
        core_links.append(net.addLink(agg, core1, bw=BASE_BW, delay='5ms', use_htb=True))
        core_links.append(net.addLink(agg, core2, bw=BASE_BW, delay='5ms', use_htb=True))

    return net, hosts, host_links, core_links


def randomize_links(host_links, core_links):
    # Random congestion
    for link in host_links:
        bw = random.randint(20, BASE_BW)
        delay = f"{random.randint(1, MAX_DELAY)}ms"
        loss = random.uniform(0, MAX_LOSS)

        link.intf1.config(bw=bw, delay=delay, loss=loss)
        link.intf2.config(bw=bw, delay=delay, loss=loss)

    # Random link failure (10% chance)
    for link in core_links:
        if random.random() < 0.1:
            link.intf1.ifconfig('down')
            link.intf2.ifconfig('down')
        else:
            link.intf1.ifconfig('up')
            link.intf2.ifconfig('up')


def parse_ping(output):
    latency = jitter = loss = 0.0

    loss_match = re.search(r'(\d+)% packet loss', output)
    if loss_match:
        loss = float(loss_match.group(1))

    rtt_match = re.search(
        r'rtt min/avg/max/mdev = ([\d\.]+)/([\d\.]+)/([\d\.]+)/([\d\.]+)',
        output
    )
    if rtt_match:
        latency = float(rtt_match.group(2))
        jitter = float(rtt_match.group(4))

    return latency, jitter, loss


def measure_metrics(hosts):
    h1 = hosts[0]
    h2 = hosts[-1]

    # ---- Ping first (clean latency/jitter) ----
    ping_out = h1.cmd(f"ping -c 10 -i 0.2 {h2.IP()}")
    latency, jitter, ping_loss = parse_ping(ping_out)

    # ---- Iperf (throughput + loss) ----
    h2.cmd("iperf -s -u > /tmp/iperf.log &")
    time.sleep(1)

    iperf_out = h1.cmd(
        f"iperf -c {h2.IP()} -u -b {IPERF_RATE} -t {TEST_DURATION}"
    )

    throughput = iperf_loss = 0.0

    t_match = re.search(r'(\d+\.?\d*) Mbits/sec', iperf_out)
    if t_match:
        throughput = float(t_match.group(1))

    loss_match = re.search(r'(\d+\.?\d*)% packet loss', iperf_out)
    if loss_match:
        iperf_loss = float(loss_match.group(1))

    congestion = throughput / BASE_BW

    return throughput, congestion, iperf_loss, latency, jitter


def run():
    cleanup()

    net, hosts, host_links, core_links = create_topology()
    net.start()
    time.sleep(3)

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "throughput_mbps",
            "congestion_ratio",
            "packet_loss_percent",
            "latency_ms",
            "jitter_ms"
        ])

        for i in range(TOTAL_SAMPLES):
            randomize_links(host_links, core_links)
            time.sleep(0.5)

            metrics = measure_metrics(hosts)
            writer.writerow(metrics)

            if i % 500 == 0:
                print(f"[+] Generated {i}/{TOTAL_SAMPLES} samples")

    net.stop()
    cleanup()
    print(f"✅ Dataset generated: {CSV_FILE}")


if __name__ == "__main__":
    setLogLevel("error")
    run()
