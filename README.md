
# Network Simulation & Emulation Environment  
## NS-3 and Mininet Installation & Usage Guide

This repository provides **step-by-step installation guides** for **NS-3** and **Mininet**, along with instructions on how to **run and test network simulation and emulation codes**.

The repository is intended for:
- Network simulation experiments
- SDN and traffic engineering research
- Academic projects and coursework
- Performance evaluation and testing

---

## 📌 Contents

- NS-3 Installation Guide
- Mininet Installation Guide
- How to Run Simulation & Emulation Codes
- Project Structure
- Notes & Troubleshooting

---

## 🧰 Requirements

Before installation, ensure your system meets the following requirements:

- Ubuntu 20.04 / 22.04 (recommended)
- Python 3.x
- Git
- Minimum 8 GB RAM (recommended for simulations)

---

## 🔹NS3

# to run the code
- cd ns-3-dev/
- ./ns3 build
- ./ns3 run "scratch/script"
- # Or run with specific parameters
- ./ns3 run "scratch/data_center_sim --cores=2 --aggs=4 --tors=8 --servers=8 --time=5 --output=dc_results.csv"

---

## 🔹Mininet
# to run the code
- cd mininet/
- #Before running any Mininet script:
- sudo mn -c
- python3 script.py


---

## 📚 References
- NS-3 Official Documentation
- Mininet Official GitHub Repository


```bash
sudo apt update && sudo apt upgrade -y
