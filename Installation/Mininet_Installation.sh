#!/bin/bash
#=======================================================================================
#This script was tested on CentOS Stream 10
#=======================================================================================
#Step 1: System Preparation
#Update the system and enable required repositories:
sudo dnf update -y
sudo reboot


#Enable EPEL and CRB repositories:
sudo dnf install -y epel-release
sudo crb enable
sudo dnf update -y


#=======================================================================================
#Step 2: Install Required Dependencies
# Install development tools, networking utilities, and kernel headers:
sudo dnf install -y \
  git make gcc gcc-c++ \
  python3 python3-pip python3-devel \
  kernel-devel kernel-headers \
  autoconf automake libtool \
  libcap-ng-devel libuuid-devel openssl-devel \
  numactl-devel libibverbs-devel elfutils-libelf-devel \
  iproute iperf iperf3 tcpdump wireshark-cli \
  bridge-utils net-tools


#=======================================================================================
#Step 3: Build and Install Open vSwitch from Source
#Open vSwitch is not available as a prebuilt package on CentOS Stream 10, so it must be compiled manually.

# Clone Open vSwitch
cd /opt
sudo git clone https://github.com/openvswitch/ovs.git
cd ovs

# Compile and Install
sudo ./boot.sh
sudo ./configure
sudo make -j$(nproc)
sudo make install
sudo ldconfig

#=======================================================================================
#Step 4: Fix PATH for Open vSwitch Binaries
#Open vSwitch installs under /usr/local. Add it to PATH:
echo 'export PATH=$PATH:/usr/local/bin:/usr/local/sbin' | sudo tee /etc/profile.d/ovs.sh
source /etc/profile.d/ovs.sh

#=======================================================================================
#Step 5: Initialize and Start Open vSwitch
#Create required directories:
sudo mkdir -p /usr/local/etc/openvswitch
sudo mkdir -p /usr/local/var/run/openvswitch

#Create OVS database:
sudo /usr/local/bin/ovsdb-tool create \
  /usr/local/etc/openvswitch/conf.db \
  /opt/ovs/vswitchd/vswitch.ovsschema

#Start OVS database server:
sudo /usr/local/sbin/ovsdb-server \
  --remote=punix:/usr/local/var/run/openvswitch/db.sock \
  --remote=db:Open_vSwitch,Open_vSwitch,manager_options \
  --pidfile --detach

#Initialize database and start switch daemon:
sudo /usr/local/bin/ovs-vsctl --no-wait init
sudo /usr/local/sbin/ovs-vswitchd --pidfile --detach

#=======================================================================================
#Step 6: Load Open vSwitch Kernel Module
#Verify and load kernel module:
lsmod | grep openvswitch
sudo modprobe openvswitch

#=======================================================================================
#Step 7: Fix Socket Permissions (Development Environment)
#To allow access to OVS socket:
sudo chmod 660 /usr/local/var/run/openvswitch/db.sock

#Verify OVS:
sudo /usr/local/bin/ovs-vsctl show

#=======================================================================================
#Step 8: Install Mininet from Source
#Clone Mininet:
cd /opt
sudo git clone https://github.com/mininet/mininet.git
cd mininet

#Install Mininet Python package:
sudo python3 setup.py install

#(Optional) Ensure mn command is available:
sudo ln -s /opt/mininet/bin/mn /usr/bin/mn

#=======================================================================================
#Step 9: Build & install Mininet helper binaries (mnexec)
# From the Mininet directory:
cd /opt/mininet
sudo make install

 #This installs:
   # - /usr/bin/mnexec
   # - other helper binaries Mininet needs

# Verify:
which mnexec
  # Expected:
   # /usr/bin/mnexec


# Make Open vSwitch visible to Mininet cleanup scripts
# Create symlinks so /bin/sh can find OVS tools:

sudo ln -s /usr/local/bin/ovs-vsctl /usr/bin/ovs-vsctl
sudo ln -s /usr/local/bin/ovs-ofctl /usr/bin/ovs-ofctl

# Verify:
which ovs-vsctl
  # Expected:
    # /usr/bin/ovs-vsctl
    
#=======================================================================================
#Step 10: Verify Mininet Installation

# Check mnexec
mnexec --help

#Check Mininet version:
mn --version

#Run test topology:
sudo mn --test pingall
  #Expected result:
  # *** Results: 0% dropped


#=======================================================================================
...
Final Notes
This installation is fully compatible with Mininet + Open vSwitch
Suitable for:
- Cloud / datacenter topology emulation
- Link-down and congestion fault injection
- Traffic capture and dataset generation
- Recommended for research and thesis work (e.g., SOANN fault detection models)
...





















