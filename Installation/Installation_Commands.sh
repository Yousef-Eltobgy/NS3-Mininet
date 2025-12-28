#!/bin/bash

# Install required dependencies:
sudo dnf update -y
sudo dnf install -y \
  gcc gcc-c++ \
  python3 python3-devel \
  git \
  cmake \
  make \
  wget \
  pkgconfig \
  sqlite sqlite-devel \
  libxml2 libxml2-devel \
  boost-devel \
  openssl-devel


#verify the installation:
gcc --version
python3 --version


# Download ns-3 (recommended: stable release):
cd ~
git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev

# Checkout a stable version (example: ns-3.42)
git checkout ns-3.42

# Configure ns-3
./ns3 configure --enable-examples --enable-tests

# Build ns-3
./ns3 build

# Test ns-3 installation:
./ns3 run hello-simulator
  # Expected output: Hello Simulator
  # this means ns-3 is installed correctly.


# Run a real simulation example
./ns3 run scratch/myfirst
  # or 
./ns3 run examples/tutorial/first



##########################################################
# (Optional) Enable visualization (NetAnim)
sudo dnf install -y qt5-qtbase-devel qt5-qtmultimedia-devel
  #then Reconfigure and rebuild:
./ns3 configure
./ns3 build


# Install NetAnim (visualizer)
cd ~
git clone https://gitlab.com/nsnam/netanim.git
cd netanim
qmake-qt5 NetAnim.pro
make
  # If qmake-qt5 is not found:
  sudo dnf install -y qt5-qttools-devel
  # Then rerun:
  qmake-qt5 NetAnim.pro
  make


# Launch NetAnim
./NetAnim
  # A GUI window should open.


# How NetAnim works with ns-3
  # Your ns-3 script generates an animation file:
  AnimationInterface anim("anim.xml");

  # Run the simulation:
  ./ns3 run scratch/my-simulation

  # Open the file in NetAnim:
  File → Open → anim.xml

# Quick sanity check
  # Run this:
  ldd NetAnim | grep Qt
    # If Qt libraries show up → everything is correct.




