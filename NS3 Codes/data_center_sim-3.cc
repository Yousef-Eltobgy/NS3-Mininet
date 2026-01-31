/*./ns3 run "scratch/data_center_sim --time=20 --traffic=extreme --load=5.0 --output=extreme_20s.csv" */
/*
 * Data Center Network Simulation with Congestion Scenarios
 * Exports: Packet size, latency, jitter, packet loss, throughput, network load, bandwidth usage
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/csma-module.h"
#include "ns3/bridge-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DataCenterSimulation");

// Structure to store detailed simulation results
struct SimulationResults {
    // Packet sample structure
    struct PacketSample {
        double size;    // in bytes
        double latency; // in seconds
        double jitter;  // in seconds
        uint32_t flowId;
    };
    
    // Flow-level metrics
    std::map<uint32_t, double> flowThroughput;
    std::map<uint32_t, double> flowLatency;
    std::map<uint32_t, double> flowJitter;
    std::map<uint32_t, double> flowPacketLoss;
    std::map<uint32_t, uint64_t> flowTxPackets;
    std::map<uint32_t, uint64_t> flowRxPackets;
    std::map<uint32_t, uint64_t> flowTxBytes;
    std::map<uint32_t, uint64_t> flowRxBytes;
    
    // Packet-level samples
    std::vector<PacketSample> packetSamples;
    
    // Summary statistics
    double totalThroughput;    // Mbps
    double avgLatency;         // seconds
    double avgJitter;          // seconds
    double avgPacketLoss;      // percentage
    double minLatency;         // seconds
    double maxLatency;         // seconds
    double stdDevLatency;      // seconds
    uint64_t totalPackets;
    uint64_t totalLostPackets;
    uint64_t totalTxBytes;
    uint64_t totalRxBytes;
    
    // Network load metrics
    double bandwidthUtilization;  // percentage
    double networkLoad;           // Mbps
    double availableBandwidth;    // Mbps
    
    // Constructor
    SimulationResults() {
        totalThroughput = 0.0;
        avgLatency = 0.0;
        avgJitter = 0.0;
        avgPacketLoss = 0.0;
        minLatency = std::numeric_limits<double>::max();
        maxLatency = 0.0;
        stdDevLatency = 0.0;
        totalPackets = 0;
        totalLostPackets = 0;
        totalTxBytes = 0;
        totalRxBytes = 0;
        bandwidthUtilization = 0.0;
        networkLoad = 0.0;
        availableBandwidth = 0.0;
    }
};

class DataCenterSim {
private:
    uint32_t nCore;
    uint32_t nAgg;
    uint32_t nTor;
    uint32_t nServersPerTor;
    double simulationTime;
    std::string outputFile;
    std::string trafficPattern;
    double loadFactor;
    NodeContainer coreSwitches;
    NodeContainer aggSwitches;
    NodeContainer torSwitches;
    NodeContainer servers;
    NetDeviceContainer serverDevices;
    Ipv4InterfaceContainer serverInterfaces;
    SimulationResults results;
    FlowMonitorHelper flowmonHelper;
    
public:
    DataCenterSim(uint32_t cores = 2, uint32_t aggs = 4, uint32_t tors = 8, 
                  uint32_t serversPerTor = 8, double simTime = 10.0,
                  const std::string& outFile = "data_center_results.csv",
                  const std::string& traffic = "normal",
                  double load = 1.0)
        : nCore(cores), nAgg(aggs), nTor(tors), nServersPerTor(serversPerTor), 
          simulationTime(simTime), outputFile(outFile), trafficPattern(traffic),
          loadFactor(load) {}
    
    void BuildTopology() {
        NS_LOG_INFO("Building Data Center Topology");
        NS_LOG_INFO("Cores: " << nCore << ", Aggs: " << nAgg << ", ToRs: " << nTor 
                    << ", Servers per ToR: " << nServersPerTor);
        NS_LOG_INFO("Traffic Pattern: " << trafficPattern << ", Load Factor: " << loadFactor);
        
        // Create nodes
        coreSwitches.Create(nCore);
        aggSwitches.Create(nAgg);
        torSwitches.Create(nTor);
        servers.Create(nTor * nServersPerTor);
        
        // Install internet stack on all nodes
        InternetStackHelper stack;
        stack.Install(coreSwitches);
        stack.Install(aggSwitches);
        stack.Install(torSwitches);
        stack.Install(servers);
        
        // Create point-to-point helpers for different links
        PointToPointHelper p2pCoreAgg;
        p2pCoreAgg.SetDeviceAttribute("DataRate", StringValue("40Gbps"));
        p2pCoreAgg.SetChannelAttribute("Delay", StringValue("5us"));
        p2pCoreAgg.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("1000p"));
        
        PointToPointHelper p2pAggTor;
        p2pAggTor.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
        p2pAggTor.SetChannelAttribute("Delay", StringValue("2us"));
        p2pAggTor.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("500p"));
        
        PointToPointHelper p2pTorServer;
        p2pTorServer.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
        p2pTorServer.SetChannelAttribute("Delay", StringValue("1us"));
        p2pTorServer.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("100p"));
        
        // Calculate total available bandwidth
        CalculateAvailableBandwidth();
        
        // Create CSMA for server connections
        CsmaHelper csma;
        csma.SetChannelAttribute("DataRate", StringValue("1Gbps"));
        csma.SetChannelAttribute("Delay", StringValue("1us"));
        
        NS_LOG_INFO("Creating Core-Aggregation Links");
        // Connect core to aggregation switches
        Ipv4AddressHelper ipv4;
        for (uint32_t i = 0; i < nCore; i++) {
            for (uint32_t j = 0; j < nAgg; j++) {
                NetDeviceContainer link = p2pCoreAgg.Install(coreSwitches.Get(i), 
                                                           aggSwitches.Get(j));
                std::stringstream subnet;
                subnet << "10." << i + 1 << "." << j + 1 << ".0";
                ipv4.SetBase(subnet.str().c_str(), "255.255.255.0");
                ipv4.Assign(link);
            }
        }
        
        NS_LOG_INFO("Creating Aggregation-ToR Links");
        // Connect aggregation to ToR switches
        uint32_t torsPerAgg = nTor / nAgg;
        for (uint32_t agg = 0; agg < nAgg; agg++) {
            for (uint32_t tor = 0; tor < torsPerAgg; tor++) {
                uint32_t torIndex = agg * torsPerAgg + tor;
                if (torIndex < nTor) {
                    NetDeviceContainer link = p2pAggTor.Install(
                        aggSwitches.Get(agg), torSwitches.Get(torIndex));
                    std::stringstream subnet;
                    subnet << "20." << agg + 1 << "." << tor + 1 << ".0";
                    ipv4.SetBase(subnet.str().c_str(), "255.255.255.0");
                    ipv4.Assign(link);
                }
            }
        }
        
        NS_LOG_INFO("Creating ToR-Server Links");
        // Connect ToR switches to servers
        for (uint32_t tor = 0; tor < nTor; tor++) {
            NodeContainer torServerGroup;
            for (uint32_t srv = 0; srv < nServersPerTor; srv++) {
                uint32_t serverIndex = tor * nServersPerTor + srv;
                torServerGroup.Add(servers.Get(serverIndex));
            }
            
            // Connect ToR to its servers using CSMA
            NetDeviceContainer torDev = csma.Install(torSwitches.Get(tor));
            NetDeviceContainer serverDev = csma.Install(torServerGroup);
            
            // Create bridge on ToR switch
            BridgeHelper bridge;
            bridge.Install(torSwitches.Get(tor), torDev);
            
            // Add server devices to container
            for (uint32_t srv = 0; srv < nServersPerTor; srv++) {
                serverDevices.Add(serverDev.Get(srv));
            }
            
            // Assign IP addresses to servers
            std::stringstream subnet;
            subnet << "30." << (tor / 256) + 1 << "." << (tor % 256) + 1 << ".0";
            ipv4.SetBase(subnet.str().c_str(), "255.255.255.0");
            Ipv4InterfaceContainer serverIfaces = ipv4.Assign(serverDev);
            
            // Store interfaces for all servers
            for (uint32_t srv = 0; srv < nServersPerTor; srv++) {
                serverInterfaces.Add(serverIfaces.Get(srv));
            }
        }
        
        // Populate routing tables
        Ipv4GlobalRoutingHelper::PopulateRoutingTables();
        
        NS_LOG_INFO("Topology built successfully");
    }
    
    void CalculateAvailableBandwidth() {
        // Core-Agg links: nCore * nAgg * 40Gbps
        double coreAggBandwidth = nCore * nAgg * 40 * 1000; // Convert Gbps to Mbps
        
        // Agg-Tor links: nAgg * (nTor/nAgg) * 10Gbps
        uint32_t torsPerAgg = nTor / nAgg;
        double aggTorBandwidth = nAgg * torsPerAgg * 10 * 1000;
        
        // Tor-Server links: nTor * nServersPerTor * 1Gbps
        double torServerBandwidth = nTor * nServersPerTor * 1 * 1000;
        
        // Use bottleneck as available bandwidth (simplified)
        results.availableBandwidth = std::min({coreAggBandwidth, aggTorBandwidth, torServerBandwidth});
        
        NS_LOG_INFO("Available Bandwidth: " << results.availableBandwidth << " Mbps");
    }
    
    void InstallNormalApplications() {
        NS_LOG_INFO("Installing Normal Applications");
        
        uint32_t port = 9;
        
        // Install packet sink on all servers
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory", 
                                   InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApps = sinkHelper.Install(servers);
        sinkApps.Start(Seconds(0.0));
        sinkApps.Stop(Seconds(simulationTime));
        
        Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
        
        for (uint32_t i = 0; i < servers.GetN(); i++) {
            uint32_t numConnections = uv->GetInteger(1, 4);
            
            for (uint32_t conn = 0; conn < numConnections; conn++) {
                uint32_t dstIndex;
                do {
                    dstIndex = uv->GetInteger(0, servers.GetN() - 1);
                } while (dstIndex == i);
                
                OnOffHelper onoff("ns3::UdpSocketFactory", 
                                 InetSocketAddress(serverInterfaces.GetAddress(dstIndex), port));
                
                uint32_t packetSizeOptions[] = {64, 512, 1024, 1500};
                uint32_t packetSize = packetSizeOptions[uv->GetInteger(0, 3)];
                onoff.SetAttribute("PacketSize", UintegerValue(packetSize));
                
                double dataRateMbps = uv->GetValue(1.0, 100.0) * loadFactor;
                std::stringstream dataRate;
                dataRate << dataRateMbps << "Mbps";
                onoff.SetAttribute("DataRate", DataRateValue(DataRate(dataRate.str())));
                
                onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=0.1]"));
                onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.05]"));
                onoff.SetAttribute("MaxBytes", UintegerValue(0));
                
                ApplicationContainer apps = onoff.Install(servers.Get(i));
                
                double startTime = uv->GetValue(0.1, 1.0);
                apps.Start(Seconds(startTime));
                apps.Stop(Seconds(simulationTime - 0.1));
            }
        }
    }
    
    void InstallHighLoadApplications() {
        NS_LOG_INFO("Installing High Load Applications");
        
        uint32_t port = 9;
        
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory", 
                                   InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApps = sinkHelper.Install(servers);
        sinkApps.Start(Seconds(0.0));
        sinkApps.Stop(Seconds(simulationTime));
        
        Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
        
        // Create hotspot: servers 0-7 communicate heavily with servers 56-63
        for (uint32_t i = 0; i < 8; i++) {
            for (uint32_t j = 56; j < 64; j++) {
                if (i != j) {
                    OnOffHelper onoff("ns3::UdpSocketFactory", 
                                     InetSocketAddress(serverInterfaces.GetAddress(j), port));
                    
                    onoff.SetAttribute("PacketSize", UintegerValue(1500));
                    double dataRateMbps = 1000.0 * loadFactor; // 1Gbps per flow
                    std::stringstream dataRate;
                    dataRate << dataRateMbps << "Mbps";
                    onoff.SetAttribute("DataRate", DataRateValue(DataRate(dataRate.str())));
                    
                    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
                    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
                    onoff.SetAttribute("MaxBytes", UintegerValue(0));
                    
                    ApplicationContainer apps = onoff.Install(servers.Get(i));
                    apps.Start(Seconds(1.0));
                    apps.Stop(Seconds(simulationTime - 1.0));
                }
            }
        }
        
        // Add background traffic
        for (uint32_t i = 8; i < servers.GetN(); i++) {
            uint32_t numConnections = uv->GetInteger(1, 3);
            
            for (uint32_t conn = 0; conn < numConnections; conn++) {
                uint32_t dstIndex;
                do {
                    dstIndex = uv->GetInteger(0, servers.GetN() - 1);
                } while (dstIndex == i);
                
                OnOffHelper onoff("ns3::UdpSocketFactory", 
                                 InetSocketAddress(serverInterfaces.GetAddress(dstIndex), port));
                
                onoff.SetAttribute("PacketSize", UintegerValue(1024));
                double dataRateMbps = uv->GetValue(10.0, 100.0) * loadFactor;
                std::stringstream dataRate;
                dataRate << dataRateMbps << "Mbps";
                onoff.SetAttribute("DataRate", DataRateValue(DataRate(dataRate.str())));
                
                onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=0.2]"));
                onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.1]"));
                onoff.SetAttribute("MaxBytes", UintegerValue(0));
                
                ApplicationContainer apps = onoff.Install(servers.Get(i));
                
                double startTime = uv->GetValue(0.5, 2.0);
                apps.Start(Seconds(startTime));
                apps.Stop(Seconds(simulationTime - 0.5));
            }
        }
    }
    
    void InstallBurstyApplications() {
        NS_LOG_INFO("Installing Bursty Applications");
        
        uint32_t port = 9;
        
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory", 
                                   InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApps = sinkHelper.Install(servers);
        sinkApps.Start(Seconds(0.0));
        sinkApps.Stop(Seconds(simulationTime));
        
        Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
        
        for (uint32_t i = 0; i < servers.GetN(); i++) {
            uint32_t numConnections = uv->GetInteger(2, 6);
            
            for (uint32_t conn = 0; conn < numConnections; conn++) {
                uint32_t dstIndex;
                do {
                    dstIndex = uv->GetInteger(0, servers.GetN() - 1);
                } while (dstIndex == i);
                
                OnOffHelper onoff("ns3::UdpSocketFactory", 
                                 InetSocketAddress(serverInterfaces.GetAddress(dstIndex), port));
                
                onoff.SetAttribute("PacketSize", UintegerValue(1500));
                double dataRateMbps = uv->GetValue(500.0, 2000.0) * loadFactor; // High burst rates
                std::stringstream dataRate;
                dataRate << dataRateMbps << "Mbps";
                onoff.SetAttribute("DataRate", DataRateValue(DataRate(dataRate.str())));
                
                // Very bursty: short on times, longer off times
                onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=0.05]"));
                onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.2]"));
                onoff.SetAttribute("MaxBytes", UintegerValue(0));
                
                ApplicationContainer apps = onoff.Install(servers.Get(i));
                
                double startTime = uv->GetValue(0.1, 0.5);
                apps.Start(Seconds(startTime));
                apps.Stop(Seconds(simulationTime - 0.1));
            }
        }
    }
    
    void InstallExtremeHighLoadApplications() {
        NS_LOG_INFO("Installing Extreme High Load Applications");
        
        uint32_t port = 9;
        
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory", 
                                   InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApps = sinkHelper.Install(servers);
        sinkApps.Start(Seconds(0.0));
        sinkApps.Stop(Seconds(simulationTime));
        
        Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
        
        // EXTREME HOTSPOT: All servers in first ToR talk to all servers in last ToR
        uint32_t firstTorStart = 0;
        uint32_t firstTorEnd = nServersPerTor - 1;
        uint32_t lastTorStart = (nTor - 1) * nServersPerTor;
        uint32_t lastTorEnd = nTor * nServersPerTor - 1;
        
        // Create massive all-to-all traffic between these two groups
        for (uint32_t i = firstTorStart; i <= firstTorEnd; i++) {
            for (uint32_t j = lastTorStart; j <= lastTorEnd; j++) {
                if (i != j) {
                    OnOffHelper onoff("ns3::UdpSocketFactory", 
                                     InetSocketAddress(serverInterfaces.GetAddress(j), port));
                    
                    onoff.SetAttribute("PacketSize", UintegerValue(1500));
                    double dataRateMbps = 900.0 * loadFactor; // Almost full 1Gbps per flow
                    std::stringstream dataRate;
                    dataRate << dataRateMbps << "Mbps";
                    onoff.SetAttribute("DataRate", DataRateValue(DataRate(dataRate.str())));
                    
                    // Continuous traffic
                    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
                    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
                    onoff.SetAttribute("MaxBytes", UintegerValue(0));
                    
                    ApplicationContainer apps = onoff.Install(servers.Get(i));
                    apps.Start(Seconds(1.0));
                    apps.Stop(Seconds(simulationTime - 1.0));
                }
            }
        }
        
        // Add more UDP background traffic
        for (uint32_t i = 0; i < servers.GetN(); i++) {
            if (i >= firstTorStart && i <= firstTorEnd) continue; // Skip already loaded servers
            if (i >= lastTorStart && i <= lastTorEnd) continue;   // Skip already loaded servers
            
            uint32_t numConnections = uv->GetInteger(2, 5);
            
            for (uint32_t conn = 0; conn < numConnections; conn++) {
                uint32_t dstIndex;
                do {
                    dstIndex = uv->GetInteger(0, servers.GetN() - 1);
                } while (dstIndex == i || 
                        (dstIndex >= firstTorStart && dstIndex <= firstTorEnd) ||
                        (dstIndex >= lastTorStart && dstIndex <= lastTorEnd));
                
                OnOffHelper onoff("ns3::UdpSocketFactory", 
                                 InetSocketAddress(serverInterfaces.GetAddress(dstIndex), port));
                
                onoff.SetAttribute("PacketSize", UintegerValue(1024));
                double dataRateMbps = uv->GetValue(200.0, 500.0) * loadFactor;
                std::stringstream dataRate;
                dataRate << dataRateMbps << "Mbps";
                onoff.SetAttribute("DataRate", DataRateValue(DataRate(dataRate.str())));
                
                onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=0.3]"));
                onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.1]"));
                onoff.SetAttribute("MaxBytes", UintegerValue(0));
                
                ApplicationContainer apps = onoff.Install(servers.Get(i));
                
                double startTime = uv->GetValue(0.5, 1.5);
                apps.Start(Seconds(startTime));
                apps.Stop(Seconds(simulationTime - 0.5));
            }
        }
    }
    
    void InstallApplications() {
        if (trafficPattern == "highload") {
            InstallHighLoadApplications();
        } else if (trafficPattern == "bursty") {
            InstallBurstyApplications();
        } else if (trafficPattern == "extreme") {
            InstallExtremeHighLoadApplications();
        } else {
            InstallNormalApplications();
        }
    }
    
    void RunSimulation() {
        NS_LOG_INFO("Starting Simulation");
        
        // Enable flow monitoring
        Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll();
        
        // Run simulation
        Simulator::Stop(Seconds(simulationTime));
        Simulator::Run();
        
        // Collect statistics
        CollectStatistics(monitor);
        
        Simulator::Destroy();
        NS_LOG_INFO("Simulation completed");
    }
    
    void CollectStatistics(Ptr<FlowMonitor> monitor) {
        NS_LOG_INFO("Collecting Statistics");
        
        monitor->CheckForLostPackets();
        Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(
            flowmonHelper.GetClassifier());
        
        std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();
        
        std::vector<double> allLatencies;
        results.totalTxBytes = 0;
        results.totalRxBytes = 0;
        results.totalPackets = 0;
        results.totalLostPackets = 0;
        results.totalThroughput = 0.0;
        double totalLatencySum = 0.0;
        double totalJitterSum = 0.0;
        
        for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator it = stats.begin();
             it != stats.end(); ++it) {
            FlowId flowId = it->first;
            FlowMonitor::FlowStats flowStats = it->second;
            
            if (flowStats.rxPackets > 0) {
                double flowThroughput = flowStats.rxBytes * 8.0 / 
                                      (simulationTime - 1.0) / 1000000.0;
                double flowLatency = flowStats.delaySum.GetSeconds() / flowStats.rxPackets;
                double flowJitter = flowStats.jitterSum.GetSeconds() / flowStats.rxPackets;
                double flowPacketLoss = (flowStats.lostPackets * 100.0) / 
                                      (flowStats.rxPackets + flowStats.lostPackets);
                
                results.flowThroughput[flowId] = flowThroughput;
                results.flowLatency[flowId] = flowLatency;
                results.flowJitter[flowId] = flowJitter;
                results.flowPacketLoss[flowId] = flowPacketLoss;
                results.flowTxPackets[flowId] = flowStats.txPackets;
                results.flowRxPackets[flowId] = flowStats.rxPackets;
                results.flowTxBytes[flowId] = flowStats.txBytes;
                results.flowRxBytes[flowId] = flowStats.rxBytes;
                
                // Collect packet samples (up to 20 per flow)
                for (uint32_t i = 0; i < std::min((uint32_t)20, flowStats.rxPackets); i++) {
                    SimulationResults::PacketSample sample;
                    sample.size = flowStats.rxBytes * 1.0 / flowStats.rxPackets;
                    sample.latency = flowLatency;
                    sample.jitter = flowJitter;
                    sample.flowId = flowId;
                    results.packetSamples.push_back(sample);
                }
                
                // Update min/max latency
                if (flowLatency < results.minLatency) {
                    results.minLatency = flowLatency;
                }
                if (flowLatency > results.maxLatency) {
                    results.maxLatency = flowLatency;
                }
                
                // Store for standard deviation calculation
                for (uint32_t i = 0; i < flowStats.rxPackets; i++) {
                    allLatencies.push_back(flowLatency);
                }
                
                // Accumulate totals
                results.totalThroughput += flowThroughput;
                totalLatencySum += flowLatency * flowStats.rxPackets;
                totalJitterSum += flowJitter * flowStats.rxPackets;
                results.totalTxBytes += flowStats.txBytes;
                results.totalRxBytes += flowStats.rxBytes;
                results.totalPackets += flowStats.rxPackets;
                results.totalLostPackets += flowStats.lostPackets;
            }
        }
        
        // Calculate averages
        if (results.totalPackets > 0) {
            results.avgLatency = totalLatencySum / results.totalPackets;
            results.avgJitter = totalJitterSum / results.totalPackets;
            results.avgPacketLoss = (results.totalLostPackets * 100.0) / 
                                  (results.totalPackets + results.totalLostPackets);
            
            // Calculate standard deviation of latency
            double variance = 0.0;
            for (double latency : allLatencies) {
                double diff = latency - results.avgLatency;
                variance += diff * diff;
            }
            if (allLatencies.size() > 1) {
                results.stdDevLatency = std::sqrt(variance / (allLatencies.size() - 1));
            }
        }
        
        // Calculate network load metrics
        results.networkLoad = results.totalThroughput;
        results.bandwidthUtilization = (results.totalThroughput / results.availableBandwidth) * 100.0;
        
        NS_LOG_INFO("Statistics collected");
    }
    
    void ExportResultsToCSV() {
        NS_LOG_INFO("Exporting results to CSV: " << outputFile);
        
        std::ofstream outFile(outputFile);
        
        outFile << "===== DATA CENTER SIMULATION RESULTS =====\n";
        outFile << "Topology: " << nCore << " Core, " << nAgg << " Agg, " 
                << nTor << " ToR, " << (nTor * nServersPerTor) << " Servers\n";
        outFile << "Simulation Time: " << simulationTime << " seconds\n";
        outFile << "Traffic Pattern: " << trafficPattern << "\n";
        outFile << "Load Factor: " << loadFactor << "\n\n";
        
        outFile << "1. SUMMARY STATISTICS\n";
        outFile << "Metric,Value,Unit\n";
        outFile << "Total Throughput," << results.totalThroughput << ",Mbps\n";
        outFile << "Average Latency," << results.avgLatency * 1000 << ",ms\n";
        outFile << "Minimum Latency," << results.minLatency * 1000 << ",ms\n";
        outFile << "Maximum Latency," << results.maxLatency * 1000 << ",ms\n";
        outFile << "Latency Std Dev," << results.stdDevLatency * 1000 << ",ms\n";
        outFile << "Average Jitter," << results.avgJitter * 1000 << ",ms\n";
        outFile << "Packet Loss Rate," << results.avgPacketLoss << ",%\n";
        outFile << "Total Packets," << results.totalPackets << ",count\n";
        outFile << "Total Lost Packets," << results.totalLostPackets << ",count\n";
        outFile << "Total Transmitted Bytes," << results.totalTxBytes << ",bytes\n";
        outFile << "Total Received Bytes," << results.totalRxBytes << ",bytes\n";
        outFile << "Packet Delivery Ratio," 
                << (100.0 * results.totalPackets / (results.totalPackets + results.totalLostPackets)) 
                << ",%\n\n";
        
        outFile << "2. NETWORK LOAD ANALYSIS\n";
        outFile << "Metric,Value,Unit\n";
        outFile << "Available Bandwidth," << results.availableBandwidth << ",Mbps\n";
        outFile << "Network Load," << results.networkLoad << ",Mbps\n";
        outFile << "Bandwidth Utilization," << results.bandwidthUtilization << ",%\n";
        outFile << "Network Load Factor," << (results.networkLoad / results.availableBandwidth) << ",ratio\n\n";
        
        outFile << "3. FLOW-LEVEL STATISTICS\n";
        outFile << "FlowID,Throughput(Mbps),Latency(ms),Jitter(ms),PacketLoss(%),"
                << "TxPackets,RxPackets,TxBytes,RxBytes\n";
        
        for (auto& flow : results.flowThroughput) {
            uint32_t flowId = flow.first;
            outFile << flowId << ","
                   << results.flowThroughput[flowId] << ","
                   << results.flowLatency[flowId] * 1000 << ","
                   << results.flowJitter[flowId] * 1000 << ","
                   << results.flowPacketLoss[flowId] << ","
                   << results.flowTxPackets[flowId] << ","
                   << results.flowRxPackets[flowId] << ","
                   << results.flowTxBytes[flowId] << ","
                   << results.flowRxBytes[flowId] << "\n";
        }
        
        outFile << "\n4. PACKET-LEVEL SAMPLE DATA\n";
        outFile << "SampleID,FlowID,PacketSize(bytes),Latency(ms),Jitter(ms)\n";
        
        for (size_t i = 0; i < results.packetSamples.size(); i++) {
            const SimulationResults::PacketSample& sample = results.packetSamples[i];
            outFile << i + 1 << ","
                   << sample.flowId << ","
                   << sample.size << ","
                   << sample.latency * 1000 << ","
                   << sample.jitter * 1000 << "\n";
        }
        
        outFile.close();
        NS_LOG_INFO("Results exported to " << outputFile);
        
        PrintSummary();
    }
    
    void PrintSummary() {
        std::cout << "\n========================================\n";
        std::cout << "DATA CENTER SIMULATION RESULTS SUMMARY\n";
        std::cout << "========================================\n";
        std::cout << "Topology: " << nCore << " Core, " << nAgg << " Agg, " 
                  << nTor << " ToR, " << (nTor * nServersPerTor) << " Servers\n";
        std::cout << "Simulation Time: " << simulationTime << " seconds\n";
        std::cout << "Traffic Pattern: " << trafficPattern << "\n";
        std::cout << "Load Factor: " << loadFactor << "\n";
        std::cout << "Output File: " << outputFile << "\n";
        std::cout << "\nPERFORMANCE METRICS:\n";
        std::cout << "  Total Throughput: " << results.totalThroughput << " Mbps\n";
        std::cout << "  Network Load: " << results.networkLoad << " Mbps\n";
        std::cout << "  Bandwidth Utilization: " << results.bandwidthUtilization << " %\n";
        std::cout << "  Average Latency: " << results.avgLatency * 1000 << " ms\n";
        std::cout << "  Min/Max Latency: " << results.minLatency * 1000 << "/" 
                  << results.maxLatency * 1000 << " ms\n";
        std::cout << "  Latency Std Dev: " << results.stdDevLatency * 1000 << " ms\n";
        std::cout << "  Average Jitter: " << results.avgJitter * 1000 << " ms\n";
        std::cout << "  Packet Loss Rate: " << results.avgPacketLoss << " %\n";
        std::cout << "  Total Packets: " << results.totalPackets << "\n";
        std::cout << "  Packet Delivery Ratio: " 
                  << (100.0 * results.totalPackets / (results.totalPackets + results.totalLostPackets)) 
                  << " %\n";
        std::cout << "========================================\n";
    }
    
    void Run() {
        BuildTopology();
        InstallApplications();
        RunSimulation();
        ExportResultsToCSV();
    }
};

int main(int argc, char *argv[]) {
    // Enable logging
    LogComponentEnable("DataCenterSimulation", LOG_LEVEL_INFO);
    
    // Simulation parameters with defaults
    uint32_t nCore = 2;
    uint32_t nAgg = 4;
    uint32_t nTor = 8;
    uint32_t nServersPerTor = 8;
    double simulationTime = 10.0;
    std::string outputFile = "data_center_results.csv";
    std::string trafficPattern = "normal";
    double loadFactor = 1.0;
    
    // Parse command line arguments using CommandLine
    CommandLine cmd(__FILE__);
    cmd.AddValue("cores", "Number of core switches", nCore);
    cmd.AddValue("aggs", "Number of aggregation switches", nAgg);
    cmd.AddValue("tors", "Number of ToR switches", nTor);
    cmd.AddValue("servers", "Number of servers per ToR", nServersPerTor);
    cmd.AddValue("time", "Simulation time in seconds", simulationTime);
    cmd.AddValue("output", "Output CSV file name", outputFile);
    cmd.AddValue("traffic", "Traffic pattern (normal/highload/bursty/extreme)", trafficPattern);
    cmd.AddValue("load", "Load factor multiplier", loadFactor);
    cmd.Parse(argc, argv);
    
    NS_LOG_INFO("Starting Data Center Simulation");
    NS_LOG_INFO("Traffic Pattern: " << trafficPattern);
    NS_LOG_INFO("Load Factor: " << loadFactor);
    NS_LOG_INFO("Output file: " << outputFile);
    
    // Create and run simulation
    DataCenterSim sim(nCore, nAgg, nTor, nServersPerTor, simulationTime, 
                      outputFile, trafficPattern, loadFactor);
    sim.Run();
    
    return 0;
}
