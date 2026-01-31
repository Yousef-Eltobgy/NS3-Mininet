#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/csma-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("EnhancedDataCenterSim");

// Function to setup TCP traffic
void SetupTcpTraffic(Ptr<Node> src, Ptr<Node> dst, uint16_t port, 
                     double startTime, double duration, uint32_t maxBytes = 0) {
    // Install packet sink on destination
    PacketSinkHelper sink("ns3::TcpSocketFactory", 
                         InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApp = sink.Install(dst);
    sinkApp.Start(Seconds(startTime));
    sinkApp.Stop(Seconds(startTime + duration));
    
    // Install bulk send on source
    Ptr<Ipv4> ipv4 = dst->GetObject<Ipv4>();
    Ipv4Address dstAddr = ipv4->GetAddress(1, 0).GetLocal();
    
    BulkSendHelper source("ns3::TcpSocketFactory", 
                         InetSocketAddress(dstAddr, port));
    source.SetAttribute("MaxBytes", UintegerValue(maxBytes));
    source.SetAttribute("SendSize", UintegerValue(1440)); // Typical MSS
    
    ApplicationContainer sourceApp = source.Install(src);
    sourceApp.Start(Seconds(startTime + 0.1));
    sourceApp.Stop(Seconds(startTime + duration - 0.1));
}

// Function to setup UDP traffic
void SetupUdpTraffic(Ptr<Node> src, Ptr<Node> dst, uint16_t port,
                     double startTime, double duration, 
                     uint32_t packetSize = 1024, double interval = 0.01) {
    // Install UDP echo server on destination
    UdpEchoServerHelper echoServer(port);
    ApplicationContainer serverApps = echoServer.Install(dst);
    serverApps.Start(Seconds(startTime));
    serverApps.Stop(Seconds(startTime + duration));
    
    // Install UDP echo client on source
    Ptr<Ipv4> ipv4 = dst->GetObject<Ipv4>();
    Ipv4Address dstAddr = ipv4->GetAddress(1, 0).GetLocal();
    
    UdpEchoClientHelper echoClient(dstAddr, port);
    echoClient.SetAttribute("MaxPackets", UintegerValue(UINT32_MAX));
    echoClient.SetAttribute("Interval", TimeValue(Seconds(interval)));
    echoClient.SetAttribute("PacketSize", UintegerValue(packetSize));
    
    ApplicationContainer clientApps = echoClient.Install(src);
    clientApps.Start(Seconds(startTime + 0.1));
    clientApps.Stop(Seconds(startTime + duration - 0.1));
}

// Enhanced statistics printing
void PrintEnhancedStatistics(Ptr<FlowMonitor> monitor, 
                            Ptr<FlowMonitorHelper> flowmonHelper,
                            double simulationTime) {
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(
        flowmonHelper->GetClassifier());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();
    
    double totalThroughput = 0;
    uint64_t totalTxPackets = 0;
    uint64_t totalRxPackets = 0;
    uint64_t totalLostPackets = 0;
    uint64_t totalTxBytes = 0;
    uint64_t totalRxBytes = 0;
    
    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << "              ENHANCED FLOW STATISTICS\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    
    for (auto it = stats.begin(); it != stats.end(); ++it) {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        
        double flowDuration = simulationTime - 2.0; // Account for startup delay
        if (flowDuration <= 0) flowDuration = simulationTime;
        
        double throughput = it->second.rxBytes * 8.0 / flowDuration / 1000000; // Mbps
        double lossRate = (it->second.lostPackets * 100.0) / it->second.txPackets;
        double avgDelay = it->second.rxPackets > 0 ? 
                         it->second.delaySum.GetSeconds() / it->second.rxPackets * 1000 : 0;
        double avgJitter = it->second.rxPackets > 1 ?
                          it->second.jitterSum.GetSeconds() / (it->second.rxPackets - 1) * 1000 : 0;
        
        std::cout << "\n┌─ Flow " << it->first << " ";
        std::cout << (t.protocol == 6 ? "[TCP]" : "[UDP]") << "\n";
        std::cout << "├─ Source:      " << t.sourceAddress << ":" << t.sourcePort << "\n";
        std::cout << "├─ Destination: " << t.destinationAddress << ":" << t.destinationPort << "\n";
        std::cout << "├─ Tx Packets:  " << it->second.txPackets << "\n";
        std::cout << "├─ Rx Packets:  " << it->second.rxPackets << "\n";
        std::cout << "├─ Lost:        " << it->second.lostPackets 
                  << " (" << std::fixed << std::setprecision(2) << lossRate << "%)\n";
        std::cout << "├─ Throughput:  " << std::setprecision(4) << throughput << " Mbps\n";
        std::cout << "├─ Avg Delay:   " << std::setprecision(4) << avgDelay << " ms\n";
        std::cout << "├─ Avg Jitter:  " << std::setprecision(4) << avgJitter << " ms\n";
        std::cout << "└─ Mean Pkt Sz: " << (it->second.rxBytes * 1.0 / it->second.rxPackets) 
                  << " bytes\n";
        
        totalThroughput += throughput;
        totalTxPackets += it->second.txPackets;
        totalRxPackets += it->second.rxPackets;
        totalLostPackets += it->second.lostPackets;
        totalTxBytes += it->second.txBytes;
        totalRxBytes += it->second.rxBytes;
    }
    
    // Summary Statistics
    double avgPacketLoss = totalTxPackets > 0 ? 
                          (totalLostPackets * 100.0) / totalTxPackets : 0;
    double networkUtilization = totalRxBytes * 8.0 / (simulationTime * 10e9) * 100; // Assuming 10Gbps
    
    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << "                  SUMMARY STATISTICS\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Total Flows:          " << stats.size() << "\n";
    std::cout << "Total Throughput:     " << std::setprecision(4) << totalThroughput << " Mbps\n";
    std::cout << "Aggregate Goodput:    " << std::setprecision(4) 
              << (totalRxBytes * 8.0 / simulationTime / 1000000) << " Mbps\n";
    std::cout << "Total Packets Sent:   " << totalTxPackets << "\n";
    std::cout << "Total Packets Recv:   " << totalRxPackets << "\n";
    std::cout << "Total Packet Loss:    " << totalLostPackets 
              << " (" << std::setprecision(2) << avgPacketLoss << "%)\n";
    std::cout << "Network Utilization:  " << std::setprecision(2) 
              << networkUtilization << "% of 10Gbps links\n";
    std::cout << "Average Flow Throughput: " << std::setprecision(4) 
              << (totalThroughput / stats.size()) << " Mbps\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
}

int main(int argc, char *argv[]) {
    // Enable logging for debugging (comment out for clean output)
    // LogComponentEnable("EnhancedDataCenterSim", LOG_LEVEL_INFO);
    
    // Command Line Configuration
    CommandLine cmd;
    uint32_t nAgg = 2;
    uint32_t nTor = 4;
    uint32_t nServersPerTor = 4;
    double simulationTime = 30.0;
    double linkDelay = 2.0; // ms
    std::string dataRate = "10Gbps";
    std::string trafficPattern = "mixed"; // mixed, eastwest, northsouth, alltoall
    bool enableTcp = true;
    bool enableUdp = true;
    uint32_t tcpMaxBytes = 10000000; // 10MB per TCP flow
    
    cmd.AddValue("agg", "Number of aggregation switches", nAgg);
    cmd.AddValue("tor", "Number of ToR switches", nTor);
    cmd.AddValue("servers", "Servers per ToR", nServersPerTor);
    cmd.AddValue("time", "Simulation time (seconds)", simulationTime);
    cmd.AddValue("delay", "Link delay (ms)", linkDelay);
    cmd.AddValue("rate", "Link data rate", dataRate);
    cmd.AddValue("pattern", "Traffic pattern (mixed/eastwest/northsouth/alltoall)", trafficPattern);
    cmd.AddValue("tcp", "Enable TCP traffic", enableTcp);
    cmd.AddValue("udp", "Enable UDP traffic", enableUdp);
    cmd.AddValue("tcpbytes", "Max bytes per TCP flow (0=unlimited)", tcpMaxBytes);
    
    cmd.Parse(argc, argv);
    
    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << "        DATA CENTER SIMULATION CONFIGURATION\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Topology:           " << nAgg << " Agg × " << nTor << " ToR × " 
              << nServersPerTor << " servers/ToR\n";
    std::cout << "Total Servers:      " << (nTor * nServersPerTor) << "\n";
    std::cout << "Simulation Time:    " << simulationTime << " seconds\n";
    std::cout << "Link Rate:          " << dataRate << "\n";
    std::cout << "Link Delay:         " << linkDelay << " ms\n";
    std::cout << "Traffic Pattern:    " << trafficPattern << "\n";
    std::cout << "TCP Enabled:        " << (enableTcp ? "Yes" : "No") << "\n";
    std::cout << "UDP Enabled:        " << (enableUdp ? "Yes" : "No") << "\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    
    // Create nodes
    Ptr<Node> core = CreateObject<Node>();  // Single core switch
    NodeContainer aggSwitches;
    aggSwitches.Create(nAgg);
    NodeContainer torSwitches;
    torSwitches.Create(nTor);
    
    uint32_t totalServers = nTor * nServersPerTor;
    NodeContainer servers;
    servers.Create(totalServers);
    
    NodeContainer allNodes;
    allNodes.Add(core);
    allNodes.Add(aggSwitches);
    allNodes.Add(torSwitches);
    allNodes.Add(servers);
    
    // Install internet stack
    InternetStackHelper internet;
    internet.Install(allNodes);
    
    // Configure point-to-point links
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue(dataRate));
    p2p.SetChannelAttribute("Delay", StringValue(std::to_string(linkDelay) + "ms"));
    
    Ipv4AddressHelper address;
    std::vector<NetDeviceContainer> allDevices;
    
    std::cout << "\nCreating network topology..." << std::endl;
    
    // Core to Aggregation links
    std::cout << "  Creating Core → Aggregation links..." << std::endl;
    address.SetBase("10.0.0.0", "255.255.255.0");
    for (uint32_t i = 0; i < nAgg; i++) {
        NetDeviceContainer link = p2p.Install(core, aggSwitches.Get(i));
        allDevices.push_back(link);
        Ipv4InterfaceContainer iface = address.Assign(link);
        address.NewNetwork();
    }
    
    // Aggregation to ToR links (full bisection bandwidth)
    std::cout << "  Creating Aggregation → ToR links..." << std::endl;
    address.SetBase("10.1.0.0", "255.255.255.0");
    for (uint32_t i = 0; i < nAgg; i++) {
        for (uint32_t j = 0; j < nTor / nAgg; j++) {
            uint32_t torIndex = i * (nTor / nAgg) + j;
            if (torIndex < nTor) {
                NetDeviceContainer link = p2p.Install(aggSwitches.Get(i), torSwitches.Get(torIndex));
                allDevices.push_back(link);
                Ipv4InterfaceContainer iface = address.Assign(link);
                address.NewNetwork();
            }
        }
    }
    
    // ToR to Server links
    std::cout << "  Creating ToR → Server links..." << std::endl;
    address.SetBase("10.2.0.0", "255.255.255.0");
    for (uint32_t i = 0; i < nTor; i++) {
        for (uint32_t j = 0; j < nServersPerTor; j++) {
            uint32_t serverIndex = i * nServersPerTor + j;
            if (serverIndex < totalServers) {
                NetDeviceContainer link = p2p.Install(torSwitches.Get(i), servers.Get(serverIndex));
                allDevices.push_back(link);
                Ipv4InterfaceContainer iface = address.Assign(link);
                address.NewNetwork();
            }
        }
    }
    
    // Enable routing
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    
    std::cout << "\nSetting up traffic patterns..." << std::endl;
    
    // Setup traffic based on selected pattern
    uint16_t basePort = 5000;
    
    if (trafficPattern == "eastwest" || trafficPattern == "mixed") {
        // East-West traffic (within same rack)
        std::cout << "  Creating East-West traffic (same rack)..." << std::endl;
        for (uint32_t tor = 0; tor < nTor; tor++) {
            for (uint32_t s = 0; s < nServersPerTor - 1; s++) {
                uint32_t srcIndex = tor * nServersPerTor + s;
                uint32_t dstIndex = tor * nServersPerTor + s + 1;
                
                if (enableTcp) {
                    SetupTcpTraffic(servers.Get(srcIndex), servers.Get(dstIndex),
                                  basePort++, 1.0 + tor * 0.5, simulationTime - 5, tcpMaxBytes);
                }
                if (enableUdp) {
                    SetupUdpTraffic(servers.Get(dstIndex), servers.Get(srcIndex),
                                  basePort++, 2.0 + tor * 0.5, simulationTime - 6);
                }
            }
        }
    }
    
    if (trafficPattern == "northsouth" || trafficPattern == "mixed") {
        // North-South traffic (cross rack)
        std::cout << "  Creating North-South traffic (cross rack)..." << std::endl;
        for (uint32_t tor1 = 0; tor1 < nTor; tor1++) {
            for (uint32_t tor2 = tor1 + 1; tor2 < nTor; tor2++) {
                uint32_t srcIndex = tor1 * nServersPerTor;
                uint32_t dstIndex = tor2 * nServersPerTor;
                
                if (enableTcp) {
                    SetupTcpTraffic(servers.Get(srcIndex), servers.Get(dstIndex),
                                  basePort++, 3.0, simulationTime - 8, tcpMaxBytes);
                }
                if (enableUdp) {
                    SetupUdpTraffic(servers.Get(dstIndex), servers.Get(srcIndex),
                                  basePort++, 4.0, simulationTime - 9);
                }
            }
        }
    }
    
    if (trafficPattern == "alltoall") {
        // All-to-all traffic (heavy)
        std::cout << "  Creating All-to-All traffic (heavy load)..." << std::endl;
        for (uint32_t i = 0; i < totalServers; i++) {
            for (uint32_t j = i + 1; j < totalServers; j++) {
                if (enableTcp) {
                    SetupTcpTraffic(servers.Get(i), servers.Get(j),
                                  basePort++, 1.0 + (i+j)*0.1, 
                                  simulationTime - 2, tcpMaxBytes/10);
                }
            }
        }
    }
    
    // Always add some background traffic
    std::cout << "  Adding background traffic..." << std::endl;
    if (enableUdp) {
        // Background UDP traffic from first to last server
        SetupUdpTraffic(servers.Get(0), servers.Get(totalServers - 1),
                      9000, 0.5, simulationTime - 1, 512, 0.02);
    }
    
    // Enable Flow Monitor
    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> flowMonitor = flowmonHelper.InstallAll();
    
    // Run simulation
    std::cout << "\nStarting simulation for " << simulationTime << " seconds..." << std::endl;
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    
    // Output flow statistics
    PrintEnhancedStatistics(flowMonitor, &flowmonHelper, simulationTime);
    
    // Optional: Export to XML for further analysis
    flowMonitor->SerializeToXmlFile("datacenter-enhanced-flow.xml", true, true);
    std::cout << "\nDetailed flow data exported to: datacenter-enhanced-flow.xml\n";
    
    Simulator::Destroy();
    
    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << "            SIMULATION COMPLETE\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    
    return 0;
}


