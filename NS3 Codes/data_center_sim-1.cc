/*./ns3 run "scratch/data_center_sim --cores=2 --aggs=4 --tors=8 --servers=8 --time=5 --output=dc_results.csv"*/
/*
 * Data Center Network Simulation with Fat-Tree Topology
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

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DataCenterSimulation");

// Structure to store simulation results
struct SimulationResults {
    std::map<uint32_t, uint64_t> totalRxBytes; // Flow ID -> Total received bytes
    std::map<uint32_t, uint64_t> totalTxBytes; // Flow ID -> Total transmitted bytes
    std::map<uint32_t, uint64_t> lostPackets;  // Flow ID -> Lost packets
    std::map<uint32_t, Time> delaySum;         // Flow ID -> Total delay
    std::map<uint32_t, Time> jitterSum;        // Flow ID -> Total jitter
    std::map<uint32_t, uint64_t> rxPackets;    // Flow ID -> Received packets
    std::vector<double> packetSizes;           // All packet sizes
    std::vector<double> latencies;             // All latencies
    std::vector<double> jitters;              // All jitters
    double totalThroughput;
    double avgPacketLoss;
    double avgLatency;
    double avgJitter;
    uint64_t totalPackets;
};

class DataCenterSim {
private:
    uint32_t nCore;
    uint32_t nAgg;
    uint32_t nTor;
    uint32_t nServersPerTor;
    double simulationTime;
    NodeContainer coreSwitches;
    NodeContainer aggSwitches;
    NodeContainer torSwitches;
    NodeContainer servers;
    NetDeviceContainer serverDevices;
    Ipv4InterfaceContainer serverInterfaces;
    SimulationResults results;
    FlowMonitorHelper flowmonHelper; // Add this to fix the error
    
public:
    DataCenterSim(uint32_t cores = 2, uint32_t aggs = 4, uint32_t tors = 8, 
                  uint32_t serversPerTor = 8, double simTime = 10.0)
        : nCore(cores), nAgg(aggs), nTor(tors), nServersPerTor(serversPerTor), 
          simulationTime(simTime) {}
    
    void BuildTopology() {
        NS_LOG_INFO("Building Data Center Topology");
        NS_LOG_INFO("Cores: " << nCore << ", Aggs: " << nAgg << ", ToRs: " << nTor 
                    << ", Servers per ToR: " << nServersPerTor);
        
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
        
        PointToPointHelper p2pAggTor;
        p2pAggTor.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
        p2pAggTor.SetChannelAttribute("Delay", StringValue("2us"));
        
        PointToPointHelper p2pTorServer;
        p2pTorServer.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
        p2pTorServer.SetChannelAttribute("Delay", StringValue("1us"));
        
        // Create CSMA for server connections (alternative to p2p)
        CsmaHelper csma;
        csma.SetChannelAttribute("DataRate", StringValue("1Gbps"));
        csma.SetChannelAttribute("Delay", StringValue("1us"));
        
        NS_LOG_INFO("Creating Core-Aggregation Links");
        // Connect core to aggregation switches (full bisection bandwidth)
        Ipv4AddressHelper ipv4;
        std::vector<NetDeviceContainer> coreAggDevices;
        std::vector<Ipv4InterfaceContainer> coreAggInterfaces;
        
        for (uint32_t i = 0; i < nCore; i++) {
            for (uint32_t j = 0; j < nAgg; j++) {
                NetDeviceContainer link = p2pCoreAgg.Install(coreSwitches.Get(i), 
                                                           aggSwitches.Get(j));
                coreAggDevices.push_back(link);
                
                std::stringstream subnet;
                subnet << "10." << i + 1 << "." << j + 1 << ".0";
                ipv4.SetBase(subnet.str().c_str(), "255.255.255.0");
                coreAggInterfaces.push_back(ipv4.Assign(link));
            }
        }
        
        NS_LOG_INFO("Creating Aggregation-ToR Links");
        // Connect aggregation to ToR switches
        std::vector<NetDeviceContainer> aggTorDevices;
        std::vector<Ipv4InterfaceContainer> aggTorInterfaces;
        
        uint32_t torsPerAgg = nTor / nAgg;
        for (uint32_t agg = 0; agg < nAgg; agg++) {
            for (uint32_t tor = 0; tor < torsPerAgg; tor++) {
                uint32_t torIndex = agg * torsPerAgg + tor;
                if (torIndex < nTor) {
                    NetDeviceContainer link = p2pAggTor.Install(
                        aggSwitches.Get(agg), torSwitches.Get(torIndex));
                    aggTorDevices.push_back(link);
                    
                    std::stringstream subnet;
                    subnet << "20." << agg + 1 << "." << tor + 1 << ".0";
                    ipv4.SetBase(subnet.str().c_str(), "255.255.255.0");
                    aggTorInterfaces.push_back(ipv4.Assign(link));
                }
            }
        }
        
        NS_LOG_INFO("Creating ToR-Server Links");
        // Connect ToR switches to servers
        std::vector<Ipv4InterfaceContainer> torServerInterfaces;
        
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
            torServerInterfaces.push_back(serverIfaces);
            
            // Store interfaces for all servers
            for (uint32_t srv = 0; srv < nServersPerTor; srv++) {
                serverInterfaces.Add(serverIfaces.Get(srv));
            }
        }
        
        // Populate routing tables
        Ipv4GlobalRoutingHelper::PopulateRoutingTables();
        
        NS_LOG_INFO("Topology built successfully");
    }
    
    void InstallApplications() {
        NS_LOG_INFO("Installing Applications");
        
        // Create random traffic between servers
        uint32_t port = 9;  // Discard port (RFC 863)
        Time interPacketInterval = Seconds(0.001);  // 1ms interval
        
        // Install packet sink on all servers
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory", 
                                   InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApps = sinkHelper.Install(servers);
        sinkApps.Start(Seconds(0.0));
        sinkApps.Stop(Seconds(simulationTime));
        
        // Create traffic between random server pairs
        Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
        
        for (uint32_t i = 0; i < servers.GetN(); i++) {
            // Each server sends to 2 other random servers
            for (int j = 0; j < 2; j++) {
                uint32_t dstIndex;
                do {
                    dstIndex = uv->GetInteger(0, servers.GetN() - 1);
                } while (dstIndex == i);
                
                // Create UDP application
                OnOffHelper onoff("ns3::UdpSocketFactory", 
                                 InetSocketAddress(serverInterfaces.GetAddress(dstIndex), port));
                
                // Random packet size between 64 and 1500 bytes
                onoff.SetAttribute("PacketSize", UintegerValue(uv->GetInteger(64, 1500)));
                onoff.SetAttribute("DataRate", DataRateValue(DataRate("10Mbps")));
                onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
                onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
                onoff.SetAttribute("MaxBytes", UintegerValue(0));  // Unlimited
                
                ApplicationContainer apps = onoff.Install(servers.Get(i));
                
                // Start with random offset
                double startTime = uv->GetValue(0.1, 1.0);
                apps.Start(Seconds(startTime));
                apps.Stop(Seconds(simulationTime - 0.1));
            }
        }
        
        NS_LOG_INFO("Applications installed");
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
        
        results.totalPackets = 0;
        results.totalThroughput = 0.0;
        double totalDelay = 0.0;
        double totalJitter = 0.0;
        uint64_t totalLostPackets = 0;
        uint64_t totalRxPackets = 0;
        
        for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator it = stats.begin();
             it != stats.end(); ++it) {
            FlowId flowId = it->first;
            FlowMonitor::FlowStats flowStats = it->second;
            
            if (flowStats.rxPackets > 0) {
                // Store flow statistics
                results.totalRxBytes[flowId] = flowStats.rxBytes;
                results.totalTxBytes[flowId] = flowStats.txBytes;
                results.lostPackets[flowId] = flowStats.lostPackets;
                results.delaySum[flowId] = flowStats.delaySum;
                results.jitterSum[flowId] = flowStats.jitterSum;
                results.rxPackets[flowId] = flowStats.rxPackets;
                
                // Calculate per-packet metrics
                for (uint32_t i = 0; i < flowStats.packetsDropped.size(); i++) {
                    if (flowStats.packetsDropped[i] > 0) {
                        totalLostPackets += flowStats.packetsDropped[i];
                    }
                }
                
                // Calculate average metrics for this flow
                double flowThroughput = flowStats.rxBytes * 8.0 / 
                                      (simulationTime - 1.0) / 1000000.0; // Mbps
                double flowLatency = flowStats.delaySum.GetSeconds() / flowStats.rxPackets;
                double flowJitter = flowStats.jitterSum.GetSeconds() / flowStats.rxPackets;
                
                // Store packet-level data (sample)
                for (uint32_t i = 0; i < std::min((uint32_t)10, flowStats.rxPackets); i++) {
                    results.packetSizes.push_back(flowStats.rxBytes * 1.0 / flowStats.rxPackets);
                    results.latencies.push_back(flowLatency);
                    results.jitters.push_back(flowJitter);
                }
                
                // Accumulate totals
                results.totalThroughput += flowThroughput;
                totalDelay += flowLatency * flowStats.rxPackets;
                totalJitter += flowJitter * flowStats.rxPackets;
                totalRxPackets += flowStats.rxPackets;
                results.totalPackets += flowStats.rxPackets;
            }
        }
        
        // Calculate averages
        if (totalRxPackets > 0) {
            results.avgLatency = totalDelay / totalRxPackets;
            results.avgJitter = totalJitter / totalRxPackets;
            results.avgPacketLoss = (totalLostPackets * 100.0) / 
                                  (totalRxPackets + totalLostPackets);
        }
        
        NS_LOG_INFO("Statistics collected");
    }
    
    void ExportResultsToCSV(const std::string& filename) {
        NS_LOG_INFO("Exporting results to CSV: " << filename);
        
        std::ofstream outFile(filename);
        
        // Write header
        outFile << "Metric,Value,Unit,Description\n";
        
        // Write summary statistics
        outFile << "Total Throughput," << results.totalThroughput << ",Mbps,Total network throughput\n";
        outFile << "Average Latency," << results.avgLatency * 1000 << ",ms,Average packet latency\n";
        outFile << "Average Jitter," << results.avgJitter * 1000 << ",ms,Average packet jitter\n";
        outFile << "Packet Loss Rate," << results.avgPacketLoss << ",%,Average packet loss rate\n";
        outFile << "Total Packets," << results.totalPackets << ",count,Total packets received\n";
        
        // Write per-flow statistics section
        outFile << "\nFlow Statistics\n";
        outFile << "FlowID,Throughput(Mbps),AvgLatency(ms),AvgJitter(ms),PacketLoss(%),TxBytes,RxBytes\n";
        
        for (auto& flow : results.totalRxBytes) {
            uint32_t flowId = flow.first;
            double throughput = results.totalRxBytes[flowId] * 8.0 / 
                              (simulationTime - 1.0) / 1000000.0;
            double latency = results.delaySum[flowId].GetSeconds() / 
                           results.rxPackets[flowId] * 1000;
            double jitter = results.jitterSum[flowId].GetSeconds() / 
                          results.rxPackets[flowId] * 1000;
            double lossRate = (results.lostPackets[flowId] * 100.0) / 
                            (results.rxPackets[flowId] + results.lostPackets[flowId]);
            
            outFile << flowId << "," 
                   << throughput << ","
                   << latency << ","
                   << jitter << ","
                   << lossRate << ","
                   << results.totalTxBytes[flowId] << ","
                   << results.totalRxBytes[flowId] << "\n";
        }
        
        // Write packet-level data sample
        outFile << "\nPacket-Level Sample Data\n";
        outFile << "Sample#,PacketSize(bytes),Latency(ms),Jitter(ms)\n";
        
        for (size_t i = 0; i < results.packetSizes.size(); i++) {
            outFile << i + 1 << ","
                   << results.packetSizes[i] << ","
                   << results.latencies[i] * 1000 << ","
                   << results.jitters[i] * 1000 << "\n";
        }
        
        // Calculate and write network load and bandwidth usage
        outFile << "\nNetwork Load Analysis\n";
        outFile << "Metric,Value,Description\n";
        
        // Estimated bandwidth usage (simplified)
        double totalBandwidth = (nCore * nAgg * 40 + nAgg * (nTor/nAgg) * 10 + 
                               nTor * nServersPerTor * 1) * 1000; // Convert to Mbps
        double bandwidthUsage = (results.totalThroughput / totalBandwidth) * 100;
        
        outFile << "Total Available Bandwidth," << totalBandwidth << ",Mbps,Sum of all link capacities\n";
        outFile << "Bandwidth Usage," << bandwidthUsage << ",%,Percentage of total bandwidth used\n";
        outFile << "Network Load," << results.totalThroughput << ",Mbps,Actual load on network\n";
        
        outFile.close();
        NS_LOG_INFO("Results exported to " << filename);
        
        // Also print summary to console
        PrintSummary();
    }
    
    void PrintSummary() {
        std::cout << "\n========================================\n";
        std::cout << "DATA CENTER SIMULATION RESULTS SUMMARY\n";
        std::cout << "========================================\n";
        std::cout << "Topology: " << nCore << " Core, " << nAgg << " Agg, " 
                  << nTor << " ToR, " << (nTor * nServersPerTor) << " Servers\n";
        std::cout << "Simulation Time: " << simulationTime << " seconds\n";
        std::cout << "\nPerformance Metrics:\n";
        std::cout << "  Total Throughput: " << results.totalThroughput << " Mbps\n";
        std::cout << "  Average Latency: " << results.avgLatency * 1000 << " ms\n";
        std::cout << "  Average Jitter: " << results.avgJitter * 1000 << " ms\n";
        std::cout << "  Packet Loss Rate: " << results.avgPacketLoss << " %\n";
        std::cout << "  Total Packets: " << results.totalPackets << "\n";
        std::cout << "========================================\n";
    }
    
    void Run() {
        BuildTopology();
        InstallApplications();
        RunSimulation();
    }
    
    SimulationResults GetResults() const { return results; }
};

int main(int argc, char *argv[]) {
    // Enable logging
    LogComponentEnable("DataCenterSimulation", LOG_LEVEL_INFO);
    // LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);
    // LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
    
    // Simulation parameters
    uint32_t nCore = 2;
    uint32_t nAgg = 4;
    uint32_t nTor = 8;
    uint32_t nServersPerTor = 8;
    double simulationTime = 10.0;
    std::string outputFile = "data_center_results.csv";
    
    // Parse command line arguments
    CommandLine cmd;
    cmd.AddValue("cores", "Number of core switches", nCore);
    cmd.AddValue("aggs", "Number of aggregation switches", nAgg);
    cmd.AddValue("tors", "Number of ToR switches", nTor);
    cmd.AddValue("servers", "Number of servers per ToR", nServersPerTor);
    cmd.AddValue("time", "Simulation time in seconds", simulationTime);
    cmd.AddValue("output", "Output CSV file name", outputFile);
    cmd.Parse(argc, argv);
    
    NS_LOG_INFO("Starting Data Center Simulation");
    NS_LOG_INFO("Output file: " << outputFile);
    
    // Create and run simulation
    DataCenterSim sim(nCore, nAgg, nTor, nServersPerTor, simulationTime);
    sim.Run();
    sim.ExportResultsToCSV(outputFile);
    
    return 0;
}
