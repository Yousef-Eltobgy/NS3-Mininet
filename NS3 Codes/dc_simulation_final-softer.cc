#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/rng-seed-manager.h"

#include <fstream>
#include <map>
#include <vector>
#include <sstream>
#include <unordered_set>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DCFaultDatasetFinal");

/* ===================== CONFIGURATION ===================== */
static const uint32_t N_CORE = 2;
static const uint32_t N_AGG  = 4;
static const uint32_t N_TOR  = 8;
static const uint32_t N_SRV  = 8;

/* ===================== DATA STRUCTURES ===================== */
struct LinkInfo {
    Ptr<PointToPointNetDevice> devA;
    Ptr<PointToPointNetDevice> devB;
    Ptr<QueueDisc> qA;
    Ptr<QueueDisc> qB;
    
    uint64_t prevTxBytes = 0;
    uint64_t prevRxBytes = 0;
    uint64_t prevDrops = 0;
    uint32_t linkId;
    bool isFaulty = false;
};

struct FlowInfo {
    uint64_t prevTxBytes = 0;
    uint64_t prevRxBytes = 0;
    uint64_t prevTxPackets = 0;
    uint64_t prevRxPackets = 0;
    uint64_t prevLostPackets = 0;
    Time prevDelaySum = Seconds(0.0);
    Time prevJitterSum = Seconds(0.0);
    uint32_t label = 0; // 0=normal, 1=link_failure, 2=congestion
};

/* ===================== GLOBALS ===================== */
std::vector<LinkInfo> links;
std::vector<NodeContainer> serverGroups;
std::map<FlowId, FlowInfo> flowInfo;
std::unordered_set<uint32_t> faultyLinks;

std::ofstream flowCsv;
std::ofstream linkCsv;

Ptr<UniformRandomVariable> g_random = CreateObject<UniformRandomVariable>();

/* ===================== SIMPLIFIED FAULT INJECTION ===================== */
void InjectLinkFailure(uint32_t linkId) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " failure at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    // The simplest approach: just mark as faulty
    link.isFaulty = true;
    faultyLinks.insert(linkId);
    
    // In ns-3, to actually break a link, we need to disable the netdevice
    // But this can cause routing issues. Instead, we'll simulate it by
    // injecting packet loss on that link
    
    Ptr<RateErrorModel> errorModel = CreateObject<RateErrorModel>();
    errorModel->SetAttribute("ErrorRate", DoubleValue(1.0)); // 100% loss
    errorModel->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
    
    link.devA->SetReceiveErrorModel(errorModel);
    link.devB->SetReceiveErrorModel(errorModel);
}

void RestoreLinkFailure(uint32_t linkId) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " restored at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    link.isFaulty = false;
    faultyLinks.erase(linkId);
    
    // Remove error model
    link.devA->SetReceiveErrorModel(nullptr);
    link.devB->SetReceiveErrorModel(nullptr);
}

void InjectCongestion() {
    NS_LOG_UNCOND("[CONGESTION] Creating congestion traffic at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    // Create a few high-bandwidth flows
    for (int i = 0; i < 3; i++) {
        uint32_t srcTor = g_random->GetInteger(0, N_TOR - 1);
        uint32_t srcSrv = g_random->GetInteger(0, N_SRV - 1);
        
        uint32_t dstTor;
        do {
            dstTor = g_random->GetInteger(0, N_TOR - 1);
        } while (dstTor == srcTor);
        
        uint32_t dstSrv = g_random->GetInteger(0, N_SRV - 1);
        
        Ptr<Node> srcNode = serverGroups[srcTor].Get(srcSrv);
        Ptr<Node> dstNode = serverGroups[dstTor].Get(dstSrv);
        
        Ptr<Ipv4> dstIpv4 = dstNode->GetObject<Ipv4>();
        Ipv4Address dstAddr = dstIpv4->GetAddress(1, 0).GetLocal();
        
        uint16_t port = 10000 + i;
        
        // Install sink
        PacketSinkHelper sink("ns3::TcpSocketFactory",
                            InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApp = sink.Install(dstNode);
        sinkApp.Start(Seconds(0.0));
        sinkApp.Stop(Seconds(60.0));
        
        // Install bulk sender
        BulkSendHelper bulk("ns3::TcpSocketFactory",
                          InetSocketAddress(dstAddr, port));
        bulk.SetAttribute("MaxBytes", UintegerValue(0)); // Unlimited
        bulk.SetAttribute("SendSize", UintegerValue(1460));
        
        ApplicationContainer bulkApp = bulk.Install(srcNode);
        bulkApp.Start(Seconds(0.0));
        bulkApp.Stop(Seconds(60.0));
    }
}

/* ===================== STATISTICS COLLECTION ===================== */
void ExportFlowStatistics(Ptr<FlowMonitor> monitor,
                         Ptr<Ipv4FlowClassifier> classifier,
                         double windowStart,
                         double windowEnd) {
    
    monitor->CheckForLostPackets();
    auto stats = monitor->GetFlowStats();
    
    for (auto &flow : stats) {
        FlowId id = flow.first;
        auto curr = flow.second;
        
        if (flowInfo.find(id) == flowInfo.end()) {
            flowInfo[id] = FlowInfo();
        }
        
        auto &info = flowInfo[id];
        
        // Calculate deltas
        uint64_t txPkts = curr.txPackets - info.prevTxPackets;
        uint64_t rxPkts = curr.rxPackets - info.prevRxPackets;
        uint64_t lost = curr.lostPackets - info.prevLostPackets;
        uint64_t rxBytes = curr.rxBytes - info.prevRxBytes;
        
        // Calculate metrics
        double delay = 0.0;
        double jitter = 0.0;
        
        if (rxPkts > 0) {
            delay = (curr.delaySum.GetSeconds() - info.prevDelaySum.GetSeconds()) 
                   / rxPkts * 1000.0;
        }
        if (rxPkts > 1) {
            jitter = (curr.jitterSum.GetSeconds() - info.prevJitterSum.GetSeconds()) 
                    / (rxPkts - 1) * 1000.0;
        }
        
        double windowDuration = windowEnd - windowStart;
        double throughput = 0.0;
        if (windowDuration > 0) {
            throughput = (rxBytes * 8.0) / (windowDuration * 1e6); // Mbps
        }
        
        // Get flow details
        auto tuple = classifier->FindFlow(id);
        
        // Simple label detection
        uint32_t label = 0; // normal
        
        // If throughput is very low, mark as faulty
        if (throughput < 1.0 && windowStart > 10.0) { // After warmup
            label = 1; // Assume link failure
        }
        // If high loss rate
        else if (lost > 0 && txPkts > 0 && (double)lost/txPkts > 0.1) {
            label = 2; // Congestion/packet loss
        }
        
        info.label = label;
        
        // Export to CSV
        flowCsv << std::fixed << windowStart << ","
                << id << ","
                << tuple.sourceAddress << ","
                << tuple.destinationAddress << ","
                << txPkts << ","
                << rxPkts << ","
                << lost << ","
                << throughput << ","
                << delay << ","
                << jitter << ","
                << label
                << "\n";
        
        // Update previous values
        info.prevTxBytes = curr.txBytes;
        info.prevRxBytes = curr.rxBytes;
        info.prevTxPackets = curr.txPackets;
        info.prevRxPackets = curr.rxPackets;
        info.prevLostPackets = curr.lostPackets;
        info.prevDelaySum = curr.delaySum;
        info.prevJitterSum = curr.jitterSum;
    }
}

void ExportLinkStatistics(double timestamp) {
    for (size_t i = 0; i < links.size(); i++) {
        auto &link = links[i];
        
        // Get queue statistics
        uint64_t drops = 0;
        if (link.qA && link.qB) {
            drops = link.qA->GetStats().nTotalDroppedPackets + 
                   link.qB->GetStats().nTotalDroppedPackets;
        }
        
        uint64_t dDrops = drops - link.prevDrops;
        
        uint32_t queueDepth = 0;
        if (link.qA && link.qB) {
            queueDepth = link.qA->GetNPackets() + link.qB->GetNPackets();
        }
        
        // Determine link status
        uint32_t status = 0; // normal
        if (link.isFaulty) {
            status = 1;
        } else if (queueDepth > 50) {
            status = 2; // congestion
        } else if (dDrops > 0) {
            status = 3; // dropping
        }
        
        // Simplified statistics
        linkCsv << std::fixed << timestamp << ","
                << i << ","
                << 0 << "," // txBytes placeholder
                << 0 << "," // rxBytes placeholder  
                << dDrops << ","
                << queueDepth << ","
                << 0 << "," // utilization placeholder
                << status
                << "\n";
        
        link.prevDrops = drops;
    }
}

/* ===================== MAIN ===================== */
int main(int argc, char *argv[]) {
    // Command line configuration
    uint32_t randomSeed = 1;
    double simulationTime = 30.0; // SHORTER for testing
    std::string outputPrefix = "dc_dataset";
    
    CommandLine cmd(__FILE__);
    cmd.AddValue("seed", "Random seed for reproducibility", randomSeed);
    cmd.AddValue("time", "Simulation time in seconds", simulationTime);
    cmd.AddValue("output", "Output file prefix", outputPrefix);
    cmd.Parse(argc, argv);
    
    // Set random seed
    RngSeedManager::SetSeed(randomSeed);
    g_random->SetAttribute("Min", DoubleValue(0.0));
    g_random->SetAttribute("Max", DoubleValue(1.0));
    
    Time::SetResolution(Time::NS);
    
    // Open output files
    flowCsv.open(outputPrefix + "_flows.csv");
    flowCsv << "time,flowId,srcIP,dstIP,txPkts,rxPkts,lostPkts,"
            << "throughputMbps,delayMs,jitterMs,label\n";
    
    linkCsv.open(outputPrefix + "_links.csv");
    linkCsv << "time,linkId,txBytes,rxBytes,drops,queueDepth,utilization,status\n";
    
    NS_LOG_UNCOND("========================================");
    NS_LOG_UNCOND("   Data Center Fault Dataset Generator  ");
    NS_LOG_UNCOND("========================================");
    NS_LOG_UNCOND("Seed: " << randomSeed);
    NS_LOG_UNCOND("Time: " << simulationTime << "s");
    NS_LOG_UNCOND("Output: " << outputPrefix << "_*.csv");
    NS_LOG_UNCOND("========================================");
    
    // Create nodes
    NodeContainer core, agg, tor;
    core.Create(N_CORE);
    agg.Create(N_AGG);
    tor.Create(N_TOR);
    
    serverGroups.resize(N_TOR);
    for (uint32_t i = 0; i < N_TOR; i++) {
        serverGroups[i].Create(N_SRV);
    }
    
    // Install internet stack with ECMP
    InternetStackHelper internet;
    Config::SetDefault("ns3::Ipv4GlobalRouting::RandomEcmpRouting", BooleanValue(true));
    internet.InstallAll();
    
    // Create P2P helper
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));
    
    // Traffic control
    TrafficControlHelper tch;
    tch.SetRootQueueDisc("ns3::FifoQueueDisc", "MaxSize", StringValue("100p"));
    
    // IP addressing
    Ipv4AddressHelper ip;
    uint32_t subnet = 0;
    
    // Install links
    auto InstallLink = [&](Ptr<Node> nodeA, Ptr<Node> nodeB) {
        NetDeviceContainer devices = p2p.Install(nodeA, nodeB);
        QueueDiscContainer queues = tch.Install(devices);
        
        LinkInfo link;
        link.devA = DynamicCast<PointToPointNetDevice>(devices.Get(0));
        link.devB = DynamicCast<PointToPointNetDevice>(devices.Get(1));
        link.qA = queues.Get(0);
        link.qB = queues.Get(1);
        link.linkId = links.size();
        
        links.push_back(link);
        
        std::ostringstream network;
        network << "10." << (subnet / 256) << "." << (subnet % 256) << ".0";
        ip.SetBase(network.str().c_str(), "255.255.255.0");
        ip.Assign(devices);
        
        subnet++;
    };
    
    // Core ↔ Aggregation
    for (uint32_t c = 0; c < N_CORE; c++) {
        for (uint32_t a = 0; a < N_AGG; a++) {
            InstallLink(core.Get(c), agg.Get(a));
        }
    }
    
    // Aggregation ↔ ToR
    uint32_t torsPerAgg = N_TOR / N_AGG;
    for (uint32_t a = 0; a < N_AGG; a++) {
        for (uint32_t t = 0; t < torsPerAgg; t++) {
            uint32_t torId = a * torsPerAgg + t;
            InstallLink(agg.Get(a), tor.Get(torId));
        }
    }
    
    // ToR ↔ Servers
    for (uint32_t t = 0; t < N_TOR; t++) {
        for (uint32_t s = 0; s < N_SRV; s++) {
            InstallLink(tor.Get(t), serverGroups[t].Get(s));
        }
    }
    
    // Enable routing
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    
    // Create realistic TCP traffic
    uint16_t basePort = 5000;
    uint32_t totalFlows = 0;
    
    for (uint32_t srcTor = 0; srcTor < N_TOR; srcTor++) {
        for (uint32_t srcSrv = 0; srcSrv < N_SRV; srcSrv++) {
            // Destination: different rack
            uint32_t dstTor = (srcTor + 1 + g_random->GetInteger(0, N_TOR - 2)) % N_TOR;
            uint32_t dstSrv = g_random->GetInteger(0, N_SRV - 1);
            
            Ptr<Node> srcNode = serverGroups[srcTor].Get(srcSrv);
            Ptr<Node> dstNode = serverGroups[dstTor].Get(dstSrv);
            
            Ptr<Ipv4> dstIpv4 = dstNode->GetObject<Ipv4>();
            Ipv4Address dstAddr = dstIpv4->GetAddress(1, 0).GetLocal();
            
            // Install PacketSink on destination
            PacketSinkHelper sink("ns3::TcpSocketFactory",
                                 InetSocketAddress(Ipv4Address::GetAny(), basePort));
            ApplicationContainer sinkApp = sink.Install(dstNode);
            sinkApp.Start(Seconds(1.0));
            sinkApp.Stop(Seconds(simulationTime - 1.0));
            
            // Install BulkSend on source
            BulkSendHelper source("ns3::TcpSocketFactory",
                                 InetSocketAddress(dstAddr, basePort));
            source.SetAttribute("MaxBytes", UintegerValue(5000000)); // 5MB per flow (SMALLER)
            source.SetAttribute("SendSize", UintegerValue(1460));
            
            ApplicationContainer sourceApp = source.Install(srcNode);
            double startTime = 1.0 + g_random->GetValue(0.0, 2.0); // Shorter range
            sourceApp.Start(Seconds(startTime));
            sourceApp.Stop(Seconds(simulationTime - 1.0));
            
            basePort++;
            totalFlows++;
        }
    }
    
    NS_LOG_UNCOND("Created " << totalFlows << " TCP flows");
    NS_LOG_UNCOND("Total links: " << links.size());
    
    // Setup Flow Monitor
    FlowMonitorHelper fmh;
    Ptr<FlowMonitor> monitor = fmh.InstallAll();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(fmh.GetClassifier());
    
    // Schedule simpler faults
    Simulator::Schedule(Seconds(5.0), []() {
        if (!links.empty()) {
            uint32_t linkId = g_random->GetInteger(0, links.size() - 1);
            InjectLinkFailure(linkId);
        }
    });
    
    Simulator::Schedule(Seconds(15.0), []() {
        if (!links.empty()) {
            uint32_t linkId = g_random->GetInteger(0, links.size() - 1);
            RestoreLinkFailure(linkId);
        }
    });
    
    Simulator::Schedule(Seconds(10.0), &InjectCongestion);
    
    // Schedule periodic statistics collection
    for (double t = 2.0; t <= simulationTime; t += 1.0) {
        Simulator::Schedule(Seconds(t), &ExportFlowStatistics,
                          monitor, classifier, t-1.0, t);
        Simulator::Schedule(Seconds(t), &ExportLinkStatistics, t);
    }
    
    // Add progress indicator
    for (double t = 5.0; t <= simulationTime; t += 5.0) {
        Simulator::Schedule(Seconds(t), []() {
            NS_LOG_UNCOND("[PROGRESS] Simulation time: " << Simulator::Now().GetSeconds() << "s");
        });
    }
    
    // Run simulation
    NS_LOG_UNCOND("\nStarting simulation...");
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    // Close files
    flowCsv.close();
    linkCsv.close();
    
    NS_LOG_UNCOND("\n========================================");
    NS_LOG_UNCOND("   Dataset Generation Complete!         ");
    NS_LOG_UNCOND("========================================");
    NS_LOG_UNCOND("Generated files:");
    NS_LOG_UNCOND("  1. " << outputPrefix << "_flows.csv - Flow statistics");
    NS_LOG_UNCOND("  2. " << outputPrefix << "_links.csv - Link statistics");
    NS_LOG_UNCOND("========================================");
    
    return 0;
}


