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

NS_LOG_COMPONENT_DEFINE("DCFaultDatasetWorking");

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
    bool isCongested = false;
    double faultSeverity = 0.0;
};

struct FlowInfo {
    uint64_t prevTxBytes = 0;
    uint64_t prevRxBytes = 0;
    uint64_t prevTxPackets = 0;
    uint64_t prevRxPackets = 0;
    uint64_t prevLostPackets = 0;
    Time prevDelaySum = Seconds(0.0);
    Time prevJitterSum = Seconds(0.0);
    uint32_t label = 0;
    double baselineThroughput = 0.0;
    bool baselineSet = false;
    uint32_t consecutiveZeros = 0;
};

/* ===================== GLOBALS ===================== */
std::vector<LinkInfo> links;
std::vector<NodeContainer> serverGroups;
std::map<FlowId, FlowInfo> flowInfo;
std::unordered_set<uint32_t> faultyLinks;
std::unordered_set<uint32_t> congestedLinks;

std::ofstream flowCsv;
std::ofstream linkCsv;

Ptr<UniformRandomVariable> g_random = CreateObject<UniformRandomVariable>();

/* ===================== WORKING FAULT INJECTION ===================== */
void InjectRealisticLinkFailure(uint32_t linkId) {
    // REALISTIC: 50-80% packet loss, not 100%
    
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    double severity = 0.5 + g_random->GetValue() * 0.3; // 50-80% loss
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " realistic failure (" 
                  << (severity*100) << "% loss) at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    link.isFaulty = true;
    link.faultSeverity = severity;
    faultyLinks.insert(linkId);
    
    // Realistic error model - allows SOME packets through
    Ptr<RateErrorModel> errorModel = CreateObject<RateErrorModel>();
    errorModel->SetAttribute("ErrorRate", DoubleValue(severity));
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
    link.faultSeverity = 0.0;
    faultyLinks.erase(linkId);
    
    link.devA->SetReceiveErrorModel(nullptr);
    link.devB->SetReceiveErrorModel(nullptr);
}

void CreateBackgroundTraffic() {
    // Create background UDP traffic for congestion (not too aggressive)
    NS_LOG_UNCOND("[TRAFFIC] Creating background traffic at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    for (int i = 0; i < 20; i++) { // More flows, but lower rate
        uint32_t srcTor = g_random->GetInteger(0, N_TOR - 1);
        uint32_t srcSrv = g_random->GetInteger(0, N_SRV - 1);
        
        uint32_t dstTor = g_random->GetInteger(0, N_TOR - 1);
        uint32_t dstSrv = g_random->GetInteger(0, N_SRV - 1);
        
        Ptr<Node> srcNode = serverGroups[srcTor].Get(srcSrv);
        Ptr<Node> dstNode = serverGroups[dstTor].Get(dstSrv);
        
        Ptr<Ipv4> dstIpv4 = dstNode->GetObject<Ipv4>();
        Ipv4Address dstAddr = dstIpv4->GetAddress(1, 0).GetLocal();
        
        uint16_t port = 30000 + i;
        
        // Moderate UDP traffic
        OnOffHelper onoff("ns3::UdpSocketFactory", 
                         InetSocketAddress(dstAddr, port));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        onoff.SetAttribute("DataRate", StringValue("100Mbps")); // MODERATE rate
        onoff.SetAttribute("PacketSize", UintegerValue(1024));
        
        ApplicationContainer app = onoff.Install(srcNode);
        double start = 0.0;
        double duration = 60.0;
        app.Start(Seconds(start));
        app.Stop(Seconds(start + duration));
    }
}

/* ===================== WORKING STATISTICS COLLECTION ===================== */
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
        
        // SIMPLE LABELING based on time windows (NOT throughput ratio)
        uint32_t label = 0; // normal by default
        
        // Time-based labeling (most reliable)
        if (windowStart >= 20.0 && windowStart < 40.0) {
            label = 1; // Link failure period
        } else if (windowStart >= 50.0 && windowStart < 70.0) {
            label = 2; // Congestion period
        }
        // All other times: label 0 (normal)
        
        // Override: if throughput is very low during normal periods, mark as failure
        if (label == 0 && throughput < 1.0 && windowStart > 10.0) {
            label = 1;
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
        } else if (queueDepth > 10) {
            status = 2; // congestion
        } else if (dDrops > 0) {
            status = 3; // dropping
        }
        
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
    double simulationTime = 80.0; // Optimal length
    std::string outputPrefix = "dc_working";
    
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
    NS_LOG_UNCOND("   WORKING DC Fault Dataset Generator   ");
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
    
    // Set TCP parameters for better performance
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1448));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    
    internet.InstallAll();
    
    // Create P2P helper
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms")); // Slightly higher delay
    
    // Traffic control with reasonable queue
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
    
    // Core â†” Aggregation
    for (uint32_t c = 0; c < N_CORE; c++) {
        for (uint32_t a = 0; a < N_AGG; a++) {
            InstallLink(core.Get(c), agg.Get(a));
        }
    }
    
    // Aggregation â†” ToR
    uint32_t torsPerAgg = N_TOR / N_AGG;
    for (uint32_t a = 0; a < N_AGG; a++) {
        for (uint32_t t = 0; t < torsPerAgg; t++) {
            uint32_t torId = a * torsPerAgg + t;
            InstallLink(agg.Get(a), tor.Get(torId));
        }
    }
    
    // ToR â†” Servers
    for (uint32_t t = 0; t < N_TOR; t++) {
        for (uint32_t s = 0; s < N_SRV; s++) {
            InstallLink(tor.Get(t), serverGroups[t].Get(s));
        }
    }
    
    // Enable routing
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    
    // Create TCP traffic - FEWER but LONGER flows
    uint16_t basePort = 5000;
    uint32_t totalFlows = 0;
    
    for (uint32_t srcTor = 0; srcTor < N_TOR; srcTor++) {
        for (uint32_t srcSrv = 0; srcSrv < N_SRV; srcSrv++) {
            // Create ONE flow per server (not two)
            uint32_t dstTor = (srcTor + 1) % N_TOR; // Simple pattern
            uint32_t dstSrv = srcSrv; // Same server number
            
            Ptr<Node> srcNode = serverGroups[srcTor].Get(srcSrv);
            Ptr<Node> dstNode = serverGroups[dstTor].Get(dstSrv);
            
            Ptr<Ipv4> dstIpv4 = dstNode->GetObject<Ipv4>();
            Ipv4Address dstAddr = dstIpv4->GetAddress(1, 0).GetLocal();
            
            // Install PacketSink
            PacketSinkHelper sink("ns3::TcpSocketFactory",
                                 InetSocketAddress(Ipv4Address::GetAny(), basePort));
            ApplicationContainer sinkApp = sink.Install(dstNode);
            sinkApp.Start(Seconds(1.0));
            sinkApp.Stop(Seconds(simulationTime - 1.0));
            
            // Install BulkSend with LIMITED size
            BulkSendHelper source("ns3::TcpSocketFactory",
                                 InetSocketAddress(dstAddr, basePort));
            
            // Moderate flow size
            source.SetAttribute("MaxBytes", UintegerValue(20000000)); // 20MB
            
            source.SetAttribute("SendSize", UintegerValue(1448));
            
            ApplicationContainer sourceApp = source.Install(srcNode);
            double startTime = 1.0 + g_random->GetValue(0.0, 2.0);
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
    
    // Schedule CLEAR fault scenarios
    // Period 1: NORMAL (0-20s)
    // Period 2: LINK FAILURE (20-40s)
    // Period 3: NORMAL (40-50s) 
    // Period 4: CONGESTION (50-70s)
    // Period 5: NORMAL (70-80s)
    
    // Link failure at 20s
    Simulator::Schedule(Seconds(20.0), []() {
        if (!links.empty()) {
            // Fail 2 random links (not all)
            for (int i = 0; i < 2; i++) {
                uint32_t linkId = g_random->GetInteger(0, links.size() - 1);
                InjectRealisticLinkFailure(linkId);
            }
        }
    });
    
    // Restore at 40s
    Simulator::Schedule(Seconds(40.0), []() {
        for (uint32_t linkId = 0; linkId < links.size(); linkId++) {
            if (links[linkId].isFaulty) {
                RestoreLinkFailure(linkId);
            }
        }
    });
    
    // Background traffic for congestion at 50s
    Simulator::Schedule(Seconds(50.0), &CreateBackgroundTraffic);
    
    // Schedule periodic statistics collection
    for (double t = 2.0; t <= simulationTime; t += 1.0) {
        Simulator::Schedule(Seconds(t), &ExportFlowStatistics,
                          monitor, classifier, t-1.0, t);
        Simulator::Schedule(Seconds(t), &ExportLinkStatistics, t);
    }
    
    // Add progress indicator
    for (double t = 20.0; t <= simulationTime; t += 20.0) {
        Simulator::Schedule(Seconds(t), []() {
            NS_LOG_UNCOND("[PROGRESS] Simulation time: " << Simulator::Now().GetSeconds() << "s");
        });
    }
    
    // Run simulation
    NS_LOG_UNCOND("\nStarting WORKING simulation...");
    NS_LOG_UNCOND("Timeline:");
    NS_LOG_UNCOND("  0-20s:  NORMAL operation");
    NS_LOG_UNCOND("  20-40s: LINK FAILURE (2 random links)");
    NS_LOG_UNCOND("  40-50s: NORMAL operation");
    NS_LOG_UNCOND("  50-70s: CONGESTION (background traffic)");
    NS_LOG_UNCOND("  70-80s: NORMAL operation");
    
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
    NS_LOG_UNCOND("  1. " << outputPrefix << "_flows.csv");
    NS_LOG_UNCOND("  2. " << outputPrefix << "_links.csv");
    NS_LOG_UNCOND("\nExpected distribution:");
    NS_LOG_UNCOND("  Normal: ~40% (0-20s, 40-50s, 70-80s)");
    NS_LOG_UNCOND("  Link Failure: ~25% (20-40s)");
    NS_LOG_UNCOND("  Congestion: ~25% (50-70s)");
    NS_LOG_UNCOND("========================================");
    
    return 0;
}
