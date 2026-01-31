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

NS_LOG_COMPONENT_DEFINE("DCFaultDatasetPerfect");

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

/* ===================== PERFECT FAULT INJECTION ===================== */
void InjectSmartLinkFailure(uint32_t linkId) {
    // KEY FIX: Only apply error model to ONE direction
    // This allows TCP ACKs to get through
    
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    double severity = 0.7; // 70% loss in one direction
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " SMART failure (" 
                  << (severity*100) << "% loss, ONE direction) at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    link.isFaulty = true;
    link.faultSeverity = severity;
    faultyLinks.insert(linkId);
    
    // Apply error model to ONE direction only (devA -> devB)
    // This allows ACKs to flow in the other direction
    Ptr<RateErrorModel> errorModel = CreateObject<RateErrorModel>();
    errorModel->SetAttribute("ErrorRate", DoubleValue(severity));
    errorModel->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
    
    // CRITICAL: Only devA drops packets, devB works normally
    link.devA->SetReceiveErrorModel(errorModel);
    // link.devB->SetReceiveErrorModel(nullptr); // Already nullptr
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
}

void CreateBackgroundTraffic() {
    NS_LOG_UNCOND("[CONGESTION] Creating background traffic at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    // Create MANY small UDP flows
    for (int i = 0; i < 30; i++) {
        uint32_t srcTor = g_random->GetInteger(0, N_TOR - 1);
        uint32_t srcSrv = g_random->GetInteger(0, N_SRV - 1);
        
        uint32_t dstTor = g_random->GetInteger(0, N_TOR - 1);
        uint32_t dstSrv = g_random->GetInteger(0, N_SRV - 1);
        
        Ptr<Node> srcNode = serverGroups[srcTor].Get(srcSrv);
        Ptr<Node> dstNode = serverGroups[dstTor].Get(dstSrv);
        
        Ptr<Ipv4> dstIpv4 = dstNode->GetObject<Ipv4>();
        Ipv4Address dstAddr = dstIpv4->GetAddress(1, 0).GetLocal();
        
        uint16_t port = 40000 + i;
        
        // UDP traffic
        OnOffHelper onoff("ns3::UdpSocketFactory", 
                         InetSocketAddress(dstAddr, port));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        onoff.SetAttribute("DataRate", StringValue("50Mbps")); // Moderate
        onoff.SetAttribute("PacketSize", UintegerValue(512)); // Small packets
        
        ApplicationContainer app = onoff.Install(srcNode);
        app.Start(Seconds(0.0));
        app.Stop(Seconds(30.0));
    }
}

/* ===================== PERFECT STATISTICS COLLECTION ===================== */
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
        
        // SIMPLE, RELIABLE time-based labeling
        uint32_t label = 0; // normal by default
        
        if (windowStart >= 20.0 && windowStart < 40.0) {
            label = 1; // Link failure period
        } else if (windowStart >= 50.0 && windowStart < 70.0) {
            label = 2; // Congestion period
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
        
        uint32_t status = 0;
        if (link.isFaulty) {
            status = 1;
        } else if (link.isCongested) {
            status = 2;
        } else if (dDrops > 0) {
            status = 3;
        }
        
        linkCsv << std::fixed << timestamp << ","
                << i << ","
                << 0 << ","
                << 0 << ","  
                << dDrops << ","
                << queueDepth << ","
                << 0 << ","
                << status
                << "\n";
        
        link.prevDrops = drops;
    }
}

/* ===================== MAIN ===================== */
int main(int argc, char *argv[]) {
    uint32_t randomSeed = 1;
    double simulationTime = 80.0;
    std::string outputPrefix = "dc_perfect";
    
    CommandLine cmd(__FILE__);
    cmd.AddValue("seed", "Random seed", randomSeed);
    cmd.AddValue("time", "Simulation time", simulationTime);
    cmd.AddValue("output", "Output prefix", outputPrefix);
    cmd.Parse(argc, argv);
    
    RngSeedManager::SetSeed(randomSeed);
    g_random->SetAttribute("Min", DoubleValue(0.0));
    g_random->SetAttribute("Max", DoubleValue(1.0));
    
    Time::SetResolution(Time::NS);
    
    flowCsv.open(outputPrefix + "_flows.csv");
    flowCsv << "time,flowId,srcIP,dstIP,txPkts,rxPkts,lostPkts,"
            << "throughputMbps,delayMs,jitterMs,label\n";
    
    linkCsv.open(outputPrefix + "_links.csv");
    linkCsv << "time,linkId,txBytes,rxBytes,drops,queueDepth,utilization,status\n";
    
    NS_LOG_UNCOND("========================================");
    NS_LOG_UNCOND("   PERFECT DC Fault Dataset Generator   ");
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
    
    // Install internet
    InternetStackHelper internet;
    Config::SetDefault("ns3::Ipv4GlobalRouting::RandomEcmpRouting", BooleanValue(true));
    
    // OPTIMIZE TCP
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    Config::SetDefault("ns3::TcpSocketBase::MinRto", TimeValue(Seconds(0.5)));
    
    internet.InstallAll();
    
    // Create links
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    
    TrafficControlHelper tch;
    tch.SetRootQueueDisc("ns3::FifoQueueDisc", "MaxSize", StringValue("200p"));
    
    Ipv4AddressHelper ip;
    uint32_t subnet = 0;
    
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
    
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    
    // Create TCP traffic
    uint16_t basePort = 5000;
    uint32_t totalFlows = 0;
    
    for (uint32_t srcTor = 0; srcTor < N_TOR; srcTor++) {
        for (uint32_t srcSrv = 0; srcSrv < N_SRV; srcSrv++) {
            uint32_t dstTor = (srcTor + 1) % N_TOR;
            uint32_t dstSrv = srcSrv;
            
            Ptr<Node> srcNode = serverGroups[srcTor].Get(srcSrv);
            Ptr<Node> dstNode = serverGroups[dstTor].Get(dstSrv);
            
            Ptr<Ipv4> dstIpv4 = dstNode->GetObject<Ipv4>();
            Ipv4Address dstAddr = dstIpv4->GetAddress(1, 0).GetLocal();
            
            // Receiver
            PacketSinkHelper sink("ns3::TcpSocketFactory",
                                 InetSocketAddress(Ipv4Address::GetAny(), basePort));
            ApplicationContainer sinkApp = sink.Install(dstNode);
            sinkApp.Start(Seconds(1.0));
            sinkApp.Stop(Seconds(simulationTime - 1.0));
            
            // Sender
            BulkSendHelper source("ns3::TcpSocketFactory",
                                 InetSocketAddress(dstAddr, basePort));
            source.SetAttribute("MaxBytes", UintegerValue(0)); // Unlimited
            source.SetAttribute("SendSize", UintegerValue(1448));
            
            ApplicationContainer sourceApp = source.Install(srcNode);
            sourceApp.Start(Seconds(1.0 + g_random->GetValue(0.0, 2.0)));
            sourceApp.Stop(Seconds(simulationTime - 1.0));
            
            basePort++;
            totalFlows++;
        }
    }
    
    NS_LOG_UNCOND("Created " << totalFlows << " unlimited TCP flows");
    NS_LOG_UNCOND("Total links: " << links.size());
    
    // Setup Flow Monitor
    FlowMonitorHelper fmh;
    Ptr<FlowMonitor> monitor = fmh.InstallAll();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(fmh.GetClassifier());
    
    // Clear timeline
    // 0-20s: Normal
    // 20-40s: Link Failure (2 links, ONE direction only)
    // 40-50s: Normal
    // 50-70s: Congestion
    // 70-80s: Normal
    
    // Link failures at 20s
    Simulator::Schedule(Seconds(20.0), []() {
        if (links.size() >= 2) {
            InjectSmartLinkFailure(0);  // Link 0
            InjectSmartLinkFailure(10); // Link 10 (different part of network)
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
    
    // Congestion at 50s
    Simulator::Schedule(Seconds(50.0), &CreateBackgroundTraffic);
    
    // Statistics collection
    for (double t = 2.0; t <= simulationTime; t += 1.0) {
        Simulator::Schedule(Seconds(t), &ExportFlowStatistics,
                          monitor, classifier, t-1.0, t);
        Simulator::Schedule(Seconds(t), &ExportLinkStatistics, t);
    }
    
    // Progress
    for (double t = 20.0; t <= simulationTime; t += 20.0) {
        Simulator::Schedule(Seconds(t), []() {
            NS_LOG_UNCOND("[PROGRESS] Time: " << Simulator::Now().GetSeconds() << "s");
        });
    }
    
    NS_LOG_UNCOND("\nStarting PERFECT simulation...");
    NS_LOG_UNCOND("Timeline:");
    NS_LOG_UNCOND("  0-20s:  NORMAL");
    NS_LOG_UNCOND("  20-40s: LINK FAILURE (links 0 & 10, ONE direction)");
    NS_LOG_UNCOND("  40-50s: NORMAL");
    NS_LOG_UNCOND("  50-70s: CONGESTION (background traffic)");
    NS_LOG_UNCOND("  70-80s: NORMAL");
    
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    flowCsv.close();
    linkCsv.close();
    
    NS_LOG_UNCOND("\n========================================");
    NS_LOG_UNCOND("   PERFECT Dataset Complete!           ");
    NS_LOG_UNCOND("========================================");
    NS_LOG_UNCOND("Expected distribution (~25% each class):");
    NS_LOG_UNCOND("  Normal: 0-20s, 40-50s, 70-80s (37.5%)");
    NS_LOG_UNCOND("  Link Failure: 20-40s (25%)");
    NS_LOG_UNCOND("  Congestion: 50-70s (25%)");
    NS_LOG_UNCOND("========================================");
    
    return 0;
}