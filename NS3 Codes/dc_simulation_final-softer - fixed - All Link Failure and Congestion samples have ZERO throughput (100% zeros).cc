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

NS_LOG_COMPONENT_DEFINE("DCFaultDatasetUltimate");

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
    uint32_t label = 0;
};

/* ===================== GLOBALS ===================== */
std::vector<LinkInfo> links;
std::vector<NodeContainer> serverGroups;
std::map<FlowId, FlowInfo> flowInfo;

std::ofstream flowCsv;
std::ofstream linkCsv;

Ptr<UniformRandomVariable> g_random = CreateObject<UniformRandomVariable>();

/* ===================== ULTIMATE FAULT INJECTION ===================== */
void InjectControlledLinkFailure(uint32_t linkId) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " CONTROLLED failure at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    link.isFaulty = true;
    
    // Apply MODERATE error model (30% loss) - allows SOME throughput
    Ptr<RateErrorModel> errorModel = CreateObject<RateErrorModel>();
    errorModel->SetAttribute("ErrorRate", DoubleValue(0.3)); // 30% loss
    errorModel->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
    
    link.devA->SetReceiveErrorModel(errorModel);
}

void RestoreLinkFailure(uint32_t linkId) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " restored at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    link.isFaulty = false;
    link.devA->SetReceiveErrorModel(nullptr);
}

void CreateModerateBackgroundTraffic() {
    NS_LOG_UNCOND("[CONGESTION] Creating MODERATE background traffic at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    // Create moderate UDP traffic (not too aggressive)
    for (int i = 0; i < 15; i++) {
        uint32_t srcTor = g_random->GetInteger(0, N_TOR - 1);
        uint32_t srcSrv = g_random->GetInteger(0, N_SRV - 1);
        
        uint32_t dstTor = g_random->GetInteger(0, N_TOR - 1);
        uint32_t dstSrv = g_random->GetInteger(0, N_SRV - 1);
        
        Ptr<Node> srcNode = serverGroups[srcTor].Get(srcSrv);
        Ptr<Node> dstNode = serverGroups[dstTor].Get(dstSrv);
        
        Ptr<Ipv4> dstIpv4 = dstNode->GetObject<Ipv4>();
        Ipv4Address dstAddr = dstIpv4->GetAddress(1, 0).GetLocal();
        
        uint16_t port = 40000 + i;
        
        // Moderate UDP traffic (20Mbps)
        OnOffHelper onoff("ns3::UdpSocketFactory", 
                         InetSocketAddress(dstAddr, port));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        onoff.SetAttribute("DataRate", StringValue("20Mbps"));
        onoff.SetAttribute("PacketSize", UintegerValue(1024));
        
        ApplicationContainer app = onoff.Install(srcNode);
        app.Start(Seconds(0.0));
        app.Stop(Seconds(30.0));
    }
}

/* ===================== ULTIMATE STATISTICS ===================== */
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
        
        uint64_t txPkts = curr.txPackets - info.prevTxPackets;
        uint64_t rxPkts = curr.rxPackets - info.prevRxPackets;
        uint64_t lost = curr.lostPackets - info.prevLostPackets;
        uint64_t rxBytes = curr.rxBytes - info.prevRxBytes;
        
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
            throughput = (rxBytes * 8.0) / (windowDuration * 1e6);
        }
        
        auto tuple = classifier->FindFlow(id);
        
        // Time-based labeling BUT with throughput adjustment
        uint32_t label = 0; // normal
        
        if (windowStart >= 20.0 && windowStart < 40.0) {
            label = 1; // Link failure period
            // During link failure, throughput should be REDUCED but not ZERO
        } else if (windowStart >= 50.0 && windowStart < 70.0) {
            label = 2; // Congestion period
            // During congestion, throughput should be MODERATELY reduced
        }
        
        // CRITICAL: If throughput is zero during normal period, mark as abnormal
        if (label == 0 && throughput < 1.0 && windowStart > 10.0) {
            // Could be early finish or other issue - exclude or mark as failure
            label = 1; // Mark as link failure
        }
        
        info.label = label;
        
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
        
        info.prevTxBytes = curr.txBytes;
        info.prevRxBytes = curr.rxBytes;
        info.prevTxPackets = curr.txPackets;
        info.prevRxPackets = curr.rxPackets;
        info.prevLostPackets = curr.lostPackets;
        info.prevDelaySum = curr.delaySum;
        info.prevJitterSum = curr.jitterSum;
    }
}

/* ===================== MAIN ===================== */
int main(int argc, char *argv[]) {
    uint32_t randomSeed = 1;
    double simulationTime = 70.0; // Slightly shorter
    std::string outputPrefix = "dc_ultimate";
    
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
    NS_LOG_UNCOND("   ULTIMATE DC Fault Dataset Generator  ");
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
    
    // Install internet with OPTIMIZED TCP
    InternetStackHelper internet;
    Config::SetDefault("ns3::Ipv4GlobalRouting::RandomEcmpRouting", BooleanValue(true));
    
    // CRITICAL: Optimize TCP for continuous flows
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1448));
    Config::SetDefault("ns3::TcpSocketBase::MinRto", TimeValue(Seconds(1.0)));
    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpNewReno"));
    
    internet.InstallAll();
    
    // Create links
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps")); // REDUCED from 1Gbps
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    
    TrafficControlHelper tch;
    tch.SetRootQueueDisc("ns3::FifoQueueDisc", "MaxSize", StringValue("100p"));
    
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
    
    // Create CONTINUOUS TCP traffic (ON/OFF pattern)
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
            
            // Sender - ON/OFF pattern for CONTINUOUS traffic
            OnOffHelper source("ns3::TcpSocketFactory",
                             InetSocketAddress(dstAddr, basePort));
            source.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=100]"));
            source.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
            source.SetAttribute("DataRate", StringValue("10Mbps")); // Moderate rate
            source.SetAttribute("PacketSize", UintegerValue(1448));
            
            ApplicationContainer sourceApp = source.Install(srcNode);
            sourceApp.Start(Seconds(1.0 + g_random->GetValue(0.0, 2.0)));
            sourceApp.Stop(Seconds(simulationTime - 1.0));
            
            basePort++;
            totalFlows++;
        }
    }
    
    NS_LOG_UNCOND("Created " << totalFlows << " CONTINUOUS TCP flows (10Mbps each)");
    NS_LOG_UNCOND("Total links: " << links.size());
    
    // Setup Flow Monitor
    FlowMonitorHelper fmh;
    Ptr<FlowMonitor> monitor = fmh.InstallAll();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(fmh.GetClassifier());
    
    // Fault schedule
    Simulator::Schedule(Seconds(20.0), []() {
        if (links.size() >= 2) {
            InjectControlledLinkFailure(0);  // 30% loss
            InjectControlledLinkFailure(10); // 30% loss
        }
    });
    
    Simulator::Schedule(Seconds(40.0), []() {
        for (uint32_t linkId = 0; linkId < links.size(); linkId++) {
            if (links[linkId].isFaulty) {
                RestoreLinkFailure(linkId);
            }
        }
    });
    
    // Congestion traffic (moderate)
    Simulator::Schedule(Seconds(50.0), &CreateModerateBackgroundTraffic);
    
    // Statistics collection
    for (double t = 2.0; t <= simulationTime; t += 1.0) {
        Simulator::Schedule(Seconds(t), &ExportFlowStatistics,
                          monitor, classifier, t-1.0, t);
    }
    
    // Progress indicator
    for (double t = 20.0; t <= simulationTime; t += 20.0) {
        Simulator::Schedule(Seconds(t), []() {
            NS_LOG_UNCOND("[PROGRESS] Time: " << Simulator::Now().GetSeconds() << "s");
        });
    }
    
    NS_LOG_UNCOND("\nStarting ULTIMATE simulation...");
    NS_LOG_UNCOND("Key improvements:");
    NS_LOG_UNCOND("  1. 100Mbps links (not 1Gbps) - more realistic congestion");
    NS_LOG_UNCOND("  2. Continuous ON/OFF TCP flows (not bulk transfers)");
    NS_LOG_UNCOND("  3. Moderate 30% link failure (not severe)");
    NS_LOG_UNCOND("  4. Optimized TCP parameters");
    
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    flowCsv.close();
    linkCsv.close();
    
    NS_LOG_UNCOND("\n========================================");
    NS_LOG_UNCOND("   ULTIMATE Dataset Complete!          ");
    NS_LOG_UNCOND("========================================");
    
    return 0;
}
