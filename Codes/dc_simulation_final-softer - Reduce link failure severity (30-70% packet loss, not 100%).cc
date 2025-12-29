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

/* ===================== OPTIMIZED FAULT INJECTION ===================== */
void InjectModerateLinkFailure(uint32_t linkId, double severity) {
    // severity: 0.3 = 30% packet loss (mild), 0.7 = 70% packet loss (severe)
    // NEVER use 1.0 (100% loss) - that creates useless all-zero samples
    
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " moderate failure (" 
                  << severity*100 << "% loss) at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    link.isFaulty = true;
    link.faultSeverity = severity;
    faultyLinks.insert(linkId);
    
    // Inject MODERATE packet loss
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

void InjectHeavyCongestion() {
    NS_LOG_UNCOND("[CONGESTION] Creating HEAVY congestion at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    // Create MANY UDP flood flows to ensure congestion
    for (int i = 0; i < 10; i++) { // Increased from 5 to 10
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
        
        uint16_t port = 20000 + i;
        
        // UDP flood with VERY high rate
        OnOffHelper onoff("ns3::UdpSocketFactory", 
                         InetSocketAddress(dstAddr, port));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        onoff.SetAttribute("DataRate", StringValue("800Mbps")); // VERY high rate
        onoff.SetAttribute("PacketSize", UintegerValue(1472));
        
        ApplicationContainer app = onoff.Install(srcNode);
        app.Start(Seconds(0.0));
        app.Stop(Seconds(40.0)); // Longer duration
        
        // Mark MULTIPLE links as congested
        for (int j = 0; j < 3; j++) {
            uint32_t congestedLink = g_random->GetInteger(0, links.size() - 1);
            links[congestedLink].isCongested = true;
            congestedLinks.insert(congestedLink);
        }
    }
}

/* ===================== OPTIMIZED STATISTICS COLLECTION ===================== */
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
        
        // Set baseline (first 10 seconds are normal)
        if (windowStart < 10.0 && !info.baselineSet && throughput > 0) {
            info.baselineThroughput = throughput;
            info.baselineSet = true;
        }
        
        // OPTIMIZED LABELING - More balanced thresholds
        uint32_t label = 0; // normal by default
        
        if (info.baselineSet && info.baselineThroughput > 1.0) {
            double throughputRatio = throughput / info.baselineThroughput;
            
            // More generous thresholds to create more balanced classes
            if (throughputRatio < 0.1) { // Severe reduction
                label = 1; // link failure
            } else if (throughputRatio < 0.6) { // Moderate reduction
                label = 2; // congestion
            }
            // throughputRatio >= 0.6 remains normal
        }
        // If no baseline or very low baseline, use absolute thresholds
        else if (throughput < 1.0) {
            label = 1; // link failure (very low throughput)
        } else if (throughput < 50.0) { // Moderate throughput
            label = 2; // congestion
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
            if (link.faultSeverity > 0.5) {
                status = 1; // severe fault
            } else {
                status = 4; // mild fault
            }
        } else if (link.isCongested) {
            status = 2;
        } else if (dDrops > 0) {
            status = 3; // dropping
        } else if (queueDepth > 30) {
            status = 5; // high queue
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
    double simulationTime = 90.0; // Even longer for more balanced data
    std::string outputPrefix = "dc_final";
    
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
    NS_LOG_UNCOND("   FINAL DC Fault Dataset Generator     ");
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
    
    // Traffic control with LARGER queue for congestion
    TrafficControlHelper tch;
    tch.SetRootQueueDisc("ns3::FifoQueueDisc", "MaxSize", StringValue("200p"));
    
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
    
    // Create TCP traffic - MORE FLOWS for better statistics
    uint16_t basePort = 5000;
    uint32_t totalFlows = 0;
    
    for (uint32_t srcTor = 0; srcTor < N_TOR; srcTor++) {
        for (uint32_t srcSrv = 0; srcSrv < N_SRV; srcSrv++) {
            // Create TWO flows per server pair for more data
            for (int flowNum = 0; flowNum < 2; flowNum++) {
                uint32_t dstTor = (srcTor + 1 + g_random->GetInteger(0, N_TOR - 2)) % N_TOR;
                uint32_t dstSrv = g_random->GetInteger(0, N_SRV - 1);
                
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
                
                // Install BulkSend with RANDOM sizes
                BulkSendHelper source("ns3::TcpSocketFactory",
                                     InetSocketAddress(dstAddr, basePort));
                
                // Random flow sizes for diversity
                double randVal = g_random->GetValue();
                if (randVal < 0.4) {
                    source.SetAttribute("MaxBytes", UintegerValue(2000000)); // 2MB
                } else if (randVal < 0.8) {
                    source.SetAttribute("MaxBytes", UintegerValue(10000000)); // 10MB
                } else {
                    source.SetAttribute("MaxBytes", UintegerValue(50000000)); // 50MB
                }
                
                source.SetAttribute("SendSize", UintegerValue(1460));
                
                ApplicationContainer sourceApp = source.Install(srcNode);
                double startTime = 1.0 + g_random->GetValue(0.0, 5.0);
                sourceApp.Start(Seconds(startTime));
                sourceApp.Stop(Seconds(simulationTime - 1.0));
                
                basePort++;
                totalFlows++;
            }
        }
    }
    
    NS_LOG_UNCOND("Created " << totalFlows << " TCP flows");
    NS_LOG_UNCOND("Total links: " << links.size());
    
    // Setup Flow Monitor
    FlowMonitorHelper fmh;
    Ptr<FlowMonitor> monitor = fmh.InstallAll();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(fmh.GetClassifier());
    
    // Schedule OPTIMIZED fault scenarios for balanced classes
    
    // 1. MILD link failure (30% loss) - creates congestion-like symptoms
    Simulator::Schedule(Seconds(15.0), []() {
        if (!links.empty()) {
            uint32_t linkId = g_random->GetInteger(0, links.size() - 1);
            InjectModerateLinkFailure(linkId, 0.3); // 30% loss
        }
    });
    
    Simulator::Schedule(Seconds(30.0), []() {
        if (!links.empty()) {
            uint32_t linkId = g_random->GetInteger(0, links.size() - 1);
            RestoreLinkFailure(linkId);
        }
    });
    
    // 2. HEAVY congestion (multiple times)
    Simulator::Schedule(Seconds(20.0), &InjectHeavyCongestion);
    Simulator::Schedule(Seconds(40.0), &InjectHeavyCongestion);
    
    // 3. MODERATE link failure (50% loss)
    Simulator::Schedule(Seconds(50.0), []() {
        if (!links.empty()) {
            uint32_t linkId = g_random->GetInteger(0, links.size() - 1);
            InjectModerateLinkFailure(linkId, 0.5); // 50% loss
        }
    });
    
    Simulator::Schedule(Seconds(70.0), []() {
        if (!links.empty()) {
            uint32_t linkId = g_random->GetInteger(0, links.size() - 1);
            RestoreLinkFailure(linkId);
        }
    });
    
    // Schedule periodic statistics collection
    for (double t = 2.0; t <= simulationTime; t += 1.0) {
        Simulator::Schedule(Seconds(t), &ExportFlowStatistics,
                          monitor, classifier, t-1.0, t);
        Simulator::Schedule(Seconds(t), &ExportLinkStatistics, t);
    }
    
    // Add progress indicator
    for (double t = 15.0; t <= simulationTime; t += 15.0) {
        Simulator::Schedule(Seconds(t), []() {
            NS_LOG_UNCOND("[PROGRESS] Simulation time: " << Simulator::Now().GetSeconds() << "s");
        });
    }
    
    // Run simulation
    NS_LOG_UNCOND("\nStarting FINAL simulation...");
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
    NS_LOG_UNCOND("\nDesigned for BALANCED ML training:");
    NS_LOG_UNCOND("  - Moderate link failures (30-50% loss)");
    NS_LOG_UNCOND("  - Heavy congestion traffic");
    NS_LOG_UNCOND("  - Optimized labeling thresholds");
    NS_LOG_UNCOND("  - Larger dataset with more flows");
    NS_LOG_UNCOND("========================================");
    
    return 0;
}

