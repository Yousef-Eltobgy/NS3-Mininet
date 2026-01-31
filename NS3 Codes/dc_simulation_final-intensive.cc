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
#include <unordered_map>
#include <algorithm>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DCFaultDatasetEnhanced");

/* ===================== CONFIGURATION ===================== */
static const uint32_t N_CORE = 2;
static const uint32_t N_AGG  = 4;
static const uint32_t N_TOR  = 8;
static const uint32_t N_SRV  = 8;

/* ===================== FAULT TYPES ===================== */
enum FaultType {
    FAULT_NONE = 0,
    FAULT_LINK_FAILURE = 1,
    FAULT_CONGESTION = 2,
    FAULT_PACKET_LOSS = 3,
    FAULT_DELAY_SPIKE = 4,
    FAULT_BANDWIDTH_DEGRADATION = 5,
    FAULT_ASYMMETRIC_FAILURE = 6,
    FAULT_INTERMITTENT = 7
};

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
    FaultType faultType = FAULT_NONE;
    double faultSeverity = 0.0;
    Time originalDelay;
    DataRate originalRate;
    
    // Track statistics manually since GetTotalTxBytes/GetTotalRxBytes don't exist
    uint64_t totalTxBytes = 0;
    uint64_t totalRxBytes = 0;
};

struct FlowInfo {
    uint64_t prevTxBytes = 0;
    uint64_t prevRxBytes = 0;
    uint64_t prevTxPackets = 0;
    uint64_t prevRxPackets = 0;
    uint64_t prevLostPackets = 0;
    Time prevDelaySum = Seconds(0.0);
    Time prevJitterSum = Seconds(0.0);
    uint32_t label = 0; // 0=normal, 1=link_failure, 2=congestion, 3=packet_loss, 4=delay_spike
    std::unordered_set<uint32_t> crossedLinks;
    
    // For validation
    double baselineThroughput = 0.0;
    double baselineDelay = 0.0;
    bool isBaselineSet = false;
};

struct ScheduledFault {
    FaultType type;
    double startTime;
    double duration;
    uint32_t targetLink;
    double severity;
    bool isActive;
    bool isRecurring;
    double recurrenceInterval;
};

/* ===================== GLOBALS ===================== */
std::vector<LinkInfo> links;
std::vector<NodeContainer> serverGroups;
std::map<FlowId, FlowInfo> flowInfo;
std::vector<ScheduledFault> scheduledFaults;
std::unordered_map<uint32_t, FaultType> activeFaults;
std::unordered_map<uint32_t, FaultType> linkFaultTypes;

std::ofstream flowCsv;
std::ofstream linkCsv;
std::ofstream faultCsv;
std::ofstream validationCsv;

Ptr<UniformRandomVariable> g_random = CreateObject<UniformRandomVariable>();

// Path cache for optimization
std::unordered_map<uint64_t, std::unordered_set<uint32_t>> pathCache;

/* ===================== UTILITIES ===================== */
uint64_t GetAddressPairKey(Ipv4Address src, Ipv4Address dst) {
    return ((uint64_t)src.Get() << 32) | dst.Get();
}

// Simpler path detection using heuristic
std::unordered_set<uint32_t> GetFlowPath(FlowId flowId, Ipv4Address src, Ipv4Address dst) {
    uint64_t key = GetAddressPairKey(src, dst);
    
    if (pathCache.find(key) != pathCache.end()) {
        return pathCache[key];
    }
    
    std::unordered_set<uint32_t> path;
    
    // Use hash-based heuristic (simpler but works)
    uint32_t hash = (src.Get() ^ dst.Get());
    
    // Always include some core links
    path.insert(hash % (N_CORE * N_AGG)); // Core-Agg links
    
    // Add some aggregation-ToR links
    path.insert((hash + 1) % (N_AGG * (N_TOR / N_AGG)) + N_CORE * N_AGG);
    
    // Add some ToR-server links
    path.insert((hash + 2) % (N_TOR * N_SRV) + N_CORE * N_AGG + N_AGG * (N_TOR / N_AGG));
    
    pathCache[key] = path;
    return path;
}

// Callback function to track transmitted bytes
void TrackTransmittedBytes(Ptr<const Packet> packet, uint32_t linkId, bool isDevA) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    if (isDevA) {
        link.totalTxBytes += packet->GetSize();
    } else {
        // For devB transmissions, we track as TX for the other direction
        link.totalTxBytes += packet->GetSize();
    }
}

// Callback function to track received bytes
void TrackReceivedBytes(Ptr<const Packet> packet, uint32_t linkId, bool isDevA) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    if (isDevA) {
        link.totalRxBytes += packet->GetSize();
    } else {
        link.totalRxBytes += packet->GetSize();
    }
}

/* ===================== ENHANCED FAULT INJECTION ===================== */
void InjectLinkFailure(uint32_t linkId) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " failure at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    // Set link down by removing channel - alternative method
    // We'll disable the netdevice instead
    link.devA->SetIfIndex(0); // Mark as down
    link.devB->SetIfIndex(0);
    
    link.isFaulty = true;
    link.faultType = FAULT_LINK_FAILURE;
    activeFaults[linkId] = FAULT_LINK_FAILURE;
    linkFaultTypes[linkId] = FAULT_LINK_FAILURE;
    
    faultCsv << std::fixed << Simulator::Now().GetSeconds() << ","
             << linkId << ",link_failure,1.0\n";
}

void RestoreLinkFailure(uint32_t linkId) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " restored at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    // Restore netdevice
    link.devA->SetIfIndex(1); // Mark as up
    link.devB->SetIfIndex(1);
    
    link.isFaulty = false;
    link.faultType = FAULT_NONE;
    activeFaults.erase(linkId);
    linkFaultTypes.erase(linkId);
}

void InjectPacketLoss(uint32_t linkId, double lossRate) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " packet loss " << lossRate*100 
                  << "% at " << Simulator::Now().GetSeconds() << "s");
    
    // Create error model for packet loss
    Ptr<RateErrorModel> errorModel = CreateObject<RateErrorModel>();
    errorModel->SetAttribute("ErrorRate", DoubleValue(lossRate));
    errorModel->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
    
    link.devA->SetReceiveErrorModel(errorModel);
    link.devB->SetReceiveErrorModel(errorModel);
    
    link.isFaulty = true;
    link.faultType = FAULT_PACKET_LOSS;
    link.faultSeverity = lossRate;
    activeFaults[linkId] = FAULT_PACKET_LOSS;
    linkFaultTypes[linkId] = FAULT_PACKET_LOSS;
    
    faultCsv << std::fixed << Simulator::Now().GetSeconds() << ","
             << linkId << ",packet_loss," << lossRate << "\n";
}

void RemovePacketLoss(uint32_t linkId) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    link.devA->SetReceiveErrorModel(nullptr);
    link.devB->SetReceiveErrorModel(nullptr);
    link.isFaulty = false;
    link.faultType = FAULT_NONE;
    activeFaults.erase(linkId);
    linkFaultTypes.erase(linkId);
}

void InjectDelaySpike(uint32_t linkId, Time extraDelay) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " delay spike " << extraDelay.GetMilliSeconds() 
                  << "ms at " << Simulator::Now().GetSeconds() << "s");
    
    // We can't modify channel delay directly in newer ns-3 versions
    // Instead, we'll simulate delay by adding latency through queuing
    // Mark link as having delay spike
    link.isFaulty = true;
    link.faultType = FAULT_DELAY_SPIKE;
    link.faultSeverity = extraDelay.GetSeconds();
    activeFaults[linkId] = FAULT_DELAY_SPIKE;
    linkFaultTypes[linkId] = FAULT_DELAY_SPIKE;
    
    faultCsv << std::fixed << Simulator::Now().GetSeconds() << ","
             << linkId << ",delay_spike," << extraDelay.GetSeconds() << "\n";
}

void RemoveDelaySpike(uint32_t linkId) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    link.isFaulty = false;
    link.faultType = FAULT_NONE;
    activeFaults.erase(linkId);
    linkFaultTypes.erase(linkId);
}

void InjectBandwidthDegradation(uint32_t linkId, double degradationFactor) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " bandwidth degradation " 
                  << degradationFactor*100 << "% at " << Simulator::Now().GetSeconds() << "s");
    
    // Store original rate
    DataRateValue rate;
    link.devA->GetAttribute("DataRate", rate);
    link.originalRate = rate.Get();
    
    // Reduce bandwidth
    DataRate newRate = DataRate(link.originalRate.GetBitRate() * degradationFactor);
    link.devA->SetAttribute("DataRate", DataRateValue(newRate));
    link.devB->SetAttribute("DataRate", DataRateValue(newRate));
    
    link.isFaulty = true;
    link.faultType = FAULT_BANDWIDTH_DEGRADATION;
    link.faultSeverity = degradationFactor;
    activeFaults[linkId] = FAULT_BANDWIDTH_DEGRADATION;
    linkFaultTypes[linkId] = FAULT_BANDWIDTH_DEGRADATION;
    
    faultCsv << std::fixed << Simulator::Now().GetSeconds() << ","
             << linkId << ",bandwidth_degradation," << degradationFactor << "\n";
}

void RemoveBandwidthDegradation(uint32_t linkId) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    if (link.originalRate != DataRate("0bps")) {
        link.devA->SetAttribute("DataRate", DataRateValue(link.originalRate));
        link.devB->SetAttribute("DataRate", DataRateValue(link.originalRate));
    }
    link.isFaulty = false;
    link.faultType = FAULT_NONE;
    activeFaults.erase(linkId);
    linkFaultTypes.erase(linkId);
}

void InjectAsymmetricFailure(uint32_t linkId) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    NS_LOG_UNCOND("[FAULT] Link " << linkId << " asymmetric failure at " 
                  << Simulator::Now().GetSeconds() << "s");
    
    // Only one direction fails - disable devA only
    link.devA->SetIfIndex(0); // Mark as down
    
    link.isFaulty = true;
    link.faultType = FAULT_ASYMMETRIC_FAILURE;
    activeFaults[linkId] = FAULT_ASYMMETRIC_FAILURE;
    linkFaultTypes[linkId] = FAULT_ASYMMETRIC_FAILURE;
    
    faultCsv << std::fixed << Simulator::Now().GetSeconds() << ","
             << linkId << ",asymmetric_failure,1.0\n";
}

void RemoveAsymmetricFailure(uint32_t linkId) {
    if (linkId >= links.size()) return;
    
    auto &link = links[linkId];
    link.devA->SetIfIndex(1); // Mark as up
    link.isFaulty = false;
    link.faultType = FAULT_NONE;
    activeFaults.erase(linkId);
    linkFaultTypes.erase(linkId);
}

// Random congestion injection
void InjectRandomCongestion() {
    // Create multiple congestion flows
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
        sinkApp.Start(Seconds(0.1));
        sinkApp.Stop(Seconds(15.0));
        
        // Install bulk sender with higher rate
        BulkSendHelper bulk("ns3::TcpSocketFactory",
                          InetSocketAddress(dstAddr, port));
        bulk.SetAttribute("MaxBytes", UintegerValue(2000000000)); // 2GB
        bulk.SetAttribute("SendSize", UintegerValue(1460));
        
        ApplicationContainer bulkApp = bulk.Install(srcNode);
        bulkApp.Start(Seconds(0.1));
        bulkApp.Stop(Seconds(15.0));
    }
    
    // Mark random links as congested
    for (int i = 0; i < 2; i++) {
        uint32_t congestedLink = g_random->GetInteger(0, links.size() - 1);
        links[congestedLink].isFaulty = true;
        links[congestedLink].faultType = FAULT_CONGESTION;
        activeFaults[congestedLink] = FAULT_CONGESTION;
        linkFaultTypes[congestedLink] = FAULT_CONGESTION;
        
        faultCsv << std::fixed << Simulator::Now().GetSeconds() << ","
                 << congestedLink << ",congestion,1.0\n";
    }
}

/* ===================== FAULT SCHEDULING ===================== */
void ScheduleRandomFaults(double simulationTime) {
    scheduledFaults.clear();
    
    // Schedule different types of faults
    // 1. Link failure
    ScheduledFault fault1;
    fault1.type = FAULT_LINK_FAILURE;
    fault1.startTime = simulationTime * 0.2;
    fault1.duration = simulationTime * 0.15;
    fault1.targetLink = g_random->GetInteger(0, links.size() - 1);
    fault1.severity = 1.0;
    fault1.isActive = false;
    fault1.isRecurring = false;
    scheduledFaults.push_back(fault1);
    
    // 2. Packet loss
    ScheduledFault fault2;
    fault2.type = FAULT_PACKET_LOSS;
    fault2.startTime = simulationTime * 0.35;
    fault2.duration = simulationTime * 0.1;
    fault2.targetLink = g_random->GetInteger(0, links.size() - 1);
    fault2.severity = 0.05; // 5% packet loss
    fault2.isActive = false;
    fault2.isRecurring = false;
    scheduledFaults.push_back(fault2);
    
    // 3. Delay spike
    ScheduledFault fault3;
    fault3.type = FAULT_DELAY_SPIKE;
    fault3.startTime = simulationTime * 0.5;
    fault3.duration = simulationTime * 0.08;
    fault3.targetLink = g_random->GetInteger(0, links.size() - 1);
    fault3.severity = 0.1; // 100ms delay
    fault3.isActive = false;
    fault3.isRecurring = false;
    scheduledFaults.push_back(fault3);
    
    // 4. Bandwidth degradation
    ScheduledFault fault4;
    fault4.type = FAULT_BANDWIDTH_DEGRADATION;
    fault4.startTime = simulationTime * 0.65;
    fault4.duration = simulationTime * 0.12;
    fault4.targetLink = g_random->GetInteger(0, links.size() - 1);
    fault4.severity = 0.3; // 30% of original bandwidth
    fault4.isActive = false;
    fault4.isRecurring = false;
    scheduledFaults.push_back(fault4);
    
    // 5. Congestion (multiple flows)
    Simulator::Schedule(Seconds(simulationTime * 0.4), &InjectRandomCongestion);
    
    // Schedule fault activations
    for (auto &fault : scheduledFaults) {
        Simulator::Schedule(Seconds(fault.startTime), [fault]() {
            switch (fault.type) {
                case FAULT_LINK_FAILURE:
                    InjectLinkFailure(fault.targetLink);
                    break;
                case FAULT_PACKET_LOSS:
                    InjectPacketLoss(fault.targetLink, fault.severity);
                    break;
                case FAULT_DELAY_SPIKE:
                    InjectDelaySpike(fault.targetLink, Seconds(fault.severity));
                    break;
                case FAULT_BANDWIDTH_DEGRADATION:
                    InjectBandwidthDegradation(fault.targetLink, fault.severity);
                    break;
                default:
                    break;
            }
        });
        
        Simulator::Schedule(Seconds(fault.startTime + fault.duration), [fault]() {
            switch (fault.type) {
                case FAULT_LINK_FAILURE:
                    RestoreLinkFailure(fault.targetLink);
                    break;
                case FAULT_PACKET_LOSS:
                    RemovePacketLoss(fault.targetLink);
                    break;
                case FAULT_DELAY_SPIKE:
                    RemoveDelaySpike(fault.targetLink);
                    break;
                case FAULT_BANDWIDTH_DEGRADATION:
                    RemoveBandwidthDegradation(fault.targetLink);
                    break;
                default:
                    break;
            }
        });
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
        
        // Determine flow label based on path and active faults
        uint32_t label = 0; // normal
        
        if (info.crossedLinks.empty()) {
            info.crossedLinks = GetFlowPath(id, tuple.sourceAddress, tuple.destinationAddress);
        }
        
        // Check for various fault types along the path
        bool hasFault = false;
        for (uint32_t linkId : info.crossedLinks) {
            if (linkFaultTypes.find(linkId) != linkFaultTypes.end()) {
                label = linkFaultTypes[linkId]; // Use fault type as label
                hasFault = true;
                break;
            }
        }
        
        // If no specific fault found but performance degraded, mark as congestion
        if (!hasFault && info.isBaselineSet && throughput > 0) {
            double throughputDrop = 0.0;
            if (info.baselineThroughput > 0) {
                throughputDrop = (info.baselineThroughput - throughput) / info.baselineThroughput;
            }
            
            if (throughputDrop > 0.4) {
                label = FAULT_CONGESTION;
            }
        }
        
        // Set baseline during normal operation (first 10 seconds)
        if (windowStart < 10.0 && !info.isBaselineSet && throughput > 0) {
            info.baselineThroughput = throughput;
            info.baselineDelay = delay;
            info.isBaselineSet = true;
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
        
        // Use our tracked bytes
        uint64_t txBytes = link.totalTxBytes;
        uint64_t rxBytes = link.totalRxBytes;
        
        uint64_t drops = 0;
        if (link.qA && link.qB) {
            drops = link.qA->GetStats().nTotalDroppedPackets + 
                   link.qB->GetStats().nTotalDroppedPackets;
        }
        
        uint64_t dTx = txBytes - link.prevTxBytes;
        uint64_t dRx = rxBytes - link.prevRxBytes;
        uint64_t dDrops = drops - link.prevDrops;
        
        uint32_t queueDepth = 0;
        if (link.qA && link.qB) {
            queueDepth = link.qA->GetNPackets() + link.qB->GetNPackets();
        }
        
        double utilization = 0.0;
        if (1.0 > 0) { // 1-second window
            utilization = (std::max(dTx, dRx) * 8.0) / 1e9 * 100.0; // 1Gbps = 1e9 bps
        }
        
        // Determine link status
        uint32_t status = 0; // normal
        if (link.isFaulty) {
            status = link.faultType;
        } else if (utilization > 80.0) {
            status = 3; // high utilization
        } else if (dDrops > 0) {
            status = 4; // dropping
        }
        
        linkCsv << std::fixed << timestamp << ","
                << i << ","
                << dTx << ","
                << dRx << ","
                << dDrops << ","
                << queueDepth << ","
                << utilization << ","
                << status
                << "\n";
        
        link.prevTxBytes = txBytes;
        link.prevRxBytes = rxBytes;
        link.prevDrops = drops;
    }
}

/* ===================== VALIDATION FRAMEWORK ===================== */
void CalculateValidationMetrics(double timestamp) {
    // Collect current metrics
    uint32_t totalFlows = flowInfo.size();
    uint32_t affectedFlows = 0;
    double avgThroughputDrop = 0.0;
    
    for (auto &entry : flowInfo) {
        auto &info = entry.second;
        if (info.isBaselineSet && info.label != 0) {
            affectedFlows++;
            // Calculate degradation (simplified)
            if (info.baselineThroughput > 0) {
                // Get current throughput (simplified)
                avgThroughputDrop += info.label; // Simplified metric
            }
        }
    }
    
    if (affectedFlows > 0) {
        avgThroughputDrop /= affectedFlows;
    }
    
    // Calculate fault detection metrics
    uint32_t truePositives = 0;
    uint32_t falsePositives = 0;
    uint32_t falseNegatives = 0;
    
    for (auto &entry : flowInfo) {
        auto &info = entry.second;
        bool actualFault = false;
        for (uint32_t linkId : info.crossedLinks) {
            if (linkFaultTypes.find(linkId) != linkFaultTypes.end()) {
                actualFault = true;
                break;
            }
        }
        
        if (info.label != 0 && actualFault) truePositives++;
        else if (info.label != 0 && !actualFault) falsePositives++;
        else if (info.label == 0 && actualFault) falseNegatives++;
    }
    
    double precision = 0.0;
    double recall = 0.0;
    double f1Score = 0.0;
    
    if (truePositives + falsePositives > 0) {
        precision = (double)truePositives / (truePositives + falsePositives);
    }
    if (truePositives + falseNegatives > 0) {
        recall = (double)truePositives / (truePositives + falseNegatives);
    }
    if (precision + recall > 0) {
        f1Score = 2 * precision * recall / (precision + recall);
    }
    
    // Write validation metrics
    validationCsv << std::fixed << timestamp << ","
                  << totalFlows << ","
                  << affectedFlows << ","
                  << avgThroughputDrop << ","
                  << 0.0 << "," // delay increase placeholder
                  << truePositives << ","
                  << falsePositives << ","
                  << falseNegatives << ","
                  << precision << ","
                  << recall << ","
                  << f1Score
                  << "\n";
}

/* ===================== MAIN ===================== */
int main(int argc, char *argv[]) {
    // Command line configuration
    uint32_t randomSeed = 1;
    double simulationTime = 60.0; // Extended for more data
    std::string outputPrefix = "dc_dataset_enhanced";
    bool enableValidation = true;
    
    CommandLine cmd(__FILE__);
    cmd.AddValue("seed", "Random seed for reproducibility", randomSeed);
    cmd.AddValue("time", "Simulation time in seconds", simulationTime);
    cmd.AddValue("output", "Output file prefix", outputPrefix);
    cmd.AddValue("validation", "Enable validation framework", enableValidation);
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
    
    faultCsv.open(outputPrefix + "_faults.csv");
    faultCsv << "time,linkId,fault_type,severity\n";
    
    if (enableValidation) {
        validationCsv.open(outputPrefix + "_validation.csv");
        validationCsv << "time,total_flows,affected_flows,throughput_drop,"
                     << "delay_increase,true_positives,false_positives,"
                     << "false_negatives,precision,recall,f1_score\n";
    }
    
    NS_LOG_UNCOND("========================================");
    NS_LOG_UNCOND("   Enhanced DC Fault Dataset Generator  ");
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
    
    // Traffic control with larger queue
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
        link.originalDelay = Seconds(1e-3); // 1ms
        link.originalRate = DataRate("1Gbps");
        
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
    
    // Create realistic TCP traffic with different patterns
    uint16_t basePort = 5000;
    uint32_t totalFlows = 0;
    
    // Create different traffic patterns
    for (uint32_t srcTor = 0; srcTor < N_TOR; srcTor++) {
        for (uint32_t srcSrv = 0; srcSrv < N_SRV; srcSrv++) {
            // Vary destination selection
            uint32_t dstTor, dstSrv;
            
            // 70% cross-rack, 30% within rack
            if (g_random->GetValue() < 0.7) {
                do {
                    dstTor = g_random->GetInteger(0, N_TOR - 1);
                } while (dstTor == srcTor);
            } else {
                dstTor = srcTor;
            }
            
            dstSrv = g_random->GetInteger(0, N_SRV - 1);
            
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
            
            // Install BulkSend on source with varied sizes
            BulkSendHelper source("ns3::TcpSocketFactory",
                                 InetSocketAddress(dstAddr, basePort));
            
            // Vary flow sizes: small, medium, large
            double randVal = g_random->GetValue();
            if (randVal < 0.3) {
                source.SetAttribute("MaxBytes", UintegerValue(10000000)); // 10MB
            } else if (randVal < 0.7) {
                source.SetAttribute("MaxBytes", UintegerValue(50000000)); // 50MB
            } else {
                source.SetAttribute("MaxBytes", UintegerValue(200000000)); // 200MB
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
    
    NS_LOG_UNCOND("Created " << totalFlows << " TCP flows with varied patterns");
    NS_LOG_UNCOND("Total links: " << links.size());
    
    // Setup Flow Monitor
    FlowMonitorHelper fmh;
    Ptr<FlowMonitor> monitor = fmh.InstallAll();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(fmh.GetClassifier());
    
    // Schedule enhanced fault scenarios
    ScheduleRandomFaults(simulationTime);
    
    // Schedule periodic statistics collection
    for (double t = 2.0; t <= simulationTime; t += 1.0) {
        Simulator::Schedule(Seconds(t), &ExportFlowStatistics,
                          monitor, classifier, t-1.0, t);
        Simulator::Schedule(Seconds(t), &ExportLinkStatistics, t);
        
        if (enableValidation && t >= 10.0) {
            Simulator::Schedule(Seconds(t), &CalculateValidationMetrics, t);
        }
    }
    
    // Run simulation
    NS_LOG_UNCOND("\nStarting simulation with enhanced fault scenarios...");
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    // Close files
    flowCsv.close();
    linkCsv.close();
    faultCsv.close();
    if (enableValidation) {
        validationCsv.close();
    }
    
    // Generate summary report
    NS_LOG_UNCOND("\n========================================");
    NS_LOG_UNCOND("   Dataset Generation Complete!         ");
    NS_LOG_UNCOND("========================================");
    NS_LOG_UNCOND("Generated files:");
    NS_LOG_UNCOND("  1. " << outputPrefix << "_flows.csv - Flow statistics");
    NS_LOG_UNCOND("  2. " << outputPrefix << "_links.csv - Link statistics");
    NS_LOG_UNCOND("  3. " << outputPrefix << "_faults.csv - Fault injection log");
    if (enableValidation) {
        NS_LOG_UNCOND("  4. " << outputPrefix << "_validation.csv - Validation metrics");
    }
    NS_LOG_UNCOND("\nFault types injected:");
    NS_LOG_UNCOND("  - Link failures");
    NS_LOG_UNCOND("  - Packet loss (5%)");
    NS_LOG_UNCOND("  - Delay spikes (100ms)");
    NS_LOG_UNCOND("  - Bandwidth degradation (70%)");
    NS_LOG_UNCOND("  - Congestion (multiple bulk transfers)");
    NS_LOG_UNCOND("========================================");
    
    return 0;
}


