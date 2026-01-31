#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include <fstream>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DCFaultCSV_ECMP");

// Function to simulate a link failure
static void
LinkFailure(Ptr<NetDevice> dev)
{
    dev->SetAttribute("ReceiveErrorModel",
        PointerValue(CreateObject<RateErrorModel>()));
}

// Function to export time-windowed statistics
void
ExportWindowedStats(Ptr<FlowMonitor> monitor,
                    Ptr<Ipv4FlowClassifier> classifier,
                    double time,
                    std::ofstream &out,
                    uint32_t label)
{
    monitor->CheckForLostPackets();
    auto stats = monitor->GetFlowStats();

    for (auto &flow : stats)
    {
        auto t = flow.second;
        double throughput = (t.rxBytes * 8.0) / (time * 1e6); // Mbps

        out << time << "," 
            << flow.first << "," 
            << t.txPackets << "," 
            << t.rxPackets << "," 
            << t.lostPackets << "," 
            << throughput << "," 
            << label << "\n";
    }
}

int main(int argc, char *argv[])
{
    // -----------------------------
    // Topology parameters
    // -----------------------------
    uint32_t nCore = 2;
    uint32_t nAgg = 4;
    uint32_t nTor = 8;
    uint32_t nServersPerTor = 8;
    double simTime = 30.0;

    NodeContainer core, agg, tor, servers;
    core.Create(nCore);
    agg.Create(nAgg);
    tor.Create(nTor);
    servers.Create(nTor * nServersPerTor);

    // Install Internet stack
    InternetStackHelper internet;
    internet.InstallAll();

    // -----------------------------
    // Links configuration
    // -----------------------------
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    p2p.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("50p"));

    Ipv4AddressHelper ip;
    std::vector<NetDeviceContainer> allLinks;

    // Core ↔ Agg (ECMP: all agg connect to both cores)
    ip.SetBase("10.0.0.0", "255.255.255.0");
    for (uint32_t c = 0; c < nCore; c++)
    {
        for (uint32_t a = 0; a < nAgg; a++)
        {
            auto link = p2p.Install(core.Get(c), agg.Get(a));
            ip.Assign(link);
            ip.NewNetwork();
            allLinks.push_back(link);
        }
    }

    // Agg ↔ ToR
    uint32_t torsPerAgg = nTor / nAgg;
    ip.SetBase("10.1.0.0", "255.255.255.0");
    for (uint32_t a = 0; a < nAgg; a++)
    {
        for (uint32_t t = 0; t < torsPerAgg; t++)
        {
            uint32_t torIdx = a * torsPerAgg + t;
            auto link = p2p.Install(agg.Get(a), tor.Get(torIdx));
            ip.Assign(link);
            ip.NewNetwork();
            allLinks.push_back(link);
        }
    }

    // ToR ↔ Servers
    ip.SetBase("10.2.0.0", "255.255.255.0");
    for (uint32_t t = 0; t < nTor; t++)
    {
        for (uint32_t s = 0; s < nServersPerTor; s++)
        {
            uint32_t idx = t * nServersPerTor + s;
            auto link = p2p.Install(tor.Get(t), servers.Get(idx));
            ip.Assign(link);
            ip.NewNetwork();
            allLinks.push_back(link);
        }
    }

    // -----------------------------
    // Routing (ECMP paths possible)
    // -----------------------------
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // -----------------------------
    // TCP Traffic (BulkSend)
    // -----------------------------
    uint16_t port = 9000;
    for (uint32_t i = 0; i < servers.GetN() - 1; i++)
    {
        Ptr<Ipv4> ipv4Dst = servers.Get(i + 1)->GetObject<Ipv4>();
        InetSocketAddress addr(ipv4Dst->GetAddress(1, 0).GetLocal(), port++);
        BulkSendHelper tcp("ns3::TcpSocketFactory", addr);
        tcp.SetAttribute("MaxBytes", UintegerValue(0)); // unlimited
        ApplicationContainer apps = tcp.Install(servers.Get(i));
        apps.Start(Seconds(1.0));
        apps.Stop(Seconds(simTime));
    }

    // -----------------------------
    // Fault injection
    // -----------------------------
    Simulator::Schedule(Seconds(20.0), &LinkFailure, allLinks[3].Get(1));

    // -----------------------------
    // Flow Monitor
    // -----------------------------
    FlowMonitorHelper fm;
    Ptr<FlowMonitor> monitor = fm.InstallAll();
    auto classifier = DynamicCast<Ipv4FlowClassifier>(fm.GetClassifier());

    // -----------------------------
    // Windowed CSV
    // -----------------------------
    std::ofstream windowed("dc_windowed_dataset.csv");
    windowed << "time,flowId,txPackets,rxPackets,lostPackets,throughputMbps,label\n";

    for (int t = 1; t <= int(simTime); ++t)
    {
        uint32_t label = 0; // normal
        if (t >= 10 && t < 20) label = 1; // congestion
        if (t >= 20) label = 2; // failure

        Simulator::Schedule(Seconds(t),
                            &ExportWindowedStats,
                            monitor,
                            classifier,
                            t,
                            std::ref(windowed),
                            label);
    }

    // -----------------------------
    // Run simulation
    // -----------------------------
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    windowed.close();

    // -----------------------------
    // Final CSV export
    // -----------------------------
    monitor->CheckForLostPackets();

    std::ofstream csv("dc_flow_dataset.csv");
    csv << "FlowId,SrcIP,DstIP,TxPackets,RxPackets,LostPackets,ThroughputMbps,DelayMs\n";

    auto stats = monitor->GetFlowStats();
    for (auto &f : stats)
    {
        auto t = classifier->FindFlow(f.first);
        double thr = f.second.rxBytes * 8.0 / simTime / 1e6; // Mbps
        double delay = f.second.rxPackets ? f.second.delaySum.GetSeconds() * 1000 / f.second.rxPackets : 0;

        csv << f.first << ","
            << t.sourceAddress << ","
            << t.destinationAddress << ","
            << f.second.txPackets << ","
            << f.second.rxPackets << ","
            << f.second.lostPackets << ","
            << thr << ","
            << delay << "\n";
    }

    csv.close();

    // -----------------------------
    // Clean up
    // -----------------------------
    Simulator::Destroy();

    std::cout << "\nSimulation finished. CSV files generated:\n"
              << "  - dc_windowed_dataset.csv\n"
              << "  - dc_flow_dataset.csv\n";

    return 0;
}
