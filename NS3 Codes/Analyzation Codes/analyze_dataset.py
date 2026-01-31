#!/usr/bin/env python3
"""
Network Fault Dataset Analysis
Run this in the same directory as your CSV files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def analyze_flows(filename="dc_dataset_flows.csv"):
    """Analyze flow statistics dataset"""
    
    print("\n" + "="*60)
    print("FLOW DATASET ANALYSIS")
    print("="*60)
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        return None
    
    # Load data
    flows = pd.read_csv(filename)
    print(f"✓ Loaded {len(flows)} flow samples")
    print(f"  Columns: {list(flows.columns)}")
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Time range: {flows['time'].min():.1f}s to {flows['time'].max():.1f}s")
    print(f"   Unique flows: {flows['flowId'].nunique()}")
    print(f"   Unique source IPs: {flows['srcIP'].nunique()}")
    print(f"   Unique destination IPs: {flows['dstIP'].nunique()}")
    
    # Label analysis
    print(f"\n🏷️  LABEL DISTRIBUTION:")
    label_counts = flows['label'].value_counts().sort_index()
    label_names = {
        0: 'Normal',
        1: 'Link Failure',
        2: 'Congestion'
    }
    
    for label, count in label_counts.items():
        name = label_names.get(label, f'Label {label}')
        percentage = count / len(flows) * 100
        print(f"   {name}: {count} samples ({percentage:.1f}%)")
    
    # Performance metrics by label
    print(f"\n📈 PERFORMANCE METRICS BY LABEL:")
    metrics = ['throughputMbps', 'delayMs', 'jitterMs', 'lostPkts']
    
    for label in sorted(flows['label'].unique()):
        subset = flows[flows['label'] == label]
        name = label_names.get(label, f'Label {label}')
        print(f"\n   {name}:")
        for metric in metrics:
            if metric in subset.columns:
                mean_val = subset[metric].mean()
                std_val = subset[metric].std()
                print(f"     {metric}: {mean_val:.2f} ± {std_val:.2f}")
    
    # Check for anomalies
    print(f"\n⚠️  DATA QUALITY CHECKS:")
    missing = flows.isnull().sum().sum()
    print(f"   Missing values: {missing}")
    
    unrealistic_throughput = len(flows[flows['throughputMbps'] > 1000])
    print(f"   Throughput > 1Gbps: {unrealistic_throughput}")
    
    negative_values = len(flows[(flows['delayMs'] < 0) | (flows['jitterMs'] < 0)])
    print(f"   Negative delays/jitter: {negative_values}")
    
    return flows

def analyze_links(filename="dc_dataset_links.csv"):
    """Analyze link statistics dataset"""
    
    print("\n" + "="*60)
    print("LINK DATASET ANALYSIS")
    print("="*60)
    
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found!")
        return None
    
    links = pd.read_csv(filename)
    print(f"✓ Loaded {len(links)} link samples")
    print(f"  Columns: {list(links.columns)}")
    
    print(f"\n🔗 LINK INFORMATION:")
    print(f"   Unique links: {links['linkId'].nunique()}")
    print(f"   Time range: {links['time'].min():.1f}s to {links['time'].max():.1f}s")
    
    # Status distribution
    print(f"\n📊 LINK STATUS DISTRIBUTION:")
    status_counts = links['status'].value_counts().sort_index()
    status_names = {
        0: 'Normal',
        1: 'Faulty',
        2: 'Congested',
        3: 'Dropping',
        4: 'High Utilization'
    }
    
    for status, count in status_counts.items():
        name = status_names.get(status, f'Status {status}')
        percentage = count / len(links) * 100
        print(f"   {name}: {count} samples ({percentage:.1f}%)")
    
    return links

def create_visualizations(flows, links):
    """Create comprehensive visualizations"""
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Throughput over time by label
    ax1 = plt.subplot(3, 3, 1)
    label_names = {0: 'Normal', 1: 'Link Failure', 2: 'Congestion'}
    
    for label in sorted(flows['label'].unique()):
        subset = flows[flows['label'] == label].copy()
        # Average per time window
        time_avg = subset.groupby('time')['throughputMbps'].mean().reset_index()
        ax1.plot(time_avg['time'], time_avg['throughputMbps'], 
                label=label_names.get(label, f'Label {label}'), linewidth=2)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Throughput (Mbps)')
    ax1.set_title('Average Throughput Over Time by Label')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Delay distribution by label
    ax2 = plt.subplot(3, 3, 2)
    delay_data = []
    labels = []
    
    for label in sorted(flows['label'].unique()):
        subset = flows[flows['label'] == label]
        delay_data.append(subset['delayMs'].dropna().values)
        labels.append(label_names.get(label, f'Label {label}'))
    
    ax2.boxplot(delay_data, labels=labels)
    ax2.set_ylabel('Delay (ms)')
    ax2.set_title('Delay Distribution by Label')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # 3. Label distribution pie chart
    ax3 = plt.subplot(3, 3, 3)
    label_counts = flows['label'].value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, len(label_counts)))
    ax3.pie(label_counts.values, labels=[label_names.get(i, f'Label {i}') for i in label_counts.index],
           autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('Label Distribution')
    
    # 4. Correlation heatmap
    ax4 = plt.subplot(3, 3, 4)
    numeric_cols = ['throughputMbps', 'delayMs', 'jitterMs', 'lostPkts', 'label']
    corr_matrix = flows[numeric_cols].corr()
    im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(numeric_cols)))
    ax4.set_xticklabels(numeric_cols, rotation=45)
    ax4.set_yticks(range(len(numeric_cols)))
    ax4.set_yticklabels(numeric_cols)
    ax4.set_title('Feature Correlation Matrix')
    plt.colorbar(im, ax=ax4)
    
    # 5. Packet loss vs throughput scatter
    ax5 = plt.subplot(3, 3, 5)
    scatter = ax5.scatter(flows['lostPkts'], flows['throughputMbps'], 
                         c=flows['label'], cmap='viridis', alpha=0.6, s=20)
    ax5.set_xlabel('Lost Packets')
    ax5.set_ylabel('Throughput (Mbps)')
    ax5.set_title('Packet Loss vs Throughput')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Label')
    
    # 6. Time series of labels
    ax6 = plt.subplot(3, 3, 6)
    time_labels = flows.groupby('time')['label'].value_counts().unstack().fillna(0)
    time_labels.plot(kind='area', ax=ax6, stacked=True, alpha=0.7)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Count')
    ax6.set_title('Label Distribution Over Time')
    ax6.legend(title='Label')
    ax6.grid(True, alpha=0.3)
    
    # 7. Link status heatmap (if link data exists)
    if links is not None:
        ax7 = plt.subplot(3, 3, 7)
        # Sample some links for readability
        sample_links = sorted(links['linkId'].unique())[:20]
        links_sample = links[links['linkId'].isin(sample_links)]
        
        link_pivot = links_sample.pivot_table(index='time', columns='linkId', 
                                            values='status', aggfunc='first')
        im = ax7.imshow(link_pivot.T, aspect='auto', cmap='tab10', vmin=0, vmax=4)
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Link ID')
        ax7.set_title('Link Status Timeline (Sample)')
        plt.colorbar(im, ax=ax7, label='Status')
    
    # 8. Queue depth over time
    if links is not None and 'queueDepth' in links.columns:
        ax8 = plt.subplot(3, 3, 8)
        queue_avg = links.groupby('time')['queueDepth'].mean().reset_index()
        ax8.plot(queue_avg['time'], queue_avg['queueDepth'], linewidth=2, color='red')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Average Queue Depth')
        ax8.set_title('Queue Depth Over Time')
        ax8.grid(True, alpha=0.3)
    
    # 9. Export summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    Dataset Summary:
    --------------------
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    Total flow samples: {len(flows):,}
    Time range: {flows['time'].min():.1f}s to {flows['time'].max():.1f}s
    
    Label Distribution:
    {' / '.join([f'{label_names.get(i, i)}: {c/len(flows)*100:.1f}%' 
                 for i, c in flows['label'].value_counts().sort_index().items()])}
    
    Performance Averages:
    Throughput: {flows['throughputMbps'].mean():.2f} Mbps
    Delay: {flows['delayMs'].mean():.2f} ms
    Jitter: {flows['jitterMs'].mean():.2f} ms
    """
    
    if links is not None:
        summary_text += f"""
    Link samples: {len(links):,}
    Unique links: {links['linkId'].nunique()}
    """
    
    ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Network Fault Dataset Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'dataset_analysis_{timestamp}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualizations saved to: {output_file}")
    
    plt.show()
    
    return output_file

def save_statistical_report(flows, links):
    """Save detailed statistical report to file"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'dataset_report_{timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("NETWORK FAULT DATASET STATISTICAL REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Flow data file: dc_dataset_flows.csv\n")
        f.write(f"Link data file: dc_dataset_links.csv\n\n")
        
        f.write("1. DATASET OVERVIEW\n")
        f.write("-"*40 + "\n")
        f.write(f"Total flow samples: {len(flows):,}\n")
        f.write(f"Time range: {flows['time'].min():.1f}s to {flows['time'].max():.1f}s\n")
        f.write(f"Sampling interval: 1 second\n")
        f.write(f"Unique flows: {flows['flowId'].nunique():,}\n\n")
        
        f.write("2. LABEL DISTRIBUTION\n")
        f.write("-"*40 + "\n")
        label_counts = flows['label'].value_counts().sort_index()
        label_names = {0: 'Normal', 1: 'Link Failure', 2: 'Congestion'}
        
        for label, count in label_counts.items():
            name = label_names.get(label, f'Label {label}')
            percentage = count / len(flows) * 100
            f.write(f"{name:<15}: {count:>8} samples ({percentage:6.2f}%)\n")
        
        f.write("\n3. PERFORMANCE METRICS\n")
        f.write("-"*40 + "\n")
        metrics = ['throughputMbps', 'delayMs', 'jitterMs', 'lostPkts']
        
        for metric in metrics:
            if metric in flows.columns:
                f.write(f"\n{metric}:\n")
                f.write(f"  Mean: {flows[metric].mean():.2f}\n")
                f.write(f"  Std:  {flows[metric].std():.2f}\n")
                f.write(f"  Min:  {flows[metric].min():.2f}\n")
                f.write(f"  Max:  {flows[metric].max():.2f}\n")
                f.write(f"  25%:  {flows[metric].quantile(0.25):.2f}\n")
                f.write(f"  50%:  {flows[metric].quantile(0.50):.2f}\n")
                f.write(f"  75%:  {flows[metric].quantile(0.75):.2f}\n")
        
        if links is not None:
            f.write("\n4. LINK STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total link samples: {len(links):,}\n")
            f.write(f"Unique links: {links['linkId'].nunique():,}\n")
            
            if 'status' in links.columns:
                f.write("\nLink status distribution:\n")
                status_counts = links['status'].value_counts().sort_index()
                status_names = {0: 'Normal', 1: 'Faulty', 2: 'Congested', 3: 'Dropping'}
                
                for status, count in status_counts.items():
                    name = status_names.get(status, f'Status {status}')
                    percentage = count / len(links) * 100
                    f.write(f"{name:<15}: {count:>8} samples ({percentage:6.2f}%)\n")
        
        f.write("\n5. DATA QUALITY\n")
        f.write("-"*40 + "\n")
        missing = flows.isnull().sum().sum()
        f.write(f"Missing values in flows: {missing}\n")
        
        if links is not None:
            missing_links = links.isnull().sum().sum()
            f.write(f"Missing values in links: {missing_links}\n")
        
        f.write("\n6. RECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        
        # Check class balance
        max_count = label_counts.max()
        min_count = label_counts.min()
        imbalance = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance > 10:
            f.write("⚠️  HIGH CLASS IMBALANCE DETECTED\n")
            f.write("   Consider techniques for imbalanced datasets:\n")
            f.write("   - Oversampling minority classes\n")
            f.write("   - Undersampling majority classes\n")
            f.write("   - Class weighting in loss function\n")
            f.write("   - Generate more data for minority classes\n")
        
        if missing > 0:
            f.write("⚠️  MISSING VALUES FOUND\n")
            f.write("   Consider:\n")
            f.write("   - Imputation (mean, median, etc.)\n")
            f.write("   - Removing samples with missing values\n")
            f.write("   - Investigating data collection issues\n")
        
        # Check for unrealistic values
        unrealistic = len(flows[flows['throughputMbps'] > 1000])
        if unrealistic > 0:
            f.write(f"⚠️  {unrealistic} samples with throughput > 1Gbps\n")
            f.write("   Consider capping or investigating these values\n")
        
        f.write("\n7. SUGGESTED NEXT STEPS\n")
        f.write("-"*40 + "\n")
        f.write("1. Generate more data with different seeds\n")
        f.write("2. Try different fault injection scenarios\n")
        f.write("3. Split data into train/validation/test sets\n")
        f.write("4. Experiment with different ML models\n")
        f.write("5. Monitor model performance on each class\n")
    
    print(f"✓ Statistical report saved to: {report_file}")
    return report_file

def main():
    """Main analysis function"""
    
    print("\n" + "="*60)
    print("NETWORK FAULT DATASET ANALYZER")
    print("="*60)
    print(f"Working directory: {os.getcwd()}")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for required packages
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"\n❌ Missing required packages: {e}")
        print("Please install with: pip3 install pandas matplotlib seaborn numpy")
        return
    
    # Analyze datasets
    flows = analyze_flows("dc_dataset_flows.csv")
    if flows is None:
        return
    
    links = analyze_links("dc_dataset_links.csv")
    
    # Create visualizations
    try:
        create_visualizations(flows, links)
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")
        print("Continuing with report generation...")
    
    # Save detailed report
    report_file = save_statistical_report(flows, links)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print(f"  1. dataset_analysis_*.png - Visualizations")
    print(f"  2. dataset_report_*.txt   - Detailed statistical report")
    print("\nNext steps:")
    print("  1. Review the generated reports")
    print("  2. Generate more data if needed")
    print("  3. Start ML model training")
    print("="*60)

if __name__ == "__main__":
    main()
