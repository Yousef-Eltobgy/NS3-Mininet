#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=== ANALYZING IMPROVED DATASET ===")

# Load improved dataset
flows = pd.read_csv('dc_improved_flows.csv')
links = pd.read_csv('dc_improved_links.csv')

print(f"Improved flows: {len(flows)} samples")
print(f"Improved links: {len(links)} samples")
print(f"Time range: {flows['time'].min():.1f}s to {flows['time'].max():.1f}s")

print("\n=== LABEL DISTRIBUTION ===")
label_counts = flows['label'].value_counts().sort_index()
label_names = {0: 'Normal', 1: 'Link Failure', 2: 'Congestion'}

for label, count in label_counts.items():
    name = label_names.get(label, f'Label {label}')
    percentage = count / len(flows) * 100
    print(f"{name}: {count} samples ({percentage:.1f}%)")

print("\n=== PERFORMANCE BY LABEL ===")
for label in sorted(flows['label'].unique()):
    subset = flows[flows['label'] == label]
    name = label_names.get(label, f'Label {label}')
    print(f"\n{name}:")
    print(f"  Throughput: {subset['throughputMbps'].mean():.2f} ± {subset['throughputMbps'].std():.2f} Mbps")
    print(f"  Delay: {subset['delayMs'].mean():.2f} ± {subset['delayMs'].std():.2f} ms")
    print(f"  Jitter: {subset['jitterMs'].mean():.2f} ± {subset['jitterMs'].std():.2f} ms")
    print(f"  Lost packets: {subset['lostPkts'].mean():.2f} ± {subset['lostPkts'].std():.2f}")

print("\n=== DATA QUALITY ===")
# Check if we have all three labels
has_congestion = 2 in flows['label'].unique()
print(f"Has congestion data: {has_congestion}")

# Check for non-zero throughput in link failures
link_failures = flows[flows['label'] == 1]
non_zero = len(link_failures[link_failures['throughputMbps'] > 0])
print(f"Link failures with throughput > 0: {non_zero}/{len(link_failures)}")

# Quick visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Throughput distribution by label
for label in sorted(flows['label'].unique()):
    subset = flows[flows['label'] == label]
    axes[0,0].hist(subset['throughputMbps'], bins=50, alpha=0.5, 
                   label=label_names.get(label, f'Label {label}'))
axes[0,0].set_xlabel('Throughput (Mbps)')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Throughput Distribution by Label')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Label distribution over time
time_labels = flows.groupby('time')['label'].value_counts().unstack().fillna(0)
time_labels.plot(kind='area', ax=axes[0,1], stacked=True, alpha=0.7)
axes[0,1].set_xlabel('Time (s)')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Label Distribution Over Time')
axes[0,1].legend(title='Label')
axes[0,1].grid(True, alpha=0.3)

# 3. Link status distribution
if not links.empty and 'status' in links.columns:
    status_counts = links['status'].value_counts().sort_index()
    status_names = {0: 'Normal', 1: 'Faulty', 2: 'Congested', 3: 'Dropping', 4: 'Mild Fault', 5: 'High Queue'}
    colors = plt.cm.Set3(np.linspace(0, 1, len(status_counts)))
    axes[1,0].pie(status_counts.values, labels=[status_names.get(i, f'Status {i}') for i in status_counts.index],
                 autopct='%1.1f%%', colors=colors[:len(status_counts)])
    axes[1,0].set_title('Link Status Distribution')

# 4. Correlation matrix
numeric_cols = ['throughputMbps', 'delayMs', 'jitterMs', 'lostPkts', 'label']
corr_matrix = flows[numeric_cols].corr()
im = axes[1,1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
axes[1,1].set_xticks(range(len(numeric_cols)))
axes[1,1].set_xticklabels(numeric_cols, rotation=45)
axes[1,1].set_yticks(range(len(numeric_cols)))
axes[1,1].set_yticklabels(numeric_cols)
axes[1,1].set_title('Feature Correlation')
plt.colorbar(im, ax=axes[1,1])

plt.suptitle('Improved Dataset Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('improved_dataset_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Chart saved to improved_dataset_analysis.png")
