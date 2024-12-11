import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read and combine all CSV data


data_50x = pd.read_csv('results/12_2024_runs/cohere_unfiltered_50x_mult.csv')
data_60x = pd.read_csv('results/12_2024_runs/cohere_unfiltered_60x_mult.csv')
data_70x = pd.read_csv('results/12_2024_runs/cohere_unfiltered_70x_mult.csv')
data_80x = pd.read_csv('results/12_2024_runs/cohere_unfiltered_80x_mult.csv')
data_90x = pd.read_csv('results/12_2024_runs/cohere_unfiltered_90x_mult.csv')

# Add multiplier column to each dataframe
data_50x['multiplier'] = 50
data_60x['multiplier'] = 60
data_70x['multiplier'] = 70
data_80x['multiplier'] = 80
data_90x['multiplier'] = 90

# Combine all data
combined_data = pd.concat([data_50x, data_60x, data_70x, data_80x, data_90x])

# Extract k value from Test Case
combined_data['k'] = combined_data['Test Case'].apply(lambda x: int(x.split('.k')[1].split('.')[0]))

# Create figure with subplots
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(20, 15))

# Plot 1: Recall vs Multiplier for different k values and concurrency
ax1 = plt.subplot(2, 2, 1)
for k in [10, 100]:
    for concurrency in [1, 10, 100]:
        data = combined_data[
            (combined_data['k'] == k) & 
            (combined_data['Concurrent Requests'] == concurrency)
        ]
        plt.plot(data['multiplier'], data['Recall'], 
                marker='o', 
                label=f'k={k}, conc={concurrency}')

plt.xlabel('Multiplier')
plt.ylabel('Recall (%)')
plt.title('Recall vs Multiplier for Different Configurations')
plt.legend()
plt.grid(True)

# Plot 2: Mean Latency vs Multiplier
ax2 = plt.subplot(2, 2, 2)
for k in [10, 100]:
    for concurrency in [1, 10, 100]:
        data = combined_data[
            (combined_data['k'] == k) & 
            (combined_data['Concurrent Requests'] == concurrency)
        ]
        plt.plot(data['multiplier'], data['Mean Latency (ms)'], 
                marker='o', 
                label=f'k={k}, conc={concurrency}')

plt.xlabel('Multiplier')
plt.ylabel('Mean Latency (ms)')
plt.title('Mean Latency vs Multiplier for Different Configurations')
plt.legend()
plt.grid(True)

# Plot 3: QPS vs Multiplier
ax3 = plt.subplot(2, 2, 3)
for k in [10, 100]:
    for concurrency in [1, 10, 100]:
        data = combined_data[
            (combined_data['k'] == k) & 
            (combined_data['Concurrent Requests'] == concurrency)
        ]
        plt.plot(data['multiplier'], data['QPS'], 
                marker='o', 
                label=f'k={k}, conc={concurrency}')

plt.xlabel('Multiplier')
plt.ylabel('QPS')
plt.title('QPS vs Multiplier for Different Configurations')
plt.legend()
plt.grid(True)

# Plot 4: Recall vs Mean Latency scatter plot
ax4 = plt.subplot(2, 2, 4)
scatter = plt.scatter(combined_data['Mean Latency (ms)'], 
                     combined_data['Recall'],
                     c=combined_data['multiplier'],
                     s=combined_data['Concurrent Requests'] * 50,
                     cmap='viridis')
plt.xlabel('Mean Latency (ms)')
plt.ylabel('Recall (%)')
plt.title('Recall vs Mean Latency Trade-off')
plt.colorbar(scatter, label='Multiplier')
plt.grid(True)

plt.tight_layout()
plt.show()