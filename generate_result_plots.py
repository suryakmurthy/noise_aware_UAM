import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from JSON file
with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/eval/test_noise_mode_001.json', 'r') as file:
    data_001 = json.load(file)

with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/eval/test_noise_mode_0001.json', 'r') as file:
    data_0001 = json.load(file)

with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/eval/test_noise_mode_00001.json', 'r') as file:
    data_00001 = json.load(file)

with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/eval/test_noise_mode_000001.json', 'r') as file:
    data_000001 = json.load(file)

# Extracting data for plotting
scenario_nums = [i for i, entry in enumerate(data_001)]
scenario_nums_mod = [scenario_nums[i] for i in range(0, len(scenario_nums)-1)]

los_totals = [entry['los'] for entry in data_001]
max_noise_inc = [entry['max_noise'] for entry in data_001]
avg_noise_inc = [entry['avg_noise'] for entry in data_001]
spd_changes = [entry['speed_change_counter'] for entry in data_001]
alt_changes = [entry['alt_change_counter'] for entry in data_001]
print("Max Noise Increase 0.001: ", np.mean(max_noise_inc) )
print("Num Alt Changes 0.001: ", np.mean(alt_changes) )

los_totals = [entry['los'] for entry in data_0001]
max_noise_inc = [entry['max_noise'] for entry in data_0001]
avg_noise_inc = [entry['avg_noise'] for entry in data_0001]
spd_changes = [entry['speed_change_counter'] for entry in data_0001]
alt_changes = [entry['alt_change_counter'] for entry in data_0001]
print("Max Noise Increase 0.0001: ", np.mean(max_noise_inc) )
print("Num Alt Changes 0.0001: ", np.mean(alt_changes) )

los_totals = [entry['los'] for entry in data_00001]
max_noise_inc = [entry['max_noise'] for entry in data_00001]
avg_noise_inc = [entry['avg_noise'] for entry in data_00001]
spd_changes = [entry['speed_change_counter'] for entry in data_00001]
alt_changes = [entry['alt_change_counter'] for entry in data_00001]
print("Max Noise Increase 0.00001: ", np.mean(max_noise_inc) )
print("Num Alt Changes 0.00001: ", np.mean(alt_changes) )

los_totals = [entry['los'] for entry in data_000001]
max_noise_inc = [entry['max_noise'] for entry in data_000001]
avg_noise_inc = [entry['avg_noise'] for entry in data_000001]
spd_changes = [entry['speed_change_counter'] for entry in data_000001]
alt_changes = [entry['alt_change_counter'] for entry in data_000001]
print("Max Noise Increase 0.000001: ", np.mean(max_noise_inc) )
print("Num Alt Changes 0.000001: ", np.mean(alt_changes) )

with open('log/eval/aircraft_mod_alt_test_safe.json', 'r') as file:
    data_safe = json.load(file)

with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/eval/test_noise.json', 'r') as file:
    data_01 = json.load(file)

with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/eval/test_noise_and_safety.json', 'r') as file:
    data_noise_and_safe = json.load(file)

avg_noise_inc_safe = [entry['avg_noise'] for entry in data_safe]
avg_noise_inc_01 = [entry['avg_noise'] for entry in data_01]
avg_noise_inc = [entry['avg_noise'] for entry in data_noise_and_safe]



noise_histogram = [item for i in range(0, len(avg_noise_inc)) for sublist in avg_noise_inc[i].values() for item in sublist]
noise_histogram_01 = [item for i in range(0, len(avg_noise_inc_01)) for sublist in avg_noise_inc_01[i].values() for item in sublist]
noise_hist_safe = [item for i in range(0, len(avg_noise_inc_safe)) for sublist in avg_noise_inc_safe[i].values() for item in sublist]


print(len(noise_histogram), len(noise_histogram_01), len(noise_hist_safe))
num_bins = 10

num_bins = 10
bar_width = 0.2  # Width of each bar
spacing = 0.1  # Spacing between groups of bars


# Define the bin edges
bins = np.histogram_bin_edges(np.concatenate((noise_histogram, noise_histogram_01, noise_hist_safe)), bins=num_bins)
# bins = np.histogram_bin_edges(noise_histogram, bins=num_bins)


# Calculate the histogram counts
counts_safe, _ = np.histogram(noise_hist_safe, bins=bins, density=True)
counts_01, _ = np.histogram(noise_histogram_01, bins=bins, density=True)
counts_both, _ = np.histogram(noise_histogram, bins=bins, density=True)

# Define the positions for the bars
positions = bins[:-1] * (bar_width * 3 + spacing)

# Plot the bars side by side
plt.bar(positions - bar_width, counts_safe, width=bar_width, color='red', alpha=0.7, label="Safety-Tuned RL")
plt.bar(positions, counts_01, width=bar_width, color='blue', alpha=0.7, label='Noise-Tuned RL')
plt.bar(positions + bar_width, counts_both, width=bar_width, color='green', alpha=0.7, label='Noise and Safety')

plt.xlabel('Cumulative Noise Increase (dB)')
plt.ylabel('Percentage of Time Steps')
plt.title(f"Cumulative Noise Increase")
plt.legend()
plt.show()