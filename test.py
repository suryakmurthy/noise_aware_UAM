import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from JSON file
with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/eval/aircraft_mod_alt_train_001_1.json', 'r') as file:
    data = json.load(file)

# Extracting data for plotting
scenario_nums = [i for i, entry in enumerate(data)]
scenario_nums_mod = [scenario_nums[i] for i in range(0, len(scenario_nums)-1)]

max_noise_inc = [entry['max_noise'] for entry in data]
avg_noise_inc = [entry['avg_noise'] for entry in data]
spd_changes = [entry['speed_change_counter'] for entry in data]
alt_changes = [entry['alt_change_counter'] for entry in data]


shield_totals = [entry['shield_total'] for entry in data]
los_totals = [entry['los'] for entry in data]


shield_totals_i = [entry['shield_total_intersection'] for entry in data]
shield_totals_j = [entry['shield_total_route'] for entry in data]

max_travel_times = [entry['max_travel_time'] for entry in data]

print("LOS mean: ", np.mean(los_totals))
print("Max Noise Increase mean: ", np.mean(max_noise_inc))
print("Avg Noise Increase mean: ", np.mean(avg_noise_inc))
print("max travel time: ", np.mean(max_travel_times))

# max_travel_times = [max_travel_times[i] for i in range(0, len(max_travel_times)-1)]
RM = 30
st_df = pd.DataFrame(shield_totals)
st_df = st_df.rolling(RM).mean()

si_df = pd.DataFrame(shield_totals_i)
si_df = si_df.rolling(RM).mean()

sr_df = pd.DataFrame(shield_totals_j)
sr_df = sr_df.rolling(RM).mean()




# plt.plot(nmac_no_tm.rolling(RM).mean())
# plt.plot(nmac_uneq.rolling(RM).mean())
# plt.plot(st_df)
# plt.xlabel('Scenario Number')
# plt.ylabel('Total Shield Interventions')
# plt.title('Scenario Number vs Total Shield Interventions (20 Vehicles)')
# plt.tight_layout()
# plt.show()

# plt.plot(sr_df)
# plt.xlabel('Scenario Number')
# plt.ylabel('Shield Interventions at Routes')
# plt.title('Scenario Number vs Shield Interventions at Routes (20 Vehicles)')
# plt.tight_layout()
# plt.show()

# plt.plot(si_df)
# plt.xlabel('Scenario Number')
# plt.ylabel('Shield Interventions at Intersections')
# plt.title('Scenario Number vs Shield Interventions at Intersections (20 Vehicles)')
# plt.tight_layout()
# plt.show()

# max_val = np.argmax(shield_totals)
# del shield_totals[max_val]
# del scenario_nums[max_val]
# del los_totals[max_val]
# del shield_totals_i[max_val]
# del shield_totals_j[max_val]
# del max_travel_times[max_val]
# del scenario_nums_mod[max_val]

# Plotting
# plt.figure(figsize=(30, 8))  # Adjust the figure size as needed

# Plotting the data in separate subplots
# plt.plot(scenario_nums, shield_totals, marker='o', linestyle='--', color='purple')
# plt.xlabel('Scenario Number')
# plt.ylabel('Total Shield Interventions')
# plt.title('Scenario Number vs Total Shield Interventions (Reward = -0.0001)')
# plt.tight_layout()
# plt.show()

# plt.plot(scenario_nums, shield_totals_i, marker='o', linestyle='--', color='blue')
# plt.xlabel('Scenario Number')
# plt.ylabel('Shield Interventions at Intersections')
# plt.title('Scenario Number vs Shield Interventions at Intersections (Reward = -0.0001)')
# plt.tight_layout()
# plt.show()

# plt.plot(scenario_nums, shield_totals_j, marker='o', linestyle='--', color='red')
# plt.xlabel('Scenario Number')
# plt.ylabel('Shield Interventions at Routes')
# plt.title('Scenario Number vs Shield Interventions at Routes (Reward = -0.0001)')
# plt.tight_layout()
# plt.show()

plt.plot(scenario_nums, max_noise_inc, marker='o', linestyle='--', color='orange')
plt.xlabel('Scenario Number')
plt.ylabel('Maximum Noise Increase')
plt.title('Scenario Number vs Maximum Noise Increase (alpha = 0, beta = 1)')
plt.tight_layout()
plt.show()

plt.plot(scenario_nums, avg_noise_inc, marker='o', linestyle='--', color='green')
plt.xlabel('Scenario Number')
plt.ylabel('Average Noise Increase')
plt.title('Scenario Number vs Average Noise Increase (alpha = 0, beta = 1)')
plt.tight_layout()
plt.show()


plt.plot(scenario_nums, spd_changes, marker='o', linestyle='--', color='green')
plt.xlabel('Scenario Number')
plt.ylabel('Number of Speed Changes')
plt.title('Scenario Number vs Speed Changes (alpha = 0, beta = 1)')
plt.tight_layout()
plt.show()

plt.plot(scenario_nums, alt_changes, marker='o', linestyle='--', color='green')
plt.xlabel('Scenario Number')
plt.ylabel('Number of Alt Changes')
plt.title('Scenario Number vs Alt Changes (alpha = 0, beta = 1)')
plt.tight_layout()
plt.show()
