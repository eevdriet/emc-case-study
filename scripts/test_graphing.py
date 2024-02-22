import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from emc.util import Paths

path = Paths.stats()

with open(path, 'r') as file:
    stats = json.load(file)


### Timing graph ###
time_dict = stats['time']
times = []
time_rates = []
for time in time_dict.keys():
    times.append(int(time))
    time_rates.append(time_dict[time]["rejected"] / (time_dict[time]["not_rejected"] + time_dict[time]["rejected"]))

# # Create a DataFrame
# df = pd.DataFrame({'Time': times, 'Values': time_rates})

# # Set Seaborn style and plot
# sns.set_theme()
# sns.lineplot(x='Time', y='Values', data=df)

# # Customize and show the plot
# plt.xlabel('Time')
# plt.ylim(0,1)
# plt.ylabel('Null Hypothesis Rejection Rate')
# plt.title('Null Hypothesis Rejection Rate over Time')
# plt.show()


### Baseline graph ###
baseline_dict = stats['baseline']
buckets = []
bucket_rates = []
for bucket in baseline_dict.keys():
    buckets.append(bucket + f"-{int(bucket)+10}")
    bucket_rates.append(baseline_dict[bucket]["rejected"] / (baseline_dict[bucket]["not_rejected"] + baseline_dict[bucket]["rejected"]))

# # Create a DataFrame
# df = pd.DataFrame({'Bars': buckets, 'Values': bucket_rates})

# # Set Seaborn style and plot
# sns.set_theme()
# sns.barplot(x='Bars', y='Values', data=df)

# # Customize and show the plot
# plt.xlabel('Baseline infection level')
# plt.ylim(0,1)
# plt.ylabel('Null Hypothesis Rejection Rate')
# plt.title('Null Hypothesis Rejection Rates per Baseline Infection Level')
# plt.show()


### Strategy graph ###
strategy_dict = stats['strategy']
strategies = []
strategy_rates = []
for strategy in strategy_dict.keys():
    strategies.append(strategy)
    strategy_rates.append(strategy_dict[strategy]["rejected"] / (strategy_dict[strategy]["not_rejected"] + strategy_dict[strategy]["rejected"]))

# # Create a DataFrame
# df = pd.DataFrame({'Bars': strategies, 'Values': strategy_rates})

# # Set Seaborn style and plot
# sns.set_theme()
# sns.barplot(x='Bars', y='Values', data=df)

# # Customize and show the plot
# plt.xlabel('Strategy')
# plt.ylim(0,1)
# plt.ylabel('Null Hypothesis Rejection Rate')
# plt.title('Null Hypothesis Rejection Rates per Strategy')
# plt.show()
    

### Frequency graph ###
frequency_dict = stats['frequency']
frequencies = []
frequency_rates = []
for frequency in frequency_dict.keys():
    frequencies.append(frequency)
    frequency_rates.append(frequency_dict[frequency]["rejected"] / (frequency_dict[frequency]["not_rejected"] + frequency_dict[frequency]["rejected"]))

# # Create a DataFrame
# df = pd.DataFrame({'Bars': frequencies, 'Values': frequency_rates})

# # Set Seaborn style and plot
# sns.set_theme()
# sns.barplot(x='Bars', y='Values', data=df)

# # Customize and show the plot
# plt.xlabel('Frequency')
# plt.ylim(0,1)
# plt.ylabel('Null Hypothesis Rejection Rate')
# plt.title('Null Hypothesis Rejection Rates per Frequency')
# plt.show()


### Res mode graph ###
resmode_dict = stats['res_mode']
resmodes = []
resmode_rates = []
for resmode in resmode_dict.keys():
    resmodes.append(resmode)
    resmode_rates.append(resmode_dict[resmode]["rejected"] / (resmode_dict[resmode]["not_rejected"] + resmode_dict[resmode]["rejected"]))

# # Create a DataFrame
# df = pd.DataFrame({'Bars': resmodes, 'Values': resmode_rates})

# # Set Seaborn style and plot
# sns.set_theme()
# sns.barplot(x='Bars', y='Values', data=df)

# # Customize and show the plot
# plt.xlabel('Resistance mode')
# plt.ylim(0,1)
# plt.ylabel('Null Hypothesis Rejection Rate')
# plt.title('Null Hypothesis Rejection Rates per Resistance Mode')
# plt.show()


### Combination graph ###
combo_dict = stats['combos']
combos = []
combo_rates = []
for combo in combo_dict.keys():
    combos.append(combo)
    combo_rates.append(combo_dict[combo]["rejected"] / (combo_dict[combo]["not_rejected"] + combo_dict[combo]["rejected"]))

# # Create a DataFrame
# df = pd.DataFrame({'Bars': combos, 'Values': combo_rates})

# # Set Seaborn style and plot
# sns.set_theme()
# sns.barplot(x='Bars', y='Values', data=df)

# # Customize and show the plot
# plt.xlabel('Combination')
# plt.ylim(0,1)
# plt.ylabel('Null Hypothesis Rejection Rate')
# plt.title('Null Hypothesis Rejection Rates per Combination')
# plt.show()


