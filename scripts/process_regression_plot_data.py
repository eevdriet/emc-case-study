from emc.util import Paths

import json
import matplotlib.pyplot as plt


def show_regression_plot() -> None:
    worms = ['ascaris', 'hookworm']
    strategies = ['sac', 'community']
    frequencies = ['1', '2']

    # Create a single figure
    fig, axs = plt.subplots(len(worms), len(strategies) * len(frequencies), figsize=(16, 8))

    # Flatten axs if it's a single row or column to make indexing easier
    if len(axs.shape) == 1:
        axs = axs.reshape(1, -1)

    for i, worm in enumerate(worms):
        for j, strategy in enumerate(strategies):
            for k, frequency in enumerate(frequencies):
                # Read data from JSON file
                path = Paths.hyperparameter_opt(f'classifier_stats_{worm}_{strategy}_{frequency}_SingleGradientBoosterBayesian.json', True)

                with open(path, 'r') as file:
                    data = json.load(file)

                # Extract metrics
                time_points = list(data.keys())[1:]  # Skip first two time points
                accuracy = [data[time]['accuracy'] for time in time_points]
                f1_score = [data[time]['f1_score'] for time in time_points]

                # Plotting in corresponding subplot
                ax = axs[i, j * len(frequencies) + k]
                ax.plot(time_points, accuracy, marker='o', label='Accuracy')
                ax.plot(time_points, f1_score, marker='o', label='F1-score')

                ax.set_title(f'{worm.capitalize()} - {strategy.capitalize()} - Frequency {frequency}')
                ax.set_xlabel('Time Points')
                ax.set_ylabel('Metrics')
                ax.set_ylim(0, 1)
                ax.legend()
                ax.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show_regression_plot()

# import json
# import matplotlib.pyplot as plt

# worms = ['ascaris', 'hookworm']
# strategies = ['sac', 'community']
# frequencies = ['1', '2']

# for worm in worms:
#     for strategy in strategies:
#         for frequency in frequencies:
#             # Read data from JSON file
#             with open(f'classifier_stats_{worm}_{strategy}_{frequency}_SingleGradientBoosterBayesian.json', 'r') as file:
#                 data = json.load(file)

#             # Extract metrics
#             time_points = list(data.keys())[1:]  # Skip first two time points
#             accuracy = [data[time]['accuracy'] for time in time_points]
#             precision = [data[time]['precision'] for time in time_points]
#             recall = [data[time]['recall'] for time in time_points]
#             f1_score = [data[time]['f1_score'] for time in time_points]

#             # Plotting
#             plt.figure(figsize=(10, 6))

#             plt.plot(time_points, accuracy, marker='o', label='Accuracy')
#             plt.plot(time_points, precision, marker='o', label='Precision')
#             plt.plot(time_points, recall, marker='o', label='Recall')
#             plt.plot(time_points, f1_score, marker='o', label='F1-score')

#             plt.title('Performance Metrics Over Time')
#             plt.xlabel('Time Points')
#             plt.ylabel('Metrics')
#             plt.xticks(rotation=45)
#             plt.legend()
#             plt.grid(True)

#             # Set y-axis limits from 0 to 1
#             plt.ylim(0, 1)

#             plt.tight_layout()
#             plt.show()
