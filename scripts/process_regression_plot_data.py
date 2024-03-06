from emc.util import Paths
from emc.data.constants import *

import json
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import json
from enum import Enum, auto


class DisplayMode(Enum):
    ALL = auto()
    SEPARATELY = auto()


def show_regression_plot(display_mode: DisplayMode = DisplayMode.ALL) -> None:
    worms = ['ascaris', 'hookworm']
    strategies = ['sac', 'community']
    frequencies = ['1', '2']

    if display_mode == DisplayMode.ALL:
        # Create a single figure
        fig, axs = plt.subplots(len(worms), len(strategies) * len(frequencies), figsize=(16, 8))

        # Flatten axs if it's a single row or column to make indexing easier
        if len(axs.shape) == 1:
            axs = axs.reshape(1, -1)

        # Adjust spacing between rows
        plt.subplots_adjust(hspace=0.3)

    elif display_mode == DisplayMode.SEPARATELY:
        fig = None
        axs = None

    lines = []  # To store the line objects for the legend
    labels = []  # To store the labels for the legend

    for i, worm in enumerate(Worm):
        worm = worm.value

        for j, strategy in enumerate(MDA_STRATEGIES):
            for k, frequency in enumerate(MDA_FREQUENCIES):
                # Read data from JSON file
                path1 = Paths.hyperparameter_opt(
                    f'classifier_stats_f1score_{worm}_{strategy}_{frequency}_SingleGradientBoosterBayesian.json', True)
                path2 = Paths.hyperparameter_opt(
                    f'classifier_stats_{worm}_{strategy}_{frequency}_SingleGradientBoosterDefault.json', True)
                path3 = Paths.hyperparameter_opt(
                    f'classifier_stats_{worm}_{strategy}_{frequency}_SingleGradientBoosterBayesian.json', True)

                with open(path1, 'r') as file:
                    data = json.load(file)

                # Extract metrics
                time_points = list(data.keys())[1:]  # Skip first time point
                accuracy = [data[time]['accuracy'] for time in time_points]
                f1_score = [data[time]['f1_score'] for time in time_points]

                with open(path2, 'r') as file:
                    data = json.load(file)

                accuracy_def = [data[time]['accuracy'] for time in time_points]
                f1_score_def = [data[time]['f1_score'] for time in time_points]

                with open(path3, 'r') as file:
                    data = json.load(file)

                accuracy_opt = [data[time]['accuracy'] for time in time_points]
                f1_score_opt = [data[time]['f1_score'] for time in time_points]

                if display_mode == DisplayMode.ALL:
                    # Plotting in corresponding subplot
                    ax = axs[i, j * len(frequencies) + k]
                elif display_mode == DisplayMode.SEPARATELY:
                    # Create a separate figure for each subplot
                    fig, ax = plt.subplots(figsize=(8, 4))

                line1, = ax.plot(time_points, f1_score_def, label='F1 score', alpha=1, color=YELLOW)
                # line2, = ax.plot(time_points, accuracy_opt, label='Optimized Accuracy by MSE', alpha=1, color='#FFA07A')
                line2, = ax.plot(time_points, accuracy_def, label='Accuracy', alpha=1, color=MAGENTA)

                # #CD5C5C')
                # # line2, = ax.plot(time_points, accuracy_opt, label='Optimized Accuracy by MSE', alpha=1, color='#FFA07A')
                # line2, = ax.plot(time_points, f1_score, label='F1 score Optimised', alpha=1, color='#008000') 
                plt.gcf().set_facecolor('none')

                if i == j == k == 0:
                    lines.extend([line1, line2])
                    labels.extend(['F1 score', 'Accuracy'])

                ax.set_title(f'{worm}-{strategy[0]}-{frequency}')
                ax.set_xticks(range(N_YEARS), labels=[str(year) if year % 2 == 0 else "" for year in range(N_YEARS)])
                ax.set_xlabel('Time (years)')
                ax.set_ylabel('Metrics')
                ax.set_ylim(0, 1)
                ax.grid(False)

                if display_mode == DisplayMode.SEPARATELY:
                    # Create a separate figure for each subplot
                    fig, ax = plt.subplots(figsize=(7, 6))

                    line1, = ax.plot(time_points, f1_score_def, label='Default accuracy', alpha=1, color=YELLOW)
                    line2, = ax.plot(time_points, f1_score, label='Optimized accuracy by F1', alpha=1, color=GREEN)

                    # Set the title below the figure
                    # ax.set_title(f'{worm.capitalize()} - {strategy.capitalize()} - Frequency {frequency}', y=-0.3)

                    ax.set_xlabel('Time (years)')
                    ax.set_ylabel('Accuracy')
                    ax.set_ylim(0, 1)
                    ax.grid(False)

                    # Set the legend below the figure
                    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2)
                    legend.get_frame().set_alpha(None)
                    legend.get_frame().set_facecolor((1, 1, 1, 0))

                    freq_str = f'{frequency}year' if frequency else 'any_freq'
                    strategy_str = strategy if strategy else 'anyone'
                    path = Paths.hyperparameter_opt(f"{worm}_{freq_str}_{strategy_str}.png", plotdata=True)

                    plt.savefig(path, bbox_inches='tight', transparent=True)
                    plt.clf()

    if display_mode == DisplayMode.ALL:
        path = Paths.hyperparameter_opt(f"plots.png", plotdata=True)
        legend = fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0))
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((1, 1, 1, 0))

        fig.savefig(path, transparent=True, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    show_regression_plot(DisplayMode.ALL)

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
