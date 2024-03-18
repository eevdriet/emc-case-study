import pandas as pd

from emc.util import Paths, Writer
from emc.data.constants import *
import matplotlib.pyplot as plt
import seaborn as sns
from emc.data.constants import *


policies = {
    'total_costs' : {
        'sac' : {
            '1' : '[0, 8, 16]',
            '2' : '[0, 5, 10, 15]'
        },
        'community' : {
            '1' : '[0, 5, 10, 15]',
            '2' : '[0, 3, 6, 9, 12, 15, 18]'
        }
    },
    'responsiveness' : {
        'sac' : {
            '1' : '[0, 4, 8, 12, 16]',
            '2' : '[0, 4, 8, 12, 16]'
        },
        'community' : {
            '1' : '[0, 4, 8, 12, 16]',
            '2' : '[0, 4, 8, 12, 16]'
        }
    },
    'financial_costs' : {
        'sac' : {
            '1' : '[0, 17]',
            '2' : '[0, 12]'
        },
        'community' : {
            '1' : '[0, 10]',
            '2' : '[0, 10]'
        }
    },
    '5year_policy' : {
        'sac' : {
            '1' : '[0, 5, 10, 15]',
            '2' : '[0, 5, 10, 15]'
        },
        'community' : {
            '1' : '[0, 5, 10, 15]',
            '2' : '[0, 5, 10, 15]'
        }
    }
}


def create_plot(method : str, worm : str, frequency : str, strategy : str, policies : dict):
    policy = policies[method][strategy][frequency]
    json_path = Paths.data('mc') / method / f"{worm}_{strategy}_{frequency}_GradientBoosterOptuna__Policy({policy}).json"

    data = Writer.read_json_file(json_path)

    json_path2 = Paths.data('policies') / f"{worm}{frequency}{strategy}" / f"total_costs_identity_{method}.json"
    model_data = Writer.read_json_file(json_path2)

    df = pd.DataFrame(data).T

    fig, axs = plt.subplots(1, 5, figsize=(15, 5))

    # Plot histogram for accuracy
    sns.histplot(df['accuracy'], color=GREEN, kde=True, ax=axs[0])
    axs[0].set_title('Distribution of Accuracy')
    axs[0].set_xlabel('Accuracy')
    axs[0].set_ylabel('Frequency')
    axs[0].axvline(x=model_data[policy]['accuracy'], color=MAGENTA, linestyle='--', label='Trained Model')  # Vertical line at 4

    # Plot histogram for financial costs
    sns.histplot(df['n_false_positives'], color=BLUE, kde=True, ax=axs[1])
    axs[1].set_title('Distribution of n_false_positives')
    axs[1].set_xlabel('n_false_positives')
    axs[1].set_ylabel('Frequency')
    axs[1].axvline(x=model_data[policy]['n_false_positives'], color=MAGENTA, linestyle='--')

    # Plot histogram for financial costs
    sns.histplot(df['n_false_negatives'], color=BLUE, kde=True, ax=axs[2])
    axs[2].set_title('Distribution of n_false_negatives')
    axs[2].set_xlabel('n_false_negatives')
    axs[2].set_ylabel('Frequency')
    axs[2].axvline(x=model_data[policy]['n_false_negatives'], color=MAGENTA, linestyle='--')

    # Plot histogram for avg_lateness
    sns.histplot(df['avg_lateness'], color=YELLOW, kde=True, ax=axs[3])
    axs[3].set_title('Distribution of Average Lateness')
    axs[3].set_xlabel('Average Lateness')
    axs[3].set_ylabel('Frequency')
    axs[3].axvline(x=model_data[policy]['avg_lateness'], color=MAGENTA, linestyle='--', label='Model Predicted')  # Vertical line at 6

    # Plot histogram for financial costs
    sns.histplot(df['financial_costs'], color=VIOLET, kde=True, ax=axs[4])
    axs[4].set_title('Distribution of Financial Costs')
    axs[4].set_xlabel('Financial Costs')
    axs[4].set_ylabel('Frequency')
    axs[4].axvline(x=model_data[policy]['financial_costs'], color=MAGENTA, linestyle='--')

    # Add legend for trained model and model predicted
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False)

    # plt.tight_layout()
    # plt.show()

    path = Paths.data('plots') / "model_accuracy" / f"{worm}{strategy}{frequency}_{method}.png"
    fig.savefig(path, transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    worms = ['ascaris', 'hookworm']

    for worm in worms:
        for method in policies:
            for strategy in policies[method]:
                for frequency in policies[method][strategy]:
                    create_plot(method, worm, frequency, strategy, policies)