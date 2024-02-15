import pandas as pd
import numpy as np


def process_data(type: str):
    data = pd.read_feather('../data/drug_efficacy_' + type + '.feather')

    # Count number of rows dropped per scenario, per simulation, per treat_time due to NaN values
    dropped_rows_per_group = data.groupby(['scenario', 'simulation', 'treat_time']).apply(lambda x: len(x) - x['post'].count()).reset_index()
    dropped_rows_per_group.columns = ['scenario', 'simulation', 'treat_time', 'skipped_NaN_tests']

    # Drop rows with NaN values in the 'post' column
    data = data.dropna(subset=['post'])

    # Group by 'scenario', 'simulation', and 'treat_time' and calculate required statistics
    grouped = data.groupby(['scenario', 'simulation', 'treat_time']).agg(
        true_a_pre=('pre', 'mean'),
        true_a_post=('post', 'mean'),
        true_total_pre=('pre', 'sum'),
        true_total_post=('post', 'sum'),
        total_useful_tests=('post', 'count')
    ).reset_index()

    # Merge dropped rows information with the grouped dataframe
    grouped = pd.merge(grouped, dropped_rows_per_group, on=['scenario', 'simulation', 'treat_time'], how='left')

    grouped['ERR'] = (1 - (grouped['true_a_post'] / grouped['true_a_pre'])) * 100
    grouped['EPG_change'] = (grouped['true_a_post'] - grouped['true_a_pre']) / grouped['true_a_pre']

    unique_treat_times = grouped['treat_time'].unique()
    print("Unique treat times:", unique_treat_times)

    print(grouped)

    grouped.to_csv('../data/' + type + '_drug_efficacy.csv', index=False)

process_data('ascaris')
# process_data('hookworm')