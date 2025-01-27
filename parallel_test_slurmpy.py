import json
import os
import pickle
import shutil
import tempfile
from itertools import islice, product

import numpy as np
import pandas as pd
import slurmpy

# read in files
UK_politeness = pd.read_csv('UK_direct_df.csv')
US_politeness = pd.read_csv('US_direct_df.csv')
UK_dialogue = pd.read_csv('UK_df.csv')
US_dialogue = pd.read_csv('US_df.csv')
dataframes = [UK_politeness, US_politeness, UK_dialogue, US_dialogue]

def chunked(iterable, chunk_size):
    """Split an iterable into chunks of a given size."""
    it = iter(iterable)
    return iter(lambda: list(islice(it, chunk_size)), [])

def clean_data(df):
    # dropped Unnamed: 0 column
    df.drop(columns=['Unnamed: 0'], inplace=True)
    filtered_df = df.loc[(df['response'] > 95) | (df['response'] < 5)]
    for id in df['person_id'].unique():
        if len(filtered_df[filtered_df['person_id'] == id]) / len(
                df[df['person_id'] == id]) > 0.8:
            df.drop(df[df['person_id'] == id].index, inplace=True)
    # only keep columns with has_intensifier = yes
    df = df[df['has intensifier?'] == 'yes']
    return df[df['has intensifier?'] == 'yes']


for i in range(len(dataframes)):
    dataframes[i] = clean_data(dataframes[i])

politeness = pd.concat([dataframes[0], dataframes[1]])
dialogue = pd.concat([dataframes[2], dataframes[3]])
politeness.groupby(['intensifier', 'predicate']).agg({
    'difference':
    'mean',
    'Z-Score Difference':
    'mean'
}).reset_index()
# create key with intensifier and predicate as a tuple and value as Z-Score Difference
# compute U_soc
U_soc = politeness.set_index(['intensifier',
                              'predicate'])['Z-Score Difference'].to_dict()
# compute dialogue
utterences = list(U_soc.keys())
predicates = list(set(w[1] for w in utterences))
intensifiers = list(set(w[0] for w in utterences))
states = np.arange(-5.7, 4.5, 0.1)
possible_phi = np.arange(0, 1, 0.1)
possible_alpha = np.arange(0.6, 2.4, 0.3)
Alpha = min(possible_alpha)
Phi = min(possible_phi)
S = min(states)
theta_to_test = list(np.arange(-3, -1, 0.4)) + list(np.arange(-1, 1, 0.1)) + list(np.arange(1, 2, 0.2))

# Compute softmax row-wise
def softmax_rowwise(arr):
    # Subtract the maximum value from each row for numerical stability
    exp_vals = np.exp(arr - np.max(arr, axis=1, keepdims=True))
    # Normalize each row
    softmax_vals = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    return softmax_vals


def gaussian(s):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-((s - 0)**2) / 2)


P_state = np.array([gaussian(s) for s in states])
measured_values = dict()
for w in intensifiers:
    for pred in predicates:
        measured_values[(w, pred)] = dialogue[(
            (dialogue['intensifier'] == w) &
            (dialogue['predicate'] == pred))]['Z-Score Difference'].values
epsilon = 0.01
theta_U_inf = dict()
P_l1 = dict()
infty = 10000000
count = 0
# define theta_U_inf
for t in theta_to_test:
    arr = np.array([
        epsilon * P_state[i] if states[i] <= t else P_state[i]
        for i in range(len(states))
    ])
    theta_U_inf[t] = np.log(arr) - np.log(np.sum(arr))





def main():
    try:
        shutil.rmtree('results')
        shutil.rmtree('slurm-scripts')
    except:
        pass

    os.mkdir('results')

    block_size = 20
    parameter_blocks = list(chunked(product(theta_to_test, repeat=5), block_size))[:10]


    for n, param in enumerate(parameter_blocks):

        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmp_file:

            json.dump(param, tmp_file)
            data_file_path = tmp_file.name

            job = slurmpy.Slurm(str(n), {"time": "02:00:00", "partition": "normal"})
            results = job.run(f"python func_to_iterate_over.py --data-file {data_file_path}")
            print(results)

    # with open('results.pkl', 'wb') as f:
    #     pickle.dump(list(results), f)
    # sys.exit()
    # for r in results:
    print(results)
    best_thetas = np.zeros((len(possible_phi), len(possible_alpha)), dtype=object)
    # for i in range(len(possible_phi)):
    #     for j in range(len(possible_alpha)):
    #         print((i, j))
    #         # print(results)
    #         for idx, log_l_arr in enumerate(results):
    #             # print(log_l_arr)
    #             print('--------------------------------------------------------------------------------')
    #             print((i, j))
    #             # print(log_l_arr[i])
    #             # print(log_l_arr[i][j])
    #             # print(log_l_arr[i][j][0])
    #         arr = np.array([log_l_arr[i][j][0] for log_l_arr in results])
    #         best_thetas[i][j] = (np.max(arr), results[np.argmax(arr)][1])

    # print(best_thetas)


if __name__ == '__main__':
    main()
