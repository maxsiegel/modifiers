import os

N = 15

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N}"
os.environ["OMP_NUM_THREADS"] = f"{N}"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from enum import IntEnum, auto

import jax
import jax.numpy as np
from tqdm import tqdm
from memo import memo

import pandas as pd

# read in files
# Read UK_df.csv as pandas dataframe
original_UK_dialogue = pd.read_csv('UK_df.csv')
original_UK_politeness = pd.read_csv('UK_direct_df.csv')
original_UK_narrator = pd.read_csv('UK_narrator_df.csv')
original_US_dialogue = pd.read_csv('US_df.csv')
original_US_politeness = pd.read_csv('US_direct_df.csv')
original_US_narrator = pd.read_csv('US_narrator_df.csv')
dataframes = [original_UK_dialogue, original_UK_politeness, original_UK_narrator, original_US_dialogue, original_US_politeness, original_US_narrator]

def elim_outliers(df):
    # dropped Unnamed: 0 column
    df.drop(columns=['Unnamed: 0'], inplace=True)
    filtered_df = df.loc[(df['response'] > 95) | (df['response'] < 5)]
    for id in df['person_id'].unique():
        if len(filtered_df[filtered_df['person_id'] == id])/len(df[df['person_id'] == id])>0.8:
            df.drop(df[df['person_id'] == id].index, inplace=True)
    df['predicate Z-score'] = df.groupby(['person_id','predicate'])['response'].transform(lambda x: (x - x.mean()) / x.std())
    # if has_intensifier = no then change 'intensifier' to 'none'
    df.loc[df['has intensifier?'] == 'no', 'intensifier'] = 'none'
    return df
for i in range(len(dataframes)):
    dataframes[i] = elim_outliers(dataframes[i])
dialogue = pd.concat([dataframes[0], dataframes[3]])
politeness = pd.concat([dataframes[1], dataframes[4]])
UK_dialogue = dataframes[0]
US_dialogue = dataframes[3]
UK_politeness = dataframes[1]
US_politeness = dataframes[4]

# end of reading in data
#-------------------------------------------------------------------------------

# compute U_soc (social Utility)
U_soc_data = politeness.groupby(['intensifier','predicate'])['predicate Z-score'].mean().to_dict()
UK_U_soc_data = UK_politeness.groupby(['intensifier','predicate'])['predicate Z-score'].mean().to_dict()
US_U_soc_data = US_politeness.groupby(['intensifier','predicate'])['predicate Z-score'].mean().to_dict()





# Define constants



epsilon = 0.01
infty = 10000000
utterences =list(U_soc_data.keys())
state_values = np.linspace(-2.8,2.8,20)
S = np.arange(0,20,1)


# Define params to iterate



costs = np.linspace(0,4,4)
possible_soc_terms = np.linspace(0,3,5)
possible_inf_terms = np.linspace(0,3,5)
theta_to_test = np.arange(0,20,2)


# Grid search (code from demo-rsa.py)



class W(IntEnum):  # utterance space
    # borings
    none_boring = auto(0)
    slightly_boring = auto()
    kind_of_boring = auto()
    quite_boring = auto()
    very_boring = auto()
    extremely_boring = auto()
    # concerneds
    none_concerned = auto()
    slightly_concerned = auto()
    kind_of_concerned = auto()
    quite_concerned = auto()
    very_concerned = auto()
    extremely_concerned = auto()
    # difficults
    none_difficult = auto()
    slightly_difficult = auto()
    kind_of_difficult = auto()
    quite_difficult = auto()
    very_difficult = auto()
    extremely_difficult = auto()
    # exhausteds
    none_exhausted = auto()
    slightly_exhausted = auto()
    kind_of_exhausted = auto()
    quite_exhausted = auto()
    very_exhausted = auto()
    extremely_exhausted = auto()
    # helpfuls
    none_helpful = auto()
    slightly_helpful = auto()
    kind_of_helpful = auto()
    quite_helpful = auto()
    very_helpful = auto()
    extremely_helpful = auto()
    # impressives
    none_impressive = auto()
    slightly_impressive = auto()
    kind_of_impressive = auto()
    quite_impressive = auto()
    very_impressive = auto()
    extremely_impressive = auto()
    # understandables
    none_understandable = auto()
    slightly_understandable = auto()
    kind_of_understandable = auto()
    quite_understandable = auto()
    very_understandable = auto()
    extremely_understandable = auto()




# Generate the mapping dictionary
U_soc_key_map = {w: (' '.join(w.name.split('_')[:-1]), w.name.split('_')[-1]) for w in W}




U_soc_array = np.array([U_soc_data[U_soc_key_map[W(i)]] for i in range(len(W))])


# Construct measured_values which is an array of 42 arrays where the i'th entry is the values people reported for the i'th utterance



# Create a list of JAX arrays
measured_values = []
for w in W:
    intensifier, predicate = U_soc_key_map[w]
    raw_values = dialogue[((dialogue['intensifier'] == intensifier) & (dialogue['predicate'] == predicate))]['predicate Z-score'].values
    measured_values.append(np.array([int(r/0.28)+10 for r in raw_values]))




possible_literal_semantics = np.array(
    [
        np.concatenate([np.repeat(np.array([epsilon]),i),np.repeat(np.array([1]),20-i)])
        for i in range(20)
    ]
)
@jax.jit
def state_prior(s):
    return np.float32(np.exp(-(state_values[s]**2)/2))  # uninformative state prior doesn't matter that it doesn't add up to 1
@jax.jit
def U_soc(soc):
    return U_soc_array[soc]
@jax.jit
def is_costly(w):
    arr = [0, 1, 1, 1, 1, 1]*7
    return np.array(arr)[w]
@jax.jit
def L(w, s,t0,t1,t2,t3,t4,t5):  # literal likelihood L(w | s)
    intensifier_semantics = np.array([  # "hard semantics"
        possible_literal_semantics[t0],  # none
        possible_literal_semantics[t1],  # slightly
        possible_literal_semantics[t2],  # kind of
        possible_literal_semantics[t3],  # quite
        possible_literal_semantics[t4],  # very
        possible_literal_semantics[t5],  # extremely
    ])
    return np.tile(intensifier_semantics, (7, 1))[w, s]
@memo
def L1[s: S, w: W](inf_term, soc_term, cost,t0,t1,t2,t3,t4,t5):
    listener: thinks[
        speaker: given(s in S, wpp=state_prior(s)),
        speaker: chooses(w in W, wpp=
            imagine[
                listener: knows(w),
                listener: chooses(s in S, wpp=L(w, s,t0,t1,t2,t3,t4,t5)) ,
                exp(inf_term * log(Pr[listener.s == s]) +
                soc_term * U_soc(w) - # U_soc = listener's EU
                cost*is_costly(w)) # U_inf = listener's surprisal
            ]
        )
    ]
    listener: observes[speaker.w] is w
    listener: chooses(s in S, wpp=Pr[speaker.s == s])
    return Pr[listener.s == s]





# Create a list of JAX arrays
measured_values = []
for w in W:
    intensifier, predicate = U_soc_key_map[w]
    raw_values = dialogue[((dialogue['intensifier'] == intensifier) & (dialogue['predicate'] == predicate))]['predicate Z-score'].values
    # measured_values.append(np.array([int(r/0.28)+10 for r in raw_values]))
    z = [int(r/0.28)+10 for r in raw_values]
    x = [0]*len(S)
    for i in z:
        x[i] += 1
    measured_values.append(x)
measured_values = np.array(measured_values)




# def compute_logloss(*params):
#     thetas = params[:6]
#     cost = params[6]
#     inf_term = params[7]
#     soc_term = params[8]
#     # compute fit e.g. log_likelihood
#     P_l1 = L1(inf_term=inf_term, soc_term=soc_term, cost=cost,thetas=thetas) # note this should be P(s|w)
#     return np.sum(np.log(P_l1)*measured_values.T)
# Function to compute log-loss
@jax.jit
def compute_logloss(params):
    t0, t1, t2, t3, t4, t5 = params[:6]
    t0 = np.int64(t0)
    t1 = np.int64(t1)
    t2 = np.int64(t2)
    t3 = np.int64(t3)
    t4 = np.int64(t4)
    t5 = np.int64(t5)
    cost = params[6]
    inf_term = params[7]
    soc_term = params[8]
    # compute fit e.g. log_likelihood
    P_l1 = L1(inf_term, soc_term, cost, t0, t1, t2, t3, t4, t5) # note this should be P(s|w)
    return np.sum(np.log(P_l1)*measured_values.T)



total_param_combos = len(costs)*len(possible_soc_terms)*len(possible_inf_terms)*len(theta_to_test)**3
in_repeat = total_param_combos//len(theta_to_test)
out_repeat = 1
# first_thetas = np.repeat(np.tile(theta_to_test, in_repeat),out_repeat)
# out_repeat = out_repeat*len(theta_to_test)
# in_repeat = in_repeat//len(theta_to_test)
# second_thetas = np.repeat(np.tile(theta_to_test, in_repeat),out_repeat)
# out_repeat = out_repeat*len(theta_to_test)
# in_repeat = in_repeat//len(theta_to_test)
# third_thetas = np.repeat(np.tile(theta_to_test, in_repeat),out_repeat)
# out_repeat = out_repeat*len(theta_to_test)
# in_repeat = in_repeat//len(theta_to_test)
fourth_thetas = np.repeat(np.tile(theta_to_test, in_repeat),out_repeat)
out_repeat = out_repeat*len(theta_to_test)
in_repeat = in_repeat//len(theta_to_test)
fifth_thetas = np.repeat(np.tile(theta_to_test, in_repeat),out_repeat)
out_repeat = out_repeat*len(theta_to_test)
in_repeat = in_repeat//len(theta_to_test)
sixth_thetas = np.repeat(np.tile(theta_to_test, in_repeat),out_repeat)
out_repeat = out_repeat*len(theta_to_test)
in_repeat = in_repeat//len(costs)
seventh_costs = np.repeat(np.tile(costs, in_repeat),out_repeat)
out_repeat = out_repeat*len(costs)
in_repeat = in_repeat//len(possible_inf_terms)
eighth_infs = np.repeat(np.tile(possible_inf_terms, in_repeat),out_repeat)
out_repeat = out_repeat*len(possible_inf_terms)
in_repeat = in_repeat//len(possible_soc_terms)
ninth_socs = np.repeat(np.tile(possible_soc_terms, in_repeat),out_repeat)




total_param_combos


num_devices = min(jax.device_count(), N)  # Use up to N cores

# Generate parameter grid correctly
param_grid = np.array(
    np.meshgrid(
        theta_to_test, theta_to_test, theta_to_test, theta_to_test, theta_to_test, theta_to_test,
        costs, possible_inf_terms, possible_soc_terms, indexing='ij')
).reshape(9, -1).T  # (num_params, num_combos) → Transpose for correct shape


# Reduce batch size to avoid OOM issues
batch_size = N * 10  # Adjust this if necessary

# Function to compute loss in parallel (Smaller Batches)
@jax.pmap
def compute_parallel(params_batch):
    return jax.vmap(compute_logloss)(params_batch)

# Compute in chunks with progress tracking
all_output = []
total_batches = (param_grid.shape[0] + batch_size - 1) // batch_size  # Number of batches

for i in tqdm(range(0, param_grid.shape[0], batch_size), desc="Processing Batches", unit="batch"):
    batch = param_grid[i : i + batch_size]

    # Pad last batch if necessary
    if batch.shape[0] % num_devices != 0:
        pad_size = num_devices - (batch.shape[0] % num_devices)
        pad_values = np.tile(batch[-1], (pad_size, 1))  # Duplicate last row to pad
        batch = np.vstack([batch, pad_values])

    batch = batch.reshape(num_devices, -1, batch.shape[1])  # Reshape for `pmap`

    # print(batch)
    output = compute_parallel(batch)
    all_output.append(jax.device_get(output).flatten())  # Free memory early

# Flatten and remove padding
all_output = np.concatenate(all_output)[: param_grid.shape[0]]
import pickle
with open('results.pkl', 'wb') as f:
    pickle.dump(all_output, f)
