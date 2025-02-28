#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy
import os
from os.path import join


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


# In[2]:


from memo import memo
import jax
import jax.numpy as np
from enum import IntEnum, auto


# Define constants

# In[3]:


epsilon = 0.01
infty = 10000000
utterences =list(U_soc_data.keys())
state_values = np.arange(-2.8,2.8,0.28)
S = np.arange(0,20,1)


# Define params to iterate

# In[4]:

from numpy import arange
theta_to_test = arange(0,20,1)
possible_inf_terms = arange(0,5,0.5)
possible_soc_terms = arange(0,5,0.5)
costs = arange(0,5,0.5)


# Grid search (code from demo-rsa.py)

# In[5]:


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


# In[6]:


# Generate the mapping dictionary
U_soc_key_map = {w: (' '.join(w.name.split('_')[:-1]), w.name.split('_')[-1]) for w in W}
U_soc_array = np.array([U_soc_data[U_soc_key_map[W(i)]] for i in range(len(W))])
UK_U_soc_array = np.array([UK_U_soc_data[U_soc_key_map[W(i)]] for i in range(len(W))])
US_U_soc_array = np.array([US_U_soc_data[U_soc_key_map[W(i)]] for i in range(len(W))])


# Construct measured_values which is an array of 42 arrays where the i'th entry is the values people reported for the i'th utterance

# In[7]:


possible_literal_semantics = np.array(
    [
        np.concatenate([np.repeat(np.array([epsilon]),i),np.repeat(np.array([1]),20-i)])
        for i in range(20)
    ]
)
@jax.jit
def state_prior(s):
    return np.float32(np.exp(-state_values[s]**2/2))  # uninformative state prior doesn't matter that it doesn't add up to 1
@jax.jit
def U_soc(soc):
    return U_soc_array[soc]
@jax.jit
def UK_U_soc(soc):
    return UK_U_soc_array[soc]

@jax.jit
def US_U_soc(soc):
    return US_U_soc_array[soc]

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
@memo
def UKL1[s: S, w: W](inf_term, soc_term, cost,t0,t1,t2,t3,t4,t5):
    listener: thinks[
        speaker: given(s in S, wpp=state_prior(s)),
        speaker: chooses(w in W, wpp=
            imagine[
                listener: knows(w),
                listener: chooses(s in S, wpp=L(w, s,t0,t1,t2,t3,t4,t5)) ,
                exp(inf_term * log(Pr[listener.s == s]) + 
                soc_term * UK_U_soc(w) - # U_soc = listener's EU
                cost*is_costly(w)) # U_inf = listener's surprisal       
            ]
        )
    ]
    listener: observes[speaker.w] is w
    listener: chooses(s in S, wpp=Pr[speaker.s == s])
    return Pr[listener.s == s]
@memo
def USL1[s: S, w: W](inf_term, soc_term, cost,t0,t1,t2,t3,t4,t5):
    listener: thinks[
        speaker: given(s in S, wpp=state_prior(s)),
        speaker: chooses(w in W, wpp=
            imagine[
                listener: knows(w),
                listener: chooses(s in S, wpp=L(w, s,t0,t1,t2,t3,t4,t5)) ,
                exp(inf_term * log(Pr[listener.s == s]) + 
                soc_term * US_U_soc(w) - # U_soc = listener's EU
                cost*is_costly(w)) # U_inf = listener's surprisal       
            ]
        )
    ]
    listener: observes[speaker.w] is w
    listener: chooses(s in S, wpp=Pr[speaker.s == s])
    return Pr[listener.s == s]


UK_measured_values = []
for w in W:
    intensifier, predicate = U_soc_key_map[w]
    raw_values = UK_dialogue[((UK_dialogue['intensifier'] == intensifier) & (UK_dialogue['predicate'] == predicate))]['predicate Z-score'].values
    # measured_values.append(np.array([int(r/0.28)+10 for r in raw_values]))
    z = [int(r/0.28)+10 for r in raw_values]
    x = [0]*len(S)
    if intensifier != 'none': # make measured_values all zero if intensifier is none
        for i in z:
            x[i] += 1
    UK_measured_values.append(x)
UK_measured_values = np.array(UK_measured_values)
# Create a list of JAX arrays
US_measured_values = []
for w in W:
    intensifier, predicate = U_soc_key_map[w]
    raw_values = US_dialogue[((US_dialogue['intensifier'] == intensifier) & (US_dialogue['predicate'] == predicate))]['predicate Z-score'].values
    # measured_values.append(np.array([int(r/0.28)+10 for r in raw_values]))
    z = [int(r/0.28)+10 for r in raw_values]
    x = [0]*len(S)
    if intensifier != 'none': # make measured_values all zero if intensifier is none
        for i in z:
            x[i] += 1
    US_measured_values.append(x)
US_measured_values = np.array(US_measured_values)

def compute_logloss(*params):
    thetas = params[:6]
    cost = params[6]
    inf_term = params[7]
    soc_term = params[8]
    # compute fit e.g. log_likelihood
    P_l1 = L1(inf_term=inf_term, soc_term=soc_term, cost=cost, t0 = thetas[0],t1=thetas[1], t2= thetas[2], t3 = thetas[3], t4 = thetas[4], t5 = thetas[5]) # note this should be P(s|w)
    return np.sum(np.log(P_l1)*UK_measured_values.T),np.sum(np.log(P_l1)*US_measured_values.T)
def UK_soc_logloss(*params):
    thetas = params[:6]
    cost = params[6]
    inf_term = params[7]
    soc_term = params[8]
    # compute fit e.g. log_likelihood
    P_l1 = UKL1(inf_term=inf_term, soc_term=soc_term, cost=cost, t0 = thetas[0],t1=thetas[1], t2= thetas[2], t3 = thetas[3], t4 = thetas[4], t5 = thetas[5]) # note this should be P(s|w)
    return np.sum(np.log(P_l1)*UK_measured_values.T)
def US_soc_logloss(*params):
    thetas = params[:6]
    cost = params[6]
    inf_term = params[7]
    soc_term = params[8]
    # compute fit e.g. log_likelihood
    P_l1 = USL1(inf_term=inf_term, soc_term=soc_term, cost=cost, t0 = thetas[0],t1=thetas[1], t2= thetas[2], t3 = thetas[3], t4 = thetas[4], t5 = thetas[5]) # note this should be P(s|w)
    return np.sum(np.log(P_l1)*US_measured_values.T)

total_param_combos = len(costs)*len(possible_soc_terms)*len(possible_inf_terms)*len(theta_to_test)**2
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
# fourth_thetas = np.repeat(np.tile(theta_to_test, in_repeat),out_repeat)
# out_repeat = out_repeat*len(theta_to_test)
# in_repeat = in_repeat//len(theta_to_test)
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

# Define batch size (adjust based on memory capacity)
BATCH_SIZE = 10000  # Modify this based on your GPU memory availability
print(f"BATCH_SIZE = {BATCH_SIZE}")

# Function to batch the computation
"""
def batch_compute(func, *params, batch_size=BATCH_SIZE):
    # Identify which parameters are arrays (batched) and which are scalars
    scalar_params = params[:4]  # First four parameters are scalars
    batched_params = params[4:]  # Remaining parameters are arrays

    # Ensure at least one parameter is an array to determine batch size
    if not batched_params or not isinstance(batched_params[0], np.ndarray):
        raise ValueError("Batched parameters are missing or incorrectly formatted.")

    total_params = len(batched_params[0])
    results = []

    for start in range(0, total_params, batch_size):
        end = min(start + batch_size, total_params)
        batch_params = [p[start:end] for p in batched_params]

        # Apply function with scalars passed normally and arrays batched
        batch_result = jax.vmap(func, in_axes=(None,)*4 + (0,)*5)(*scalar_params, *batch_params).block_until_ready()
        results.append(batch_result)

    return np.concatenate(results)
"""
def batch_compute(func, *params, batch_size=BATCH_SIZE):
    """
    Handles batch computation while keeping scalar arguments separate.
    Supports functions that return either a single output or a tuple of outputs.
    """
    scalar_params = params[:4]  # First four parameters are scalars
    batched_params = params[4:]  # Remaining parameters are arrays

    if not batched_params or not isinstance(batched_params[0], np.ndarray):
        raise ValueError("Batched parameters are missing or incorrectly formatted.")

    total_params = len(batched_params[0])
    results = []
    results_tuple = None

    for start in range(0, total_params, batch_size):
        end = min(start + batch_size, total_params)
        batch_params = [p[start:end] for p in batched_params]

        # Apply function and check if the output is a tuple
        batch_result = jax.vmap(func, in_axes=(None,)*4 + (0,)*5)(*scalar_params, *batch_params)
        
        if isinstance(batch_result, tuple):
            if results_tuple is None:
                results_tuple = [[] for _ in range(len(batch_result))]
            for i, res in enumerate(batch_result):
                res.block_until_ready()
                results_tuple[i].append(np.array(res))
        else:
            batch_result.block_until_ready()
            results.append(np.array(batch_result))

    if results_tuple is not None:
        return tuple(np.concatenate(res) for res in results_tuple)
    return np.concatenate(results)


try:
    os.mkdir('UK_US_standard')
    os.mkdir('US_specific')
    os.mkdir('UK_specific')
except:
    pass

"""
# Compute UK-specific results in batches
for t1 in theta_to_test:
    for t2 in theta_to_test:
        all_output = []
        for t3 in theta_to_test:
            for t4 in theta_to_test:
                output = batch_compute(UK_soc_logloss, t1, t2, t3, t4, fifth_thetas, sixth_thetas, seventh_costs, eighth_infs, ninth_socs)
                output = numpy.array(jax.device_get(output))
                all_output.append(output)
            print(t1, t2, t3)
        np.save(f'UK_specific_soc_feb21_{t1}_{t2}.npy', numpy.array(all_output).flatten())
        del all_output

# Compute US-specific results in batches
"""

"""
for t1 in theta_to_test:
    for t2 in theta_to_test:
        all_output = []
        for t3 in theta_to_test:
            for t4 in theta_to_test:
                output = batch_compute(US_soc_logloss, t1, t2, t3, t4, fifth_thetas, sixth_thetas, seventh_costs, eighth_infs, ninth_socs)
                output = numpy.array(jax.device_get(output))
                all_output.append(output)
            print(t1, t2, t3)
        np.save(join('US_specific', f'US_specific_soc_feb21_{t1}_{t2}.npy'), numpy.array(all_output).flatten())
        del all_output

"""
# Compute standard UK/US results in batches
for t1 in theta_to_test:
    for t2 in theta_to_test:
        UK_all_output = []
        US_all_output = []
        for t3 in theta_to_test:
            for t4 in theta_to_test:
                UK_output, US_output = batch_compute(compute_logloss, t1, t2, t3, t4, fifth_thetas, sixth_thetas, seventh_costs, eighth_infs, ninth_socs)
                UK_output = numpy.array(jax.device_get(UK_output))
                US_output = numpy.array(jax.device_get(US_output))
                UK_all_output.append(UK_output)
                US_all_output.append(US_output)
            print(t1, t2, t3)
        np.save(join('UK_US_standard', f'UK_standard_feb21_{t1}_{t2}.npy'), numpy.array(UK_all_output).flatten())
        del UK_all_output
        np.save(join('UK_US_standard', f'US_standard_feb21_{t1}_{t2}.npy'), numpy.array(US_all_output).flatten())
        del US_all_output
