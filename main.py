import pandas as pd
import os
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dataset_name = ''
directory = '' + dataset_name

data_list = []
description = {}

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path) and filename.endswith('.json'):
        with open(file_path, 'r') as file:
            data = json.load(file)
            if filename == "description.json":
                description = data
            else:
                data_list.append(data)

df = pd.DataFrame(data_list)

print(description)
print(df)

output_directory = '' + dataset_name
os.makedirs(output_directory, exist_ok=True)

probabilities = description['probabilities']
print("probabilities: ", probabilities)

if 'num_of_nodes' in df.columns and 'average_iterations' in df.columns:
    for p_add, p_remove in probabilities:
        filtered_df = df[(df['p_add'] == p_add) & (df['p_remove'] == p_remove)]

        plt.figure(figsize=(10, 8))
        plt.plot(filtered_df['num_of_nodes'], filtered_df['average_iterations'], marker='o', linestyle='', label=f'P={p_add}_{p_remove}')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Average Number of Iterations')
        plt.title(f'Graph for P={p_add}_{p_remove}')
        plt.legend()
        plt.grid(True)

        filename = f'graph_p_{p_add}_{p_remove}.png'
        plt.savefig(os.path.join(output_directory, filename))
        plt.close()

if 'num_of_nodes' in df.columns and 'average_iterations' in df.columns:
    plt.figure(figsize=(10, 8))
    for p_add, p_remove in probabilities:
        filtered_df = df[(df['p_add'] == p_add) & (df['p_remove'] == p_remove)]

        plt.plot(filtered_df['num_of_nodes'], filtered_df['average_iterations'], marker='o', linestyle='',
                 label=f'P={p_add}_{p_remove}')

    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Number of Iterations')
    plt.title('Graphs for All Probabilities')
    plt.legend()
    plt.grid(True)

    filename = 'combined_graph.png'
    plt.savefig(os.path.join(output_directory, filename))
    plt.show()

import numpy as np
from scipy.optimize import curve_fit

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_exponential_curve(x_data, y_data):
    params, _ = curve_fit(exponential_func, x_data, y_data)
    return params

def plot_exponential_fit(x_data, y_data, params, p_add, p_remove):
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = exponential_func(x_fit, *params)

    plt.figure(figsize=(10, 8))
    plt.plot(x_data, y_data, 'o', label='Data')
    plt.plot(x_fit, y_fit, '-', label='Exponential Fit')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Number of Iterations')
    plt.title(f'Exponential Fit for P={p_add}_{p_remove}')
    plt.legend()
    plt.grid(True)

    filename = f'exponential_fit_p_{p_add}_{p_remove}.png'
    plt.savefig(os.path.join(output_directory, filename))
    plt.close()

if 'num_of_nodes' in df.columns and 'average_iterations' in df.columns:
    for p_add, p_remove in probabilities:
        filtered_df = df[(df['p_add'] == p_add) & (df['p_remove'] == p_remove)]

        x_data = filtered_df['num_of_nodes']
        y_data = filtered_df['average_iterations']

        params = fit_exponential_curve(x_data, y_data)

        plot_exponential_fit(x_data, y_data, params, p_add, p_remove)

        params_filename = f'params_p_{p_add}_{p_remove}.txt'
        with open(os.path.join(output_directory, params_filename), 'w') as f:
            f.write(f'Parameters for P={p_add}_{p_remove}: a={params[0]}, b={params[1]}, c={params[2]}\n')
            f.write(f'Covariance: {np.cov(x_data, y_data)}\n')
        print(f'Parameters for P={p_add}_{p_remove}: a={params[0]}, b={params[1]}, c={params[2]}')
        print(f'Covariance: {np.cov(x_data, y_data)}')
        print(f'Fitted graph saved as {params_filename}')