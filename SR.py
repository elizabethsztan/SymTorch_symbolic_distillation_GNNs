import sys
sys.path.append("/Users/liz/PhD/SymTorch_project/SymTorch/src")


import symtorch as SymTorch
from symtorch.mlp_sr import MLP_SR
from plot_linear_rep import get_message_features
from model import load_model, get_edge_index
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils import load_data
import numpy as np
import argparse
import os
import pickle
import torch
import json


print("SymTorch imported successfully!")

niterations = 6000
num_points = 3000
_, _, test_data = load_data('spring')
save_path = f'pysr_objects/spring/{niterations}'
os.makedirs(save_path, exist_ok=True)

model = load_model(dataset_name='spring', model_type='bottleneck', num_epoch=100)

model.edge_model = MLP_SR(model.edge_model, mlp_name = 'bottleneck')

X_test, y_test = test_data
edge_index = get_edge_index(X_test.shape[1])

dataset = DataLoader(
    [Data(x=X_test[i], edge_index=edge_index, y=y_test[i]) for i in range(len(X_test))],
    batch_size=len(X_test),
    shuffle=False)

all_inputs = []

for datapoint in dataset:
    source_nodes = datapoint.x[datapoint.edge_index[0]]
    target_nodes = datapoint.x[datapoint.edge_index[1]]
    x = torch.cat((source_nodes, target_nodes), dim=1)
    all_inputs.append(x)
all_inputs = torch.cat(all_inputs, dim=0)

# Define variable transformations for physics-relevant quantities
# Input format: [x1, y1, nu_x1, nu_y1, q1, m1, x2, y2, nu_x2, nu_y2, q2, m2]
variable_transforms = [
    lambda x: x[:, 0] - x[:, 6],      # dx = x1 - x2
    lambda x: x[:, 1] - x[:, 7],      # dy = y1 - y2  
    lambda x: torch.sqrt((x[:, 0] - x[:, 6])**2 + (x[:, 1] - x[:, 7])**2),  # r = sqrt(dx^2 + dy^2)
    lambda x: x[:, 4],                # q1 (charge 1)
    lambda x: x[:, 10],               # q2 (charge 2)
    lambda x: x[:, 5],                # m1 (mass 1) 
    lambda x: x[:, 11]                # m2 (mass 2)
]

variable_names = ['dx', 'dy', 'r', 'q1', 'q2', 'm1', 'm2']

print(f"Input tensor shape: {all_inputs.shape}")
print("Running symbolic regression...")

np.random.seed(290402)
idx = np.random.choice(len(all_inputs), size=num_points, replace=False)
inputs_subset = all_inputs[idx]

# Run symbolic regression
result = model.edge_model.interpret(
    inputs_subset,
    variable_transforms=variable_transforms,
    variable_names=variable_names, 
    niterations=niterations,
    save_path=save_path, 
    parsimony = 0.05, 
    complexity_of_constants = 1,
    maxsize = 23
)

print("Symbolic regression completed!")
print(f"Results saved to: {save_path}")

