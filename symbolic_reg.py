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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    # parser.add_argument('--num_points', type=int, default = 5_000)
    # parser.add_argument('--niterations', type=int, default = 1_000)
    # parser.add_argument("--num_epoch", type=str, default = 100)
    
    args = parser.parse_args()

    dataset = args.dataset_name
    model_type = args.model_type
    num_epoch = 100
    niterations = 7000
    num_points = 5000

    _, _, test_data = load_data(f'{dataset}')
    save_path = f'pysr_objects/{dataset}/{niterations}'
    os.makedirs(save_path, exist_ok=True)

    model = load_model(dataset_name=dataset, model_type=model_type, num_epoch=num_epoch)

    if model.type != 'pruning':
        model.edge_model = MLP_SR(model.edge_model, mlp_name = model_type)

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
        lambda x: torch.sqrt((x[:, 0] - x[:, 6])**2 + (x[:, 1] - x[:, 7])**2) +1e-2,  # r = sqrt(dx^2 + dy^2)
        lambda x: x[:, 4],                # q1 (charge 1)
        lambda x: x[:, 10],               # q2 (charge 2)
        lambda x: x[:, 5],                # m1 (mass 1) 
        lambda x: x[:, 11]                # m2 (mass 2)
    ]
    variable_names = ['dx', 'dy', 'r', 'q_one', 'q_two', 'm_one', 'm_two']

    print(f"Input tensor shape: {all_inputs.shape}")
    print("Running symbolic regression...")

    if model.type == 'standard' or model.type == 'L1':

        messages = model.edge_model(all_inputs).detach().numpy()
        msg_importance = messages.std(axis=0)
        dim0 = np.argsort(msg_importance)[-1]
        dim1 = np.argsort(msg_importance)[-2]
    
    elif model.type == 'KL':
        raw_outputs = model.edge_model(all_inputs).detach().numpy()
        messages = raw_outputs[:, 0::2]

        msg_importance = messages.std(axis=0)
        dim0 = np.argsort(msg_importance)[-1]
        dim1 = np.argsort(msg_importance)[-2]


    np.random.seed(290402)
    idx = np.random.choice(len(all_inputs), size=num_points, replace=False)
    inputs_subset = all_inputs[idx]

    if model.type == 'standard' or model.type == 'L1' or model.type == 'KL':

        # Run symbolic regression
        result = model.edge_model.interpret(
            inputs_subset,
            output_dim = dim0,
            variable_transforms=variable_transforms,
            variable_names=variable_names, 
            niterations=niterations,
            save_path=save_path, 
            parsimony = 0.05, 
            complexity_of_constants = 1,
            maxsize = 25, 
            elementwise_loss = "loss(prediction, target) = abs(prediction - target)",
            batching=True,
            unary_operators=[
            "inv(x) = 1/x",
            "exp",
            "log"
            ],
            constraints={'exp': (1), 'log': (1)}, 
            complexity_of_operators={'exp': 3, 'sin':3, 'log':3}
        )

        result = model.edge_model.interpret(
            inputs_subset,
            output_dim = dim1,
            variable_transforms=variable_transforms,
            variable_names=variable_names, 
            niterations=niterations,
            save_path=save_path, 
            parsimony = 0.05, 
            complexity_of_constants = 1,
            maxsize = 25, 
            elementwise_loss = "loss(prediction, target) = abs(prediction - target)",
            batching=True,
            unary_operators=[
            "inv(x) = 1/x",
            "exp",
            "log"
            ],
            constraints={'exp': (1), 'log': (1)}, 
            complexity_of_operators={'exp': 3, 'sin':3, 'log':3}
        )

    else:
        result = model.edge_model.interpret(
            inputs_subset,
            variable_transforms=variable_transforms,
            variable_names=variable_names, 
            niterations=niterations,
            save_path=save_path, 
            parsimony = 0.05, 
            complexity_of_constants = 1,
            maxsize = 25, 
            batching = True,
            unary_operators=[
                "inv(x) = 1/x",
                "exp",
                "log"
            ],
            constraints={'exp': (1), 'log': (1)}, 
            complexity_of_operators={'exp': 3, 'sin':3, 'log':3},
            elementwise_loss="loss(prediction, target) = abs(prediction - target)"
        )

    print("Symbolic regression completed!")
    print(f"Results saved to: {save_path}")

