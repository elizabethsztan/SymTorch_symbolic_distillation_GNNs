import symtorch 
from symtorch import MLP_SR
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
    """
    Perform symbolic regression analysis on trained GNN models.
    
    This function loads a trained GNN model and applies symbolic regression to
    discover interpretable mathematical expressions that describe the learned
    message representations. The analysis focuses on the most important message
    dimensions (based on variance) and uses physics-relevant variable transforms.
    
    The workflow:
    1. Load trained model and test data
    2. Extract edge features from all node pairs
    3. Apply model to get message representations
    4. Identify most important message dimensions
    5. Run symbolic regression with physics-aware variable transforms
    6. Save results for analysis
    
    Command line arguments:
        --dataset_name: Dataset type ('charge', 'r1', 'r2', 'spring')
        --model_type: Model variant ('standard', 'bottleneck', 'KL', 'L1', 'pruning')
    
    The function uses hardcoded parameters:
        - num_epoch: 100 (training epochs for model loading)
        - niterations: 7000 (symbolic regression iterations)
        - num_points: 5000 (data points for regression)
    
    Output:
        Saves symbolic regression results to pysr_objects/{dataset}/{niterations}/
        
    Note:
        Different model types require different handling:
        - standard/L1: Use top 2 message dimensions by variance
        - KL: Extract mean components from variational output
        - bottleneck/pruning: Use all available dimensions
    """
    parser = argparse.ArgumentParser(description='Run symbolic regression on trained GNN models')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Dataset name: charge, r1, r2, or spring')
    parser.add_argument('--model_type', type=str, required=True,
                       help='Model type: standard, bottleneck, KL, L1, or pruning')
    
    args = parser.parse_args()

    # Extract arguments and set hyperparameters
    dataset = args.dataset_name
    model_type = args.model_type
    num_epoch = 100      # Model training epochs
    niterations = 7000   # Symbolic regression iterations
    num_points = 5000    # Data points for regression analysis

    # Load test data and create output directory
    _, _, test_data = load_data(f'{dataset}')
    save_path = f'pysr_objects/{dataset}/{niterations}'
    os.makedirs(save_path, exist_ok=True)

    # Load trained model
    model = load_model(dataset_name=dataset, model_type=model_type, num_epoch=num_epoch)

    # Wrap edge model with SymTorch for symbolic regression (except pruning models)
    if model_type != 'pruning':
        model.edge_model = MLP_SR(model.edge_model, mlp_name=model_type)

    # Prepare test data for edge feature extraction
    X_test, y_test = test_data
    edge_index = get_edge_index(X_test.shape[1])  # Full connectivity graph

    # Create DataLoader for batch processing
    dataset = DataLoader(
        [Data(x=X_test[i], edge_index=edge_index, y=y_test[i]) for i in range(len(X_test))],
        batch_size=len(X_test),
        shuffle=False)

    # Extract edge features by concatenating source and target node features
    all_inputs = []
    for datapoint in dataset:
        source_nodes = datapoint.x[datapoint.edge_index[0]]  # Source node features
        target_nodes = datapoint.x[datapoint.edge_index[1]]  # Target node features
        x = torch.cat((source_nodes, target_nodes), dim=1)   # Concatenate for edge features
        all_inputs.append(x)
    all_inputs = torch.cat(all_inputs, dim=0)  # Shape: [num_edges_total, 2*node_dim]

    # Define physics-relevant variable transformations
    # Input format: [x1, y1, vx1, vy1, q1, m1, x2, y2, vx2, vy2, q2, m2]
    variable_transforms = [
        lambda x: x[:, 0] - x[:, 6],      # dx = x1 - x2 (relative position x)
        lambda x: x[:, 1] - x[:, 7],      # dy = y1 - y2 (relative position y)
        lambda x: torch.sqrt((x[:, 0] - x[:, 6])**2 + (x[:, 1] - x[:, 7])**2) + 1e-2,  # r = distance + small epsilon
        lambda x: x[:, 4],                # q1 (charge of particle 1)
        lambda x: x[:, 10],               # q2 (charge of particle 2)
        lambda x: x[:, 5],                # m1 (mass of particle 1)
        lambda x: x[:, 11]                # m2 (mass of particle 2)
    ]
    variable_names = ['dx', 'dy', 'r', 'q_one', 'q_two', 'm_one', 'm_two']

    print(f"Input tensor shape: {all_inputs.shape}")
    print("Running symbolic regression...")

    # Identify most important message dimensions for regression
    if model_type == 'standard' or model_type == 'L1':
        # For standard models, use all message dimensions
        important_dims_info = model.edge_model.get_importance(all_inputs)
        important_dims = important_dims_info['importance']

        dim0 = important_dims[0]
        dim1 = important_dims[1]
    
    elif model_type == 'KL':
        # For KL models, extract mean components from variational output
        raw_outputs = model.edge_model(all_inputs).detach().numpy()
        messages = raw_outputs[:, 0::2]  # Extract mean components (every other element)
        logvars = raw_outputs[:, 1::2]

        KL_div =  (np.exp(logvars) + messages**2 - logvars)/2
        KL_mean = KL_div.mean(axis=0)
        most_important = np.argsort(KL_mean)[-2:]
        dim0 = most_important[1] * 2  # Most important dimension index (even)
        dim1 = most_important[0] * 2  # Second most important dimension index (even)


    # Sample subset of data for symbolic regression (for computational efficiency)
    np.random.seed(290402)  # Fixed seed for reproducibility
    idx = np.random.choice(len(all_inputs), size=num_points, replace=False)
    inputs_subset = all_inputs[idx]

    # For models with multiple dimensions, analyse top 2 most important dimensions
    sr_params = {'niterations':niterations,
                    'parsimony':0.05,                    # Simplicity preference
                    'complexity_of_constants':1,         # Penalty for complex constants
                    'maxsize': 25,                        # Max expression size
                    'elementwise_loss':"loss(prediction, target) = abs(prediction - target)",
                    'batching':True,
                    'unary_operators':["inv(x) = 1/x", "exp", "log"],
                    'constraints':{'exp': (1), 'log': (1)},
                    'complexity_of_operators':{'exp': 3, 'sin': 3, 'log': 3}}
    
    fit_params = {'variable_names':variable_names}

    # Run symbolic regression on selected message dimensions
    if model_type == 'standard' or model_type == 'L1' or model_type == 'KL':

        
        # First dimension (most important)
        result = model.edge_model.distill(
            inputs_subset,
            output_dim=dim0,
            variable_transforms=variable_transforms,
            sr_params = sr_params,
            fit_params = fit_params, 
            save_path=save_path)

        # Second dimension (second most important)
        result = model.edge_model.distill(
            inputs_subset,
            output_dim=dim1,
            variable_transforms=variable_transforms,
            sr_params = sr_params,
            fit_params = fit_params, 
            save_path=save_path
        )

    else:
        # For bottleneck and pruning models, analyse all available dimensions
        result = model.edge_model.distill(
            inputs_subset,
            variable_transforms=variable_transforms,
            sr_params = sr_params,
            fit_params = fit_params, 
            save_path=save_path
        )

    print("Symbolic regression completed!")
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
