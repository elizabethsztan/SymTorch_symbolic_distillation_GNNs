# This file is to run SR purely on the input data to check that the GNN is necessary to find the forces


from model import *
from utils import *
import pysr
from pysr import PySRRegressor


seed = 290402 

sim = 'r1'
niterations = 7000
idx = 0
num_points = 6000

if sim == 'spring':
    data_path = 'datasets/spring_n=4_dim=2_nt=1000_dt=0.01.pt'
    def force_func(X_train, idx=0):
        acc_x = np.zeros(X_train.shape[0])
        acc_y = np.zeros(X_train.shape[0])
        m0 = X_train[:, idx, -1]

        for j in range(4):
            if j == idx:
                continue
            dx = X_train[:, idx, 0] - X_train[:, j, 0]
            dy = X_train[:, idx, 1] - X_train[:, j, 1]
            r = np.sqrt(dx**2 + dy**2) + 1e-2

            acc_x += -2*(r-1)*(dx/r)/m0
            acc_y += -2*(r-1)*(dy/r)/m0

        return np.stack([acc_x, acc_y], axis=1)
    
elif sim == 'r1':
    data_path = 'datasets/r1_n=4_dim=2_nt=1000_dt=0.005.pt'
    def force_func(X_train, idx=0):

        acc_x = np.zeros(X_train.shape[0])
        acc_y = np.zeros(X_train.shape[0])
        # m0 = X_train[:, idx, -1]
        # q0 = X_train[:, idx, -2]

        for j in range(4):
            if j == idx:
                continue
            dx = X_train[:, idx, 0] - X_train[:, j, 0]
            dy = X_train[:, idx, 1] - X_train[:, j, 1]
            r = np.sqrt(dx**2 + dy**2) + 1e-2
            mj = X_train[:, j, -1]

            force_mag = -mj/r
            acc_x += force_mag*(dx/r)
            acc_y += force_mag*(dy/r)

        return np.stack([acc_x, acc_y], axis=1)
    
elif sim == 'r2':
    data_path = 'datasets/r2_n=4_dim=2_nt=1000_dt=0.001.pt'

    def force_func(X_train, idx=0):
        acc_x = np.zeros(X_train.shape[0])
        acc_y = np.zeros(X_train.shape[0])
        # m0 = X_train[:, idx, -1]
        # q0 = X_train[:, idx, -2]

        for j in range(4):
            if j == idx:
                continue
            dx = X_train[:, idx, 0] - X_train[:, j, 0]
            dy = X_train[:, idx, 1] - X_train[:, j, 1]
            r = np.sqrt(dx**2 + dy**2) + 1e-2
            mj = X_train[:, j, -1]

            force_mag = -mj/(r**2)
            acc_x += force_mag*(dx/r)
            acc_y += force_mag*(dy/r)

        return np.stack([acc_x, acc_y], axis=1)

elif sim == 'charge':
    data_path = 'datasets/charge_n=4_dim=2_nt=1000_dt=0.001.pt'
    def force_func(X_train, idx=0):

        acc_x = np.zeros(X_train.shape[0])
        acc_y = np.zeros(X_train.shape[0])
        m0 = X_train[:, idx, -1]
        charge0 = X_train[:, idx, -2]

        for j in range(4):
            if j == idx:
                continue
            dx = X_train[:, idx, 0] - X_train[:, j, 0]
            dy = X_train[:, idx, 1] - X_train[:, j, 1]
            r = np.sqrt(dx**2 + dy**2) + 1e-2
            chargej = X_train[:, j, -2]

            # Force from -grad(charge_i * charge_j / r) = -charge_i * charge_j / r^2
            force_mag = charge0*chargej/(r**2)
            acc_x += force_mag*(dx/r)/m0
            acc_y += force_mag*(dy/r)/m0

        return np.stack([acc_x, acc_y], axis=1)


train_data, _, _ = load_and_process(data_path, seed)

X_train, y_train = train_data
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

print('checking that force function is as expected:')
calculated = force_func(X_train, idx = idx)
print('force func shape:', calculated.shape)
print('force func first 5:', calculated[:5])
print('accelerations first 5:', y_train[:5, idx, :])

config = {'parsimony': 0.05, 
            'complexity_of_constants': 1, 
            'maxsize': 23}

regressor = PySRRegressor(
    maxsize=config['maxsize'],
    niterations=niterations,
    binary_operators=["+", "*"],
    unary_operators=[
        "inv(x) = 1/x",
        "exp",
        "log"
    ],
    extra_sympy_mappings={
        "inv": lambda x: 1 / x
    },
    constraints={'exp': (1), 'log': (1)},
    complexity_of_operators={"exp": 3, "log": 3, "^": 3},
    complexity_of_constants=config['complexity_of_constants'],
    elementwise_loss="loss(prediction, target) = abs(prediction - target)",
    parsimony=config['parsimony'],
    batching=True, 
    output_directory = 'sr_on_inputs_results',
    run_id = f'{sim}'
)

sr_inputs = []
sr_inputs.append (X_train[:, idx, -1])
sr_inputs.append (X_train[:, idx, -2])
sr_inputs_names = ['m0', 'q0']

for j in range (4):
    if j == idx:
        continue
    dx = X_train[:, idx, 0] - X_train[:, j, 0]
    dy = X_train[:, idx, 1] - X_train[:, j, 1]
    r = np.sqrt(dx**2 + dy**2) + 1e-2

    mj = X_train[:, j, -1]
    qj = X_train[:, j, -2]
    
    sr_inputs.append(mj)
    sr_inputs.append(qj)
    sr_inputs.append(dx)
    sr_inputs.append(dy)
    sr_inputs.append(r)

    sr_inputs_names.append(f'm{j}')
    sr_inputs_names.append(f'q{j}')
    sr_inputs_names.append(f'dx{j}')
    sr_inputs_names.append(f'dy{j}')
    sr_inputs_names.append(f'r{j}')

sr_inputs = np.array(sr_inputs).T
sr_targets = y_train[:,0,:]


chosen_idx =np.random.choice(len(sr_inputs), size=num_points, replace=False)

sr_inputs_subset = sr_inputs[chosen_idx]
sr_targets_subset = sr_targets[chosen_idx]

regressor.fit(sr_inputs_subset, sr_targets_subset, variable_names = sr_inputs_names)