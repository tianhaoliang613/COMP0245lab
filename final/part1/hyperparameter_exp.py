from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from rollout_loader import load_rollouts # Ensure rollout_loader.py is ready for Part1
from pathlib import Path
import os
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("-" * 50) 


DATA_PART1_DIR = Path(__file__).resolve().parent / "data_part1"
DATA_PART1_DIR.mkdir(exist_ok=True) # Ensure folder exists
print(f"Loading Part 1 data from: {DATA_PART1_DIR}")

rollouts = load_rollouts(indices=range(50), directory=DATA_PART1_DIR) 

X_list = []
y_list = []

for rollout in rollouts:
    q_des_traj = np.array(rollout.q_d_all)
    q_mes_traj = np.array(rollout.q_mes_all)
    tau_traj = np.array(rollout.tau_cmd_all)
    error = q_des_traj - q_mes_traj
    X_list.append(error)
    y_list.append(tau_traj)

X = np.vstack(X_list)
y = np.vstack(y_list)

# Data splitting (60% train, 20% validation, 20% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2


class RobotTorqueDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

class MLPNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPNet, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, 128)
        self.activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x


class MLPNet2(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPNet2, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, 256)
        self.activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class MLPNet5(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPNet5, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, 512)
        self.activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class MLPNet3(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPNet3, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, 256)
        self.activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(256, 128)
        self.activation = nn.ReLU()
        self.hidden_layer_3 = nn.Linear(128, 64)
        self.activation = nn.ReLU()
        self.hidden_layer_4 = nn.Linear(64, 32)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        x = self.hidden_layer_3(x)
        x = self.activation(x)
        x = self.hidden_layer_4(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class MLPNet4(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPNet4, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, 512)
        self.activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(512, 256)
        self.activation = nn.ReLU()
        self.hidden_layer_3 = nn.Linear(256, 128)
        self.activation = nn.ReLU()
        self.hidden_layer_4 = nn.Linear(128, 64)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        x = self.hidden_layer_3(x)
        x = self.activation(x)
        x = self.hidden_layer_4(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class MLPNet6(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPNet6, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, 1024)
        self.activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(1024, 512)
        self.activation = nn.ReLU()
        self.hidden_layer_3 = nn.Linear(512, 256)
        self.activation = nn.ReLU()
        self.hidden_layer_4 = nn.Linear(256, 128)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(128, output_size)
    def forward(self, x):
        x = self.activation(self.hidden_layer_1(x))
        x = self.activation(self.hidden_layer_2(x))
        x = self.activation(self.hidden_layer_3(x))
        x = self.activation(self.hidden_layer_4(x))
        x = self.output_layer(x)
        return x

# ==============================================================================
# 3. Create reusable experiment function
# ==============================================================================
def run_experiment(experiment_name, model_class, learning_rate, use_standardization, batch_size,
                   X_train, y_train, X_val, y_val, X_test, y_test):
    
    print(f"\nRunning experiment: {experiment_name}")
    print(f"Model: {model_class.__name__}, LR: {learning_rate}, Standardization: {use_standardization}, Batch size: {batch_size}")

    if use_standardization:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_proc = scaler_X.fit_transform(X_train)
        y_train_proc = scaler_y.fit_transform(y_train)
        X_val_proc = scaler_X.transform(X_val)
        y_val_proc = scaler_y.transform(y_val)
        X_test_proc = scaler_X.transform(X_test)
    else:
        X_train_proc, y_train_proc = X_train, y_train
        X_val_proc, y_val_proc = X_val, y_val
        X_test_proc = X_test

    train_dataset = RobotTorqueDataset(X_train_proc, y_train_proc)
    val_dataset = RobotTorqueDataset(X_val_proc, y_val_proc)
    test_dataset = RobotTorqueDataset(X_test_proc, y_test) 
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = model_class(input_size=input_dim, output_size=output_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 50 

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                total_val_loss += criterion(model(inputs), targets).item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    # --- Evaluation ---
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            all_predictions.append(model(inputs).cpu().numpy())
    y_pred_proc = np.vstack(all_predictions)
    
    if use_standardization:
        y_pred = scaler_y.inverse_transform(y_pred_proc)
    else:
        y_pred = y_pred_proc
        
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Return results for group plotting
    return {
        'name': experiment_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'mse': mse,
        'r2': r2,
        'model_name': model_class.__name__,
        'learning_rate': learning_rate,
        'use_standardization': use_standardization,
        'batch_size': batch_size
    }

def plot_experiment_group(group_name, results_list):
    """Plot comprehensive results for an experiment group"""
    print(f"\nGenerating experiment group plot: {group_name}")
    
    plt.figure(figsize=(14, 8))
    
    # Set colors and line styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    # Plot training and validation loss curves
    for i, result in enumerate(results_list):
        color = colors[i % len(colors)]
        style = line_styles[i % len(line_styles)]
        epochs = range(1, len(result['train_losses']) + 1)
        
        # Plot training loss (solid line)
        plt.plot(epochs, result['train_losses'], 
                color=color, linestyle=style, linewidth=2, alpha=0.8,
                label=f"{result['name']} - Training (MSE: {result['mse']:.4f})")
        
        # Plot validation loss (dashed line)
        plt.plot(epochs, result['val_losses'], 
                color=color, linestyle=':', linewidth=2, alpha=0.6,
                label=f"{result['name']} - Validation (R²: {result['r2']:.4f})")
    
    plt.title(f'{group_name} - Training and Validation Loss Curves', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    # Save figure
    results_dir = Path(__file__).resolve().parent / "experiment_results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / f"{group_name}.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Plot saved: {results_dir / f'{group_name}.png'}")
    
    # Print summary results
    print(f"\n{group_name} Results Summary:")
    print("-" * 60)
    for result in results_list: 
        print(f"  {result['name']:30} MSE: {result['mse']:8.4f}  R²: {result['r2']:7.4f}")
    print("-" * 60)

# ==============================================================================
# 4. Execute our designed experiments
# ==============================================================================
if __name__ == '__main__':  

    # --- Experiment Group 1: Model Architecture Comparison ---
    print("\n" + "="*50)
    print("Experiment Group 1: Model Architecture Comparison")
    print("="*50)
    
    exp1_results = []
    exp1_configs = [
        ("MLPNet", MLPNet),
        ("MLPNet2", MLPNet2), 
        ("MLPNet3", MLPNet3),
        ("MLPNet4", MLPNet4),
        ("MLPNet5", MLPNet5),
        ("MLPNet6", MLPNet6)
    ]
    
    for name, model_class in exp1_configs:
        result = run_experiment(name, model_class, 0.0005, True, 64, 
                              X_train, y_train, X_val, y_val, X_test, y_test)
        exp1_results.append(result)
    
    plot_experiment_group("Exp1_Model_Architecture_Comparison", exp1_results)

    # --- Experiment Group 2: Learning Rate Comparison ---
    print("\n" + "="*50)
    print("Experiment Group 2: Learning Rate Comparison") 
    print("="*50)
    
    exp2_results = []
    exp2_configs = [
        ("LR_0.005", 0.005),
        ("LR_0.001", 0.001),
        ("LR_0.0005", 0.0005), 
        ("LR_0.0001", 0.0001)
    ]
    
    for name, lr in exp2_configs:
        result = run_experiment(name, MLPNet4, lr, True, 64,
                              X_train, y_train, X_val, y_val, X_test, y_test)
        exp2_results.append(result)
    
    plot_experiment_group("Exp2_Learning_Rate_Comparison", exp2_results)
   
    # --- Experiment Group 3: Standardization Comparison ---
    print("\n" + "="*50)
    print("Experiment Group 3: Standardization Comparison")
    print("="*50)
    
    exp3_results = []
    exp3_configs = [
        ("With_Standardization", True),
        ("Without_Standardization", False)
    ]
    
    for name, use_std in exp3_configs:
        result = run_experiment(name, MLPNet4, 0.0005, use_std, 64,
                              X_train, y_train, X_val, y_val, X_test, y_test)
        exp3_results.append(result)
    
    plot_experiment_group("Exp3_Standardization_Comparison", exp3_results)
    
    # --- Experiment Group 4: Batch Size Comparison ---
    print("\n" + "="*50)
    print("Experiment Group 4: Batch Size Comparison")
    print("="*50)
    
    exp4_results = []
    exp4_configs = [
        ("BatchSize_32", 32),
        ("BatchSize_64", 64),   
        ("BatchSize_128", 128)
    ]
    
    for name, batch_size in exp4_configs:
        result = run_experiment(name, MLPNet4, 0.0005, True, batch_size,
                              X_train, y_train, X_val, y_val, X_test, y_test)
        exp4_results.append(result)
    
    plot_experiment_group("Exp4_Batch_Size_Comparison", exp4_results)
    
    print("\n" + "="*80)
    print("All experiments completed! Please check the experiment_results folder for generated plots.")
    print("="*80)