# FedRMPC Implementation

This repository contains the source code implementation for the paper **"FedRMPC: Federated Robust Model Predictive Control via Uncertainty Regularization"**.

## 🛠️ Project Structure

The codebase is organized into modular components handling simulation, control logic, federated learning, and visualization.

```bash
.
├── config.py           # Global Configuration
│   ├── Simulation settings (steps, agents, obstacles)
│   ├── Vehicle dynamics parameters (mass, drag, limits)
│   ├── MPC hyperparameters (horizon, weights, robust beta)
│   └── Federated Learning hyperparameters (rounds, epochs, lr)
│
├── controllers.py      # Control Algorithms
│   ├── PIDController: Baseline PID control
│   ├── LinearMPC: Standard Linear MPC implementation
│   ├── RobustMPC: Tube-based Robust MPC baseline
│   ├── AdaptiveMPC: Adaptive MPC baseline
│   └── FedRMPCController: Our proposed controller with dynamic uncertainty regularization
│
├── federated.py        # Federated Learning Logic
│   ├── FedClient: Handles local training loops and data buffering
│   └── FedServer: Manages global model aggregation (FedAvg/FedProx) and variance calculation
│
├── models.py           # Neural Network Architecture
│   └── BNN: Bayesian Neural Network implementation using Monte Carlo Dropout
│
├── utils.py            # Simulation Environment & Utilities
│   ├── VehicleModel: Kinematic bicycle model dynamics
│   ├── CrossingEnv: Multi-agent intersection environment with obstacles
│   └── Plotting functions for trajectories, radar charts, and metrics
│
└── main.py             # Execution Entry Point
    └── Orchestrates data generation, training, and experiment execution
```

## 📦 Requirements

The code requires Python 3.8+ and the following libraries. You can install them using pip:

```bash
pip install torch numpy pandas matplotlib seaborn scipy tabulate
```

## 🚀 Usage

The project is designed to be run via the `main.py` script, which automatically executes the entire pipeline sequentially.

```bash
python main.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
