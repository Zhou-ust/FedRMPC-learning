# config.py
import torch
import os
import matplotlib.pyplot as plt

class Config:
    # --- 路径与硬件 ---
    RESULTS_DIR = 'results'
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running real simulation on: {DEVICE}")

    # --- IEEE / Nature 绘图风格设置 ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 28,
        'axes.labelsize': 32,
        'axes.titlesize': 34,
        'xtick.labelsize': 32,
        'ytick.labelsize': 32,
        'legend.fontsize': 32,
        'axes.linewidth': 2,
        'lines.linewidth': 5.0,
        'lines.markersize': 16,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'mathtext.fontset': 'stix',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'text.color': 'black',
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    # --- 物理仿真 ---
    DT = 0.1
    SIM_STEPS = 180
    NUM_AGENTS = 4
    OBSTACLE_DENSITY = 3

    # --- 车辆动力学 ---
    L = 2.5
    MAX_STEER = 0.6
    MAX_ACCEL = 2.0
    DRAG = 0.05
    MASS = 1000.0

    # --- MPC 参数 ---
    HORIZON = 12
    Q_TRACKING = 10.0   # 追踪权重
    R_CONTROL = 0.1     # 控制能耗权重
    UNCERTAINTY_WEIGHT = 5.0  # 不确定性惩罚权重

    # 鲁棒膨胀系数：将不确定性转化为物理距离的倍率 (Geometric Robustness)
    ROBUST_BETA = 2.5

    OBSTACLE_PENALTY = 5000.0
    SAFETY_MARGIN = 1.2

    # PID & Adaptive 参数
    KP_STEER = 2.0
    KP_ACCEL = 2.0
    ADAPT_LR = 0.1

    # --- 联邦学习参数 ---
    ROUNDS = 25
    LOCAL_EPOCHS = 10
    BATCH_SIZE = 64
    LR = 0.005
    HIDDEN_DIM = 128
    DROPOUT = 0.15
    PROXIMAL_MU = 0.05

    # --- 配色方案 ---
    COLORS = {
        'FedRMPC': '#e9a2a3',       # 粉红
        'Full FedRMPC': '#e9a2a3',
        'Robust MPC': '#94cee5',    # 天蓝
        'Linear MPC': '#b6d09e',    # 嫩绿
        'w/o Uncertainty': '#94cee5',
        'w/o Federated': '#f5cfa8', # 淡橙
        'Baseline': '#b6d09e',
        'PID': '#f5cfa8',
        'Centralized': '#000000',
        'FedAvg': '#94cee5',
        'FedProx': '#e9a2a3',
        'Local': '#b6d09e'
    }