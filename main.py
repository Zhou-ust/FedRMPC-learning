# main.py
import torch
import numpy as np
import pandas as pd
from tabulate import tabulate
import copy
import os
import matplotlib.pyplot as plt
from config import Config
from utils import (
    VehicleModel,
    CrossingEnv,
    plot_crossing_comparison,
    plot_radar_chart,
    plot_ablation_comparison,
    plot_robustness_bars,
    plot_ablation_reliability,
    plot_ablation_efficiency,
    plot_uncertainty_surface,
    font_bold, font_new_roman
)
from controllers import PIDController, LinearMPC, RobustMPC, AdaptiveMPC, FedRMPCController
from federated import FedServer, FedClient
from models import BNN


def run_simulation(controllers, env, noise=0.0, disturbance=0.0, mass_error=0.0):
    starts = [
        np.array([-20, -2, 0, 0]), np.array([20, 2, 0, np.pi]),
        np.array([-2, -20, 0, np.pi / 2]), np.array([2, 20, 0, -np.pi / 2])
    ]
    targets = [
        np.array([20, -2, 0, 0]), np.array([-20, 2, 0, np.pi]),
        np.array([-2, 20, 0, np.pi / 2]), np.array([2, -20, 0, -np.pi / 2])
    ]

    masses = [1000 * (1 + mass_error), 1000 * (1 - mass_error), 1000, 1000]
    physics = [VehicleModel(mass=m) for m in masses]

    states = copy.deepcopy(starts)
    trajs = {i: {'traj': [starts[i]], 'success': False} for i in range(4)}
    last_u = {i: np.zeros(2) for i in range(4)}
    active_agents = [True] * 4

    min_dists = []
    costs = 0
    jerks = 0
    violation_steps = 0

    for _ in range(Config.SIM_STEPS):
        for i in range(4):
            if not active_agents[i]: continue

            try:
                u = controllers[i].get_action(states[i], targets[i])
            except:
                u = np.array([0, -1.0])

            s_next = physics[i].step(states[i], u, noise_std=noise, disturbance=disturbance)

            jerk = np.linalg.norm(u - last_u[i]) / Config.DT
            jerks += jerk
            last_u[i] = u

            states[i] = s_next
            trajs[i]['traj'].append(s_next)

            d = env.get_min_dist(s_next[0], s_next[1])
            costs += np.linalg.norm(u)

            record_dist = max(0.0, d)
            min_dists.append(record_dist)

            if 0 < d < Config.SAFETY_MARGIN:
                violation_steps += 1

            if d < 0.0:
                trajs[i]['success'] = False
                active_agents[i] = False
                remaining_steps = Config.SIM_STEPS - (_ + 1)
                min_dists.extend([0.0] * remaining_steps)

            elif np.linalg.norm(s_next[:2] - targets[i][:2]) < 2.5:
                trajs[i]['success'] = True
                active_agents[i] = False

    total_active_steps = sum([len(t['traj']) for t in trajs.values()])
    violation_rate = violation_steps / max(1, total_active_steps)

    success_rate = sum([t['success'] for t in trajs.values()]) / 4

    return {
        'Success Rate': success_rate,
        'Avg Min Dist': np.mean(min_dists) if min_dists else 0,
        'Control Cost': costs / 4,
        'Smoothness': jerks / 4,
        'Violation Rate': violation_rate,
        'Trajectories': trajs
    }


def train_models_for_experiments():
    print("  [Pre-training] Training BNN models with Physics Randomization (Domain Randomization)...")
    data = []

    # 生成多样化训练数据以支持鲁棒性测试
    # 覆盖从 800kg 到 1300kg 的质量范围
    mass_distribution = [800, 900, 1000, 1100, 1200, 1300]

    for m in mass_distribution:
        # 随机化物理参数（如阻力），模拟模型不匹配
        p = VehicleModel(mass=m, drag=np.random.uniform(0.02, 0.10))

        s = np.random.rand(4)
        for _ in range(300):
            s = np.array([
                np.random.uniform(-22, 22),
                np.random.uniform(-22, 22),
                np.random.uniform(0, 12),
                np.random.uniform(-np.pi, np.pi)
            ])
            u = np.random.uniform(-1, 1, 2)

            # 混合噪声注入：包含低噪声与高噪声，使模型学会输出合适方差
            noise_level = np.random.choice([0.0, 0.05, 0.15, 0.3], p=[0.4, 0.3, 0.2, 0.1])
            disturbance = np.random.choice([0.0, 1.0], p=[0.8, 0.2])

            sn = p.step(s, u, noise_std=noise_level, disturbance=disturbance)
            data.append((s, u, sn))

    print(f"  Collected {len(data)} domain-randomized samples.")

    srv = FedServer()
    clients = [FedClient(i) for i in range(4)]

    # 模拟 Non-IID 数据分布
    np.random.shuffle(data)

    chunk = len(data) // 4
    for i in range(4):
        for d in data[i * chunk:(i + 1) * chunk]: clients[i].add_data(*d)

    w_global = srv.global_model.state_dict()
    for r in range(Config.ROUNDS):
        for c in clients: c.train(w_global, mode='FedProx')
        w_global = srv.aggregate(clients)

    global_bnn = BNN().to(Config.DEVICE)
    global_bnn.load_state_dict(w_global)

    local_bnns = []
    for i in range(4):
        c = FedClient(i)
        # 重新分配数据用于本地对比
        for d in data[i * chunk:(i + 1) * chunk]: c.add_data(*d)
        w_local = c.model.state_dict()
        for _ in range(Config.ROUNDS):
            c.train(w_local, mode='FedAvg')
            w_local = c.model.state_dict()
        l_bnn = BNN().to(Config.DEVICE)
        l_bnn.load_state_dict(w_local)
        local_bnns.append(l_bnn)

    return global_bnn, local_bnns


def exp1_comparison(global_bnn):
    print(">>> Exp 1: MPC Comparison...")
    env = CrossingEnv(seed=123)

    plot_uncertainty_surface(global_bnn, env, "exp1_uncertainty_heatmap.pdf")

    ctls = {
        'Linear MPC': [LinearMPC(env) for _ in range(4)],
        'Robust MPC': [RobustMPC(env) for _ in range(4)],
        'FedRMPC': [FedRMPCController(env, global_bnn) for _ in range(4)]
    }

    res_list = [];
    all_trajs = {}
    for n, c in ctls.items():
        print(f"  Testing {n}...")
        r = run_simulation(c, env)
        res_list.append({
            'Method': n,
            'Success Rate': r['Success Rate'],
            'Avg Min Dist': r['Avg Min Dist'],
            'Control Cost': r['Control Cost'],
            'Smoothness': r['Smoothness'],
            'Violation Rate': r['Violation Rate']
        })
        all_trajs[n] = r['Trajectories']

    df = pd.DataFrame(res_list)
    df.to_excel(os.path.join(Config.RESULTS_DIR, 'exp1_metrics.xlsx'))

    starts = [np.array([-20, -2]), np.array([20, 2]), np.array([-2, -20]), np.array([2, 20])]
    targets = [np.array([20, -2]), np.array([-20, 2]), np.array([-2, 20]), np.array([2, -20])]

    plot_crossing_comparison(env, all_trajs, targets, starts, "exp1_trajectories.pdf")
    plot_radar_chart(df, "exp1_radar_chart.pdf")


def exp2_fl_analysis():
    print(">>> Exp 2: FL Metrics...")
    phy = VehicleModel()
    data = []
    for _ in range(4):
        for _ in range(100):
            s = np.array([np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(0, 10),
                          np.random.uniform(-np.pi, np.pi)])
            u = np.random.uniform(-1, 1, 2)
            sn = phy.step(s, u)
            data.append((s, u, sn))

    modes = ['Local', 'FedAvg', 'FedProx']
    hist = {'Round': range(1, Config.ROUNDS + 1)}

    for mode in modes:
        srv = FedServer();
        cli = [FedClient(i) for i in range(4)]
        chunk = len(data) // 4
        for i in range(4):
            for d in data[i * chunk:(i + 1) * chunk]: cli[i].add_data(*d)
        mses, drifts, vars = [], [], []
        w_g = srv.global_model.state_dict()

        for r in range(Config.ROUNDS):
            loc_mse, loc_drift = [], []
            for c in cli:
                w_in = c.model.state_dict() if mode == 'Local' else w_g
                mse, _, drift = c.train(w_in, mode=mode)
                loc_mse.append(mse);
                loc_drift.append(drift)

            if mode != 'Local':
                vars.append(srv.calc_model_variance(cli))
                w_g = srv.aggregate(cli)
            else:
                vars.append(0.0)

            mses.append(np.mean(loc_mse));
            drifts.append(np.mean(loc_drift))

        hist[f'{mode}_MSE'] = mses
        hist[f'{mode}_Drift'] = drifts
        hist[f'{mode}_Var'] = vars

    df = pd.DataFrame(hist)
    df.to_excel(os.path.join(Config.RESULTS_DIR, 'exp2_full_metrics.xlsx'))

    # 绘制 MSE 曲线
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    for m in modes:
        ax1.plot(hist['Round'], hist[f'{m}_MSE'], marker='o', lw=4, markersize=12, label=m,
                 color=Config.COLORS.get(m, 'k'))
    ax1.set_xlabel("Rounds", fontproperties=font_bold, fontsize=28)
    ax1.set_ylabel("MSE", fontproperties=font_bold, fontsize=28)
    ax1.legend(frameon=False, prop=font_new_roman)
    ax1.grid(True, ls=':', alpha=0.6, color='gray')
    ax1.tick_params(axis='both', colors='black', labelsize=24, width=3, length=8)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, "exp2_mse.pdf"))
    plt.close()

    # 绘制 Drift 曲线
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    for m in ['FedAvg', 'FedProx']:
        ax2.plot(hist['Round'], hist[f'{m}_Drift'], marker='s', lw=4, markersize=12, ls='--', label=f"{m} Drift",
                 color=Config.COLORS.get(m, 'k'))
    ax2.set_xlabel("Rounds", fontproperties=font_bold, fontsize=28)
    ax2.set_ylabel("Drift (L2 Norm)", fontproperties=font_bold, fontsize=28)
    ax2.legend(frameon=False, prop=font_new_roman)
    ax2.grid(True, ls=':', alpha=0.6, color='gray')
    ax2.tick_params(axis='both', colors='black', labelsize=24, width=3, length=8)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, "exp2_drift.pdf"))
    plt.close()

    # 绘制 Variance 曲线
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    for m in ['FedAvg', 'FedProx']:
        ax3.plot(hist['Round'], hist[f'{m}_Var'], marker='^', lw=4, markersize=12, ls='-.', label=f"{m} Variance",
                 color=Config.COLORS.get(m, 'k'))
    ax3.set_xlabel("Rounds", fontproperties=font_bold, fontsize=28)
    ax3.set_ylabel("Model Variance", fontproperties=font_bold, fontsize=28)
    ax3.legend(frameon=False, prop=font_new_roman)
    ax3.grid(True, ls=':', alpha=0.6, color='gray')
    ax3.tick_params(axis='both', colors='black', labelsize=24, width=3, length=8)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, "exp2_var.pdf"))
    plt.close()


def run_exp3_robustness(global_bnn):
    print(">>> Exp 3: Robustness (Multi-Metric)...")
    env = CrossingEnv(seed=None)
    scenarios = [
        {'name': 'Low Noise', 'noise': 0.05, 'dist': 0.0, 'mass': 0.0},
        {'name': 'High Noise', 'noise': 0.25, 'dist': 0.0, 'mass': 0.0},
        {'name': 'Wind', 'noise': 0.0, 'dist': 1.0, 'mass': 0.0},
        {'name': 'Model Error', 'noise': 0.0, 'dist': 0.0, 'mass': 0.5}
    ]
    res = []

    methods_config = [
        ('Robust MPC', lambda: [RobustMPC(env) for _ in range(4)]),
        ('FedRMPC', lambda: [FedRMPCController(env, global_bnn) for _ in range(4)])
    ]

    for sc in scenarios:
        print(f"  Running Scenario: {sc['name']}")
        for method_name, get_controllers in methods_config:
            controllers = get_controllers()
            r = run_simulation(controllers, env,
                               noise=sc['noise'],
                               disturbance=sc['dist'],
                               mass_error=sc['mass'])

            # 收集三大核心指标
            res.append({
                'Scenario': sc['name'],
                'Method': method_name,
                'Safety Margin': r['Avg Min Dist'],
                'Safety Compliance': (1.0 - r['Violation Rate']) * 100.0,
                'Control Cost': r['Control Cost']
            })

    df = pd.DataFrame(res)
    df.to_excel(os.path.join(Config.RESULTS_DIR, 'exp3_robustness_multi.xlsx'))

    plot_robustness_bars(df, "exp3_robustness_metrics.pdf")


def run_exp4_ablation(global_bnn, local_bnns):
    print(">>> Exp 4: Ablation...")
    env = CrossingEnv(seed=123)
    configs = {}
    configs['Full FedRMPC'] = [FedRMPCController(env, global_bnn, uncertainty_aware=True) for _ in range(4)]
    configs['w/o Uncertainty'] = [FedRMPCController(env, global_bnn, uncertainty_aware=False) for _ in range(4)]
    configs['w/o Federated'] = [FedRMPCController(env, local_bnns[i], uncertainty_aware=True) for i in range(4)]

    results = [];
    all_trajs = {}
    for name, ctls in configs.items():
        print(f"  Testing {name}...")
        r = run_simulation(ctls, env)
        results.append({
            'Config': name,
            'Success': r['Success Rate'],
            'Cost': r['Control Cost'],
            'Safety': r['Avg Min Dist'],
            'Jerk': r['Smoothness']
        })
        all_trajs[name] = r['Trajectories']

    df = pd.DataFrame(results)
    df.to_excel(os.path.join(Config.RESULTS_DIR, 'exp4_ablation.xlsx'))

    starts = [np.array([-20, -2]), np.array([20, 2]), np.array([-2, -20]), np.array([2, 20])]
    targets = [np.array([20, -2]), np.array([-20, 2]), np.array([-2, 20]), np.array([2, -20])]

    plot_ablation_comparison(env, all_trajs, targets, starts, "exp4_ablation_traj.pdf")
    plot_ablation_reliability(df, "exp4_ablation_reliability.pdf")
    plot_ablation_efficiency(df, "exp4_ablation_efficiency.pdf")


if __name__ == "__main__":
    torch.manual_seed(666)
    np.random.seed(100)
    global_model, local_models = train_models_for_experiments()
    exp1_comparison(global_model)
    exp2_fl_analysis()
    run_exp3_robustness(global_model)
    run_exp4_ablation(global_model, local_models)
    print(f"All experiments finished. Results saved to {Config.RESULTS_DIR}")