# utils.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import pandas as pd
import os
import seaborn as sns
from config import Config
import matplotlib.font_manager as fm
import torch

# 通用字体设置
font_new_roman = fm.FontProperties(family='Times New Roman', style='normal', size=28)
font_bold = fm.FontProperties(family='Times New Roman', weight='bold', size=30)
font_legend = fm.FontProperties(family='Times New Roman', style='normal', size=26)


class VehicleModel:
    def __init__(self, mass=Config.MASS, drag=Config.DRAG):
        self.L = Config.L
        self.dt = Config.DT
        self.mass = mass
        self.drag = drag

    def step(self, state, control, noise_std=0.0, disturbance=0.0):
        x, y, v, yaw = state
        steer, accel = control
        steer = np.clip(steer, -Config.MAX_STEER, Config.MAX_STEER)
        accel = np.clip(accel, -Config.MAX_ACCEL, Config.MAX_ACCEL)
        beta = np.arctan(0.5 * np.tan(steer))
        dx = v * np.cos(yaw + beta) + np.random.normal(0, noise_std)
        dy = v * np.sin(yaw + beta) + np.random.normal(0, noise_std)
        dv = (accel - self.drag * v + disturbance)
        dyaw = (v / self.L) * np.sin(beta)
        new_x = x + dx * self.dt
        new_y = y + dy * self.dt
        new_v = max(0.0, v + dv * self.dt)
        new_yaw = yaw + dyaw * self.dt
        return np.array([new_x, new_y, new_v, new_yaw])


class CrossingEnv:
    def __init__(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self.obstacles = []
        self._gen_obstacles()

    def _gen_obstacles(self):
        # 必经之路陷阱
        self.obstacles.append({'x': -9.0, 'y': -0.5, 'r': 2.2})
        self.obstacles.append({'x': 9.0, 'y': 0.5, 'r': 2.2})
        self.obstacles.append({'x': -0.5, 'y': -9.0, 'r': 2.2})
        self.obstacles.append({'x': 0.5, 'y': 9.0, 'r': 2.2})
        self.obstacles.append({'x': 0.0, 'y': 0.0, 'r': 2.5})

        density = Config.OBSTACLE_DENSITY
        for _ in range(int(8 * density)):
            x = np.random.uniform(-16, 16)
            y = np.random.uniform(-16, 16)
            if np.linalg.norm([x, y]) < 8.0: continue
            if abs(x) > 15 and abs(y) > 15: continue
            self.obstacles.append({'x': x, 'y': y, 'r': np.random.uniform(1.2, 1.8)})

    def get_min_dist(self, x, y):
        min_d = float('inf')
        for obs in self.obstacles:
            d = np.sqrt((x - obs['x']) ** 2 + (y - obs['y']) ** 2) - obs['r']
            if d < min_d: min_d = d
        return min_d


# --- 绘图函数 ---

def plot_crossing_comparison(env, all_results, targets, starts, filename):
    if not filename.endswith('.pdf'): filename = filename.replace('.png', '.pdf')

    font_title_huge = fm.FontProperties(family='Times New Roman', weight='bold', size=52)
    font_tick_huge = fm.FontProperties(family='Times New Roman', style='normal', size=42)
    font_legend_huge = fm.FontProperties(family='Times New Roman', weight='bold', size=42)

    fig, axs = plt.subplots(1, 4, figsize=(48, 12))
    axs = axs.flatten()

    method_styles = {
        'Linear MPC': {'ls': ':', 'lw': 8.0, 'z': 2},
        'Robust MPC': {'ls': '--', 'lw': 8.0, 'z': 3},
        'FedRMPC': {'ls': '-', 'lw': 10.0, 'z': 5}
    }

    for i in range(Config.NUM_AGENTS):
        ax = axs[i]
        for obs in env.obstacles:
            ax.add_patch(patches.Circle((obs['x'], obs['y']), obs['r'], color='#D2B48C', alpha=0.6, zorder=1))

        for method_name, style in method_styles.items():
            if method_name in all_results:
                res = all_results[method_name][i]
                t = np.array(res['traj'])
                if len(t) > 0:
                    ax.plot(t[:, 0], t[:, 1], color=Config.COLORS[method_name],
                            ls=style['ls'], lw=style['lw'], alpha=0.9, zorder=style['z'])
                    if not res['success']:
                        ax.scatter(t[-1, 0], t[-1, 1], marker='x', color='#8B0000', s=800, zorder=25, lw=8)

        ax.scatter(starts[i][0], starts[i][1], marker='s', color='k', s=400, zorder=20)
        ax.scatter(targets[i][0], targets[i][1], marker='*', color='#FFD700', edgecolors='k', s=1000, zorder=20)

        ax.set_title(f"Agent {i + 1}", fontproperties=font_title_huge, pad=30)
        ax.set_xlim(-23, 23);
        ax.set_ylim(-23, 23)
        ax.grid(True, ls=':', alpha=0.5, color='gray', linewidth=2)
        ax.set_aspect('equal')

        ax.tick_params(axis='both', colors='black', labelsize=42, width=4, length=12, pad=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_tick_huge)

    lines = [
        Line2D([0], [0], color=Config.COLORS['Linear MPC'], lw=8.0, ls=':', label='Linear MPC'),
        Line2D([0], [0], color=Config.COLORS['Robust MPC'], lw=8.0, ls='--', label='Robust MPC'),
        Line2D([0], [0], color=Config.COLORS['FedRMPC'], lw=10.0, ls='-', label='FedRMPC'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=24, label='Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700', markeredgecolor='k', markersize=36,
               label='Goal'),
        Line2D([0], [0], marker='x', color='#8B0000', markerfacecolor='#8B0000', markersize=24, lw=6, label='Collision')
    ]
    fig.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6, frameon=False,
               prop=font_legend_huge)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()


def plot_radar_chart(df, filename):
    if not filename.endswith('.pdf'): filename = filename.replace('.png', '.pdf')

    methods = ['Linear MPC', 'Robust MPC', 'FedRMPC']
    df_plot = df[df['Method'].isin(methods)].copy()

    penalty = 5000.0
    df_plot['Real Cost'] = df_plot['Control Cost'] + (1.0 - df_plot['Success Rate']) * penalty
    df_plot['Efficiency'] = 1.0 / (df_plot['Real Cost'] + 1e-6)
    df_plot['Compliance'] = 1.0 - df_plot['Violation Rate']

    metrics = ['Success Rate', 'Compliance', 'Efficiency', 'Avg Min Dist']
    labels = ['Success Rate', 'Safety Compliance', 'Efficiency', 'Safety Margin']

    for m in metrics:
        mn, mx = df_plot[m].min(), df_plot[m].max()
        if mx - mn > 1e-9:
            df_plot[m] = (df_plot[m] - mn) / (mx - mn)
        else:
            df_plot[m] = 1.0
        df_plot[m] = df_plot[m] * 0.8 + 0.2

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 4)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels, color='black', size=30, fontproperties=font_bold)
    ax.tick_params(pad=80)

    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [], color="grey", size=10)
    plt.ylim(0, 1.0)

    for method in methods:
        if method not in df_plot['Method'].values: continue
        row = df_plot[df_plot['Method'] == method].iloc[0]
        values = row[metrics].tolist()
        values += values[:1]
        color = Config.COLORS[method]
        ax.plot(angles, values, color=color, linewidth=7, label=method)
        ax.fill(angles, values, color=color, alpha=0.15)

    plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), frameon=True, prop=font_new_roman)
    ax.grid(True, color='gray', linestyle=':', linewidth=2.5, alpha=0.5)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(2.5)

    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()


def plot_ablation_comparison(env, results_dict, targets, starts, filename):
    if not filename.endswith('.pdf'): filename = filename.replace('.png', '.pdf')
    fig, axs = plt.subplots(1, 4, figsize=(44, 11))
    axs = axs.flatten()

    method_styles = {
        'Full FedRMPC': {'ls': '-', 'lw': 7.0, 'z': 5},
        'w/o Uncertainty': {'ls': '--', 'lw': 5.0, 'z': 3},
        'w/o Federated': {'ls': ':', 'lw': 5.0, 'z': 2}
    }

    for i in range(Config.NUM_AGENTS):
        ax = axs[i]
        for obs in env.obstacles:
            ax.add_patch(patches.Circle((obs['x'], obs['y']), obs['r'], color='#C0B2A6', alpha=0.5, zorder=1))

        for method_name, style in method_styles.items():
            if method_name in results_dict and i in results_dict[method_name]:
                res = results_dict[method_name][i]
                t = np.array(res['traj'])
                if len(t) > 0:
                    color = Config.COLORS.get(method_name, 'black')
                    ax.plot(t[:, 0], t[:, 1], color=color, ls=style['ls'], lw=style['lw'], alpha=0.8, zorder=style['z'])
                    if not res['success']:
                        ax.scatter(t[-1, 0], t[-1, 1], marker='x', color='#8B0000', s=600, zorder=25, lw=7)

        ax.scatter(starts[i][0], starts[i][1], marker='s', color='k', s=300, zorder=20)
        ax.scatter(targets[i][0], targets[i][1], marker='*', color='#FFD700', edgecolors='k', s=700, zorder=20)

        ax.set_title(f"Agent {i + 1}", fontproperties=font_bold, pad=25, fontsize=36)
        ax.set_xlim(-22, 22);
        ax.set_ylim(-22, 22)
        ax.grid(True, ls=':', alpha=0.6, color='gray')
        ax.set_aspect('equal')
        ax.tick_params(axis='both', colors='black', labelsize=30, width=3, length=8)
        for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontproperties(font_new_roman)

    legend_elements = []
    for name in ['Full FedRMPC', 'w/o Uncertainty', 'w/o Federated']:
        if name in results_dict:
            style = method_styles[name]
            color = Config.COLORS.get(name)
            legend_elements.append(Line2D([0], [0], color=color, lw=style['lw'], ls=style['ls'], label=name))

    legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=18, label='Start'))
    legend_elements.append(
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700', markeredgecolor='k', markersize=26,
               label='Goal'))
    legend_elements.append(Line2D([0], [0], marker='x', color='#8B0000', markerfacecolor='#8B0000', markersize=18, lw=5,
                                  label='Collision'))

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6, frameon=False,
               prop=font_bold)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()


def plot_robustness_bars(df, filename):
    """
    绘制鲁棒性测试的多指标对比图 (3个子图: Safety Margin, Safety Compliance, Control Cost)
    """
    if not filename.endswith('.pdf'): filename = filename.replace('.png', '.pdf')

    # 设置画布：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    # 定义指标
    metrics = [
        {'col': 'Safety Margin', 'ylabel': 'Safety Margin (m)', 'title': '(a) Safety Margin (Higher is Better)'},
        {'col': 'Safety Compliance', 'ylabel': 'Compliance (%)', 'title': '(b) Safety Compliance (Higher is Better)'},
        {'col': 'Control Cost', 'ylabel': 'Control Cost', 'title': '(c) Control Cost (Lower is Better)'}
    ]

    sns.set_style("whitegrid")
    # 强制设置全局字体和线宽，避免被 seaborn 覆盖
    plt.rcParams['axes.linewidth'] = 3.0
    plt.rcParams['axes.edgecolor'] = 'black'

    palette = [Config.COLORS['Robust MPC'], Config.COLORS['FedRMPC']]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # 绘制
        sns.barplot(data=df, x='Scenario', y=metric['col'], hue='Method',
                    palette=palette, ax=ax, edgecolor='black', linewidth=2.5, alpha=0.9)

        # 设置标签
        ax.set_ylabel(metric['ylabel'], fontproperties=font_bold, fontsize=32, color='black')
        ax.set_xlabel("", fontsize=0)
        ax.set_title(metric['title'], fontproperties=font_bold, fontsize=34, pad=20)

        # 刻度设置
        ax.tick_params(axis='x', labelsize=28, colors='black', pad=10)
        ax.tick_params(axis='y', labelsize=28, colors='black')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_new_roman)

        # 边框
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3.0)

        # 图例仅在第一张图显示
        if i == 0:
            ax.legend(prop=font_new_roman, fontsize=28, loc='upper left', frameon=True, framealpha=0.9)
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # 数值标签
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=22, fontproperties=font_bold)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), dpi=300)
    plt.close()


def plot_ablation_reliability(df, filename):
    if not filename.endswith('.pdf'): filename = filename.replace('.png', '.pdf')
    df = df[df['Config'] != 'Baseline'].copy()
    fig, ax1 = plt.subplots(figsize=(16, 11))
    configs = df['Config']
    x = np.arange(len(configs))
    width = 0.5

    bars1 = ax1.bar(x, df['Success'], width, label='Success Rate',
                    color=Config.COLORS['FedRMPC'], alpha=0.9, edgecolor='black', lw=0)
    ax1.set_ylabel('Success Rate', color='black', fontproperties=font_bold, fontsize=34)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=30, width=3, length=8)
    for label in ax1.get_yticklabels(): label.set_fontproperties(font_new_roman)
    ax1.set_ylim(0, 1.15)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, df['Safety'], color=Config.COLORS['Robust MPC'], marker='o', lw=7, markersize=18,
                  label='Safety Margin', mec='k', mew=3)
    ax1_twin.set_ylabel('Safety Margin (m)', color='black', fontproperties=font_bold, fontsize=34)
    ax1_twin.tick_params(axis='y', labelcolor='black', labelsize=30, width=3, length=8)
    for label in ax1_twin.get_yticklabels(): label.set_fontproperties(font_new_roman)

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=15, fontproperties=font_bold, fontsize=30, color='black')
    ax1.grid(True, axis='y', linestyle='-', alpha=0.4, color='gray')

    for ax in [ax1, ax1_twin]:
        for spine in ax.spines.values(): spine.set_edgecolor('black'); spine.set_linewidth(3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2,
                    frameon=False, prop=font_legend)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), dpi=300)
    plt.close()


def plot_ablation_efficiency(df, filename):
    if not filename.endswith('.pdf'): filename = filename.replace('.png', '.pdf')
    df = df[df['Config'] != 'Baseline'].copy()
    fig, ax2 = plt.subplots(figsize=(16, 11))
    configs = df['Config']
    x = np.arange(len(configs))
    width = 0.5

    bars2 = ax2.bar(x, df['Cost'], width, label='Control Cost',
                    color=Config.COLORS['w/o Uncertainty'], alpha=0.9, edgecolor='black', lw=0)
    ax2.set_ylabel('Control Cost', color='black', fontproperties=font_bold, fontsize=34)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=30, width=3, length=8)
    for label in ax2.get_yticklabels(): label.set_fontproperties(font_new_roman)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, df['Jerk'], color='#8B4513', marker='s', lw=7, markersize=18, linestyle='--', label='Jerk',
                  mec='k', mew=3)
    ax2_twin.set_ylabel('Jerk Magnitude', color='black', fontproperties=font_bold, fontsize=34)
    ax2_twin.tick_params(axis='y', labelcolor='black', labelsize=30, width=3, length=8)
    for label in ax2_twin.get_yticklabels(): label.set_fontproperties(font_new_roman)

    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=15, fontproperties=font_bold, fontsize=30, color='black')
    ax2.grid(True, axis='y', linestyle='-', alpha=0.4, color='gray')

    for ax in [ax2, ax2_twin]:
        for spine in ax.spines.values(): spine.set_edgecolor('black'); spine.set_linewidth(3)

    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_twin.get_legend_handles_labels()
    ax2_twin.legend(lines3 + lines4, labels3 + labels4, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2,
                    frameon=False, prop=font_legend)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), dpi=300)
    plt.close()


def plot_uncertainty_surface(model, env, filename='exp_uncertainty_map.png'):
    if not filename.endswith('.pdf'): filename = filename.replace('.png', '.pdf')
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(x, y)

    states = np.zeros((100 * 100, 4))
    states[:, 0] = X.flatten()
    states[:, 1] = Y.flatten()
    states[:, 2] = 5.0
    states[:, 3] = 0.0

    actions = np.zeros((100 * 100, 2))

    s_tensor = torch.FloatTensor(states).to(Config.DEVICE)
    a_tensor = torch.FloatTensor(actions).to(Config.DEVICE)

    preds = []
    model.train()
    with torch.no_grad():
        for _ in range(20):
            preds.append(model(s_tensor, a_tensor))

    preds = torch.stack(preds)
    var = preds.var(dim=0).mean(dim=1).cpu().numpy()
    Z = var.reshape(100, 100)

    fig, ax = plt.subplots(figsize=(16, 14))
    cp = ax.contourf(X, Y, Z, levels=100, cmap='OrRd', alpha=0.9)
    cbar = fig.colorbar(cp)

    # 字体与刻度设置
    cbar.ax.set_ylabel('Epistemic Uncertainty', fontproperties=font_bold, fontsize=46)
    cbar.ax.tick_params(labelsize=38)

    for obs in env.obstacles:
        ax.add_patch(
            patches.Circle((obs['x'], obs['y']), obs['r'], edgecolor='black', facecolor='none', lw=3, zorder=10))

    ax.set_xlabel("X (m)", fontproperties=font_bold, fontsize=52)
    ax.set_ylabel("Y (m)", fontproperties=font_bold, fontsize=52)

    ax.tick_params(labelsize=42, width=5, length=12)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_new_roman)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), dpi=300)
    plt.close()