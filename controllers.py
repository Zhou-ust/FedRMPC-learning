# controllers.py
import numpy as np
from scipy.optimize import minimize
from config import Config
from utils import VehicleModel


class BaseController:
    def get_action(self, state, target): raise NotImplementedError


# --- PID ---
class PIDController(BaseController):
    def __init__(self):
        self.kp_s = Config.KP_STEER
        self.kp_a = Config.KP_ACCEL

    def get_action(self, state, target):
        x, y, v, yaw = state
        target_yaw = np.arctan2(target[1] - y, target[0] - x)
        yaw_err = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        steer = np.clip(self.kp_s * yaw_err, -Config.MAX_STEER, Config.MAX_STEER)
        dist = np.linalg.norm([x - target[0], y - target[1]])
        accel = np.clip(self.kp_a * (dist - 2.0), -Config.MAX_ACCEL, Config.MAX_ACCEL)
        return np.array([steer, accel])


# --- MPC Base ---
class OptimizationMPC(BaseController):
    def __init__(self, env):
        self.env = env
        self.model = VehicleModel()
        self.horizon = Config.HORIZON
        self.u_prev = np.zeros(self.horizon * 2)

    def solve(self, state, target, cost_func):
        bounds = [(-Config.MAX_STEER, Config.MAX_STEER), (-Config.MAX_ACCEL, Config.MAX_ACCEL)] * self.horizon
        try:
            res = minimize(cost_func, self.u_prev, args=(state, target),
                           method='SLSQP', bounds=bounds, options={'ftol': 0.1, 'maxiter': 5})
            self.u_prev = np.roll(res.x, -2)
            return res.x[:2]
        except:
            return np.array([0., -0.5])


# --- Linear MPC (Baseline) ---
class LinearMPC(OptimizationMPC):
    def get_action(self, state, target): return self.solve(state, target, self._cost)

    def _cost(self, u_flat, state, target):
        u = u_flat.reshape(self.horizon, 2)
        curr = state.copy()
        cost = 0.0
        for t in range(self.horizon):
            curr = self.model.step(curr, u[t])
            # 仅包含追踪与控制代价，无显式避障
            cost += Config.Q_TRACKING * np.linalg.norm(curr[:2] - target[:2])
            cost += Config.R_CONTROL * np.sum(u[t] ** 2)
        return cost


# --- Robust MPC (Baseline) ---
class RobustMPC(OptimizationMPC):
    def get_action(self, state, target):
        return self.solve(state, target, self._cost)

    def _cost(self, u_flat, state, target):
        u = u_flat.reshape(self.horizon, 2)
        curr = state.copy()
        cost = 0.0
        for t in range(self.horizon):
            curr = self.model.step(curr, u[t])
            cost += Config.Q_TRACKING * np.linalg.norm(curr[:2] - target[:2])
            cost += Config.R_CONTROL * np.sum(u[t] ** 2)
            # 传统势场法 (固定安全余量)
            d = self.env.get_min_dist(curr[0], curr[1])
            if d < 1.2:
                cost += Config.OBSTACLE_PENALTY * (1.2 - d) ** 2
        return cost


# --- Adaptive MPC (Baseline) ---
class AdaptiveMPC(OptimizationMPC):
    def __init__(self, env):
        super().__init__(env)
        self.est_drag = 0.0

    def update_model(self, s, a, sn):
        pred = self.model.step(s, a)
        self.est_drag += Config.ADAPT_LR * (pred[2] - sn[2])
        self.est_drag = np.clip(self.est_drag, 0.0, 0.1)
        self.model.drag = self.est_drag

    def get_action(self, s, t): return self.solve(s, t, self._cost)

    def _cost(self, u_flat, s, t):
        u = u_flat.reshape(self.horizon, 2)
        curr = s.copy()
        cost = 0.0
        for i in range(self.horizon):
            curr = self.model.step(curr, u[i])
            cost += Config.Q_TRACKING * np.linalg.norm(curr[:2] - t[:2])
        return cost


# --- FedRMPC (Ours) ---
class FedRMPCController(OptimizationMPC):
    def __init__(self, env, bnn_model, uncertainty_aware=True):
        super().__init__(env)
        self.bnn_model = bnn_model
        self.use_unc = uncertainty_aware
        # 缓存当前不确定性估计
        self.current_unc_estimate = 0.0

    def get_action(self, state, target):
        # 在优化前基于当前状态评估不确定性
        if self.use_unc:
            dummy_u = np.zeros(2)
            _, self.current_unc_estimate = self.bnn_model.predict_uncertainty(state, dummy_u, num_samples=10)
        else:
            self.current_unc_estimate = 0.0

        return self.solve(state, target, self._cost)

    def _cost(self, u_flat, state, target):
        u = u_flat.reshape(self.horizon, 2)
        curr = state.copy()
        cost = 0.0

        # 获取基础不确定性 (t=0)
        base_unc = self.current_unc_estimate

        for t in range(self.horizon):
            curr = self.model.step(curr, u[t])
            dist_phys = self.env.get_min_dist(curr[0], curr[1])

            # 1. 基础追踪代价
            cost += Config.Q_TRACKING * np.linalg.norm(curr[:2] - target[:2])
            cost += Config.R_CONTROL * np.sum(u[t] ** 2)

            if self.use_unc:
                # 动态安全管 (Dynamic Safety Tube / Geometric Robustness)

                # A. 不确定性传播：假设不确定性随预测步长线性增长
                uncertainty_factor = base_unc * (1.0 + 0.15 * t)

                # B. 计算 "风险感知距离"：物理距离减去不确定性膨胀项
                dist_risk = dist_phys - (Config.ROBUST_BETA * uncertainty_factor)

                # C. 鲁棒避障惩罚
                effective_margin = Config.SAFETY_MARGIN
                if dist_risk < effective_margin:
                    penetration = effective_margin - dist_risk
                    cost += Config.OBSTACLE_PENALTY * (penetration ** 2)

                # D. 正则项：引导车辆前往低不确定性区域
                cost += Config.UNCERTAINTY_WEIGHT * uncertainty_factor

            else:
                # 对照组：无不确定性感知，使用固定安全距离
                if dist_phys < Config.SAFETY_MARGIN:
                    cost += Config.OBSTACLE_PENALTY * (Config.SAFETY_MARGIN - dist_phys) ** 2

        return cost