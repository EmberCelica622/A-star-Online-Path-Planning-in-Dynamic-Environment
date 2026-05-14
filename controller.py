import numpy as np


class DifferentialDriveRobot:
    def __init__(self, x, y, theta):
        self.state = np.array([x, y, theta], dtype=float)
        self.history = [self.state.copy()]
        self.control_history = []

    def step(self, v, w, dt):
        x, y, theta = self.state

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        self.state = np.array([x, y, theta], dtype=float)
        self.history.append(self.state.copy())
        self.control_history.append([v, w])


class TrajectoryTracker:
    def __init__(self, k_rho=1.2, k_alpha=4.0, k_beta=-1.0,
                 v_max=1.0, w_max=2.0):
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.k_beta = k_beta
        self.v_max = v_max
        self.w_max = w_max

    @staticmethod
    def wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def compute_control(self, state, ref):
        """
        state = [x, y, theta]
        ref   = [x_ref, y_ref, theta_ref]
        """
        x, y, theta = state
        x_r, y_r, theta_r = ref

        dx = x_r - x
        dy = y_r - y

        rho = np.hypot(dx, dy)
        alpha = self.wrap_angle(np.arctan2(dy, dx) - theta)
        beta = self.wrap_angle(theta_r - theta - alpha)

        v = self.k_rho * rho
        w = self.k_alpha * alpha + self.k_beta * beta

        v = np.clip(v, -self.v_max, self.v_max)
        w = np.clip(w, -self.w_max, self.w_max)

        return v, w, rho

    def follow_trajectory(self, robot, trajectory, dt=0.05):
        errors = []

        for ref in trajectory:
            v, w, rho = self.compute_control(robot.state, ref)
            robot.step(v, w, dt)
            errors.append(rho)

        return np.array(errors)


def build_reference_trajectory(smooth_path_xy, env_resolution):
    """
    Input:
        smooth_path_xy in grid coordinates
    Output:
        reference trajectory in world coordinates [x, y, theta]
    """
    if len(smooth_path_xy) < 2:
        raise ValueError("Smooth path too short.")

    path_world = smooth_path_xy * env_resolution

    refs = []
    for i in range(len(path_world) - 1):
        p = path_world[i]
        p_next = path_world[i + 1]
        heading = np.arctan2(p_next[1] - p[1], p_next[0] - p[0])
        refs.append([p[0], p[1], heading])

    refs.append([path_world[-1, 0], path_world[-1, 1], refs[-1][2]])
    return np.array(refs, dtype=float)