import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


class GridEnvironment:
    def __init__(self, width=60, height=40, resolution=0.2, robot_radius=0.2):
        """                     
        width, height: number of cells
        resolution: meters per cell
        robot_radius: meters
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.robot_radius = robot_radius

        # Ground-truth occupancy map: 0 free, 1 obstacle
        self.static_occupancy = np.zeros((height, width), dtype=np.uint8)
        self.dynamic_occupancy = np.zeros((height, width), dtype=np.uint8)
        self.original = np.zeros((height, width), dtype=np.uint8)   # static before inflation
        self.occupancy = np.zeros((height, width), dtype=np.uint8)  # static + dynamic
        self.experience_map = np.zeros((height, width), dtype=np.float32)
        
        # Belief map: probability that each cell is occupied, initialized unknown
        self.belief = np.full((height, width), 0.5, dtype=np.float32)
        self.fused_belief = self.belief.copy()

        # Entropy map derived from belief
        self.entropy = np.zeros((height, width), dtype=np.float32)

        # Keep compatibility with planner naming
        self.uncertainty = self.entropy
        # self.confidence = np.zeros((height, width), dtype=np.float32)


    def add_rectangle_obstacle(self, x_min, y_min, x_max, y_max, dynamic=False):
        target = self.dynamic_occupancy if dynamic else self.static_occupancy
        target[y_min:y_max + 1, x_min:x_max + 1] = 1

    def inflate_obstacles(self):
        inflation_cells = int(np.ceil(2 * self.robot_radius / self.resolution))

        self.original = self.static_occupancy.copy()

        free_mask = 1 - self.static_occupancy
        dist_to_obstacle = distance_transform_edt(free_mask)
        inflated = dist_to_obstacle <= inflation_cells

        self.static_occupancy = np.where(inflated, 1, 0).astype(np.uint8)
        self.refresh_occupancy()

    def refresh_occupancy(self):
        self.occupancy = np.logical_or(
            self.static_occupancy,
            self.dynamic_occupancy
        ).astype(np.uint8)

    def build_belief_map(self, observed_region="partial", p_free=0.1, p_occ=0.9):
        """
        Build an occupancy-probability map.

        Parameters
        ----------
        observed_region : str
            "partial" or "full"
        p_free : float
            Occupancy probability assigned to observed free cells
        p_occ : float
            Occupancy probability assigned to observed obstacle cells
        """
        self.belief.fill(0.5)

        if observed_region == "full":
            observed_mask = np.ones_like(self.occupancy, dtype=bool)

        elif observed_region == "none":
            observed_mask = np.zeros_like(self.occupancy, dtype=bool)

        elif observed_region == "partial":
            observed_mask = np.zeros_like(self.occupancy, dtype=bool)

            # left half observed
            observed_mask[:, : self.width // 2] = True

            # central horizontal observed corridor
            observed_mask[self.height // 3: 2 * self.height // 3, :] = True

            # circular observed area near start
            yy, xx = np.indices((self.height, self.width))
            cx, cy, r = int(self.width * 0.18), int(self.height * 0.18), 10
            circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            observed_mask = np.logical_or(observed_mask, circle_mask)

        else:
            raise ValueError(f"Unknown observed_region: {observed_region}")

        # Observed cells become confident
        self.belief[np.logical_and(observed_mask, self.occupancy == 0)] = p_free
        self.belief[np.logical_and(observed_mask, self.occupancy == 1)] = p_occ

        # Unobserved cells remain 0.5
        self.compute_entropy_map()

    def build_experience_map(self):
        self.experience_map.fill(0.0)
        self.experience_map[6:79, 7:46] = 4
        self.experience_map[6:79, 60:84] = 4

    def update_fused_belief(self, exp_weight=0.9):
        """
        Fuse current online belief with experiential map.
        exp_weight in [0,1]:
            1   -> only current belief
            0   -> only experiential map
        """
        self.fused_belief = (
            exp_weight * self.belief + (1-exp_weight) * self.experience_map
        ).astype(np.float32)

    def update_belief_from_observation(
        self,
        robot_grid_pos,
        sensor_range=8,
        p_free=0.1,
        p_occ=0.9,
        noise_std=0.1
    ):

        gx, gy = robot_grid_pos

        yy, xx = np.indices((self.height, self.width))

        dist = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)

        observed_mask = dist <= sensor_range

        normalized_dist = np.clip(dist / sensor_range, 0.0, 1.0)

        # Confidence decays with distance
        new_confidence = 1.0 - normalized_dist
        # new_confidence[~observed_mask] = 0.0

        # Observation model
        # occ_observation = 0.5 + new_confidence * (p_occ - 0.5)
        # free_observation = 0.5 + new_confidence * (p_free - 0.5)
        occ_observation = np.ones_like(new_confidence, dtype=float) * p_occ
        free_observation = np.ones_like(new_confidence, dtype=float) * p_free

        # Gaussian noise
        occ_observation += np.random.normal(0, noise_std, new_confidence.shape) * new_confidence**2
        free_observation += np.random.normal(0, noise_std, new_confidence.shape) * new_confidence**2

        occ_observation = np.clip(occ_observation, 0, 1.0)
        free_observation = np.clip(free_observation, 0.0, 1.0)

        observation = self.belief.copy()

        occ_mask = np.logical_and(observed_mask, self.occupancy == 1)
        free_mask = np.logical_and(observed_mask, self.occupancy == 0)

        observation[occ_mask] = occ_observation[occ_mask]
        observation[free_mask] = free_observation[free_mask]

        # ==========================================
        # Confidence-weighted belief fusion
        # ==========================================

        old_belief = self.belief
        # old_conf = self.confidence

        alpha = new_confidence**2

        fused_belief = (1-alpha) * old_belief + alpha * observation

        # fused_confidence = np.maximum(old_conf, new_confidence)

        update_mask = observed_mask

        self.belief[update_mask] = fused_belief[update_mask]
        # self.confidence[update_mask] = fused_confidence[update_mask]

        self.compute_entropy_map()

    def compute_entropy_map(self):
        """
        Binary entropy:
            H(p) = -p log p - (1-p) log (1-p)
        """
        eps = 1e-6
        p = np.clip(self.belief, eps, 1.0 - eps)

        entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

        max_entropy = np.max(entropy)
        if max_entropy > 0:
            entropy = entropy / max_entropy

        self.entropy = entropy.astype(np.float32)
        self.uncertainty = self.entropy

    def update_dynamic_obstacles(self, moving_obstacles):
        self.dynamic_occupancy.fill(0)

        yy, xx = np.indices((self.height, self.width))

        for obs in moving_obstacles:
            ox, oy = obs.grid_position()
            mask = (xx - ox) ** 2 + (yy - oy) ** 2 <= obs.radius ** 2
            self.dynamic_occupancy[mask] = 1

        self.refresh_occupancy()

    def world_to_grid(self, x, y):
        gx = int(round(x / self.resolution))
        gy = int(round(y / self.resolution))
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution
        y = gy * self.resolution
        return x, y

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    def is_free(self, gx, gy):
        return self.in_bounds(gx, gy) and self.occupancy[gy, gx] == 0

    def create_demo_map(self):
        """
        Build a demo map with corridors and obstacles.
        """
        # border walls
        # self.add_rectangle_obstacle(0, 0, self.width - 1, 1)
        # self.add_rectangle_obstacle(0, self.height - 2, self.width - 1, self.height - 1)
        # self.add_rectangle_obstacle(0, 0, 1, self.height - 1)
        # self.add_rectangle_obstacle(self.width - 2, 0, self.width - 1, self.height - 1)

        # internal obstacles
        self.add_rectangle_obstacle(10, 8, 12, 10)
        self.add_rectangle_obstacle(22, 2, 24, 4)
        self.add_rectangle_obstacle(30, 18, 32, 20)
        self.add_rectangle_obstacle(42, 5, 44, 7)
        self.add_rectangle_obstacle(55, 25, 57, 27)

        # small blocks
        # self.add_rectangle_obstacle(18, 30, 24, 34)
        # self.add_rectangle_obstacle(49, 30, 54, 34)

    def create_mall_map(self):
        # Boundary
        self.add_rectangle_obstacle(0, 0, self.width - 1, 1)
        self.add_rectangle_obstacle(0, self.height - 2, self.width - 1, self.height - 1)
        self.add_rectangle_obstacle(0, 0, 1, self.height - 1)
        self.add_rectangle_obstacle(self.width - 2, 0, self.width - 1, self.height - 1)

        # Shelves
        shelf_width = 6
        shelf_height = 16
        aisle_gap_x = 8
        aisle_gap_y = 10

        x_positions = [10, 24, 38, 62, 76]
        y_positions = [8, 34, 60]

        for x in x_positions:
            for y in y_positions:
                # Asle
                if 45 <= x <= 55:
                    continue
                self.add_rectangle_obstacle(
                    x, y,
                    x + shelf_width - 1,
                    y + shelf_height - 1
                )

    def plot_layer(self, data, title="Layer", cmap="viridis", ax=None, start=None, goal=None, mask=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(data, cmap=cmap, origin="lower")
        if mask:
            obs = np.ma.masked_where(self.occupancy == 0, self.occupancy)
            ax.imshow(obs, cmap="gray", origin="lower", alpha=0.3)

        if start is not None:
            ax.plot(start[0], start[1], "go", markersize=8, label="Start")
        if goal is not None:
            ax.plot(goal[0], goal[1], "bo", markersize=8, label="Goal")

        ax.set_title(title)
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax)
        return ax

    def plot_map(self, ax=None, show_uncertainty=True, start=None, goal=None, show_colorbar=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if show_uncertainty:
            im = ax.imshow(self.entropy, cmap="YlOrRd", origin="lower", alpha=0.75)
            if show_colorbar:
                plt.colorbar(im, ax=ax)
        else:
            ax.imshow(np.zeros_like(self.original), cmap="gray", origin="lower", alpha=0.0)

        obs = np.ma.masked_where(self.occupancy == 0, self.occupancy)
        ax.imshow(obs, cmap="gray", origin="lower", alpha=0.9)

        static_obs = np.ma.masked_where(self.static_occupancy == 0, self.static_occupancy)
        ax.imshow(static_obs, cmap="gray", origin="lower", alpha=0.9)

        dynamic_obs = np.ma.masked_where(self.dynamic_occupancy == 0, self.dynamic_occupancy)
        ax.imshow(dynamic_obs, cmap="Blues", origin="lower", alpha=0.9)

        if start is not None:
            ax.plot(start[0], start[1], "go", markersize=8, label="Start")
        if goal is not None:
            ax.plot(goal[0], goal[1], "bo", markersize=8, label="Goal")

        ax.set_title("Entropy Map + Occupancy")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")                                      
        ax.legend(loc="upper right")
        return ax
    
    def plot_experience_map(self, ax=None, start=None, goal=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(self.experience_map, cmap="Purples", origin="lower", alpha=0.85)
        static_obs = np.ma.masked_where(self.static_occupancy == 0, self.static_occupancy)
        ax.imshow(static_obs, cmap="gray", origin="lower", alpha=0.8)

        if start is not None:
            ax.plot(start[0], start[1], "go", markersize=8, label="Start")
        if goal is not None:
            ax.plot(goal[0], goal[1], "bo", markersize=8, label="Goal")

        ax.set_title("Experiential Risk Map")
        ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax)
        return ax
    

class MovingObstacle:
    def __init__(self, x, y, speed, axis="x", x_min=None, x_max=None, y_min=None, y_max=None, radius=2):
        self.x = float(x)
        self.y = float(y)
        self.speed = float(speed)
        self.axis = axis
        self.direction = 1
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.radius = radius

    def step(self, env):
        if self.axis == "x":
            self.x += self.direction * self.speed
            if self.x <= self.x_min or self.x >= self.x_max:
                self.direction *= -1
                self.x = np.clip(self.x, self.x_min, self.x_max)

        elif self.axis == "y":
            self.y += self.direction * self.speed
            if self.y <= self.y_min or self.y >= self.y_max:
                self.direction *= -1
                self.y = np.clip(self.y, self.y_min, self.y_max)

    def grid_position(self):
        return int(round(self.x)), int(round(self.y))
    

if __name__=='__main__':
    env = GridEnvironment(width=100, height=100, resolution=0.25, robot_radius=0.25)
    env.create_mall_map()
    env.inflate_obstacles()
    env.build_belief_map(observed_region="none", p_free=0.1, p_occ=0.9)
    env.build_experience_map()
    env.update_fused_belief(exp_weight=0.35)
    env.plot_experience_map()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ground truth
    env.plot_layer(
        env.occupancy,
        title="Ground Truth Map",
        cmap="gray_r",
        ax=axes[0]
    )

    # Belief map
    env.plot_layer(
        env.belief,
        title="Belief Map (P occupied)",
        cmap="viridis",
        ax=axes[1]
    )

    # Entropy map
    env.plot_layer(
        env.entropy,
        title="Entropy Map (Uncertainty)",
        cmap="YlOrRd",
        ax=axes[2]
    )

    plt.tight_layout()
    plt.show()
