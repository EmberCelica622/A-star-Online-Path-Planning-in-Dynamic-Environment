import heapq
import numpy as np
from scipy.interpolate import splprep, splev


class AStarPlanner:
    def __init__(self, env, lambda_uncertainty=2.5, lambda_belief=3.0, diagonal=True, fused_map=False):
        self.env = env
        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_belief = lambda_belief
        self.diagonal = diagonal
        self.fused_map = fused_map

        if diagonal:
            self.moves = [
                (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
                (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2)),
            ]
        else:
            self.moves = [
                (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)
            ]

    def heuristic(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def cost(self, current, nxt):
        """
        Risk-aware cost:
            movement cost
            + lambda_uncertainty * entropy cost
            + lambda_belief * belief cost
        """
        dx = nxt[0] - current[0]
        dy = nxt[1] - current[1]
        move_cost = abs(dx) + abs(dy)

        entropy_cost = self.env.entropy[nxt[1], nxt[0]]
        if self.fused_map:
            belief_cost = self.env.fused_belief[nxt[1],nxt[0]]
        else:
            belief_cost = self.env.belief[nxt[1], nxt[0]]

        return (
            move_cost
            + self.lambda_uncertainty * entropy_cost
            + self.lambda_belief * belief_cost
        )

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def plan(self, start, goal):
        if not self.env.is_free(*start):
            raise ValueError("Start is not in free space.")
        if not self.env.is_free(*goal):
            raise ValueError("Goal is not in free space.")

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}
        closed_set = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            if current in closed_set:
                continue
            closed_set.add(current)

            for dx, dy, _ in self.moves:
                nxt = (current[0] + dx, current[1] + dy)

                if not self.env.is_free(*nxt):
                    continue

                tentative_g = g_score[current] + self.cost(current, nxt)

                if nxt not in g_score or tentative_g < g_score[nxt]:
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    f_score[nxt] = tentative_g + self.heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f_score[nxt], nxt))

        return None


def prune_path(path, max_gap=2):
    if path is None or len(path) < 3:
        return path

    pruned = [path[0]]
    last_prune_index = 0

    for i in range(1, len(path) - 1):
        x0, y0 = pruned[-1]
        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y1)

        turning = v1[0] * v2[1] - v1[1] * v2[0] != 0
        too_far = (i-last_prune_index) > max_gap

        if turning or too_far:
            pruned.append(path[i])
            last_prune_index = i

    pruned.append(path[-1])
    return pruned


def smooth_path(path, num_points=300, smoothing=1.0):
    if path is None or len(path) < 3:
        return np.array(path, dtype=float)

    path = np.array(path, dtype=float)
    x = path[:, 0]
    y = path[:, 1]

    try:
        tck, _ = splprep([x, y], s=smoothing)
        u_fine = np.linspace(0, 1, num_points)
        x_s, y_s = splev(u_fine, tck)
        return np.vstack((x_s, y_s)).T
    except Exception:
        return path


def compute_path_metrics(env, path):
    if path is None or len(path) < 2:
        return {"length": np.inf, "entropy_cost": np.inf}

    length = 0.0
    entropy_cost = 0.0
    for i in range(1, len(path)):
        p0 = np.array(path[i - 1], dtype=float)
        p1 = np.array(path[i], dtype=float)
        length += np.linalg.norm(p1 - p0)
        gx, gy = int(round(path[i][0])), int(round(path[i][1]))
        entropy_cost += env.entropy[gy, gx] * np.linalg.norm(p1 - p0)

    return {"length": float(length), "entropy_cost": float(entropy_cost)}