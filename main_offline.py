import numpy as np
import matplotlib.pyplot as plt

from env import GridEnvironment
from planner import AStarPlanner, prune_path, smooth_path, compute_path_metrics
from controller import DifferentialDriveRobot, TrajectoryTracker, build_reference_trajectory


def plot_belief_and_entropy(env, start, goal):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    env.plot_layer(
        env.occupancy,
        title="Ground Truth Occupancy",
        cmap="gray_r",
        ax=axes[0],
        start=start,
        goal=goal
    )

    env.plot_layer(
        env.belief,
        title="Belief Map: P(Occupied)",
        cmap="viridis",
        ax=axes[1],
        start=start,
        goal=goal
    )

    env.plot_layer(
        env.entropy,
        title="Entropy Map",
        cmap="YlOrRd",
        ax=axes[2],
        start=start,
        goal=goal
    )

    plt.tight_layout()
    plt.show()


def plot_results(env, raw_path, pruned_path, smooth_grid_path, robot, errors, start, goal):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    env.plot_map(ax=ax, show_uncertainty=True, start=start, goal=goal)

    if raw_path is not None:
        raw = np.array(raw_path)
        ax.plot(raw[:, 0], raw[:, 1], "c--", linewidth=1.5, label="A* Raw Path")

    if pruned_path is not None:
        pruned = np.array(pruned_path)
        ax.plot(pruned[:, 0], pruned[:, 1], "m.-", linewidth=2, label="Pruned Path")

    if smooth_grid_path is not None:
        ax.plot(smooth_grid_path[:, 0], smooth_grid_path[:, 1], "b-", linewidth=2.5, label="Smoothed Path")

    robot_hist = np.array(robot.history)
    robot_grid = robot_hist[:, :2] / env.resolution
    ax.plot(robot_grid[:, 0], robot_grid[:, 1], "g-", linewidth=2.0, label="Executed Trajectory")

    ax.legend(loc="upper right")
    ax.set_title("Planning and Tracking")

    ax2 = axes[1]
    ax2.plot(errors, linewidth=2)
    ax2.set_title("Tracking Error")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Position Error [m]")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # --------------------------------------------------
    # 1. Build environment
    # --------------------------------------------------
    env = GridEnvironment(width=60, height=40, resolution=0.2, robot_radius=0.2)
    env.create_demo_map()
    env.inflate_obstacles()

    # Build entropy-based uncertainty
    env.build_belief_map(observed_region="partial", p_free=0.1, p_occ=0.9)

    # Start/goal in grid coordinates
    start = (4, 4)
    goal = (55, 35)

    # Visualize occupancy / belief / entropy
    plot_belief_and_entropy(env, start, goal)

    # --------------------------------------------------
    # 2. Risk-aware A* planning
    # --------------------------------------------------
    planner = AStarPlanner(env, lambda_uncertainty=2.5, diagonal=True)
    raw_path = planner.plan(start, goal)

    if raw_path is None:
        print("No feasible path found.")
        return

    metrics_raw = compute_path_metrics(env, raw_path)

    # --------------------------------------------------
    # 3. Path post-processing
    # --------------------------------------------------
    pruned_path = prune_path(raw_path)
    smooth_grid_path = smooth_path(pruned_path, num_points=350, smoothing=0.5)

    # --------------------------------------------------
    # 4. Build reference trajectory
    # --------------------------------------------------
    ref_traj = build_reference_trajectory(smooth_grid_path, env_resolution=env.resolution)

    x0 = start[0] * env.resolution
    y0 = start[1] * env.resolution
    theta0 = 0.0

    robot = DifferentialDriveRobot(x0, y0, theta0)
    tracker = TrajectoryTracker(
        k_rho=1.5,
        k_alpha=5.0,
        k_beta=-1.2,
        v_max=0.9,
        w_max=2.5
    )

    # --------------------------------------------------
    # 5. Simulate tracking
    # --------------------------------------------------
    dt = 0.05
    errors = tracker.follow_trajectory(robot, ref_traj, dt=dt)

    # --------------------------------------------------
    # 6. Print summary
    # --------------------------------------------------
    print("========== Planning Results ==========")
    print(f"Raw path points: {len(raw_path)}")
    print(f"Pruned path points: {len(pruned_path)}")
    print(f"Smoothed trajectory points: {len(smooth_grid_path)}")
    print(f"Raw path length (grid units): {metrics_raw['length']:.3f}")
    print(f"Raw cumulative entropy cost: {metrics_raw['entropy_cost']:.3f}")
    print(f"Mean tracking error [m]: {np.mean(errors):.4f}")
    print(f"Max tracking error [m]: {np.max(errors):.4f}")

    # --------------------------------------------------
    # 7. Visualization
    # --------------------------------------------------
    plot_results(env, raw_path, pruned_path, smooth_grid_path, robot, errors, start, goal)


if __name__ == "__main__":
    main()