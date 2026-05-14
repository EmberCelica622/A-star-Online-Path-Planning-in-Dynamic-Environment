import numpy as np
import matplotlib.pyplot as plt
import os

from env import GridEnvironment
from planner import AStarPlanner, prune_path, smooth_path, compute_path_metrics
from controller import DifferentialDriveRobot, TrajectoryTracker, build_reference_trajectory


def plot_belief_and_entropy(env, start, goal):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    env.plot_layer(
        1-env.original,
        title="Ground Truth Occupancy",
        cmap="gray",
        ax=axes[0],
        start=start,
        goal=goal,
        mask=True
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


def plot_online_step(
    env,
    start,
    goal,
    curr,
    raw_path,
    pruned_path,
    smooth_grid_path,
    executed_path,
    step_idx,
    fig=None,
    ax=None,
    show_plot=True,
    save_plot=False,
    save_dir="online_plots",
    pause_time=0.3
):
    """
    Update online planning visualization in the same figure window.
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))

    ax.clear()

    env.plot_map(ax=ax, show_uncertainty=True, start=start, goal=goal, show_colorbar= False)

    if raw_path is not None and len(raw_path) > 0:
        raw = np.array(raw_path)
        ax.plot(raw[:, 0], raw[:, 1], "c--", linewidth=1.5, label="Current A* Path")

    if pruned_path is not None and len(pruned_path) > 0:
        pruned = np.array(pruned_path)
        ax.plot(pruned[:, 0], pruned[:, 1], "m.-", linewidth=2.0, label="Pruned Path")

    if smooth_grid_path is not None and len(smooth_grid_path) > 0:
        smooth = np.array(smooth_grid_path)
        ax.plot(smooth[:, 0], smooth[:, 1], "b-", linewidth=2.5, label="Smoothed Local Path")

    if executed_path is not None and len(executed_path) > 0:
        executed = np.array(executed_path)
        ax.plot(executed[:, 0], executed[:, 1], "g-", linewidth=2.5, label="Executed Path")

    ax.plot(curr[0], curr[1], "ro", markersize=8, label="Current Robot Position")

    ax.set_title(f"Online Planning Step {step_idx}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"online_step_{step_idx:03d}.png")
        fig.savefig(filename, dpi=200, bbox_inches="tight")

    if show_plot:
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(pause_time)

    return fig, ax


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
    if errors is not None:
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
    env.build_belief_map(observed_region="none", p_free=0.1, p_occ=0.9)

    # Start/goal in grid coordinates
    start = (4, 4)
    goal = (55, 35)
    sensor_range = 10
    # env.update_belief_from_observation(start, sensor_range=sensor_range)

    # Visualization parameters
    plot_belief_and_entropy(env, start, goal)
    save_dir = 'online_plots'
    show_online_plot = True
    save_online_plot = True
    fig_online, ax_online = None, None

    if show_online_plot:
        plt.ion()
        fig_online, ax_online = plt.subplots(figsize=(9, 7))


    # Build robot dynamics
    x0, y0, theta0 = start[0], start[1], 0.0
    robot = DifferentialDriveRobot(x0, y0, theta0)
    tracker = TrajectoryTracker(
        k_rho=1.5,
        k_alpha=5.0,
        k_beta=-1.2, 
        v_max=0.9,
        w_max=2.5
    )

    # --------------------------------------------------
    # 2. Online risk-aware A* planning
    # --------------------------------------------------
    planner = AStarPlanner(env, lambda_uncertainty=2.5, lambda_belief=3.0, diagonal=True)

    not_finish = True
    curr = start
    i = 0
    horizon = int(np.ceil(sensor_range*0.4))

    executed_path = [start]
    max_iter = 100
    goal_tolerance = 2.0

    while not_finish and i < max_iter:
        env.update_belief_from_observation(curr, sensor_range=sensor_range)

        full_path = planner.plan(curr, goal)
        if full_path is None:
            print(f"No feasible path found at posistion {curr}")
            return
        
        local_horizon = min(horizon, len(full_path))
        raw_path = full_path[:local_horizon]

        for pos in raw_path[1:]:
            env.update_belief_from_observation(pos, sensor_range=sensor_range)

        metrics_raw = compute_path_metrics(env, raw_path)

        # --------------------------------------------------
        # 3. Path post-processing
        # --------------------------------------------------
        pruned_path = prune_path(raw_path)
        smooth_grid_path = smooth_path(pruned_path, num_points=20*len(raw_path), smoothing=0.5)

        # --------------------------------------------------
        # 4. Build reference trajectory
        # --------------------------------------------------
        # ref_traj = build_reference_trajectory(smooth_grid_path, env_resolution=env.resolution)

        # x0 = start[0] * env.resolution
        # y0 = start[1] * env.resolution
        # theta0 = 0.0
    

        # --------------------------------------------------
        # 5. Simulate tracking
        # --------------------------------------------------
        # dt = 0.05
        # errors = tracker.follow_trajectory(robot, ref_traj, dt=dt)
        errors = None
        curr = raw_path[-1]     # discrete excecution
        executed_path.extend(raw_path[1:])


        # --------------------------------------------------
        # 6. Print check point summary
        # --------------------------------------------------
        
        print("================ Planning Results ================")
        print(f"Check point: {i}")
        print(f"Raw path points: {len(raw_path)}")
        print(f"Pruned path points: {len(pruned_path)}")
        print(f"Smoothed trajectory points: {len(smooth_grid_path)}")
        print(f"Raw path length (grid units): {metrics_raw['length']:.3f}")
        print(f"Raw cumulative entropy cost: {metrics_raw['entropy_cost']:.3f}")
        # print(f"Mean tracking error [m]: {np.mean(errors):.4f}")
        # print(f"Max tracking error [m]: {np.max(errors):.4f}")

        # Goal check
        if np.linalg.norm(np.array(curr) - np.array(goal)) <= goal_tolerance:
            print("Goal reached.")
            not_finish = False

        # --------------------------------------------------
        # 7. Visualization
        # --------------------------------------------------
        fig_online, ax_online = plot_online_step(
            env=env,
            start=start,
            goal=goal,
            curr=curr,
            raw_path=raw_path,
            pruned_path=pruned_path,
            smooth_grid_path=smooth_grid_path,
            executed_path=executed_path,
            step_idx=i,
            fig=fig_online,
            ax=ax_online,
            pause_time=0.4,
            show_plot=show_online_plot,
            save_plot=save_online_plot,
            save_dir=save_dir)

        i += 1


    if show_online_plot:
            plt.ioff()
            plt.show()
    # plot_results(env, raw_path, pruned_path, smooth_grid_path, robot, errors, start, goal)
    plot_belief_and_entropy(env, start, goal)


if __name__ == "__main__":
    main()