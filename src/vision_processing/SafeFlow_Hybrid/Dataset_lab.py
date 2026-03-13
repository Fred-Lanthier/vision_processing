"""
Maze trajectory data generation — inspired by SafeFlow (Dai et al., 2025).

Pipeline:
  1. Define a 2D grid maze as an occupancy map
  2. Find a reference path through it (A* on the grid)
  3. Use MPPI (Model Predictive Path Integral) to generate many smooth
     trajectories that follow the reference path
  4. Optionally add elliptical obstacles (unseen during training)
  5. Plot everything

This is a simplified, standalone version — no MuJoCo, no D4RL dependency.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from collections import deque
import heapq
import os

# ============================================================
# 1. MAZE DEFINITION
# ============================================================
# The maze is a binary grid: 0 = free, 1 = wall.
# We define it manually to look like the SafeFlow paper maze.
# Each cell represents a square region in continuous space.

def create_maze():
    """
    Create a maze similar to the one in SafeFlow Fig. 2.
    The maze goes from (0,0) to (COLS, ROWS) in continuous space.
    Robot navigates from lower-left to lower-right.
    The bottom path is blocked so trajectories must go through the interior.
    """
    # 13 rows x 13 cols maze
    # 1 = wall, 0 = free space
    maze = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1],  # top
        [1,0,0,0,0,0,1,0,0,0,0,0,1],
        [1,0,1,1,1,0,1,0,1,1,1,0,1],
        [1,0,1,0,0,0,0,0,0,0,1,0,1],
        [1,0,1,0,1,1,1,1,1,0,1,0,1],
        [1,0,0,0,1,0,0,0,1,0,0,0,1],
        [1,1,1,0,1,0,1,0,1,0,1,1,1],  # middle
        [1,0,0,0,0,0,1,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,1,0,0,0,1],
        [1,0,1,0,1,1,1,1,1,0,1,0,1],
        [1,0,1,0,0,0,0,0,0,0,1,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1],  # bottom
    ], dtype=np.float32)

    # Flip vertically so row 0 is at the bottom in the array
    maze = maze[::-1]
    return maze


# ============================================================
# 2. A* PATH FINDER (reference path on the grid)
# ============================================================

def astar(maze, start_cell, goal_cell):
    """
    Simple A* on a grid. Returns a list of (row, col) cells.
    start_cell, goal_cell: (row, col) in the maze array.
    """
    rows, cols = maze.shape
    open_set = []
    heapq.heappush(open_set, (0, start_cell))
    came_from = {}
    g_score = {start_cell: 0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_cell:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = current[0]+dr, current[1]+dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0:
                tentative_g = g_score[current] + 1
                neighbor = (nr, nc)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal_cell)
                    heapq.heappush(open_set, (f, neighbor))

    return None  # no path found


def cells_to_continuous(cell_path, cell_size=1.0):
    """
    Convert grid cell indices to continuous (x, y) coordinates.
    Center of cell (row, col) → (col + 0.5, row + 0.5) * cell_size.
    """
    pts = []
    for (r, c) in cell_path:
        x = (c + 0.5) * cell_size
        y = (r + 0.5) * cell_size
        pts.append([x, y])
    return np.array(pts)


def smooth_path(path, sigma=1.0):
    """Apply Gaussian smoothing to a path for continuity."""
    from scipy.ndimage import gaussian_filter1d
    smoothed = np.copy(path)
    smoothed[:, 0] = gaussian_filter1d(path[:, 0], sigma=sigma)
    smoothed[:, 1] = gaussian_filter1d(path[:, 1], sigma=sigma)
    return smoothed


def interpolate_path(path, num_points=100):
    """Resample a path to have a fixed number of evenly spaced points."""
    # Compute cumulative arc length
    diffs = np.diff(path, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    # Interpolate at evenly spaced arc-length values
    target_lengths = np.linspace(0, total_length, num_points)
    new_path = np.zeros((num_points, 2))
    new_path[:, 0] = np.interp(target_lengths, cum_length, path[:, 0])
    new_path[:, 1] = np.interp(target_lengths, cum_length, path[:, 1])
    return new_path


# ============================================================
# 3. MPPI — Model Predictive Path Integral (simplified)
# ============================================================
# The idea: at each step, sample K random control sequences,
# simulate them forward, score them based on how close they
# stay to the reference path, pick a weighted average.
# This produces diverse but smooth trajectories.

def is_collision(x, y, maze, cell_size=1.0):
    """Check if continuous point (x,y) is inside a wall."""
    col = int(x / cell_size)
    row = int(y / cell_size)
    rows, cols = maze.shape
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return True
    return maze[row, col] == 1


def mppi_generate_trajectory(
    reference_path,
    maze,
    cell_size=1.0,
    num_samples=200,       # K: number of rollouts per step
    horizon=10,            # planning horizon (steps ahead)
    noise_sigma=0.3,       # std of random perturbations
    lambda_cost=1.0,       # temperature for weighting
    dt=0.1,                # integration timestep
    num_steps=None,        # total trajectory length
    collision_penalty=100.0,
    reference_weight=5.0,
    velocity_weight=0.1,
):
    """
    Generate one trajectory using MPPI around a reference path.

    The "dynamics" are simple: x_{t+1} = x_t + u_t * dt
    (single integrator in 2D).
    """
    if num_steps is None:
        num_steps = len(reference_path)

    # Start at the first point of the reference
    start = reference_path[0].copy()
    goal = reference_path[-1].copy()

    trajectory = [start.copy()]
    x = start.copy()

    # Mean control sequence (warm start)
    u_mean = np.zeros((horizon, 2))

    for step in range(num_steps - 1):
        # Progress ratio to pick nearest reference point
        progress = step / max(num_steps - 1, 1)
        ref_idx = min(int(progress * (len(reference_path) - 1)),
                      len(reference_path) - 1)

        # Sample K perturbations around the current mean
        noise = np.random.randn(num_samples, horizon, 2) * noise_sigma
        U_samples = u_mean[None, :, :] + noise  # (K, horizon, 2)

        # Simulate each rollout
        costs = np.zeros(num_samples)
        for k in range(num_samples):
            x_sim = x.copy()
            for h in range(horizon):
                x_sim = x_sim + U_samples[k, h] * dt

                # Reference tracking cost
                future_ref_idx = min(ref_idx + h + 1, len(reference_path) - 1)
                ref_pt = reference_path[future_ref_idx]
                costs[k] += reference_weight * np.linalg.norm(x_sim - ref_pt)**2

                # Collision cost
                if is_collision(x_sim[0], x_sim[1], maze, cell_size):
                    costs[k] += collision_penalty

                # Control effort
                costs[k] += velocity_weight * np.linalg.norm(U_samples[k, h])**2

        # MPPI weighting: softmin
        costs -= np.min(costs)  # numerical stability
        weights = np.exp(-costs / lambda_cost)
        weights /= np.sum(weights) + 1e-10

        # Weighted average control
        u_mean = np.einsum('k,khi->hi', weights, U_samples)

        # Apply the first control
        x = x + u_mean[0] * dt

        # Shift the mean sequence for warm start
        u_mean = np.roll(u_mean, -1, axis=0)
        u_mean[-1] = 0.0

        trajectory.append(x.copy())

    return np.array(trajectory)


# ============================================================
# 4. ELLIPTICAL OBSTACLES (added after training, like the paper)
# ============================================================

def ellipse_obstacle(center, a, b):
    """
    Returns a barrier function h(s) for an elliptical obstacle.
    h(s) > 0 means safe (outside the ellipse).

    h(s) = (s - c)^T Q (s - c) - 1
    with Q = diag(1/a^2, 1/b^2).
    """
    c = np.array(center)
    Q = np.diag([1.0/a**2, 1.0/b**2])

    def h(s):
        diff = s - c
        return diff @ Q @ diff - 1.0

    return h, c, a, b


# ============================================================
# 5. PLOTTING
# ============================================================

def plot_maze_with_trajectories(
    maze, trajectories, cell_size=1.0,
    obstacles=None, title="Dataset", figsize=(5, 4.5),
    save_path=None
):
    """
    Plot the maze as filled blue cells, trajectories as colored lines,
    and optional elliptical obstacles.
    """
    rows, cols = maze.shape
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw wall cells as blue rectangles
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1:
                rect = patches.Rectangle(
                    (c * cell_size, r * cell_size),
                    cell_size, cell_size,
                    linewidth=0,
                    facecolor='steelblue',
                    alpha=0.9
                )
                ax.add_patch(rect)

    # Draw elliptical obstacles
    if obstacles is not None:
        for (_, center, a, b) in obstacles:
            ellipse = patches.Ellipse(
                center, 2*a, 2*b,
                linewidth=1.5,
                edgecolor='navy',
                facecolor='steelblue',
                alpha=0.7
            )
            ax.add_patch(ellipse)

    # Draw trajectories
    cmap = plt.cm.plasma
    n_traj = len(trajectories)
    for i, traj in enumerate(trajectories):
        color = cmap(i / max(n_traj - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1],
                linewidth=0.8, alpha=0.7, color=color)

    # Mark start and goal with dots
    if len(trajectories) > 0:
        ax.plot(trajectories[0][0, 0], trajectories[0][0, 1],
                'go', markersize=6, label='Start')
        ax.plot(trajectories[0][-1, 0], trajectories[0][-1, 1],
                'r*', markersize=8, label='Goal')

    ax.set_xlim(0, cols * cell_size)
    ax.set_ylim(0, rows * cell_size)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig, ax


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(42)

    # --- Step 1: Create maze ---
    maze = create_maze()
    cell_size = 1.0
    rows, cols = maze.shape
    print(f"Maze size: {rows} x {cols}")

    # --- Step 2: Find reference path with A* ---
    # Start: lower-left free cell, Goal: lower-right free cell
    # In our maze, row 1 is the bottom corridor
    start_cell = (1, 1)   # (row, col)
    goal_cell = (1, 11)

    cell_path = astar(maze, start_cell, goal_cell)
    if cell_path is None:
        print("ERROR: No path found in maze!")
        return

    print(f"A* path length: {len(cell_path)} cells")

    # Convert to continuous coordinates, smooth, and interpolate
    ref_path = cells_to_continuous(cell_path, cell_size)
    ref_path = smooth_path(ref_path, sigma=1.5)
    ref_path = interpolate_path(ref_path, num_points=80)

    # --- Step 3: Generate trajectories with MPPI ---
    # To get diversity (multiple routes through the maze), we generate
    # multiple reference paths using A* with different waypoints,
    # then run MPPI around each one.
    num_trajectories = 30
    print(f"Generating {num_trajectories} trajectories with MPPI...")

    # Find alternative routes by forcing waypoints at different maze locations
    # These are free cells at different heights in the maze
    waypoints_options = [
        [],                          # direct A* path
        [(3, 5)],                    # go through col 5, row 3
        [(7, 5)],                    # go through middle
        [(9, 3)],                    # go up-left
        [(7, 7)],                    # go through center
        [(9, 9)],                    # go up-right
        [(5, 3), (9, 5)],            # double waypoint: up-left then up
        [(5, 9), (9, 7)],            # double waypoint: right then up
    ]

    # Generate multiple reference paths
    ref_paths = []
    for wps in waypoints_options:
        # Chain: start → wp1 → wp2 → ... → goal
        full_cells = []
        prev = start_cell
        valid = True
        for wp in wps:
            sub = astar(maze, prev, wp)
            if sub is None:
                valid = False
                break
            full_cells.extend(sub[:-1])
            prev = wp
        if not valid:
            continue
        sub = astar(maze, prev, goal_cell)
        if sub is None:
            continue
        full_cells.extend(sub)

        ref = cells_to_continuous(full_cells, cell_size)
        ref = smooth_path(ref, sigma=1.5)
        ref = interpolate_path(ref, num_points=80)
        ref_paths.append(ref)

    if not ref_paths:
        ref_paths = [ref_path]  # fallback

    print(f"  Found {len(ref_paths)} distinct reference routes")

    trajectories = []
    for i in range(num_trajectories):
        # Pick a reference path (cycle through them)
        rp = ref_paths[i % len(ref_paths)]

        traj = mppi_generate_trajectory(
            reference_path=rp,
            maze=maze,
            cell_size=cell_size,
            num_samples=150,
            horizon=8,
            noise_sigma=0.25 + np.random.rand() * 0.2,
            lambda_cost=0.8,
            dt=0.15,
            num_steps=80,
            collision_penalty=80.0,
            reference_weight=3.0,
            velocity_weight=0.05,
        )
        trajectories.append(traj)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{num_trajectories} done")

    # --- Step 4: Define test-time elliptical obstacles (like the paper) ---
    # These are NOT seen during "training" — used only for evaluation
    obs1 = ellipse_obstacle(center=[4.0, 5.5], a=1.8, b=1.0)
    obs2 = ellipse_obstacle(center=[9.0, 4.0], a=1.5, b=0.8)
    obs3 = ellipse_obstacle(center=[7.0, 8.5], a=0.8, b=1.2)
    obstacles = [obs1, obs2, obs3]

    # --- Step 5: Plot ---
    # Figure A: Dataset trajectories (no obstacles shown — training view)
    fig1, ax1 = plot_maze_with_trajectories(
        maze, trajectories, cell_size,
        obstacles=None,
        title="(a) Dataset — Training Trajectories",
        save_path="maze_dataset.png"
    )

    # Figure B: Same trajectories with test-time obstacles overlaid
    fig2, ax2 = plot_maze_with_trajectories(
        maze, trajectories, cell_size,
        obstacles=obstacles,
        title="(b) Dataset + Unseen Obstacles",
        save_path="maze_with_obstacles.png"
    )

    # --- Save trajectory data as .npz for later use (e.g., training FM) ---
    traj_array = np.array(trajectories)  # shape: (N, T, 2)
    np.savez(
        "maze_trajectories.npz",
        trajectories=traj_array,
        reference_path=ref_path,
        maze=maze,
    )
    print(f"\nSaved {num_trajectories} trajectories to maze_trajectories.npz")
    print(f"Trajectory shape: {traj_array.shape}")

    plt.show()


if __name__ == "__main__":
    main()