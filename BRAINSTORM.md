# Brainstorm: Perception & CBF Architecture

Date: 2026-05-12  
Context: Wrist-camera assisted feeding pipeline (Panda + RealSense, ROS Noetic)

---

## The Yellow and Red Clouds — What They Are

Inside `test_novelty/cbf_safety_node_Bernstein.py`, the obstacle cloud from
`/perception/cleaned_obstacles` (or `/perception/persistent_obstacles`) goes
through three successive filters every CBF cycle:

1. **Workspace crop** — keep only points within 40 cm of the TCP (fork tip).
2. **Self-filter** — discard points on the robot body (Bernstein SDF < 5 mm on
   body links 0–8 only, not finger/fork links).
3. **Split into two visualisation layers**:
   - **Red** (`obs_global`, 100 pts) — the 100 survivors with the smallest SDF
     to any robot link. These are the most imminently dangerous points and they
     are the sole input to the QP solver.
   - **Yellow** — every valid point that passed steps 1–2 but did not make the
     top-100 cut. Visualised in RViz, never used by the CBF.

There is also a dedicated **fork-local pass** (`obs_fork`, 50 pts): the 50
points within 15 cm of the fork tip, fed into a second CUDA-graph QP pass.
This partially compensates for the body-biased top-100 selection but does not
fully solve it.

---

## Identified Problems

### P1 — No memory outside the camera field of view

The wrist camera has a narrow, ego-centric frustum. `perception_processing_node.py`
publishes a purely live, single-frame cloud at 30 Hz.
Once the arm moves toward the food, the camera pans away from the person, and
the person disappears from the obstacle map entirely — even though they are
still physically present. The CBF then has zero signal about the person and
will not deflect.

This is the central limitation of the architecture. All improvements below are
attempts to address it without requiring an additional fixed camera.

### P2 — `persistent_cloud_node.py` Python-dict performance

The existing node (`scripts/persistent_cloud_node.py`) has the right
architectural intent (voxel accumulation + freespace carving) but the inner
loops operate on Python `dict` objects one key at a time:

- `_add_to_grid` iterates over points in a Python `for` loop with individual
  `dict.get()` / `dict.__setitem__` calls.
- `_check_freespace_grid` builds a `keys_list`, projects all centers at once
  (good, vectorised numpy), but then loops over `np.where(in_fov)[0]` in
  Python to update individual dict entries (slow, O(N) Python overhead).

At 50 000 committed voxels this becomes the bottleneck on the CPU, competing
with the 30 Hz perception pipeline on the same core.

The `PersistentCloud.py` module (`scripts/PersistentCloud.py`, `WorldCloud`
class) already implements the same concept using **sorted numpy arrays** with
`np.searchsorted`-based binary merges — fully vectorised. The two
implementations should be unified: `persistent_cloud_node.py` should use
`WorldCloud` as its backend instead of raw Python dicts.

### P3 — Freespace carving is fragile for dynamic obstacles

The current freespace logic requires a per-voxel projection into the depth
image and a TF lookup at the depth timestamp. Several failure modes:

- **Fixed free threshold (4 frames at 30 Hz = 133 ms)**: a person moving at
  0.5 m/s leaves ghost voxels for 133 ms after they leave a position. During
  those 133 ms the CBF deflects away from where the person was, not where they
  are.
- **TF timestamp coupling**: if the TF tree lags or the depth-cloud timestamp
  pair is mismatched, `lookup_transform` raises and the whole freespace check
  is silently skipped. Committed voxels are never cleared.
- **Static and dynamic obstacles share one grid** with the same thresholds.
  A table voxel and a person voxel are treated identically. This is incorrect:
  the table never moves (high commit threshold is fine, slow freespace is fine),
  but a person can cross the workspace in under a second.

### P4 — Self-filter ingestion bug (critical)

Line 365 in `cbf_safety_node_Bernstein.py`:

```python
not_self = sdf_body > 0.005
pts_inside = pts_inside[not_self]
```

This removes **all** obstacle points within 5 mm of any body link surface. If
the robot physically enters an obstacle — CBF violation due to tracking error,
joint disturbance, or a person pushing the arm — the obstacle points inside
that 5 mm zone vanish from `pts_inside`. The CBF then receives zero signal
about the penetration and cannot push back out. The deeper the arm enters the
obstacle, the more obstacle points are filtered: the failure is
self-reinforcing.

The filter is intended to strip depth-noise self-hits (depth pixels that land
on the robot's own surface). Those returns have SDF < 0 (inside the mesh).
Real obstacle points at the robot surface have SDF ≥ 0 and should never be
removed.

### P5 — Body-biased top-100 red selection

The 100 red points are chosen globally by smallest SDF over all links. If
link1 or link2 (large, slow links near the base) are close to the table edge,
they can consume most of the 100 slots, leaving the fork tip — the most
safety-critical part during a feeding task — under-represented in the QP. The
`obs_fork` second pass partially compensates but it is a separate, independent
QP pass, not an integrated selection.

### P6 — dt cap hardcoded to 1/150 s (live bug)

In `cbf_safety_node_Bernstein.py`, line 295 computes `dt_fixed = 1.0 /
self.rate_hz` but never uses it. The spike cap on line 310-311 is:

```python
if dt > 0.1:
    dt = 0.0066   # hardcoded 1/150 — wrong at the current 50 Hz rate
```

The node actually runs at `rate_hz = 50` (20 ms nominal dt). A GPU spike
(thermal throttle, memory allocation, CUDA sync bubble) that causes dt to
exceed 100 ms will have it clamped to 6.66 ms — a 15× underestimate. The
velocity command `dq_nom = (q_next_nom - last_q_safe) / dt` then becomes 15×
too large, causing a violent lurch that saturates `max_joint_velocity` or
trips the Franka torque watchdog.

---

## Proposed Approaches

### A — GPU-native TSDF-style freespace integration

**Why this is better than the current approach:**

The current freespace check asks "is this stored voxel still observed?". This
requires projecting every stored voxel into the image and is bounded by the
number of stored voxels (up to 50 000).

Forward raycasting inverts the question: "what does each depth pixel tell us
about the world?". Every valid depth pixel simultaneously:
- Marks all voxels along its ray at distances < `depth - truncation` as **free**
- Marks the voxel at `depth ± truncation` as **occupied**

This is fully parallel in both directions. More importantly, when a person
moves, every depth pixel behind their old position immediately generates a
free signal for those voxels — dynamic removal happens in **1 frame**, not
after `free_thresh` frames. The freespace signal is geometrically correct
because it follows the actual ray, not a threshold comparison.

**Practical implementation at 2 cm voxel resolution:**

A 4 m × 3 m × 2 m workspace at 2 cm = 200 × 150 × 100 = 3 M voxels.
As an `int8` GPU tensor that is 3 MB — fits trivially in VRAM.
Indexing becomes direct tensor scatter, not hash lookup.

```python
# Pseudocode — fully vectorised, no Python loops
ray_dirs = precomputed_per_pixel_directions  # [H*W, 3], world frame
depth_flat = depth_image.flatten()           # [H*W]
valid = (depth_flat > 0.01) & (depth_flat < max_range)

# Step along each ray at voxel_size intervals
t = torch.arange(0, max_range, voxel_size, device=dev)  # [S]
# points: [H*W, S, 3] — all ray march points for all valid pixels
points = cam_origin + t[None, :, None] * ray_dirs[:, None, :]

# Free mask: step depth < pixel depth - truncation
step_depth = (points - cam_origin).norm(dim=-1)  # [H*W, S]
is_free = step_depth < (depth_flat[:, None] - truncation)
is_occ  = (step_depth - depth_flat[:, None]).abs() < truncation

# Convert to voxel indices, scatter into occupancy grid
vox = (points / voxel_size).long()
occupancy_grid.scatter_(free_indices, -1)  # decrement
occupancy_grid.scatter_(occ_indices,  +1)  # increment
```

**Memory note:** marching all 640×480 pixels at 100 steps = 30 M points per
frame. To keep this tractable, march only pixels with valid depth (typically
60-80% of the image) and cap the march depth to the actual measured depth
(no need to march past the surface). In practice this is 10-20 M operations
per frame, well within GPU throughput at 30 Hz.

### B — Separate static and dynamic layers with independent decay

**Why one grid is wrong:**

Static (table, bowl, wall) and dynamic (person) obstacles have fundamentally
different temporal properties. Treating them identically means either the
static layer flickers (too-aggressive freespace) or the dynamic layer has
ghost voxels (too-conservative freespace).

**Two-layer design:**

| Layer | Contents | Commit threshold | Free threshold | Age timeout |
|-------|----------|-----------------|----------------|-------------|
| Static | Table, bowl, walls, tray | 5 frames | 15 frames | never (or ~10 s) |
| Dynamic | Person, moving arm | 2 frames | 2 frames | 1–2 s (~60 frames) |

The age timeout on the dynamic layer is the key: even if the person goes
completely outside the FOV, their voxels expire after `max_age` frames. The
static layer has no timeout (or a very long one) because the table will always
be there.

**Separation heuristic without running SAM3/SAM2 on everything:**

You do not need segmentation to separate the layers. Use **motion as the
discriminant**:

- Points that are consistently present across many consecutive frames →
  static layer (table etc.)
- Points that appear or disappear within a few frames, or that shift between
  consecutive frames → dynamic layer

A simple implementation: any voxel with `hit_count > N_static` (e.g., 30
consecutive observations over 1 s) is promoted to the static layer and no
longer subject to aggressive freespace carving. New or recently-appeared
voxels remain in the dynamic layer until they earn that count.

This requires no additional neural network or SAM query.

### C — Fix the self-filter threshold (P4)

Change the self-filter to remove only points geometrically **inside** the
robot mesh (true depth-noise self-hits), not points that are close but outside:

```python
# Current — removes real obstacle points at the robot surface
not_self = sdf_body > 0.005

# Proposed — only removes points inside the mesh (SDF < 0)
# Allow 3 mm noise tolerance around the surface
not_self = sdf_body > -0.003
```

Rationale: depth-noise self-hits from the robot's own surface produce depth
readings on or inside the mesh, giving SDF values ≤ 0. Real obstacle points
touching the robot surface have SDF ≥ 0 by definition. The current 5 mm
positive threshold is the source of the ingestion blind-spot.

### D — Fix the dt spike cap (P6)

One-line fix in `cbf_safety_node_Bernstein.py`:

```python
# Current — wrong, hardcoded to 1/150 s even though rate_hz = 50
if dt > 0.1:
    dt = 0.0066

# Fix — use the already-computed dt_fixed
if dt > 3.0 * dt_fixed:   # cap at 3x nominal (60 ms at 50 Hz)
    dt = dt_fixed
```

Using `3 * dt_fixed` instead of an absolute value makes the cap
rate-independent. The tighter bound (60 ms instead of 100 ms) also means a
smaller maximum velocity error on a spike.

### E — Per-zone budgeted obstacle selection (P5)

Replace the single global top-100 with explicit zone budgets:

| Zone | Budget | Selection criterion |
|------|--------|---------------------|
| Fork tip (r < 15 cm) | 50 pts | Closest by SDF to end-effector links |
| Mid-arm (15–40 cm) | 30 pts | Closest by SDF to links 3–6 |
| Global (any range) | 20 pts | Globally closest to any link |

This guarantees the fork tip always receives 50 dedicated CBF slots regardless
of what link1 is doing near the table. The dual-graph structure already in the
node (`graph_global` + `graph_fork`) maps naturally onto this: `obs_fork`
becomes the 50-pt fork-zone set and `obs_global` the remaining 50 pts from
mid-arm + global zones combined.

---

## Summary of Why These Approaches Fit the Use Case

The feeding task has a specific structure that makes the above choices
tractable:

1. **The workspace is small and mostly static.** A single dense 3 MB GPU
   tensor covers it entirely. There is no need for an unbounded map.
2. **There is exactly one significant dynamic obstacle: the person.** The
   separation heuristic (static vs dynamic by motion/age) does not need to be
   general — it just needs to distinguish the person from the table.
3. **The wrist camera always sees in the direction of motion.** When the arm
   moves toward the food, the camera is looking at the food area. The person
   is likely visible when the arm is near the start/approach, which is exactly
   when the FOV coverage is highest. The combination of short dynamic-layer
   age timeout (voxels expire in ~1 s) and static-layer persistence gives the
   right memory profile for this task.
4. **The CBF is the last line of defense, not the primary planner.** It does
   not need to reason about the full world — it needs to avoid collision in the
   next 100–200 ms. A 50 Hz CBF with a 1–2 s dynamic map gives 50–100 frames
   of look-back, which covers the full arm movement duration (~3–5 s).
