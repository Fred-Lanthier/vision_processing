# Optimization Notes: green_cube_feeding_casf.launch Pipeline

Date: 2026-05-12  
Scope: every node launched by `launch/green_cube_feeding_casf.launch`

---

## Pipeline-wide: PointCloud2 publishing via `.tolist()`

**Affected files (all the same fix):**
- `scripts/perception_processing_node.py:68`
- `scripts/point_cloud_projector_node.py:247, 267`
- `scripts/condition_pcd_from_perception.py:91`
- `scripts/persistent_cloud_node.py:261, 282`

`pc2.create_cloud_xyz32` expects an iterable of `(x, y, z)` tuples. Calling it
with `.tolist()` first converts the entire numpy array to nested Python lists —
one Python `float` object per coordinate. For a 5000-point cloud that is 15 000
allocations per publish call, repeated at 30 Hz.

**Fix — build the message directly from the numpy buffer:**

```python
from sensor_msgs.msg import PointCloud2, PointField

def make_cloud_msg(header, pts_np):
    pts = pts_np.astype(np.float32)
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(pts)
    msg.is_bigendian = False
    msg.point_step = 12          # 3 × float32
    msg.row_step = 12 * len(pts)
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    msg.data = pts.tobytes()
    msg.is_dense = True
    return msg
```

No list allocation, ~100× faster for large clouds. This helper can be shared
across all five nodes.

---

## `nominal_fork_pose_follower.py` — critical, 150 Hz loop

### Issue 1 — MoveIt IK service at 150 Hz (`compute_command`, line ~334)

With `use_planner_joint_trajectory = false` (set in the launch at line 88),
`compute_ik` calls the `/compute_ik` ROS service every cycle. Each call
involves a Python IPC round-trip, MoveIt planning-scene locking, and a
KDL/TRAC-IK solver invocation. At 150 Hz this saturates the ROS service
thread. `fast_ik_module` (pybind11 Pinocchio, already used by
`casf_generative_node_PC.py`) does the same IK in ~0.5 ms with no IPC.

**Fix — replace the service call with a direct C++ call:**

```python
# In __init__, load once:
from vision_processing import fast_ik_module
self.ik_solver = fast_ik_module.FastIK(urdf_path, "panda_TCP")

# In compute_command, replace compute_ik(...):
se3_tcp = ...   # 4×4 numpy array built from pose_tcp
q_seed = np.concatenate([current_joints, [0.0, 0.0]])
q9 = self.ik_solver.solve(se3_tcp, q_seed)
ik_sol = q9[:7] if q9 is not None else None
```

### Issue 2 — TF lookup at 150 Hz (`current_fork_transform`, line ~202)

`tf_listener.lookupTransform('world', tcp_frame, rospy.Time(0))` acquires a
lock on the C++ TF tree at every cycle. `robot_state_publisher` already
broadcasts FK-derived transforms from `/joint_states`, so the transform is
always available — but the lookup itself has non-trivial overhead at high rate.

Since `current_joints` (from `/joint_states`) is already stored, FK can be
computed directly, eliminating the TF dependency entirely:

```python
# In __init__, load once (same URDFLayer used by casf_generative_node_PC):
from third_party.RDF.urdf_layer import URDFLayer
self.robot_layer = URDFLayer(urdf_path, device='cpu', ...)

# In current_fork_transform(), replace lookupTransform:
q = torch.tensor(np.concatenate([self.current_joints, [0,0]]),
                 dtype=torch.float32).unsqueeze(0)
poses = self.robot_layer._native_forward_kinematics(q)
T = poses['fork_tip'][0].numpy()   # 4×4 in world frame
```

This makes `current_fork_transform` deterministic, latency-free, and
independent of TF tree availability.

---

## `persistent_cloud_node.py` — high impact, 30 Hz

### Issue — Python dict voxel grid

`_add_to_grid` iterates over points in a Python `for` loop with individual
`dict.get()` / `dict.__setitem__` calls. `_check_freespace_grid` projects all
voxel centers at once (good) but then loops `for i in np.where(in_fov)[0]` in
Python to update per-voxel dicts (bad). At 50 000 committed voxels this
becomes a CPU bottleneck competing with the 30 Hz perception pipeline.

`scripts/PersistentCloud.py` already implements the same voxel concept using
**sorted numpy arrays with `np.searchsorted`** — fully vectorised. The
`WorldCloud` class there is a drop-in backend for `persistent_cloud_node.py`.

**Fix — replace the dict store with `WorldCloud`:**

```python
from PersistentCloud import WorldCloud

# In __init__:
self._obs_world  = WorldCloud(voxel_size=self.obs_voxel_size,
                               max_age=self.obs_free_thresh * 50,
                               max_points=self.obs_max_voxels)
self._tgt_world  = WorldCloud(voxel_size=self.tgt_voxel_size,
                               max_age=self.tgt_free_thresh * 50,
                               max_points=self.tgt_max_voxels)

# In _obs_cb, replace _add_to_grid + _check_freespace_grid + _evict_grid:
self._obs_world.integrate(pts)   # one vectorised call
```

The freespace carving loop (`_check_freespace_grid`) also needs to be
vectorised — replace the `for i in np.where(in_fov)[0]` loop with numpy
boolean-indexed array operations on the hit/free count arrays.

---

## `cbf_safety_node_Bernstein.py` — safety-critical fix + minor

### Issue 1 — Hardcoded dt spike cap (line 310–311)

```python
# Current — wrong at rate_hz = 50 (nominal dt = 20 ms)
if dt > 0.1:
    dt = 0.0066   # hardcoded 1/150
```

`dt_fixed = 1.0 / self.rate_hz` is computed at line 295 but never used.
A GPU spike that makes dt exceed 100 ms clamps it to 6.66 ms — a 15×
underestimate. The velocity command `dq_nom = (q_next_nom - last_q_safe) / dt`
then becomes 15× too large, causing a violent lurch or joint-limit trip.

**Fix — one line:**

```python
if dt > 3.0 * dt_fixed:   # cap at 3× nominal (60 ms at 50 Hz)
    dt = dt_fixed
```

### Issue 2 — Visualization marker at every cycle (line ~239)

`publish_safe_trajectory_marker` computes 16 forward kinematics evaluations
and constructs 17 RViz `Marker` objects every cycle at 50 Hz. This is pure
visualization overhead.

**Fix — throttle to every 5th cycle:**

```python
self._viz_counter = getattr(self, '_viz_counter', 0) + 1
if self._viz_counter % 5 == 0:
    self.publish_safe_trajectory_marker(self.current_q, q_safe)
```

### Issue 3 — Self-filter threshold removes real obstacle points (line 365)

```python
not_self = sdf_body > 0.005   # removes points within 5 mm of robot surface
```

If the robot physically enters an obstacle (CBF violated due to joint error or
external push), obstacle points within 5 mm of any body link disappear from
`pts_inside`. The CBF then sees no obstacle and cannot push back out —
self-reinforcing failure. Depth-noise self-hits have SDF ≤ 0 (inside the mesh);
real contact points have SDF ≥ 0.

**Fix — filter only points geometrically inside the robot mesh:**

```python
not_self = sdf_body > -0.003   # keep surface contacts, remove only self-interior
```

Same fix applies to `casf_generative_node_PC.py:495`.

---

## `casf_generative_node_PC.py` — medium impact, 3 Hz

### Issue 1 — TCP pose conversion loop (`generate_safe_trajectory`, line ~540)

```python
# Current — H=16 Python iterations
for k in range(H):
    T_world_fork = se3_fork_np[k]
    se3_tcp_list.append(self.fork_se3_to_tcp_se3(T_world_fork))
```

`T_tcp_fork_tip_inv` is a fixed constant. The entire batch is a single
batched matrix multiply:

```python
# In __init__, precompute once on GPU:
self.T_tcp_fork_tip_inv_gpu = torch.from_numpy(
    self.T_tcp_fork_tip_inv).float().to(self.device)

# In generate_safe_trajectory, replace the loop:
# se3_fork_gpu: [H, 4, 4]
se3_tcp_gpu = se3_fork_gpu @ self.T_tcp_fork_tip_inv_gpu   # [H, 4, 4]
se3_tcp_list = list(se3_tcp_gpu.cpu().numpy())             # FastIK still needs list
```

### Issue 2 — `TemporalEnsembler.get_ensembled_trajectory` (line ~37)

Nested Python loops: `O(buffer_size × H)` with list appends and `np.average`
called H=16 times per planning step. The buffer is maxlen=3.

**Fix — vectorise over the buffer dimension:**

```python
def get_ensembled_trajectory(self, num_steps):
    if not self.buffer:
        return None
    preds   = np.stack([p for _, p in self.buffer])           # [B, H, D]
    ages    = np.array([self.step_counter - ts - 1
                        for ts, _ in self.buffer], dtype=np.float32)  # [B]
    weights = np.exp(-0.5 * ages)[:, None, None]              # [B, 1, 1]
    weights /= weights.sum(axis=0, keepdims=True)
    return (preds[:, :num_steps] * weights).sum(axis=0)       # [H, D]
```

Same pattern applies to the identical copy in `casf_generative_node.py`.

### Issue 3 — Random sampling in callbacks uses CPU numpy

`obstacle_callback:289` and `cloud_callback:301` use `np.random.choice` on
CPU after the data is already on GPU. Replace with `torch.randperm` to stay
on GPU and avoid the round-trip:

```python
# Instead of: idx = np.random.choice(len(points), N, replace=False)
idx = torch.randperm(len(points_gpu), device=self.device)[:N]
points_gpu = points_gpu[idx]
```

---

## `perception_processing_node.py` — medium impact, 30 Hz

### Issue — CPU/GPU round-trip for Radius Outlier Removal (line 62–63)

```python
pts_np = pts_inside.cpu().numpy()                                      # GPU→CPU
pts_filtered_np = fast_perception_module.radius_outlier_removal(...)   # C++ PCL
# result is CPU numpy; GPU upload happens implicitly when CBF reads it
```

GPU-native ROR avoids the transfer:

```python
# For N < ~5000 (post-crop typical size) this is faster than the CPU round-trip
dist = torch.cdist(pts_inside, pts_inside)          # [N, N]
neighbor_count = (dist < 0.05).sum(dim=1) - 1      # exclude self
pts_filtered = pts_inside[neighbor_count >= 5]
pts_filtered_np = pts_filtered.cpu().numpy()
```

For larger N (> 10k), keep the C++ path or chunk the `torch.cdist` call.

---

## `point_cloud_projector_node.py` — low impact, 30 Hz

### Issue — Full-image obstacle mask allocation (`sync_cb`, line 250–251)

```python
obs_mask = np.ones_like(mask_cv) * 255   # allocates HxW uint8
obs_mask[mask_cv > 0] = 0
```

**Fix — in-place inversion, no new allocation:**

```python
obs_mask = ((mask_cv == 0).view(np.uint8)) * 255
```

---

## `sam3_sam2_node.py` — no action needed

Already correctly split into a RealSense/ROS producer thread and a GPU
inference thread (`ai_worker`) with a single-frame lock. The bottleneck is
SAM2 inference which is irreducible without a smaller model.

---

## Summary Table

| Node | Rate | Issue | Impact | Effort |
|------|------|-------|--------|--------|
| `nominal_fork_pose_follower` | 150 Hz | MoveIt IK service → `fast_ik_module` | **Critical** | Medium |
| `nominal_fork_pose_follower` | 150 Hz | TF lookup → direct FK | **Critical** | Medium |
| `cbf_safety_node_Bernstein` | 50 Hz | `dt = 0.0066` hardcoded cap | **Safety** | Trivial |
| `cbf_safety_node_Bernstein` | 50 Hz | Self-filter SDF threshold too aggressive | **Safety** | Trivial |
| All 5 publishing nodes | 30–150 Hz | `.tolist()` in PointCloud2 publish | High | Low |
| `persistent_cloud_node` | 30 Hz | Python dict → `WorldCloud` numpy backend | High | Medium |
| `casf_generative_node_PC` | 3 Hz | TCP pose loop → batched GPU matmul | Medium | Low |
| `casf_generative_node_PC` | 3 Hz | `TemporalEnsembler` vectorisation | Medium | Low |
| `casf_generative_node_PC` | 3 Hz | CPU `np.random.choice` → GPU `torch.randperm` | Medium | Trivial |
| `perception_processing_node` | 30 Hz | GPU→CPU→GPU ROR round-trip | Medium | Low |
| `cbf_safety_node_Bernstein` | 50 Hz | Visualization marker every cycle | Low | Trivial |
| `point_cloud_projector_node` | 30 Hz | obs_mask full-image allocation | Low | Trivial |
| `sam3_sam2_node` | 30 Hz | Already optimal | — | — |
