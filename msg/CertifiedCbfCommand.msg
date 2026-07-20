# Atomic output of one GPU CBF solve.  The velocity and certificate must stay
# together: applying a velocity with h/gradients from another solve invalidates
# the sampled-data reflex check.
std_msgs/Header header
uint32 selection_generation
uint32 row_count
float64[7] solve_joint_position
float64[7] safe_joint_velocity
float64[] barrier_value
float64[] environment_hdot
# Row-major [row_count, 7].
float64[] barrier_gradient
