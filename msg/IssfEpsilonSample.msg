# Each message summarizes one fixed controller-rate window.  The controller
# evaluates all ticks, but only publishes the worst samples so calibration
# bags stay small without losing disturbance peaks.
bool has_normal
float64 normal_adverse_projection

bool has_braking
float64 braking_adverse_projection
