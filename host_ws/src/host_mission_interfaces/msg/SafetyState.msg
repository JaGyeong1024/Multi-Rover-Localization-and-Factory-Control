# rover -> host. What the rover safety gate is doing and why. Reasons are data, not states.
std_msgs/Header header
string robot
string state                  # OK | LIMITED | STOPPED
string[] reasons              # STALE_POSE STALE_CMD STALE_LEAD ESTOP NOT_ALIGNED LIDAR_STOP LIDAR_SLOW
                              # POSE_JUMP OFF_PATH STUCK QUALITY_LOW INPUT_TIMEOUT
float32 speed_scale           # 0..1 scale applied by the gate
float32 blocked_x             # LIDAR_STOP: estimated global position of the blocker (NaN if none)
float32 blocked_y
