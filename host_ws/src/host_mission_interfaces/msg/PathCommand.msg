# host -> rover. The host plans; the rover only follows this path and stops by itself.
# Acceptance: take it if epoch is larger; with the same epoch take it if cmd_id >= current.
# Same cmd_id = keep-alive only (do not reload the path). Host republishes at 2 Hz.
# Freshness is judged by the rover's own receive time, not header.stamp.
std_msgs/Header header        # frame_id = uwb_global
string robot                  # leader | follower
uint32 epoch                  # host start time [s]; grows when the host restarts
uint32 cmd_id                 # bumps when the path or mode changes (within an epoch)
string mode                   # FOLLOW | STOP
geometry_msgs/Pose2D[] poses  # path points (x, y, theta) ~0.05 m apart; last = goal pose
float32 speed_cap             # m/s cap (0 = rover default)
float32 goal_pos_tol          # m
float32 goal_yaw_tol          # rad (0 = heading not checked)
bool pick_at_goal             # true -> report HOLDING on arrival (pick point); else PARKED
string lead_robot             # robot to keep a gap to ("" = solo)
float32 gap                   # distance to keep to lead_robot, measured along the path [m]
bool pivot_first              # rotate in place at poses[0] to pivot_heading before tracking (FOLLOW only)
float64 pivot_heading         # rad, uwb_global; only meaningful when pivot_first
bool fixture_ahead            # goal faces a known fixture/object: rover may skip the front stop on the final straight run-in
