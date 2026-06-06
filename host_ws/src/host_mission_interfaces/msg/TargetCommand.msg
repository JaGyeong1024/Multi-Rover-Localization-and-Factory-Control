# host -> rover boundary contract. Host sends a TARGET + mode; the rover owns
# routing (over the shared lane graph) and control. Deadman: rover STOPs if this
# command goes stale (header.stamp older than its timeout).
std_msgs/Header header             # stamp = issue time (deadman source); frame_id = uwb_global
string robot                       # leader | follower
uint32 cmd_id                      # monotonic; rover echoes in NavFeedback (supersede detection)
string mode                        # GOTO | HOLD | STOP | RELEASE
int32 target_node                  # graph node id to reach (-1 = use target_coord)
geometry_msgs/Point target_coord   # alternative target; rover snaps to nearest lane node
bool stop_and_hold                 # on arrival, hold position until next command
float64 speed_cap                  # m/s cap (0 = rover default)
