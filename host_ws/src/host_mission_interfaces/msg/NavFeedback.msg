# rover -> host navigation progress (fast path for the host sequencer; the host
# also cross-checks arrival against /{robot}/global/position).
std_msgs/Header header
string robot
uint32 cmd_id                      # which TargetCommand this answers (match to trust)
string state                       # ROUTING | TRACKING | ARRIVED | HOLDING | STOPPED | IDLE | ERROR
int32 current_node                 # nearest graph node
float64 progress                   # 0..1 along the current route
string detail                      # free text (ERROR cause, etc.)
