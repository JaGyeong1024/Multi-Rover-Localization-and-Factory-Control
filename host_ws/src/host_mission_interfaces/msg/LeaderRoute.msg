# Leader's planned route over the shared lane graph. The LEADER publishes this on
# /leader/route; the HOST relays it (domain bridge leader 25 -> host 30 -> follower 73,
# host adds NO following logic). The FOLLOWER consumes it to platoon: it decides where to
# JOIN the shared path and where to BRANCH off to its own target node, then builds its own
# route. Host stays target-based (TargetCommand); this is the rover↔rover route share.
#
# node_ids index the SHARED lane graph (tools/path_editor graph.json / lane_graph_map.json)
# — host, leader and follower all share it, so coordinates are DERIVED from the graph, not
# carried here (avoids drift; keeps the message small). frame for derived coords = uwb_global.
std_msgs/Header header        # stamp = issue time (freshness / supersede); frame_id = uwb_global
string robot                  # whose route this is — "leader"
uint32 cmd_id                 # the TargetCommand.cmd_id this route fulfils (correlate / supersede)
int32 target_node             # route destination node (== TargetCommand.target_node)
int32[] node_ids              # ordered lane-graph node ids, current position -> target_node
