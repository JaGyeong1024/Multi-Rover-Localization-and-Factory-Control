# Host's world-model snapshot of one robot (dashboard + introspection).
string name                        # leader | follower
geometry_msgs/Pose pose            # last /{robot}/global/position
int32 lane_node                    # nearest graph node to pose
string health                      # GOOD | DEGRADED | LOST | UNKNOWN
string current_task                # "" | mission item id
string state                       # per-robot mission state (role-specific):
                                   #  leader:   IDLE|MOVING|AT_PICKUP|PICKING|RETURNING|PAUSED|FAULT|ABORTED
                                   #  follower: IDLE|MOVING|AT_PICKUP|RECEIVING|DELIVERING|UNLOADING|...
