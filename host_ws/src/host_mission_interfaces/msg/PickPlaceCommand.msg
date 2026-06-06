# host -> picker (leader / teammate A). Request a pick&place once the robots are
# positioned at the part's lane node. The arm grasps the part and places it on the
# carrier. Host waits for PickPlaceStatus DONE/FAILED.
std_msgs/Header header
string robot                       # picker, e.g. "leader"
uint32 cmd_id                      # monotonic per request
string part_id                     # "gear_01"
geometry_msgs/Point part_coord     # uwb_global (z ignored)
string target_robot                # carrier to place onto, e.g. "follower"
