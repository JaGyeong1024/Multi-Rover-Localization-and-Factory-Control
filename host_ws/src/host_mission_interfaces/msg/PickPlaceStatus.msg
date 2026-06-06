# picker (teammate A) -> host. pick&place progress. Drives the host's PICK_PLACE
# phase (DONE -> part loaded; FAILED -> item failed) and is shown by 관제 as the
# pick&place sub-state.
std_msgs/Header header
string robot                       # picker
uint32 cmd_id                      # echoes PickPlaceCommand.cmd_id
string part_id
string state                       # IDLE | APPROACH | PICKING | PLACING | DONE | FAILED
string detail
