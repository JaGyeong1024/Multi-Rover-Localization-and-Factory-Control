# One mission item = one part fetch, with its derived lane stop node.
string id                          # e.g. "fetch_gear_01"
Part part
int32 lane_node                    # graph node id nearest to part.coord (the stop)
float64 x                          # lane-stop coord (node x) — convenience for dashboard
float64 y
float64 yaw
string status                      # pending | active | done | failed
float64 ready_in                   # s until this item's shelf is off cooldown (0 = ready now)
builtin_interfaces/Time enqueued_at
