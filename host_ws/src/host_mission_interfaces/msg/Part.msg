# A part to be fetched. Coordinate-only; the lane stop node is derived by projection.
string id                     # e.g. "gear_01"
geometry_msgs/Point coord     # uwb_global meters (z ignored)
string state                  # QUEUED | ASSIGNED | PICKED | LOADED | DELIVERED
