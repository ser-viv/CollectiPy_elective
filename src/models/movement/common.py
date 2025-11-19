# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Fabio Oddi
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

def apply_motion_state(agent) -> None:
    """Translate the discrete motion state into velocity commands."""
    linear = 0.0
    angular = 0.0
    if agent.motion == agent.FORWARD:
        linear = agent.max_absolute_velocity
    elif agent.motion == agent.LEFT:
        angular = agent.max_angular_velocity
    elif agent.motion == agent.RIGHT:
        angular = -agent.max_angular_velocity
    # stop -> (0,0)
    agent.linear_velocity_cmd = linear
    agent.angular_velocity_cmd = angular
