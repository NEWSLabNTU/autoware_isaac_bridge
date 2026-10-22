"""Traffic signals as Autoware's V2X input (carla-scenario-bridge roadmap 044).

The host (csb's world server) holds each signal's colour, as SSv2 commands it; `EgoRos`
publishes them on `/perception/traffic_light_recognition/external/traffic_signals`, the
input `traffic_light_arbiter` takes beside its own recognition, as acb does on CARLA: one
`TrafficLightGroup` per Lanelet2 regulatory element, one solid circle at confidence 1 (the
simulator's own state, not a recognition); a lamp that shows nothing is UNKNOWN at 0.
"""

TOPIC = "/perception/traffic_light_recognition/external/traffic_signals"

# autoware_perception_msgs/TrafficLightElement's constants
UNKNOWN, RED, AMBER, GREEN = 0, 1, 2, 3
CIRCLE = 1
SOLID_ON = 2


def element(colour):
    """`(color, shape, status, confidence)` for a colour name: OFF, RED, AMBER or GREEN."""
    c = {"RED": RED, "AMBER": AMBER, "GREEN": GREEN}.get(colour)
    if c is None:
        return UNKNOWN, UNKNOWN, UNKNOWN, 0.0
    return c, CIRCLE, SOLID_ON, 1.0
