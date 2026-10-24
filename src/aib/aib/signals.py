"""Traffic signals as Autoware's V2X input (scenario-world-bridge roadmap 044).

The host (swb's world server) holds each signal's colour, as SSv2 commands it; `EgoRos`
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


# A regulatory element whose lights disagree shows the most restrictive known colour (acb's
# `group_colour`, swb_isaac_world's `signals.RESTRICTIVENESS`); OFF is "unknown", the least.
RESTRICTIVENESS = {"OFF": 0, "GREEN": 1, "AMBER": 2, "RED": 3}


class Cycle:
    """Traffic lights cycling as CARLA's do, for the stand-alone runner (no SSv2 commands them).

    CARLA turns a junction's lights green one at a time: each green for `green_s`, then amber
    for `amber_s`, while the rest of its group is red (10 s and 3 s are CARLA's defaults).
    The groups come from the map dir's `carla/signal_groups.yaml`
    (scenario-world-bridge's `scripts/generate_signal_groups.py`, from CARLA's own groups);
    which regulatory elements each light is, from its signal table (`traffic_lights.yaml`).
    Every group starts its cycle at sim time 0."""

    def __init__(self, groups, elements, green_s=10.0, amber_s=3.0):
        """`groups`: `[[OpenDRIVE id, ...]]`, each in the order it turns green; `elements`:
        OpenDRIVE id -> `[regulatory element id, ...]`."""
        if green_s <= 0 or amber_s < 0:
            raise ValueError(f"green {green_s} s, amber {amber_s} s: green must be > 0")
        self.groups = [[str(od) for od in g] for g in groups if g]
        self.elements = {str(od): [int(e) for e in els] for od, els in elements.items()}
        self.green_s, self.amber_s = float(green_s), float(amber_s)
        grouped = {od for g in self.groups for od in g}
        # A light of the table in no group would never be published: it cycles on its own.
        self.groups += [[od] for od in sorted(set(self.elements) - grouped)]

    @classmethod
    def from_map_dir(cls, map_dir, green_s=10.0, amber_s=3.0):
        """The cycle for a map dir: its signal table and `carla/signal_groups.yaml`."""
        from pathlib import Path

        import yaml

        d = Path(map_dir)
        tables = [d / "traffic_lights.yaml", d / "carla" / "traffic_lights.yaml"]
        table = next((p for p in tables if p.exists()), None)
        if table is None:
            raise FileNotFoundError(f"no signal table in {d} (traffic_lights.yaml or "
                                    "carla/traffic_lights.yaml)")
        groups_file = d / "carla" / "signal_groups.yaml"
        if not groups_file.exists():
            raise FileNotFoundError(
                f"no {groups_file}: write it from CARLA holding the town, with "
                "scenario-world-bridge's scripts/generate_signal_groups.py <Town> --map "
                f"{d} --port <CARLA port>")
        elements = {}
        for s in (yaml.safe_load(table.read_text()) or {}).get("signals", []):
            if s.get("regulatory_element_ids"):
                elements.setdefault(str(s["opendrive_id"]), []).extend(
                    s["regulatory_element_ids"])
        groups = [g["lights"] for g in
                  (yaml.safe_load(groups_file.read_text()) or {}).get("groups", [])]
        return cls(groups, elements, green_s, amber_s)

    def colours(self, t):
        """OpenDRIVE id -> colour name at sim time `t` (s)."""
        turn = self.green_s + self.amber_s
        out = {}
        for g in self.groups:
            phase = t % (turn * len(g))
            current = int(phase // turn)
            for i, od in enumerate(g):
                if i != current:
                    out[od] = "RED"
                else:
                    out[od] = "GREEN" if phase - i * turn < self.green_s else "AMBER"
        return out

    def groups_at(self, t):
        """`[(regulatory element id, colour name)]` at sim time `t`, what `EgoRos.
        publish_signals` takes: one per element, the most restrictive of its lights."""
        by_element = {}
        for od, colour in self.colours(t).items():
            for e in self.elements.get(od, []):
                held = by_element.get(e)
                if held is None or RESTRICTIVENESS[colour] > RESTRICTIVENESS[held]:
                    by_element[e] = colour
        return sorted(by_element.items())
