"""A progressive bump stop at the suspension's real travel, as a force on the body (scensim
issue 077, after roadmap 066 R2). Pure.

PhysX's sweep wheel query climbs a sharp curb the raycast cannot (issue 077), but at the real
travel its rigid limit throws the car off a 10 cm cleat at speed (issue 029). With PhysX's
travel made longer, its frame raised so the droop stays the real one, the limit lies where
nothing on the road reaches it; what stops the wheel at the real travel is this: over the last
`bump` m of it, a force rising with the square of the compression to `rate` (N/m) at full
compression, and a damper while engaged, pushing the body up at the wheel's attachment.

PhysX reports each wheel's jounce (compression from full droop) after a step; the host applies
the force before the next. The sweep's jounce is not smooth: meeting a step's face, it jumps
by most of the step's height within one PhysX step (8 cm at 2 m/s on a 10 cm curb). So the
stop acts on the jounce as read, not on one extrapolated from the last two, and its damper
sees a compression speed no faster than `v_max`: from a jump that speed would be metres per
second, and the force would throw the car. The force is capped at `cap` times the stop's
force at full compression.
"""
from dataclasses import dataclass


@dataclass
class BumpStop:
    """One vehicle's stops, one per wheel (the same at every wheel)."""

    travel: float  # the real travel, m: full compression is jounce == travel
    bump: float = 0.03  # engagement before full compression, m
    rate: float = 0.0  # the stop's rate at full compression, N/m
    damping: float = 0.0  # N s/m while engaged, on the compression speed
    v_max: float = 0.5  # m/s: the compression speed the damper sees, at most
    cap: float = 2.0  # the force's cap, times rate * bump

    def __post_init__(self):
        if not 0.0 < self.bump < self.travel:
            raise ValueError(f"bump stop engagement {self.bump} m: between 0 and the travel "
                             f"({self.travel} m)")
        if self.rate <= 0.0:
            raise ValueError(f"bump stop rate {self.rate} N/m: positive")
        self.prev = None

    def reset(self):
        self.prev = None

    def force(self, j, v):
        """The stop's force (N, pushing the body up, never pulling) at compression `j` (m from
        full droop) and compression speed `v` (m/s)."""
        over = j - (self.travel - self.bump)
        if over <= 0.0:
            return 0.0
        # rate/bump * over^2 reaches rate * bump at full compression; its slope there is
        # 2 rate: the stop stiffens as it goes
        v = min(max(v, -self.v_max), self.v_max)
        f = self.rate / self.bump * over * over + self.damping * v
        return min(max(f, 0.0), self.cap * self.rate * self.bump)

    def step(self, jounce, dt):
        """Each wheel's force for the coming step, from the jounces PhysX reported after the
        last one (None for a wheel PhysX has none for)."""
        prev = self.prev or [None] * len(jounce)
        out = []
        for j, jp in zip(jounce, prev):
            if j is None:
                out.append(0.0)
                continue
            v = 0.0 if jp is None or dt <= 0 else (j - jp) / dt
            out.append(self.force(j, v))
        self.prev = list(jounce)
        return out
