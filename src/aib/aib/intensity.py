"""A lidar's intensity from the vehicle spec's reflectance table (scensim roadmap 084, Track A).

intensity = reflectance[class] x |cos incidence|, clamped to 0-255: the rule CARLA's semantic
lidar follows with the same table (sim_vehicle_spec, a lidar's `carla.intensity`; acb applies
it there). On Isaac the class comes from the prim each ray hit: the RTX annotator gives each
point a stable object id and the hit's normal (with the OmniLidar's auxOutputType FULL); the
StableIdMap annotator maps an id to its prim path; the prim, or its nearest ancestor, carries
SimReady's `semantics:labels:wikidata_qcode`, which QCODE_TAG turns into CARLA's tag.

Pure functions here (numpy only); the Isaac side is aib.isaac.sensors.Lidar.
"""

import numpy as np

QCODE_ATTR = "semantics:labels:wikidata_qcode"

# SimReady's Wikidata Q-codes, as CARLA's USD export labels its assets, to CARLA 0.9.16's
# semantic tags (carla.CityObjectLabel names, the names the spec's reflectance takes). Read off
# the exports on 2026-11-18, each code checked against the prims it labels: Town01 (17 codes
# on 4434 prims; 5 meshes unlabelled) and the 190 vehicle, walker and prop assets the world
# server dresses its entities in (all labelled but vehicle.ford.crown), so its NPCs, walkers,
# props and the ego's body need no label of their own.
QCODE_TAG = {
    "Q34442": "Roads",            # road (and the manhole covers in it)
    "Q177749": "Sidewalks",       # sidewalk
    "Q1392287": "RoadLines",      # road surface marking, crosswalks
    "Q186131": "Terrain",         # terrain
    "Q36133": "Ground",           # ground
    "Q41176": "Buildings",        # building
    "Q42948": "Walls",            # wall
    "Q148571": "Fences",          # fence
    "Q1188866": "GuardRail",      # guard rail
    "Q2180428": "Poles",          # street light (CARLA tags lamp posts Poles)
    "Q8004": "TrafficLight",      # traffic light
    "Q170285": "TrafficSigns",    # traffic sign
    "Q187997": "Vegetation",      # trees, foliage
    "Q283": "Water",              # water
    "Q12280": "Bridge",           # bridge
    "Q2302426": "Static",         # street furniture: benches, bins, kennels
    "Q55107540": "Static",        # decals: dirt, debris
    "Q1420": "Car",               # motor car
    "Q43193": "Truck",            # truck (and vans)
    "Q5638": "Bus",               # bus
    "Q34493": "Motorcycle",       # motorcycle
    "Q11442": "Bicycle",          # bicycle
    "Q11998775": "Rider",         # a two-wheeler's rider (CARLA's bike and motorbike exports)
    "Q221488": "Pedestrians",     # pedestrian (the walker exports)
}


class Table:
    """A lidar's reflectance table (the rig's `isaac.intensity`: `default` and `reflectance` by
    CARLA tag)."""

    def __init__(self, spec):
        self.default = float(spec.get("default", 0))
        self.by_tag = {str(k): float(v) for k, v in (spec.get("reflectance") or {}).items()}

    def of_qcode(self, qcode):
        """The reflectance of a class, by its Q-code (None: unlabelled)."""
        tag = QCODE_TAG.get(qcode) if qcode else None
        return self.by_tag.get(tag, self.default) if tag else self.default


def object_ids(raw):
    """The RTX annotator's per-point object ids (16 bytes each, little-endian), as an (n, 2)
    uint64 array: low and high halves, comparable and hashable row by row."""
    a = np.ascontiguousarray(raw)
    return a.view(np.uint8).reshape(-1, 16).view("<u8").reshape(-1, 2)


def stable_id_map(raw):
    """StableIdMap's buffer -> {(low, high): prim path}. Its layout (Isaac's own decoder,
    isaacsim.sensors.rtx LidarRtx.decode_stable_id_mapping): a uint32 entry count at the end;
    each entry six uint32 -- the 128-bit id, the label's length and its offset."""
    b = bytes(np.ascontiguousarray(raw).view(np.uint8))
    if len(b) < 4:
        return {}
    n = int.from_bytes(b[-4:], "little")
    entries = np.frombuffer(b[: n * 24], "<u4").reshape(-1, 6)
    out = {}
    for e in entries:
        lo = int(e[0]) | (int(e[1]) << 32)
        hi = int(e[2]) | (int(e[3]) << 32)
        out[(lo, hi)] = b[int(e[5]): int(e[5]) + int(e[4])].decode("utf8").rstrip("\x00 ")
    return out


def incidence_cos(xyz, normals):
    """|cos| of each ray (sensor origin -> point, in `xyz`'s frame) against its hit normal
    (the same frame); 0 where either is degenerate."""
    p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    lp = np.linalg.norm(p, axis=1)
    ln = np.linalg.norm(n, axis=1)
    ok = (lp > 1e-9) & (ln > 1e-9)
    c = np.zeros(len(p))
    c[ok] = np.abs(np.einsum("ij,ij->i", p[ok], n[ok])) / (lp[ok] * ln[ok])
    return np.clip(c, 0.0, 1.0)


def intensity(reflectance, cos):
    """uint8 intensity: reflectance x |cos incidence|, rounded, clamped to 0-255."""
    v = np.rint(np.asarray(reflectance, dtype=np.float64) * np.asarray(cos, dtype=np.float64))
    return np.clip(v, 0, 255).astype(np.uint8)


def per_point_reflectance(ids, reflectance_of_ids):
    """Each point's reflectance from its id ((n, 2) uint64, `object_ids`). Each distinct id is
    looked up once: `reflectance_of_ids` takes the list of distinct (low, high) ids and returns
    their reflectances in order (a dict {(low, high): reflectance} works too)."""
    if len(ids) == 0:
        return np.zeros(0)
    keyed = np.ascontiguousarray(ids, dtype=np.uint64).view([("lo", "<u8"), ("hi", "<u8")]).ravel()
    uniq, inverse = np.unique(keyed, return_inverse=True)
    keys = [(int(u["lo"]), int(u["hi"])) for u in uniq]
    if callable(reflectance_of_ids):
        vals = np.asarray(reflectance_of_ids(keys), dtype=np.float64)
    else:
        vals = np.array([reflectance_of_ids[k] for k in keys], dtype=np.float64)
    return vals[inverse.reshape(-1)]


_MIX = np.uint64(0x9E3779B97F4A7C15)


def id_hash(ids):
    """One uint64 per (low, high) object id ((n, 2) uint64): low ^ (high x golden-ratio), for
    a vectorized lookup. Ids are a prim's stable id; two colliding would share a class, and at
    30k ids in a town the chance is ~1e-11."""
    ids = np.ascontiguousarray(ids, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return ids[:, 0] ^ (ids[:, 1] * _MIX)


class IdReflectance:
    """Object id -> reflectance, kept as sorted uint64 hashes and their values: a scan is one
    searchsorted; ids not kept go to `resolve`, which takes a list of (low, high) and returns
    (reflectances, keep): an id it could not place (keep False) gets its reflectance for this
    scan only and is asked about again in the next. A call returns (reflectance per point,
    placed per point)."""

    def __init__(self, resolve):
        self.resolve = resolve
        self.keys = np.zeros(0, dtype=np.uint64)
        self.vals = np.zeros(0, dtype=np.float64)

    def __call__(self, ids):
        if len(ids) == 0:
            return np.zeros(0), np.zeros(0, bool)
        ids = np.asarray(ids)
        h = id_hash(ids)
        out = np.empty(len(h))
        placed = np.ones(len(h), bool)
        if len(self.keys):
            pos = np.minimum(np.searchsorted(self.keys, h), len(self.keys) - 1)
            known = self.keys[pos] == h
            out[known] = self.vals[pos[known]]
        else:
            known = np.zeros(len(h), bool)
        if not known.all():
            new_h, first, inv = np.unique(h[~known], return_index=True, return_inverse=True)
            new_ids = ids[~known][first]
            vals, keep = self.resolve([(int(a), int(b)) for a, b in new_ids])
            vals = np.asarray(vals, dtype=np.float64)
            keep = np.asarray(keep, dtype=bool)
            out[~known] = vals[inv.reshape(-1)]
            placed[~known] = keep[inv.reshape(-1)]
            if keep.any():
                keys = np.concatenate([self.keys, new_h[keep]])
                order = np.argsort(keys, kind="stable")
                self.keys = keys[order]
                self.vals = np.concatenate([self.vals, vals[keep]])[order]
        return out, placed


# Bodies a world server spawned after the lidar's StableIdMap was made are not in it (Isaac
# maps the prims there are when the map is made; 084): their points take the class of the
# entity box they fall in. The world server registers where its entities are, in-process.
_entity_source = None


def set_entity_source(fn):
    """`fn()` -> [(world 4x4 row-major matrix of the entity frame (USD's, rows are axes),
    box centre in the entity frame, box dims, qcode)], or None to unregister."""
    global _entity_source
    _entity_source = fn


def entity_source():
    return _entity_source


BOX_MARGIN = 0.2  # metres around a declared box: a body's mirrors and wheels stick out of it


def box_classes(points, boxes, margin=BOX_MARGIN):
    """For each world point, the index of the entity box it falls in (the nearest box centre
    when it falls in several), or -1. `boxes`: [(matrix, centre, dims, qcode)] as
    set_entity_source's; a USD matrix maps a row vector: world = local @ m."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    best = np.full(len(p), -1)
    best_d = np.full(len(p), np.inf)
    ph = np.hstack([p, np.ones((len(p), 1))])
    for i, (m, centre, dims, _q) in enumerate(boxes):
        inv = np.linalg.inv(np.asarray(m, dtype=np.float64).reshape(4, 4))
        local = (ph @ inv)[:, :3] - np.asarray(centre, dtype=np.float64)
        inside = np.all(np.abs(local) <= np.asarray(dims, dtype=np.float64) / 2 + margin, axis=1)
        d = np.linalg.norm(local, axis=1)
        take = inside & (d < best_d)
        best[take], best_d[take] = i, d[take]
    return best
