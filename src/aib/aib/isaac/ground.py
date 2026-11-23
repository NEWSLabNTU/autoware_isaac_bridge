"""A flat ground plane built here, with no network (scensim issue 117).

Isaac's `Scene.add_default_ground_plane` asks `get_assets_root_path` for its asset root,
NVIDIA's S3 bucket, and references `Isaac/Environments/Grid/default_environment.usd` from it:
a run with no route to the bucket fails at start. This builds the same thing for physics
locally, with pxr's own ground plane: an infinite collision plane (axis Z) under a visible
square, and the contact values Isaac's default plane bound to it.

The tyres never read the material: aib's tyre friction tables name no ground material, so
every wheel takes its table's default on any ground. Other bodies do (a chassis grounding, a
prop on a flat world), so the values stay Isaac's.

What it leaves out is the default environment's picture: its grid texture, and its sphere
light (intensity 100000, 2.5 m above the origin). With no light on the stage,
`aib.runner.ensure_light` adds its sun and sky, as for any world that brings none.

The constants are pure; `add_ground_plane` needs Isaac's Python.
"""

PATH = "/World/defaultGroundPlane"  # the world server's path for it (scensim_isaac_world)
STATIC_FRICTION = 0.5  # Isaac 6.2's add_default_ground_plane's values
DYNAMIC_FRICTION = 0.5
RESTITUTION = 0.8
HALF_SIZE = 1000.0  # m, the visible square's; the collision plane is infinite
COLOR = (0.5, 0.5, 0.5)


def add_ground_plane(stage, path=PATH, z=0.0):
    """The flat ground at `path`, at height `z`. Returns its collision plane's prim."""
    from pxr import Gf, PhysicsSchemaTools, Usd, UsdPhysics, UsdShade

    PhysicsSchemaTools.addGroundPlane(stage, path, "Z", HALF_SIZE, Gf.Vec3f(0.0, 0.0, z),
                                      Gf.Vec3f(*COLOR))
    material = UsdShade.Material.Define(stage, f"{path}/physicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(STATIC_FRICTION)
    api.CreateDynamicFrictionAttr(DYNAMIC_FRICTION)
    api.CreateRestitutionAttr(RESTITUTION)
    colliders = [p for p in Usd.PrimRange(stage.GetPrimAtPath(path))
                 if p.HasAPI(UsdPhysics.CollisionAPI)]
    for prim in colliders:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return colliders[0]
