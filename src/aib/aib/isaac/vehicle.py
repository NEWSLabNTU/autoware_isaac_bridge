"""The ego's PhysX vehicle and its actuation. Import after `SimulationApp`.

The vehicle's rigid body frame is Autoware's `base_link`: the rear axle centre on the ground,
x forward, z up, so the pose read back is the one the vehicle status reports and Autoware
expects. Geometry and physics are the rig's `VehicleParams` (generated from the vehicle
spec), the suspension tuned per axle to one frequency so the car sits level.

PhysX drives it with no engine model ("drive none"): each wheel's drive torque, brake torque
and steer angle are set directly, every physics step, from the car's onboard computers
(`sim_vehicle_ecu`: the drive unit, the brake module with ABS, traction control and ESC, the
power steering; roadmap 055), which see only what the car's own sensors read: the four wheel
speeds and the body's yaw rate and accelerations. The linkage splits their tire angle into
Ackermann angles (`aib.control`). The road loads (rolling resistance, aerodynamic drag:
`aib.params.road_load`) are a force on the body at its centre of mass (PhysxForceAPI), set
every step from the body's speed. Each step's wheel states (spin, both slips, the tire's
force, contact) are read back for the sensors and the host's surface layer.

Each wheel has its own tire friction table (`/World/vehicleTireFriction/<vehicle>/<wheel>`), with
no ground materials: its default value is the wheel's friction on whatever it stands on, the
rig's `tire_friction` until a host writes another (`set_wheel_friction`). A host with a
surface map (swb's surface layer, roadmap 047) writes it every substep from the wheel's
ground (`contacts`: where, on what, the load it carries and the tire's rolling direction, the
inputs of swb's soil layer too); PhysX takes the edit on the next step. The friction-vs-slip graph is
fixed once the simulation runs (PhysX refuses the edit), so a surface's own slip curve is the
host's to apply through the same friction value. The tire's stiffness, sideways and
lengthways, does take while playing (054 step 1): a host scales it per wheel
(`set_wheel_stiffness`) so the force peaks at its ground's own slip (054 T1). The tables live outside the vehicle and stay
when it goes: PhysX refuses to remove a tire friction table while the simulation plays
("removing tire friction tables while playing is not permitted"), so one under the vehicle
failed at every despawn; a vehicle spawned again at the same path takes its tables back.
"""
import math
from typing import NamedTuple

from omni.physx.bindings._physx import (VEHICLE_WHEEL_STATE_GROUND_HIT_POSITION,
                                        VEHICLE_WHEEL_STATE_GROUND_SHAPE,
                                        VEHICLE_WHEEL_STATE_IS_ON_GROUND,
                                        VEHICLE_WHEEL_STATE_ROTATION_SPEED,
                                        VEHICLE_WHEEL_STATE_SUSPENSION_FORCE,
                                        VEHICLE_WHEEL_STATE_TIRE_FORCE,
                                        VEHICLE_WHEEL_STATE_TIRE_LATERAL_DIRECTION,
                                        VEHICLE_WHEEL_STATE_TIRE_LATERAL_SLIP,
                                        VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_DIRECTION,
                                        VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_SLIP)
from omni.physx.scripts.utils import set_custom_metadata
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

from ..control import Command, State, ackermann
from ..deps import use
from ..frames import qconj, qmul, qrot
from ..params import G, VehicleParams, road_load

use("sim_vehicle_ecu")

from sim_vehicle_ecu import Calibration, Ecu, Request, Sensors  # noqa: E402

GROUPS = "/World/vehicleCollisionGroups"
STIFFNESS_STEP = 0.02  # a stiffness share closer than this to the one written is not written
TIRE_FRICTION = "/World/vehicleTireFriction"
WHEELS = (("frontLeft", True, 1), ("frontRight", True, -1), ("rearLeft", False, 1),
          ("rearRight", False, -1))


class WheelState(NamedTuple):
    """What PhysX reports for one wheel after a step."""

    rotation_speed: float = 0.0  # rad/s, positive rolling forward
    slip: float = 0.0  # longitudinal slip: + the wheel spins faster than the road, - slower
    on_ground: bool = True
    slip_angle: float = 0.0  # rad, PhysX's lateral slip: the contact's velocity off the wheel
    force: tuple = (0.0, 0.0)  # the tire's force on the ground, N: (along, across) the wheel


class Contact(NamedTuple):
    """Where one wheel touched the ground in the last step."""

    on_ground: bool = False
    position: tuple = (0.0, 0.0, 0.0)  # the suspension ray's hit, world frame
    shape: str = ""  # the collider's prim path ("" when off the ground)
    load: float = 0.0  # the suspension force's magnitude, N: what the ground carries
    direction: tuple = (1.0, 0.0, 0.0)  # the tire's rolling direction, world frame, unit


def ensure_vehicle_scene(stage):
    """The physics scene's vehicle context and the collision groups.
    Call before the simulation starts: PhysX ignores the context's axes once it runs."""
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            ctx = PhysxSchema.PhysxVehicleContextAPI.Apply(prim)
            ctx.CreateUpdateModeAttr(PhysxSchema.Tokens.velocityChange)
            ctx.CreateVerticalAxisAttr(PhysxSchema.Tokens.posZ)
            ctx.CreateLongitudinalAxisAttr(PhysxSchema.Tokens.posX)
    if not stage.GetPrimAtPath(GROUPS):
        UsdGeom.Scope.Define(stage, GROUPS)
        chassis = UsdPhysics.CollisionGroup.Define(stage, f"{GROUPS}/chassis")
        query = UsdPhysics.CollisionGroup.Define(stage, f"{GROUPS}/groundQuery")
        # A vehicle's suspension raycasts do not hit its own (or any) chassis.
        chassis.CreateFilteredGroupsRel().AddTarget(query.GetPath())
        query.CreateFilteredGroupsRel().AddTarget(chassis.GetPath())


class Vehicle:
    def __init__(self, stage, path, params: VehicleParams, calibration: Calibration, box_center,
                 box_dims, pose):
        """`params`: the PhysX vehicle (a rig's `vehicle`); `calibration`: its onboard
        computers' (a rig's `ecu`)."""
        self.stage = stage
        self.path = path
        self.p = p = params
        self.ecu = Ecu(calibration)
        ensure_vehicle_scene(stage)
        root = UsdGeom.Xform.Define(stage, path)
        prim = root.GetPrim()
        self.translate = root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        self.orient = root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        self._set_xform(pose)
        set_custom_metadata(prim, PhysxSchema.Tokens.referenceFrameIsCenterOfMass, False)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(p.mass)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*p.com))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*p.diagonal_inertia()))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body.CreateSleepThresholdAttr(0)
        body.CreateStabilizationThresholdAttr(0)
        body.CreateDisableGravityAttr(True)  # the vehicle simulation applies gravity
        # The road loads: a force at the centre of mass, in the body frame, set every step.
        force = PhysxSchema.PhysxForceAPI.Apply(prim)
        force.CreateModeAttr(PhysxSchema.Tokens.force)
        force.CreateWorldFrameEnabledAttr(False)
        force.CreateForceEnabledAttr(True)
        self.road_load = force.CreateForceAttr(Gf.Vec3f(0, 0, 0))

        v = PhysxSchema.PhysxVehicleAPI.Apply(prim)
        v.CreateVehicleEnabledAttr(True)
        v.CreateSuspensionLineQueryTypeAttr(PhysxSchema.Tokens.raycast)
        v.CreateSubStepThresholdLongitudinalSpeedAttr(p.substep_threshold_speed)
        v.CreateLowForwardSpeedSubStepCountAttr(p.physx_substeps)
        v.CreateHighForwardSpeedSubStepCountAttr(p.physx_substeps)
        v.CreateMinPassiveLongitudinalSlipDenominatorAttr(p.min_passive_slip_denominator)
        v.CreateMinActiveLongitudinalSlipDenominatorAttr(p.min_active_slip_denominator)
        v.CreateMinLateralSlipDenominatorAttr(p.min_lateral_slip_denominator)
        v.CreateLongitudinalStickyTireThresholdSpeedAttr(p.sticky_tire_threshold_speed)
        v.CreateLongitudinalStickyTireThresholdTimeAttr(p.sticky_tire_threshold_time)
        v.CreateLongitudinalStickyTireDampingAttr(p.longitudinal_sticky_tire_damping)
        v.CreateLateralStickyTireThresholdSpeedAttr(p.sticky_tire_threshold_speed)
        v.CreateLateralStickyTireThresholdTimeAttr(p.sticky_tire_threshold_time)
        v.CreateLateralStickyTireDampingAttr(p.lateral_sticky_tire_damping)

        r, half_t, L = p.wheel_radius, p.wheel_tread / 2, p.wheel_base
        (front_axle, rear_axle), sag = p.suspension()
        # The suspension frame sits so that, sagged under the static load, the wheel centre is
        # at z = r: base_link on the ground at rest.
        frame_z = r + (p.suspension_travel - sag)
        self.wheels = []
        self.wheel_paths = [f"{path}/{name}" for name, _, _ in WHEELS]
        self.wheel_states = None  # WheelState per wheel, after a step
        self.contacts = None  # Contact per wheel, after a step
        self.friction = []  # each wheel's table's defaultFrictionValue attribute
        self.friction_values = [p.tire_friction] * len(WHEELS)
        self.tires = []  # each wheel's PhysxVehicleTireAPI
        self.tire_base = []  # each wheel's rig (lateral graph (x, y), longitudinal) stiffness
        self.stiffness_values = [(1.0, 1.0)] * len(WHEELS)  # shares of tire_base, as written
        for i, (name, is_front, side) in enumerate(WHEELS):
            x, y = (L if is_front else 0.0), side * half_t
            sprung, k, c = front_axle if is_front else rear_axle
            rest_load = sprung * G
            wp = f"{path}/{name}"
            w = UsdGeom.Xform.Define(stage, wp)
            w.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(x, y, r))
            wprim = w.GetPrim()
            att = PhysxSchema.PhysxVehicleWheelAttachmentAPI.Apply(wprim)
            wheel = PhysxSchema.PhysxVehicleWheelAPI.Apply(wprim)
            wheel.CreateRadiusAttr(r)
            wheel.CreateWidthAttr(p.wheel_width)
            wheel.CreateMassAttr(p.wheel_mass)
            wheel.CreateMoiAttr(0.5 * p.wheel_mass * r * r)
            wheel.CreateDampingRateAttr(p.wheel_damping_rate)
            # PhysX tire: lateral stiffness rises with load up to lat_stiff_max_load rest
            # loads, where it is lat_stiff_value per N of rest load; longitudinal per g.
            tire = PhysxSchema.PhysxVehicleTireAPI.Apply(wprim)
            tire.CreateLateralStiffnessGraphAttr(
                Gf.Vec2f(p.lat_stiff_max_load, p.lat_stiff_value * rest_load))
            tire.CreateLongitudinalStiffnessAttr(p.long_stiff_value * G)
            tire.CreateCamberStiffnessAttr(0.0)
            self.tires.append(tire)
            self.tire_base.append(((p.lat_stiff_max_load, p.lat_stiff_value * rest_load),
                                   p.long_stiff_value * G))
            tire.CreateFrictionVsSlipGraphAttr([Gf.Vec2f(*pt) for pt in p.friction_vs_slip])
            table = PhysxSchema.PhysxVehicleTireFrictionTable.Define(
                stage, f"{TIRE_FRICTION}/{path.strip('/').replace('/', '_')}/{name}")
            table.CreateGroundMaterialsRel()
            table.CreateFrictionValuesAttr([])
            self.friction.append(table.CreateDefaultFrictionValueAttr(p.tire_friction))
            tire.CreateFrictionTableRel().AddTarget(table.GetPath())
            susp = PhysxSchema.PhysxVehicleSuspensionAPI.Apply(wprim)
            susp.CreateSprungMassAttr(sprung)
            susp.CreateSpringStrengthAttr(k)
            susp.CreateSpringDamperRateAttr(c)
            susp.CreateTravelDistanceAttr(p.suspension_travel)
            att.CreateIndexAttr(i)
            att.CreateSuspensionTravelDirectionAttr(Gf.Vec3f(0, 0, -1))
            att.CreateSuspensionFramePositionAttr(Gf.Vec3f(x, y, frame_z))
            att.CreateSuspensionFrameOrientationAttr(Gf.Quatf(1, 0, 0, 0))
            att.CreateCollisionGroupRel().AddTarget(f"{GROUPS}/groundQuery")
            comp = PhysxSchema.PhysxVehicleSuspensionComplianceAPI.Apply(wprim)
            comp.CreateWheelToeAngleAttr([])
            comp.CreateWheelCamberAngleAttr([])
            app_point = Gf.Vec4f(0.0, 0.0, 0.0, r - frame_z)
            comp.CreateSuspensionForceAppPointAttr([app_point])
            comp.CreateTireForceAppPointAttr([app_point])
            ctl = PhysxSchema.PhysxVehicleWheelControllerAPI.Apply(wprim)
            self.wheels.append((is_front, ctl.CreateDriveTorqueAttr(0.0),
                                ctl.CreateBrakeTorqueAttr(0.0), ctl.CreateSteerAngleAttr(0.0)))
            # what an operator sees: a wheel (a host with a body hides it)
            cyl = UsdGeom.Cylinder.Define(stage, f"{wp}/render")
            cyl.CreateRadiusAttr(r)
            cyl.CreateHeightAttr(p.wheel_width)
            cyl.CreateAxisAttr(UsdGeom.Tokens.y)
            cyl.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.08, 0.08)])

        # The chassis: the declared box, kept clear of the ground so only the wheels touch it.
        cx, cy, cz = box_center
        dx, dy, dz = box_dims
        bottom = max(cz - dz / 2, r + 0.05)
        top = cz + dz / 2
        cube = UsdGeom.Cube.Define(stage, f"{path}/body")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.45, 0.85)])
        UsdGeom.XformCommonAPI(cube.GetPrim()).SetTranslate(Gf.Vec3d(cx, cy, (bottom + top) / 2))
        UsdGeom.XformCommonAPI(cube.GetPrim()).SetScale(Gf.Vec3f(dx, dy, top - bottom))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.CollisionGroup.Get(stage, f"{GROUPS}/chassis").GetCollidersCollectionAPI() \
            .CreateIncludesRel().AddTarget(cube.GetPath())
        self.body_path = cube.GetPath()
        self.tire_angle = 0.0
        self.torques = None  # the last step's sim_vehicle_ecu.Actuation
        self.last = None  # (position, quat, velocity in world) of the last substep

    # --- Pose ------------------------------------------------------------------------------

    def _set_xform(self, pose):
        (x, y, z), (w, qx, qy, qz) = pose
        self.translate.Set(Gf.Vec3d(x, y, z))
        n = math.sqrt(w * w + qx * qx + qy * qy + qz * qz) or 1.0
        self.orient.Set(Gf.Quatd(w / n, qx / n, qy / n, qz / n))

    def teleport(self, pose, physx):
        """Put it at `pose` at rest (a scenario's teleport)."""
        self._set_xform(pose)
        body = self.stage.GetPrimAtPath(self.path)
        UsdPhysics.RigidBodyAPI(body).GetVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
        UsdPhysics.RigidBodyAPI(body).GetAngularVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
        try:
            physx.set_vehicle_to_rest_state(self.path)
        except Exception:  # noqa: BLE001 - not simulated yet: the USD pose is all there is
            pass
        self.last = None
        self.wheel_states = None
        self.contacts = None
        self.road_load.Set(Gf.Vec3f(0, 0, 0))
        self.ecu.reset()
        self.tire_angle = 0.0

    # --- Actuation -------------------------------------------------------------------------

    def apply(self, cmd: Command, dt, state: State):
        """One physics step's actuation: the vehicle interface's `cmd` as the by-wire request,
        the car's sensors as the last step left them, through the onboard computers."""
        p = self.p
        request = Request(cmd.acceleration, cmd.velocity, cmd.tire_angle, cmd.gear)
        self.torques = act = self.ecu.step(request, self.sensors(state), dt)
        self.tire_angle = act.tire_angle
        if p.ackermann:
            left, right = ackermann(self.tire_angle, p.wheel_base, p.wheel_tread)
        else:
            left = right = self.tire_angle
        angles = (left, right, 0.0, 0.0)  # WHEELS' order
        for (_, d, b, s), drive, brake, angle in zip(self.wheels, act.drive, act.brake, angles):
            d.Set(drive)
            b.Set(brake)
            s.Set(angle)
        self.road_load.Set(Gf.Vec3f(road_load(p, state.velocity[0]), 0.0, 0.0))

    def sensors(self, state: State):
        """What the car's own sensors read after the last step: the wheel speed sensors and
        the brake module's yaw-rate and acceleration sensor (body frame). None before PhysX
        has stepped the wheels."""
        if self.wheel_states is None:
            return None
        return Sensors(tuple(w.rotation_speed for w in self.wheel_states),
                       state.angular_velocity[2], state.acceleration[0], state.acceleration[1])

    def set_wheel_stiffness(self, i, lateral, longitudinal):
        """Wheel `i`'s tire stiffness from the next step on, as shares of the rig's: the lateral
        stiffness graph's value and the longitudinal stiffness. Written only when either moves
        by more than STIFFNESS_STEP of itself, so a host may call it every substep."""
        last = self.stiffness_values[i]
        if all(abs(new / old - 1.0) <= STIFFNESS_STEP for new, old in
               zip((lateral, longitudinal), last)):
            return
        tire = self.tires[i]
        (x, y), long_base = self.tire_base[i]
        tire.GetLateralStiffnessGraphAttr().Set(Gf.Vec2f(x, y * float(lateral)))
        tire.GetLongitudinalStiffnessAttr().Set(long_base * float(longitudinal))
        self.stiffness_values[i] = (lateral, longitudinal)

    def set_wheel_friction(self, i, mu):
        """Wheel `i`'s (WHEELS order) friction from the next step on; written only when it
        changes, so a host may call it every substep."""
        if abs(mu - self.friction_values[i]) > 1e-6:
            self.friction[i].Set(float(mu))
            self.friction_values[i] = mu

    # --- State -----------------------------------------------------------------------------

    def _wheel_states(self, physx):
        """Each wheel's spin and longitudinal slip, and where it touches the ground, after the
        step (None, None if PhysX has none)."""
        out, contacts = [], []
        for wp in self.wheel_paths:
            try:
                w = physx.get_wheel_state(wp)
            except Exception:  # noqa: BLE001 - not simulated yet
                return None, None
            if not w:
                return None, None
            on_ground = bool(w[VEHICLE_WHEEL_STATE_IS_ON_GROUND])
            f = w[VEHICLE_WHEEL_STATE_TIRE_FORCE]
            out.append(WheelState(
                float(w[VEHICLE_WHEEL_STATE_ROTATION_SPEED]),
                float(w[VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_SLIP]),
                on_ground, float(w[VEHICLE_WHEEL_STATE_TIRE_LATERAL_SLIP]),
                (_dot(f, w[VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_DIRECTION]),
                 _dot(f, w[VEHICLE_WHEEL_STATE_TIRE_LATERAL_DIRECTION]))))
            if on_ground:
                f = w[VEHICLE_WHEEL_STATE_SUSPENSION_FORCE]
                contacts.append(Contact(
                    True, tuple(float(c) for c in w[VEHICLE_WHEEL_STATE_GROUND_HIT_POSITION]),
                    str(w[VEHICLE_WHEEL_STATE_GROUND_SHAPE]),
                    math.sqrt(sum(float(c) ** 2 for c in f)),
                    tuple(float(c) for c in w[VEHICLE_WHEEL_STATE_TIRE_LONGITUDINAL_DIRECTION])))
            else:
                contacts.append(Contact())
        return out, contacts

    def sample(self, physx, dt, state: State):
        """Read the body after one physics step; velocities by difference (base_link frame)."""
        t = physx.get_rigidbody_transformation(self.path)
        pos = tuple(float(c) for c in t["position"])
        rx, ry, rz, rw = (float(c) for c in t["rotation"])
        q = (rw, rx, ry, rz)
        if self.last is None or dt <= 0:
            vel_w = (0.0, 0.0, 0.0)
            omega = (0.0, 0.0, 0.0)
            acc = (0.0, 0.0, 0.0)
        else:
            lp, lq, lv = self.last
            vel_w = tuple((a - b) / dt for a, b in zip(pos, lp))
            dq = qmul(qconj(lq), q)  # body-frame rotation over the step
            if dq[0] < 0:
                dq = tuple(-c for c in dq)
            s = math.sqrt(dq[1] ** 2 + dq[2] ** 2 + dq[3] ** 2)
            angle = 2 * math.atan2(s, dq[0])
            omega = (0.0, 0.0, 0.0) if s < 1e-12 else tuple(c / s * angle / dt for c in dq[1:])
            acc = qrot(qconj(q), tuple((a - b) / dt for a, b in zip(vel_w, lv)))
        self.last = (pos, q, vel_w)
        self.wheel_states, self.contacts = self._wheel_states(physx)
        state.position = pos
        state.quat = q
        state.velocity = qrot(qconj(q), vel_w)
        state.angular_velocity = omega
        state.acceleration = acc
        state.tire_angle = self.tire_angle


def _dot(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b))
