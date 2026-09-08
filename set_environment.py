__doc__ = """This file is for setting an environment for Elastica arm reaching to a fixed target and matching arms
orientation with the target. Actuation torques acting on arm can generate torques in normal, binormal and tangent
direction. Environment set in this file is interfaced with stable-baselines3 and Gymnasium.

MIGRATED TO PYELASTICA 1.0.0 (Python 3.12 / .venv_SoftArm). All API changes verified against the
installed 1.0.0 build by source inspection and end-to-end smoke tests:
  * gym -> gymnasium (reset returns (state, info); step returns 5-tuple)
  * rod constructor: `nu` removed (raises ValueError in 1.0.0), `poisson_ratio` removed
    (raises NameError) -> pass shear_modulus = E / (2*(1+0.5)) = E/3
  * damping: AnalyticalLinearDamper physical protocol with
    translational/rotational_damping_constant = NU * (base_length/n_elem) = 0.75 for NU=30
    (derived from 0.1.0.post5 kernels: F = nu*v*l, tau = nu*w*l -> c = nu*l; exact equivalence,
    except the two end nodes damp translationally at 2x -- negligible, see thesis notes)
  * OneEndFixedRod -> OneEndFixedBC ; FreeRod -> FreeBC (custom constraints call super().__init__(**kwargs))
  * stepping: extend_stepper_interface/do_step is deprecated AND broken in 1.0.0
    (`from warning import warn` typo crashes) -> use self.StatefulStepper.step(simulator, time, dt)
  * np.infty -> np.inf (NumPy 2.x removed np.infty)
  * `is` -> `==` for int literal comparisons (Python 3.12)

PICK-AND-PLACE TASK (Route B, extended). The target sphere is a real dynamic body: it has mass,
gravity acts on it, and it rests on a gated anchor (support/pedestal) until picked. A genuine
FixedJoint (spring-damper, forces on BOTH bodies) is registered between rod tip and sphere at
reset (PyElastica requires all connections before finalize) and is gated by a runtime flag:
  reach + orient (policy 1) -> both gates pass -> joint ON, anchor OFF (ATTACH)
  -> the rod carries the sphere's true weight along the dome T2(s) (policy 2)
     -> the load reaches place_position (ARRIVAL, m_place <= place_radius)
  -> the tip is turned to point DOWN at the drop-off (policy 3)
     -> joint OFF (DETACH / RELEASE) once m_place < place_radius AND p_down <=
        orient_tier_inner. The object is set down and the floor catches it.
  Attach, arrival and detach are all automatic consequences of their phase's threshold
  being met -- never learned actions, never rewarded. There is no separate "settle" term.

REWARD -- THREE-PHASE SWITCH (never a sum). Which reward is live is decided by two
runtime latches, attach_state["attached"]/released and _arrived:
  R1  while NOT attached                     -- reach + orient 
  R2  while attached and NOT yet arrived     -- carry along the trajectpory
  R3  once arrived (attached or released)    -- orient the tip down, then release
  n       = ||x_tip - x_sphere||,  p = 1 - (q_tip . q_target)^2       (phase 1)
  m       = ||x_sphere - T2(s(t))||                                    (phase 2)
  m_place = ||x_sphere - place_position||                              (phases 2, 3)
  p_down  = (1 + t_tip . y_hat)/2      0 = tip straight down           (phase 3)

  R1 = -n^2 - 0.5 p^2 + phi1(n, p)
  R2 = -min(m, 2.5)^2 + 1.0*[m<=0.30] + 3.0*[m<=0.15]        <- tracking (2x weight)
                      + 0.5*[m_place<=0.15] + 1.5*[m_place<=0.05]   <- arrival
  R3 = -p_down^2 + 0.5*[p_down<=0.30] + 1.5*[p_down<=0.15]   inside the release band
       -(1.0 + m_place^2)                                    outside it (recovery slope)

ALL THREE use phase 1's shaping form, -(error)^2, so each is NEGATIVE while the policy
performs badly and rises toward positive as it improves (measured bad/mid/good:
R1 -1.41/-0.17/+5.96, R2 -1.00/+0.91/+6.00, R3 -1.00/-0.25/+2.00). That swing is what
gives phase 1 its textbook learning curve and makes phases 2 and 3 behave the same way.

NO ORIENTATION IN PHASE 2. The term that used to live there asked the tip to HOLD THE
GRASP POSE for the whole carry -- geometrically impossible while descending to a
floor-level drop-off (measured over 4996 releases: p_place median 0.865, only 1.3%
below its threshold). Posture is phase 3's entire job now. Measured consequence of the
old arrangement: tracking and releasing competed, and over 300 test episodes per seed
the best tracker (0.088 m median error) released in only 17% of grasps while the worst
(0.240 m) released in 100%.

IMPORTANT: place_position must sit ON A SURFACE (the floor, at ~floor_height + sphere_radius).
After detach the object falls and the floor catches it AT place_position, so m stays ~0 and the
placement stays rewarded. A MID-AIR place_position would make the object fall away after release
(m grows, R2 goes sharply negative), teaching the agent NOT to release -- so keep place_position
at ~floor_height + sphere_radius.

An always-on floor-plane constraint (FloorForSphere, `floor_height`, default 0.0) clamps the
sphere to floor_height + radius and zeroes its velocity there; it is registered unconditionally
whenever PICK_AND_RELEASE=True and is geometrically dormant until the sphere actually reaches
that height (pre-pick the anchor holds it above the floor; while carried it is typically above
the floor). Nothing is ever teleported.

VERIFIED JOINT STABILITY MAP (dt=2.5e-4, E=1e7, r=0.05, L=1, n_elem=40):
  !! THESE CONDITIONS ARE NOT THE ONES THE Case_2 RUNS USE. Every pick-and-place run here
  !! uses sim_dt=2.0e-4 and n_elem=20 -- a smaller step and LONGER elements, both of which
  !! move the axial CFL limit the safe way. Re-measured under the actual Case_2 settings
  !! (probe_mass_stability.py, 30 episodes/density, k=1e3 nu=10 unchanged):
  !!     density  100 / 300 / 500 / 1000  ->  0% NaN blow-ups at every one.
  !! So the "density 1000 -> UNSTABLE" row below does NOT apply as written to this
  !! configuration; it is the result for the 40-element, 2.5e-4 discretisation. Keep the
  !! map for that case, but size payload experiments from the re-measured numbers.
  sphere density 100  (0.052 kg, 0.51 N):  k=1e3, nu=10 -> stable, 0.5 mm sag  (DEFAULT)
                                            k=1e2, nu=1  -> stable, 5 mm sag
  sphere density 1000 (0.524 kg, 5.13 N):  k=1e2, nu=1  -> stable, 5.3 cm sag
                                            k=1e3, nu=10 -> UNSTABLE (blows ~step 200)
  kt must stay small (rod element rotational inertia ~1.2e-7 kg m^2): kt=0.5, nut=1e-3 verified.
  nu >= 30 on the joint is explicit-unstable. Root cause of all instabilities: the baseline sim
  runs at exactly the axial CFL limit (sqrt(E/rho)=100 m/s -> l/c = 2.5e-4 = dt), so stiff point
  loads at the tip need gentle k/nu or a smaller sim_dt.
  NOTE: the release gate detaches the joint (attached=False) AS the object reaches
  place_position, BEFORE the object settles onto the floor. So the floor constraint engages only
  AFTER detach, when the joint is already off -- the compliant-joint-vs-hard-floor-clamp
  interaction is thus avoided by construction. (If a run somehow keeps the object attached at
  floor level -- e.g. place_radius set very large so detach never fires until on the floor --
  the earlier concern returns; fix with smaller K_ATTACH/NU_ATTACH or a softer floor.)
"""

from collections import defaultdict
import copy
import os
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from post_processing import plot_video_with_sphere, plot_video_with_sphere_2D

from MuscleTorquesWithBspline.BsplineMuscleTorques import (
    MuscleTorquesWithVaryingBetaSplines,
)

from elastica._calculus import _isnan_check  # verified importable in 1.0.0
from elastica import *


# Set base simulator class
# 1.0.0: Damping mixin is REQUIRED for the AnalyticalLinearDamper registration below.
class BaseSimulator(
    BaseSystemCollection, Constraints, Connections, Contact, Forcing, CallBacks, Damping
):
    pass


class GatedFixedJoint(FixedJoint):
    """A genuine PyElastica FixedJoint (spring-damper coupling applying equal-and-opposite
    forces/torques to BOTH bodies through the standard force machinery) whose action is
    gated by attach_state["attached"]. Registered before finalize() -- as PyElastica
    requires -- and switched at runtime, following the same pre-registered/live-state
    pattern as MuscleTorquesWithVaryingBetaSplines and WallBoundaryForSphere.

    On the first gated step it captures the current relative rotation between the rod tip
    element and the sphere as the joint's rest rotation (convention verified from 1.0.0
    source: rel_rot = C1 @ C2.T), so the grasp freezes whatever relative orientation
    existed at the attach instant, applying zero torque at that moment.

    IT DOES THE SAME FOR TRANSLATION. PyElastica's FreeJoint force is a ZERO-REST-LENGTH
    spring: it pulls the rod tip NODE and the sphere CENTRE toward coincidence, so its
    equilibrium is the rod buried at the ball's centre rather than the two surfaces in
    contact. Two things follow, and both are wrong:

      1. A GRASP-INSTANT FORCE SPIKE. The attach gate fires at dist < attach_radius
         (0.05 m), so a zero-rest spring fires k * 0.05 the moment it engages -- 50 N at
         k=1e3, but 5000 N at the k=1e5 needed to actually hold a 1-4 kg payload, i.e.
         >1000 m/s^2 on the ball. Stiffening the joint without this capture replaces a
         floating payload with a violent snap at every grasp.
      2. THE WRONG REST GEOMETRY. A gripper holds an object at the pose it grasped it
         in; it does not drag the object's centre onto the fingertip.

    So the tip->sphere offset is captured at the attach instant IN THE TIP'S LOCAL FRAME
    and carried as the rest translation. The joint then holds the full 6-DOF relative
    pose that existed at the grasp -- which is what "rigid attachment" means -- and, like
    the rotation capture, applies zero force at that moment. The offset rotates with the
    tip, so the ball follows the arm's orientation instead of sliding around it.
    """

    def __init__(self, k, nu, kt, nut, attach_state, **kwargs):
        super().__init__(k=k, nu=nu, kt=kt, nut=nut, **kwargs)
        self.attach_state = attach_state
        self._rest_captured = False
        self._rest_offset_local = None

    def apply_forces(
        self, system_one, index_one, system_two, index_two, time=np.float64(0.0)
    ):
        if not self.attach_state["attached"]:
            # Re-arm alongside the rotation capture, so a later grasp freezes the pose
            # held at THAT grasp rather than inheriting an earlier one.
            self._rest_offset_local = None
            return
        p1 = system_one.position_collection[..., index_one]
        p2 = system_two.position_collection[..., index_two]
        # Force indices address NODES; director_collection is per ELEMENT, so index_one
        # = -1 gives the tip element's frame -- the correct local frame for the tip.
        C1 = system_one.director_collection[..., index_one]
        if self._rest_offset_local is None:
            self._rest_offset_local = C1 @ (p2 - p1)
        target = p1 + C1.T @ self._rest_offset_local
        # Same sign convention as FreeJoint.apply_forces, with (p2 - p1) replaced by the
        # deviation from the captured rest pose.
        contact_force = self.k * (p2 - target) + self.nu * (
            system_two.velocity_collection[..., index_two]
            - system_one.velocity_collection[..., index_one]
        )
        system_one.external_forces[..., index_one] += contact_force
        system_two.external_forces[..., index_two] -= contact_force

    def apply_torques(
        self, system_one, index_one, system_two, index_two, time=np.float64(0.0)
    ):
        if self.attach_state["attached"]:
            if not self._rest_captured:
                C1 = system_one.director_collection[..., index_one]
                C2 = system_two.director_collection[..., index_two]
                self.rest_rotation_matrix = C1 @ C2.T
                self._rest_captured = True
            super().apply_torques(system_one, index_one, system_two, index_two, time)
        else:
            # Re-arm the capture whenever the joint is OFF, so a later attach freezes
            # the pose held at THAT grasp rather than inheriting an earlier one. With
            # a fresh simulator per reset the flag is already False at every real
            # grasp, so this changes nothing today; it makes the class correct for any
            # caller that detaches and re-attaches within one simulator, where a
            # latched rest rotation would fire a torque spike on the re-grasp.
            self._rest_captured = False


class GatedSphereAnchor(FreeBC):
    """Physically: the sphere rests on a support/pedestal at its initial position until it
    is picked. Implemented as a constraint (pre-registered, runtime-gated) that pins the
    sphere until the first attachment; from then on it is permanently off and the sphere
    is carried purely by the rod through the joint.
    """

    def __init__(self, attach_state, anchor_position, **kwargs):
        super().__init__(**kwargs)
        self.attach_state = attach_state
        self.anchor_position = anchor_position.reshape(3, 1)

    def constrain_values(self, system, time):
        if not self.attach_state["ever_attached"]:
            system.position_collection[:] = self.anchor_position

    def constrain_rates(self, system, time):
        if not self.attach_state["ever_attached"]:
            system.velocity_collection[:] = 0.0


class FloorForSphere(FreeBC):
    """Always-on ground-plane constraint on the target sphere: whenever the sphere's
    y-position is at or below floor_height + sphere_radius, it is clamped to that
    resting height and its velocity is zeroed. Unlike GatedSphereAnchor (permanently
    off after first attach), this constraint is unconditional -- always registered,
    always live -- and is simply geometrically dormant whenever the sphere sits above
    that height (held by the anchor pre-pick, or carried above the floor while
    transported). It engages the instant the sphere reaches ground level, stopping any
    fall so the sphere visibly settles instead of falling forever.
    """

    def __init__(self, floor_height, **kwargs):
        super().__init__(**kwargs)
        self.floor_height = floor_height

    def constrain_values(self, system, time):
        rest_height = self.floor_height + system.radius
        if system.position_collection[1] < rest_height:
            system.position_collection[1] = rest_height

    def constrain_rates(self, system, time):
        rest_height = self.floor_height + system.radius
        if system.position_collection[1] <= rest_height:
            # Support force: stop only the DOWNWARD velocity so the sphere rests
            # ON the floor instead of being welded to it. The earlier version
            # zeroed the whole velocity vector, which pinned the sphere to the
            # first floor point it touched -- any mid-carry ground graze froze
            # the object permanently and no carry could complete. Lifting the
            # sphere back off the floor (vy > 0) and horizontal motion stay free.
            if system.velocity_collection[1] < 0.0:
                system.velocity_collection[1] = 0.0
            # Ground friction: damp horizontal sliding (~2% per sim step) without
            # blocking a sustained pull from the attached joint.
            system.velocity_collection[0] *= 0.98
            system.velocity_collection[2] *= 0.98
            # Contact friction on SPIN. Nothing else damps the sphere's rotation --
            # dampen() is registered on the rod only, and the gated joint drives the
            # sphere's orientation through its rotational spring (kt=0.5, nut=1e-3)
            # while carried. At release the joint switches off and the sphere kept
            # that angular momentum forever, visibly spinning on the floor in the
            # videos. Same 2%/step as the sliding damping, and like it this is
            # geometrically dormant until the object is actually resting on the
            # ground. Affects post-release frames only: no reward reads sphere
            # orientation (m and m_place use its POSITION, p_down uses the ROD tip),
            # so nothing learned or measured changes.
            system.omega_collection[:] *= 0.98


class Environment(gym.Env):
    """

    Custom environment that follows Gymnasium interface. This environment, generates an
    arm (Cosserat rod) and target (rigid sphere). Target position and orientation are fixed.
    Controller has to select control points (stored in action) and input to step class method.
    Control points have to be in between [-1,1] and are used to generate a beta spline. This beta spline is scaled
    by the torque scaling factor (alpha or beta) and muscle torques acting along arm computed. Muscle torques bend
    and twist the arm and to reach the target and match arm orientation with target.

    PICK-AND-PLACE EXTENSION, THREE POLICIES. Once the arm satisfies BOTH the proximity
    gate and the orientation gate, a real mechanical coupling (GatedFixedJoint) engages and
    the support anchor disengages (ATTACH). The reward is a THREE-phase switch, never a sum:
    R1(n, p) while NOT attached (the paper's reach+orient reward, untouched); R2 while
    carrying and not yet arrived (dome tracking + arrival, NO orientation); R3 once the load
    has reached place_position (tip-down posture, then release). The joint disengages only
    in R3, and only when BOTH m_place < place_radius AND p_down <= orient_tier_inner hold.
    Attach, arrival and detach are pure consequences of their thresholds -- never learned,
    no bonus. There is no separate settle reward; place_position must sit on the floor (see
    module docstring) so the object stays at the target after detach.

    Attributes
    ----------
    dim : float
        Dimension of the problem.
        If dim=2.0 2D problem only muscle torques in normal direction is activated.
        If dim=2.5 or 3.0 3D problem muscle torques in normal and binormal direction are activated.
        If dim=3.5 3D problem muscle torques in normal, binormal and tangent direction are activated.
    n_elem : int
        Cosserat rod number of elements.
    final_time : float
        Final simulation time.
    time_step : float
        Simulation time-step.
    number_of_control_points : int
        Number of control points for beta-spline that generate muscle torques.
    alpha : float
        Muscle torque scaling factor for normal/binormal directions.
    beta : float
        Muscle torque scaling factor for tangent directions (generates twist).
    target_position :  numpy.ndarray
        1D (3,) array containing data with 'float' type.
        Initial target position, If mode is 2 or 4 target randomly placed.
    num_steps_per_update : int
        Number of Elastica simulation steps, before updating the actions by control algorithm.
    action : numpy.ndarray
        1D (n_torque_directions * number_of_control_points,) array containing data with 'float' type.
        Action returns control points selected by control algorithm to the Elastica simulation. n_torque_directions
        is number of torque directions, this is controlled by the dim.
    action_space : spaces.Box
        1D (n_torque_direction * number_of_control_poinst,) array containing data with 'float' type in range [-1., 1.].
    obs_state_points : int
        Number of arm (Cosserat rod) points used for state information.
    observation_space : spaces.Box
        1D ( total_number_of_states,) array containing data with 'float' type.
        State information of the systems are stored in this variable.
    mode : int
        There are 4 modes available.
        mode=1 fixed target position to be reached (default)
        mode=2 randomly placed fixed target position to be reached. Target position changes every reset call.
        mode=3 moving target on fixed trajectory.
        mode=4 randomly moving target.
    COLLECT_DATA_FOR_POSTPROCESSING : boolean
        If true data from simulation is collected for post-processing. If false post-processing making videos
        and storing data is not done.
    E : float
        Young's modulus of the arm (Cosserat rod).
    NU : float
        Dissipation constant of the arm (Cosserat rod). Converted internally to the
        AnalyticalLinearDamper physical-protocol coefficient NU * (base_length / n_elem).
    PICK_AND_RELEASE : boolean
        If True (default) the gated pick-and-place machinery (sphere gravity, anchor,
        joint, floor, phase-2 carry reward R2, attach + detach gates) is active. Set False
        to recover the pure reach/orient task (required for modes 3 and 4, where the target
        moves).
    SPHERE_DENSITY : float
        Density of the target sphere (kg/m^3). Default 100 (0.052 kg, 0.51 N). The
        original Case 2 value of 1000 requires K_ATTACH=1e2, NU_ATTACH=1 (see stability
        map in module docstring) or a smaller sim_dt.
    K_ATTACH, NU_ATTACH, KT_ATTACH, NUT_ATTACH : float
        Gated joint translational stiffness/damping and rotational stiffness/damping.
        Defaults (1e3, 10, 0.5, 1e-3) verified stable for SPHERE_DENSITY=100.
    attach_radius : float
        Proximity gate: attachment requires tip-to-sphere distance below this. Default 0.05.
    orient_tol : float
        Orientation gate: attachment requires quaternion orientation distance below this.
        Default 0.05 (matches the innermost reward tier of the original reward).
    place_position : numpy.ndarray, required when PICK_AND_RELEASE=True
        1D (3,) fixed, constant place/drop-off location (NOT randomized, NOT
        mode-dependent). After attach, R2 drives the sphere here; on reaching it (see
        place_radius) the joint detaches and the object is set down.
        MUST sit on the floor, y ~= floor_height + sphere_radius, so the object rests at
        place_position after detach (a mid-air target makes it fall away -- see module
        docstring).
    place_radius : float
        Release/detach gate: the joint disengages once the sphere is within this distance
        of place_position (AND the orientation gate below passes). Default 0.05.
    place_orient_tol : float
        DEPRECATED / UNUSED. Read by nothing. The phase-3 release gate DOES test
        orientation, but against orient_tier_inner (a p_down direction tolerance), not
        this quaternion tolerance. Kept only so existing callers that pass it do not
        break -- use orient_tier_inner / orient_tier_outer instead.
    attach_deadline : float or None
        Sim time (s) by which an episode must attach, or it is TRUNCATED (default 2.0;
        None disables). Compute lever only -- no reward/penalty/bonus attached; see the
        __init__ comment. Only applies pre-attach: a grasped episode always runs its
        full length so the dome carry can finish.
    dome_apex : float
        Transport apex height H of the phase-2 dome carry path T2(s) (see __init__
        comment / _dome_point). Default 0.75 (inside the verified comfortable reach
        band 0.57-0.83 m; grasps above H get a plain descent chord, bump = 0).
    carry_speed : float
        Speed (m/s) at which the phase-2 goal point rides the dome; carry_duration is
        derived from the actual path length at the attach instant. Default 0.5, the
        verified upper bound the compliant loaded arm can pursue.
    place_orientation : numpy.ndarray
        1D (4,) phase-2 target tip orientation quaternion q_place. Not a constructor
        argument; it is set in reset() to the grasp pose q* (= target_tip_orientation) so
        R2 rewards holding the grasp orientation while transporting the sphere.
    floor_height : float
        Height (y-coordinate) of an always-on ground-plane constraint (FloorForSphere)
        on the sphere: clamps sphere position to floor_height + sphere_radius and
        zeroes its velocity once reached. Registered whenever PICK_AND_RELEASE=True,
        independent of attach state (geometrically dormant above that height). Default 0.0.
    """

    # Required for Gymnasium interface
    metadata = {"render_modes": ["human"]}

    """
    FOUR modes: (specified by mode)
    1. fixed target position to be reached (default: need target_position parameter)
    2. random fixed target position to be reached
    3. fixed trajectory to be followed
    4. random trajectory to be followed 
    """

    def __init__(
        self,
        final_time,
        num_steps_per_update,
        number_of_control_points,
        alpha,
        beta,
        target_position,
        COLLECT_DATA_FOR_POSTPROCESSING=False,
        sim_dt=2.5e-4,
        n_elem=40,
        mode=1,
        dim=3.5,
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        final_time : float
            Final simulation time.
        n_elem : int
            Arm (Cosserat rod) number of elements.
        num_steps_per_update : int
            Number of Elastica simulation steps, before updating the actions by control algorithm.
        number_of_control_points : int
            Number of control points for beta-spline that generate muscle torques.
        alpha : float
            Muscle torque scaling factor for normal/binormal directions.
        beta : float
            Muscle torque scaling factor for tangent directions (generates twist).
        target_position :  numpy.ndarray
            1D (3,) array containing data with 'float' type.
            Initial target position, If mode is 2 or 4 target randomly placed.
        COLLECT_DATA_FOR_POSTPROCESSING : boolean
            If true data from simulation is collected for post-processing. If false post-processing making videos
            and storing data is not done.
        sim_dt : float
            Simulation time-step
        mode : int
            There are 4 modes available.
            mode=1 fixed target position to be reached (default)
            mode=2 randomly placed fixed target position to be reached. Target position changes every reset call.
            mode=3 moving target on fixed trajectory.
            mode=4 randomly moving target.
        *args
            Variables length arguments. Currently, *args are not used.
        **kwargs
            Arbitrary keyword arguments. In addition to the original E, NU, target_v,
            boundary, acti_diff_coef, acti_coef, max_rate_of_change_of_activation:
            * PICK_AND_RELEASE, SPHERE_DENSITY, K_ATTACH, NU_ATTACH, KT_ATTACH,
              NUT_ATTACH, attach_radius, orient_tol, place_position, place_radius,
              place_orient_tol, floor_height : see class docstring.
        """
        super(Environment, self).__init__()
        self.dim = dim
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        # Simulation parameters
        self.final_time = final_time
        self.h_time_step = sim_dt  # this is a stable time step
        self.total_steps = int(self.final_time / self.h_time_step)
        self.time_step = np.float64(float(self.final_time) / self.total_steps)
        print("Total steps", self.total_steps)

        # ---- Video: CAPTURE rate and PLAYBACK rate are separate ----
        # rendering_fps is the CAPTURE rate: step_skip is derived from it, so the
        # sim stores `rendering_fps` frames per SIMULATED second (60 -> one frame
        # every 83 sim steps at sim_dt=2e-4). It is NOT a playback setting, and it
        # never affects the physics: step_skip only gates the diagnostic recorders
        # (data callbacks + the muscle-torque profile recorder); torques are
        # computed every step regardless.
        self.rendering_fps = 60
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))
        # video_fps is the PLAYBACK rate written into the video file. Slow motion
        # factor = rendering_fps / video_fps, so 60/15 = 4x slower than real time
        # (a 5 s episode becomes a 20 s video, showing every captured frame).
        # These two used to be the SAME variable, which made "lower the fps" a
        # no-op on duration: halving it halved the captured frames AND the
        # playback rate, giving a choppier video of identical length.
        self.video_fps = kwargs.get("video_fps", 15)

        # Number of control points
        self.number_of_control_points = number_of_control_points

        # Actuation torque scaling factor in normal/binormal direction
        self.alpha = alpha

        # Actuation torque scaling factor in tangent direction
        self.beta = beta

        # target position
        self.target_position = target_position

        # learning step define through num_steps_per_update
        self.num_steps_per_update = num_steps_per_update
        self.total_learning_steps = int(self.total_steps / self.num_steps_per_update)
        print("Total learning steps", self.total_learning_steps)

        if self.dim == 2.0:
            # normal direction activation (2D)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(self.number_of_control_points)
        if self.dim == 3.0 or self.dim == 2.5:
            # normal and/or binormal direction activation (3D)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2 * self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(2 * self.number_of_control_points)
        if self.dim == 3.5:
            # normal, binormal and/or tangent direction activation (3D)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3 * self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(3 * self.number_of_control_points)

        self.obs_state_points = 10
        num_points = int(n_elem / self.obs_state_points)
        num_rod_state = len(np.ones(n_elem + 1)[0::num_points])

        # 8: 4 points for velocity and 4 points for orientation
        # 11: 3 points for target position plus 8 for velocity and orientation
        # +3: active-phase tracking target (sphere position before attach; dome
        #     waypoint T2(s(t)) while carrying; place_position after release) so
        #     the policy can see the moving phase-2 goal it must pursue.
        # +1 more when observe_mass is set (see the kwarg comment): the payload mass.
        # The estimator occupies the SAME single slot as observe_mass -- it replaces the
        # true mass, it does not sit beside it. If both were present the policy would
        # simply ignore the noisy estimate in favour of the exact value next to it, the
        # estimator would never influence control, and the claim that the controller
        # senses its payload would be false. They are therefore mutually exclusive.
        _mass_dim = 1 if (bool(kwargs.get("observe_mass", False))
                          or kwargs.get("mass_estimator", None) is not None) else 0
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_rod_state * 3 + 8 + 11 + 3 + _mass_dim,),
            dtype=np.float64,
        )

        # here we specify 4 tasks that can possibly used
        self.mode = mode

        if self.mode == 2:
            assert "boundary" in kwargs, "need to specify boundary in mode 2"
            self.boundary = kwargs["boundary"]
            # Paper's Case 2: target is drawn uniformly at random from the boundary
         

        if self.mode == 3:
            assert "target_v" in kwargs, "need to specify target_v in mode 3"
            self.target_v = kwargs["target_v"]

        if self.mode == 4:
            assert (
                "boundary" and "target_v" in kwargs
            ), "need to specify boundary and target_v in mode 4"
            self.boundary = kwargs["boundary"]
            self.target_v = kwargs["target_v"]

        # Collect data is a boolean. If it is true callback function collects
        # rod parameters defined by user in a list.
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING

        self.time_tracker = np.float64(0.0)

        self.acti_diff_coef = kwargs.get("acti_diff_coef", 9e-1)

        self.acti_coef = kwargs.get("acti_coef", 1e-1)

        # Weight on the R2 action-rate penalty, w * mean((a_t - a_{t-1})^2). 0.0 = OFF,
        # which is the default so nothing already trained or configured changes. See the
        # block in the R2 reward for the calibration behind w = 1.0.
        self.action_rate_penalty = float(kwargs.get("action_rate_penalty", 0.0))
        if self.action_rate_penalty < 0.0:
            # A negative weight would PAY for jitter -- the exact opposite of the intent,
            # and it would fail silently as a slowly worsening curve. Refuse it.
            raise ValueError(
                "action_rate_penalty must be >= 0 (0 = off), got "
                f"{self.action_rate_penalty}"
            )
        # Weight on the R2 action-MAGNITUDE penalty, w * mean(a_t^2). 0.0 = OFF and is
        # the default, so nothing already trained or configured changes.
        #
        # THIS IS NOT A SUBSTITUTE FOR action_rate_penalty; the two tax different things
        # and are independent. mean(a^2) cannot see jitter at all -- a command sitting
        # flat at 0.7 and one alternating +/-0.7 every step have IDENTICAL magnitude
        # penalty. What it does see is effort, including co-contraction (opposing muscles
        # firing together: large |a|, little net motion).
        #
        # CALIBRATION. Measured on the trained _P2rate policies during the carry:
        #   mean |a|         0.69     (actions live in [-1, 1] -- near saturation)
        #   mean ||a||^2/d   0.563
        #   mean ||da||^2/d  0.239    <- 2.4x smaller
        # so the SAME weight costs 2.4x more here than on the rate term: w=1.0 would take
        # ~21% of a 2.76 mean step reward. w = 0.3 matches the ~6% budget the rate penalty
        # was sized against.
        #
        # BE AWARE of what it can and cannot reduce. mean|a| = 0.69 is largely the load-
        # bearing activation needed to hold the sphere against gravity through the carry;
        # the policy cannot pay that down without relaxing, i.e. dropping the load or
        # falling behind the dome. The reducible part is the co-contraction on top.
        self.action_penalty = float(kwargs.get("action_penalty", 0.0))
        if self.action_penalty < 0.0:
            # A negative weight would PAY for effort, and would fail silently as a
            # slowly worsening curve rather than as an error. Refuse it.
            raise ValueError(
                f"action_penalty must be >= 0 (0 = off), got {self.action_penalty}"
            )
        self.max_rate_of_change_of_activation = kwargs.get(
            "max_rate_of_change_of_activation", np.inf
        )

        self.E = kwargs.get("E", 1e7)

        self.NU = kwargs.get("NU", 30)

        self.n_elem = n_elem

        # ------------------- PICK-AND-PLACE PARAMETERS -------------------
        # All gated machinery is per-episode state managed inside reset()/step();
        
        self.PICK_AND_RELEASE = kwargs.get("PICK_AND_RELEASE", True)
        if self.PICK_AND_RELEASE and self.mode in (3, 4):
            raise ValueError(
                "PICK_AND_RELEASE requires a stationary target (mode 1 or 2). "
                "Pass PICK_AND_RELEASE=False for modes 3/4."
            )
        # Sphere is a real dynamic body. Default density 100 -> 0.052 kg, 0.51 N.
        # Original Case 2 used 1000 (0.524 kg): stable only with K_ATTACH=1e2,
        # NU_ATTACH=1 at sim_dt=2.5e-4 (5.3 cm grasp sag) -- see stability map.
        self.SPHERE_DENSITY = kwargs.get("SPHERE_DENSITY", 100)
        # ---- PAYLOAD MASS RANDOMISATION ----
        # A list of densities to draw from at every reset, instead of the single fixed
        # SPHERE_DENSITY. The sphere is rebuilt inside reset(), so the draw simply sets
        # SPHERE_DENSITY before that happens and the new mass is real from step 1.
        # None (default) = the single fixed density, exactly as every run so far.
        #
        # DENSITY, NOT RADIUS, is the thing to vary: base_radius sets the drop-off height
        # (floor_height + radius), the attach gate distance and how the object reads on
        # screen, so changing it would move the task itself. Density changes mass alone:
        #     mass = density * (4/3) * pi * 0.05^3 = density * 5.236e-4 kg
        #
        # The draw uses self.np_random, so a run's mass sequence is fixed by its SEED and
        # is reproducible -- and each seed sees its own stream, exactly as it does for the
        # pick target and the drop-off.
        # ---- PAYLOAD MASS IN THE OBSERVATION ----
        # Appends ONE extra entry: payload mass / rod mass, dimensionless. Raw density
        # (100-7639) or raw kg (0.05-4.0) would sit orders of magnitude away from the
        # position entries, which are O(1) metres; dividing by the rod's 7.854 kg puts the
        # intended 1-4 kg range at 0.13-0.51, on the same scale as everything else.
        #
        # THIS CHANGES THE OBSERVATION DIMENSION, 55 -> 56, so a policy trained with it
        # set cannot be loaded by a run without it, and vice versa. Default False keeps
        # every existing policy (_ACR_P1, _P2act, _P3tight, ...) loadable; the new
        # mass-aware pipeline passes observe_mass=True explicitly.
        #
        # It is reported in EVERY phase, including phase 1 where the payload provably does
        # not affect the dynamics (rod and sphere are decoupled before the grasp -- the
        # joint transmits nothing and the anchor pins the sphere, verified bit-identical
        # at 5x mass). Policy 1 will simply learn to ignore it. It is carried there anyway
        # because a single observation_space must serve all phases.
        self.observe_mass = bool(kwargs.get("observe_mass", False))
        # ---- PAYLOAD ESTIMATOR (frozen) ----
        # Path to a .npz written by train_estimator.py. When set, observation entry 56
        # carries the ESTIMATED payload instead of the true one, and the true mass never
        # reaches the policy at all -- it exists only as a training label inside that file's
        # provenance. The model is FIXED here: it is a sensor, not something that learns
        # alongside the policy.
        self.mass_estimator = kwargs.get("mass_estimator", None)
        self._est_model = None
        self._est_kind = None
        self._mass_estimate = 0.0          # fraction of rod mass; what obs[56] reports
        if self.mass_estimator is not None:
            assert not self.observe_mass, (
                "observe_mass and mass_estimator are mutually exclusive: the first is the "
                "ORACLE baseline (true mass), the second is the estimator. Enabling both "
                "would let the policy ignore the estimate and read the true value instead."
            )
            import estimator_features as _ef
            self._ef = _ef
            # TWO BACKENDS, chosen by file extension. Both answer the same question and
            # return PERCENT of rod mass, so nothing downstream of here changes:
            #   .npz -> ridge, a 29-weight dot product (deterministic, no torch)
            #   .pt  -> the tuned MLP ensemble from train_estimator_nn.py
            if str(self.mass_estimator).endswith(".pt"):
                import estimator_nn_model as _enn
                self._est_kind = "nn"
                self._enn = _enn
                self._est_model = _enn.load(self.mass_estimator, _ef.FEATURE_NAMES)
            else:
                self._est_kind = "ridge"
                _m = np.load(self.mass_estimator)
                assert list(_m["feature_names"]) == _ef.FEATURE_NAMES, (
                    "estimator was trained on a different feature set than "
                    "estimator_features.py now defines -- refusing to run, it would be fed "
                    "inputs it never saw."
                )
                self._est_model = dict(coef=_m["coef"], mu=_m["mu"], sd=_m["sd"])
            self.probe_steps = int(kwargs.get("probe_steps", 107))
            self.probe_amp = float(kwargs.get("probe_amp", 0.10))
            self.probe_hz = float(kwargs.get("probe_hz", 20.0))
            self.probe_seconds = float(kwargs.get("probe_seconds", 0.15))
        self.sphere_density_set = kwargs.get("sphere_density_set", None)
        if self.sphere_density_set is not None:
            self.sphere_density_set = [float(d) for d in self.sphere_density_set]
            if not self.sphere_density_set or min(self.sphere_density_set) <= 0:
                raise ValueError(
                    "sphere_density_set must be a non-empty list of positive densities, "
                    f"got {self.sphere_density_set}"
                )
        # Gated joint coefficients (verified stable for SPHERE_DENSITY=100)
        # JOINT GAINS ARE A PROPERTY OF THE GRIPPER, NOT OF THE PAYLOAD -- do not re-tune
        # them per experiment. The old defaults (1e3, 10) were chosen for the 0.052 kg
        # payload in use at the time, and silently broke when phase 2 moved to 1-4 kg:
        # the penalty spring's deflection is F/k, so at k=1e3 a 4 kg load sagged 3.9 cm
        # statically and up to 45 cm under the accelerations of a carry, leaving the ball
        # visibly floating off the tip while still being dragged along on the spring.
        # Worse, that deflection is scored: R2 measures the SPHERE's tracking error, so
        # ~0.09-0.15 m of it was manufactured by joint compliance against a 0.15 m inner
        # tier -- unreachable no matter how well the arm was controlled.
        #
        # 1e5 is set from a RIGIDITY criterion at the heaviest payload, then checked for
        # stability at the lightest, so one fixed value spans the range and well past it:
        #   4 kg   -> 0.4 mm sag (0.8% of the sphere radius)
        #   7.9 kg -> 0.8 mm   (100% of rod mass, 2x outside the training range)
        #   78 kg  -> 7.7 mm   (20x outside)
        # Stability is set by the LIGHTEST payload (highest sqrt(k/m)), and even 0.05 kg
        # resolves the joint oscillation in 22 steps at sim_dt=2e-4. Verified non-diverging
        # at 1e3/1e4/1e5/1e6 across 0.05-4 kg.
        # NU_ATTACH 100 -> 800 (2026-09-04). At k=1e5 the joint is very underdamped at
        # c=100 (zeta = c/(2*sqrt(k*m)) = 0.09-0.18 across 0.8-3.1 kg), so it rings about
        # its rest pose and the surfaces dip in and out of contact by a few mm. Measured,
        # seeded at exact contact and driven hard for 700 steps: the worst overlap at the
        # 40% payload falls from -4.7 mm at c=100 to -2.2 mm at c=800, and the outward
        # excursion from +3.9 mm to +1.3 mm. zeta becomes 0.71-1.43, i.e. properly damped.
        # Explicit-integration check c*dt/m_p stays at 0.20 even for the lightest payload.
        self.K_ATTACH = kwargs.get("K_ATTACH", 1e5)
        self.NU_ATTACH = kwargs.get("NU_ATTACH", 800)
        # ROTATIONAL PAIR, 2026-09-04: 0.5 -> 100 and 1e-3 -> 0.8. Same disease as the
        # translational pair -- tuned for a 0.052 kg payload, then the payload grew and the
        # rotational inertia grew with it (I = (2/5) m r^2, so 60x from 0.052 kg to 3.1 kg).
        #
        # DERIVED, not tuned. For a sphere I = (2/5) m r^2, so scaling the translational
        # gains by the radius of gyration squared, (2/5) r^2 = 1e-3 m^2, makes the
        # rotational and translational natural frequencies and damping ratios IDENTICAL at
        # every payload mass:
        #     kt  = K_ATTACH  * (2/5) r^2 = 1e5 * 1e-3 = 100  N.m/rad
        #     nut = NU_ATTACH * (2/5) r^2 = 800 * 1e-3 = 0.8  N.m.s/rad
        # giving w_n = 357 rad/s at the 10% payload and 178 at 40%, matching translation
        # exactly, with zeta 1.43 / 0.71. Explicit-integration check nut*dt/I peaks at 0.20
        # for the lightest payload, the same margin as the translational side.
        #
        # WHY IT MATTERED MOST FOR PHASE 3. At kt=0.5 the joint was w_n = 2 Hz and
        # zeta = 0.013 -- essentially undamped -- so the payload's orientation lagged and
        # rang relative to the tip. Measured over a driven carry, the tip-to-payload angle
        # drifted from the pose captured at the grasp by a median 9.7-13.9 deg and up to
        # 27 deg. Phase 3's release gate (p_down <= 0.15) is about 46 deg off vertical, so
        # the joint alone consumed over half the tolerance, and the orientation policy 3
        # commanded at the tip was NOT the orientation the payload had. With the new pair
        # that drift falls to a median 0.26-0.78 deg, max 2.0 deg.
        self.KT_ATTACH = kwargs.get("KT_ATTACH", 100.0)
        self.NUT_ATTACH = kwargs.get("NUT_ATTACH", 0.8)
        # Attach gate (reach + orient) and release gate (place position + orient).
        # BOTH attach and detach are automatic consequences of their phase's threshold
        # being met -- never learned actions, never rewarded (they mirror each other).
        # PHASE-1 DEADLINE: an episode that has not attached by this sim time is
        # TRUNCATED (not terminated -- see step()). Purely a compute lever: ~86% of
        # episodes never attach and would otherwise burn the full 5 s producing zero
        # phase-2 data. Cutting them roughly DOUBLES the episodes (and hence attaches
        # and phase-2 transitions) per timestep budget, and -- since mode 2 redraws a
        # random target every reset -- also doubles target diversity per unit compute.
        # Default 2.0 s is exactly the FULL episode length of the _pickplace_posrelease
        # run that produced 662 attaches (median attach 0.56 s, p90 1.54 s), so it
        # cannot cut off any attach behaviour the policy has ever demonstrated.
        # Touches NO reward, penalty or bonus. Set None to disable.
        self.attach_deadline = kwargs.get("attach_deadline", 2.0)
        # ---- GRASP GEOMETRY: SURFACE-REFERENCED (changed 2026-09-04) ----
        # attach_radius USED to be compared against ||tip_node - payload_centre|| with a
        # default of 0.05 -- which is the payload's own radius. That gate could therefore
        # only fire once the rod tip had reached the payload's CENTRE, i.e. once two solid
        # bodies were ~5 cm inside one another. The rest-pose capture then froze that
        # overlap for the whole carry.
        #
        # WHY IT WAS EVER LIKE THAT: in the ORIGINAL Case 2 there is no grasping. The
        # sphere is a TARGET MARKER and its 0.05 radius IS the reach tolerance -- the ball
        # drawn on screen is the tolerance region, so the tip entering it is the tip
        # arriving, not penetration. The pick-and-place extension gave that same sphere a
        # second job as a rigid payload with real volume, and the two jobs need different
        # thresholds. This is the fix.
        #
        # NOW: the gate is measured on SURFACE SEPARATION,
        #     s = ||tip_node - payload_centre|| - (r_tip + r_payload)
        # so s = 0 is exactly touching and s < 0 is interpenetration. The grasp fires at
        # s < attach_clearance, approached from outside, so the captured pose is always at
        # contact and never inside. attach_clearance is ~ the tip travel in one control
        # step (3.5 m/s * 1.4 ms = 4.9 mm), so overshoot and clearance cancel and the
        # captured separation lands at ~0.
        self.attach_clearance = kwargs.get("attach_clearance", 0.005)
        # GRASP DIRECTION. Clearance alone does not make a grasp a TIP grasp: it stops the
        # payload intersecting the arm, but still admits it resting against the side of the
        # shaft. Measured with clearance-only gating, grasps landed on elements 16-17 of 19
        # -- touching the arm two elements back from the end, not held off the tip.
        # This requires the payload to lie inside a cone about the tip's own axis:
        #     cos(angle between (payload_centre - tip_node) and the tip tangent) >= this
        # 0.5 = a 60 deg half-angle. Because the nearest point of a capsule's centreline is
        # the end node exactly when the payload is in the tip's forward hemisphere, any
        # positive value also makes _body_gap and _surface_gap agree at the grasp.
        self.attach_cone = kwargs.get("attach_cone", 0.5)
        # Kept only so an explicit attach_radius= in old scripts still raises no error;
        # nothing reads it for the gate any more.
        self.attach_radius = kwargs.get("attach_radius", 0.05)
        self.orient_tol = kwargs.get("orient_tol", 0.05)
        self.place_radius = kwargs.get("place_radius", 0.05)
        self.place_orient_tol = kwargs.get("place_orient_tol", 0.05)

        # ---- Phase-2 DOME carry trajectory (Next_Training_Plans.txt, 2026-07-31) ----
        # Standard time-parametrized moving-goal tracking (same style as modes 3/4):
        #   T2(s) = g + (p - g)*s + max(H - 0.5*(g_y + p_y), 0) * sin(pi*s) * y_hat
        #   s(t)  = clamp((t - t_attach) / carry_duration, 0, 1)
        # where g = sphere position at the attach instant, p = place_position and
        # H = dome_apex. T2 is only a moving GOAL POINT (exerts no force); the rod
        # physically carries the sphere and R2's distance term tracks T2(s(t)).
        # Every point of the path is reachable by construction: both endpoints are
        # (g is where the arm actually grasped; p is reachability-verified, gap
        # 0.077 m < the 0.088 m calibration point the policy placed within 0.05 m
        # of), and the sin bump lifts the chord MIDPOINT to H = 0.75, inside the
        # comfortable reach band 0.57-0.83 m (see check_dome_reachable.py). With
        # asymmetric endpoints (high grasp, floor-level p) the true crest sits a
        # little above H -- bounded by max(g_y, ~0.86) over the pick box, still
        # under the ~0.9 m demonstrated reach ceiling. For grasps above H the
        # bump is 0 and the path is the plain descent chord.
        # carry_duration is computed at attach time from the actual path length so
        # the goal moves at carry_speed (default 0.5 m/s, the verified upper bound
        # the compliant loaded arm can pursue).
        # ---- THREE-POLICY SPLIT: one policy per phase ----
        # phase = 0 : combined. All three rewards can fire in one episode, switching on
        #             the attach and arrival latches. This is the HAND-OFF/test mode.
        # phase = 1 : POLICY 1 -- reach + orient + grasp ONLY. The episode ENDS at the
        #             attach, so only R1 is ever paid.
        # phase = 2 : POLICY 2 -- loaded carry ONLY. Starts already grasped (see
        #             _phase2_warmup) and ENDS at the ARRIVAL, so only R2 is ever paid.
        #             It can never release -- that gate lives in phase 3.
        # phase = 3 : POLICY 3 -- placement posture ONLY. Starts already at the drop-off
        #             (see _phase3_warmup) and ENDS at the RELEASE, so only R3 is paid.
        # Each phase manufactures its own start state, so any one can be retrained
        # WITHOUT the others: phase 2 needs no policy 1, phase 3 needs no policy 2.
        # phase2_policy1 / phase3_policy2 opt into real replay instead (see those).
        self.phase = kwargs.get("phase", 0)
        if self.phase not in (0, 1, 2, 3):
            raise ValueError(
                "phase must be 0 (combined), 1 (reach), 2 (carry) or 3 (place)"
            )
        if self.phase in (2, 3) and not self.PICK_AND_RELEASE:
            raise ValueError("phases 2 and 3 (carry / place) require PICK_AND_RELEASE=True")
        # ---- PHASE-3 DOWNWARD-ORIENTATION tiers (p_down, 0 = tip straight down) ----
        # p_down = (1 + t_tip . y_hat)/2, so a tier value converts to an angle from
        # straight down as acos(1 - 2*p_down):  0.30 -> 66 deg, 0.15 -> 46 deg.
        # LOOSENED from the 0.10 (37 deg) first proposed. Measured over the five
        # hand-off episodes, restricted to genuinely ATTACHED frames (post-release
        # frames are unloaded and let the freed rod point anywhere -- they read far
        # better than they are), the best p_down reached while the load was inside
        # place_radius was 0.027 / 0.409 / 0.020 / 0.069 / 0.111. A 0.10 gate is
        # met by 3 of 5 seeds and just missed by a 4th, so it sits right at the edge
        # of what has been demonstrated; 0.15 clears 4 of 5 with margin and still
        # means "pointing down" in any reasonable sense.
        # LOOSENED 0.15 -> 0.20 (46 deg -> 53 deg) with the drop-off now randomised.
        # Measured over the 146 sampled poses inside the 0.50-0.60 m spawn annulus,
        # the fraction where the loaded arm can point down at all:
        #     p_down <= 0.10  61%      <= 0.20  78%   <- knee
        #     p_down <= 0.15  67%      <= 0.25  78%
        #                              <= 0.30  78%
        # There is a real gain to 0.20 (+11 points) and NOTHING beyond it -- the
        # remaining 22% of poses cannot be oriented downward at any tolerance, so
        # loosening further would only weaken the claim without adding feasibility.
        # 0.15 was calibrated when the drop-off was a single fixed point; a different
        # drop-off every episode is a harder problem and warrants the extra margin.
        # NOTE: orient_tier_inner is ALSO the release gate, not just a bonus tier.
        self.orient_tier_outer = kwargs.get("orient_tier_outer", 0.35)
        self.orient_tier_inner = kwargs.get("orient_tier_inner", 0.20)
        # Phase-3 warm-up: the TRAINED policy 2 carries the load to the drop-off, so
        # policy 3 trains from the states it will actually inherit at test time.
        # phase3_policy2 unset (default) = INDEPENDENT synthetic start, so phase 3 can
        # be retrained without policy 2 exactly as phase 2 retrains without policy 1.
        # Set it only to validate against a real policy-2 arrival distribution.
        self.phase3_policy2 = kwargs.get("phase3_policy2", None)
        self.phase3_warmup_max_time = kwargs.get("phase3_warmup_max_time", 4.0)
        # Load-height band accepted by the synthetic warm-up, and the (longer than
        # phase 2's) actuation window that settles the arm toward the floor.
        self.phase3_load_y = kwargs.get("phase3_load_y", (0.0, 0.12))
        self.phase3_warmup_time = kwargs.get("phase3_warmup_time", (0.60, 1.40))
        self.phase3_warmup_max_tries = kwargs.get("phase3_warmup_max_tries", 12)
        self._policy2_model = None
        # place_position is REWRITTEN each phase-3 episode by the synthetic warm-up,
        # so keep the configured one to restore at every reset.
        self._place_position_cfg = None
        # ---- PHASE-2 TRACKING TIERS, TIED TO THE GATE (phase-1 pattern) ----
        # DERIVED from place_radius rather than hardcoded, so the tier and the gate can
        # never drift apart again:
        #     outer = 2 * place_radius = 0.10     inner = place_radius = 0.05
        # This mirrors phase 1 exactly, where the tiers are 0.05*2 and 0.05 and the inner
        # one EQUALS attach_radius -- so in phase 1 "maximum reward" and "the gate fires"
        # are the same condition, and a policy that maximises R1 attaches by construction.
        #
        # Phase 2 previously used 0.30/0.15, three times looser than place_radius. That
        # broke the correspondence: a policy could sit at full tracking reward while still
        # 3x too far to arrive. Measured on _P2mass, terminal error at s=1 was 0.104-0.166 m
        # against a 0.05 m arrival radius, and not one seed got inside -- arrival happened
        # ~40% of the time, essentially by luck. Since T2(1) = place_position exactly,
        # tracking to the gate IS arriving, so the two thresholds should be one number.
        #
        # KNOWN RISK, recorded because it is the thing to watch. Phase 1 only has to hit
        # 0.05 ONCE, on a STATIC target, and its episode ends there. Phase 2 must hold it
        # against a goal moving at carry_speed for the whole carry. On the trained _P2mass
        # policies only 5% of carry steps are currently inside 0.05 (19% inside 0.08, 36%
        # inside 0.10, 65% inside the old 0.15), and the load's own oscillation is
        # 20-71 mm -- comparable to the 50 mm band, so the inner tier will flicker on
        # ripple the policy cannot fully suppress. If learning stalls, the first thing to
        # try is track_tier_inner=0.08 rather than reverting the whole scheme.
        # 0.30 / 0.15, the original values. They were briefly tied to the gate
        # (2*place_radius and place_radius = 0.10/0.05) to mirror phase 1, where the inner
        # tier EQUALS attach_radius so "max reward" and "gate fires" coincide. Measured
        # over a 1.5M-step A/B/C test (1 seed, rho=300, fixed drop-off) that was WORSE on
        # every axis:
        #     tiers            track med   err @ s=1   arrived
        #     0.30 / 0.15          0.086       0.092     100%    <- restored
        #     0.10 / 0.05          0.073       0.250      20%
        # At 0.05 the inner tier fires on only ~5% of carry steps, so there is almost no
        # gradient toward it and the policy settles for the outer tier; at 0.15 it fires on
        # 65% and pulls continuously. Phase 1 gets away with tier = gate because it hits
        # 0.05 ONCE on a STATIC target and its episode ends there -- phase 2 would have to
        # HOLD it against a goal moving at carry_speed, with the load's own 20-71 mm
        # oscillation already comparable to the 50 mm band.
        self.track_tier_outer = kwargs.get("track_tier_outer", 0.30)
        self.track_tier_inner = kwargs.get("track_tier_inner", 0.15)
        # Phase 2 is now tracking + arrival ONLY (all orientation moved to phase 3),
        # so TRACKING carries the larger share: 1.0/3.0 against arrival's 0.5/1.5.
        # The 1:3 outer:inner ratio of the proven phase-1 pattern is preserved in
        # both -- only the scale differs, doubling tracking's weight.
        self.track_bonus_outer = kwargs.get("track_bonus_outer", 1.0)
        self.track_bonus_inner = kwargs.get("track_bonus_inner", 3.0)
        # ARRIVAL bonuses (restored 2026-09-04 to the _P2act/_P2mass values). Paid on
        # m_place = ||x_sphere - place_position||, gated at 3*place_radius and
        # place_radius, so they reward getting the LOAD to the drop-off -- distinct from
        # the tracking tiers above, which reward following T2(s(t)) along the way.
        # Together with the tiers these put max R2 back at 1.0 + 3.0 + 0.5 + 1.5 = 6.0.
        # Set either to 0.0 to run tracking-only, as _P2mobs did.
        self.arrival_bonus_outer = kwargs.get("arrival_bonus_outer", 0.5)
        self.arrival_bonus_inner = kwargs.get("arrival_bonus_inner", 1.5)
        # R3 orientation bonuses. Defaults chosen so R3's ceiling (-0 + 1.5 + 4.5 = 6.0)
        # matches R1's and R2's, rather than sitting 3x below them as it did. Paid on
        # p_down, the tip's deviation from pointing straight down.
        self.orient_bonus_outer = kwargs.get("orient_bonus_outer", 1.5)
        self.orient_bonus_inner = kwargs.get("orient_bonus_inner", 4.5)
        # Bound on the tracking error entering -m^2. Set OUTSIDE the reachable set
        # (max load-to-goal separation is 2.0 m), so it only ever truncates a
        # numerically diverged episode -- see the reward block for the measurements.
        self.track_error_clip = kwargs.get("track_error_clip", 2.5)
        # A velocity-matching term, -w*min(|v_obj - v_goal|, clip)^2, was added here and
        # then REMOVED after measurement. A vs B in the same A/B/C test isolated it
        # (identical tiers, the penalty the only difference): tracking got 2.4x WORSE
        # (0.177 vs 0.073) and arrival fell 20% -> 0%, and it did not even reduce what it
        # targeted -- |v_err| was 1.214 with it against 1.223 for the control. R2 is
        # position-only again. _dome_velocity() is kept: the diagnostics still use it.
        # ---- SETTLE WINDOW (between the grasp and the start of the carry) ----
        # The trajectory used to be sampled AT the attach instant, which is the worst
        # possible moment: measured over the hand-off episodes the load is moving at
        # 1.4-7.5 m/s when the grasp fires, so it leaves the (stationary) goal
        # immediately and the opening excursion reached 0.20-0.53 m -- consuming
        # 2-47% of the whole carry before tracking could even begin. In phase-2
        # TRAINING the same thing happened for a different reason: the warm-up holds
        # a constant torque to bend the rod and then zeroes it, and the elastic
        # springback alone produced 0.175 m of error in 0.17 s with NO commanded
        # action at all.
        # So the carry now starts only once the load has actually settled:
        #     t_carry = min{ t >= t_att : ||v_sphere|| <= settle_speed },
        #               capped at t_att + settle_max_time
        #     a       = x_sphere(t_carry)          (NOT x_sphere(t_att))
        #     tau     = clip((t - t_carry)/T_c, 0, 1)
        # T(s), B, T_c and the arc-length inversion are all unchanged. While settling,
        # s is frozen at 0 so the goal sits at the grasp point and R2 pays the policy
        # to arrest the momentum. The dome is then REBUILT from where it stabilised,
        # rather than demanding a return to the original grasp point -- after a 0.5 m
        # excursion that return may simply not be achievable.
        # settle_speed defaults to carry_speed: do not start the schedule until the
        # load is moving slower than the goal will.
        self.settle_speed = kwargs.get("settle_speed", None)   # None -> carry_speed
        self.settle_max_time = kwargs.get("settle_max_time", 0.5)
        # A MINIMUM settle duration is required before the velocity test may pass:
        # at the hand-over instant the rod is in a held equilibrium and its velocity
        # is momentarily low, so an instantaneous test exits immediately and the
        # transient then develops anyway (measured: settle exited at 0.000 s while
        # the peak error still reached 0.233 m).
        self.settle_min_time = kwargs.get("settle_min_time", 0.15)
        self.dome_apex = kwargs.get("dome_apex", 0.75)
        self.carry_speed = kwargs.get("carry_speed", 0.5)
        # Every commanded point T2(s) must be a position the arm can actually occupy,
        # otherwise the policy is asked to put the object where the rod cannot go and
        # the tracking error can never close. The rod is a 1.0 m cantilever, so |T2(s)|
        # <= dome_reach_max is a necessary condition (a bent rod reaches strictly less
        # than its length). Measured over grasps on the reachable sphere: 23% of domes
        # violated 0.95 m and the worst commanded 1.087 m -- beyond the rod entirely.
        # _setup_dome shrinks the arch until it complies.
        self.dome_reach_max = kwargs.get("dome_reach_max", 0.95)
        # PHASE-2 hand-over state (see _phase2_warmup). Spawning at the straight-rod
        # tip (0,1,0) is degenerate: that is the arm's MAXIMUM reach, so the dome from
        # it can only descend (measured dy/ds < 0 everywhere) and no apex value fixes
        # it -- any visible arch would sit outside the workspace. The warm-up instead
        # drives the rod to a real loaded configuration whose grasp height matches the
        # range phase 1 actually hands over at.
        # phase2_policy1 unset (default) = INDEPENDENT synthetic start (below), so
        # phase 2 can be retrained without policy 1. Set it to a trained policy 1 and
        # every phase-2 episode instead begins from a REAL grasp: policy 1 is rolled
        # out until its own attach gate fires, then the hand-over happens exactly as
        # in phase 0. Same trade as phase3_policy2 -- true hand-over distribution in
        # exchange for the independence.
        # Path to a file of REAL policy-1 attach states (see harvest_attach_states.py).
        # When set, every phase-2 episode RESTORES one instead of synthesising a grasp,
        # so policy 2 trains on the dynamics it actually inherits -- measured hand-over
        # tip speed 3.52 m/s and curvature 3.59 for real attaches, against 1.80 and
        # 2.79 for synthetic ones. Restore fidelity verified at 6.4e-8 m after one step.
        # Path to WRITE attach states to while phase 1 trains. Every time the attach
        # gate fires, the full state is appended and the file rewritten. Recording is
        # passive: it reads state and touches NO reward, gate or dynamics, so phase-1
        # training is bit-identical to a run without it. States are stored in
        # chronological order, so "only the converged policy's attaches" is just a
        # tail slice (measured: 540-780 attaches per seed, ~250-310 after convergence).
        # Path to WRITE arrival states to while phase 2 trains -- the mirror of
        # record_attach_states. Recorded when the arrival latch fires, i.e. the load
        # genuinely inside place_radius of the REAL drop-off, so phase 3 can start
        # from states policy 2 actually produces instead of a fabricated drop-off.
        # Passive: reads state only, touches no reward or gate.
        self.record_arrival_states = kwargs.get("record_arrival_states", None)
        self._rec_arrival = None
        # Path to READ arrival states from in phase 3. When set, _phase3_warmup
        # restores one instead of synthesising a start, and place_position stays the
        # REAL configured drop-off rather than being moved under a flailed rod.
        self.phase3_arrival_states = kwargs.get("phase3_arrival_states", None)
        self._arrival_states = None
        self.record_attach_states = kwargs.get("record_attach_states", None)
        self._rec_attach = None
        self.phase2_attach_states = kwargs.get("phase2_attach_states", None)
        self._attach_states = None
        # ---- HELD-OUT SPLIT of the attach bank (validation) ----
        # The bank is a FIXED set (300 states/seed) and phase 2 samples the whole of it,
        # so a phase-2 test re-uses training initial conditions: the drop-off is drawn
        # fresh every episode and is genuinely unseen, but the state the carry starts
        # from is not. This splits the bank by index so a controller can be validated on
        # initial conditions it never trained on.
        #   phase2_attach_split = 0.8, side "train"   -> indices [0, 240)
        #   phase2_attach_split = 0.8, side "holdout" -> indices [240, 300)
        # None (default) = the whole bank, exactly as every run so far. The split is by
        # INDEX, not random, so "train" and "holdout" are reproducible without carrying a
        # seed around, and the bank's own order is already the arbitrary harvest order.
        self.phase2_attach_split = kwargs.get("phase2_attach_split", None)
        self.phase2_attach_side = kwargs.get("phase2_attach_side", "train")
        if self.phase2_attach_split is not None:
            if not 0.0 < float(self.phase2_attach_split) < 1.0:
                raise ValueError(
                    "phase2_attach_split must be in (0, 1) or None (= whole bank), got "
                    f"{self.phase2_attach_split}"
                )
            if self.phase2_attach_side not in ("train", "holdout"):
                raise ValueError(
                    "phase2_attach_side must be 'train' or 'holdout', got "
                    f"{self.phase2_attach_side!r}"
                )
        self.phase2_policy1 = kwargs.get("phase2_policy1", None)
        self.phase2_warmup_max_time = kwargs.get("phase2_warmup_max_time", 4.0)
        self._policy1_model = None
        self.phase2_warmup_amp = kwargs.get("phase2_warmup_amp", 0.6)
        self.phase2_warmup_time = kwargs.get("phase2_warmup_time", (0.15, 0.60))
        # CALIBRATED AGAINST REAL PHASE-1 HAND-OVERS. The warm-up used to reject on a
        # hand-picked grasp height y in [0.3, 0.9], which was a guess. Phase 1's own
        # training logs contain 3292 real ATTACHED events across the five seeds, and
        # they record the two dome OBSERVABLES -- apex height and path length -- so
        # the warm-up now accepts a candidate grasp only if the dome it would produce
        # falls inside the p5-p95 band phase 1 actually produced:
        #     apex_y  0.720 - 0.880   (median 0.800)
        #     length  0.820 - 1.524 m (median 1.180)
        # Matching the observables is the right move because the inverse problem is
        # ill-posed: measured over grasp heights 0.3 -> 0.9, apex_y moves only 0.755
        # -> 0.901 (the bump compensates) and is confounded with horizontal offset,
        # so apex_y cannot identify a grasp height to invert back to.
        # It also kills the straight-rod degeneracy for free: g = (0,1,0) gives
        # apex_y = 1.0, outside the band, so it can never be accepted.
        self.phase2_apex_range = kwargs.get("phase2_apex_range", (0.72, 0.88))
        self.phase2_length_range = kwargs.get("phase2_length_range", (0.82, 1.52))
        # Kept only as a coarse sanity guard; the apex/length test above is the real
        # acceptance criterion.
        self.phase2_grasp_y = kwargs.get("phase2_grasp_y", (0.2, 0.95))
        # Raised from 8: the acceptance region is now a calibrated 2-D band rather
        # than a wide height window, so more draws are needed to land in it.
        self.phase2_warmup_max_tries = kwargs.get("phase2_warmup_max_tries", 30)

        # ---- PICK-AND-PLACE: fixed, constant drop-off location + floor ----
        # place_position has no sensible default -- it must be a deliberately chosen
        # constant (same style as the "boundary" requirement for mode 2, above).
        self.place_position = None
        if self.PICK_AND_RELEASE:
            assert "place_position" in kwargs, (
                "PICK_AND_RELEASE requires place_position: a fixed (3,) location, "
                "constant across all episodes (not randomized, not mode-dependent), "
                "that the arm must carry the sphere to. MUST sit on the floor, "
                "y ~= floor_height + sphere_radius, so the object rests at "
                "place_position after detach (a mid-air target makes it fall away "
                "after release -- see module docstring)."
            )
            self.place_position = np.array(kwargs["place_position"], dtype=np.float64)
            # kept so a randomised drop-off can reuse its FLOOR HEIGHT each episode
            self._place_position_cfg = self.place_position.copy()
        # ---- RANDOMISED DROP-OFF (mirrors phase 1's randomised pick target) ----
        # Phase 1 draws its target uniformly from a BOX because at y in [0.3, 0.9] the
        # arm is dexterous almost everywhere. The drop-off cannot use a box: it must sit
        # ON THE FLOOR (see the module docstring -- a mid-air drop-off makes the object
        # fall away after release), and at floor level the LOADED arm is near the edge
        # of its envelope, where pointing down costs most of the remaining margin.
        #
        # A point is admissible only if the arm can BOTH reach it loaded AND point down
        # there (p_down <= orient_tier_inner), otherwise phase 3 can never release.
        # Measured over 2982 loaded floor-level samples (map_dropoff_workspace.py),
        # fraction that could also point down, by distance from the base:
        #     0.3-0.4 m   0%      the arm cannot curl tightly enough this close in
        #     0.4-0.5 m  34%
        #     0.5-0.6 m  67%   <- the usable band
        #     0.6-0.7 m  28%
        #     0.7-0.8 m   4%
        #     0.8-0.9 m   0%
        # This explains both earlier observations: moving the drop-off from 0.568 m to
        # 0.642 m failed (28% orientable there), while phase 3's synthetic warm-up
        # appeared to succeed at 0.64-0.84 m only because a FLAILING rod reached those
        # points -- not a loaded carry delivering to them.
        #
        # So the drop-off is sampled from an ANNULUS at floor height, not a box:
        # uniform in azimuth (the arm is symmetric about its base) and uniform in
        # radius over the measured band. The default range is the 0.5-0.6 m sweet spot
        # rather than the wider 0.45-0.62 first proposed -- below 0.5 m only ~34% of
        # poses are orientable, which would spawn drop-offs the arm cannot complete.
        # The height is taken from the configured place_position, so it stays on the
        # floor. randomize_place=False keeps the single fixed drop-off.
        self.randomize_place = kwargs.get("randomize_place", False)
        self.place_radius_range = kwargs.get("place_radius_range", (0.50, 0.60))

        # Always-on ground-plane height for FloorForSphere (see reset()).
        self.floor_height = kwargs.get("floor_height", 0.0)
        # -------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """

        This class method, resets and creates the simulation environment. First,
        Elastica rod (or arm) is initialized and boundary conditions acting on the rod defined.
        Second, target and if there are obstacles are initialized and append to the
        simulation. Finally, call back functions are set for Elastica rods and rigid bodies.

        Returns
        -------
        state : numpy.ndarray
        info : dict

        """
        super().reset(seed=seed)

        self.simulator = BaseSimulator()

        # ---- per-episode DROP-OFF, drawn like phase 1 draws its pick target ----
        # Uses self.np_random, so the sequence is fixed by the run's SEED: each seed
        # sees its own stream of drop-offs, exactly as it sees its own pick targets.
        # Drawn HERE, before the sphere, the dome or any warm-up reads place_position.
        if self.randomize_place and self._place_position_cfg is not None:
            r_lo, r_hi = self.place_radius_range
            r = float(self.np_random.uniform(r_lo, r_hi))
            th = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            self.place_position = np.array(
                [r * np.cos(th), self._place_position_cfg[1], r * np.sin(th)],
                dtype=np.float64)

        # setting up test params
        n_elem = self.n_elem
        start = np.zeros((3,))
        direction = np.array([0.0, 1.0, 0.0])  # rod direction: pointing upwards
        normal = np.array([0.0, 0.0, 1.0])
        binormal = np.cross(direction, normal)

        density = 1000
        E = self.E  # Young's Modulus
        poisson_ratio = 0.5

        # Set the arm properties after defining rods
        base_length = 1.0  # rod base length
        radius_tip = 0.05  # radius of the arm at the tip
        radius_base = 0.05  # radius of the arm at the base

        radius_along_rod = np.linspace(radius_base, radius_tip, n_elem)

        # Arm is shearable Cosserat rod
        # 1.0.0: `nu` and `poisson_ratio` are removed from the constructor (both raise).
        # Poisson's ratio enters via shear_modulus; damping is registered separately below.
        self.shearable_rod = CosseratRod.straight_rod(
            n_elem,
            start,
            direction,
            normal,
            base_length,
            base_radius=radius_along_rod,
            density=density,
            youngs_modulus=E,
            shear_modulus=E / (2.0 * (1.0 + poisson_ratio)),
        )

        # Now rod is ready for simulation, append rod to simulation
        self.simulator.append(self.shearable_rod)

        # Ground-plane contact for the ARM. The floor is a real surface at
        # floor_height (same plane the sphere rests on): whenever a rod node
        # reaches it, a normal repulsion (spring k * penetration) plus damping
        # pushes the node back, so the arm rests on the floor instead of passing
        # through it. Dormant until the arm actually touches down. k is kept well
        # below the rod's own axial stiffness (EA/l ~ 1.6e6) so it adds no new
        # stability limit at sim_dt=2e-4.
        floor_plane = Plane(
            plane_origin=np.array([0.0, self.floor_height, 0.0]),
            plane_normal=np.array([0.0, 1.0, 0.0]),
        )
        self.simulator.append(floor_plane)
        self.simulator.detect_contact_between(self.shearable_rod, floor_plane).using(
            RodPlaneContact, k=1.0e5, nu=10.0
        )

        # 1.0.0 damping: physical protocol of AnalyticalLinearDamper.
        # c = NU * element_length reproduces the old constructor-nu damping rates exactly
        # (translational nu/(rho*A) and rotational nu/(rho*I), all axes), derived from the
        # verified 0.1.0.post5 kernels F = nu*v*l, tau = nu*w*l. For NU=30: c = 0.75.
        self.simulator.dampen(self.shearable_rod).using(
            AnalyticalLinearDamper,
            translational_damping_constant=self.NU * (base_length / n_elem),
            rotational_damping_constant=self.NU * (base_length / n_elem),
            time_step=self.h_time_step,
        )

        if self.mode != 2:
            # fixed target position to reach
            target_position = self.target_position

        if self.mode == 2:
            # Paper's Case 2: random target drawn uniformly from the boundary box
            # every reset (changes each episode; the seed fixes the random sequence).
            t_x = self.np_random.uniform(self.boundary[0], self.boundary[1])
            t_y = self.np_random.uniform(self.boundary[2], self.boundary[3])
            if self.dim == 2.0 or self.dim == 2.5:
                t_z = self.np_random.uniform(self.boundary[4], self.boundary[5]) * 0
            elif self.dim == 3.0 or self.dim == 3.5:
                t_z = self.np_random.uniform(self.boundary[4], self.boundary[5])
            print("Target position:", t_x, t_y, t_z)
            target_position = np.array([t_x, t_y, t_z])

        if self.mode == 4:
            # random target position to reach with boundary
            t_x = self.np_random.uniform(self.boundary[0], self.boundary[1])
            t_y = self.np_random.uniform(self.boundary[2], self.boundary[3])
            if self.dim == 2.0 or self.dim == 2.5:
                t_z = self.np_random.uniform(self.boundary[4], self.boundary[5]) * 0
            elif self.dim == 3.0 or self.dim == 3.5:
                t_z = self.np_random.uniform(self.boundary[4], self.boundary[5])

            print("Target position:", t_x, t_y, t_z)
            target_position = np.array([t_x, t_y, t_z])

        # PHASES 2 and 3 with a SYNTHETIC warm-up: the episode starts ALREADY GRASPED,
        # so the object spawns at the rod tip -- the only place a grasp can physically
        # be. The tip of the freshly built straight rod is the grasp point g, and the
        # dome is built from it in the phase-2 block after finalize().
        # Phase 3 must be included: it also starts attached (it runs the phase-2
        # warm-up first). Spawning its object at the random mode-2 target instead left
        # the gated joint straddling a gap of up to ~1 m at the instant it switched on,
        # so a k=1e3 spring fired ~1000 N at a 0.05 kg ball on the first step.
        # With a POLICY-1 REPLAY warm-up the object must instead spawn where policy 1
        # expects to find it -- the random mode-2 target -- so the override is skipped.
        if self.phase in (2, 3) and self.phase2_policy1 is None:
            # SPAWNED AT SURFACE CONTACT, not at the tip node (fixed 2026-09-04).
            # This used to be the bare tip node, i.e. zero separation between two solid
            # bodies -- the payload fully inside the rod. That was consistent with the old
            # centre-referenced attach gate, but it is the wrong geometry now that the
            # gate is surface-referenced, and with the stiff joint it made the synthetic
            # warm-up DIVERGE outright rather than merely look wrong: the rest pose was
            # captured at ~0 offset, so the joint had no lever and any relative motion
            # produced unbounded corrective forces.
            # The payload is therefore placed one contact distance out along the tip's own
            # axis, which is exactly the pose the attach gate now produces for a real
            # grasp, so the synthetic start and the live hand-off agree.
            _P = self.shearable_rod.position_collection
            _tip = _P[..., -1].copy()
            _axis = _tip - _P[..., -2]
            _n = np.linalg.norm(_axis)
            _axis = _axis / _n if _n > 1e-12 else np.array([0.0, 1.0, 0.0])
            # r_payload is the Sphere's base_radius set just below; r_tip is the rod's
            # own tip radius. Both are fixed geometry, so this needs no live sphere.
            _contact = float(self.shearable_rod.radius[-1]) + 0.05
            target_position = _tip + _axis * _contact

        # initialize sphere -- a real dynamic body (mass from density; gravity added below)
        # 1.0.0: Sphere requires an ndarray center (a plain list crashes on center.size)
        target_position = np.array(target_position, dtype=np.float64)
        # Draw this episode's payload BEFORE the sphere is built, so the mass is real
        # from step 1 rather than being patched in afterwards.
        if self.sphere_density_set is not None:
            self.SPHERE_DENSITY = float(
                self.sphere_density_set[
                    int(self.np_random.integers(len(self.sphere_density_set)))
                ]
            )
        self.sphere = Sphere(
            center=target_position,  # initialize target position of the ball
            base_radius=0.05,
            density=self.SPHERE_DENSITY if self.PICK_AND_RELEASE else 1000,
        )

        # JOINT-DAMPING STABILITY GUARD. The joint damper is integrated explicitly, so it
        # is stable only while nu * dt < m_payload; above that a single step overshoots
        # and the run diverges to NaN a few hundred steps later, inside PyElastica's
        # dissipation kernel, with nothing pointing back here. That is exactly what the
        # original rho=100 payload (0.052 kg) does against NU_ATTACH=800: nu*dt/m = 3.06.
        # Fail loudly at reset instead. The bound is a property of the gains and the
        # timestep -- do NOT silence it by re-tuning NU_ATTACH per payload, which is the
        # mistake that produced the mass-dependent grasp in the first place; either raise
        # the payload above the floor or lower sim_dt.
        if self.PICK_AND_RELEASE:
            _m = float(self.sphere.mass.sum())
            _floor = self.NU_ATTACH * self.time_step
            assert _m > _floor, (
                f"payload {_m:.4f} kg is below the joint-damping stability floor "
                f"{_floor:.4f} kg (= NU_ATTACH {self.NU_ATTACH} * sim_dt {self.time_step}). "
                f"nu*dt/m = {_floor/_m:.2f} must stay below 1. Raise the payload "
                f"(>= {_floor/7.853982*100:.1f}% of rod mass), lower NU_ATTACH, or lower sim_dt."
            )

        if self.mode == 3:
            self.dir_indicator = 1
            self.sphere_initial_velocity = self.target_v
            self.sphere.velocity_collection[..., 0] = [
                self.sphere_initial_velocity,
                0.0,
                0.0,
            ]

        if self.mode == 4:

            self.trajectory_iteration = 0  # for changing directions
            self.rand_direction_1 = np.pi * self.np_random.uniform(0, 2)
            if self.dim == 2.0 or self.dim == 2.5:
                self.rand_direction_2 = np.pi / 2.0
            elif self.dim == 3.0 or self.dim == 3.5:
                self.rand_direction_2 = np.pi * self.np_random.uniform(0, 2)

            self.v_x = (
                self.target_v
                * np.cos(self.rand_direction_1)
                * np.sin(self.rand_direction_2)
            )
            self.v_y = (
                self.target_v
                * np.sin(self.rand_direction_1)
                * np.sin(self.rand_direction_2)
            )
            self.v_z = self.target_v * np.cos(self.rand_direction_2)

            self.sphere.velocity_collection[..., 0] = [
                self.v_x,
                self.v_y,
                self.v_z,
            ]
            self.boundaries = np.array(self.boundary)

        if self.mode == 1:
            theta_x = 0
            theta_y = np.pi / 4
            theta_z = 0
        if self.mode == 2 or self.mode == 4:
            theta_x = 0
            theta_y = self.np_random.uniform(-np.pi / 2, np.pi / 2)
            theta_z = 0

        # set the orientation of target sphere
        theta = np.array([theta_x, theta_y, theta_z])
        print(theta)
        R = np.array(
            [
                [
                    -np.sin(theta[1]),
                    np.sin(theta[0]) * np.cos(theta[1]),
                    np.cos(theta[0]) * np.cos(theta[1]),
                ],
                [
                    np.cos(theta[1]) * np.cos(theta[2]),
                    np.sin(theta[0]) * np.sin(theta[1]) * np.cos(theta[2])
                    - np.sin(theta[2]) * np.cos(theta[0]),
                    np.sin(theta[1]) * np.cos(theta[0]) * np.cos(theta[2])
                    + np.sin(theta[0]) * np.sin(theta[2]),
                ],
                [
                    np.sin(theta[2]) * np.cos(theta[1]),
                    np.sin(theta[0]) * np.sin(theta[1]) * np.sin(theta[2])
                    + np.cos(theta[0]) * np.cos(theta[2]),
                    np.sin(theta[1]) * np.sin(theta[2]) * np.cos(theta[0])
                    - np.sin(theta[0]) * np.cos(theta[2]),
                ],
            ]
        )
        self.sphere.director_collection[..., 0] = R
        self.simulator.append(self.sphere)

        Q = self.sphere.director_collection[..., 0]
        # Compute target tip orientation using quaternions.
        # We add target and arm tip orientations difference to reward function.
        qw = np.sqrt(1 + Q[0, 0] + Q[1, 1] + Q[2, 2]) / 2
        qx = (Q[2, 1] - Q[1, 2]) / (4 * qw)
        qy = (Q[0, 2] - Q[2, 0]) / (4 * qw)
        qz = (Q[1, 0] - Q[0, 1]) / (4 * qw)
        self.target_tip_orientation = np.array([qw, qx, qy, qz])

        # Phase-2 placement orientation q_place: defaults to the grasp pose q*
        # (= target_tip_orientation). R2 rewards holding this tip orientation while the
        # sphere is transported to place_position.
        self.place_orientation = self.target_tip_orientation

        # ------------------- PICK-AND-PLACE REGISTRATION -------------------
        # PyElastica requires all forcing/constraints/connections BEFORE finalize().
        # The joint and anchor are therefore registered here and gated at runtime via
        # self.attach_state -- the same pre-registered/live-state pattern used by
        # MuscleTorquesWithVaryingBetaSplines. No manual toggling between runs, ever:
        # attach_state is rebuilt fresh each reset and flipped by the reach/orient gates.
        if self.PICK_AND_RELEASE:
            self.attach_state = {"attached": False, "ever_attached": False}

            # Gravity on the sphere: it is a real weighted body. The ARM stays
            # gravity-free exactly as in the original Case 2 setup.
            self.simulator.add_forcing_to(self.sphere).using(
                GravityForces, acc_gravity=np.array([0.0, -9.80665, 0.0])
            )

            # Support/pedestal holding the sphere until it is picked
            self.simulator.constrain(self.sphere).using(
                GatedSphereAnchor,
                attach_state=self.attach_state,
                anchor_position=np.array(target_position, dtype=np.float64).copy(),
            )

            # Real bidirectional mechanical coupling: rod tip (node/element -1) <-> sphere (0)
            self.simulator.connect(
                first_rod=self.shearable_rod,
                second_rod=self.sphere,
                first_connect_idx=-1,
                second_connect_idx=0,
            ).using(
                GatedFixedJoint,
                k=self.K_ATTACH,
                nu=self.NU_ATTACH,
                kt=self.KT_ATTACH,
                nut=self.NUT_ATTACH,
                attach_state=self.attach_state,
            )

            # Always-on ground plane for the sphere (see FloorForSphere docstring).
            # Registered unconditionally alongside gravity/anchor/joint. Geometrically
            # dormant pre-pick (anchor holds the sphere above floor_height + radius) and
            # while the sphere is carried above the floor; engages only if the sphere
            # reaches ground level.
            self.simulator.constrain(self.sphere).using(
                FloorForSphere,
                floor_height=self.floor_height,
            )
        # ---------------------------------------------------------------------

        class WallBoundaryForSphere(FreeBC):
            """

            This class generates a bounded space that sphere can move inside. If sphere
            hits one of the boundaries (walls) of this space, it is reflected in opposite direction
            with the same velocity magnitude.

            """

            def __init__(self, boundaries, **kwargs):
                super().__init__(**kwargs)
                self.x_boundary_low = boundaries[0]
                self.x_boundary_high = boundaries[1]
                self.y_boundary_low = boundaries[2]
                self.y_boundary_high = boundaries[3]
                self.z_boundary_low = boundaries[4]
                self.z_boundary_high = boundaries[5]

            def constrain_values(self, sphere, time):
                pos_x = sphere.position_collection[0]
                pos_y = sphere.position_collection[1]
                pos_z = sphere.position_collection[2]

                radius = sphere.radius

                vx = sphere.velocity_collection[0]
                vy = sphere.velocity_collection[1]
                vz = sphere.velocity_collection[2]

                if (pos_x - radius) < self.x_boundary_low:
                    sphere.velocity_collection[:] = np.array([-vx, vy, vz])

                if (pos_x + radius) > self.x_boundary_high:
                    sphere.velocity_collection[:] = np.array([-vx, vy, vz])

                if (pos_y - radius) < self.y_boundary_low:
                    sphere.velocity_collection[:] = np.array([vx, -vy, vz])

                if (pos_y + radius) > self.y_boundary_high:
                    sphere.velocity_collection[:] = np.array([vx, -vy, vz])

                if (pos_z - radius) < self.z_boundary_low:
                    sphere.velocity_collection[:] = np.array([vx, vy, -vz])

                if (pos_z + radius) > self.z_boundary_high:
                    sphere.velocity_collection[:] = np.array([vx, vy, -vz])

            def constrain_rates(self, sphere, time):
                pass

        if self.mode == 4:
            self.simulator.constrain(self.sphere).using(
                WallBoundaryForSphere, boundaries=self.boundaries
            )

        # Add boundary constraints as fixing one end
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )

        # Add muscle torques acting on the arm for actuation.
        # MuscleTorquesWithVaryingBetaSplines uses the control points selected by RL to
        # generate torques along the arm.
        self.torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)
        self.spline_points_func_array_normal_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_normal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("normal"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_normal_dir,
        )

        self.torque_profile_list_for_muscle_in_binormal_dir = defaultdict(list)
        self.spline_points_func_array_binormal_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_binormal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("binormal"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_binormal_dir,
        )

        self.torque_profile_list_for_muscle_in_twist_dir = defaultdict(list)
        self.spline_points_func_array_twist_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_twist_dir,
            muscle_torque_scale=self.beta,
            direction=str("tangent"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_twist_dir,
        )

        # Call back function to collect arm data from simulation
        class ArmMuscleBasisCallBack(CallBackBaseClass):
            """
            Call back function for Elastica rod or arm
            """

            def __init__(
                self,
                step_skip: int,
                callback_params: dict,
            ):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["directors"].append(
                        system.director_collection.copy()
                    )
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    return

        # Call back function to collect target sphere data from simulation
        class RigidSphereCallBack(CallBackBaseClass):
            """
            Call back function for target sphere
            """

            def __init__(self, step_skip: int, callback_params: dict):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["directors"].append(
                        system.director_collection.copy()
                    )
                    self.callback_params["radius"].append(copy.deepcopy(system.radius))
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    return

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            # Collect data using callback function for postprocessing
            self.post_processing_dict_rod = defaultdict(list)
            # List which collected data will be append
            # set the diagnostics for rod and collect data
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                ArmMuscleBasisCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_rod,
            )

            self.post_processing_dict_sphere = defaultdict(list)
            # List which collected data will be append
            # set the diagnostics for target sphere and collect data
            self.simulator.collect_diagnostics(self.sphere).using(
                RigidSphereCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_sphere,
            )

        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()

        # 1.0.0: extend_stepper_interface/do_step is deprecated and its deprecation shim
        # is broken (crashes on `from warning import warn`). The stepper instance itself
        # is now stateful: self.StatefulStepper.step(simulator, time, dt).

        # pick-and-place episode flag: becomes True once the object is placed (detached
        # at the dome end). attach_state itself is (re)built in the registration block
        # above, fresh each episode. Must be set BEFORE get_state() (which reads it to
        # pick the active-phase tracking target).
        self.released = False

        # Dome carry path state -- filled in at the attach instant (the dome start g is
        # by definition the grasp position, unknown until then), or right below when
        # phase 2 starts already grasped.
        self._dome_start = None
        self._dome_bump = 0.0
        self._carry_t0 = 0.0
        self._carry_duration = None
        # settle-window state: True between the grasp and the start of the carry
        self._settling = False
        self._settle_t0 = 0.0
        # phase-1 flag: set when the grasp fires, ends the episode (see step()).
        self._phase1_done = False
        # ARRIVAL LATCH: False = phase 2 (carry) owns the reward, True = phase 3
        # (orient + release) owns it. Latched in the phase-2 block the step the load
        # first reaches the drop-off, and never cleared -- phase 3 keeps the episode
        # even if its own rotation pushes the load back out of the band.
        self._arrived = False
        # phase-2 flag: set at the arrival, ends a PHASE-2 TRAINING episode there
        # (policy 3 owns everything past that point, so policy 2 must not be scored
        # on it). Ignored when phase == 0 or 3.
        self._phase2_done = False
        # phase-3 flag: set at the RELEASE, ends a PHASE-3 TRAINING episode there.
        # Without it the episode continues with the object on the floor and the rod
        # FREED -- an unloaded rod points down trivially, so R3 would pay up to
        # 3.0/step for the rest of the episode for doing nothing. That post-release
        # reward is unearned, dwarfs the loaded orienting the policy is meant to
        # learn, and is exactly the unloaded-frame artefact that made earlier
        # orientation numbers look far better than they were. Ignored when phase is
        # 0 (the hand-off must run on so the placement can be filmed and settle).
        self._phase3_done = False

        # Restart the episode clock HERE, before the phase-2 dome is built. _setup_dome
        # stamps _carry_t0 = time_tracker, so if the clock were still holding the PREVIOUS
        # episode's end time (~final_time) the dome's schedule would start in the future:
        # _carry_progress() would compute (0 - final_time)/carry_duration, clamp to 0, and
        # s would stay 0 for the WHOLE episode -- the commanded goal would never leave the
        # grasp point and the policy would be rewarded for simply holding the object still.
        self.time_tracker = np.float64(0.0)
        self._episode_t0 = self.time_tracker      # for the carry-fits check

        # PHASE 2 (policy 2): begin the episode already holding the object. The joint
        # is switched on and the anchor off via the same runtime flags the attach gate
        # uses, then the warm-up drives the rod to a real loaded configuration and the
        # dome is built from the grasp point it ends at -- so from step 1 the reward is
        # R2 and the task is purely the loaded carry. No reward/bonus/penalty changes;
        # the episode simply starts where phase 1 would have handed over.
        if self.phase in (2, 3):
            self.attach_state["attached"] = True
            self.attach_state["ever_attached"] = True

        # PHASE 2 only. Phase 3 deliberately SKIPS this: the phase-2 warm-up spends up
        # to phase2_warmup_max_tries draws rejection-sampling a grasp whose dome matches
        # phase 1's logged apex/length band, and _phase3_warmup then flails the rod low
        # and moves the drop-off under it -- destroying that grasp and never reading the
        # dome again. Phase 3 needs only "object attached to the tip", which the tip
        # spawn already gives it. (phase3_policy2 replay builds its own dome, since it
        # drives policy 2, which does read it.)
        if self.phase == 2:
            path = self._phase2_warmup()
            g = self._dome_start
            print(
                " PHASE 2 START -- grasp (%.2f, %.2f, %.2f), dome: rise %+.2f m,"
                " apex_y %.2f, length %.2f m, carry %.2f s"
                % (g[0], g[1], g[2], path[:, 1].max() - g[1], path[:, 1].max(),
                   self._dome_arclen[-1], self._carry_duration)
            )

        # PHASE 3 (policy 3): the episode must START at the drop-off, holding the load.
        # Random torques cannot put a flailing rod within 5 cm of a specific point, so
        # the TRAINED policy 2 carries it there and policy 3 takes over from where it
        # stops -- the same states policy 3 inherits at test time, so no distribution
        # gap opens at the phase-2 -> phase-3 hand-over.
        if self.phase == 3:
            # place_position was drawn per-episode at the top of reset(). The two
            # phase-3 start modes treat it differently, deliberately:
            #   arrival-state restore : keeps it -- the state was recorded at the REAL
            #                           drop-off, so the geometry must match.
            #   synthetic warm-up     : OVERWRITES it, since that mode fabricates a
            #                           drop-off underneath a flailed rod. The
            #                           per-episode draw is therefore irrelevant there.
            reached = self._phase3_warmup()
            self._arrived = True
            m0 = float(np.linalg.norm(
                self.sphere.position_collection[..., 0] - self.place_position))
            print(
                " PHASE 3 START -- %s, load %.3f m from the drop-off (%.2f, %.2f, %.2f)%s"
                % (
                    ("real policy-2 arrival" if self.phase3_arrival_states
                     else "policy-2 replay" if self.phase3_policy2 else "synthetic"),
                    m0, self.place_position[0], self.place_position[1],
                    self.place_position[2],
                    "" if reached else "  (WARM-UP MISSED THE BAND)",
                )
            )

        # ---- CALIBRATION PROBE (frozen estimator only) ----
        # Runs here, at the end of reset and OUTSIDE the policy's action loop, so the
        # policy never sees these steps. It gives the estimate the observation reports
        # from step 1: for phase 1 the arm is empty and this reads ~0; for phases 2 and 3
        # the warm-up has already attached the payload, so it reads the load.
        # The probe MOVES the arm, so phase 2's dome -- built from the grasp point by the
        # warm-up above -- is rebuilt from where the payload actually ends up, otherwise
        # the carry would be commanded from a point the load has already left.
        if self._est_model is not None:
            self._run_probe()
            if self.phase == 2 and self._dome_start is not None:
                _p = self.sphere.position_collection[..., 0].copy()
                if np.all(np.isfinite(_p)):
                    self._setup_dome(_p)
            self.time_tracker = np.float64(0.0)
            self._episode_t0 = self.time_tracker

        # set state
        state = self.get_state()

        # reset on_goal
        self.on_goal = 0
        # reset current_step
        self.current_step = 0
        # (time_tracker was already reset above, before the phase-2 dome setup)
        # reset previous_action
        self.previous_action = None

        # wall-clock timer for the per-episode compute-time report printed at truncation
        self._episode_wall_start = time.perf_counter()

        # After resetting the environment return state information
        return state, {}

    def sampleAction(self):
        """
        Sample usable random actions are returned.

        Returns
        -------
        numpy.ndarray
            1D (3 * number_of_control_points,) array containing data with 'float' type, in range [-1, 1].
        """
        random_action = (np.random.rand(1 * self.number_of_control_points) - 0.5) * 2
        return random_action

    def _phase3_warmup(self):
        """PHASE-3 start state: the load held at the drop-off.

        DEFAULT (phase3_policy2 unset): INDEPENDENT of policy 2, by the same inversion
        that makes phase 2 independent of policy 1. _phase2_warmup does not drive the
        rod to a predetermined grasp point -- it flails, rejection-samples the height,
        and then BUILDS THE DOME FROM WHEREVER THE ROD ENDED UP. Phase 3 inverts the
        same way: rather than steering the rod to the drop-off (which random torques
        essentially never hit within 5 cm), it puts THE DROP-OFF WHERE THE ROD IS.

        Concretely: flail until the load is within place_radius of FLOOR HEIGHT --
        the long warm-ups that _phase2_warmup rejects for flopping to the floor are
        exactly the ones wanted here -- then set the drop-off directly beneath the
        load, at the configured drop-off height, with a little horizontal jitter so
        episodes start spread across the band instead of always dead centre.

        The drop-off therefore MOVES between phase-3 training episodes. That is safe
        because it is observable: once _arrived is latched, get_state() reports
        place_position as the tracking target, so the policy is told where it is. It
        also makes policy 3 learn "point down at the drop-off you are shown" rather
        than memorising one location.

        OPTIONAL (phase3_policy2 set): replay a trained policy 2 to the drop-off
        instead. Higher fidelity -- the true arrival poses -- at the cost of the
        independence, so it is a validation mode rather than the default.

        Touches NO reward, bonus or penalty -- only the state phase 3 starts from.

        Returns True if the load ended up inside the band.
        """
        if self.phase3_arrival_states is not None:
            return self._phase3_warmup_restore()
        if self.phase3_policy2 is None:
            return self._phase3_warmup_synthetic()
        return self._phase3_warmup_replay()

    def _phase3_warmup_restore(self):
        """PHASE-3 start restored from a REAL policy-2 arrival state.

        The synthetic warm-up fabricates its start: it flails the rod low and then moves
        place_position underneath wherever the load ended up. That guarantees the
        episode begins "arrived", but the arm's shape and momentum bear no relation to
        one that has just completed a carry -- and the drop-off is not the real one.

        Restoring a recorded arrival fixes both: the configuration is exactly what
        policy 2 produced on reaching the drop-off, and place_position stays the REAL
        configured target, so phase 3 trains on the actual task geometry.

        Touches NO reward, bonus or penalty -- only the state phase 3 starts from.
        """
        if self._arrival_states is None:
            d = np.load(self.phase3_arrival_states)
            self._arrival_states = {k: d[k] for k in d.files}
            print(" PHASE 3: loaded %d real policy-2 arrival states from %s"
                  % (len(self._arrival_states["q_place"]),
                     self.phase3_arrival_states), flush=True)
        bank = self._arrival_states
        i = int(self.np_random.integers(len(bank["q_place"])))
        for k in ("position_collection", "velocity_collection",
                  "director_collection", "omega_collection"):
            getattr(self.shearable_rod, k)[:] = bank[f"rod_{k}"][i]
            getattr(self.sphere, k)[:] = bank[f"sph_{k}"][i]
        self.place_orientation = bank["q_place"][i].copy()
        self.target_tip_orientation = self.place_orientation
        # Restore the drop-off this state was recorded at, overriding the per-episode
        # random draw -- otherwise the load sits at the OLD target while phase 3 aims
        # at a new one, and the episode starts ~1 m away instead of arrived.
        if "place_position" in bank:
            self.place_position = bank["place_position"][i].copy()
        self.time_tracker = np.float64(0.0)
        self._episode_t0 = self.time_tracker
        # No dome is needed: _arrived is latched by the caller, so get_state reports
        # place_position as the tracking target and _carry_progress is never consulted.
        return True

    def _phase3_warmup_synthetic(self):
        """Independent phase-3 start: flail low, then put the drop-off under the load."""
        n_cp = self.number_of_control_points
        drop_y = float(self.place_position[1])
        lo, hi = self.phase3_load_y
        ok = False
        for _ in range(self.phase3_warmup_max_tries):
            amp = self.phase2_warmup_amp
            act = self.np_random.uniform(-amp, amp, 3 * n_cp)
            self.spline_points_func_array_normal_dir[:] = act[:n_cp]
            self.spline_points_func_array_binormal_dir[:] = act[n_cp : 2 * n_cp]
            self.spline_points_func_array_twist_dir[:] = act[2 * n_cp :]
            # deliberately LONGER than the phase-2 warm-up: those are the runs that
            # let the arm settle toward the floor, which phase 2 rejects and phase 3
            # wants (see _phase2_warmup's measured 25-58% acceptance note).
            t_warm = self.np_random.uniform(*self.phase3_warmup_time)
            for _ in range(max(int(t_warm / self.time_step), 1)):
                self.time_tracker = self.StatefulStepper.step(
                    self.simulator, self.time_tracker, self.time_step
                )
            pos = self.sphere.position_collection[..., 0]
            if np.all(np.isfinite(pos)) and lo <= pos[1] <= hi:
                ok = True
                break

        pos = self.sphere.position_collection[..., 0].copy()
        if not np.all(np.isfinite(pos)):
            return False
        # Drop-off directly beneath the load at the configured height, plus horizontal
        # jitter. Budgeted so the total offset stays inside place_radius, i.e. the
        # episode always starts legitimately "arrived".
        dy = float(pos[1]) - drop_y
        room = max(self.place_radius ** 2 - dy ** 2, 0.0) ** 0.5
        ang = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        rad = float(self.np_random.uniform(0.0, 0.8 * room))
        self.place_position = np.array(
            [pos[0] + rad * np.cos(ang), drop_y, pos[2] + rad * np.sin(ang)],
            dtype=np.float64,
        )
        self.time_tracker = np.float64(0.0)
        self._episode_t0 = self.time_tracker
        return ok and abs(dy) <= self.place_radius

    def _phase3_warmup_replay(self):
        """Validation mode: let a TRAINED policy 2 carry the load to the drop-off."""
        # policy 2 reads the dome waypoint from the observation, so it must exist
        # before the replay starts. (The synthetic phase-3 warm-up needs no dome.)
        if self._dome_start is None:
            self._phase2_warmup_synthetic()
        model = self._load_policy2()
        if model is None:
            return False
        n_cp = self.number_of_control_points
        max_steps = max(
            1,
            int(self.phase3_warmup_max_time
                / (self.time_step * self.num_steps_per_update)),
        )
        reached = False
        for _ in range(max_steps):
            act, _ = model.predict(self.get_state(), deterministic=True)
            self.spline_points_func_array_normal_dir[:] = act[:n_cp]
            self.spline_points_func_array_binormal_dir[:] = act[n_cp : 2 * n_cp]
            self.spline_points_func_array_twist_dir[:] = act[2 * n_cp :]
            for _ in range(self.num_steps_per_update):
                self.time_tracker = self.StatefulStepper.step(
                    self.simulator, self.time_tracker, self.time_step
                )
            pos = self.sphere.position_collection[..., 0]
            if not np.all(np.isfinite(pos)):
                break
            if np.linalg.norm(pos - self.place_position) <= self.place_radius:
                reached = True
                break
        # Hand phase 3 a FULL episode: the clock is rewound so the replay does not eat
        # policy 3's step budget. Safe because phase 3 never reads the dome -- once
        # _arrived is set, get_state() reports place_position as the tracking target,
        # so _carry_progress() is not consulted again this episode.
        self.time_tracker = np.float64(0.0)
        self._episode_t0 = self.time_tracker
        return reached

    def _load_policy2(self):
        """Lazily load the phase-2 policy used to seed phase-3 episodes."""
        if self._policy2_model is not None:
            return self._policy2_model
        if self.phase3_policy2 is None:
            raise ValueError(
                "phase 3 needs phase3_policy2=<path to a trained policy 2> so each "
                "episode can start at the drop-off"
            )
        from stable_baselines3 import SAC   # local: keeps the env import SB3-free

        self._policy2_model = SAC.load(self.phase3_policy2, device="cpu")
        return self._policy2_model

    def _dome_stats(self, g):
        """(apex_y, arc length) of the dome a grasp at g WOULD produce, without
        committing any state. Replicates _setup_dome's bump and reachability shrink
        exactly, so the warm-up can test a candidate grasp against the real phase-1
        distribution before accepting it.
        """
        bump = max(self.dome_apex - 0.5 * (g[1] + self.place_position[1]), 0.0)
        s = np.linspace(0.0, 1.0, 101)
        r_limit = max(
            self.dome_reach_max,
            float(np.linalg.norm(g)),
            float(np.linalg.norm(self.place_position)),
        )
        for _ in range(40):
            path = g + (self.place_position - g) * s[:, None]
            path[:, 1] += bump * np.sin(np.pi * s)
            if bump <= 0.0 or np.linalg.norm(path, axis=1).max() <= r_limit:
                break
            bump *= 0.85
        path = g + (self.place_position - g) * s[:, None]
        path[:, 1] += bump * np.sin(np.pi * s)
        return (
            float(path[:, 1].max()),
            float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        )

    def _dome_point(self, s):
        """Phase-2 dome carry path, s in [0, 1]:
        T2(s) = g + (p - g)*s + bump*sin(pi*s)*y_hat, with g the grasp position,
        p = place_position and bump = max(dome_apex - 0.5*(g_y + p_y), 0).
        T2(0) = g and T2(1) = place_position exactly (sin term vanishes at both
        ends), so the path starts where the arm grasped and ends at the drop-off.
        """
        point = self._dome_start + (self.place_position - self._dome_start) * s
        point[1] += self._dome_bump * np.sin(np.pi * s)
        return point

    def _dome_velocity(self, s):
        """Velocity of the phase-2 goal T2(s(t)), ANALYTIC.

        s advances at constant SPEED along arc length, so |dT2/dt| = carry_speed exactly
        and only the direction varies:
            dT2/ds  = (p - g) + bump*pi*cos(pi*s)*y_hat
            v_goal  = carry_speed * dT2/ds / |dT2/ds|
        Computed in closed form rather than by finite-differencing T2 between steps: the
        dome is REBUILT when the settle window closes, so a difference across that
        instant sees a discontinuous jump and reports hundreds of m/s (measured: 651,
        523, 245 m/s spikes). The closed form has no such artefact.

        Returns the zero vector whenever the goal is not moving -- while settling (s is
        frozen at 0) and once the schedule has clamped at s = 1 -- so a velocity penalty
        built on this asks the load to be STILL exactly when the goal is still.
        """
        if self._dome_start is None or self._settling or s >= 0.999:
            return np.zeros(3)
        d = (self.place_position - self._dome_start).astype(np.float64).copy()
        d[1] += self._dome_bump * np.pi * np.cos(np.pi * s)
        n = np.linalg.norm(d)
        return d * (self.carry_speed / n) if n > 1e-12 else np.zeros(3)

    def _setup_dome(self, g):
        """Build the dome carry path from grasp point g to place_position, and start
        its constant-speed clock. Called at the attach instant (combined / phase-1
        training) or at reset() when phase 2 starts already grasped. Returns the
        sampled path so the caller can report its apex/length.
        """
        self._dome_start = g
        self._dome_bump = max(
            self.dome_apex - 0.5 * (g[1] + self.place_position[1]), 0.0
        )
        self._dome_s_grid = np.linspace(0.0, 1.0, 101)
        # REACHABILITY: shrink the arch until every commanded point is inside the
        # arm's reach. Both endpoints are reachable by construction (g is where the
        # rod tip actually is; place_position is a verified drop-off), and the chord
        # between them cannot exceed max(|g|,|p|) -- only the sin bump can push a
        # point out, so reducing the bump is sufficient and never moves the endpoints.
        # The limit is the largest radius already demonstrated reachable for THIS
        # episode: the grasp point g (the rod tip is physically there, so |g| is
        # reachable by proof), the verified drop-off, or the conservative default --
        # whichever is greatest. Shrinking the bump alone is sufficient because the
        # chord between two reachable endpoints never exceeds max(|g|,|p|) (the norm
        # is convex, so it peaks at an endpoint); only the sin term can push out.
        r_limit = max(
            self.dome_reach_max,
            float(np.linalg.norm(g)),
            float(np.linalg.norm(self.place_position)),
        )
        for _ in range(40):
            path = np.array([self._dome_point(s) for s in self._dome_s_grid])
            if (
                self._dome_bump <= 0.0
                or np.linalg.norm(path, axis=1).max() <= r_limit
            ):
                break
            self._dome_bump *= 0.85
        path = np.array([self._dome_point(s) for s in self._dome_s_grid])
        # cumulative arc length: drives BOTH the total duration and the
        # constant-speed s(t) inversion in _carry_progress
        self._dome_arclen = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        )
        self._carry_duration = max(
            self._dome_arclen[-1] / self.carry_speed, self.time_step
        )
        self._carry_t0 = self.time_tracker
        # Enter the settle window: s stays 0 (goal parked at the grasp point) until
        # the load has slowed, then _update_settle rebuilds the dome from where it
        # actually stabilised and starts the clock. See the settle_speed comment.
        self._settling = True
        self._settle_t0 = self.time_tracker
        # The goal must be able to finish its path before the episode truncates,
        # otherwise s never reaches 1, the object is never commanded to the drop-off
        # and no release can fire. Warn loudly rather than fail silently.
        remaining = self.final_time - (self._carry_t0 - self._episode_t0)
        if self._carry_duration > remaining:
            print(
                " WARNING: carry needs %.2f s but only %.2f s of episode remain --"
                " the goal will not reach place_position before truncation."
                % (self._carry_duration, remaining)
            )
        return path

    def _record_state(self, path, buf_name):
        """Append the full state to `path`. Shared by the attach and arrival recorders.

        PASSIVE: reads state only. No reward, gate, threshold or dynamic is touched.

        Stores position, velocity, directors and angular velocity for BOTH bodies plus
        the grasp pose q_place -- not merely the point in space, because the tip can be
        at the same point with a straight or curled rod, stationary or whipping, and
        those produce completely different continuations. Rewrites every 25 entries so
        a run killed part-way still leaves a usable set, in chronological order so
        "converged-policy states only" is a tail slice.
        """
        keys = ("position_collection", "velocity_collection",
                "director_collection", "omega_collection")
        buf = getattr(self, buf_name)
        if buf is None:
            buf = {f"rod_{k}": [] for k in keys}
            buf.update({f"sph_{k}": [] for k in keys})
            buf["q_place"] = []
            # The drop-off MUST travel with the state. An arrival state means "the load
            # is at ITS drop-off"; with a randomised drop-off, reset() draws a fresh one
            # and a restore would otherwise place the load ~1 m from the new target
            # instead of arrived (measured 0.55-1.19 m before this was added).
            buf["place_position"] = []
            setattr(self, buf_name, buf)
        for k in keys:
            a, b = getattr(self.shearable_rod, k), getattr(self.sphere, k)
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                return                       # never bank a diverged state
        for k in keys:
            buf[f"rod_{k}"].append(getattr(self.shearable_rod, k).copy())
            buf[f"sph_{k}"].append(getattr(self.sphere, k).copy())
        buf["q_place"].append(self.place_orientation.copy())
        buf["place_position"].append(np.array(self.place_position, dtype=np.float64))
        n = len(buf["q_place"])
        if n % 25 == 0:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.savez_compressed(path, **{k: np.array(v) for k, v in buf.items()})
            print(" recorded %d states -> %s" % (n, path), flush=True)

    def _record_arrival(self):
        """State at the arrival latch, for phase 3 to start from."""
        self._record_state(self.record_arrival_states, "_rec_arrival")

    def _record_attach(self):
        """State at the phase-1 attach gate, for phase 2 to start from."""
        self._record_state(self.record_attach_states, "_rec_attach")

    def _update_settle(self):
        """End the settle window once the load has slowed, and start the carry there.

        Called once per env step while _settling. On exit the dome is REBUILT from the
        object's current position, so T(0) coincides exactly with where the load is and
        the tracking error starts at zero with the load already slow -- instead of the
        goal sitting on a point the object left at several m/s.
        """
        if not self._settling:
            return
        v = float(np.linalg.norm(self.sphere.velocity_collection[..., 0]))
        v_thresh = (self.carry_speed if self.settle_speed is None
                    else self.settle_speed)
        elapsed = self.time_tracker - self._settle_t0
        timed_out = elapsed >= self.settle_max_time
        if elapsed < self.settle_min_time:
            return
        if v > v_thresh and not timed_out:
            return
        # capture the duration BEFORE _setup_dome, which re-stamps _settle_t0
        self._settle_dur = float(elapsed)
        # _setup_dome re-enters the settle window, so clear the flag AFTER it returns
        pos = self.sphere.position_collection[..., 0].copy()
        if np.all(np.isfinite(pos)):
            self._setup_dome(pos)
        self._settling = False
        self._carry_t0 = self.time_tracker

    def _payload_radius(self):
        r = self.sphere.radius
        return float(r[0]) if np.ndim(r) else float(r)

    def _probe_routine(self, i):
        """The CALIBRATION ROUTINE: a fixed 20 Hz sine, byte-identical to the one
        collect_calibration_data.py recorded the training set under. If these two ever
        differ the estimator is fed an excitation it never saw, so this is deliberately a
        single expression duplicated in one other place and nowhere else."""
        t = i / self.probe_steps
        return np.full(3 * self.number_of_control_points,
                       self.probe_amp * np.sin(2 * np.pi * self.probe_hz * t
                                               * self.probe_seconds))

    def _run_probe(self):
        """Apply the routine, read the arm's response, update self._mass_estimate.

        RUNS OUTSIDE THE POLICY'S ACTION LOOP -- called from reset() and from the attach
        gate, never from step(). The policy therefore never sees these steps, collects no
        reward for them, and simply finds a number already in its observation. Episode
        length and every reward term are untouched.

        The arm MOVES during the probe (that is the point), so any caller that depends on
        the arm's pose -- phase 2's dome, built from the grasp point -- must rebuild
        afterwards. reset() and the attach gate both do.
        """
        if self._est_model is None:
            return
        R, n_cp = self.shearable_rod, self.number_of_control_points
        shape0 = np.array([
            float(np.linalg.norm(R.position_collection[..., -1])),
            float(np.linalg.norm(R.kappa, axis=0).mean()),
            float(np.linalg.norm(R.kappa, axis=0).max()),
            float(R.position_collection[1, -1]),
            float(np.linalg.norm(R.velocity_collection[..., -1])),
        ])
        tip_p, tip_v, tip_dir, force, kappa, ke = [], [], [], [], [], []
        for i in range(self.probe_steps):
            act = self._probe_routine(i)
            self.spline_points_func_array_normal_dir[:] = act[:n_cp]
            self.spline_points_func_array_binormal_dir[:] = act[n_cp: 2 * n_cp]
            self.spline_points_func_array_twist_dir[:] = act[2 * n_cp:]
            for _ in range(self.num_steps_per_update):
                self.time_tracker = self.StatefulStepper.step(
                    self.simulator, self.time_tracker, self.time_step)
            if not np.all(np.isfinite(R.position_collection)):
                return                       # keep the previous estimate rather than NaN
            tip_p.append(R.position_collection[..., -1].copy())
            tip_v.append(R.velocity_collection[..., -1].copy())
            tip_dir.append(R.director_collection[..., -1].copy())
            force.append(R.internal_forces[..., -1].copy())
            kappa.append(np.linalg.norm(R.kappa, axis=0).copy())
            ke.append(0.5 * float((R.mass * (R.velocity_collection ** 2).sum(axis=0)).sum()))
        f = self._ef.extract(np.array(tip_p), np.array(tip_v), np.array(tip_dir),
                             np.array(force), np.array(kappa), np.array(ke), shape0,
                             self.time_step * self.num_steps_per_update)
        m = self._est_model
        if self._est_kind == "nn":
            pct = self._enn.predict_pct(m, f)
        else:
            pct = float(np.dot(np.append((f - m["mu"]) / m["sd"], 1.0), m["coef"]))
        # the model outputs PERCENT of rod mass; the observation carries a FRACTION.
        # Clipped to the range the oracle could ever produce, so the policy is never shown
        # a value outside the domain observe_mass would have given it.
        self._mass_estimate = float(np.clip(pct / 100.0, 0.0, 1.0))

    def _surface_gap(self, dist):
        """TIP-to-payload surface separation, from their centre distance. 0 = touching.

        This is the REACH measure: it says how far the tip is from the payload's surface,
        which is what phase 1 is being asked to minimise. It models the rod's tip as a
        sphere at the tip node, so it says nothing about the rest of the arm -- use
        _body_gap for anything that must not let the payload intersect the shaft.
        """
        return float(dist) - (float(self.shearable_rod.radius[-1]) + self._payload_radius())

    def _tip_cone(self):
        """cos of the angle between the tip's tangent and the direction to the payload.
        +1 = the payload sits straight off the end; 0 = square to the side; <0 = behind."""
        P = self.shearable_rod.position_collection
        tip = P[..., -1]
        ax = tip - P[..., -2]
        n = np.linalg.norm(ax)
        if n < 1e-12:
            return 1.0
        v = self.sphere.position_collection[..., 0] - tip
        m = np.linalg.norm(v)
        if m < 1e-12:
            return 1.0
        return float(np.dot(ax / n, v / m))

    def _body_gap(self):
        """Separation between the payload and the ENTIRE rod, not just its tip node.

        WHY THIS EXISTS. The rod is a swept cylinder: its material fills a radius around
        the WHOLE centreline, not just a ball at the last node. A gate written against the
        tip node alone constrains the DISTANCE to the tip but not the DIRECTION, so a
        payload sitting alongside the shaft passes it while being buried in the arm.
        Measured on the _SM smoke run, which used exactly that gate: the payload sat a
        clean 0.104 m from the tip node (gate satisfied) while its nearest approach to the
        centreline was 0.022-0.034 m against a required 0.100 m -- roughly 7 cm inside the
        arm, pressed against element 17 of 19 rather than off the end. The rigid joint then
        froze it there for the whole carry.

        Returns min over rod segments of (distance from the payload centre to the segment)
        minus (r_rod + r_payload). Because the nearest point on a capsule's centreline is
        the end node exactly when the payload lies in the tip's forward hemisphere, this
        also implicitly enforces the grasp DIRECTION: a payload beside or behind the tip
        scores lower than one off the end, and cannot pass a gate that requires ~0.
        """
        P = self.shearable_rod.position_collection
        c = self.sphere.position_collection[..., 0]
        A = P[:, :-1]
        AB = P[:, 1:] - A
        denom = np.einsum("ij,ij->j", AB, AB)
        t = np.where(denom > 1e-18, np.einsum("ij,ij->j", c[:, None] - A, AB) / np.maximum(denom, 1e-18), 0.0)
        t = np.clip(t, 0.0, 1.0)
        d = np.linalg.norm(c[:, None] - (A + t * AB), axis=0).min()
        # radius varies per element; the tip radius is the relevant bound at the end and
        # the rod is near-uniform here, so use the max as the conservative choice.
        return float(d) - (float(self.shearable_rod.radius.max()) + self._payload_radius())

    def _tip_quaternion(self):
        """Unit quaternion of the rod tip director. Shared by get_state() and by the
        q_place capture, so both use exactly the same convention. The max() guards
        against a marginally negative trace producing a NaN from sqrt.
        """
        Q = self.shearable_rod.director_collection[..., -1]
        qw = np.sqrt(max(1.0 + Q[0, 0] + Q[1, 1] + Q[2, 2], 1e-12)) / 2.0
        qx = (Q[2, 1] - Q[1, 2]) / (4 * qw)
        qy = (Q[0, 2] - Q[2, 0]) / (4 * qw)
        qz = (Q[1, 0] - Q[0, 1]) / (4 * qw)
        return np.array([qw, qx, qy, qz])

    def _phase2_warmup(self):
        """PHASE-2 start state: the object grasped, and the dome built from that grasp.

        DEFAULT (phase2_policy1 unset): the SYNTHETIC warm-up below -- independent of
        policy 1, at the cost of being a proxy for its hand-over distribution.

        OPTIONAL (phase2_policy1 set): roll out the TRAINED policy 1 until its own
        attach gate fires, and hand over from the grasp it actually produced. This is
        the real hand-over distribution -- the "buffering true attach states" the
        synthetic warm-up's docstring calls the principled version -- and removes the
        proxy caveat entirely. Costs the independence (phase 2 then needs policy 1)
        and some wall-clock, since only a fraction of policy-1 rollouts grasp.
        """
        if self.phase2_attach_states is not None:
            return self._phase2_warmup_restore()
        if self.phase2_policy1 is None:
            return self._phase2_warmup_synthetic()
        return self._phase2_warmup_replay()

    def _load_policy1(self):
        """Lazily load the phase-1 policy used to seed phase-2 episodes."""
        if self._policy1_model is None:
            from stable_baselines3 import SAC   # local: keeps the env import SB3-free

            self._policy1_model = SAC.load(self.phase2_policy1, device="cpu")
        return self._policy1_model

    def _rollout_policy1(self):
        """Drive the sim with the trained policy 1 until its attach gate fires.

        Reproduces the phase-0 attach EXACTLY -- same gate (dist < attach_radius AND
        orientation_dist < orient_tol), same q_place capture -- so a banked grasp is
        indistinguishable from one policy 1 produces at test time. Returns True on a
        grasp, leaving the arm holding the load; False if the budget ran out.
        """
        model = self._load_policy1()
        self.attach_state["attached"] = False
        self.attach_state["ever_attached"] = False
        n_cp = self.number_of_control_points
        max_steps = max(
            1,
            int(self.phase2_warmup_max_time
                / (self.time_step * self.num_steps_per_update)),
        )
        for _ in range(max_steps):
            act, _ = model.predict(self.get_state(), deterministic=False)
            self.spline_points_func_array_normal_dir[:] = act[:n_cp]
            self.spline_points_func_array_binormal_dir[:] = act[n_cp : 2 * n_cp]
            self.spline_points_func_array_twist_dir[:] = act[2 * n_cp :]
            for _ in range(self.num_steps_per_update):
                self.time_tracker = self.StatefulStepper.step(
                    self.simulator, self.time_tracker, self.time_step
                )
            tip = self.shearable_rod.position_collection[..., -1]
            if not np.all(np.isfinite(tip)):
                return False
            dist = np.linalg.norm(tip - self.sphere.position_collection[..., 0])
            q_tip = self._tip_quaternion()
            if (
                abs(self._body_gap()) < self.attach_clearance
                and self._tip_cone() >= self.attach_cone
                and (1.0 - np.dot(q_tip, self.target_tip_orientation) ** 2)
                < self.orient_tol
            ):
                self.attach_state["attached"] = True
                self.attach_state["ever_attached"] = True
                self.place_orientation = q_tip
                self.target_tip_orientation = self.place_orientation
                zero = np.zeros(n_cp)
                self.spline_points_func_array_normal_dir[:] = zero
                self.spline_points_func_array_binormal_dir[:] = zero
                self.spline_points_func_array_twist_dir[:] = zero
                return True
        return False

    def _phase2_warmup_replay(self):
        """PHASE-2 start from a REAL policy-1 grasp, rolled out fresh this episode.

        Deliberately per-reset rather than from a pre-built bank of snapshots. A bank
        would reuse ONE finalized simulator across every rollout, and three pieces of
        state leak across that boundary: GatedFixedJoint._rest_captured latches on the
        first grasp (so later grasps inherit the first one's rest rotation),
        GatedSphereAnchor.anchor_position is fixed at registration (so re-randomising
        the pick target is undone by the anchor on the next step, leaving every
        rollout the SAME reach problem), and target_tip_orientation is overwritten at
        each grasp (so the attach gate then scores against the previous grasp's tip
        pose). reset() builds a fresh simulator, so none of that can happen here.

        The cost is honest: policy 1 grasps roughly 2 rollouts in 10, and a failure
        burns the whole warm-up budget before falling back -- about double the
        per-episode cost for real hand-over data. Which is why the DEFAULT is the
        synthetic warm-up, now calibrated against phase 1's own logged distribution.

        Touches NO reward, bonus or penalty -- only the state the carry starts from.
        """
        if self._rollout_policy1():
            # Clock rewound BEFORE the dome is built: _setup_dome stamps _carry_t0 =
            # time_tracker, so rewinding afterwards would leave the schedule starting
            # in the future and pin s at 0 for the whole episode.
            self.time_tracker = np.float64(0.0)
            self._episode_t0 = self.time_tracker
            return self._setup_dome(self.sphere.position_collection[..., 0].copy())

        # Rate-limited: policy 1 grasps ~2 rollouts in 10, so printing every failure
        # would bury the training log under ~8 lines per 10 resets.
        self._p1_miss = getattr(self, "_p1_miss", 0) + 1
        if self._p1_miss % 25 == 1:
            print(
                " PHASE 2 warm-up: policy 1 has failed to grasp %d time(s) in %.1f s"
                " -- falling back to the synthetic warm-up."
                % (self._p1_miss, self.phase2_warmup_max_time), flush=True
            )
        tip = self.shearable_rod.position_collection[..., -1].copy()
        if np.all(np.isfinite(tip)):
            self.sphere.position_collection[..., 0] = tip
            self.sphere.velocity_collection[:] = 0.0
        self.attach_state["attached"] = True
        self.attach_state["ever_attached"] = True
        return self._phase2_warmup_synthetic()

    def _phase2_warmup_restore(self):
        """PHASE-2 start restored from a REAL policy-1 attach state.

        No proxy and no rollout cost: the states were harvested once from the trained
        policy 1, so every episode begins in a configuration policy 1 genuinely
        produces -- same rod shape, same velocities, same grasp pose. The settle window
        then runs exactly as it does for a live hand-off, so the carry starts from a
        stabilised load rather than from the transient.

        Touches NO reward, bonus or penalty -- only the state the carry starts from.
        """
        if self._attach_states is None:
            d = np.load(self.phase2_attach_states)
            self._attach_states = {k: d[k] for k in d.files}
            print(" PHASE 2: loaded %d real policy-1 attach states from %s"
                  % (len(self._attach_states["q_place"]), self.phase2_attach_states),
                  flush=True)
        bank = self._attach_states
        n_bank = len(bank["q_place"])
        # Held-out split, if one was asked for. Default (None) samples the whole bank,
        # which is what every run so far did.
        lo, hi = 0, n_bank
        if self.phase2_attach_split is not None:
            cut = int(round(float(self.phase2_attach_split) * n_bank))
            cut = min(max(cut, 1), n_bank - 1)          # never empty on either side
            lo, hi = (0, cut) if self.phase2_attach_side == "train" else (cut, n_bank)
        i = int(self.np_random.integers(lo, hi))
        for k in ("position_collection", "velocity_collection",
                  "director_collection", "omega_collection"):
            getattr(self.shearable_rod, k)[:] = bank[f"rod_{k}"][i]
            getattr(self.sphere, k)[:] = bank[f"sph_{k}"][i]
        # ---- REPAIR STALE GEOMETRY (added 2026-09-04) ----
        # Banks harvested before the gate became surface-referenced encode the OLD
        # convention: the attach gate fired on ||tip - centre|| < 0.05, which is the
        # payload's own radius, so every banked grasp has the payload ~5 cm INSIDE the
        # rod (measured median surface gap -0.052 m on attach_states_P1mass_*).
        # The joint is rigid now, so restoring one unrepaired would freeze that overlap
        # for the whole carry -- training phase 2 against exactly the geometry the
        # surface-referenced gate was introduced to remove.
        #
        # Only the PAYLOAD's position is corrected, and only when it is interpenetrating:
        # it is pushed back out along the existing tip->payload direction to contact. The
        # rod's shape, velocity, directors and angular velocity -- the part of the state
        # phase 2 actually inherits, and the part that made real hand-over data worth
        # harvesting -- are untouched. A bank recorded under the new gate already sits at
        # a positive gap and passes through unchanged.
        _tip = self.shearable_rod.position_collection[..., -1]
        _c = self.sphere.position_collection[..., 0]
        _d = float(np.linalg.norm(_c - _tip))
        if self._body_gap() < 0.0 and _d > 1e-12:
            _contact = _d - self._surface_gap(_d)
            self.sphere.position_collection[..., 0] = _tip + (_c - _tip) / _d * _contact
            if not getattr(self, "_bank_repair_warned", False):
                self._bank_repair_warned = True
                print(" PHASE 2: attach bank %s predates the surface-referenced gate;"
                      " payload repositioned from %+.4f m to surface contact."
                      % (self.phase2_attach_states, self._surface_gap(_d)), flush=True)
        # q_place is the pose held at that grasp; the observation must report the same
        # quantity R2 is scored against, exactly as the other warm-ups do.
        self.place_orientation = bank["q_place"][i].copy()
        self.target_tip_orientation = self.place_orientation
        # Clock rewound BEFORE the dome is built: _setup_dome stamps _carry_t0, so
        # rewinding afterwards would leave the schedule starting in the future.
        self.time_tracker = np.float64(0.0)
        self._episode_t0 = self.time_tracker
        return self._setup_dome(self.sphere.position_collection[..., 0].copy())

    def _phase2_warmup_synthetic(self):
        """PHASE-2 hand-over state: drive the rod to a REAL loaded configuration.

        Spawning the object at the straight-rod tip (0,1,0) is degenerate -- that is
        the arm's maximum reach, so the dome from it descends monotonically (measured
        dy/ds < 0 everywhere) and policy 2 never sees the lift the split exists to
        train. No dome_apex value fixes this: any visible arch from y=1.0 would sit
        beyond the workspace.

        So the episode still starts attached, but the rod is first driven with a random
        held actuation, carrying the object with it. REJECTION SAMPLING on the
        resulting grasp height is what makes this work -- measured over fixed
        (amplitude, duration) settings only 25-58% of warm-ups landed in the pick
        range: short ones leave the tip near y=1.0 (degeneracy intact), long ones let
        the rod flop to the floor (y~0.05, below the drop-off).

        ACCEPTANCE IS CALIBRATED AGAINST REAL PHASE-1 HAND-OVERS, not a guessed height
        window: a draw is kept only if the dome it would produce has apex height and
        path length inside the p5-p95 band measured over 3292 real ATTACHED events in
        phase 1's own training logs (phase2_apex_range / phase2_length_range). Verified
        over 60 warm-ups -- real vs synthetic medians: apex 0.800 vs 0.816, length
        1.180 vs 1.290 m, 0/60 outside the band. phase2_grasp_y survives only as a
        coarse pre-filter.

        Touches NO reward, bonus or penalty -- only the state the carry starts from.

        REMAINING CAVEAT: the two dome observables now match phase 1's, but this is
        still a proxy -- rod SHAPE and VELOCITY at hand-over are unconstrained, and the
        hard p5/p95 cut clips the tails (synthetic p5/p95 sit inside the real ones).
        Restoring true attach states from a trained policy 1 remains the
        principled version.
        """
        n_cp = self.number_of_control_points
        lo, hi = self.phase2_grasp_y
        zero = np.zeros(n_cp)
        for _ in range(self.phase2_warmup_max_tries):
            amp = self.phase2_warmup_amp
            act = self.np_random.uniform(-amp, amp, 3 * n_cp)
            self.spline_points_func_array_normal_dir[:] = act[:n_cp]
            self.spline_points_func_array_binormal_dir[:] = act[n_cp : 2 * n_cp]
            self.spline_points_func_array_twist_dir[:] = act[2 * n_cp :]
            t_warm = self.np_random.uniform(*self.phase2_warmup_time)
            for _ in range(max(int(t_warm / self.time_step), 1)):
                self.time_tracker = self.StatefulStepper.step(
                    self.simulator, self.time_tracker, self.time_step
                )
            tip = self.shearable_rod.position_collection[..., -1]
            if not (np.all(np.isfinite(tip)) and lo <= tip[1] <= hi):
                continue
            # Accept only if the dome this grasp produces looks like the ones phase 1
            # actually handed over (see phase2_apex_range / phase2_length_range).
            apex, length = self._dome_stats(
                self.sphere.position_collection[..., 0].copy()
            )
            if (
                self.phase2_apex_range[0] <= apex <= self.phase2_apex_range[1]
                and self.phase2_length_range[0] <= length <= self.phase2_length_range[1]
            ):
                break
        else:
            # Every try missed the hand-over band (or blew up). Say so rather than
            # silently training on an out-of-range grasp.
            tip = self.shearable_rod.position_collection[..., -1]
            print(
                " WARNING: phase-2 warm-up did not match the real phase-1 hand-over"
                " band (apex %.2f-%.2f, length %.2f-%.2f) after %d tries"
                " -- using the last draw anyway (tip y = %.3f)."
                % (self.phase2_apex_range[0], self.phase2_apex_range[1],
                   self.phase2_length_range[0], self.phase2_length_range[1],
                   self.phase2_warmup_max_tries, tip[1])
            )
        # The held warm-up command is deliberately NOT zeroed here. Zeroing it removed
        # the torque holding the rod bent, and the elastic springback that followed
        # produced 0.175 m of tracking error in 0.17 s with no commanded action at all
        # -- the "roaming before it starts tracking" seen in the videos. Leaving the
        # command in place hands policy 2 a rod in equilibrium; its own first action
        # then replaces the command, which is a controlled change rather than a
        # release to zero.
        # q_place = the tip pose ACTUALLY held at hand-over, so R2's orientation term
        # means "hold the grasp" -- the same thing it means in phase 0/1.
        self.place_orientation = self._tip_quaternion()
        # The OBSERVATION carries target_tip_orientation as its orientation reference.
        # R2 scores against place_orientation, so in phase 2 the two must be the SAME
        # quantity -- otherwise the policy is graded on holding a pose it cannot see
        # (measured mismatch 0.45-0.96 before this line, i.e. essentially unrelated).
        # Phase 0/1 need no such line: their attach gate forces the two within
        # orient_tol. R1 and the attach gate never run in phase 2, so this override
        # affects the observation only -- no reward term changes.
        self.target_tip_orientation = self.place_orientation
        # Rewind the clock BEFORE building the dome, so the warm-up's own sim time is
        # not charged to the carry. _setup_dome stamps _carry_t0 = time_tracker and
        # warns when carry_duration exceeds the episode remainder; with up to
        # phase2_warmup_max_tries draws of 0.15-0.60 s each the warm-up could consume
        # 2.8 s of a 5 s episode, and 4 of 35 measured domes were then unable to reach
        # the drop-off before truncation -- episodes on which no arrival could ever
        # fire. The step budget is governed by current_step, not by time_tracker, so
        # rewinding costs the episode nothing. Same rewind as both replay warm-ups.
        self.time_tracker = np.float64(0.0)
        self._episode_t0 = self.time_tracker
        return self._setup_dome(self.sphere.position_collection[..., 0].copy())

    def _carry_progress(self):
        """Time-parametrization of the dome goal, CONSTANT SPEED along the path:
        the elapsed fraction clamp((t - t_attach)/carry_duration, 0, 1) is mapped
        to arc length and inverted to s via the arc-length table sampled at the
        attach instant (plain uniform-s scheduling is NOT constant speed -- it ran
        ~0.36 m/s on the climb but ~0.8 m/s on the final descent, violating the
        <= 0.5 m/s bound the loaded arm can pursue). After release (or once s
        reaches 1) the goal sits at the dome end = place_position forever, which
        reproduces the previous fixed-drop-off behavior for the rest of the
        episode (R2 keeps scoring the completed placement).
        """
        if self._carry_duration is None:
            return 1.0
        if self._settling:
            return 0.0          # goal parked at the grasp point until the load settles
        frac = np.clip(
            (self.time_tracker - self._carry_t0) / self._carry_duration, 0.0, 1.0
        )
        return float(
            np.interp(frac * self._dome_arclen[-1], self._dome_arclen, self._dome_s_grid)
        )

    def get_state(self):
        """
        Returns current state of the system to the controller.

        Returns
        -------
        numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
        """

        rod_state = self.shearable_rod.position_collection
        r_s_a = rod_state[0]  # x_info
        r_s_b = rod_state[1]  # y_info
        r_s_c = rod_state[2]  # z_info

        num_points = int(self.n_elem / self.obs_state_points)
        ## get full 3D state information
        rod_compact_state = np.concatenate(
            (
                r_s_a[0 : len(r_s_a) + 1 : num_points],
                r_s_b[0 : len(r_s_b) + 1 : num_points],
                r_s_c[0 : len(r_s_b) + 1 : num_points],
            )
        )

        rod_compact_velocity = self.shearable_rod.velocity_collection[..., -1]
        rod_compact_velocity_norm = np.array([np.linalg.norm(rod_compact_velocity)])
        rod_compact_velocity_dir = (
            rod_compact_velocity / rod_compact_velocity_norm
            if rod_compact_velocity_norm[0] > 0.0 else np.zeros(3)
        )

        sphere_compact_state = self.sphere.position_collection.flatten()  # 2
        sphere_compact_velocity = self.sphere.velocity_collection.flatten()
        sphere_compact_velocity_norm = np.array(
            [np.linalg.norm(sphere_compact_velocity)]
        )
        sphere_compact_velocity_dir = (
            sphere_compact_velocity / sphere_compact_velocity_norm
            if sphere_compact_velocity_norm[0] > 0.0 else np.zeros(3)
        )

        self.rod_tip_orientation = self._tip_quaternion()

        # Active-phase tracking target: the point the policy must currently chase.
        # Phase 1 (not attached): the pick target = sphere position (duplicates
        # sphere_compact_state, keeping the layout phase-independent). Phase 2
        # (attached / released): the dome waypoint T2(s(t)) -- the ONLY place the
        # moving goal enters the observation, without which the carry along the
        # dome would be unobservable to the (Markovian) policy.
        # Phase 3 (arrived): the drop-off itself. The dome is finished by then, and
        # reporting place_position keeps the target well-defined after _phase3_warmup
        # rewinds the clock (which would otherwise send _carry_progress back to s=0
        # and make the observed goal jump to the start of the dome).
        if self.PICK_AND_RELEASE and self._arrived:
            tracking_target = np.asarray(self.place_position, dtype=np.float64).copy()
        elif self.PICK_AND_RELEASE and (
            self.attach_state["attached"] or self.released
        ):
            tracking_target = self._dome_point(self._carry_progress())
        else:
            tracking_target = self.sphere.position_collection[..., 0].copy()

        state = np.concatenate(
            (
                # rod information
                rod_compact_state,
                rod_compact_velocity_norm,
                rod_compact_velocity_dir,
                self.rod_tip_orientation,
                # target information
                sphere_compact_state,
                sphere_compact_velocity_norm,
                sphere_compact_velocity_dir,
                self.target_tip_orientation,
                # active-phase tracking target (see above)
                tracking_target,
                # ONE payload slot, filled by exactly one of two sources:
                #   observe_mass    -> the TRUE mass / rod mass    (oracle baseline)
                #   mass_estimator  -> the ESTIMATED mass / rod mass, from the calibration
                #                      routine. The true value is never read here.
                (np.array([self.sphere.mass.sum() / self.shearable_rod.mass.sum()])
                 if self.observe_mass
                 else (np.array([self._mass_estimate])
                       if self._est_model is not None else np.empty(0))),
            )
        )

        return state

    def step(self, action):
        """
        This method integrates the simulation number of steps given in num_steps_per_update, using the actions
        selected by the controller and returns state information, reward, and done booleans.

        Parameters
        ----------
        action :  numpy.ndarray
            1D (n_torque_directions * number_of_control_points,) array containing data with 'float' type.
            Action returns control points selected by control algorithm to the Elastica simulation. n_torque_directions
            is number of torque directions, this is controlled by the dim.

        Returns
        -------
        state : numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
        reward : float
            Reward after the integration.
        terminated : boolean
            True if the episode ended due to NaN blow-up.
        truncated : boolean
            True if the episode ended due to reaching final simulation time.
        info : dict

        """

        # action contains the control points for actuation torques in different directions in range [-1, 1]
        self.action = action

        # set binormal activations to 0 if solving 2D case
        if self.dim == 2.0:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
            self.spline_points_func_array_twist_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
        elif self.dim == 2.5:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
            self.spline_points_func_array_twist_dir[:] = action[
                self.number_of_control_points :
            ]
        # apply binormal activations if solving 3D case
        elif self.dim == 3.0:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = action[
                self.number_of_control_points :
            ]
            self.spline_points_func_array_twist_dir[:] = (
                action[: self.number_of_control_points] * 0.0
            )
        elif self.dim == 3.5:
            self.spline_points_func_array_normal_dir[:] = action[
                : self.number_of_control_points
            ]
            self.spline_points_func_array_binormal_dir[:] = action[
                self.number_of_control_points : 2 * self.number_of_control_points
            ]
            self.spline_points_func_array_twist_dir[:] = action[
                2 * self.number_of_control_points :
            ]

        # Do multiple time step of simulation for <one learning step>
        # 1.0.0 stepping API: stateful stepper instance
        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.StatefulStepper.step(
                self.simulator, self.time_tracker, self.time_step
            )

        if self.mode == 3:
            ##### (+1, 0, 0) -> (0, -1, 0) -> (-1, 0, 0) -> (0, +1, 0) -> (+1, 0, 0) #####
            if (
                self.current_step
                % (1.0 / (self.h_time_step * self.num_steps_per_update))
                == 0
            ):
                if self.dir_indicator == 1:
                    self.sphere.velocity_collection[..., 0] = [
                        0.0,
                        -self.sphere_initial_velocity,
                        0.0,
                    ]
                    self.dir_indicator = 2
                elif self.dir_indicator == 2:
                    self.sphere.velocity_collection[..., 0] = [
                        -self.sphere_initial_velocity,
                        0.0,
                        0.0,
                    ]
                    self.dir_indicator = 3
                elif self.dir_indicator == 3:
                    self.sphere.velocity_collection[..., 0] = [
                        0.0,
                        +self.sphere_initial_velocity,
                        0.0,
                    ]
                    self.dir_indicator = 4
                elif self.dir_indicator == 4:
                    self.sphere.velocity_collection[..., 0] = [
                        +self.sphere_initial_velocity,
                        0.0,
                        0.0,
                    ]
                    self.dir_indicator = 1
                else:
                    print("ERROR")

        if self.mode == 4:
            self.trajectory_iteration += 1
            if self.trajectory_iteration == 500:
                # print('changing direction')
                self.rand_direction_1 = np.pi * self.np_random.uniform(0, 2)
                if self.dim == 2.0 or self.dim == 2.5:
                    self.rand_direction_2 = np.pi / 2.0
                elif self.dim == 3.0 or self.dim == 3.5:
                    self.rand_direction_2 = np.pi * self.np_random.uniform(0, 2)

                self.v_x = (
                    self.target_v
                    * np.cos(self.rand_direction_1)
                    * np.sin(self.rand_direction_2)
                )
                self.v_y = (
                    self.target_v
                    * np.sin(self.rand_direction_1)
                    * np.sin(self.rand_direction_2)
                )
                self.v_z = self.target_v * np.cos(self.rand_direction_2)

                self.sphere.velocity_collection[..., 0] = [
                    self.v_x,
                    self.v_y,
                    self.v_z,
                ]
                self.trajectory_iteration = 0

        self.current_step += 1

        # End the settle window if the load has slowed (see _update_settle). Must run
        # BEFORE get_state/the reward, so both see the same schedule this step.
        self._update_settle()

        # observe current state: current as sensed signal
        state = self.get_state()

        # Phase-1 quantities: tip-to-sphere distance (n) and tip-vs-target orientation (p).
        # Computed every step -- they drive the phase-1 reward AND the attach gate, and are
        # reused in the end-of-episode diagnostic print.
        dist = np.linalg.norm(
            self.shearable_rod.position_collection[..., -1]
            - self.sphere.position_collection[..., 0]
        )
        # R1's REACH TERM, SURFACE-REFERENCED (2026-09-04).
        # It used to be -||tip - payload_CENTRE||^2, whose optimum is dist = 0: the tip
        # node AT the payload's centre, i.e. the arm fully inside it. That is not a
        # reachable configuration for two solid bodies, and it is what the trained policy
        # actually chased -- measured, it drove the tip through the payload and ended up
        # holding it against the shaft (element 17 of 19, ~7 cm inside the arm). With the
        # direction cone enabled that policy grasped 0 times in 25 episodes, because it
        # never approaches tip-first.
        # Squaring the SURFACE gap instead puts the optimum exactly at contact: negative
        # (interpenetrating) is penalised just as much as positive (too far), so there is
        # no gradient pushing the tip into the payload.
        _reach_gap = self._surface_gap(dist)
        reward_dist = -np.square(_reach_gap)
        orientation_dist = (
            1.0 - np.dot(self.rod_tip_orientation, self.target_tip_orientation) ** 2
        )
        orientation_penalty = -((orientation_dist) ** 2)

        """ Reward Engineering -- THREE-PHASE SWITCH, never a sum.
            R1  while NOT attached                  reach + orient (paper's reward, UNTOUCHED)
            R2  attached, NOT yet arrived           carry along the dome, NO orientation
            R3  arrived (attached or released)      orient the tip down, then release
        Which one is live is decided by two runtime latches -- attach_state["attached"]
        (set by the attach gate) and _arrived (set the step the load first reaches
        place_position) -- never by a step count.
            n       = ||x_tip - x_sphere||,  p = 1 - (q_tip . q_target)^2
            m       = ||x_sphere - T2(s(t))||       distance to the MOVING dome goal
            m_place = ||x_sphere - place_position|| distance to the FIXED drop-off
            p_down  = (1 + t_tip . y_hat)/2         0 = tip straight down
        T2(s) is the DOME carry path from the grasp point g to place_position (see
        _dome_point); s(t) clamps at 1 with T2(1) = place_position.
        GATES, all automatic consequences of a threshold -- never learned, never bonused:
            attach   n < attach_radius AND p < orient_tol
            arrival  m_place <= place_radius                 (R2 -> R3; latches _arrived)
            release  m_place < place_radius AND p_down <= orient_tier_inner   (R3 only)
        Phase 2 can NEVER release. Re-attach after release is blocked. In phase-N training
        the episode also ends at that phase's own gate (see _phase1/2/3_done).
        """
        done = False
        reward = 0.0
        # A NaN blow-up is a TERMINATION (the future really is worthless); a time
        # limit or a gate is a TRUNCATION (bootstrap V(s')). Previously a state-NaN
        # landing on the final step was reported as truncated, so SB3 bootstrapped
        # off a -10000 step.
        nan_blowup = False

        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

        if invalid_values_condition == True:
            print(" Nan detected in the position, exiting simulation now")
            nan_blowup = True
            self.shearable_rod.position_collection = np.zeros(
                self.shearable_rod.position_collection.shape
            )
            reward = -10000
            state = self.get_state()
            done = True

        if not done:
            if self.PICK_AND_RELEASE and (
                self.attach_state["attached"] or self.released
            ):
                # -------------- PHASES 2 and 3 -- LOADED CARRY / PLACE --------------
                # Active while ATTACHED and after RELEASE. Which of the two pays is
                # decided by the _arrived latch, NOT by a step count:
                #   not arrived -> R2, carry along the dome (below)
                #   arrived     -> R3, orient down and release (above)
                # Phase 1 is OFF here.
                #
                # m tracks the MOVING dome goal T2(s(t)), not the fixed drop-off; s(t)
                # advances at carry_speed and clamps at 1, where T2(1) = place_position.
                # Used by R2 only -- R3 never reads it (verified: R3 is exactly a
                # function of m_place and p_down).
                # R2, as actually implemented below (this line previously described an
                # OLDER design -- tiers on m at 2*place_radius/place_radius with 0.5/1.5
                # bonuses, and no arrival term at all -- which no longer matches the code):
                #   R2 = -min(m, track_error_clip)^2                      clip 2.5
                #      + track_bonus_outer * [m <= track_tier_outer]      +1.0 at 0.30
                #      + track_bonus_inner * [m <= track_tier_inner]      +3.0 at 0.15
                #      + 0.5 * [m_place <= 3*place_radius]                arrival, 0.15
                #      + 1.5 * [m_place <= place_radius]                  arrival, 0.05
                #      - action_rate_penalty * mean((a_t - a_{t-1})^2)    default off
                #      - action_penalty      * mean(a_t^2)                default off
                # Tracking keys off m (the MOVING goal); arrival keys off m_place (the
                # FIXED drop-off). Max attainable = 1.0+3.0+0.5+1.5 = 6.0 per step.
                #
                # Distance from the sphere to the FIXED drop-off. Drives the arrival
                # tiers and the arrival latch in R2, and the whole of R3 -- everything
                # about the drop-off must key off THIS, never off m (distance to the
                # moving waypoint), which stays small all along the dome.
                m_place = np.linalg.norm(
                    self.sphere.position_collection[..., 0] - self.place_position
                )

                # Tip direction, needed by phase 3 and by the release print.
                # p_down = (1 + t_tip . y_hat)/2, i.e. (1 + t_tip[1])/2 -- the dot is with
                # WORLD UP, hardcoded as the y component below. (This comment used to name
                # an "n_place" drop-off pointer director; no such variable exists or is
                # read here.) 0 when the tip is exactly anti-parallel to y_hat, i.e.
                # pointing straight down, 1 when pointing straight up. A DIRECTION
                # dot product, not a quaternion distance -- 1-(q.q')^2 is sign-invariant
                # and so cannot express "anti-parallel", and it would also constrain the
                # roll, which is meaningless for a sphere.
                t_tip = self.shearable_rod.director_collection[2, :, -1]
                t_norm = np.linalg.norm(t_tip)
                p_down = (
                    (1.0 + float(t_tip[1] / t_norm)) / 2.0 if t_norm > 0.0 else 1.0
                )

                if self._arrived:
                    # ========== PHASE 3 (R3) -- ORIENT DOWN, THEN RELEASE ==========
                    # Split at the release band:
                    #   OUTSIDE : -m_place^2 and nothing else. Under a pure gate every
                    #             state outside scores the SAME, so a policy pushed out
                    #             by its own rotation gets no gradient home and stalls
                    #             in a flat reward desert. Drift is not hypothetical
                    #             here: the load rides 2.9 cm from the tip, so a 90 deg
                    #             tip rotation swings it 4.3 cm -- comparable to the
                    #             0.05 band itself. This term is the way back, NOT a
                    #             reaching objective: phase 3 is never paid for
                    #             travelling to the drop-off, only for not losing it.
                    #   INSIDE  : orientation only -- position is a pure precondition,
                    #             exactly the check-then-orient structure.
                    # Inside is >= 0 and outside is strictly < 0, so stepping out of
                    # the band can never pay better than staying in it.
                    if m_place > self.place_radius:
                        # Offset by 1.0 so being outside is worse than the WORST in-band
                        # state (-p_down^2 = -1.0 at p_down = 1). Without it a badly
                        # oriented state inside the band would score below simply
                        # stepping outside, creating an incentive to leave the band.
                        reward = -(1.0 + np.square(m_place).sum())
                    else:
                        # SHAPING -- phase 1's form: -(error)^2. R1 weights orientation
                        # at 0.5 because there reaching outranks aligning; here
                        # orienting IS the task, so it carries weight 1.0. Negative
                        # (-1.0/step) when the tip points up and rising toward 0 as it
                        # comes down -- the same negative-to-positive swing that gives
                        # phase 1 its curve. No living cost needed.
                        # R3 BONUSES RESCALED x3 (2026-09-07): 0.5/1.5 -> 1.5/4.5.
                        # R1 and R2 both top out at 6.0 per step; R3 topped out at 2.0, so
                        # phase 3 was learning against a reward -- and therefore a gradient
                        # -- three times smaller than the phases either side of it, for no
                        # reason other than that nobody had compared the ceilings. In the
                        # smoke run R2 reached +3.38/step (56% of its ceiling) while R3
                        # reached +0.17 (8.5% of its), and part of that gap was simply the
                        # ceiling. The release-gate feasibility test showed the task IS
                        # reachable, so an under-scaled reward is a live suspect for phase
                        # 3 under-performing rather than the task being impossible.
                        #
                        # ONLY THE BONUSES ARE SCALED, not the -p_down^2 shaping term.
                        # Scaling the whole branch would also shrink the jump the policy
                        # gets for ENTERING the band (outside pays -(1 + m^2) ~ -1.0), and
                        # that jump is what draws it in. Leaving the quadratic alone keeps
                        # entry equally attractive and makes staying, and orienting, worth
                        # three times more. The 1:3 outer:inner ratio is unchanged, and it
                        # now matches R1's own tier ratio (1.5 outer / 4.5 inner) exactly.
                        #
                        # THE TIERS ARE NOT TOUCHED. orient_tier_inner is ALSO the release
                        # gate, so moving it would change when the episode can end, not
                        # just how much reward it pays.
                        reward = -np.square(p_down)
                        if p_down <= self.orient_tier_outer:
                            reward += self.orient_bonus_outer
                        if p_down <= self.orient_tier_inner:
                            reward += self.orient_bonus_inner

                    # RELEASE GATE -- lives ONLY here. Phase 2 can no longer release.
                    # BOTH conditions must hold: the load inside place_radius AND the
                    # tip pointing down. Detach stays an automatic consequence of the
                    # thresholds -- not a learned action, NO bonus.
                    if self.attach_state["attached"]:
                        if (
                            m_place < self.place_radius
                            and p_down <= self.orient_tier_inner
                        ):
                            self.attach_state["attached"] = False
                            self.released = True
                            if self.phase == 3:
                                self._phase3_done = True
                            print(
                                " RELEASED / PLACED at step %d (place dist %.4f, p_down %.4f)"
                                % (self.current_step, m_place, p_down)
                            )
                else:
                    # ====== PHASE 2 (R2) -- CARRY ALONG THE DOME, ARRIVE AT p ======
                    # m tracks the MOVING dome goal T2(s(t)). Computed HERE, not in the
                    # shared preamble, because R3 never reads it -- and phase-3 training
                    # has no dome at all (its warm-up skips the phase-2 one), so
                    # _dome_point would fail on _dome_start = None.
                    m = np.linalg.norm(
                        self.sphere.position_collection[..., 0]
                        - self._dome_point(self._carry_progress())
                    )

                    # TRACKING vs the moving dome goal. Tiers stay at track_tier
                    # (0.15/0.30), NOT place_radius: those were the RELEASE precision
                    # and were never the right scale for tracking a goal moving at
                    # carry_speed. Measured consequence of the tight tiers: the best
                    # tracking ever demonstrated (0.147 m) earned -0.02/step,
                    # indistinguishable from mediocre, because -m^2 improves by only
                    # 0.47/step across the whole 0.7 -> 0.15 m operating range while
                    # 97% of the reward sat inside the last 10 cm.
                    #
                    # ALL ORIENTATION HAS LEFT PHASE 2 -- no p_down, no q_place, no
                    # bonus and no penalty. Posture is phase 3's entire job now.
                    # Tracking is weighted 2x arrival (1.0/3.0 against 0.5/1.5): with
                    # orientation gone this policy has exactly two things to do, and
                    # following the path is the one being asked for.
                    # SHAPING -- phase 1's form: -(error)^2, at weight 1.0, exactly as
                    # R1 uses reward_dist = -dist^2. It is NEGATIVE whenever tracking is
                    # poor (-0.49/step at m=0.7, -1.0 at m=1.0) and rises toward 0 as
                    # tracking improves, which is what gives phase 1 its textbook curve
                    # and does the same here. No living cost is needed or wanted: the
                    # negative region is already present.
                    #
                    # BOUNDED. The clip is placed OUTSIDE the reachable set, so
                    # it is provably a no-op for every state the policy can occupy:
                    # the rod reaches at most ~1.0 m from its base and the dome is
                    # clamped to stay inside that reach, so the largest separation
                    # between load and goal in any physical state is 2.0 m -- below
                    # the 2.5 m clip. It can therefore only engage on a numerically
                    # diverged rod. Measured over the previous run, unbounded -m^2
                    # let 2-8 episodes per seed reach -139k to -789k against a +3.9k
                    # median (the worst averages -221/step, i.e. m ~ 14.9 m -- far
                    # outside the workspace), forcing the critic to fit TD targets
                    # 200x out of scale. Tiers, bonuses, arrival and release are
                    # untouched, as is the ordering of every reachable state.
                    reward = -np.square(min(float(m), self.track_error_clip))

                    if m <= self.track_tier_outer:
                        reward += self.track_bonus_outer

                    if m <= self.track_tier_inner:
                        reward += self.track_bonus_inner

                    # ---- ARRIVAL BONUSES (RESTORED 2026-09-04) ----
                    # +0.5 inside 3*place_radius, +1.5 inside place_radius, exactly the
                    # values _P2act and _P2mass trained with. Max R2 returns to 6.0/step.
                    #
                    # WHY THEY WERE REMOVED, AND WHY THAT REASONING FAILED. The argument
                    # was redundancy: the dome ENDS at the drop-off (T2(1) = place_position
                    # exactly), so tracking T2(s) to s = 1 is arriving by construction, and
                    # paying separately for proximity lets an episode be rewarded for
                    # sitting near the endpoint without having followed the path there.
                    # That holds ONLY IF the payload is where the tip is -- formally, if
                    # the joint's constraint violation is small against place_radius.
                    # It was not: K_ATTACH=1e3 under a 1-4 kg payload left the load
                    # 0.09-0.15 m behind the tip (up to 0.45 m in transients) against a
                    # 0.05 m place_radius, so "tracking the dome" and "arriving at the
                    # drop-off" had decoupled and the redundancy being relied on did not
                    # exist. R2 lost its arrival signal while the tracking tiers were
                    # simultaneously made unreachable by the same compliance -- see the
                    # K_ATTACH note. Restoring them returns the reward to the last
                    # configuration that trained successfully, so the rigid-grasp fix is
                    # the ONLY change from a known-good baseline.
                    #
                    # NOTE the arrival LATCH below is separate and unaffected: it is what
                    # hands the episode to phase 3.
                    if m_place <= 3.0 * self.place_radius:
                        reward += self.arrival_bonus_outer

                    if m_place <= self.place_radius:
                        reward += self.arrival_bonus_inner

                    # NO RELEASE IN PHASE 2. The gate moved to phase 3 in full: this
                    # policy can never set the object down, however close it gets.
                    #
                    # ARRIVAL LATCH -- set AFTER this step's R2 is paid, so the
                    # arriving step still collects the arrival bonus and phase 3 takes
                    # over from the NEXT step. Same convention as the phase-1 attach.
                    # It is a LATCH, not a live test: once arrived, phase 3 owns the
                    # episode even if its own rotation pushes the load back out of the
                    # band -- that is what R3's recovery slope is for.
                    if m_place <= self.place_radius:
                        self._arrived = True
                        if self.record_arrival_states is not None:
                            self._record_arrival()
                        if self.phase == 2:
                            self._phase2_done = True

                    # ---- ACTION-RATE PENALTY (R2 only, additive) ----
                    # Nothing above is altered: the tiers, bonuses, clip, arrival and
                    # latch are byte-identical. This only subtracts a cost for CHANGING
                    # the command between consecutive control steps.
                    #
                    # WHY. There is no regularisation anywhere in this environment --
                    # max_rate_of_change_of_activation is inf -- so the policy may slam
                    # every control point across the full [-1, 1] range every 1.4 ms.
                    # Measured: tip speeds peak at 5-7 m/s while the goal moves at 0.5,
                    # and peak/median tip speed is 3.3x. sim_dt was ruled out as the
                    # cause three times (roughness flat then RISING across 1x, 4x, 20x
                    # refinement), so the jitter is the command, not the integrator.
                    #
                    # WEIGHT. Measured mean square action change is 0.256/step against a
                    # typical step reward of 3.99, so w=1.0 costs ~6.4%. The inner->outer
                    # tier gap is 2.0, so halving the jitter gains 0.128 while dropping a
                    # tier costs 2.0: tracking stays strictly dominant and smoothing is
                    # only bought where it is free. Default 0.0 = OFF, so every existing
                    # config and the trained policies are unaffected.
                    # DISABLED. Command smoothing is now done in the PLANT instead, via a
                    # finite max_rate_of_change_of_activation, so penalising the command
                    # in the reward as well would charge twice for the same thing.
                    # Re-enable by uncommenting; the kwarg still exists and defaults to 0.
                    # if self.action_rate_penalty and self.previous_action is not None:
                    #     reward -= self.action_rate_penalty * float(
                    #         np.square(action - self.previous_action).mean()
                    #     )

                    # ---- ACTION-MAGNITUDE PENALTY (R2 only, additive) ----
                    # Independent of the rate penalty above: either, both or neither may
                    # be on. Needs no previous_action, so unlike the rate term it also
                    # applies on the first control step of an episode.
                    # DISABLED, same reason as the rate penalty above.
                    # if self.action_penalty:
                    #     reward -= self.action_penalty * float(np.square(action).mean())

            else:
                # -------------------- PHASE 1 -- REACH + ORIENT (R1) --------------------
                # Paper's reward, BYTE-IDENTICAL. Active only while NOT attached and NOT
                # yet released.
                reward = 1.0 * reward_dist + 0.5 * orientation_penalty
                # (A commented-out action-rate penalty with weight 0.1 used to sit here,
                # in the PHASE-1 branch. It was dead code and never ran, but it was read
                # as the live weight at least once. The real term is action_rate_penalty,
                # it lives in R2 only, and it was trained at w = 1.0. Removed to stop it
                # being mistaken for the implemented penalty again.)

                # TIERS RE-REFERENCED TO SURFACE SEPARATION (2026-09-04), to follow the
                # attach gate. The paper's tolerances (0.10 outer, 0.05 inner) were
                # distances to the target's CENTRE, correct when the sphere was a target
                # marker whose radius WAS the tolerance. Now that the sphere is a rigid
                # payload the gate fires at contact, so leaving these on centre distance
                # would put BOTH tiers inside the payload -- unreachable, and every bonus
                # here would be dead code while only -dist^2 survived. The same 0.10 and
                # 0.05 are kept, now as clearances ABOVE the payload's surface.
                #
                # NOTE what this gives up: the paper deliberately set the inner tier equal
                # to the gate so "max R1" and "the grasp fires" were one condition. That
                # only means something for a POINT target. With a surface target the tiers
                # shape the approach and contact is a separate event, so the two no longer
                # coincide. Deliberate, and stated rather than silently dropped.
                # BONUSES ALSO REQUIRE THE TIP-SIDE APPROACH. Distance alone does not
                # make a grasp a tip grasp: the payload can sit against the side of the
                # shaft at the same tip distance. The attach gate rejects that, so if the
                # bonuses did not also require it the policy would be paid full reward for
                # poses the gate can never fire on -- max R1 and "the grasp fires" would
                # point in different directions, which is exactly how the arm learned to
                # bury its tip. Gating both on the same cone restores the paper's property
                # that maximising R1 makes the grasp happen by construction.
                gap = self._surface_gap(dist)
                cone_ok = self._tip_cone() >= self.attach_cone

                if gap < 0.05 * 2.0 and cone_ok:
                    reward += 0.5
                    reward += 0.5 * (1 - orientation_dist)
                    if np.isclose(orientation_dist, 0.0, atol=0.05 * 2.0).all():
                        reward += 0.5

                # for this specific case, check on_goal parameter
                if gap < 0.05 and cone_ok:
                    reward += 1.5
                    reward += 1.5 * (1 - orientation_dist)
                    if np.isclose(orientation_dist, 0.0, atol=0.05).all():
                        reward += 1.5

                # ATTACH GATE: both reach and orient must pass. Attach is a consequence,
                # not a learned action -- NO bonus. The reward for THIS step is the phase-1
                # reward above (completing the reach); the switch to R2 takes effect from
                # the next step.
                if (
                    self.PICK_AND_RELEASE
                    and (not self.attach_state["attached"])
                    and (not self.released)
                ):
                    # GATE ON _body_gap, NOT _surface_gap. The tip-node measure
                    # constrains distance but not direction, so it admits grasps with the
                    # payload buried against the shaft (measured: 7 cm inside the arm on
                    # the _SM run). abs() makes it a SHELL around contact: approaching
                    # from outside the first crossing sits at ~+attach_clearance, and a
                    # pass that overshoots straight through simply does not grasp, rather
                    # than latching onto an interpenetrating pose.
                    if (
                        abs(self._body_gap()) < self.attach_clearance
                        and self._tip_cone() >= self.attach_cone
                        and orientation_dist < self.orient_tol
                    ):
                        self.attach_state["attached"] = True
                        self.attach_state["ever_attached"] = True
                        # q_place = the pose ACTUALLY held at the grasp, so R2's
                        # orientation term means "hold the grasp" in every phase.
                        # The gate has already guaranteed orientation_dist <
                        # orient_tol (0.05), so this differs from the old value
                        # (the sphere's director) by at most that tolerance.
                        self.place_orientation = self._tip_quaternion()
                        # Keep the OBSERVATION's orientation reference equal to the one
                        # R2 scores against, exactly as phase-2 training does. Without
                        # this the combined (phase-0) hand-off would feed policy 2 a
                        # slightly different quantity than it trained on. The attach
                        # gate has already forced the two within orient_tol, so the
                        # shift is at most 0.05; R1 and the attach gate never run again
                        # in this episode, so nothing else reads it.
                        self.target_tip_orientation = self.place_orientation
                        # The dome starts at the actual grasp position g (reachable
                        # by construction: the arm is holding the object there) and
                        # ends at place_position.
                        path = self._setup_dome(
                            self.sphere.position_collection[..., 0].copy()
                        )
                        # RE-PROBE AT THE GRASP. The reset probe measured an EMPTY arm;
                        # now there is a payload, so the estimate must be refreshed or the
                        # policy would carry a load while still reading ~0. Runs outside
                        # the policy's action loop, and the dome is rebuilt afterwards
                        # because the probe moves the arm.
                        if self._est_model is not None:
                            self._run_probe()
                            _p = self.sphere.position_collection[..., 0].copy()
                            if np.all(np.isfinite(_p)):
                                path = self._setup_dome(_p)
                        path_length = self._dome_arclen[-1]
                        # PHASE 1 TRAINING (policy 1) ends at the grasp -- the carry
                        # is policy 2's job, so there is nothing left to learn here.
                        if self.phase == 1:
                            self._phase1_done = True
                        if self.record_attach_states is not None:
                            self._record_attach()
                        print(
                            " ATTACHED at step %d (dist %.4f, orient %.4f) -- dome:"
                            " apex_y %.2f, length %.2f m, carry %.2f s"
                            % (
                                self.current_step,
                                dist,
                                orientation_dist,
                                path[:, 1].max(),
                                path_length,
                                self._carry_duration,
                            )
                        )

        # PHASE-1 DEADLINE (see attach_deadline in __init__): give up on an episode
        # that has not attached in time. This is a TRUNCATION -- the episode simply
        # stops being collected; no reward, penalty or bonus is applied here, and the
        # flag logic below routes it to `truncated` so SB3 still bootstraps V(s').
        # Marking it `terminated` instead would zero the future value, which WOULD
        # change the learning problem. Only fires pre-attach: once the object is
        # grasped the episode always runs its full length so the carry can finish.
        deadline_hit = bool(
            (not done)  # a NaN blow-up already ended this episode -> stays terminated
            and self.PICK_AND_RELEASE
            and self.attach_deadline is not None
            and (not self.attach_state["ever_attached"])
            and self.time_tracker > self.attach_deadline
        )
        # PHASE 1 (policy 1) also ends the moment the grasp succeeds -- same truncation
        # route as the deadline (no reward, no bonus, V(s') still bootstrapped).
        phase1_hit = bool((not done) and self._phase1_done)
        # PHASE 2 (policy 2) ends at the ARRIVAL, by the same truncation route: policy
        # 3 owns everything past the drop-off, so policy 2 must not be scored on it.
        phase2_hit = bool((not done) and self._phase2_done)
        # PHASE 3 (policy 3) ends at the RELEASE -- its task is complete there.
        phase3_hit = bool((not done) and self._phase3_done)

        if (
            self.current_step >= self.total_learning_steps
            or deadline_hit
            or phase1_hit
            or phase2_hit
            or phase3_hit
        ):
            done = True
            episode_wall_time = time.perf_counter() - self._episode_wall_start
            print(
                " Episode compute time: %.1f s (%d env steps, %.1f steps/s)"
                % (
                    episode_wall_time,
                    self.current_step,
                    self.current_step / episode_wall_time,
                )
            )
            if reward > 0:
                print(
                    " Reward greater than 0! Reward: %0.3f, Distance: %0.3f, Orientation: %0.3f -- %0.3f, %0.3f "
                    % (reward, dist, orientation_dist, reward_dist, orientation_penalty)
                )
            else:
                print(
                    " Finished simulation. Reward: %0.3f, Distance: %0.3f, Orientation: %0.3f -- %0.3f, %0.3f"
                    % (reward, dist, orientation_dist, reward_dist, orientation_penalty)
                )
        """ Done is a boolean to reset the environment before episode is completed """

        # COPY, not a reference. SB3's VecEnv can hand the same ndarray buffer back on
        # the next step; storing the reference would make previous_action alias the
        # current action and the action-rate penalty would silently evaluate to zero.
        self.previous_action = np.array(action, dtype=np.float64, copy=True)

        invalid_values_condition_state = _isnan_check(state)
        if invalid_values_condition_state == True:
            print(
                " Nan detected in the state other than position data, exiting simulation now"
            )
            reward = -10000
            state = np.zeros(state.shape)
            done = True
            nan_blowup = True

        # Gymnasium 5-tuple: NaN blow-up -> terminated; time limit OR the pre-attach
        # deadline -> truncated (both are "we stopped collecting", not "the future is
        # worthless", so SB3 must bootstrap V(s') for them -- it does this via the
        # TimeLimit.truncated info key that the gymnasium->VecEnv shim sets from
        # `truncated`). terminated is now defined as "done for any OTHER reason",
        # which keeps the NaN case exactly as before.
        truncated = bool(
            (not nan_blowup)
            and (
                self.current_step >= self.total_learning_steps
                or deadline_hit
                or phase1_hit
                or phase2_hit
                or phase3_hit
            )
        )
        terminated = bool(nan_blowup or (done and not truncated))

        return state, reward, terminated, truncated, {"ctime": self.time_tracker}

    def render(self, mode="human"):
        """
        This method does nothing, it is here for interfacing with Gymnasium.

        Parameters
        ----------
        mode

        Returns
        -------

        """
        return

    def post_processing(self, filename_video, SAVE_DATA=False, **kwargs):
        """
        Make video 3D of arm movement in time, and store the arm, target, obstacles, and actuation
        data.

        Parameters
        ----------
        filename_video : str
            Names of the videos to be made for post-processing.
        SAVE_DATA : boolean
            If true collected data in simulation saved.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """

        if self.COLLECT_DATA_FOR_POSTPROCESSING:

            plot_video_with_sphere_2D(
                [self.post_processing_dict_rod],
                [self.post_processing_dict_sphere],
                video_name="2d_" + filename_video,
                fps=self.video_fps,
                step=1,
                vis2D=False,
                **kwargs,
            )

            plot_video_with_sphere(
                [self.post_processing_dict_rod],
                [self.post_processing_dict_sphere],
                video_name="3d_" + filename_video,
                fps=self.video_fps,
                step=1,
                vis2D=False,
                **kwargs,
            )

            if SAVE_DATA == True:

                save_folder = os.path.join(os.getcwd(), "data")
                os.makedirs(save_folder, exist_ok=True)

                # Transform nodal to elemental positions
                position_rod = np.array(self.post_processing_dict_rod["position"])
                position_rod = 0.5 * (position_rod[..., 1:] + position_rod[..., :-1])

                # Pick-and-place drop-off info for the POVray renderer (static
                # pointer sphere at the phase-2 release location).
                place_data = {}
                if self.place_position is not None:
                    place_data["place_position"] = self.place_position
                    place_data["place_radius"] = self.place_radius
                # Trajectory carry path of the last episode (if an attach happened) --
                # for drawing T2(s) as a marker trail / dashed curve in POVray
                # and the matplotlib videos.
                if self._dome_start is not None:
                    place_data["dome_path"] = np.array(
                        [self._dome_point(s) for s in np.linspace(0.0, 1.0, 101)]
                    )

                np.savez(
                    os.path.join(save_folder, "arm_data.npz"),
                    position_rod=position_rod,
                    radii_rod=np.array(self.post_processing_dict_rod["radius"]),
                    n_elems_rod=self.shearable_rod.n_elems,
                    directors_rod=np.array(self.post_processing_dict_rod["directors"]),
                    position_sphere=np.array(
                        self.post_processing_dict_sphere["position"]
                    ),
                    radii_sphere=np.array(self.post_processing_dict_sphere["radius"]),
                    directors_sphere=np.array(
                        self.post_processing_dict_sphere["directors"]
                    ),
                    **place_data,
                )

                np.savez(
                    os.path.join(save_folder, "arm_activation.npz"),
                    torque_mag=np.array(
                        self.torque_profile_list_for_muscle_in_normal_dir["torque_mag"]
                    ),
                    torque_muscle=np.array(
                        self.torque_profile_list_for_muscle_in_normal_dir["torque"]
                    ),
                )

        else:

            raise RuntimeError(
                "call back function is not called anytime during simulation, "
                "change COLLECT_DATA=True"
            )