import numpy as np
import os

# Initialize directory
if not os.path.exists("./images"):
    os.makedirs("./images")

# Load elastic arm data
arm_and_target_data = np.load("../data/arm_data.npz")
arm_position = arm_and_target_data["position_rod"]
arm_radius = arm_and_target_data["radii_rod"]
arm_n_elem = arm_and_target_data["n_elems_rod"]

# Load target data. Normalize shapes so both data eras work:
#   paper-era files:  position_sphere (T,3,1), radii_sphere (T,1)
#   current Case_2:   position_sphere (T,3,1), radii_sphere (T,)
target_position = arm_and_target_data["position_sphere"]
if target_position.ndim == 3:
    target_position = target_position[..., -1]          # -> (T, 3)
target_radius = arm_and_target_data["radii_sphere"]
target_radius = np.asarray(target_radius).reshape(len(target_radius), -1)[:, -1]  # -> (T,)

# Plot directors for rod tip and sphere
arm_directors = arm_and_target_data["directors_rod"]
target_directors = arm_and_target_data["directors_sphere"]
vector_radius = 0.025 # Radius of lines
color_order = ["SlateBlue", "LimeGreen", "OrangeRed"]

# Phase-2 drop-off (pick-and-place): static pointer sphere marking where the
# release must happen. Read from the npz when present (saved by the current
# Case_2 post_processing); fall back to the training constant for older files.
if "place_position" in arm_and_target_data.files:
    place_position = arm_and_target_data["place_position"]
    place_radius = float(arm_and_target_data["place_radius"])
else:
    place_position = np.array([0.4, 0.05, -0.4])   # logging_bio_args.py constant
    place_radius = 0.05

# place_directors (the old q_place triad) is no longer drawn -- see the
# RELEASE-ORIENTATION REFERENCE block below for why.

# Attach visual cue: the target sphere changes color while it is being carried
# (tip holding it) and after it is parked at the drop-off. Heuristic on saved
# positions only; arm_position is elemental (half-element short of the tip
# node), hence the generous threshold.
CARRY_THRESHOLD = 0.1
target_color_free = "color rgb<1,0.5,0.4>"       # phase 1: reach target
target_color_carried = "color rgb<1,0.85,0.2>"   # attached / being carried
target_color_placed = "color rgb<0.3,0.9,0.4>"   # resting at the drop-off

scale=3 #simulation arm is 1m; scale/camera tuned together (paper setting)

# POVray transform shared by every object in the scene: positions are written
# as <z, x, y>*scale and then pushed through this chain. Keep them identical
# for any new object or it will not line up with the arm.
transform_suffix = "scale<4,4,4> rotate<0,90,90> translate<2,0,4>    }\n"

time = arm_position.shape[0]

for k in range(time):
    file1 = open("images/moving_arm%03d.inc"%k,"w")

    # Arm
    # TIP GEOMETRY.  The drawn rod used to stop short of the real one, which showed as a
    # false ~2 cm gap between the tip and the payload even when the simulation had them
    # in contact.  Two compounding causes:
    #   1. position_rod stores ELEMENT CENTRES (set_environment.py: 0.5*(p[1:]+p[:-1])),
    #      so the last control point sits half an element behind the true tip node.
    #   2. POV-Ray's b_spline sphere_sweep APPROXIMATES its control points -- the swept
    #      surface never reaches the first or last one, costing roughly another element.
    # Fixed by extrapolating the true end nodes back from the stored centres and then
    # repeating each end point so the b-spline is pulled onto it, staying smooth between.
    # NOTE the radius takes the same `scale` factor as the positions; dropping it renders
    # the rod scale-times too thin.
    _pts = [(arm_position[k][0][i], arm_position[k][1][i], arm_position[k][2][i],
             arm_radius[k][i]) for i in range(0, arm_n_elem)]
    def _extrap(a, b):          # half an element beyond `a`, directed away from `b`
        return (a[0] + 0.5*(a[0]-b[0]), a[1] + 0.5*(a[1]-b[1]),
                a[2] + 0.5*(a[2]-b[2]), a[3])
    _pts = [_extrap(_pts[0], _pts[1])] + _pts + [_extrap(_pts[-1], _pts[-2])]
    _pts = [_pts[0]]*2 + _pts + [_pts[-1]]*2
    file1.writelines("sphere_sweep\n{b_spline %d"% len(_pts))
    for _x, _y, _z, _r in _pts:
        file1.writelines(",\n<%f,%f,%f>,%f" % (scale*_z, scale*_x, scale*_y, scale*_r))
    file1.writelines("\ntexture{")
    file1.writelines("pigment{ color rgb<0.45,0.39,1>  transmit %f }"%(0.1))
    file1.writelines("finish{ phong 1 } }")
    file1.writelines(transform_suffix)

    # Base of arm
    file1.writelines("sphere\n{")
    for i in range(0, 1):
        file1.writelines(
            "\n<%f,%f,%f>,%f" % (scale * arm_position[k][2][i] , scale * (arm_position[k][0][i]) ,
                                 scale * arm_position[k][1][i] , 1.5 * scale * arm_radius[k][i]))
    file1.writelines("\ntexture{")
    file1.writelines("pigment{ color rgb<0.75,0.75,0.75>  transmit %f }" % (0.1))
    file1.writelines("finish{ phong 1 } }")
    file1.writelines(transform_suffix)

    # Target sphere, colored by task state (free / carried / placed)
    tip_to_sphere = np.linalg.norm(arm_position[k, :, -1] - target_position[k])
    sphere_to_place = np.linalg.norm(target_position[k] - place_position)
    if tip_to_sphere < CARRY_THRESHOLD:
        target_color = target_color_carried
    elif sphere_to_place < 2.0 * place_radius:
        target_color = target_color_placed
    else:
        target_color = target_color_free
    file1.writelines("sphere\n{")
    file1.writelines(
        "\n<%f,%f,%f>,%f" % (
        scale * target_position[k][2], scale * (target_position[k][0]), scale * target_position[k][1],
        scale * target_radius[k]))
    file1.writelines("\ntexture{")
    file1.writelines("pigment{ %s  transmit %f }" % (target_color, 0.1))
    file1.writelines("finish{ phong 1 } }")
    file1.writelines(transform_suffix)

    # Phase-2 drop-off pointer: static release-zone marker. Radius deliberately
    # capped at place_radius (= the target sphere's own size); visibility comes
    # from the brighter, more opaque green rather than a bigger shape.
    file1.writelines("sphere\n{")
    file1.writelines(
        "\n<%f,%f,%f>,%f" % (
        scale * place_position[2], scale * place_position[0], scale * place_position[1],
        scale * place_radius))
    file1.writelines("\ntexture{")
    file1.writelines("pigment{ color rgb<0.1,0.9,0.25>  transmit %f }" % (0.35))
    file1.writelines("finish{ phong 1 } }")
    file1.writelines(transform_suffix)

    # Drawing rod tip directors using vector
    for idx in range(3):
        arm_tip_vector_start = arm_position[k, :,-1]
        arm_tip_vector_end = arm_position[k, :, -1] + 5*arm_radius[k,-1] * arm_directors[k, idx, :, -1] # directors(time, vectorstart, vectorend, element index)
        file1.writelines("object\n{")
        file1.writelines("Vector\n(")
        file1.writelines(
            "\n<%f,%f,%f>,   <%f, %f, %f>, %f)" % (
                scale * (arm_tip_vector_start[2]),
                scale * (arm_tip_vector_start[0]),
                scale * (arm_tip_vector_start[1]),
                scale * (arm_tip_vector_end[2] ),
                scale * (arm_tip_vector_end[0] ),
                scale * (arm_tip_vector_end[1] ),
                vector_radius,
                ))
        file1.writelines("\ntexture{")
        file1.writelines(str("pigment{ color "+color_order[idx]+" transmit %f }") % (0.1))
        file1.writelines("finish{ phong 1 } }")
        file1.writelines(transform_suffix)

    # Drawing target directors using vector (live sphere frame)
    for idx in range(3):
        target_vector_start = target_position[k]
        target_vector_end = target_position[k] + 5 * target_radius[k] * target_directors[k, idx, :,
                                                                        -1]  # directors(time, vectorstart, vectorend, element index)
        file1.writelines("object\n{")
        file1.writelines("Vector\n(")
        file1.writelines(
            "\n<%f,%f,%f>,   <%f, %f, %f>, %f)" % (
                scale * (target_vector_start[2]),
                scale * (target_vector_start[0]),
                scale * (target_vector_start[1]),
                scale * (target_vector_end[2]),
                scale * (target_vector_end[0]),
                scale * (target_vector_end[1]),
                vector_radius,
            ))
        file1.writelines("\ntexture{")
        file1.writelines(str("pigment{ color "+color_order[idx]+" transmit %f }") % (0.1))
        file1.writelines("finish{ phong 1 } }")
        file1.writelines(transform_suffix)

    # TARGET TIP FRAME at the drop-off: what the rod tip must LOOK LIKE to release.
    #
    #   OrangeRed, pointing straight DOWN = the target d3 (tangent). Same colour as the
    #       tip's own d3 arrow, so the release condition is met exactly when the tip's
    #       orange arrow lines up with this one.
    #       Gate: p_down = (1 + t_tip . y_hat)/2 <= 0.15, i.e. within 46 deg of vertical.
    #   SlateBlue / LimeGreen, faint = d1 and d2. Drawn PALE on purpose: the reward
    #       constrains ONLY d3. It uses a direction dot product precisely because roll
    #       is unconstrained -- the quaternion form 1-(q.q')^2 was abandoned as it is
    #       sign-invariant (cannot express "anti-parallel") AND it pinned the roll,
    #       which is meaningless for a sphere. A solid triad would wrongly imply the
    #       tip must match d1/d2 as well.
    #
    # RAISED ORIGIN: place_position sits at floor + sphere_radius, so a frame drawn
    # there sends its down arrow through the ground plane. The origin is lifted so the
    # d3 head lands just above the drop-off instead of piercing the floor.
    #
    # REPLACES a triad that drew target_directors[0] -- the sphere's INITIAL orientation
    # (the old q_place), which nothing reads any more. That triad pointed d3 UP and so
    # implied the tip should ALIGN with it: the old, geometrically impossible objective
    # (measured p_place median 0.865, only 1.3% of releases under its threshold).
    #
    # PURELY VISUAL: this file only reads the saved npz. It cannot affect p_down, the
    # release gate, or anything in set_environment.py.
    # BACK AT THE TARGET POSITION, scaled down. Two constraints fix the geometry:
    # the drop-off pointer sphere spans y in [0, 2*place_radius] (centre
    # place_position, radius place_radius), so an arrow shorter than that from the
    # centre is swallowed by the sphere -- while one long enough to emerge downward
    # goes through the floor. Both are avoided by hanging the frame just above the
    # sphere: the d3 head lands exactly on its top, touching the target without
    # entering it or the ground. Tune SIZE to scale the whole key.
    SIZE = 4.0
    d3_len, ax_len = SIZE * place_radius, 0.8 * SIZE * place_radius
    frame_origin = place_position + np.array(
        [0.0, place_radius + d3_len, 0.0])
    for vec, col, trans in (
            (np.array([0.0, -1.0, 0.0]) * d3_len, "OrangeRed", 0.10),
            (np.array([1.0, 0.0, 0.0]) * ax_len, "SlateBlue", 0.70),
            (np.array([0.0, 0.0, 1.0]) * ax_len, "LimeGreen", 0.70)):
        v_end = frame_origin + vec
        file1.writelines("object\n{")
        file1.writelines("Vector\n(")
        file1.writelines(
            "\n<%f,%f,%f>,   <%f, %f, %f>, %f)" % (
                scale * (frame_origin[2]),
                scale * (frame_origin[0]),
                scale * (frame_origin[1]),
                scale * (v_end[2]),
                scale * (v_end[0]),
                scale * (v_end[1]),
                vector_radius,
            ))
        file1.writelines("\ntexture{")
        file1.writelines(str("pigment{ color "+col+" transmit %f }") % (trans))
        file1.writelines("finish{ phong 1 } }")
        file1.writelines(transform_suffix)

    file1.close()
    file2 = open("images/moving_arm%03d.pov" % k, "w")
    file2.writelines("#include \"../camera_position.inc\"\n")
    file2.writelines("#include \"moving_arm%03d.inc\"\n" % k)
    # OPTIONAL PER-SEED CAMERA. camera_position.inc instantiates camera{} at its case-21
    # block; POVray honours the LAST camera statement in the scene, so writing another one
    # here overrides it for this render only -- no edit to the shared .inc, and seeds
    # rendered without these variables keep their existing framing exactly.
    # Set CAM_POS/CAM_LOOK as "x,y,z" and CAM_ANGLE as a number.
    cam_pos = os.environ.get("CAM_POS")
    if cam_pos:
        cam_look = os.environ.get("CAM_LOOK", "-1.42,6.17,5.65")
        cam_ang = os.environ.get("CAM_ANGLE", "31.4")
        file2.writelines(
            "camera{ location <%s>\n        right x*image_width/image_height\n"
            "        angle %s\n        look_at <%s>\n      }\n"
            % (cam_pos, cam_ang, cam_look))
        # camera_position.inc puts a dim flash light AT the old camera; add the matching
        # one here so a mirrored viewpoint is not lit from behind.
        file2.writelines(
            "light_source{ <%s> color rgb<0.9,0.9,1>*0.1 }\n" % cam_pos)
    file2.close()
