from datetime import datetime
import pdr
from load_kernels import spice
import numpy as np
import scipy
import math
from project_ellipsoid import ellipsoid_to_equi, equi_to_ellipsoid

ORIGIN_BODY = "Sun"
SATELLITE_DICT = {"TITAN": "SATURN BARYCENTER"}
TARGETS = ["SUN", "SATURN BARYCENTER", "TITAN", "ENCELADUS", "DIONE"]
RADIUS = {
    "SATURN": [60268, 60268, 54364, 58232],
    "SATURN BARYCENTER": [60268, 60268, 54364, 58232],
    "TITAN": [
        250 + 2574.73 + 0.09,
        250 + 2574.73 + 0.09,
        250 + 2574.73 - 0.09,
        250 + 2574.73,
    ],
    "ENCELADUS": [513.2 / 2, 502.8 / 2, 496.6 / 2, 252.1],
    "DIONE": [1128.8 / 2, 1128.8 / 2, 1119.2 / 2, 561.4],
}  # km

def normalize(arr):
    return np.array(arr)/np.linalg.norm(arr)
def axis_rotmat(axis,angle=None):
    if angle is None:
        if np.linalg.norm(axis)==0:
            return np.eye(3)
        angle = np.linalg.norm(axis)
        axis = axis/angle
    if angle==0:
        return np.eye(3)
    axis = normalize(axis)
    sa = np.sin(angle)
    ca = np.cos(angle)
    omc = (1-np.cos(angle))
    return np.array([
        [ca+axis[0]**2*omc,axis[0]*axis[1]*omc-axis[2]*sa,axis[0]*axis[2]*omc+axis[1]*sa],
        [axis[1]*axis[0]*omc+axis[2]*sa,ca+axis[1]**2*omc,axis[1]*axis[2]*omc-axis[0]*sa],
        [axis[2]*axis[0]*omc-axis[1]*sa,axis[2]*axis[1]*omc+axis[0]*sa,ca+axis[2]**2*omc]
    ])

def get_img(file_name, justdate=False):
    data = pdr.read(file_name)
    if justdate:
        info_dict = dict(
            [
                [el.split("=")[0].strip(), "=".join(el.split("=")[1:]).strip()]
                for el in data["LABEL"].split("\n")
            ]
        )
        if "IMAGE_MID_TIME" not in info_dict:
            return
        return datetime.strptime(info_dict["IMAGE_MID_TIME"], "%Y-%jT%H:%M:%S.%f")
    if "LABEL" not in data:
        return
    if "IMAGE" not in data:
        return
    info_dict = dict(
        [
            [el.split("=")[0].strip(), "=".join(el.split("=")[1:]).strip()]
            for el in data["LABEL"].split("\n")
        ]
    )
    print(info_dict)
    if "IMAGE_MID_TIME" not in info_dict:
        return
    img_date = datetime.strptime(info_dict["IMAGE_MID_TIME"], "%Y-%jT%H:%M:%S.%f")
    if justdate:
        return img_date
    return [data["IMAGE"], img_date, info_dict]


def draw_saturn_rings(orientation, timestamp, imscale=1, imsize=1024):
    saturn_pos, light_time = spice.spkpos(
        "SATURN BARYCENTER",  # Target: the spacecraft
        timestamp,  # Ephemeris time (ET)
        "J2000",  # Reference frame
        "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
        "Cassini",  # Observer (or 'SUN', or another body)
    )
    rotated = np.matmul(orientation, saturn_pos)
    rotated[2] *= -1
    saturn_orientation = np.matmul(orientation, saturn_north_pole)
    saturn_orientation/=np.linalg.norm(saturn_orientation)
    saturn_a = np.cross(saturn_orientation,np.array([1,0,0]))
    saturn_a/=np.linalg.norm(saturn_a)
    saturn_b = np.cross(saturn_orientation,saturn_a)
    saturn_b/=np.linalg.norm(saturn_b)
    x_ = np.linspace(-imscale, imscale, num=imsize, endpoint=False)
    y_ = np.linspace(-imscale, imscale, num=imsize, endpoint=False)
    x, y = np.meshgrid(x_, y_)
    v = np.dstack([x, y, np.ones_like(x)])
    print(np.sum(saturn_orientation*rotated),np.sum(v*saturn_orientation[None,None,:],axis=-1))
    # plt.imshow(np.sum(v*saturn_orientation[None,None,:],axis=-1))
    # plt.title("np.sum(v*saturn_orientation[None,None,:],axis=-1)")
    # plt.show()
    t = np.sum(saturn_orientation*rotated)/np.sum(v*saturn_orientation[None,None,:],axis=-1)
    # plt.imshow(np.minimum(100000,np.maximum(-100000,t)))
    # plt.title("t")
    # plt.show()
    on_ring_plane = t[:,:,None]*v
    dist_from_saturn = np.linalg.norm(on_ring_plane-rotated[None,None,:],axis=-1)
    # plt.title("dist_from_saturn")
    is_rings = np.logical_and(dist_from_saturn<60300+80000,dist_from_saturn>60300+7000)
    # plt.imshow(is_rings*(-t))
    # plt.show()
    return is_rings*(-t)

def target_ellipsoid(position, orientation, timestamp, target_name):
    print("position, orientation, timestamp, target_name")
    print(position, orientation, timestamp, target_name)
    target = target_name[1:-1]

    _, light_time = spice.spkpos(
        target,  # Target: the spacecraft
        timestamp,  # Ephemeris time (ET)
        "J2000",  # Reference frame
        "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
        "Cassini",  # Observer (or 'SUN', or another body)
    )
    target_pos, _ = spice.spkpos(
        target,  # Target: the spacecraft
        timestamp - light_time,  # Ephemeris time (ET)
        "J2000",  # Reference frame
        "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
        "Cassini",  # Observer (or 'SUN', or another body)
    )
    print("target_posA",target, timestamp , light_time,"Cassini")
    print("target_posA",target_pos, target , SATELLITE_DICT)
    if target in SATELLITE_DICT:
        orbits_around_pos_then, _ = spice.spkpos(
            SATELLITE_DICT[target],  # Target: the spacecraft
            timestamp - light_time,  # Ephemeris time (ET)
            "J2000",  # Reference frame
            "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
            ORIGIN_BODY,  # Observer (or 'SUN', or another body)
        )
        orbits_around_pos_now, _ = spice.spkpos(
            SATELLITE_DICT[target],  # Target: the spacecraft
            timestamp - light_time,  # Ephemeris time (ET)
            "J2000",  # Reference frame
            "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
            ORIGIN_BODY,  # Observer (or 'SUN', or another body)
        )
        print("target_posB",target_pos, orbits_around_pos_now, orbits_around_pos_then)
        target_pos = target_pos + orbits_around_pos_now - orbits_around_pos_then
    rotated = np.matmul(orientation, target_pos - position)
    rotated[2] *= -1
    print("rotated",rotated)
    in_xy = rotated[:2] / rotated[2]
    target_center = in_xy
    target_center_xyz = rotated

    if target not in RADIUS:
        return None

    r = RADIUS[target][0]
    p = np.linalg.norm(target_center_xyz)
    theta = 2 * np.arcsin(r / p)
    phi_low = (
        np.arctan2(np.linalg.norm(target_center_xyz[:2]), target_center_xyz[2])
        - theta / 2
    )
    phi_high = (
        np.arctan2(np.linalg.norm(target_center_xyz[:2]), target_center_xyz[2])
        + theta / 2
    )
    print("phi_low,phi_high",phi_low,phi_high)
    radius_on_screen = (np.tan(phi_high) - np.tan(phi_low)) / 2

    radius_on_screen_ = RADIUS[target][-1] / target_center_xyz[2]

    r = RADIUS[target][2]
    p = np.linalg.norm(target_center_xyz)
    theta = 2 * np.arcsin(r / p)
    phi_low = (
        np.arctan2(np.linalg.norm(target_center_xyz[:2]), target_center_xyz[2])
        - theta / 2
    )
    phi_high = (
        np.arctan2(np.linalg.norm(target_center_xyz[:2]), target_center_xyz[2])
        + theta / 2
    )
    print("phi_low,phi_high",phi_low,phi_high)
    radius2_on_screen = (np.tan(phi_high) - np.tan(phi_low)) / 2

    saturn_orientation = np.matmul(orientation, saturn_north_pole)
    north_pole_angle_away_from_screen = np.arctan2(
        saturn_orientation[2], np.linalg.norm(saturn_orientation[:2])
    )
    longer_axis = saturn_orientation[:2] / np.linalg.norm(saturn_orientation[:2])
    shorter_axis = np.array(longer_axis[::-1])
    shorter_axis[0] *= -1
    shorter_axis = radius_on_screen * shorter_axis
    longer_axis = (
        np.sqrt(
            np.sin(north_pole_angle_away_from_screen) ** 2 * radius_on_screen**2
            + np.cos(north_pole_angle_away_from_screen) ** 2 * radius2_on_screen**2
        )
        * longer_axis
    )
    print("longer_axis,shorter_axis,north_pole_angle_away_from_screen")
    print(longer_axis,shorter_axis,north_pole_angle_away_from_screen)
    M_line = np.cross(
        np.array([0, 0, 1]), saturn_orientation / np.linalg.norm(saturn_orientation)
    )
    return (
        np.eye(3) * np.array(RADIUS[target])[None, :3],
        axis_rotmat(
            M_line / np.linalg.norm(M_line), -np.arcsin(np.linalg.norm(M_line))
        ),
        target_center_xyz,
    )



def target_ellipse(position, orientation, timestamp, target_name):
    target = target_name[1:-1]

    _, light_time = spice.spkpos(
        target,  # Target: the spacecraft
        timestamp,  # Ephemeris time (ET)
        "J2000",  # Reference frame
        "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
        "Cassini",  # Observer (or 'SUN', or another body)
    )
    target_pos, _ = spice.spkpos(
        target,  # Target: the spacecraft
        timestamp - light_time,  # Ephemeris time (ET)
        "J2000",  # Reference frame
        "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
        ORIGIN_BODY,  # Observer (or 'SUN', or another body)
    )
    if target in SATELLITE_DICT:
        orbits_around_pos_then, _ = spice.spkpos(
            SATELLITE_DICT[target],  # Target: the spacecraft
            timestamp - light_time,  # Ephemeris time (ET)
            "J2000",  # Reference frame
            "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
            ORIGIN_BODY,  # Observer (or 'SUN', or another body)
        )
        orbits_around_pos_now, _ = spice.spkpos(
            SATELLITE_DICT[target],  # Target: the spacecraft
            timestamp - light_time,  # Ephemeris time (ET)
            "J2000",  # Reference frame
            "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
            ORIGIN_BODY,  # Observer (or 'SUN', or another body)
        )
        target_pos = target_pos + orbits_around_pos_now - orbits_around_pos_then
    rotated = np.matmul(orientation, target_pos - position)
    rotated[2] *= -1
    in_xy = rotated[:2] / rotated[2]
    target_center = in_xy
    target_center_xyz = rotated

    if target not in RADIUS:
        return None

    r = RADIUS[target][0]
    p = np.linalg.norm(target_center_xyz)
    theta = 2 * np.arcsin(r / p)
    phi_low = (
        np.arctan2(np.linalg.norm(target_center_xyz[:2]), target_center_xyz[2])
        - theta / 2
    )
    phi_high = (
        np.arctan2(np.linalg.norm(target_center_xyz[:2]), target_center_xyz[2])
        + theta / 2
    )
    radius_on_screen = (np.tan(phi_high) - np.tan(phi_low)) / 2

    radius_on_screen_ = RADIUS[target][-1] / target_center_xyz[2]

    r = RADIUS[target][2]
    p = np.linalg.norm(target_center_xyz)
    theta = 2 * np.arcsin(r / p)
    phi_low = (
        np.arctan2(np.linalg.norm(target_center_xyz[:2]), target_center_xyz[2])
        - theta / 2
    )
    phi_high = (
        np.arctan2(np.linalg.norm(target_center_xyz[:2]), target_center_xyz[2])
        + theta / 2
    )
    radius2_on_screen = (np.tan(phi_high) - np.tan(phi_low)) / 2

    saturn_orientation = np.matmul(orientation, saturn_north_pole)
    north_pole_angle_away_from_screen = np.arctan2(
        saturn_orientation[2], np.linalg.norm(saturn_orientation[:2])
    )
    longer_axis = saturn_orientation[:2] / np.linalg.norm(saturn_orientation[:2])
    shorter_axis = np.array(longer_axis[::-1])
    shorter_axis[0] *= -1
    shorter_axis = radius_on_screen * shorter_axis
    longer_axis = (
        np.sqrt(
            np.sin(north_pole_angle_away_from_screen) ** 2 * radius_on_screen**2
            + np.cos(north_pole_angle_away_from_screen) ** 2 * radius2_on_screen**2
        )
        * longer_axis
    )
    return [
        target_center[0]
        + np.cos(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * longer_axis[0]
        + np.sin(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * shorter_axis[0],
        target_center[1]
        + np.cos(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * longer_axis[1]
        + np.sin(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * shorter_axis[1],
    ]

def target_ellipsoid_ellipse(R, M, C):
    print("C",C)
    projected_C = C[:2] / C[2]
    # R = np.matmul(M,R)
    r = R[0, 0]
    p = np.linalg.norm(C)
    theta = 2 * np.arcsin(r / p)
    phi_low = np.arctan2(np.linalg.norm(C[:2]), C[2]) - theta / 2
    phi_high = np.arctan2(np.linalg.norm(C[:2]), C[2]) + theta / 2
    radius_on_screen = (np.tan(phi_high) - np.tan(phi_low)) / 2

    radius_on_screen_ = np.sum(R) / 3 / C[2]


    r = R[2, 2]
    p = np.linalg.norm(C)
    theta = 2 * np.arcsin(r / p)
    phi_low = np.arctan2(np.linalg.norm(C[:2]), C[2]) - theta / 2
    phi_high = np.arctan2(np.linalg.norm(C[:2]), C[2]) + theta / 2
    radius2_on_screen = (np.tan(phi_high) - np.tan(phi_low)) / 2

    saturn_orientation = -np.matmul(M, np.array([0, 0, 1]))
    north_pole_angle_on_screen = np.arctan2(
        saturn_orientation[1], saturn_orientation[0]
    )
    north_pole_angle_away_from_screen = np.arctan2(
        saturn_orientation[2], np.linalg.norm(saturn_orientation[:2])
    )
    longer_axis = saturn_orientation[:2] / np.linalg.norm(saturn_orientation[:2])
    shorter_axis = np.array(longer_axis[::-1])
    shorter_axis[0] *= -1
    shorter_axis = radius_on_screen * shorter_axis
    longer_axis = (
        np.sqrt(
            np.sin(north_pole_angle_away_from_screen) ** 2 * radius_on_screen**2
            + np.cos(north_pole_angle_away_from_screen) ** 2 * radius2_on_screen**2
        )
        * longer_axis
    )
    

    return [
        projected_C[0]
        + np.cos(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * longer_axis[0]
        + np.sin(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * shorter_axis[0],
        projected_C[1]
        + np.cos(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * longer_axis[1]
        + np.sin(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * shorter_axis[1],
        ]



# testing to get ring north pole vvv
rings_ra = 40.589
rings_dec = 83.537
def ring_loss(x_in):
    global saturn_north_pole
    rings_ra= 40.589+x_in[0]*100
    rings_dec= 83.537+x_in[1]*100
    saturn_north_pole = np.array([np.cos(rings_ra*math.pi/180)*np.cos(rings_dec*math.pi/180),np.sin(rings_ra*math.pi/180)*np.cos(rings_dec*math.pi/180),np.sin(rings_dec*math.pi/180)])

    loss_val = 0
    position = np.array([0.,0.,0.])
    orientation_new=np.array( [[ 0.08430472,  0.06998057,  0.99397959],
                         [-0.69480352,  0.71915173,  0.00829853],
                         [-0.71424141, -0.69132012,  0.10925063]])
    spicey_time = 370959807.39634377
    R_ring, M_ring, C_ring = target_ellipsoid(
        position, orientation_new, spicey_time, target_name="'SATURN BARYCENTER'"
    )
    # print("C_ring", C_ring)
    R_ring[0, 0] = 60300 + 80000
    R_ring[1, 1] = 60300 + 80000
    R_ring[2, 2] = 1
    print("R_ring, M_ring, C_ring",R_ring, M_ring, C_ring)
    ring_ellipse = np.array(target_ellipsoid_ellipse(R_ring, M_ring, C_ring))
    print("ring_ellipse",ring_ellipse)
    print("np.where(np.absolute(ring_ellipse[1])<0.02)[0]",np.where(np.absolute(ring_ellipse[1])<0.02)[0])
    ring_top = np.max(ring_ellipse[0][np.where(np.absolute(ring_ellipse[1])<0.02)[0]])
    ring_bottom = np.min(ring_ellipse[0][np.where(np.absolute(ring_ellipse[1])<0.02)[0]])
    target_ring_top = -4*10**(-5)+0.0001# taking 0.002772 to be the center
    target_ring_bottom = -0.00038
    loss_val += (ring_top-target_ring_top)**2+(ring_bottom-target_ring_bottom)**2#+titan_loss

    return loss_val
# testing to get ring north pole ^^^
res = scipy.optimize.minimize(ring_loss,[0,0])
print(res)
rings_ra = 40.589+res.x[0]*100
rings_dec = 83.537+res.x[1]*100
saturn_north_pole = np.array([np.cos(rings_ra * math.pi / 180) * np.cos(rings_dec * math.pi / 180),
                              np.sin(rings_ra * math.pi / 180) * np.cos(rings_dec * math.pi / 180),
                              np.sin(rings_dec * math.pi / 180)])


def render(position, orientation, timestamp, color=None, target_name=None):
    precision = 0.86400  # seconds
    duration_back = 100  # * precision
    
    target_positions = {}
    for target in TARGETS:
        print("target",target)
        print(target.lower(), target_name[1:-1].lower())
        #if target_name is not None:
        #    if target.lower() != target_name[1:-1].lower():
        #        continue
        target_path_xyz = []
        target_path = []
        for back_time in range(duration_back):
            _, light_time = spice.spkpos(
                target,  # Target: the spacecraft
                timestamp - precision * back_time,  # Ephemeris time (ET)
                "J2000",  # Reference frame
                "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
                "Cassini",  # Observer (or 'SUN', or another body)
            )
            target_pos, _ = spice.spkpos(
                target,  # Target: the spacecraft
                timestamp - precision * back_time - light_time,  # Ephemeris time (ET)
                "J2000",  # Reference frame
                "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
                ORIGIN_BODY,  # Observer (or 'SUN', or another body)
            )
            if target in SATELLITE_DICT:
                orbits_around_pos_then, _ = spice.spkpos(
                    SATELLITE_DICT[target],  # Target: the spacecraft
                    timestamp
                    - precision * back_time
                    - light_time,  # Ephemeris time (ET)
                    "J2000",  # Reference frame
                    "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
                    ORIGIN_BODY,  # Observer (or 'SUN', or another body)
                )
                orbits_around_pos_now, _ = spice.spkpos(
                    SATELLITE_DICT[target],  # Target: the spacecraft
                    timestamp - light_time,  # Ephemeris time (ET)
                    "J2000",  # Reference frame
                    "NONE",  # Aberration correction (use 'LT+S' for light time + stellar aberration)
                    ORIGIN_BODY,  # Observer (or 'SUN', or another body)
                )
                target_pos = target_pos + orbits_around_pos_now - orbits_around_pos_then
            rotated = np.matmul(orientation, target_pos - position)
            rotated[2] *= -1
            in_xy = rotated[:2] / rotated[2]
            target_path.append(in_xy)
            target_path_xyz.append(rotated)
        target_path = np.array(target_path)
        target_path_xyz = np.array(target_path_xyz)
        #if target_name is None:
        #    plt.plot(
        #        np.array(target_path)[:, 0],
        #        np.array(target_path)[:, 1],
        #        label=target,
        #        color=color,
        #    )
        print(target, RADIUS)
        if target not in RADIUS:
            continue

        saturn_orientation = np.matmul(orientation, saturn_north_pole)
        M_line = np.cross(
            np.array([0, 0, 1]), saturn_orientation / np.linalg.norm(saturn_orientation)
        )
        M = axis_rotmat(
            M_line / np.linalg.norm(M_line), -np.arcsin(np.linalg.norm(M_line))
        )

        #if target_name is None:
        #    render_ellipsoid(
        #        np.eye(3) * RADIUS[target][:3], M, target_path_xyz[0], color=color
        #    )

        r = RADIUS[target][0]
        p = np.linalg.norm(target_path_xyz[0])
        theta = 2 * np.arcsin(r / p)
        phi_low = (
            np.arctan2(np.linalg.norm(target_path_xyz[0, :2]), target_path_xyz[0, 2])
            - theta / 2
        )
        phi_high = (
            np.arctan2(np.linalg.norm(target_path_xyz[0, :2]), target_path_xyz[0, 2])
            + theta / 2
        )
        radius_on_screen = (np.tan(phi_high) - np.tan(phi_low)) / 2
        
        print("srscsc",target)
        if target_name is not None:
            target_positions[target] = [target_path[0], np.absolute(radius_on_screen)]
    return target_positions

def render_ellipsoid(R, M, C, color="green", ax=None, plotting=False):
    if plotting:
        if ax is None:
            ax = plt.gca()
    projected_C = C[:2] / C[2]
    # R = np.matmul(M,R)
    r = R[0, 0]
    p = np.linalg.norm(C)
    theta = 2 * np.arcsin(r / p)
    phi_low = np.arctan2(np.linalg.norm(C[:2]), C[2]) - theta / 2
    phi_high = np.arctan2(np.linalg.norm(C[:2]), C[2]) + theta / 2
    radius_on_screen = (np.tan(phi_high) - np.tan(phi_low)) / 2

    radius_on_screen_ = np.sum(R) / 3 / C[2]

    if plotting:
        ax.plot(
            projected_C[0]
            + np.cos(np.linspace(0, 2 * math.pi, num=100, endpoint=True))
            * radius_on_screen_,
            projected_C[1]
            + np.sin(np.linspace(0, 2 * math.pi, num=100, endpoint=True))
            * radius_on_screen_,
            color="red",
        )

    r = R[2, 2]
    p = np.linalg.norm(C)
    theta = 2 * np.arcsin(r / p)
    phi_low = np.arctan2(np.linalg.norm(C[:2]), C[2]) - theta / 2
    phi_high = np.arctan2(np.linalg.norm(C[:2]), C[2]) + theta / 2
    radius2_on_screen = (np.tan(phi_high) - np.tan(phi_low)) / 2

    saturn_orientation = -np.matmul(M, np.array([0, 0, 1]))
    north_pole_angle_on_screen = np.arctan2(
        saturn_orientation[1], saturn_orientation[0]
    )
    north_pole_angle_away_from_screen = np.arctan2(
        saturn_orientation[2], np.linalg.norm(saturn_orientation[:2])
    )
    longer_axis = saturn_orientation[:2] / np.linalg.norm(saturn_orientation[:2])
    shorter_axis = np.array(longer_axis[::-1])
    shorter_axis[0] *= -1
    shorter_axis = radius_on_screen * shorter_axis
    longer_axis = (
        np.sqrt(
            np.sin(north_pole_angle_away_from_screen) ** 2 * radius_on_screen**2
            + np.cos(north_pole_angle_away_from_screen) ** 2 * radius2_on_screen**2
        )
        * longer_axis
    )

    if plotting:
        ax.plot(
            projected_C[0]
            + np.cos(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * longer_axis[0]
            + np.sin(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * shorter_axis[0],
            projected_C[1]
            + np.cos(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * longer_axis[1]
            + np.sin(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * shorter_axis[1],
            color=color,
        )
    else:
        return [projected_C[0]
            + np.cos(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * longer_axis[0]
            + np.sin(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * shorter_axis[0],
            projected_C[1]
            + np.cos(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * longer_axis[1]
            + np.sin(np.linspace(0, 2 * math.pi, num=100, endpoint=True)) * shorter_axis[1]]
