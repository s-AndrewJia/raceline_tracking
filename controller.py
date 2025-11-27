import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

def get_steer_rate(delta_r, delta, min_steer_rate, max_steer_rate) -> float:
    K_p = 10.0
    steer_rate = K_p * (delta_r - delta)

    steer_rate = np.clip(steer_rate, min_steer_rate, max_steer_rate)

    return steer_rate

def get_acceleration(v_r, v, min_acceleration, max_acceleration) -> float:
    K_p = 10.0
    acceleration = K_p * (v_r - v)

    acceleration = np.clip(acceleration, min_acceleration, max_acceleration)

    return acceleration

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    # [steer angle, velocity]
    assert(desired.shape == (2,))

    delta_r = desired[0]
    delta = state[2]
    min_steer_rate = parameters[7]
    max_steer_rate = parameters[9]

    steer_rate = get_steer_rate(delta_r, delta, min_steer_rate, max_steer_rate)

    v_r = desired[1]
    v = state[3]
    min_acceleration = parameters[8]
    max_acceleration = parameters[10]

    acceleration = get_acceleration(v_r, v, min_acceleration, max_acceleration)

    return np.array([steer_rate, acceleration])

def get_next_ind(centerline, ind, lookahead):
    N = centerline.shape[0]
    total_dist = 0.0
    next_ind = ind

    while total_dist < lookahead:
        next_ind = (next_ind + 1) % N
        p_curr = centerline[(next_ind - 1) % N]
        p_next = centerline[next_ind]
        step = np.linalg.norm(p_next - p_curr)
        total_dist += step

    return next_ind

def get_delta_r(s_x, s_y, phi, l_wb, centerline, curr_ind, min_steering_angle, max_steering_angle) -> float:
    lookahead = 12
    next_ind = get_next_ind(centerline, curr_ind, lookahead)
    r_x, r_y = centerline[next_ind]

    phi_des = np.arctan2(r_y - s_y, r_x - s_x)
    alpha = phi_des - phi
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

    delta_r = np.arctan2(2 * l_wb * np.sin(alpha), lookahead)

    delta_r = np.clip(delta_r, min_steering_angle, max_steering_angle)

    return delta_r

def get_v_r(racetrack, N, centerline, curr_ind, min_velocity, max_velocity) -> float:
    prev_ind = (curr_ind - 3) % N
    next_ind = (curr_ind + 3) % N

    p_prev = centerline[prev_ind]
    p_curr = centerline[curr_ind]
    p_next = centerline[next_ind]

    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    curv_now = np.abs(np.cross(v1, v2) /
                    (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

    horizon = 10
    p_future = centerline[(curr_ind + horizon) % N]
    v_future = p_future - p_curr
    curv_future = np.abs(
        np.cross(v2, v_future) /
        (np.linalg.norm(v2) * np.linalg.norm(v_future) + 1e-9)
    )

    curvature = 0.7 * curv_now + 0.3 * curv_future

    v_max = 60.0
    k_speed = 3.0
    v_r = v_max * np.exp(-k_speed * curvature)

    right = racetrack.right_boundary[curr_ind]
    left  = racetrack.left_boundary[curr_ind]
    track_width = np.linalg.norm(right - left)
    width_factor = np.clip((track_width - 7) / 7, 0.7, 1.3)

    v_r *= width_factor

    v_r = float(np.clip(v_r, 15.0, v_max))

    v_r = np.clip(v_r, min_velocity, max_velocity)

    return v_r

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    s_x, s_y = state[0], state[1]
    phi = state[4]
    l_wb = parameters[0]

    min_steering_angle = parameters[1]
    max_steering_angle = parameters[4]

    min_velocity = parameters[2]
    max_velocity = parameters[5]

    centerline = racetrack.centerline
    N = centerline.shape[0]

    diffs = centerline - np.array([s_x, s_y])
    dists = np.linalg.norm(diffs, axis=1)
    curr_ind = np.argmin(dists)

    delta_r = get_delta_r(s_x, s_y, phi, l_wb, centerline, curr_ind, min_steering_angle, max_steering_angle)

    v_r = get_v_r(racetrack, N, centerline, curr_ind, min_velocity, max_velocity)

    return np.array([delta_r, v_r]).T
