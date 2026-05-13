import math

import numpy as np
from matplotlib import pyplot as plt


def get_k(R, M, C, arr_shape, imscale=1):
    x_ = np.linspace(-imscale, imscale, num=arr_shape[0], endpoint=False)
    y_ = np.linspace(-imscale, imscale, num=arr_shape[1], endpoint=False)
    x, y = np.meshgrid(x_, y_)
    v = np.dstack([x, y, np.ones_like(x)])
    RMv = np.matmul(
        np.matmul(np.linalg.inv(R), np.linalg.inv(M))[None, None, :, :],
        v[:, :, :, None],
    )[:, :, :, 0]
    RMC = np.matmul(np.matmul(np.linalg.inv(R), np.linalg.inv(M)), C)
    a = np.sum(RMv * RMv, axis=-1)
    b = -2 * np.sum(RMv * RMC[None, None, :], axis=-1)
    c = np.dot(RMC, RMC) - 1
    k_a = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    k_b = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    mid_k = ((k_a + k_b) / 2).astype(float)
    mid_k = np.nan_to_num(mid_k, nan=-1000)
    for i in range(5):
        mid_k[mid_k == -1000] = np.roll(mid_k, 1, axis=1)[mid_k == -1000]
        mid_k[mid_k == -1000] = np.roll(mid_k, -1, axis=1)[mid_k == -1000]
        mid_k[mid_k == -1000] = np.roll(mid_k, 1, axis=0)[mid_k == -1000]
        mid_k[mid_k == -1000] = np.roll(mid_k, -1, axis=0)[mid_k == -1000]
    return mid_k


def ellipsoid_to_equi(ellipsoid_arr, R, M, C, k=None, imscale=1, imsize=600):
    if k is None:
        k = get_k(R, M, C, ellipsoid_arr.shape, imscale=imscale)
    phi_ = np.linspace(-math.pi / 2, math.pi / 2, num=imsize, endpoint=False)
    theta_ = np.linspace(-math.pi, math.pi, num=2 * imsize, endpoint=False)
    phi, theta = np.meshgrid(phi_, theta_)
    xyz = np.dstack(
        [np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)]
    )
    xyz = np.matmul(R, xyz[:, :, :, None])[:, :, :, 0]

    rotated_xyz = C + np.matmul(M, xyz[:, :, :, None])[:, :, :, 0]
    xy = rotated_xyz[:, :, :2] / rotated_xyz[:, :, 2, None]
    xy = xy / imscale

    paint = ellipsoid_arr
    xy_idx = np.floor((1 + xy) / 2 * ellipsoid_arr.shape[0])
    xy_idx_outside = np.logical_or(
        np.logical_or(
            xy_idx[:, :, 0] >= ellipsoid_arr.shape[0],
            xy_idx[:, :, 1] >= ellipsoid_arr.shape[1],
        ),
        np.logical_or(xy_idx[:, :, 0] < 0, xy_idx[:, :, 1] < 0),
    )
    xy_idx = np.minimum(ellipsoid_arr.shape[0] - 1, np.maximum(0, xy_idx)).astype(int)
    painted_k = np.reshape(
        k[np.ndarray.flatten(xy_idx[:, :, 1]), np.ndarray.flatten(xy_idx[:, :, 0])],
        xy.shape[:2],
    )
    z_close = (rotated_xyz[:, :, 2] <= painted_k).astype(int)

    invalid = np.ones_like(xy_idx[:, :, 0])
    invalid *= z_close
    invalid *= np.logical_not(xy_idx_outside).astype(int)
    painted_strip = paint[
        np.ndarray.flatten(xy_idx[:, :, 1]), np.ndarray.flatten(xy_idx[:, :, 0])
    ]
    painted_surface = np.reshape(painted_strip, xy.shape[:2])
    painted_surface[invalid == 0] = np.nan
    return painted_surface


def equi_to_ellipsoid(equi_arr, R, M, C, imscale=1, imsize=600):
    x_ = np.linspace(-imscale, imscale, num=imsize, endpoint=False)
    y_ = np.linspace(-imscale, imscale, num=imsize, endpoint=False)
    x, y = np.meshgrid(x_, y_)
    v = np.dstack([x, y, np.ones_like(x)])
    RMv = np.matmul(
        np.matmul(np.linalg.inv(R), np.linalg.inv(M))[None, None, :, :],
        v[:, :, :, None],
    )[:, :, :, 0]
    RMC = np.matmul(np.matmul(np.linalg.inv(R), np.linalg.inv(M)), C)
    a = np.sum(RMv * RMv, axis=-1)
    b = -2 * np.sum(RMv * RMC[None, None, :], axis=-1)
    c = np.dot(RMC, RMC) - 1
    k_a = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    k_b = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    k = np.minimum(k_a, k_b)
    mid_k = ((k_a + k_b) / 2).astype(float)
    mid_k = np.nan_to_num(mid_k, nan=-1000)
    for i in range(5):
        mid_k[mid_k == -1000] = np.roll(mid_k, 1, axis=1)[mid_k == -1000]
        mid_k[mid_k == -1000] = np.roll(mid_k, -1, axis=1)[mid_k == -1000]
        mid_k[mid_k == -1000] = np.roll(mid_k, 1, axis=0)[mid_k == -1000]
        mid_k[mid_k == -1000] = np.roll(mid_k, -1, axis=0)[mid_k == -1000]
    # plt.imshow(mid_k)
    # plt.show()
    k[k < 0] = np.nan
    RMkvC = np.matmul(
        np.matmul(np.linalg.inv(R), np.linalg.inv(M))[None, None, :, :],
        (k[:, :, None] * v - C[None, None, :])[:, :, :, None],
    )[:, :, :, 0]
    reconstructed_phi = np.arcsin(RMkvC[:, :, 2])
    reconstructed_theta = np.arctan2(RMkvC[:, :, 1], RMkvC[:, :, 0])

    phi_idx = np.nan_to_num(
        (reconstructed_phi / (math.pi) + 0.5) * equi_arr.shape[1], nan=-1
    )
    theta_idx = np.nan_to_num(
        (reconstructed_theta / (2 * math.pi) + 0.5) * equi_arr.shape[0], nan=-1
    )
    xy_idx_outside = np.logical_or(
        np.logical_or(
            phi_idx >= equi_arr.shape[1],
            theta_idx >= equi_arr.shape[0],
        ),
        np.logical_or(phi_idx < 0, theta_idx < 0),
    )
    phi_idx = np.minimum(equi_arr.shape[1] - 1, np.maximum(0, phi_idx)).astype(int)
    theta_idx = np.minimum(equi_arr.shape[0] - 1, np.maximum(0, theta_idx)).astype(int)
    invalid = np.ones_like(phi_idx)
    invalid *= np.logical_not(xy_idx_outside).astype(int)
    painted_strip = equi_arr[np.ndarray.flatten(theta_idx), np.ndarray.flatten(phi_idx)]
    painted_surface = np.reshape(painted_strip, phi_idx.shape).astype(float)
    painted_surface[invalid == 0] = np.nan
    return painted_surface


if __name__ == "__main__":
    plt.subplots(1, 2)

    r_abc = [1, 1, 1]
    R = np.array([[r_abc[2], 0, 0], [0, r_abc[1], 0], [0, 0, r_abc[0]]])

    i = 3
    j = 4
    rot_theta = i * math.pi / 9
    ellipsoid_rot = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rot_theta), np.sin(rot_theta)],
            [0, -np.sin(rot_theta), np.cos(rot_theta)],
        ]
    )
    rot_phi = j * math.pi / 9
    ellipsoid_rot_b = np.array(
        [
            [np.cos(rot_phi), np.sin(rot_phi), 0],
            [-np.sin(rot_phi), np.cos(rot_phi), 0],
            [0, 0, 1],
        ]
    )
    up_vec = np.array([1, 0, 0])
    up_vec = up_vec / np.linalg.norm(up_vec)
    right_vec = np.cross(up_vec, np.array([0, 0, 1]))
    right_vec = right_vec / np.linalg.norm(right_vec)
    fore_vec = np.cross(up_vec, right_vec)
    M = np.vstack([up_vec, right_vec, fore_vec])
    M = np.matmul(ellipsoid_rot, np.matmul(ellipsoid_rot_b, M))

    C = np.array([2, 0, 4])

    painting = equi_to_ellipsoid(
        np.remainder(np.arange(20)[None, :] + np.arange(10)[:, None], 2), R, M, C
    )

    plt.subplot(1, 2, 1).imshow(painting)

    plt.subplot(1, 2, 2).imshow(
        equi_to_ellipsoid(ellipsoid_to_equi(painting, R, M, C), R, M, C)
    )
    plt.show()
