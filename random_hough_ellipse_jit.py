import time

import numpy as np
from numba import njit

# from numba.typed import List
from numpy.typing import NDArray

Candidate = tuple[int, float, float, float, float, float]

# CannyArgs = tuple[float, float, NDArray | None, int, bool]


class RHTException(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        return self.message if self.message else "something wrong!"


@njit
def average_candidate_weight(old_cand: float, new_cand: float, score: int) -> float:
    return (old_cand * score + new_cand) / (score + 1)


@njit
def average_candidate(old_cand: Candidate, new_cand: Candidate) -> Candidate:
    score = old_cand[0] + 1
    return (
        score,
        average_candidate_weight(old_cand[1], new_cand[1], score),
        average_candidate_weight(old_cand[2], new_cand[2], score),
        average_candidate_weight(old_cand[3], new_cand[3], score),
        average_candidate_weight(old_cand[4], new_cand[4], score),
        average_candidate_weight(old_cand[5], new_cand[5], score),
    )


@njit
def is_candidate_similar(
    old_cand: Candidate,
    new_cand: Candidate,
    similar_center_dist: float,
    similar_major_axis_dist: float,
    similar_minor_axis_dist: float,
    similar_angle_dist: float,
) -> bool:
    if abs(old_cand[3] - new_cand[3]) > similar_major_axis_dist:
        return False

    if abs(old_cand[4] - new_cand[4]) > similar_minor_axis_dist:
        return False

    angle_dist = abs(old_cand[5] - new_cand[5])
    angle_b_180 = new_cand[5] + (np.pi if new_cand[5] < 0 else -np.pi)
    angle_dist_180 = abs(old_cand[5] - angle_b_180)
    angle_final = min(angle_dist, angle_dist_180)
    if angle_final > similar_angle_dist:
        return False

    center_dist_squared = (old_cand[1] - new_cand[1]) ** 2 + (
        old_cand[2] - new_cand[2]
    ) ** 2
    if center_dist_squared > similar_center_dist**2:
        return False

    return True


@njit
def find_center(pt: NDArray, edge: NDArray, line_fitting_area: int) -> NDArray:
    m, c = 0, 0
    m_arr = []
    c_arr = []

    for i in range(len(pt)):
        # find tangent line
        xstart = pt[i][0] - line_fitting_area // 2
        xend = pt[i][0] + line_fitting_area // 2 + 1
        ystart = pt[i][1] - line_fitting_area // 2
        yend = pt[i][1] + line_fitting_area // 2 + 1
        crop = edge[ystart:yend, xstart:xend].T
        proximal_point = np.vstack(np.where(crop == 255)).T.astype(np.float32)
        proximal_point[:, 0] += xstart
        proximal_point[:, 1] += ystart

        # fit straight line
        coef_matrix = np.vstack(
            (proximal_point[:, 0], np.ones(len(proximal_point[:, 0]), dtype=np.float32))
        ).T
        dependent_variable = proximal_point[:, 1]
        m, c = np.linalg.lstsq(coef_matrix, dependent_variable, rcond=-1)[0]
        m_arr.append(m)
        c_arr.append(c)

    # find intersection
    slope_arr = []
    intercept_arr = []
    for i, j in zip([0, 1], [1, 2]):
        # intersection
        coef_matrix = np.array([[m_arr[i], -1], [m_arr[j], -1]], dtype=np.float32)
        dependent_variable = np.array([-c_arr[i], -c_arr[j]], dtype=np.float32)
        t12 = np.linalg.solve(coef_matrix, dependent_variable)
        # middle point
        m1 = ((pt[i][0] + pt[j][0]) / 2, (pt[i][1] + pt[j][1]) / 2)
        # bisector
        slope = (m1[1] - t12[1]) / (m1[0] - t12[0])
        intercept = (m1[0] * t12[1] - t12[0] * m1[1]) / (m1[0] - t12[0])

        slope_arr.append(slope)
        intercept_arr.append(intercept)

    # find center
    coef_matrix = np.array([[slope_arr[0], -1], [slope_arr[1], -1]], dtype=np.float32)
    dependent_variable = np.array(
        [-intercept_arr[0], -intercept_arr[1]], dtype=np.float32
    )
    center = np.linalg.solve(coef_matrix, dependent_variable)

    if (
        center[0] < 0
        or center[0] >= edge.shape[1]
        or center[1] < 0
        or center[1] >= edge.shape[0]
    ):
        raise RHTException("center is out of image!")

    return center


@njit
def calculate_rotation_angle(a: float, b: float, c: float) -> float:
    if a == c:
        angle = 0
    else:
        angle = 0.5 * np.arctan((2 * b) / (a - c))

    if a > c:
        if b < 0:
            angle += 0.5 * np.pi
        elif b > 0:
            angle -= 0.5 * np.pi
    return float(angle)


@njit
def find_semi_axis(
    pt: NDArray,
    center: NDArray,
    major_axis_bound: tuple[int, int],
    minor_axis_bound: tuple[int, int],
    max_flattening: float,
) -> tuple[float, float, float]:
    # shift to origin
    npt = pt - center

    # semi axis
    coef_matrix = np.array(
        [
            [npt[0][0] ** 2, 2 * npt[0][0] * npt[0][1], npt[0][1] ** 2],
            [npt[1][0] ** 2, 2 * npt[1][0] * npt[1][1], npt[1][1] ** 2],
            [npt[2][0] ** 2, 2 * npt[2][0] * npt[2][1], npt[2][1] ** 2],
        ],
        dtype=np.float32,
    )
    dependent_variable = np.array([1, 1, 1], dtype=np.float32)
    u, v, w = np.linalg.solve(coef_matrix, dependent_variable)

    if u * w - v**2 > 0:
        angle = calculate_rotation_angle(u, v, w)
        axis_coef = np.array(
            [
                [np.sin(angle) ** 2, np.cos(angle) ** 2],
                [np.cos(angle) ** 2, np.sin(angle) ** 2],
            ],
            dtype=np.float32,
        )
        axis_ans = np.array([u, w], dtype=np.float32)
        a, b = np.linalg.solve(axis_coef, axis_ans)
    else:
        raise RHTException("no valid semi axis!")

    if a > 0 and b > 0:
        major = 1 / np.sqrt(min(a, b))
        minor = 1 / np.sqrt(max(a, b))

        if (major_axis_bound[0] < 2 * major < major_axis_bound[1]) and (
            minor_axis_bound[0] < 2 * minor < minor_axis_bound[1]
        ):
            flattening = (major - minor) / major
            if flattening < max_flattening:
                return major, minor, angle

    raise RHTException("no valid semi axis!")


@njit
def ellipse_out_of_mask(
    mask_shape: tuple[int, int],
    center: NDArray,
    axis: tuple[float, float],
    angle: float,
    num_points: int = 50,
) -> bool:
    thetas = np.linspace(0, 2 * np.pi, num_points) - angle
    tan_thetas = np.tan(thetas)

    x_values = (
        np.where((thetas > np.pi / 2) & (thetas < 3 * np.pi / 2), -1, 1)
        * axis[0]
        * axis[1]
        / (np.sqrt(axis[1] ** 2 + (axis[0] ** 2) * np.square(tan_thetas)))
    )

    rotation_matrix = np.array(
        [[-np.sin(angle), np.cos(angle)], [np.cos(angle), np.sin(angle)]]
    )

    coordinates = np.dot(
        rotation_matrix, np.vstack((x_values * tan_thetas, x_values))
    ) + center.reshape(-1, 1)

    return np.any(
        (coordinates < -1)
        | (coordinates > np.array((mask_shape[1], mask_shape[0])).reshape(-1, 1))
    )  # type: ignore


@njit
def random_hough_ellipse(
    edge_img: NDArray,
    max_iter: int,
    edge_img_for_pick: NDArray | None = None,
    line_fitting_area: int = 5,
    similar_center_dist: float = 5.0,
    similar_major_axis_dist: float = 5.0,
    similar_minor_axis_dist: float = 5.0,
    similar_angle_dist: float = 0.2,
    major_axis_bound: tuple[int, int] = (10, 1000),
    minor_axis_bound: tuple[int, int] = (10, 1000),
    max_flattening: float = 0.8,
    candidates_update_interval: int = 1000,
    max_candidates: int = 1000,
    returned_candidates: int = 10,
    target_score: int = 100,
) -> list[Candidate]:
    edge = edge_img.copy()
    if edge_img_for_pick is not None:
        edge_for_pick = edge_img_for_pick.copy()
    else:
        edge_for_pick = edge_img.copy()

    edge_for_pick = np.vstack(np.where(edge_for_pick.T == 255)).T.astype(np.int32)

    candidates: list[Candidate] = [(0, 0.0, 0.0, 0.0, 0.0, 0.0)]

    np.random.seed(0)

    for i in range(max_iter):
        if i >= candidates_update_interval and i % candidates_update_interval == 0:
            candidates = sorted(candidates, reverse=True)[:max_candidates]
            if candidates[0][0] >= target_score:
                return candidates[:returned_candidates]

        points = edge_for_pick[np.random.choice(len(edge_for_pick), 3, replace=False)]

        try:
            center = find_center(points, edge, line_fitting_area)
            major, minor, angle = find_semi_axis(
                points, center, major_axis_bound, minor_axis_bound, max_flattening
            )
        except Exception:
            continue

        if ellipse_out_of_mask(
            (edge.shape[0], edge.shape[1]), center, (major, minor), angle
        ):
            continue

        new_candidate = (1, center[0], center[1], major, minor, angle)
        similar_idx = -1
        for j, cand in enumerate(candidates):
            if is_candidate_similar(
                cand,
                new_candidate,
                similar_center_dist,
                similar_major_axis_dist,
                similar_minor_axis_dist,
                similar_angle_dist,
            ):
                similar_idx = j
                break

        if similar_idx == -1:
            candidates.append(new_candidate)
        else:
            candidates[similar_idx] = average_candidate(
                candidates[similar_idx], new_candidate
            )

    return sorted(candidates, reverse=True)[:returned_candidates]


def compile_jit_functions():
    start_time = time.perf_counter()
    ellipse_arr = (
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        * 255
    )
    average_candidate(
        (1, 50.0, 50.0, 100.0, 100.0, 0.0),
        (1, 50.0, 50.0, 100.0, 100.0, 0.0),
    )
    is_candidate_similar(
        (
            1,
            50.0,
            50.0,
            100.0,
            100.0,
            0.0,
        ),
        (
            1,
            50.0,
            50.0,
            100.0,
            100.0,
            0.0,
        ),
        10.0,
        10.0,
        5.0,
        0.2,
    )
    find_center(np.array([[1, 4], [5, 1], [7, 7]]), ellipse_arr, 2)
    find_semi_axis(
        np.array([[5, 20], [25, 5], [25, 35]]),
        np.array([13.5, 10.7]),
        (5, 100),
        (5, 100),
        0.8,
    )
    ellipse_out_of_mask((100, 100), np.array((50, 50)), (60.0, 20.0), np.pi / 4)
    random_hough_ellipse(ellipse_arr, 1, None, 2)
    print("Compile time:", time.perf_counter() - start_time)
