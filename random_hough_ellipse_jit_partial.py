import random
import time

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def cand_average_weight(old, now, score):
    return (old * score + now) / (score + 1)


@njit
def cand_is_similar(
    self_p,
    self_q,
    self_a,
    self_b,
    self_angle,
    candidate_p,
    candidate_q,
    candidate_a,
    candidate_b,
    candidate_angle,
    major_axis_limit,
    minor_axis_limit,
    center_limit,
    angle_limit,
):
    major_axis_dist = abs(max(candidate_a, candidate_b) - max(self_a, self_b))
    if major_axis_dist > major_axis_limit:
        return False

    minor_axis_dist = abs(min(candidate_a, candidate_b) - min(self_a, self_b))
    if minor_axis_dist > minor_axis_limit:
        return False

    angle_dist = abs(self_angle - candidate_angle)
    angle_180 = candidate_angle + (np.pi if candidate_angle < 0 else -np.pi)
    angle_dist_180 = abs(self_angle - angle_180)
    angle_final = min(angle_dist, angle_dist_180)
    if angle_final > angle_limit:
        return False

    center_dist_squared = (self_p - candidate_p) ** 2 + (self_q - candidate_q) ** 2
    if center_dist_squared > center_limit**2:
        return False

    return True


@njit
def ed_find_center(pt, edge, line_fitting_area):
    m, c = 0, 0
    m_arr = []
    c_arr = []

    # pt[0] is p1; pt[1] is p2; pt[2] is p3;
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

    return center


@njit
def ed_calculate_rotation_angle(a, b, c):
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
def ed_find_semi_axis(pt, center, major_axis_bound, minor_axis_bound, max_flattening):
    # shift to origin
    npt = [(p[0] - center[0], p[1] - center[1]) for p in pt]

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
        angle = ed_calculate_rotation_angle(u, v, w)
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
        return -1, -1, -1

    if a > 0 and b > 0:
        major = 1 / np.sqrt(min(a, b))
        minor = 1 / np.sqrt(max(a, b))

        if (major_axis_bound[0] < 2 * major < major_axis_bound[1]) and (
            minor_axis_bound[0] < 2 * minor < minor_axis_bound[1]
        ):
            flattening = (major - minor) / major
            if flattening < max_flattening:
                return major, minor, angle

    return -1, -1, -1


def compile_jit_functions():
    start_time = time.time()
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
    cand_average_weight(1.0, 2.0, 3.0)
    cand_is_similar(
        50.0,
        50.0,
        100.0,
        100.0,
        0.0,
        50.0,
        50.0,
        100.0,
        100.0,
        0.0,
        10.0,
        10.0,
        5.0,
        0.2,
    )
    ed_find_center(((1, 4), (5, 1), (7, 7)), ellipse_arr, 2)
    ed_find_semi_axis(
        ((5, 20), (25, 5), (25, 35)),
        np.array([13.5, 10.7]),
        (5, 100),
        (5, 100),
        0.8,
    )
    print("Compile time:", time.time() - start_time)


class EllipseDetectorInfo:
    MaxIter = 1000
    LineFittingArea = 7
    MajorAxisBound: tuple[int, int] = (60, 250)
    MinorAxisBound: tuple[int, int] = (60, 250)
    MaxFlattening = 0.8
    SimilarCenterDist = 5
    SimilarMajorAxisDist = 10
    SimilarMinorAxisDist = 10
    SimilarAngleDist = np.pi / 18
    MinEvalScorePerIter = 0


class Candidate:
    def __init__(
        self,
        p: int,
        q: int,
        a: float,
        b: float,
        angle: float,
        info: EllipseDetectorInfo,
    ):
        self.p = p
        self.q = q
        self.a = a
        self.b = b
        self.angle = angle
        self.score = 1
        self.info = info

    def average(self, candidate: "Candidate"):
        self.score += 1
        self.p = self.__average_weight(self.p, candidate.p, self.score)
        self.q = self.__average_weight(self.q, candidate.q, self.score)
        self.a = self.__average_weight(self.a, candidate.a, self.score)
        self.b = self.__average_weight(self.b, candidate.b, self.score)
        self.angle = self.__average_weight(self.angle, candidate.angle, self.score)

    def is_similar(self, candidate: "Candidate"):
        return cand_is_similar(
            self.p,
            self.q,
            self.a,
            self.b,
            self.angle,
            candidate.p,
            candidate.q,
            candidate.a,
            candidate.b,
            candidate.angle,
            self.info.SimilarMajorAxisDist,
            self.info.SimilarMinorAxisDist,
            self.info.SimilarCenterDist,
            self.info.SimilarAngleDist,
        )

    @staticmethod
    def __average_weight(old, now, score) -> float:
        return cand_average_weight(old, now, score)

    def __str__(self):
        return "{:6.2f}, {:6.2f}, {:6.2f}, {:6.2f}, {:6.2f}, {:6.2f}".format(
            self.p, self.q, self.a, self.b, self.angle, self.score
        )


class Accumulator:
    def __init__(self, info: EllipseDetectorInfo):
        self.accumulator: list[Candidate] = []
        self.info = info

    def sort_candidates(self):
        self.accumulator = sorted(
            self.accumulator, key=lambda candidate: candidate.score, reverse=True
        )

    def get_best_candidates(self):
        self.sort_candidates()
        return self.accumulator[:3]

    def clean_candidates(self, curr_iter: int):
        if curr_iter >= 1000 and curr_iter % 1000 == 0:
            self.sort_candidates()
            self.accumulator = self.accumulator[:1000]

    def evaluate_candidate(self, new_candidate: Candidate):
        index = self.__get_similar_index(new_candidate)
        if index == -1:
            self.__add(new_candidate)
        else:
            self.__merge(index, new_candidate)

    def __add(self, candidate: Candidate):
        self.accumulator.append(candidate)

    def __merge(self, index: int, candidate: Candidate):
        self.accumulator[index].average(candidate)

    def __get_similar_index(self, new_candidate: Candidate):
        similar_idx = -1
        for idx, candidate in enumerate(self.accumulator):
            if candidate.is_similar(new_candidate):
                return idx
        return similar_idx

    def __str__(self):
        text = ""
        print(
            "{: <6}, {: <6}, {: <6}, {: <6}, {: <6}, {: <6}".format(
                "p", "q", "a", "b", "angle", "score"
            )
        )
        for idx, candidate in enumerate(self.accumulator):
            if idx >= 3:
                break
            if candidate.score > 1:
                text += str(candidate)
                text += "\n"
        return text


class EllipseDetector:
    def __init__(
        self,
        info: EllipseDetectorInfo,
        edge: NDArray,
        edge_for_pick: NDArray | None = None,
    ):
        self.edge = edge.copy()
        if edge_for_pick is None:
            edge_for_pick = self.edge
        self.edge_for_pick = [p for p in np.array(np.where(edge_for_pick == 255)).T]

        self.info = info

        self.accumulator = Accumulator(info)

        self.timings = {}

    def run(self):
        start_time = time.time()

        random.seed(0)

        self.__find_candidate()

        find_candidate_time = time.time()
        self.timings["find_candidate"] = find_candidate_time - start_time

        best_candidate = self.accumulator.get_best_candidates()
        # print(self.accumulator)

        best_candidate_time = time.time()

        self.timings["best_candidate"] = best_candidate_time - find_candidate_time
        self.timings["total"] = best_candidate_time - start_time

        return best_candidate

    def __find_candidate(self):
        total_pick_time = 0
        total_center_time = 0
        total_semi_axis_time = 0
        total_evaluate_time = 0

        for i in range(self.info.MaxIter):
            self.accumulator.clean_candidates(i)

            start_time = time.time()

            # randomly pick 3 points
            point_package = self.__randomly_pick_point()

            pick_time = time.time()
            total_pick_time += pick_time - start_time

            # find center
            try:
                center = self.__find_center(point_package)
            except np.linalg.LinAlgError:  # Singular matrix
                continue
            except RHTException:
                continue
            except ZeroDivisionError:
                continue

            center_time = time.time()
            total_center_time += center_time - pick_time

            # find axis
            try:
                semi_major, semi_minor, angle = self.__find_semi_axis(
                    point_package, center
                )
            except np.linalg.LinAlgError:  # Singular matrix
                continue
            except RHTException:
                continue

            semi_axis_time = time.time()
            total_semi_axis_time += semi_axis_time - center_time

            center = (int(round(center[0])), int(round(center[1])))
            candidate = Candidate(
                center[0], center[1], semi_major, semi_minor, angle, self.info
            )
            self.accumulator.evaluate_candidate(candidate)

            evaluate_time = time.time()
            total_evaluate_time += evaluate_time - semi_axis_time

        self.timings["find_candidate_pick"] = total_pick_time
        self.timings["find_candidate_center"] = total_center_time
        self.timings["find_candidate_semi_axis"] = total_semi_axis_time
        self.timings["find_candidate_evaluate"] = total_evaluate_time

    def __randomly_pick_point(self) -> list[tuple[int, int]]:
        ran = random.sample(self.edge_for_pick, 3)
        return [(ran[0][1], ran[0][0]), (ran[1][1], ran[1][0]), (ran[2][1], ran[2][0])]

    def __find_center(self, pt: list[tuple[int, int]]) -> NDArray:
        center = ed_find_center(tuple(pt), self.edge, self.info.LineFittingArea)

        self.__assert_point_in_image(center)

        return center

    def __find_semi_axis(
        self, pt: list[tuple[int, int]], center: NDArray
    ) -> tuple[float, float, float]:
        major, minor, angle = ed_find_semi_axis(
            tuple(pt),
            center,
            self.info.MajorAxisBound,
            self.info.MinorAxisBound,
            self.info.MaxFlattening,
        )
        if major == -1:
            raise RHTException
        return major, minor, angle

    def __assert_point_in_image(self, point: NDArray):
        if (
            point[0] < 0
            or point[0] >= self.edge.shape[1]
            or point[1] < 0
            or point[1] >= self.edge.shape[0]
        ):
            raise RHTException("center is out of image!")


class RHTException(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        return self.message if self.message else "some thing wrong!"
