import random
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class EllipseDetectorInfo:
    MaxIter: int = 1000
    LineFittingArea: int = 7
    MajorAxisBound: tuple[int, int] = (60, 250)
    MinorAxisBound: tuple[int, int] = (60, 250)
    MaxFlattening: float = 0.8
    SimilarCenterDist: int = 5
    SimilarMajorAxisDist: int = 10
    SimilarMinorAxisDist: int = 10
    SimilarAngleDist: float = np.pi / 18


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
        center_dist = np.sqrt((self.p - candidate.p) ** 2 + (self.q - candidate.q) ** 2)
        angle_dist = abs(self.angle - candidate.angle)
        if candidate.angle >= 0:
            angle180 = candidate.angle - np.pi
        else:
            angle180 = candidate.angle + np.pi
        angle_dist180 = abs(self.angle - angle180)
        angle_final = min(angle_dist, angle_dist180)

        # axis dist
        major_axis_dist = abs(max(candidate.a, candidate.b) - max(self.a, self.b))
        minor_axis_dist = abs(min(candidate.a, candidate.b) - min(self.a, self.b))
        if (
            (major_axis_dist < self.info.SimilarMajorAxisDist)
            and (minor_axis_dist < self.info.SimilarMinorAxisDist)
            and (center_dist < self.info.SimilarCenterDist)
            and (angle_final < self.info.SimilarAngleDist)
        ):
            return True
        else:
            return False

    @staticmethod
    def __average_weight(old, now, score) -> float:
        return (old * score + now) / (score + 1)

    def __str__(self):
        return "{:6.2f}, {:6.2f}, {:6.2f}, {:6.2f}, {:6.2f}, {:6.2f}".format(
            self.p, self.q, self.a, self.b, self.angle, self.score
        )


class Accumulator:
    def __init__(self):
        self.accumulator = []

    def get_best_candidates(self):
        self.accumulator = sorted(
            self.accumulator, key=lambda candidate: candidate.score, reverse=True
        )
        return self.accumulator[:3]

    def evaluate_candidate(self, new_candidate: Candidate):
        index = self.__get_similar_index(new_candidate)
        if index == -1:
            self.__add(new_candidate)
        else:
            self.__merge(index, new_candidate)

    def __add(self, candidate: Candidate):
        self.accumulator.append(candidate)

    def __merge(self, index, candidate: Candidate):
        self.accumulator[index].average(candidate)

    def __get_similar_index(self, new_candidate: Candidate):
        similar_idx = -1
        if len(self.accumulator) > 0:
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
    ):
        self.edge = edge.copy()
        self.edge_for_pick = [p for p in np.array(np.where(self.edge == 255)).T]

        self.info = info

        self.accumulator = Accumulator()

        # self.timings = {} # TIMINGS

    def run(self):
        # start_time = time.perf_counter() # TIMINGS

        random.seed(0)

        self.__find_candidate()

        # find_candidate_time = time.perf_counter() # TIMINGS
        # self.timings["find_candidate"] = find_candidate_time - start_time # TIMINGS

        best_candidate = self.accumulator.get_best_candidates()
        # print(self.accumulator)

        # best_candidate_time = time.perf_counter() # TIMINGS

        # self.timings["best_candidate"] = best_candidate_time - find_candidate_time # TIMINGS
        # self.timings["total"] = best_candidate_time - start_time # TIMINGS

        return best_candidate

    def __find_candidate(self):
        # total_pick_time = 0 # TIMINGS
        # total_center_time = 0 # TIMINGS
        # total_semi_axis_time = 0 # TIMINGS
        # total_evaluate_time = 0 # TIMINGS

        # self.timings["find_candidate_center_straight"] = 0 # TIMINGS
        # self.timings["find_candidate_center_intersection"] = 0 # TIMINGS
        # self.timings["find_candidate_center_center"] = 0 # TIMINGS

        for _ in range(self.info.MaxIter):
            # start_time = time.perf_counter() # TIMINGS

            # randomly pick 3 points
            point_package = self.__randomly_pick_point()

            # pick_time = time.perf_counter() # TIMINGS
            # total_pick_time += pick_time - start_time # TIMINGS

            # find center
            try:
                center = self.__find_center(point_package)
            except np.linalg.LinAlgError:  # Singular matrix
                continue
            except RHTException:
                continue

            # center_time = time.perf_counter() # TIMINGS
            # total_center_time += center_time - pick_time # TIMINGS

            # find axis
            try:
                semi_major, semi_minor, angle = self.__find_semi_axis(
                    point_package, center
                )
            except np.linalg.LinAlgError:  # Singular matrix
                continue
            except RHTException:
                continue

            # semi_axis_time = time.perf_counter() # TIMINGS
            # total_semi_axis_time += semi_axis_time - center_time # TIMINGS

            center = (int(round(center[0])), int(round(center[1])))
            axis = (int(round(semi_major)), int(round(semi_minor)))

            if self.__ellipse_out_of_mask(center, axis, angle):
                continue

            candidate = Candidate(
                center[0], center[1], semi_major, semi_minor, angle, self.info
            )
            self.accumulator.evaluate_candidate(candidate)

            # evaluate_time = time.perf_counter() # TIMINGS
            # total_evaluate_time += evaluate_time - semi_axis_time # TIMINGS

        # self.timings["find_candidate_pick"] = total_pick_time # TIMINGS
        # self.timings["find_candidate_center"] = total_center_time # TIMINGS
        # self.timings["find_candidate_semi_axis"] = total_semi_axis_time # TIMINGS
        # self.timings["find_candidate_evaluate"] = total_evaluate_time # TIMINGS

    # def __canny_edge_detector(self):
    #     edged_image = cv2.Canny(
    #         self.image,
    #         self.info.CannyT1,
    #         self.info.CannyT2,
    #         None,
    #         self.info.CannyApperture,
    #     )
    #     self.edge = edged_image

    def __randomly_pick_point(self) -> list[tuple[int, int]]:
        ran = random.sample(self.edge_for_pick, 3)
        return [(ran[0][1], ran[0][0]), (ran[1][1], ran[1][0]), (ran[2][1], ran[2][0])]

    def __find_center(self, pt: list[tuple[int, int]]) -> NDArray:
        # start_time = time.perf_counter() # TIMINGS

        m, c = 0, 0
        m_arr = []
        c_arr = []

        # pt[0] is p1; pt[1] is p2; pt[2] is p3;
        for i in range(len(pt)):
            # find tangent line
            xstart = pt[i][0] - self.info.LineFittingArea // 2
            xend = pt[i][0] + self.info.LineFittingArea // 2 + 1
            ystart = pt[i][1] - self.info.LineFittingArea // 2
            yend = pt[i][1] + self.info.LineFittingArea // 2 + 1
            crop = self.edge[ystart:yend, xstart:xend].T
            proximal_point = np.array(np.where(crop == 255)).T
            proximal_point[:, 0] += xstart
            proximal_point[:, 1] += ystart

            # fit straight line
            A = np.vstack([proximal_point[:, 0], np.ones(len(proximal_point[:, 0]))]).T
            m, c = np.linalg.lstsq(A, proximal_point[:, 1], rcond=None)[0]
            m_arr.append(m)
            c_arr.append(c)

        # straight_time = time.perf_counter() # TIMINGS
        # self.timings["find_candidate_center_straight"] += straight_time - start_time # TIMINGS

        # find intersection
        slope_arr = []
        intercept_arr = []
        for i, j in zip([0, 1], [1, 2]):
            # intersection
            coef_matrix = np.array([[m_arr[i], -1], [m_arr[j], -1]])
            dependent_variable = np.array([-c_arr[i], -c_arr[j]])
            t12 = np.linalg.solve(coef_matrix, dependent_variable)
            # middle point
            m1 = ((pt[i][0] + pt[j][0]) / 2, (pt[i][1] + pt[j][1]) / 2)
            # bisector
            slope = (m1[1] - t12[1]) / (m1[0] - t12[0])
            intercept = (m1[0] * t12[1] - t12[0] * m1[1]) / (m1[0] - t12[0])

            slope_arr.append(slope)
            intercept_arr.append(intercept)

        # intersection_time = time.perf_counter() # TIMINGS
        # self.timings["find_candidate_center_intersection"] += (intersection_time - straight_time) # TIMINGS

        # find center
        coef_matrix = np.array([[slope_arr[0], -1], [slope_arr[1], -1]])
        dependent_variable = np.array([-intercept_arr[0], -intercept_arr[1]])
        center = np.linalg.solve(coef_matrix, dependent_variable)
        self.__assert_point_in_image(center)

        # center_time = time.perf_counter() # TIMINGS
        # self.timings["find_candidate_center_center"] += center_time - intersection_time # TIMINGS

        return center

    def __find_semi_axis(
        self, pt: list[tuple[int, int]], center: NDArray
    ) -> tuple[float, float, float]:
        # shift to origin
        npt = [(p[0] - center[0], p[1] - center[1]) for p in pt]

        # semi axis
        x1, y1, x2, y2, x3, y3 = np.array(npt).flatten()
        coef_matrix = np.array(
            [
                [x1**2, 2 * x1 * y1, y1**2],
                [x2**2, 2 * x2 * y2, y2**2],
                [x3**2, 2 * x3 * y3, y3**2],
            ]
        )
        dependent_variable = np.array([1, 1, 1])
        A, B, C = np.linalg.solve(coef_matrix, dependent_variable)

        if A * C - B**2 > 0:
            angle = self.__calculate_rotation_angle(A, B, C)
            axis_coef = np.array(
                [
                    [np.sin(angle) ** 2, np.cos(angle) ** 2],
                    [np.cos(angle) ** 2, np.sin(angle) ** 2],
                ]
            )
            axis_ans = np.array([A, C])
            a, b = np.linalg.solve(axis_coef, axis_ans)
        else:
            raise RHTException

        if a > 0 and b > 0:
            major = 1 / np.sqrt(min(a, b))
            minor = 1 / np.sqrt(max(a, b))
            if self.__assert_diameter(major, minor):
                return major, minor, angle
        raise RHTException

    def __assert_diameter(self, major: float, minor: float) -> bool:
        if (self.info.MajorAxisBound[0] < 2 * major < self.info.MajorAxisBound[1]) and (
            self.info.MinorAxisBound[0] < 2 * minor < self.info.MinorAxisBound[1]
        ):
            flattening = (major - minor) / major
            if flattening < self.info.MaxFlattening:
                return True
        return False

    def __assert_point_in_image(self, point: NDArray):
        if (
            point[0] < 0
            or point[0] >= self.edge.shape[1]
            or point[1] < 0
            or point[1] >= self.edge.shape[0]
        ):
            raise RHTException("center is out of image!")

    def __calculate_rotation_angle(self, a: NDArray, b: NDArray, c: NDArray) -> float:
        if a == c:
            angle = 0
        else:
            angle = 0.5 * np.arctan((2 * b) / (a - c))

        if a > c:
            if b < 0:
                angle += 0.5 * np.pi  # +90 deg
            elif b > 0:
                angle -= 0.5 * np.pi  # -90 deg
        return float(angle)

    def __ellipse_out_of_mask(
        self, center: tuple[int, int], axis: tuple[int, int], angle: float
    ):
        larger_shape = (self.edge.shape[0] + 2, self.edge.shape[1] + 2)
        ref_mask = np.ones(larger_shape, dtype=np.uint8)
        ref_mask[0, :] = ref_mask[-1, :] = ref_mask[:, 0] = ref_mask[:, -1] = 0

        ellipse, out_of_mask = np.zeros_like(ref_mask), np.zeros_like(ref_mask)
        cv2.ellipse(
            ellipse,
            center,
            axis,
            angle * 180 / np.pi,
            0,
            360,
            color=(255,),
            thickness=1,
        )

        out_of_mask[(ellipse == 255) & (ref_mask == 0)] = 1

        return np.sum(out_of_mask) > 0


class RHTException(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        return self.message if self.message else "some thing wrong!"
