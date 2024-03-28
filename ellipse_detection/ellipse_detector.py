from .ellipse_candidate_maker import EllipseCandidateMaker
from .ellipse_estimator import EllipseEstimator
from .ellipse_merger import EllipseMerger
from .segment_detector import SegmentDetector


class EllipseDetector(object):
    def __init__(
        self,
        canny_threshold_1=70,
        canny_threshold_2=90,
        gaussian_k_size=(5, 5),
        gaussian_sigma_x=3,
        straight_ratio_threshold=30,
        ellipse_center_reject_distance=50,
        num_bin_n_accumulator=100,
        num_bin_rho_accumulator=180,
        max_major_semi_axis_len=500,
        lying_threshold=0.1,
        identify_threshold=0.1,
        axis_ratio_threshold=0.9,
    ):
        self.CANNY_THRESHOLD_1 = canny_threshold_1
        self.CANNY_THRESHOLD_2 = canny_threshold_2
        self.GAUSSIAN_K_SIZE = gaussian_k_size
        self.GAUSSIAN_SIGMA_X = gaussian_sigma_x
        self.STRAIGHT_RATIO_THRESHOLD = straight_ratio_threshold
        self.ELLIPSE_CENTER_REJECT_DISTANCE = ellipse_center_reject_distance
        self.NUM_BIN_N_ACCUMULATOR = num_bin_n_accumulator
        self.NUM_BIN_RHO_ACCUMULATOR = num_bin_rho_accumulator
        self.MAX_MAJOR_SEMI_AXIS_LEN = max_major_semi_axis_len
        self.LYING_THRESHOLD = lying_threshold
        self.IDENTIFY_THRESHOLD = identify_threshold
        self.AXIS_RATIO_THRESHOLD = axis_ratio_threshold

    def detect(self, image, image_edge=None):
        """Detect ellipse from image.

        Args:
            image: A numpy as array indicats gray scale image.

        Returns:
            Array of Ellipse instance that was detected from image.
        """

        if len(image.shape) != 2:
            raise RuntimeError()

        seg_detector = SegmentDetector(
            self.CANNY_THRESHOLD_1,
            self.CANNY_THRESHOLD_2,
            self.GAUSSIAN_K_SIZE,
            self.GAUSSIAN_SIGMA_X,
            self.STRAIGHT_RATIO_THRESHOLD,
        )
        segments = seg_detector.detect(image, image_edge)

        ellipse_cand_maker = EllipseCandidateMaker(self.ELLIPSE_CENTER_REJECT_DISTANCE)
        ellipse_cands = ellipse_cand_maker.make(segments)

        ellipse_estimator = EllipseEstimator(
            self.NUM_BIN_N_ACCUMULATOR,
            self.NUM_BIN_RHO_ACCUMULATOR,
            self.MAX_MAJOR_SEMI_AXIS_LEN,
            self.LYING_THRESHOLD,
        )
        ellipses = ellipse_estimator.estimate(ellipse_cands)

        ellipse_merger = EllipseMerger(
            image.shape[1],
            image.shape[0],
            self.IDENTIFY_THRESHOLD,
            self.AXIS_RATIO_THRESHOLD,
        )
        ellipses = ellipse_merger.merge(ellipses)

        return ellipses
