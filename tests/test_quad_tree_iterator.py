import pytest

from util.math_utils import Interval2D
from util.algorithms import ImageSegmentationIterator


def test_iterator_2x2():
    iterator = ImageSegmentationIterator(interval=Interval2D(min_x=0, max_x=1,
                                                             min_y=0, max_y=1),
                                         is_uniform=lambda x: False)

    bag_of_regions = set()

    for region in iterator:
        bag_of_regions.add(region)

    expected_regions = {
        Interval2D(0, 0, 0, 0),
        Interval2D(0, 0, 1, 1),
        Interval2D(1, 1, 0, 0),
        Interval2D(1, 1, 1, 1)
    }

    assert bag_of_regions == expected_regions


def test_iterator_3x3():
    iterator = ImageSegmentationIterator(interval=Interval2D(min_x=0, max_x=2,
                                                             min_y=0, max_y=2),
                                         is_uniform=lambda x: False)

    bag_of_regions = set()

    for region in iterator:
        bag_of_regions.add(region)

    expected_regions = {
        Interval2D(0, 0, 0, 0),
        Interval2D(1, 1, 0, 0),
        Interval2D(2, 2, 0, 0),

        Interval2D(0, 0, 1, 1),
        Interval2D(1, 1, 1, 1),
        Interval2D(2, 2, 1, 1),

        Interval2D(0, 0, 2, 2),
        Interval2D(1, 1, 2, 2),
        Interval2D(2, 2, 2, 2),
    }

    assert bag_of_regions == expected_regions
