"""Unit tests for the jicbioimage.segment package."""

import unittest

import numpy as np


class GenericUnitTests(unittest.TestCase):

    def test_version_is_string(self):
        import jicbioimage.segment
        self.assertTrue(isinstance(jicbioimage.segment.__version__, str))


class ConnectedComponentsTests(unittest.TestCase):

    def setUp(self):
        from jicbioimage.core.io import AutoWrite
        AutoWrite.on = False

    def test_connected_components(self):
        from jicbioimage.segment import connected_components
        from jicbioimage.core.image import SegmentedImage
        ar = np.array([[1, 1, 0, 0, 0],
                       [1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 2, 2, 2],
                       [0, 0, 2, 2, 2]], dtype=np.uint8)
        segmentation = connected_components(ar)
        self.assertTrue(isinstance(segmentation, SegmentedImage))
        self.assertEqual(segmentation.identifiers, set([1, 2, 3]))

    def test_connected_components_background_option(self):
        from jicbioimage.segment import connected_components
        from jicbioimage.core.image import SegmentedImage
        ar = np.array([[1, 1, 0, 0, 0],
                       [1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 2, 2, 2],
                       [0, 0, 2, 2, 2]], dtype=np.uint8)
        segmentation = connected_components(ar, background=1)
        self.assertTrue(isinstance(segmentation, SegmentedImage))
        self.assertEqual(segmentation.identifiers, set([1, 2]))

    def test_connected_components_connectivity_option(self):
        from jicbioimage.segment import connected_components
        from jicbioimage.core.image import SegmentedImage
        ar = np.array([[1, 1, 0, 0, 0],
                       [1, 1, 0, 0, 0],
                       [0, 0, 1, 1, 1],
                       [0, 0, 1, 1, 1],
                       [0, 0, 1, 1, 1]], dtype=np.uint8)

        segmentation = connected_components(ar, connectivity=1)
        self.assertTrue(isinstance(segmentation, SegmentedImage))
        self.assertEqual(segmentation.identifiers, set([1, 2, 3, 4]))

        segmentation = connected_components(ar, connectivity=2)
        self.assertTrue(isinstance(segmentation, SegmentedImage))
        self.assertEqual(segmentation.identifiers, set([1, 2]))

    def test_connected_components_acts_like_a_transform(self):
        from jicbioimage.segment import connected_components
        from jicbioimage.core.image import Image
        ar = np.array([[1, 1, 0, 0, 0],
                       [1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 2, 2, 2],
                       [0, 0, 2, 2, 2]], dtype=np.uint8)
        im = Image.from_array(ar)
        self.assertEqual(len(im.history), 1)
        segmentation = connected_components(im)
        self.assertEqual(len(segmentation.history), 2)
        self.assertEqual(segmentation.history[-1],
                         "Applied connected_components transform")


class WatershedWithSeedsTests(unittest.TestCase):

    def setUp(self):
        from jicbioimage.core.io import AutoWrite
        AutoWrite.on = False

    def test_watershed_with_seeds(self):
        from jicbioimage.segment import watershed_with_seeds
        from jicbioimage.core.image import SegmentedImage

        ar = np.array([[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 9, 0, 0],
                       [9, 9, 9, 9, 9, 9],
                       [0, 0, 0, 9, 0, 0],
                       [0, 0, 0, 9, 0, 0]], dtype=np.uint8)

        sd = np.array([[1, 0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [3, 0, 0, 0, 0, 4]], dtype=np.uint8)

        segmentation = watershed_with_seeds(image=ar, seeds=sd)
        self.assertTrue(isinstance(segmentation, SegmentedImage))
        self.assertEqual(segmentation.identifiers, set([1, 2, 3, 4]))

    def test_watershed_with_seeds_mask_option(self):
        from jicbioimage.segment import watershed_with_seeds
        ar = np.array([[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 9, 0, 0],
                       [9, 9, 9, 9, 9, 9],
                       [0, 0, 0, 9, 0, 0],
                       [0, 0, 0, 9, 0, 0]], dtype=np.uint8)

        sd = np.array([[1, 0, 0, 0, 0, 2],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [3, 0, 0, 0, 0, 4]], dtype=np.uint8)

        ma = np.array([[1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 0, 0, 0]], dtype=bool)

        segmentation = watershed_with_seeds(image=ar, seeds=sd, mask=ma)
        self.assertEqual(segmentation.identifiers, set([1, 2, 3]))
        mask_size = len(segmentation[np.where(segmentation == 0)])
        self.assertEqual(mask_size, 6)
