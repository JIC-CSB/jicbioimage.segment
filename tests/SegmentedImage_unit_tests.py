"""Tests for the :class:`jicbioimage.segment.SegmentedImage` class."""

import unittest

import io
import numpy as np
import PIL


class SegmentedImageTests(unittest.TestCase):

    def test_identifiers(self):

        from jicbioimage.segment import SegmentedImage

        input_array = np.array([[0, 0, 0],
                                [1, 1, 1],
                                [2, 2, 2]])

        segmented_image = SegmentedImage.from_array(input_array)

        self.assertEqual(segmented_image.identifiers, set([1, 2]))

    def test_number_of_segments(self):

        from jicbioimage.segment import SegmentedImage

        input_array = np.array([[0, 0, 0],
                                [1, 1, 1],
                                [2, 2, 2]])

        segmented_image = SegmentedImage.from_array(input_array)

        self.assertEqual(segmented_image.number_of_segments, 2)

    def test_region_by_identifier(self):

        from jicbioimage.segment import SegmentedImage

        input_array = np.array([[0, 0, 0],
                                [1, 1, 1],
                                [2, 2, 2]])

        segmented_image = SegmentedImage.from_array(input_array)

        with self.assertRaises(ValueError):
            segmented_image.region_by_identifier(0)

        with self.assertRaises(ValueError):
            segmented_image.region_by_identifier(0.5)

        with self.assertRaises(ValueError):
            segmented_image.region_by_identifier(-1)

        from jicbioimage.segment import Region

        selected_region = segmented_image.region_by_identifier(1)

        self.assertTrue(isinstance(selected_region, Region))

        expected_output = Region.select_from_array(input_array, 1)
        self.assertTrue(np.array_equal(selected_region,
                                       expected_output))

    def test_background(self):

        from jicbioimage.segment import SegmentedImage

        input_array = np.array([[0, 0, 0],
                                [1, 1, 1],
                                [2, 2, 2]])

        segmented_image = SegmentedImage.from_array(input_array)

        from jicbioimage.segment import Region

        background = segmented_image.background

        expected_output = Region.select_from_array(input_array, 0)
        self.assertTrue(np.array_equal(background,
                                       expected_output))

    def test_false_colour_image(self):

        from jicbioimage.segment import SegmentedImage

        input_array = np.array([[0, 0, 0],
                                [1, 1, 1],
                                [2, 2, 2]])

        segmented_image = SegmentedImage.from_array(input_array)

        false_color_image = segmented_image.false_color_image

        from jicbioimage.core.util.array import false_color

        self.assertTrue(np.array_equal(false_color_image,
                                       false_color(input_array)))

    def test_grayscale_image(self):

        from jicbioimage.segment import SegmentedImage

        input_array = np.array([[0, 0, 0],
                                [1, 1, 1],
                                [2, 2, 2]])

        segmented_image = SegmentedImage.from_array(input_array)

        grayscale_image = segmented_image.grayscale_image
        self.assertTrue(np.array_equal(grayscale_image, input_array))

    def test_png(self):

        from jicbioimage.segment import SegmentedImage

        input_array = np.array([[0, 0, 0],
                                [1, 1, 1],
                                [2, 2, 2]])

        segmented_image = SegmentedImage.from_array(input_array)

        png = segmented_image.png()
        ar = np.asarray(PIL.Image.open(io.BytesIO(png)))

        self.assertTrue(np.array_equal(ar, segmented_image.false_color_image))

    def test_remove_region(self):

        from jicbioimage.segment import SegmentedImage

        input_array = np.array([[0, 0, 0],
                                [1, 1, 1],
                                [2, 2, 2]])

        expected_output = np.array([[0, 0, 0],
                                    [1, 1, 1],
                                    [0, 0, 0]])

        segmented_image = SegmentedImage.from_array(input_array)

        segmented_image.remove_region(2)

        self.assertTrue(np.array_equal(segmented_image, expected_output))

if __name__ == '__main__':
    unittest.main()
