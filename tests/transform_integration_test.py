"""Transform integration tests."""

import unittest
import os
import os.path
import shutil
import numpy as np

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, 'data')
TMP_DIR = os.path.join(HERE, 'tmp')


class TransformationDecoratorIntegrationTest(unittest.TestCase):

    def setUp(self):
        from jicbioimage.core.io import AutoName
        AutoName.count = 0
        if not os.path.isdir(TMP_DIR):
            os.mkdir(TMP_DIR)

    def tearDown(self):
        from jicbioimage.core.io import AutoName
        AutoName.count = 0
        shutil.rmtree(TMP_DIR)


    def test_can_return_segmented_image(self):
        from jicbioimage.core.image import Image
        from jicbioimage.segment import SegmentedImage
        from jicbioimage.core.transform import transformation
        from jicbioimage.core.io import AutoName
        AutoName.directory = TMP_DIR

        @transformation
        def test_segmentation(image):
            return image.view(SegmentedImage)

        image = Image.from_array(np.zeros((50, 50), dtype=np.uint8))
        self.assertTrue(isinstance(image, Image))
        segmentation = test_segmentation(image)
        self.assertTrue(isinstance(segmentation, SegmentedImage))
        self.assertEqual(len(segmentation.history), 2)
        self.assertEqual(segmentation.history[-1],
                         "Applied test_segmentation transform")

if __name__ == '__main__':
    unittest.main()
