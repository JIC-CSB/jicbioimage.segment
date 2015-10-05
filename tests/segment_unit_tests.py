"""Unit tests for the jicbioimage.segment package."""

import unittest

class SegmentTests(unittest.TestCase):

    def test_version_is_string(self):
        # This throws an error if the function cannot be imported.
        import jicbioimage.segment
        self.assertTrue(isinstance(jicbioimage.segment.__version__, str))
