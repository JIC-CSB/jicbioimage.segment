"""Module containing image segmentation functions.
"""

import numpy as np
import skimage.measure

from jicbioimage.core.image import SegmentedImage
from jicbioimage.core.transform import transformation

__version__ = "0.0.1"


@transformation
def connected_components(image, connectivity=2, background=None):
    """Return :class:`jicbioimage.core.image.SegmentedImage`.

    This function wraps the :func:``skimage.measure.label`` function.

    :param image: input :class:`jicbioimage.core.image.Image`
    :param connectivity: maximum number of orthagonal hops to consider a
                         pixel/voxel as a neighbor
    :param background: consider all pixels with this value (int) as background
    """
    ar = skimage.measure.label(image, connectivity=connectivity,
                               background=background)

    # The :class:`jicbioimage.core.image.SegmentedImage` assumes that zero is
    # background.  So we need to change the identifier of any pixels that are
    # marked as zero if there is no background in the input image.
    if background is None:
        ar[np.where(ar == 0)] = np.max(ar) + 1
    else:
        if np.min(ar) == -1:
            # Work around skimage.measure.label behaviour pre version 0.12.
            # Pre version 0.12 the background in skimage was labeled -1 and the
            # first component was labelled with 0.
            # The jicbioimage.core.image.SegmentedImage assumes that the
            # background is labelled 0.
            ar[np.where(ar == 0)] = np.max(ar) + 1
            ar[np.where(ar == -1)] = 0

    segmentation = SegmentedImage.from_array(ar)
    return segmentation
