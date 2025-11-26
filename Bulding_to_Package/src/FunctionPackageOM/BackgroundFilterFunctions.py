from scipy.ndimage import gaussian_filter
from pyxem.utils.diffraction import regional_filter
from skimage.filters.rank import mean as rank_mean
from skimage.morphology import square
import numpy as np


## define prefilter like hyperspy bkgrd substraction - difference of Gaussians
def PreFilterDifferenceOfGaussians(frame, min_sigma=1, max_sigma=1.5):
    """Background removal using difference of Gaussians.

    Parameters
    ----------
    frame : NumPy 2D array
    min_sigma : float
    max_sigma : float

    Returns
    -------
    background_removed : Numpy 2D array

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.data.dummy_data.dummy_data.get_cbed_signal()
    >>> s_rem = dt._background_removal_single_frame_dog(s.data[0, 0])

    """
    blur_max = gaussian_filter(frame, max_sigma)
    blur_min = gaussian_filter(frame, min_sigma)
    return np.maximum(np.where(blur_min > blur_max, frame, 0) - blur_max, 0)


## difference of Gaussians with h-Dome
def PreFilterDifferenceOfGaussiansWithHDome(frame, min_sigma=1, max_sigma=1.5, h=0.7):
    """Background removal using difference of Gaussians.

    """
    
    blur_max = gaussian_filter(frame, max_sigma)
    blur_min = gaussian_filter(frame, min_sigma)

    differenceOfGaussians = np.maximum(np.where(blur_min > blur_max, frame, 0) - blur_max, 0)

    """Background removal using h-dome filter."""
    max_value = np.max(differenceOfGaussians)
    bg_subtracted = rank_mean(
        regional_filter(differenceOfGaussians / max_value, h = h), footprint=square(3)
    )
    bg_subtracted = bg_subtracted / np.max(bg_subtracted)
    return bg_subtracted


## difference of Gaussians with h-Dome
def PreFilterDifferenceOfGaussiansWithHDomeAndMask(frame, min_sigma=1, max_sigma=1.5, h=0.7):
    """Background removal using difference of Gaussians.

    """
    
    blur_max = gaussian_filter(frame, max_sigma)
    blur_min = gaussian_filter(frame, min_sigma)

    differenceOfGaussians = np.maximum(np.where(blur_min > blur_max, frame, 0) - blur_max, 0)

    """Background removal using h-dome filter."""
    max_value = np.max(differenceOfGaussians)
    bg_subtracted = rank_mean(
        regional_filter(differenceOfGaussians / max_value, h = h), footprint=square(3)
    )
    bg_subtracted = bg_subtracted / np.max(bg_subtracted)

    ## apply mask ontop
    imageBackgroundSubtractedAndMasked = ApplyPreLogFilterWithMask(bg_subtracted)
    
    return imageBackgroundSubtractedAndMasked


## make prefilter with masking of center part

## define filter functions
def ApplyPreLogFilterWithMask(image):
    # Base-10 logarithm of (1 + x)
    filteredImage = np.log10(1 + image)
    mask = py4DSTEM.process.utils.utils.sector_mask((180, 135), (85.71045949530398, 63.72273423375648), 14.9780148620565, angle_range=(0, 360))
    filteredImageMasked =  np.where(~mask, filteredImage, 0)
    
    return filteredImageMasked


## define filter functions
def ApplyPreFilterWithMask(image):
    # Base-10 logarithm of (1 + x)
    filteredImage = image
    mask = py4DSTEM.process.utils.utils.sector_mask((180, 135), (85.71045949530398, 63.72273423375648), 16.9780148620565, angle_range=(0, 360))
    filteredImageMasked =  np.where(~mask, filteredImage, 0)
    
    return filteredImageMasked

