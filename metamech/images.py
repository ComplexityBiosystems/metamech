# Codes used for the paper:
# "Automatic Design of Mechanical Metamaterial Actuators"
# by S. Bonfanti, R. Guerra, F. Font-Clos, R. Rayneau-Kirkhope, S. Zapperi
# Center for Complexity and Biosystems, University of Milan
# (c) University of Milan
#
#
######################################################################
#
# End User License Agreement (EULA)
# Your access to and use of the downloadable code (the "Code") is subject
# to a non-exclusive,  revocable, non-transferable,  and limited right to
# use the Code for the exclusive purpose of undertaking academic,
# governmental, or not-for-profit research. Use of the Code or any part
# thereof for commercial purposes is strictly prohibited in the absence
# of a Commercial License Agreement from the University of Milan. For
# information contact the Technology Transfer Office of the university
# of Milan (email: tto@unimi.it)
#
#######################################################################
"""
Functions to work with images of mechanical metamaterials
"""

from typing import Tuple
from typing import Dict
from typing import Callable
from typing import Optional

import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import find_objects

import typing
if typing.TYPE_CHECKING:
    from keras.models import Sequential


def perturbe_image(img: np.array, block_size: int = 4, block_color: str = "black", keep_white_border: bool = False):
    """
    Perturbe image by sliding a small square over it.

    This is useful e.g. to test how perturbations affect a models predictions.
    This function slides a block of block_size pixels and block_color color over 
    an image img, and returns an array where the first indexing dimension corresponds
    to the possible perturbations. 

    Parameters
    ----------
    img: np.array(W, H, C)
        The image to be perturbed, channels_last.
    block_size: int
        Sets the size of the square that is slided over the image.
        The square will be (block_size x block_size). The default
        is 4.
    block_color: str
        Color of the sliding block. Either "black" or "white", at the moment.

    Returns
    -------
    img_pert: np.array(P, W, H, C)
        The array of perturbed images, new index P indexes perturbation.
    """
    w_pixels, h_pixels, _ = img.shape
    assert w_pixels % block_size == 0
    assert h_pixels % block_size == 0
    if block_color == "black":
        cover_value = 0
    elif block_color == "white":
        cover_value = 1
    else:
        raise ValueError(
            "Invalid value of 'block_color', must be either 'black' or 'white'.")

    img_pert = []
    crop_zone = find_objects(img == 0)[0]
    for w in range(0, w_pixels, block_size):
        for h in range(0, h_pixels, block_size):
            imgg = img.copy()
            imgg[w:w+block_size, h:h+block_size, :] = cover_value
            if keep_white_border:
                imgg = _colorize_margin(
                    imgg, crop_zone=crop_zone, fill_value=1)
            img_pert.append(imgg)
    img_pert = np.array(img_pert)
    return img_pert


def occlusion_matrix(model: "Sequential", img: np.array, block_size: int = 4, block_color: str = "black", order: int = 3, keep_white_border: bool = False):
    """
    Compute the occlusion matrix of an image given a regressive model.

    The occlusion matrix of an image given a regressive model tells you how
    much the prediction of the model changes when a small colored square is
    placed on top of the image at certain location.

    Plotting an occlusion heatmap helps understand which areas of the image
    are of particular importance according to the model.

    Parameters
    ----------
    model: keras.Model, function, ...
        A model or function such that takes img as input and returns a float.
        Typically model has been trained in some regression task, y = model(img)
    img: np.array(W, H, C)
        The image to be occluded.
    block_size: int
        Sets the size of the square that is slided over the image.
        The square will be (block_size x block_size). The default
        is 4.
    block_color: str
        Color of the sliding block. Either "black" or "white", at the moment.
    order: int
        Interpolation order to zoom the image and recover the original number of pixels.
    keep_white_border : bool
        Whether to restore the whiteness of the border around the image, even if the model
        gives non-white predictions for it. This is useful when models have undefined behaviour
        outside the 'active area' of the images.
    """

    w_pixels, h_pixels, _ = img.shape
    y = model.predict(np.array([img]))[0][0]
    img_pert = perturbe_image(
        img=img, block_size=block_size, block_color=block_color, keep_white_border=keep_white_border)
    y_pert = model.predict(img_pert)
    dy = y_pert.reshape(1, int(w_pixels/block_size),
                        int(h_pixels/block_size))[0] - y

    # interpolate
    dy_zoomed = zoom(dy, block_size, order=order)
    # remove border from interpolated image as well
    if keep_white_border:
        crop_zone = find_objects(img[:, :, 0] == 0)[0]
        # notice that now dy has white = transparent = value 0 = center of colorscale
        dy_zoomed = _colorize_margin(
            dy_zoomed, crop_zone=crop_zone, fill_value=0)

    return dy_zoomed


def _colorize_margin(img, crop_zone: Tuple[slice, slice], fill_value: float = 1):
    """
    Colorize the margin of an image

    The fill value is in grayscale values, so 1 is white

    Parameters
    ----------
    img : np.array(W, H)
        [description]
    crop_zone : Tuple[slice, slice]
        [description]
    fill_value : Optional[float], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    # find the crop zone, a slicer that selects the active part of the image
    # active part is defined as being black (value = 0)
    # create a colored background, fill value is 1 for white
    img_margin_colored = fill_value * np.ones_like(img)
    # then copy the image back in, but only the part inside
    # the crop zone
    img_margin_colored[crop_zone] = img[crop_zone]
    return img_margin_colored
