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
from typing import Tuple
import keras


def keras_resnet50_regression(
    input_shape: Tuple[int, int, int],
    loss: str = "mean_squared_error",
    optimizer=keras.optimizers.Adam()
):
    """Create a ResNet50 regressor. 


    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of images when converted to arrays

    Returns
    -------
    keras.engine.sequential.Sequential
        ResNet50 (not pretrained) with a final dense layer for regression
    """

    resnet = keras.applications.ResNet50(
        include_top=False,
        pooling="avg",
        weights=None,
        input_shape=input_shape,
    )

    model = keras.models.Sequential()
    model.add(resnet)
    model.add(keras.layers.Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    return model
