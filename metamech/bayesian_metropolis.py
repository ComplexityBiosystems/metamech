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
from .metropolis import Metropolis
from .actuator import Actuator
from .spring import LinearSpring
from .images import occlusion_matrix

from typing import List
from typing import Set
from typing import Tuple
from typing import Dict
from typing import Optional
from typing import Union

from itertools import cycle
from pathlib import Path
import numpy as np
from pandas.io.pickle import to_pickle

import keras


class BayesianMetropolis(Metropolis):

    def __init__(
        self,
        actuator: Actuator,
        priors_model: keras.models.Sequential,
        priors_block_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__(actuator)

        # set up model to compute priors
        self.priors_model = priors_model
        self._priors_block_size = priors_block_size

        # setup the figure to plot configurations
        # in raster mode, used for ML predictions
        self._setup_viz()
        self._compute_edges_priors()

    def to_pickle(self, path: Union[str, Path]) -> None:
        """
        Save BayesianMetropolis instance as serialized pickle file.


        Parameters
        ----------
        path : Union[str, Path]
            File path where the pickled object will be stored.

        Raises
        ------
        RuntimeError
            If file already exists
        RuntimeError
            If parent dir does not exist
        """
        assert isinstance(path, (str, Path))
        path = Path(path)
        # make sure directory exists
        directory = path.parent
        if not directory.is_dir():
            raise RuntimeError(
                f"Trying to write to non-existing directory {directory}")
        # make sure file does not exist
        if path.exists():
            raise RuntimeError(
                f"File already exists {path}. Refusing to overwrite!")
        # pickle
        # save a ref to the model
        priors_model = self.priors_model
        self.priors_model = None
        to_pickle(self, path)
        self.priors_model = priors_model

    def _compute_edges_priors(
        self,
        block_size: Optional[Tuple[int, int]] = None,
        keep_white_border: bool = True
    ) -> None:
        """compute the priors using heatmaps of expected change in efficiency"""
        if block_size is None:
            block_size = self._priors_block_size
        self._update_deltaeff_heatmaps(
            block_size=block_size,
            keep_white_border=keep_white_border
        )
        self._edges_delta_eff: Dict[LinearSpring, float] = {}
        # to identify pixels that matter
        current_image = self._edges_to_array(self.lattice.edges)[:, :, 0]
        for edge in self.lattice._possible_edges:
            # are we removing or adding?
            edges_tmp = self.lattice.edges.copy()
            if edge in edges_tmp:
                change_type = "remove_bond"
                edges_tmp.remove(edge)
            else:
                change_type = "add_bond"
                edges_tmp.add(edge)
            # identify pixels that correspond to the edge
            new_image = self._edges_to_array(edges_tmp)[:, :, 0]
            heatmap_locator = (new_image != current_image)
            heatmap = self._priors_heatmaps[change_type]
            # assign weight to the edge
            expected_delta_eff = np.mean(heatmap[heatmap_locator])
            self._edges_delta_eff[edge] = expected_delta_eff

    def _update_deltaeff_heatmaps(
        self,
        block_size: Tuple[int, int] = (50, 50),
        dpi: int = 100,
        keep_white_border: bool = False
    ) -> None:
        """
        Update the heatmaps used to compute priors of edges.

        Slides white/black blocks over the image of the current
        configuration and evaluates the change in efficiency
        that the ML priors model predicts.

        Parameters
        ----------
        block_size : Tuple[int, int], optional
            Size of the sliding block, by default (32, 32)
        dpi : int, optional
            Dots per inch used in heatmaps. Must match dpi of ML model.
        keep_white_border : bool
            Whether to restore the whiteness of the border around the image, even if the model
            gives non-white predictions for it. This is useful when models have undefined behaviour
            outside the 'active area' of the images.
        """
        self._priors_heatmaps: Dict[str, Dict[Tuple[int, int], float]] = {}
        for change_type in ["add_bond", "remove_bond"]:
            self._priors_heatmaps[change_type] = self._compute_expected_deltaeff_heatmap(
                block_size=block_size,
                change_type=change_type,
                dpi=dpi,
                keep_white_border=keep_white_border
            )

    def _compute_expected_deltaeff_heatmap(
        self,
        change_type="add_bond",
        block_size: Tuple[int, int] = (64, 64),
        dpi: int = 100,
        keep_white_border: bool = False
    ) -> Dict[Tuple[int, int], float]:
        # translate to black/white colors,
        # the logic is that you must
        # paint with black to add a bond,
        # paint with white to remove a bond
        block_color = {
            "add_bond": "black",
            "remove_bond": "white"
        }[change_type]
        # for the moment we only work with same
        # block size in w and h
        # but it shouldn't be too difficult to
        # deal with different values
        assert block_size[0] == block_size[1]
        _block_size = block_size[0]
        # get image representation of current configuration
        img = self._edges_to_array(self.lattice.edges, dpi=dpi)
        # compute the expected change in efficiency
        expected_deltaeff_heatmap = occlusion_matrix(
            model=self.priors_model,
            img=img,
            block_size=_block_size,
            block_color=block_color,
            keep_white_border=keep_white_border
        )
        return expected_deltaeff_heatmap
