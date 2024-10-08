# ---------------------------------------------------------------
# Copyright (c) 2023 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class BDD100KDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(BDD100KDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='_train_id.png', **kwargs)
        self.valid_mask_size = [1280, 720]
