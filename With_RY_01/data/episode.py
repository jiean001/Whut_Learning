try:
    from ..options.default_settings import Tensor
except:
    from options.default_settings import Tensor
import numpy as np

class Episode:
    def __init__(self,
                 support_images,
                 support_labels,
                 query_images,
                 query_labels,
                 support_unlabel_images):

        self._support_images = support_images
        self._support_labels = support_labels
        self._query_images = query_images
        self._query_labels = query_labels
        self._support_unlabel_images = support_unlabel_images

    def get_data_dict(self):
        return {'xs': Tensor(self._support_images).transpose(1, 3).unsqueeze(0),
                'xu': Tensor(self._support_unlabel_images).transpose(1, 3).unsqueeze(0),
                'xq': Tensor(self._query_images).transpose(1, 3).unsqueeze(0),
                'yq': Tensor(self._query_labels).unsqueeze(0)}

    def get_data_split(self):
        return np.swapaxes(self._support_images, 1, 3),\
               np.swapaxes(self._support_unlabel_images, 1, 3),\
               np.swapaxes(self._query_images, 1, 3), \
               self._query_labels
