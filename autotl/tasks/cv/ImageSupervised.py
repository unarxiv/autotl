from autotl.fileutils import temp_folder_generator
from ..supervised import Supervised
from constant import Constant
class ImageSupervised(Supervised):
    def __init__(self, 
                 verbose=False,
                 path=None,
                 resume=False,
                 searcher_args=None,
                 augment=None):
        super().__init__(verbose)
        if searcher_args is None:
            searcher_args = {}
        if path is None:
            path = temp_folder_generator()
        if augment is None:
            augment = Constant.DATA_AUGMENTATION
        