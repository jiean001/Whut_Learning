# import sys
# sys.path.append('/home/anna/code/open/ICCV/demos')

try:
    from ..options.base_options import BaseOptions
except:
    from options.base_options import BaseOptions

opt = BaseOptions().parse()