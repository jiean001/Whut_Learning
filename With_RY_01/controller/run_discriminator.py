try:
    from ..utils.tensorboard_util import TB_Visualizer
    from ..networks import discriminator_networks
    from ..options.default_settings import Tensor
    from ..models.model_factory import get_discriminator
except:
    from utils.tensorboard_util import TB_Visualizer
    from networks import discriminator_networks
    from options.default_settings import Tensor
    from models.model_factory import get_discriminator
import numpy as np


class Run_Discriminator():
    def __init__(self, discriminator_options, data_options, global_options, tb_v):
        super(Run_Discriminator, self).__init__()
        self.discriminator_options = discriminator_options
        self.data_options = data_options
        self.discriminator = get_discriminator(self.discriminator_options['model_name'], self.discriminator_options)
        self.global_options = global_options
        if global_options['cuda']:
            self.discriminator.cuda(device=global_options['gpu_ids'][0])
        self.tb_v = tb_v

    def print_network(self):
        print(self.discriminator)

    def add_graph(self):
        x, y = self.randomly_generate()
        self.tb_v.add_graph(self.discriminator.get_graph(), (x, y))

    def randomly_generate(self):
        b = 1
        way = self.data_options['way']
        shot = self.data_options['shot']
        c = 1
        h = 28
        w = 28

        x = Tensor(np.random.random((b, way * shot, c, h, w)))
        y = Tensor(np.random.randint(0, way, (b, way * shot))).long()
        return x, y

    def randomly_test(self):
        x, y = self.randomly_generate()
        print(self.discriminator.forward(x, y))
