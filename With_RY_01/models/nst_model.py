try:
    from ..models.model_factory import get_generator
    from ..networks import NST_Generator
    from ..utils.model_utils import weights_init_normal
    from ..models.base_model import BaseModel
except:
    from models.model_factory import get_generator
    from networks import NST_Generator
    from utils.model_utils import weights_init_normal
    from models.base_model import BaseModel
import torch
from torch.autograd import Variable
from collections import OrderedDict


# 只定义模型,没有数据
class Content_Generator_GAN(BaseModel):
    def name(self):
        return 'Content_Generator_GAN'

    def __init__(self, data_options, generator_options, discriminator_options, global_options):
        super(Content_Generator_GAN, self).__init__(data_options, generator_options, discriminator_options, global_options)
        # super.initialization()
        # super.__init__(data_options, generator_options, discriminator_options, global_options)

        self.input_style = self.Tensor(self.data_options['batch_size'],
                                       self.generator_options['input_nc']*self.generator_options['style_num'],
                                       self.data_options['fineSize'],
                                       self.data_options['fineSize'])
        self.input_std = self.Tensor(self.data_options['batch_size'],
                                     self.generator_options['input_nc'], self.data_options['fineSize'],
                                     self.data_options['fineSize'],)
        self.input_gt = self.Tensor(self.data_options['batch_size'],
                                     self.generator_options['input_nc'], self.data_options['fineSize'],
                                     self.data_options['fineSize'],)

        self.netG = get_generator(self.generator_options)
        if not self.global_options['is_training'] or self.global_options['is_retrain']:
            self.load_network(self.netG, 'ContentG', self.global_options['which_epoch'])
        else:
            self.weights_init_normal(self.netG)

        self.criterionL1 = torch.nn.L1Loss()
        self.MSELoss = torch.nn.MSELoss()

        if self.global_options['is_training'] or self.global_options['is_retrain']:
            self.old_lr = self.generator_options['lr']
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.generator_options['lr'],
                                                betas=(self.generator_options['beta1'], 0.999))

        if self.is_cuda:
            self.netG.cuda(device=self.global_options['gpu_ids'][0])

    def set_input(self, input):
        style_imgs = input['style_imgs']
        std_img = input['std_img']
        gt_img = input['gt_img']

        self.input_style.resize_(style_imgs.size()).copy_(style_imgs)
        self.input_std.resize_(std_img.size()).copy_(std_img)
        self.input_gt.resize_(gt_img.size()).copy_(gt_img)

    def test(self):
        self._input_style = Variable(self.input_style, volatile=True)
        self._input_std = Variable(self.input_std, volatile=True)
        self._input_gt = Variable(self.input_gt, volatile=True)
        self.generate_letter = self.netG.forward(self._input_style, self._input_std)

        self.test_G_L1 = self.criterionL1(self.generate_letter, self._input_gt)
        self.test_G_MSE = self.MSELoss(self.generate_letter, self._input_gt)

    def get_current_test_errors(self):
        return OrderedDict([('test_loss_G_L1', self.test_G_L1.data.item()),
                            ('test_loss_G_MSE', self.test_G_MSE.data.item())
                            ])

    def forward(self):
        self._input_style = Variable(self.input_style)
        self._input_std = Variable(self.input_std)
        self._input_gt = Variable(self.input_gt, volatile=True)
        self.generate_letter = self.netG.forward(self._input_style, self._input_std)

    def backward_G(self):
        self.loss_G_L1 = self.criterionL1(self.generate_letter, self._input_gt)
        self.loss_G_MSE = self.MSELoss(self.generate_letter, self._input_gt)
        self.loss_G = (self.loss_G_L1 + self.loss_G_MSE) * 0.5
        self.loss_G.backward()

    def get_current_errors(self):
        return OrderedDict([('loss_G_L1', self.loss_G_L1.data.item()),
                            ('loss_G_MSE', self.loss_G_MSE.data.item())
        ])

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def init_model(self):
        weights_init_normal(self.netG)

    def get_crt_generate_img(self):
        return self.generate_letter

    def get_crt_gt_img(self):
        return self._input_gt

