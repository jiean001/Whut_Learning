try:
    from ..models.model_factory import get_classifier
    from ..models.model_factory import get_discriminator
    from ..networks import classifier_networks
    from ..networks import discriminator_networks
    from ..data.data_factory import get_dataloader
    from ..data.data_loader import Customed_DataLoader
    from ..utils.model_utils import weights_init_normal
    from ..options.default_settings import Tensor
except:
    from models.model_factory import get_classifier
    from models.model_factory import get_discriminator
    from networks import classifier_networks
    from networks import discriminator_networks
    from data.data_factory import get_dataloader
    from data.data_loader import Customed_DataLoader
    from utils.model_utils import weights_init_normal
    from options.default_settings import Tensor
import torch
from torch.autograd import Variable
from collections import OrderedDict


class GAN_Protypical_MLP():
    def __init__(self, classifier_options, discriminator_options, global_options, valid_0_dim, fake_0_dim):
        super(GAN_Protypical_MLP, self).__init__()
        self.classifier_options = classifier_options
        self.discriminator_options = discriminator_options
        self.global_options = global_options
        self.valid_0_dim = valid_0_dim
        self.fake_0_dim =fake_0_dim

        self.classifier = get_classifier(self.classifier_options['model_name'], self.classifier_options)
        self.discriminator = get_discriminator(self.discriminator_options['model_name'], self.discriminator_options)

        if global_options['cuda']:
            self.classifier.cuda(device=self.global_options['gpu_ids'][0])
            self.discriminator.cuda(device=self.global_options['gpu_ids'][0])

        # test
        if self.global_options['is_testing']:
            self.load_mode()
        # training or retrain
        else:
            self.classifier_optimizor = self.init_optim_cls(self.classifier)
            self.classifier_lr_scheduler = self.init_lr_scheduler_cls(self.classifier_optimizor)
            self.discriminator_optimizor = self.init_optim_dis(self.discriminator)
            self.discriminator_lr_scheduler = self.init_lr_scheduler_dis(self.discriminator_optimizor)
            self.adversarial_loss = torch.nn.BCELoss()
            if global_options['cuda']:
                self.adversarial_loss.cuda()
            # only in training, can we use discriminator
            self.valid, self.fake = self.get_init_valid_fake_data()

            # retain
            if self.global_options['is_retrain']:
                self.load_model()
            # training
            else:
                self.init_model()


    def init_model(self):
        weights_init_normal(self.classifier)
        weights_init_normal(self.discriminator)

    def load_model(self):
        # todo
        pass

    def get_init_valid_fake_data(self):
        valid = Variable(Tensor(self.valid_0_dim, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(self.fake_0_dim, 1).fill_(0.0), requires_grad=False)
        return valid, fake

    def init_optim_cls(self, model):
        if self.classifier_options['optim'] == 'Adam':
            return torch.optim.Adam(params=model.parameters(), lr=self.classifier_options['lr'])
        else:
            assert 1 == 2

    def init_lr_scheduler_cls(self, optim):
        return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                               gamma=self.classifier_options['lrG'],
                                               step_size=self.classifier_options['lrS'])

    def init_optim_dis(self, model):
        if self.discriminator_options['optim'] == 'Adam':
            return torch.optim.Adam(params=model.parameters(), lr=self.discriminator_options['lr'])
        else:
            assert 1 == 2

    def init_lr_scheduler_dis(self, optim):
        return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                               gamma=self.discriminator_options['lrG'],
                                               step_size=self.discriminator_options['lrS'])
    def update_lr_scheduler(self):
        self.classifier_lr_scheduler.step()
        self.discriminator_lr_scheduler.step()

    def train(self, sample):
        # support unlabeled data --> labeled data
        # fake data
        xu = sample['xu']
        yu, yu_logits = self.classifier.forward(sample, run_type=0)
        # true data
        xq = sample['xq']
        yq = sample['yq']

        if self.global_options['cuda']:
            xu = xu.cuda()
            yu = yu.cuda()
            xq = xq.cuda()
            yq = yq.cuda()

        tmp = self.discriminator(xu, yu)
        self.g_loss = self.adversarial_loss(tmp, self.valid)
        self.g_loss.backward()
        self.classifier_optimizor.step()

        self.real_loss = self.adversarial_loss(self.discriminator(xq, yq), self.valid)
        self.fake_loss = self.adversarial_loss(self.discriminator(xu, yu.detach()), self.fake)
        self.d_loss = (self.real_loss + self.fake_loss) / 2

        self.d_loss.backward()
        self.discriminator_optimizor.step()

    # return {loss, acc}
    def test(self, sample):
        return self.classifier.forward_test(sample, run_type=1)[2]

    def get_current_errors(self):
        return OrderedDict([('loss_gan_classifier', self.g_loss.data.item()),
                            ('loss_gan_fake', self.fake_loss.data.item()),
                            ('loss_gan_real', self.real_loss.data.item()),
                            ('loss_gan_dis', self.d_loss.data.item())
        ])


