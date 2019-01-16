try:
    from ..models.model_factory import get_classifier
    from ..models.model_factory import get_discriminator
    from ..networks import classifier_networks
    from ..networks import discriminator_networks
    from ..data.data_factory import get_dataloader
    from ..data.data_loader import Customed_DataLoader
    from ..utils.model_utils import weights_init_normal
    from ..options.default_settings import Tensor
    from ..utils.data_utils import cal_one_hot
except:
    from models.model_factory import get_classifier
    from models.model_factory import get_discriminator
    from networks import classifier_networks
    from networks import discriminator_networks
    from data.data_factory import get_dataloader
    from data.data_loader import Customed_DataLoader
    from utils.model_utils import weights_init_normal
    from options.default_settings import Tensor
    from utils.data_utils import cal_one_hot
import torch
from torch.autograd import Variable
from collections import OrderedDict


class GAN_Proto_Related_Conti_G_Network():
    def __init__(self, classifier_options, discriminator_options, global_options, valid_0_dim,
                 fake_0_dim, tb_v):
        super(GAN_Proto_Related_Conti_G_Network, self).__init__()
        self.classifier_options = classifier_options
        self.discriminator_options = discriminator_options
        self.global_options = global_options
        self.valid_0_dim = valid_0_dim
        self.fake_0_dim = fake_0_dim
        self.tb_v = tb_v
        self.is_need_add_graph = False

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
            self.classifier_optimizor.zero_grad()
            self.discriminator_optimizor.zero_grad()
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
                self.is_need_add_graph = True

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

    def print_model(self):
        print(self.classifier)
        print(self.discriminator)

    def train(self, sample, more_attention_D=True, more_attention_C=False):
        # todo
        # self.classifier.train(mode=True)
        # self.discriminator.f.train(mode=True)
        # self.discriminator.g.train(mode=True)

        xu, xq, yq_one_hot, xs = self.get_data(sample=sample)

        if more_attention_D:
            k_d = 6
        else:
            k_d = 20

        if more_attention_C:
            k_c = 3
        else:
            k_c = 1

        for i in range(k_d):
            self.train_D(xs=xs, xu=xu, xq=xq, yq_one_hot=yq_one_hot)
        for i in range(k_c):
            self.train_C(xs=xs, xu=xu)



    def get_data(self, sample):
        # support unlabeled data --> labeled data
        # fake data
        xu = sample['xu']

        # true data
        xq = sample['xq']
        yq = sample['yq']

        # labeled data
        xs = sample['xs']

        batch_size = yq.size(0)
        num_query_or_unlabeled = yq.size(1)
        way = self.classifier_options['way']
        yq_one_hot = cal_one_hot(yq, batch_size, num_query_or_unlabeled, way)

        if self.global_options['cuda']:
            xu = xu.cuda()
            xq = xq.cuda()
            yq_one_hot = yq_one_hot.cuda()
            xs = xs.cuda()

        return xu, xq, yq_one_hot, xs

    def train_D(self, xs, xu, xq, yq_one_hot):
        z_proto, z_xu, predict_label_p = self.classifier.forward_classifier(xs=xs, xu=xu)
        z_xq = self.classifier.forward_encoder(xq, input_type='xq')

        fake_dis = self.discriminator.forward_g(z_xu.detach(), predict_label_p.detach(), z_proto.detach(), input_type='unlabeled')
        real_dis = self.discriminator.forward_g(z_xq.detach(), yq_one_hot.view(*predict_label_p.size()[:]), z_proto.detach(), input_type='query')

        self.real_loss = self.adversarial_loss(real_dis, self.valid)
        self.fake_loss = self.adversarial_loss(fake_dis, self.fake)
        self.d_loss = (self.real_loss*0.4 + self.fake_loss*0.6) / 1
        self.d_loss.backward()
        self.discriminator_optimizor.step()

    def train_C(self, xs, xu):
        z_proto, z_xu, predict_label_p = self.classifier.forward_classifier(xs=xs, xu=xu)
        fake_dis = self.discriminator.forward_g(z_xu, predict_label_p, z_proto, input_type='unlabeled')
        self.loss_C = self.adversarial_loss(fake_dis, self.valid)
        self.loss_C.backward()
        self.classifier_optimizor.step()

    def test(self, sample):
        return self.classifier.forward_test(sample, run_type=1)[2]
        # todo
        # self.classifier.eval()
        # self.discriminator.f.eval()
        # self.discriminator.g.eval()

        # xu = sample['xu']
        # probu, yu, yu_logits = self.classifier.forward(sample, run_type=0)
        # # true data
        # xq = sample['xq']
        # yq = sample['yq']
        # # labeled data
        # xs = sample['xs']
        #
        # if self.global_options['cuda']:
        #     xu = xu.cuda()
        #     yu = yu.cuda()
        #     xq = xq.cuda()
        #     yq = yq.cuda()
        #     xs = xs.cuda()
        #
        # # set the proto
        # self.discriminator.calculate_prototypical(xs)
        # # calculate the loss
        # real_loss = self.adversarial_loss(self.discriminator.evaluate_forward(xq, yq), self.valid)
        # fake_loss = self.adversarial_loss(self.discriminator.evaluate_forward(xu, yu.detach()), self.fake)
        #
        # acc = self.classifier.forward_test(sample, run_type=1)[2]
        # # return self.classifier.forward_test(sample, run_type=1)[2]
        # return {"test_real_loss": real_loss.data.item(), "test_fake_loss": fake_loss.data.item(), ** acc}

    def get_current_errors(self):
        return OrderedDict([#('loss_gan_classifier', self.g_loss.data.item()),
                            ('loss_classifier', self.loss_C.data.item()),
                            ('loss_gan_fake', self.fake_loss.data.item()),
                            ('loss_gan_real', self.real_loss.data.item()),
                            ('loss_gan_dis', self.d_loss.data.item())
        ])

    def get_current_errors_C(self):
        return OrderedDict([#('loss_gan_classifier', self.g_loss.data.item()),
                            ('loss_classifier', self.loss_C.data.item())
        ])

    def get_current_errors_D(self):
        return OrderedDict([#('loss_gan_classifier', self.g_loss.data.item()),,
                            ('loss_gan_fake', self.fake_loss.data.item()),
                            ('loss_gan_real', self.real_loss.data.item()),
                            ('loss_gan_dis', self.d_loss.data.item())
        ])

