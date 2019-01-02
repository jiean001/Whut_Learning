try:
    from ..options.base_options import BaseOptions
    from ..utils.tensorboard_util import TB_Visualizer
    from ..models.model_factory import get_classifier
    from ..networks import classifier_networks
    from ..options.default_settings import Tensor
    from ..data.omniglot import OmniglotDataset
except:
    from options.base_options import BaseOptions
    from utils.tensorboard_util import TB_Visualizer
    from models.model_factory import get_classifier
    from networks import classifier_networks
    from options.default_settings import Tensor
    from data.omniglot import OmniglotDataset
import numpy as np
import torch
import os
from tqdm import tqdm

dataset_root = r'/home/share/dataset/FSL/'
dataset_name = 'omniglot'
dataset_folder = os.path.join(dataset_root, dataset_name)

class Run_Classifier():
    def __init__(self, classifier_options, data_options, global_options, tb_v):
        super(Run_Classifier, self).__init__()
        self.classifier_options = classifier_options
        self.data_options = data_options
        self.classifier = get_classifier(self.classifier_options['model_name'], self.classifier_options)
        self.global_options = global_options
        if global_options['cuda']:
            self.classifier.cuda(device=self.global_options['gpu_ids'][0])
        self.tb_v = tb_v

        self.meta_train_dataset = OmniglotDataset(dataset_folder,
                                                  split='train',
                                                  num_way=data_options['way'],
                                                  num_shot=data_options['shot'],
                                                  num_query=data_options['query'],
                                                  label_ratio=1)

        self.meta_test_dataset = OmniglotDataset(dataset_folder,
                                                 split='test',
                                                 num_way=data_options['test_way'],
                                                 num_shot=data_options['shot'],
                                                 num_query=data_options['query'],
                                                 label_ratio=1)

        self.optimizor = self.init_optim()
        self.lr_scheduler = self.init_lr_scheduler(self.optimizor)

    def train_one_batch(self, optimizor):
        samples = self.meta_train_dataset.next()
        xs = samples['xs']
        xu = samples['xu']
        xq = samples['xq']
        yq = samples['yq']
        for i in range(1, self.data_options['batch_size']):
            sample = self.meta_train_dataset.next()
            _xs = sample['xs']
            _xu = sample['xu']
            _xq = sample['xq']
            _yq = sample['yq']

            xs = torch.cat((xs, _xs), 0)
            xu = torch.cat((xu, _xu), 0)
            xq = torch.cat((xq, _xq), 0)
            yq = torch.cat((yq, _yq), 0)

        samples2 = {}
        samples2['xs'] = xs
        samples2['xu'] = xu
        samples2['xq'] = xq
        samples2['yq'] = yq

        optimizor.zero_grad()
        loss = self.classifier.forward(samples2)[1]
        loss.backward()
        optimizor.step()
        return loss


    def test_one_batch(self):
        samples = self.meta_test_dataset.next()
        xs = samples['xs']
        xu = samples['xu']
        xq = samples['xq']
        yq = samples['yq']
        for i in range(1, self.data_options['batch_size']):
            sample = self.meta_test_dataset.next()
            _xs = sample['xs']
            _xu = sample['xu']
            _xq = sample['xq']
            _yq = sample['yq']

            xs = torch.cat((xs, _xs), 0)
            xu = torch.cat((xu, _xu), 0)
            xq = torch.cat((xq, _xq), 0)
            yq = torch.cat((yq, _yq), 0)

        samples2 = {}
        samples2['xs'] = xs
        samples2['xu'] = xu
        samples2['xq'] = xq
        samples2['yq'] = yq

        acc = self.classifier.forward_test(samples2)[2]['acc']
        return acc

    def train(self):
        # for epoch in tqdm(range(self.global_options['epoches'])):
        for epoch in range(self.global_options['epoches']):
            # print('=== Epoch: {} ==='.format(epoch))
            for batch in range(self.data_options['train_episodes']):
                loss = self.train_one_batch(optimizor=self.optimizor)
                self.tb_v.add_loss({'sss':loss}, epoch*self.data_options['train_episodes'] + batch)

            self.lr_scheduler.step()

            if epoch % 1 == 0:
                acc = []
                for epoch_val in range(10):
                    acc.append(self.test_one_batch())
                self.tb_v.add_loss({'acc mean': np.array(acc).mean()}, epoch)
                print(np.array(acc).mean())

    def init_optim(self):
        return torch.optim.Adam(params=self.classifier.parameters(), lr=self.classifier_options['lr'])

    def init_lr_scheduler(self, optim):
        '''
        Initialize the learning rate scheduler
        '''
        return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                               gamma=self.classifier_options['lrG'],
                                               step_size=self.classifier_options['lrS'])

    def print_network(self):
        print(self.classifier)

    def add_graph(self):
        sample = self.get_randomly_data()
        input_data = sample['xs'][0]
        self.tb_v.add_graph(self.classifier.get_graph(), (input_data,))

    def get_randomly_data(self):
        # randomly generate test data
        b = 2
        way = self.data_options['way']
        shot = self.data_options['shot']
        query = self.data_options['query']
        c = 1
        h = 28
        w = 28

        xs = Tensor(np.random.random((b, way, shot, c, h, w)))
        xq = Tensor(np.random.random((b, way, query, c, h, w)))
        yq = Tensor(np.random.randint(0, way, (b, way * query)))

        sample = {'xs': xs.view(b, way * shot, c, h, w),
                  'xq': xq.view(b, way * query, c, h, w),
                  'yq': yq
                  }
        return sample

    def randomly_test(self):
        sample = self.get_randomly_data()
        self.classifier.forward(sample, run_type=1)
