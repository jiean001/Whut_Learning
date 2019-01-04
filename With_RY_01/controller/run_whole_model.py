try:
    from ..data.data_factory import get_dataloader
    from ..data.data_loader import Customed_DataLoader
    from ..models.gan_ssl_protypical_mlp import GAN_Protypical_MLP
    from ..models.gan_ssl_protypical_related_network import GAN_Protypical_Related_Network
except:
    from data.data_factory import get_dataloader
    from data.data_loader import Customed_DataLoader
    from models.gan_ssl_protypical_mlp import GAN_Protypical_MLP
    from models.gan_ssl_protypical_related_network import GAN_Protypical_Related_Network
from tqdm import tqdm


class Run_Model():
    def __init__(self, classifier_options, discriminator_options, data_options, global_options, tb_v):
        super(Run_Model, self).__init__()
        self.global_options = global_options
        self.data_options = data_options
        self.classifier_options = classifier_options
        self.tb_v = tb_v
        fake_0_dim = self.data_options['batch_size']*self.data_options['way']*self.data_options['unlabeled']
        valid_0_dim = self.data_options['batch_size'] * self.data_options['way'] * self.data_options['query']

        print('model name is:', self.global_options['model_name'])
        if self.global_options['model_name'] == 'protypical_mlp_network':
            self.model = GAN_Protypical_MLP(classifier_options, discriminator_options, global_options,
                                            valid_0_dim, fake_0_dim)
        elif self.global_options['model_name'] == 'protypical_related_network':
            self.model = GAN_Protypical_Related_Network(classifier_options, discriminator_options, global_options,
                                            valid_0_dim, fake_0_dim, tb_v)
        else:
            raise ValueError("Unknown model {:s}".format(self.global_options['model_name']))
        self.meta_train_dataset, self.meta_val_or_test_dataset = self.init_dataset()

    def init_dataset(self):
        meta_train_dataset = None
        self.data_options['episodes'] = self.data_options['train_episodes']
        data_val_or_test_options = self.data_options.copy()
        data_val_or_test_options['label_ratio'] = 1
        data_val_or_test_options['way'] = self.data_options['test_way']
        data_val_or_test_options['shot'] = self.data_options['test_shot']
        #data_val_or_test_options['unlabeled'] = 0
        data_val_or_test_options['query'] = self.data_options['test_query']
        data_val_or_test_options['episodes'] = self.data_options['test_episodes']

        if self.global_options['is_training']:
            data_val_or_test_options['split'] = 'val'
            meta_train_dataset = get_dataloader(self.data_options)
        else:
            data_val_or_test_options['split'] = 'test'
        meta_val_or_test_dataset = get_dataloader(data_val_or_test_options)
        return meta_train_dataset, meta_val_or_test_dataset

    # only for draw graph
    def randomly_generate(self):
        import numpy as np
        try:
            from ..options.default_settings import Tensor
        except:
            from options.default_settings import Tensor

        b = self.data_options['batch_size']
        way = self.data_options['way']
        shot = self.data_options['shot']
        c, h, w = self.classifier_options['x_dim']

        x = Tensor(np.random.random((b, way * shot, c, h, w)))
        y = Tensor(np.random.randint(0, way, (b, way * shot))).long()
        return x, y

    def add_graph(self, sample):
        input_data = sample['xs'][0].cuda().float()
        print(input_data.size())
        self.tb_v.add_graph(self.model.classifier.get_graph(), input_data)

        # x, y = self.randomly_generate()
        # self.tb_v.add_graph(self.model.discriminator.get_graph(), (x, y))

    def train(self):
        haha = 0

        for epoch in tqdm(range(self.global_options['epoches'])):
            for i, sample in tqdm(enumerate(self.meta_train_dataset.load_data())):
                self.model.train(sample=sample)
                loss = self.model.get_current_errors()
                self.tb_v.add_loss(errors=loss, scalar_x=epoch*self.data_options['train_episodes']+i)

            for j, test_sample in tqdm(enumerate(self.meta_val_or_test_dataset.load_data())):
                evalution = self.model.test(test_sample)
                self.tb_v.add_loss(errors=evalution, scalar_x=epoch*self.data_options['test_episodes']+j)
                haha += 1
                #print('ecpoch:%d' %(epoch), evalution['acc'])
            self.model.update_lr_scheduler()

    def print_model(self):
        self.model.print_model()




