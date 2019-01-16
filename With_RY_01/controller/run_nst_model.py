try:
    from ..data.data_factory import get_dataloader
    from ..data.data_loader import NST_Customed_DataLoader
    from ..models.nst_model import Content_Generator_GAN
except:
    from data.data_factory import get_dataloader
    from data.data_loader import NST_Customed_DataLoader
    from models.nst_model import Content_Generator_GAN

from tqdm import tqdm
import signal

class RUN_NST_Model():
    def __init__(self, data_options, generator_options, discriminator_options, global_options, tb_v):
        super(RUN_NST_Model, self).__init__()
        self.train_data, self.val_data = self.init_train_data(data_options)
        self.model = Content_Generator_GAN(data_options, generator_options, discriminator_options, global_options)

        signal.signal(signal.SIGINT, self.sigint_handel)
        signal.signal(signal.SIGHUP, self.sigint_handel)
        signal.signal(signal.SIGTERM, self.sigint_handel)
        self.is_sigint_up = False
        self.start_epoch = global_options['start_epoch']
        self.epoches = global_options['epoches']
        self.tb_v = tb_v

    def init_train_data(self, data_options):
        train_data = get_dataloader(data_options)
        val_options = data_options.copy()
        val_options['dataset_type'] = 'val'
        val_data =get_dataloader(val_options)
        return train_data, val_data

    def train(self):
        cor_train = 0
        cor_val = 0
        for epoch in range(self.start_epoch, self.epoches):
            # training
            for j, sample in tqdm(enumerate(self.train_data.load_data())):
                cor_train += 1
                # ctrl+c　后保存模型
                if self.is_sigint_up:
                    print('exit')
                    self.model.save('interrupt_latest_%d_%d' % (epoch, j))
                    exit()
                # 喂数据
                self.model.set_input(sample)
                # 更新一次参数
                self.model.optimize_parameters()
                # 获取errors
                errors = self.model.get_current_errors()
                # 画图
                self.tb_v.add_loss(errors, scalar_x=cor_train)
                if j % 10 == 0:
                    generate_imgs = self.model.get_crt_generate_img()
                    gt_imgs = self.model.get_crt_gt_img()
                    self.tb_v.add_img(tag='generate', img=generate_imgs, iter=cor_train)
                    self.tb_v.add_img(tag='ground_truth', img=gt_imgs, iter=cor_train)

            # validation
            for j, sample in tqdm(enumerate(self.train_data.load_data())):
                cor_val += 1
                # ctrl+c　后保存模型
                if self.is_sigint_up:
                    print('exit')
                    self.model.save('interrupt_latest_%d_%d' % (epoch, j))
                    exit()
                # 喂数据
                self.model.set_input(sample)
                # 更新一次参数
                self.model.test()
                # 获取errors
                errors = self.model.get_current_test_errors()
                # 画图
                self.tb_v.add_loss(errors, scalar_x=cor_val)

    def sigint_handel(self, signum, frame):
        global is_sigint_up
        self.is_sigint_up = True
        print('catched interrupt signal')

