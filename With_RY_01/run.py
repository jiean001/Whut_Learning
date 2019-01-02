try:
    from .options.base_options import BaseOptions
    from .utils.tensorboard_util import TB_Visualizer
    from .controller.run_classifier import Run_Classifier
    from .controller.run_discriminator import Run_Discriminator
    from controller.run_gan_ssl_proto_mlp import Run_GAN_Protypical_MLP
except:
    from options.base_options import BaseOptions
    from utils.tensorboard_util import TB_Visualizer
    from controller.run_classifier import Run_Classifier
    from controller.run_discriminator import Run_Discriminator
    from controller.run_gan_ssl_proto_mlp import Run_GAN_Protypical_MLP


options = BaseOptions()
global_options = options.global_opt
data_options = options.data_opt
classifier_options = options.classifier_opt
discriminator_options = options.discriminator_opt
tb_v = TB_Visualizer(log_dir=global_options['log_dir'], comment=global_options['tfX_comment'], use_tensorboardX=global_options['use_tensorboardX'])

# run_cls = Run_Classifier(classifier_options, data_options, global_options, tb_v)
# run_cls.train()

# run_dis = Run_Discriminator(discriminator_options, data_options, global_options, tb_v)
# # run_dis.add_graph()
# run_dis.randomly_test()
# tb_v.close()

model = Run_GAN_Protypical_MLP(classifier_options, discriminator_options, data_options, global_options, tb_v)
model.tain()
