try:
    from .options.base_options import BaseOptions
    from .utils.tensorboard_util import TB_Visualizer
    from .controller.run_classifier import Run_Classifier
    from .controller.run_discriminator import Run_Discriminator
    from .controller.run_whole_model import Run_Model
    from .controller.run_nst_model import RUN_NST_Model
except:
    from options.base_options import BaseOptions
    from utils.tensorboard_util import TB_Visualizer
    from controller.run_classifier import Run_Classifier
    from controller.run_discriminator import Run_Discriminator
    from controller.run_whole_model import Run_Model
    from controller.run_nst_model import RUN_NST_Model


options = BaseOptions()
global_options = options.global_opt
generator_options = options.generator_options
data_options = options.data_opt
classifier_options = options.classifier_opt
discriminator_options = options.discriminator_opt
# print(global_options['log_dir'])
tb_v = TB_Visualizer(log_dir=global_options['log_dir'], comment=global_options['tfX_comment'], use_tensorboardX=global_options['use_tensorboardX'])

# run_cls = Run_Classifier(classifier_options, data_options, global_options, tb_v)
# run_cls.train()

# run_dis = Run_Discriminator(discriminator_options, data_options, global_options, tb_v)
# run_dis.test_flatten()
# run_dis.add_graph()
# run_dis.randomly_test()
# tb_v.close()

# model = Run_Model(classifier_options, discriminator_options, data_options, global_options, tb_v)
# model.print_model()
# model.train()

model = RUN_NST_Model(data_options=data_options, generator_options=generator_options,
                      discriminator_options=discriminator_options, global_options=global_options, tb_v=tb_v)
model.train()

