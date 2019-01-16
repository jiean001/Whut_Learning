import os
import torch


def save_network(network, network_label, epoch_label, gpu_ids, save_dir):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if len(gpu_ids) and torch.cuda.is_available():
        network.cuda(device=gpu_ids[0])


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def load_network(network, network_label, epoch_label, save_dir, config_file=None, print_weights=False, ignore_BN=False):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    if not config_file:
        load_base_network(network, weights, print_weights, ignore_BN)
    else:
        model_dict = network.state_dict()
        # new_weights = {}
        f = open(config_file, 'r')
        S_C_dict = json.load(f)
        for s_key in S_C_dict.keys():
            model_dict[s_key] = weights[S_C_dict[s_key]]
        # model_dict.update(new_weights)
        self.load_base_network(network, model_dict, print_weights, ignore_BN)
