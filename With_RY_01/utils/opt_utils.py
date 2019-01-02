#########################################################
# Create on 2018-12-29
#
# Author: jiean001
#
# opt
#########################################################


def filter_opt(opt, tag):
    ret = {}
    for k, v in opt.items():
        tokens = k.split('.')
        if tokens[0] == tag:
            ret['.'.join(tokens[1:])] = v
    return ret


def filter_multi_opt(opt, tag_lst):
    ret_lst = []
    for _ in tag_lst:
        ret_lst.append({})
    for k, v in opt.items():
        tokens = k.split('.')
        if tokens[0] in tag_lst:
            if isinstance(v, str) and ',' in v:
                ret_lst[tag_lst.index(tokens[0])]['.'.join(tokens[1:])] = list(map(int, v.split(',')))
            else:
                ret_lst[tag_lst.index(tokens[0])]['.'.join(tokens[1:])] = v
    ret_dict = {}
    for i in range(len(tag_lst)):
        ret_dict[tag_lst[i]] = ret_lst[i]
    return ret_dict
