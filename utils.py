import os
import logging
import torch
import shutil

def set_gpu(args):
    if args.device == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.device.split(',')]
        print('use gpu:', gpu_list)
        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    return gpu_list.__len__()


def allocate_tensors():
    """
    init data tensors
    :return: data tensors
    """
    tensors = dict()
    tensors['support_data'] = torch.FloatTensor()
    tensors['support_label'] = torch.LongTensor()
    tensors['query_data'] = torch.FloatTensor()
    tensors['query_label'] = torch.LongTensor()
    return tensors


def set_logging_config(logdir):
    """
    set logging configuration
    :param logdir: directory put logs
    :return: None
    """
    if not os.path.exists(logdir): os.makedirs(logdir)
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s", level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')), logging.StreamHandler(os.sys.stdout)])


def save_checkpoint(state, is_best, exp_name):
    """
    save the checkpoint during training stage
    :param state: content to be saved
    :param is_best: if DPGN model's performance is the best at current step
    :param exp_name: experiment name
    :return: None
    """
    torch.save(state, os.path.join('{}'.format(exp_name), 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join('{}'.format(exp_name), 'checkpoint.pth.tar'), os.path.join('{}'.format(exp_name), 'model_best.pth.tar'))


def adjust_learning_rate(optimizers, lr, iteration, dec_lr_step, lr_adj_base):
    """
    adjust learning rate after some iterations
    :param optimizers: the optimizers
    :param lr: learning rate
    :param iteration: current iteration
    :param dec_lr_step: decrease learning rate in how many step
    :return: None
    """
    new_lr = lr * (lr_adj_base ** (int(iteration / dec_lr_step)))
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def label2matrix(label, device):
    """
    convert ground truth labels into ground truth matrix
    :param label: ground truth labels
    :param device: the gpu device that holds the ground truth matrix
    :return: ground truth matrix
    """
    # get size
    num_samples = label.size(1)
    # reshape
    label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
    label_j = label_i.transpose(1, 2)
    # compute matrix
    matrix = torch.eq(label_i, label_j).float().to(device)
    return matrix


def one_hot_encode(num_classes, class_idx, device):
    """
    one-hot encode the ground truth
    :param num_classes: number of total class
    :param class_idx: belonging class's index
    :param device: the gpu device that holds the one-hot encoded ground truth label
    :return: one-hot encoded ground truth label
    """
    class_idx = class_idx.to(device)  
    eye_tensor = torch.eye(num_classes, device=device)
    return eye_tensor[class_idx]


# Demo: 5way-1shot-1query
def preprocessing(num_ways, num_shots, num_queries, batch_size, device):
    # set size of support set, query set and total number of data in single task
    num_supports = num_ways * num_shots                   # num_supports: 5*1=5
    num_samples = num_supports + num_queries * num_ways   # num_samples: 5 + 1*5 =10 

    # set clsm mask (to distinguish support and query clsm)
    support_clsm_mask = torch.zeros(batch_size, num_samples, num_samples).to(device)
    support_clsm_mask[:, :num_supports, :num_supports] = 1
    # print('support_clsm_mask:', support_clsm_mask.shape)  # support_clsm_mask: torch.Size([16, 10, 10])
    # print('support_clsm_mask[0]:', support_clsm_mask[0])
    ''' support_clsm_mask[0]: tensor([[1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                      [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                      [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                      [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                      [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')'''
    
    query_clsm_mask = 1 - support_clsm_mask
    # print('query_clsm_mask:', query_clsm_mask.shape) # query_clsm_mask: torch.Size([16, 10, 10])
    # print('query_clsm_mask[0]:', query_clsm_mask[0])
    '''query_clsm_mask[0]: tensor([[0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
                                   [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
                                   [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
                                   [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
                                   [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]], device='cuda:0')'''

    return num_supports, num_samples, query_clsm_mask
    # print('num_supports:', num_supports)               # 5
    # print('num_samples:', num_samples)                 # 10
    # print('query_clsm_mask:', query_clsm_mask.shape)   # torch.Size([16, 10, 10])


def set_tensors(tensors, batch):
    """
    set data to initialized tensors
    :param tensors: initialized data tensors
    :param batch: current batch of data
    :return: None
    """
    support_data, support_label, query_data, query_label = batch
    tensors['support_data'].resize_(support_data.size()).copy_(support_data)
    tensors['support_label'].resize_(support_label.size()).copy_(support_label)
    tensors['query_data'].resize_(query_data.size()).copy_(query_data)
    tensors['query_label'].resize_(query_label.size()).copy_(query_label)


# Demo: 5way-1shot-1query
def initialize_emb_clsm(batch, num_supports, tensors, batch_size, num_queries, num_ways, device):
    """
    :param batch: data batch
    :param num_supports: number of samples in support set
    :param tensors: initialized tensors for holding data
    :param batch_size: how many tasks per batch
    :param num_queries: number of samples in query set
    :param num_ways: number of classes for each few-shot task
    :param device: the gpu device that holds all data
    """
    # allocate data in this batch to specific variables
    set_tensors(tensors, batch)
    support_data = tensors['support_data'].squeeze(0)        
    support_label = tensors['support_label'].squeeze(0)      
    query_data = tensors['query_data'].squeeze(0)            
    query_label = tensors['query_label'].squeeze(0)          
    all_data = torch.cat([support_data, query_data], 1)      
    all_label = torch.cat([support_label, query_label], 1)   
    all_label_in_clsm = label2matrix(all_label, device)      

    #######################################
    # Initialize L-channel CLSM:
    #######################################
    # 1)initialize L-channel clsm
    L_cls_matrix = all_label_in_clsm.clone()         # light_cls_matrix: torch.Size([16, 10, 10])
    # print('L_cls_matrix[0]:', L_cls_matrix[0])
    '''
    L_cls_matrix[0]: tensor([
            [1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]], device='cuda:0')
    '''
    # 2)L-channel consistent initialization:
    L_cls_matrix[:, num_supports:, :num_supports] = 1. / num_supports
    L_cls_matrix[:, :num_supports, num_supports:] = 1. / num_supports
    L_cls_matrix[:, num_supports:, num_supports:] = 0
    # print('L_cls_matrix_norm[0]:', L_cls_matrix[0])
    '''
    L_cls_matrix_norm: tensor([
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
            [0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
            [0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
            [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]], device='cuda:0')
    '''
    for i in range(num_ways * num_queries): 
        L_cls_matrix[:, num_supports + i, num_supports + i] = 1
    # print('L_channel_cls_matrix:', L_channel_cls_matrix.shape) # L_channel_cls_matrix: torch.Size([16, 10, 10])
    # print('L_cls_matrix_final[0]:', L_cls_matrix[0])
    '''
    tensor([
        [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
        [0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
        [0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
        [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
        [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]], device='cuda:0')
    '''
    
    ########################################
    # 3)initialize A-channel clsm
    ########################################
    A_cls_matrix = L_cls_matrix.clone()           
    
    ########################################
    # 4)initialize B-channel clsm
    ########################################
    B_cls_matrix = L_cls_matrix.clone()          

    return support_data, support_label, query_data, query_label, all_data, all_label_in_clsm, \
                 L_cls_matrix, A_cls_matrix, B_cls_matrix
