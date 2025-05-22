# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2025
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os, random, logging, argparse, sys, time
from tqdm import tqdm
import importlib.util
from loader.dataloader import MiniImagenet, TieredImagenet, Cifar, FC100, CUB200, Aircraft, Meta_iNat, Tiered_Meta_iNat, DataLoader
from utils import set_logging_config, adjust_learning_rate, save_checkpoint, allocate_tensors, \
            preprocessing, initialize_emb_clsm, one_hot_encode
from model.CoSeLearner import CoSeLearner
from model.CoSeDistiller import CoSeLearner_stu
# torch.autograd.set_detect_anomaly(True)
from colorama import init, Fore
init()  # Init Colorama

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
    
# CoSeDistiller Trainer
class CoSeDistiller_Trainer(object):
    def __init__(self, teacher_model, student_net, data_loader, log, arg, config, best_step):
        """
           :param teacher network: CoSeLearner
           :param student network: CoSeLearner
           :param data_loader: data loader
           :param log: logger
           :param arg: command line arguments
           :param config: model configurations
           :param best_step: starting step (step at best eval acc or 0 if starts from scratch)
        """
        self.arg = arg
        self.config = config
        self.train_opt = config['train_config']
        self.eval_opt = config['eval_config']
        self.test_opt = config['test_config']
        self.data_loader = data_loader
        self.log = log
        print(f'Using devices: {self.arg.device}')
        self.arg.device = torch.device(f'cuda:{self.arg.device}')
        # initialize variables
        self.tensors = allocate_tensors()
        for key, tensor in self.tensors.items(): 
            self.tensors[key] = tensor.to(self.arg.device)

        ##########################################################
        # 1.Set device:
        ##########################################################
        self.coseteacher = teacher_model.to(arg.device)
        self.coselearner = student_net.to(arg.device)

        ##########################################################
        # 2.Set optimizer and loss:
        ##########################################################
        self.network_params = list(self.coselearner.parameters())
        self.optimizer = optim.Adam(params=self.network_params, lr=self.arg.distill_lr, weight_decay=self.train_opt['weight_decay'])
        self.clsm_loss = nn.BCELoss(reduction='none')
        self.pred_loss = nn.CrossEntropyLoss(reduction='none')
        self.color_loss = DistillKL(4)

        ###########################################################
        # 3.initialize other global variables
        ###########################################################
        self.global_step = best_step
        self.best_step = best_step
        self.val_acc = 0
        self.test_acc = 0

    ##############################################################################
    # Training Function:
    ##############################################################################
    def train(self):
        num_supports, num_samples, query_cls_mask = \
                                    preprocessing(self.train_opt['num_ways'], self.train_opt['num_shots'], \
                                    self.train_opt['num_queries'], self.train_opt['batch_size'], self.arg.device)

        # main training loop, batch size is the number of tasks
        for iteration, batch in tqdm(enumerate(self.data_loader['train']()), desc=f"Training"):
            self.optimizer.zero_grad() # init grad
            self.global_step += 1      # set current step
            
            ############################################################################################################
            # 1.Initialize small sample tasks and classification matrix:
            ############################################################################################################
            support_data, support_label, query_data, query_label, all_data, all_label_in_clsm, \
            L_cls_matrix, A_cls_matrix, B_cls_matrix = initialize_emb_clsm(batch,
                                                                        num_supports,                  
                                                                        self.tensors,
                                                                        self.train_opt['batch_size'],  
                                                                        self.train_opt['num_queries'], 
                                                                        self.train_opt['num_ways'],   
                                                                        self.arg.device)

            ############################################################################################################
            # 2.CoSeLearner forward propagation:
            ############################################################################################################
            self.coselearner.train()
            L_cls_similarity, L_emb_similarity, A_cls_similarity, B_cls_similarity = \
                                                 self.coselearner(all_data, L_cls_matrix, A_cls_matrix, B_cls_matrix)
            with torch.no_grad(): 
                L_cls_sim_teacher, L_emb_sim_teacher, A_cls_sim_teacher, B_cls_sim_teacher = \
                                                 self.coseteacher(all_data, L_cls_matrix, A_cls_matrix, B_cls_matrix)
            
            ############################################################################################################
            # 3.Calculate similarity and loss values:
            ############################################################################################################
            query_acc_gens, loss_cls, query_cls_loss_gens = \
                                                        self.compute_train_loss_pred(all_label_in_clsm,       
                                                                                        L_cls_similarity,     
                                                                                        A_cls_similarity,    
                                                                                        B_cls_similarity,                                                                                            
                                                                                        L_emb_similarity,     
                                                                                        query_cls_mask,       
                                                                                        num_supports, support_label, query_label)
                                                        
            loss_color = self.color_loss(L_emb_similarity[-1], L_emb_sim_teacher[-1])
            total_loss = loss_cls + self.arg.color_factor * loss_color         

            #############################################################################################################
            # 4.Backpropagation:
            #############################################################################################################
            total_loss.backward()
            self.optimizer.step()
            # adjust learning rate
            adjust_learning_rate(optimizers=[self.optimizer],  lr=self.train_opt['lr'], iteration=self.global_step,
                                 dec_lr_step=self.train_opt['dec_lr'], lr_adj_base =self.train_opt['lr_adj_base'])
            # log training info
            if self.global_step % self.arg.log_step == 0:
                self.log.info('step : {}  cls_loss : {:8f}  color_loss : {:8f} query_acc : {}'.format(self.global_step,
                                                                                      query_cls_loss_gens[-1],
                                                                                      self.arg.color_factor*loss_color,
                                                                                      query_acc_gens[-1]))
            
            ##############################################################################################################
            # 5.evaluation
            ##############################################################################################################
            if self.global_step % self.eval_opt['interval'] == 0:
                is_best = 0
                test_acc = self.eval(partition='val')
                if test_acc > self.test_acc:
                    is_best = 1
                    self.test_acc = test_acc
                    self.best_step = self.global_step

                # log evaluation info
                self.log.info('test_acc : {}         step : {} '.format(test_acc, self.global_step))
                self.log.info('test_best_acc : {}    step : {}'.format( self.test_acc, self.best_step))

                # save checkpoints (best and newest)
                save_checkpoint({
                    'iteration': self.global_step,
                    'coselearner_state_dict': self.coselearner.state_dict(),
                    'test_acc': self.test_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, os.path.join('result', self.config['dataset_name'], self.arg.checkpoint_dir, self.arg.exp_name+str('_{}stu-gen'.format(self.arg.gen_stu))))
    # Training loss and predicted values:
    def compute_train_loss_pred(self, all_label_in_clsm,   
                                L_clsm_similarity,        
                                A_clsm_similarity,        
                                B_clsm_similarity,        
                                L_emb_similarity,         
                                query_cls_mask,           
                                num_supports, support_label, query_label):

        ##########################################
        # 1.Calculate predicted values
        ##########################################
        query_pred_gens = [
            torch.bmm(clsm_similarity[:, num_supports:, :num_supports], one_hot_encode(self.train_opt['num_ways'], support_label.long(), self.arg.device))
            for clsm_similarity in L_clsm_similarity]     
        query_acc_gens = [
            torch.eq(torch.max(query_pred_gen, -1)[1], query_label.long()).float().mean()
            for query_pred_gen in query_pred_gens]
        
        ##########################################
        # 2.cls loss
        ##########################################
        # L Loss
        total_clsm_loss_gens_L = [self.clsm_loss((1 - clsm_similarity), (1 - all_label_in_clsm))
                                            for clsm_similarity in L_clsm_similarity]
        # A Loss
        total_clsm_loss_gens_A = [self.clsm_loss((1 - A_similarity), (1 - all_label_in_clsm))
                                            for A_similarity in A_clsm_similarity]
        # B Loss
        total_clsm_loss_gens_B = [self.clsm_loss((1 - B_similarity), (1 - all_label_in_clsm))
                                            for B_similarity in B_clsm_similarity]
        # Combine L, A, and B losses (weighted)
        color_loss_coeff = 0.1
        total_clsm_loss_gens = [total_clsm_loss_L + color_loss_coeff * (total_clsm_loss_A + total_clsm_loss_B)
        for (total_clsm_loss_L, total_clsm_loss_A, total_clsm_loss_B) in zip(total_clsm_loss_gens_L, total_clsm_loss_gens_A, total_clsm_loss_gens_B)]
        # 1) positive query
        pos_query_clsm_loss_gens = [
            torch.sum(total_clsm_loss_generation * query_cls_mask * all_label_in_clsm)
            / torch.sum(query_cls_mask * all_label_in_clsm)
            for total_clsm_loss_generation in total_clsm_loss_gens]      
        # 2) negtive query
        neg_query_clsm_loss_gens = [
            torch.sum(total_clsm_loss_generation * query_cls_mask * (1 - all_label_in_clsm))
            / torch.sum(query_cls_mask * (1 - all_label_in_clsm))
            for total_clsm_loss_generation in total_clsm_loss_gens]
        # 3) weighted clsm loss for balancing pos/neg
        query_clsm_loss_gens = [
            pos_query_clsm_loss_generation + neg_query_clsm_loss_generation
            for (pos_query_clsm_loss_generation, neg_query_clsm_loss_generation) in zip(pos_query_clsm_loss_gens, neg_query_clsm_loss_gens)]

        ##########################
        # 3.embedding loss
        ##########################
        query_emb_pred_gens_ = [
            torch.bmm(emb_similarity[:, num_supports:, :num_supports], one_hot_encode(self.train_opt['num_ways'], support_label.long(), self.arg.device))
            for emb_similarity in L_emb_similarity]
        query_emb_pred_loss = [
            self.pred_loss(query_emb_pred_gen.view(-1, query_emb_pred_gen.size(-1)), query_label.view(-1)) \
            for query_emb_pred_gen in query_emb_pred_gens_ ]
        
        ##########################
        # 4.each loss of gens
        ##########################
        total_loss_gens = [
            query_clsm_loss_generation + 0.1 * query_emb_pred_loss_
            for (query_clsm_loss_generation, query_emb_pred_loss_) in zip(query_clsm_loss_gens, query_emb_pred_loss)]
        total_loss = []

        # Use generation weight or not: no use by default
        usage=False
        num_gen = self.arg.gen_stu
        if usage:
            if num_gen>1:
                total_loss = [total_loss_gens[i].view(-1) * self.config['generation_weight']
                    for i in range(num_gen - 1)]
                total_loss += [total_loss_gens[-1].view(-1) * 1.0]
            else: 
                total_loss += [total_loss_gens[-1].view(-1) * 1.0]
        else:
            if num_gen>1:
                total_loss = [total_loss_gens[i].view(-1) for i in range(num_gen)]
            else: 
                total_loss += [total_loss_gens[-1].view(-1) * 1.0]
                
        total_loss = torch.mean(torch.cat(total_loss, 0))
        
        return query_acc_gens, total_loss, query_clsm_loss_gens

    ##############################################################################
    # evaluation function:
    ##############################################################################
    def eval(self, partition='test', log_flag=True):
        """ evaluation function
        :param partition: which part of data is used
        :param log_flag: if log the evaluation info
        :return: None
        """
        if partition=='val': 
            iteration= self.eval_opt['iteration']
            num_supports, num_samples, query_cls_mask = preprocessing(
                self.eval_opt['num_ways'],
                self.eval_opt['num_shots'],
                self.eval_opt['num_queries'],
                self.eval_opt['batch_size'],
                self.arg.device)
        elif partition=='test': 
            iteration= self.test_opt['iteration']        
            num_supports, num_samples, query_cls_mask = preprocessing(
                self.test_opt['num_ways'],
                self.test_opt['num_shots'],
                self.test_opt['num_queries'],
                self.test_opt['batch_size'],
                self.arg.device)
        query_cls_loss_gens = []
        query_emb_acc_gens = []

        # main training loop, batch size is the number of tasks
        for current_iteration, batch in tqdm(enumerate(self.data_loader[partition]()), desc=f"Testing on {partition}({iteration}it)"):
            
            ############################################################################################################
            # 1.Initialization task
            ############################################################################################################
            support_data, support_label, query_data, query_label, all_data, all_label_in_clsm, \
            L_cls_matrix, A_cls_matrix, B_cls_matrix = initialize_emb_clsm(batch,
                                                                            num_supports,
                                                                            self.tensors,
                                                                            self.eval_opt['batch_size'],
                                                                            self.eval_opt['num_queries'],
                                                                            self.eval_opt['num_ways'],
                                                                            self.arg.device)

            ############################################################################################################
            # 2.set as train mode
            ############################################################################################################
            self.coselearner.train()
            L_cls_similarity, _, _, _  = self.coselearner(all_data, L_cls_matrix, A_cls_matrix, B_cls_matrix)

            ############################################################################################################
            # 3.Calculate similarity and loss values
            ############################################################################################################
            query_emb_acc_gens, query_cls_loss_gens = \
                                        self.compute_eval_loss_pred(query_cls_loss_gens,
                                                                    query_emb_acc_gens,
                                                                    all_label_in_clsm,
                                                                    L_cls_similarity,
                                                                    query_cls_mask,
                                                                    num_supports,
                                                                    support_label,query_label)

        # logging
        if log_flag:
            self.log.info('------------------------------------')
            self.log.info('step : {}  {}_cls_loss : {}  {}_emb_acc : {}'.format(
                self.global_step, partition, np.array(query_cls_loss_gens).mean(),
                partition, np.array(query_emb_acc_gens).mean()))

            self.log.info('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                          (current_iteration,
                           np.array(query_emb_acc_gens).mean() * 100,
                           np.array(query_emb_acc_gens).std() * 100,
                           1.96 * np.array(query_emb_acc_gens).std() / np.sqrt(float(len(np.array(query_emb_acc_gens)))) * 100))
            self.log.info('------------------------------------')

        return np.array(query_emb_acc_gens).mean()

    # Validate loss and predictions:
    def compute_eval_loss_pred(self,
                               query_clsm_losses, query_accs,
                               all_label_in_clsm,
                               L_clsm_similarity,
                               query_clsm_mask,
                               num_supports, support_label, query_label):

        #############################################################################################
        # 1.Calculate the predicted value of the query and its similarity
        #############################################################################################
        clsm_similarity = L_clsm_similarity[-1]
        # prediction
        query_emb_pred = torch.bmm(
            clsm_similarity[:, num_supports:, :num_supports],
            one_hot_encode(self.eval_opt['num_ways'], support_label.long(), self.arg.device))
        # test accuracy
        query_acc = torch.eq(torch.max(query_emb_pred, -1)[1], query_label.long()).float().mean()
        query_accs += [query_acc.item()]

        #############################################################################################
        # 2.Calculate query CLSM loss:
        #############################################################################################
        full_clsm_loss = self.clsm_loss(1 - clsm_similarity, 1 - all_label_in_clsm)
        pos_query_clsm_loss = torch.sum(full_clsm_loss * query_clsm_mask * all_label_in_clsm) / torch.sum(
            query_clsm_mask * all_label_in_clsm)
        neg_query_clsm_loss = torch.sum(full_clsm_loss * query_clsm_mask * (1 - all_label_in_clsm)) / torch.sum(
            query_clsm_mask * (1 - all_label_in_clsm))
        # weighted loss for balancing pos/neg
        query_clsm_loss = pos_query_clsm_loss + neg_query_clsm_loss
        query_clsm_losses += [query_clsm_loss.item()]

        return query_accs, query_clsm_losses


##################
# main function
##################
def main():
    ##########################################################################################################################################
    # 1.Hyperparameters:
    ##########################################################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='device ID of gpu')
    parser.add_argument('--dataset_root', type=str, default='/home/ssdData/qcfData/Benchmark_FewShot', help='root directory of dataset')
    parser.add_argument('--config', type=str, default=os.path.join('.', 'config', '5way_1shot_resnet12_mini-imagenet.py'),
                        help='config file with parameters of the experiment. It is assumed that the config file is placed under the directory ./config')
    parser.add_argument('--checkpoint_teacher', type=str, default=os.path.join('.', 'checkpoints_meta'),
                        help='path that checkpoint will be saved and loaded. It is assumed that the checkpoint file is placed under the directory ./checkpoints_meta')
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('.', 'checkpoints_distill'),
                        help='path that checkpoint will be saved and loaded. It is assumed that the checkpoint file is placed under the directory ./checkpoints_distill')
    parser.add_argument('--display_step', type=int, default=100, help='display training information in how many step')
    parser.add_argument('--log_step', type=int, default=5, help='log information in how many steps')
    parser.add_argument('--log_dir', type=str, default=os.path.join('.', 'logs_distill'), 
                        help='path that log will be saved. It is assumed that the checkpoint file is placed under the directory ./logs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    parser.add_argument('--gen_stu', type=int, default=3, help='generation of student')
    parser.add_argument('--color_factor', type=int, default=1e-4, help='coefficient of color loss in total loss')
    parser.add_argument('--distill_lr', type=int, default=0.005, help='KD learning rate during CoSeDistiller training')
    args_opt = parser.parse_args()
    config_file = args_opt.config

    # Set train and test datasets and the corresponding data  
    print(f'Using devices: {args_opt.device}')
    device = torch.device(f'cuda:{args_opt.device}')
    
    # Load module
    spec = importlib.util.spec_from_file_location("config_module", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Obtain config
    config = config_module.config
    train_opt = config['train_config']
    eval_opt = config['eval_config']
    test_opt = config['test_config']
    # args_opt.exp_name = '{}way_{}shot_{}query_{}'.format(train_opt['num_ways'], train_opt['num_shots'], train_opt['num_queries'], config['dataset_name'])
    args_opt.exp_name = '{}way_{}shot_{}query_{}_{}gen'.format(train_opt['num_ways'], train_opt['num_shots'], train_opt['num_queries'], 
                                                               config['dataset_name'], config['num_generation'])
    # set_logging_config(os.path.join(args_opt.log_dir, args_opt.exp_name))
    set_logging_config(os.path.join('result', config['dataset_name'], args_opt.log_dir, args_opt.exp_name+str('_{}stu-gen'.format(args_opt.gen_stu))))
    logger = logging.getLogger('main')

    # Load the configuration params of the experiment
    logger.info('Launching experiment from: {}'.format(config_file))
    logger.info('Generated logs will be saved to: {}'.format(args_opt.log_dir))
    logger.info('Generated checkpoints will be saved to: {}'.format(args_opt.checkpoint_dir))
    print()

    logger.info('-------------command line arguments-------------')
    logger.info(args_opt)
    print()
    logger.info('-------------configs-------------')
    logger.info(config)

    # set random seed
    np.random.seed(args_opt.seed)
    torch.manual_seed(args_opt.seed)
    torch.cuda.manual_seed_all(args_opt.seed)
    random.seed(args_opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    ######################################################################################################################################
    # 2.Load the datasets
    ######################################################################################################################################
    # 1)Coarse-grained datasets
    if config['dataset_name'] == 'mini-imagenet':
        dataset = MiniImagenet
        print('Dataset: MiniImagenet')
    elif config['dataset_name'] == 'tiered-imagenet':
        dataset = TieredImagenet
        print('Dataset: TieredImagenet')
    elif config['dataset_name'] == 'cifar-fs':
        dataset = Cifar
        print('Dataset: Cifar')
    elif config['dataset_name'] == 'fc100':
        dataset = FC100
        print('Dataset: FC100')
    # 2)fine-grained datasets
    elif config['dataset_name'] == 'cub-200-2011':
        dataset = CUB200
        print('Dataset: CUB200-200-2011')
    elif config['dataset_name'] == 'aircraft-fs':
        dataset = Aircraft
        print('Dataset: Aircraft-Fewshot')
    elif config['dataset_name'] == 'meta-iNat':
        dataset = Meta_iNat
        print('Dataset: Meta-iNat')
    elif config['dataset_name'] == 'tiered-meta-iNat':
        dataset = Tiered_Meta_iNat
        print('Dataset: Tiered-Meta-iNat')
    else:
        logger.info('Invalid dataset: {}, please specify a dataset from mini-imagenet, tiered-imagenet, \
        cifar-fs, cub-200-2011, meta-iNat and tiered-meta-iNat.'.format(config['dataset_name']))
        exit()
    # 3) Load the datasets
    dataset_train = dataset(root=args_opt.dataset_root, partition='train')
    dataset_valid = dataset(root=args_opt.dataset_root, partition='val')
    dataset_test = dataset(root=args_opt.dataset_root, partition='test')
    train_loader = DataLoader(dataset_train,num_tasks=train_opt['batch_size'],
                              num_ways=train_opt['num_ways'],
                              num_shots=train_opt['num_shots'],
                              num_queries=train_opt['num_queries'],
                              epoch_size=train_opt['iteration'])
    valid_loader = DataLoader(dataset_valid,num_tasks=eval_opt['batch_size'],
                              num_ways=eval_opt['num_ways'],
                              num_shots=eval_opt['num_shots'],
                              num_queries=eval_opt['num_queries'],
                              epoch_size=eval_opt['iteration'])
    test_loader = DataLoader(dataset_test,num_tasks=test_opt['batch_size'],
                              num_ways=test_opt['num_ways'],
                              num_shots=test_opt['num_shots'],
                              num_queries=test_opt['num_queries'],
                              epoch_size=test_opt['iteration'])
    data_loader = {'train': train_loader, 'val': valid_loader, 'test': test_loader}


    ######################################################################################################################################
    # 3.Instantiate-teacher model and student network
    ######################################################################################################################################
    encoder_flag = True if args_opt.exp_name.__contains__('cifar') or args_opt.exp_name.__contains__('fc100') else False
    # 1) Establish teacher networks and student networks:
    if config['backbone'] == 'coselearner':
        cose_teacher = CoSeLearner(encoder_flag, config['emb_size'], config['num_generation'], train_opt['dropout'],
                       train_opt['num_ways'] * train_opt['num_shots'],
                       train_opt['num_ways'] * train_opt['num_shots'] + train_opt['num_ways'] * train_opt['num_queries'],
                       train_opt['loss_indicator'], config['distance_metric']).to(device)
        cose_student = CoSeLearner_stu(encoder_flag, config['emb_size'], args_opt.gen_stu, train_opt['dropout'],
                       train_opt['num_ways'] * train_opt['num_shots'],
                       train_opt['num_ways'] * train_opt['num_shots'] + train_opt['num_ways'] * train_opt['num_queries'],
                       train_opt['loss_indicator'], config['distance_metric']).to(device)
        print('Backbone: CoSeLearner')
    else:
        logger.info('Invalid backbone: {}, please specify another backbone model.'.format(config['backbone']))
        exit()
    
    # 2) Load the pre-trained model (teacher network + student network)
    # [1]teacher network loads the pre-trained model:
    try:
        teacher_checkpoint = torch.load(os.path.join('result', config['dataset_name'], args_opt.checkpoint_teacher, args_opt.exp_name, 'model_best.pth.tar'))
        # print(str(os.path.join('result', config['dataset_name'], args_opt.checkpoint_teacher, args_opt.exp_name)))
        logger.info('Teacher model pack loaded')
        best_step = teacher_checkpoint['iteration']
        cose_teacher.load_state_dict(teacher_checkpoint['coselearner_state_dict'])
        logger.info('best validation accuracy of Teacher is: {}, at step: {}'.format(teacher_checkpoint['test_acc'], best_step))
    except Exception as e:
        print(Fore.BLUE+"Specify a directory of pre-trained models for the teacher network!")
        sys.exit()
    
    # [2]Students load pre-trained models on the network:
    if not os.path.exists(os.path.join('result', config['dataset_name'], args_opt.checkpoint_dir, args_opt.exp_name+str('_{}stu-gen'.format(args_opt.gen_stu)))):
        os.makedirs(os.path.join('result', config['dataset_name'], args_opt.checkpoint_dir, args_opt.exp_name+str('_{}stu-gen'.format(args_opt.gen_stu))))
        logger.info('no checkpoint for model: {}, make a new one at {}'.format(args_opt.exp_name, os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)))
        best_step = 0
    else:
        if not os.path.exists(os.path.join('result', config['dataset_name'], args_opt.checkpoint_dir, 
                                           args_opt.exp_name+str('_{}stu-gen'.format(args_opt.gen_stu)), 'model_best.pth.tar')): best_step = 0
        else:
            logger.info('find a student checkpoint, loading checkpoint from {}'.format(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)))
            best_checkpoint = torch.load(os.path.join('result', config['dataset_name'], args_opt.checkpoint_dir, 
                                                      args_opt.exp_name+str('_{}stu-gen'.format(args_opt.gen_stu)), 'model_best.pth.tar'))
            logger.info('Student model pack loaded')
            best_step = best_checkpoint['iteration']
            cose_student.load_state_dict(best_checkpoint['coselearner_state_dict'])
            logger.info('current best validation accuracy of Student is: {}, at step: {}'.format(best_checkpoint['test_acc'], best_step))
    
    
    ######################################################################################################################################
    # 4.Perform network training and validation
    ######################################################################################################################################
    trainer = CoSeDistiller_Trainer(teacher_model=cose_teacher, student_net=cose_student, data_loader=data_loader, \
                                                    log=logger, arg=args_opt, config=config, best_step=best_step)

    if args_opt.mode == 'train': trainer.train()
    elif args_opt.mode == 'eval': trainer.eval()
    else:
        print('select a mode')
        exit()


if __name__ == '__main__':
    main()