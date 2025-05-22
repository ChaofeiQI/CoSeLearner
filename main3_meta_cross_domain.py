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
import numpy as np
import os, random, logging, argparse, time
from tqdm import tqdm 
import importlib.util
from loader.dataloader import Places365, Stanford_Car, CropDisease, EuroSAT, DataLoader
from utils import set_logging_config, adjust_learning_rate, save_checkpoint, allocate_tensors, \
            preprocessing, initialize_emb_clsm, one_hot_encode
from model.CoSeLearner import CoSeLearner
# torch.autograd.set_detect_anomaly(True)

# CoSeLearner Trainer
class CoSeLearner_Trainer(object):
    def __init__(self, network, data_loader, log, arg, config, best_step):
        """:param network: CoSeLearner
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
        # 1.set device:
        ##########################################################
        self.coselearner = network.to(arg.device)

        ##########################################################
        # 2.set optimizer and loss:
        ##########################################################
        self.network_params = list(self.coselearner.parameters())
        self.optimizer = optim.Adam(params=self.network_params, lr=self.train_opt['lr'], weight_decay=self.train_opt['weight_decay'])
        self.clsm_loss = nn.BCELoss(reduction='none')
        self.pred_loss = nn.CrossEntropyLoss(reduction='none')

        ###########################################################
        # 3.initialize other global variables
        ###########################################################
        self.global_step = best_step
        self.best_step = best_step
        self.val_acc = 0
        self.test_acc = 0

    ##############################################################################
    # Verification function:
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
        query_cls_loss_generations = []
        query_emb_acc_generations = []

        # main training loop, batch size is the number of tasks
        for current_iteration, batch in tqdm(enumerate(self.data_loader[partition]()), desc=f"Testing on {partition}({iteration}it)"):
            
            ############################################################################################################
            # 1.Initialize small sample tasks and classification matrix:
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
            query_emb_acc_generations, query_cls_loss_generations = \
                                        self.compute_eval_loss_pred(query_cls_loss_generations,
                                                                    query_emb_acc_generations,
                                                                    all_label_in_clsm,
                                                                    L_cls_similarity,
                                                                    query_cls_mask,
                                                                    num_supports,
                                                                    support_label,query_label)

        # logging
        if log_flag:
            self.log.info('------------------------------------')
            self.log.info('step : {}  {}_cls_loss : {}  {}_emb_acc : {}'.format(
                self.global_step, partition, np.array(query_cls_loss_generations).mean(),
                partition, np.array(query_emb_acc_generations).mean()))

            self.log.info('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                          (current_iteration,
                           np.array(query_emb_acc_generations).mean() * 100,
                           np.array(query_emb_acc_generations).std() * 100,
                           1.96 * np.array(query_emb_acc_generations).std() / np.sqrt(float(len(np.array(query_emb_acc_generations)))) * 100))
            self.log.info('------------------------------------')

        return np.array(query_emb_acc_generations).mean()

    # Validate loss and predictions:
    def compute_eval_loss_pred(self,
                               query_clsm_losses, query_emb_accs,
                               all_label_in_clsm,
                               L_clsm_similarity,
                               query_clsm_mask,
                               num_supports, support_label, query_label):

        #############################################################################################
        # 1.Calculate the predicted values of the query instances and their similarity
        #############################################################################################
        clsm_similarity = L_clsm_similarity[-1]
        # prediction
        query_emb_pred = torch.bmm(
            clsm_similarity[:, num_supports:, :num_supports],
            one_hot_encode(self.eval_opt['num_ways'], support_label.long(), self.arg.device))
        # test accuracy
        query_emb_acc = torch.eq(torch.max(query_emb_pred, -1)[1], query_label.long()).float().mean()
        query_emb_accs += [query_emb_acc.item()]

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

        return query_emb_accs, query_clsm_losses


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
    # parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('.', 'results/checkpoints_meta'),
    #                     help='path that checkpoint will be saved and loaded. It is assumed that the checkpoint file is placed under the directory ./checkpoints_meta')
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('.', 'checkpoints_meta'),
                        help='path that checkpoint will be saved and loaded. It is assumed that the checkpoint file is placed under the directory ./checkpoints_meta')
    parser.add_argument('--display_step', type=int, default=100, help='display training information in how many step')
    parser.add_argument('--log_step', type=int, default=5, help='log information in how many steps')
    # parser.add_argument('--log_dir', type=str, default=os.path.join('.', 'results/logs_meta'), 
    #                     help='path that log will be saved. It is assumed that the checkpoint file is placed under the directory ./logs_meta')
    parser.add_argument('--log_dir', type=str, default=os.path.join('.', 'logs_meta_cross_domain'), 
                        help='path that log will be saved. It is assumed that the checkpoint file is placed under the directory ./logs_meta')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    args_opt = parser.parse_args()
    config_file = args_opt.config

    # Set train and test datasets and the corresponding data loaders
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
    args_opt.exp_name = '{}way_{}shot_{}query_{}'.format(train_opt['num_ways'], train_opt['num_shots'], train_opt['num_queries'], config['dataset_name'])
    set_logging_config(os.path.join('result', config['dataset_name'], args_opt.log_dir, args_opt.exp_name+str('_{}gen'.format(config['num_generation']))))
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
    # 1)Cross-domain datasets
    if config['dataset_name'] == 'places':
        dataset = Places365
        print('Dataset: Places365')
    elif config['dataset_name'] == 'cars':
        dataset = Stanford_Car
        print('Dataset: Stanford_Car')
    elif config['dataset_name'] == 'CropDisease':
        dataset = CropDisease
        print('Dataset: CropDisease')
    elif config['dataset_name'] == 'EuroSAT':
        dataset = EuroSAT
        print('Dataset: EuroSAT')        
    else:
        logger.info('Invalid dataset: {}, please specify a dataset from places, cars, \
        CropDisease and EuroSAT.'.format(config['dataset_name']))
        exit()
    # 2)Load the datasets
    dataset_test = dataset(root=args_opt.dataset_root, partition='test')
    test_loader = DataLoader(dataset_test,
                             num_tasks=test_opt['batch_size'],
                             num_ways=test_opt['num_ways'],
                             num_shots=test_opt['num_shots'],
                             num_queries=test_opt['num_queries'],
                             epoch_size=test_opt['iteration'])

    data_loader = {'test': test_loader}


    ######################################################################################################################################
    # 3.Instantiate CoSeLearner
    ######################################################################################################################################
    encoder_flag = True if args_opt.exp_name.__contains__('cifar') or args_opt.exp_name.__contains__('fc100') else False
    
    # 1) Instantiate CoSeLearner:
    if config['backbone'] == 'coselearner':
        coselearner = CoSeLearner(encoder_flag, config['emb_size'], config['num_generation'], train_opt['dropout'],
                      train_opt['num_ways'] * train_opt['num_shots'],
                      train_opt['num_ways'] * train_opt['num_shots'] + train_opt['num_ways'] * train_opt['num_queries'],
                      train_opt['loss_indicator'], config['distance_metric']).to(device)
        print('Backbone: CoSeLearner')
    else:
        logger.info('Invalid backbone: {}, please specify another backbone model.'.format(config['backbone']))
        exit()
    
    # 2) Load pre-trained model:
    if not os.path.exists(os.path.join('result', config['dataset_name'], args_opt.checkpoint_dir, args_opt.exp_name+str('_{}gen'.format(config['num_generation'])))):
        os.makedirs(os.path.join('result', config['dataset_name'],args_opt.checkpoint_dir, args_opt.exp_name+str('_{}gen'.format(config['num_generation']))))
        logger.info('no checkpoint for model: {}, make a new one at {}'.format(args_opt.exp_name, os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)))
        best_step = 0
    else:
        if not os.path.exists(os.path.join('result', config['dataset_name'], args_opt.checkpoint_dir, 
                                                    args_opt.exp_name+str('_{}gen'.format(config['num_generation'])), 'model_best.pth.tar')): best_step = 0
        else:
            logger.info('find a checkpoint, loading checkpoint from {}'.format(os.path.join(
                'result', config['dataset_name'], args_opt.checkpoint_dir, args_opt.exp_name)))
            best_checkpoint = torch.load(os.path.join('result', config['dataset_name'], args_opt.checkpoint_dir, 
                                                    args_opt.exp_name+str('_{}gen'.format(config['num_generation'])), 'model_best.pth.tar'))
            logger.info('best model pack loaded')
            best_step = best_checkpoint['iteration']
            coselearner.load_state_dict(best_checkpoint['coselearner_state_dict'])
            logger.info('current best validation accuracy is: {}, at step: {}'.format(best_checkpoint['test_acc'], best_step))
    
    
    ######################################################################################################################################
    # 4.Construct CoSeLearner and execute network training and validation
    ######################################################################################################################################
    trainer = CoSeLearner_Trainer(network=coselearner, data_loader=data_loader, log=logger,arg=args_opt, config=config, best_step=best_step)

    if args_opt.mode == 'eval': trainer.eval()
    else:
        print('select a mode')
        exit()


if __name__ == '__main__':
    main()