from collections import OrderedDict

num_query=15
#######################################
# Base超参设置
#######################################
config = OrderedDict()
config['dataset_name'] = 'cub-200-2011'
config['num_generation'] = 5
config['num_loss_generation'] = 3
config['generation_weight'] = 0.5
config['distance_metric'] = 'l1'
config['emb_size'] = 128
config['backbone'] = 'coselearner'


#######################################
# 训练阶段超参
#######################################
train_opt = OrderedDict()
train_opt['num_ways'] = 5
train_opt['num_shots'] = 1
train_opt['num_queries'] = num_query   
train_opt['batch_size'] = 4
train_opt['iteration'] = 2000
train_opt['lr'] = 1e-3
train_opt['weight_decay'] = 1e-6
train_opt['dec_lr'] = 1000
train_opt['lr_adj_base'] = 0.5
train_opt['dropout'] = 0.1
train_opt['loss_indicator'] = [1, 1, 1, 0] 


#######################################
# 验证阶段超参
#######################################
eval_opt = OrderedDict()
eval_opt['num_ways'] = 5
eval_opt['num_shots'] = 1
eval_opt['num_queries'] = num_query    
eval_opt['batch_size'] = 4
eval_opt['iteration'] = 200
eval_opt['interval'] = 40


#######################################
# 测试阶段超参
#######################################
test_opt = OrderedDict()
test_opt['num_ways'] = 5
test_opt['num_shots'] = 1
test_opt['num_queries'] = num_query    
test_opt['batch_size'] = 4
test_opt['iteration'] = 5000


###################################
config['train_config'] = train_opt
config['eval_config'] = eval_opt
config['test_config'] = test_opt
###################################