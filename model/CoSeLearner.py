# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2025
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch, time
import numpy as np
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from e2cnn import gspaces   
from e2cnn import nn as enn
import warnings
from functools import partial
from joblib import Parallel, delayed                                    
from pathos.multiprocessing import ProcessingPool as Pool               
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor 
from colorama import init, Fore
init()  # Init Colorama
warnings.filterwarnings("ignore")

################
# 1.Color Shunt
################
def Tensor_rgb_to_lab(image):
    # non negative
    image = torch.clamp(image, 0, None)
    image = torch.where(image > 0.0031308, 1.055 * (image ** (1/2.4)) - 0.055, 12.92 * image)
    image *= 100.0
    # RGB to XYZ
    mat = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]], device=image.device)    
    xyz = torch.matmul(image, mat.T)
    xyz = torch.clamp(xyz, min=0)
    # XYZ to LAB
    xyz_ref = torch.tensor([95.047, 100.000, 108.883], device=image.device)
    xyz /= xyz_ref
    xyz = torch.where(xyz > 0.008856, xyz ** (1/3), (xyz * 7.787) + (16/116))
    l_c = 116 * xyz[..., 1] - 16
    a_c = 500 * (xyz[..., 0] - xyz[..., 1])
    b_c = 200 * (xyz[..., 1] - xyz[..., 2])
    # Normalize to [-1, 1]
    l_c = (l_c / 100) * 2 - 1
    a_c = (a_c + 88) / 176 * 2 - 1
    b_c = (b_c + 107) / 214 * 2 - 1
    # channels stack    
    lab = torch.stack([l_c, a_c, b_c], dim=-1)
    return lab
def Tensor_rgb_to_hsv(image):
    # Ensure that the pixel values of the input image are within the range [0, 1].
    image = torch.clamp(image, 0, 1)
    # Separate RGB channels
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    # Find the maximum and minimum values
    max_val = torch.max(image, dim=-1)[0]
    min_val = torch.min(image, dim=-1)[0]
    delta = max_val - min_val
    # Calculate H (hue)
    h = torch.zeros_like(max_val)
    # Calculate hue when delta is not zero
    non_zero_delta = delta > 0
    h[non_zero_delta] = torch.where(
        max_val[non_zero_delta] == r[non_zero_delta],
        ((g[non_zero_delta] - b[non_zero_delta]) / delta[non_zero_delta]) % 6,
        torch.where(
            max_val[non_zero_delta] == g[non_zero_delta],
            ((b[non_zero_delta] - r[non_zero_delta]) / delta[non_zero_delta]) + 2,
            ((r[non_zero_delta] - g[non_zero_delta]) / delta[non_zero_delta]) + 4
        ))
    h[non_zero_delta] = h[non_zero_delta] / 6.0
    # Calculate S (saturation)
    s = torch.where(max_val > 0, delta / max_val, torch.zeros_like(max_val))
    # Calculate V (brightness)
    v = max_val
    # Normalize to the range [-1, 1]
    h = h * 2 - 1
    s = s * 2 - 1
    v = v * 2 - 1
    # Channel stacking
    hsv = torch.stack([h, s, v], dim=-1)
    return hsv
def Tensor_rgb_to_hsl(image):
    # Ensure that the image pixel values are within the range [0, 1].
    image = torch.clamp(image, 0, 1)
    # Separate RGB channels
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    # Find the maximum and minimum values
    max_val = torch.max(image, dim=-1)[0]
    min_val = torch.min(image, dim=-1)[0]
    delta = max_val - min_val
    # Calculate H (hue)
    h = torch.zeros_like(max_val)
    non_zero_delta = delta > 0
    h[non_zero_delta] = torch.where(
        max_val[non_zero_delta] == r[non_zero_delta],
        ((g[non_zero_delta] - b[non_zero_delta]) / delta[non_zero_delta]) % 6,
        torch.where(
            max_val[non_zero_delta] == g[non_zero_delta],
            ((b[non_zero_delta] - r[non_zero_delta]) / delta[non_zero_delta]) + 2,
            ((r[non_zero_delta] - g[non_zero_delta]) / delta[non_zero_delta]) + 4
        ))
    h[non_zero_delta] = h[non_zero_delta] / 6.0
    # Calculate L (brightness)
    l = (max_val + min_val) / 2
    # Calculate S (saturation)
    s = torch.where(delta > 0,
                    delta / (1 - torch.abs(2 * l - 1)),
                    torch.zeros_like(l))
    # Normalize to the range [-1, 1]
    h = h * 2 - 1
    s = s * 2 - 1
    l = l * 2 - 1
    # Channel stacking
    hsl = torch.stack([h, s, l], dim=-1)
    return hsl
def Tensor_rgb_to_yuv(image):
    # Ensure that the image pixel values are within the range of [0, 1].
    image = torch.clamp(image, 0, 1)
    # The conversion matrix from RGB to YUV
    mat = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001]
    ], device=image.device)
    # matrix multiplication
    yuv = torch.matmul(image, mat.T)
    # Normalize to the range [-1, 1]
    yuv[..., 0] = (yuv[..., 0] / 1.0) * 2 - 1    # Y [0, 1]
    yuv[..., 1] = (yuv[..., 1] / 0.436) * 2 - 1  # U [-0.436, 0.436]
    yuv[..., 2] = (yuv[..., 2] / 0.615) * 2 - 1  # V [-0.615, 0.615]
    return yuv
def ColorConversion(image_batch, mode='CIELab'):
    image_batch_rgb = (image_batch + 3) / 6 
    # We offer a variety of color space strategies
    with torch.no_grad():
        if mode == 'CIELab':
            image_batch_lab = Tensor_rgb_to_lab(image_batch_rgb.permute(0, 2, 3, 1))
            return image_batch_lab.permute(0, 3, 1, 2).to(image_batch.device)
        elif mode == 'HSV':
            image_batch_hsv = Tensor_rgb_to_hsv(image_batch_rgb.permute(0, 2, 3, 1))
            return image_batch_hsv.permute(0, 3, 1, 2).to(image_batch.device)            
        elif mode == 'HSL':
            image_batch_hsl = Tensor_rgb_to_hsl(image_batch_rgb.permute(0, 2, 3, 1))
            return image_batch_hsl.permute(0, 3, 1, 2).to(image_batch.device)   
        elif mode == 'YUV':
            image_batch_yuv = Tensor_rgb_to_yuv(image_batch_rgb.permute(0, 2, 3, 1))
            return image_batch_yuv.permute(0, 3, 1, 2).to(image_batch.device)  
        else: 
            raise ValueError("Undefined color space {} conversion function".format(mode))
def ColorShunt(data, color_mode='CIELab'):
    if color_mode=='RGB':
        color_shunt = data.squeeze(1)                                             
    elif color_mode in ['CIELab', 'HSV', 'HSL', 'YUV']:
        color_shunt = ColorConversion(data.squeeze(1), color_mode).to(data.device)
    else: 
        raise ValueError(f"Unsupported color space: {color_mode}")
    return color_shunt[:,0:3,:,:]

##########################
# 2.Feature Echelon Module
##########################
class CoSeAtt(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads  
        head_dim = dim // num_heads  
        self.scale = qk_scale or head_dim ** -0.5  
        self.attn_drop = nn.Dropout(attn_drop)     
        self.proj_L = nn.Linear(dim//3, dim//3)    
        self.proj_A = nn.Linear(dim//3, dim//3)    
        self.proj_B = nn.Linear(dim//3, dim//3)    
        self.proj_drop = nn.Dropout(proj_drop)     

    def forward(self, x):
        Batch, N, C = x.shape          
        split_size = x.shape[-1] // 3  
        L, A, B = torch.split(x, split_size, dim=-1) 
        L_sp = L.reshape(Batch, N, self.num_heads, C // (3*self.num_heads)).permute(0, 2, 1, 3) 
        A_sp = A.reshape(Batch, N, self.num_heads, C // (3*self.num_heads)).permute(0, 2, 1, 3)
        B_sp = B.reshape(Batch, N, self.num_heads, C // (3*self.num_heads)).permute(0, 2, 1, 3)

        # 1)VQK layer: L, B, A
        attn_L = (A_sp @ B_sp.transpose(-2, -1)) * self.scale        
        attn_L = attn_L.softmax(dim=-1)                              
        attn_L = self.attn_drop(attn_L)                              
        L_a = (attn_L @ L_sp).transpose(1, 2).reshape(Batch, N, C//3)
        # Projection
        L_p = self.proj_L(L_a)        
        L_p = self.proj_drop(L_p)     

        # 2)KVQ layer: L, A, B
        attn_A = (B_sp @ L_sp.transpose(-2, -1)) * self.scale         
        attn_A = attn_A.softmax(dim=-1)                              
        attn_A = self.attn_drop(attn_A)                               
        A_a = (attn_A @ A_sp).transpose(1, 2).reshape(Batch, N, C//3) 
        # Projection
        A_p = self.proj_A(A_a)      
        A_p = self.proj_drop(A_p)   
        
        # 3)QKV layer: L, A, B
        attn_B = (L_sp @ A_sp.transpose(-2, -1)) * self.scale         
        attn_B = attn_B.softmax(dim=-1)                               
        attn_B = self.attn_drop(attn_B)                               
        B_a = (attn_B @ B_sp).transpose(1, 2).reshape(Batch, N, C//3) 
        # Projection
        B_p = self.proj_B(B_a)         
        B_p = self.proj_drop(B_p)      

        # 4)LAB projection channels stack    
        Lab_Proj = torch.cat([L_p, A_p, B_p], dim=-1)
        Lab_attn = torch.stack([attn_L, attn_A, attn_B], dim=-1)
        
        return Lab_Proj, Lab_attn
class CoSeAtten(nn.Module):
    def __init__(self, dim, 
                 num_heads=3, qkv_bias=False, qk_scale=None, attn_drop=0., drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CoSeAtt(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                
    def forward(self, x, return_attention=False):        
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return y, attn
        else:
            return y, None
class ColorProjection(nn.Module):
    def __init__(self, in_features=384, proj_layer=240, drop=0.):
        super().__init__()
        self.L_Proj = nn.Sequential(nn.Linear(in_features, proj_layer, bias=True),
                                    nn.BatchNorm1d(proj_layer))
        self.A_Proj = nn.Sequential(nn.Linear(in_features, proj_layer, bias=True),
                                    nn.BatchNorm1d(proj_layer))
        self.B_Proj = nn.Sequential(nn.Linear(in_features, proj_layer, bias=True),
                                    nn.BatchNorm1d(proj_layer))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        hidden_layer1, hidden_layer2 = int(x.shape[2]//3), 2*int(x.shape[2]//3)
        L= x[:,:,0:hidden_layer1].reshape(x.size(0), -1)              
        A= x[:,:,hidden_layer1:hidden_layer2].reshape(x.size(0), -1)  
        B= x[:,:,hidden_layer2:].reshape(x.size(0), -1)               
        Embeded_L=self.drop(self.L_Proj(L))         
        Embeded_A=self.drop(self.A_Proj(A))         
        Embeded_B=self.drop(self.B_Proj(B))         
        CoFu = torch.cat([Embeded_L, Embeded_A, Embeded_B], dim=-1)
        return CoFu
class FeatureEchelon(nn.Module):
    def __init__(self, shunt_layer=480, proj_layer=128, encoder_flag=False, **kwargs):
        super(FeatureEchelon, self).__init__()
        self.emb_dim = shunt_layer * 25 //3  if not encoder_flag else shunt_layer//3
        self.feat_dim = [384, 5, 5]
        #################################
        # Feature extractor echelon
        #################################
        # 1.Feature Sentinels:
        self.sentinels = nn.Sequential(
                       nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1, bias=False, groups=3),  # 1.Feature scout：特征侦察兵
                       nn.BatchNorm2d(num_features=int(96)),
                       nn.MaxPool2d(kernel_size=2),
                       nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # 2.Feature Integrators:
        self.integrators = nn.Sequential(
                       nn.Conv2d(in_channels=96, out_channels=shunt_layer//2, kernel_size=3, bias=False, groups=3),
                       nn.BatchNorm2d(num_features=int(shunt_layer//2)),
                       nn.MaxPool2d(kernel_size=2),
                       nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # 3.Feature Abstractors:
        self.abstractors = nn.Sequential(
                       nn.Conv2d(in_channels=shunt_layer//2, out_channels=shunt_layer, kernel_size=3, padding=1, bias=False, groups=3),
                       nn.BatchNorm2d(num_features=int(shunt_layer)),
                       nn.MaxPool2d(kernel_size=2),
                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
                       nn.Dropout2d(0.5))
        # 4.Feature Directors:
        self.directors = nn.Sequential(
                       nn.Conv2d(in_channels=shunt_layer, out_channels=shunt_layer, kernel_size=3, padding=1, bias=False, groups=3),
                       nn.BatchNorm2d(num_features=int(shunt_layer)),
                       nn.MaxPool2d(kernel_size=2),
                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
                       nn.Dropout2d(0.5))
        # 5.Feature General:
        self.depth=1
        self.CoSeAtten = nn.ModuleList([
                       CoSeAtten(dim=480, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., drop=0.,
                       norm_layer= partial(nn.LayerNorm, eps=1e-6)) for i in range(self.depth)])
        norm_layer= partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(480)
        self.general = ColorProjection(self.emb_dim, proj_layer)

    def forward(self, inputs):
        # 1.Feature Sentinels:
        feat_sen = self.sentinels(inputs)
        # 2.Feature Integrators:
        feat_int = self.integrators(feat_sen)
        # 3.Feature Abstractors:
        feat_abs = self.abstractors(feat_int)
        # 4.Feature Directors:
        feat_dir = self.directors(feat_abs)
        # 5.Feature General:
        feat_atten = feat_dir.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.CoSeAtten):
            if i < len(self.CoSeAtten) - 1:
                feat_atten, attn = blk(feat_atten)
            else:
                feat_atten, attn = blk(feat_atten, return_attention=False)
        feat_atten = self.norm(feat_atten)                 
        feat_gen = self.general(feat_atten)         

        return feat_gen

##########################
# 3.Color Pattern Module
##########################
class SimMetric(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """ Compute Similarity Metric
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(SimMetric, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c*2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c*2),
                       nn.LeakyReLU()]
        if self.dropout > 0: layer_list += [nn.Dropout2d(p=self.dropout)]
        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]
        if self.dropout > 0: layer_list += [nn.Dropout2d(p=self.dropout)]
        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.cls_matrix_transform = nn.Sequential(*layer_list)

    def forward(self, embedding_gen, cls_matrix, distance_metric):
        ##########################################
        # 1.Calculate the similarity of embedding.
        ##########################################
        emb_i = embedding_gen.unsqueeze(2)             
        emb_j = torch.transpose(emb_i, 1, 2)            
        if distance_metric == 'l2':   emb_similarity = (emb_i - emb_j)**2      
        elif distance_metric == 'l1': emb_similarity = torch.abs(emb_i - emb_j)
        embed_sim = -torch.sum(emb_similarity, 3)                              
        ###################################
        # 2.Calculate cls_ij
        ###################################
        trans_similarity = torch.transpose(emb_similarity, 1, 3)
        cls_ij = torch.sigmoid(self.cls_matrix_transform(trans_similarity))
        # cls_matrix normalization:
        diagonal_mask = 1.0 - torch.eye(embedding_gen.size(1)).unsqueeze(0).repeat(embedding_gen.size(0), 1, 1).to(cls_matrix.get_device())
        cls_matrix *= diagonal_mask
        cls_matrix_sum = torch.sum(cls_matrix, -1, True)
        try:
            cls_ij = F.normalize(cls_ij.squeeze(1).clone() * cls_matrix.clone(), p=1, dim=-1) * cls_matrix_sum
        except Exception as e:
            print(f"Error during computation: {e}")
        diagonal_reverse_mask = torch.eye(embedding_gen.size(1)).unsqueeze(0).to(cls_matrix.get_device())
        cls_ij += (diagonal_reverse_mask + 1e-6)
        cls_ij /= torch.sum(cls_ij, dim=2).unsqueeze(-1)

        return cls_ij, embed_sim
class embed_update(nn.Module):
    def __init__(self, in_c, out_c):
        """
        :param in_c: number of input channel for the fc layer
        :param out_c:number of output channel for the fc layer
        """
        super(embed_update, self).__init__()
        self.embedding_transform = nn.Sequential(*[
                                                nn.Linear(in_features=in_c, out_features=out_c, bias=True),
                                                nn.LeakyReLU()
                                                ])
        self.out_c = out_c

    def forward(self, cls_matrix, embedding):
        """
        :param cls_matrix: current generation's classification matrix of L channel
        :param embedding: last generation's embedded feature of A channel or B channel
        :return: current generation's embedded feature of A channel or B channel
        """
        meta_batch = cls_matrix.size(0)
        num_sample = cls_matrix.size(1)
        #######################################
        # 1.[cls_matirx, embedding]->embedding
        #######################################
        embedding = torch.cat([cls_matrix[:, :, :self.out_c], embedding], dim=2)
        embedding = embedding.view(meta_batch*num_sample, -1)        
        #######################################
        # 2.embedding transformation
        #######################################
        embedding = self.embedding_transform(embedding)
        embedding = embedding.view(meta_batch, num_sample, -1)
        
        return embedding   
class L_embed_update(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(L_embed_update, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]
        if self.dropout > 0: layer_list += [nn.Dropout2d(p=self.dropout)]
        self.L_embedding_transform = nn.Sequential(*layer_list)

    def forward(self, cls_matrix, L_embedding):
        # get size
        meta_batch = L_embedding.size(0)
        num_sample = L_embedding.size(1)
        # get eye matrix (batch_size x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_sample).unsqueeze(0).repeat(meta_batch, 1, 1).to(cls_matrix.get_device())
        matrix_feat = F.normalize(cls_matrix * diag_mask, p=1, dim=-1)
        aggr_feat = torch.bmm(matrix_feat, L_embedding)
        embed_feat = torch.cat([L_embedding, aggr_feat], -1).transpose(1, 2)
        # non-linear transform
        embed_feat = self.L_embedding_transform(embed_feat.unsqueeze(-1))
        L_embedding = embed_feat.transpose(1, 2).squeeze(-1)
     
        return L_embedding
class ColorPattern(nn.Module):
    def __init__(self, emb_size, num_generations, dropout, num_support_sample, num_sample, loss_indicator, dis_metric, **kwargs):
        """
        :param num_generations: number of total generations
        :param dropout: dropout rate
        :param num_support_sample: number of support sample
        :param num_sample: number of sample
        :param loss_indicator: indicator of what losses are using
        :param dis_metric: metric for distance
        """
        super(ColorPattern, self).__init__()
        self.emb_size = emb_size
        self.generation = num_generations 
        self.dropout = dropout            
        self.num_support_sample = num_support_sample
        self.num_sample = num_sample
        self.loss_indicator = loss_indicator
        self.dis_metric = dis_metric
        
        L_cls_matrix_init = SimMetric(self.emb_size, self.emb_size, dropout=self.dropout)
        self.add_module('L_cls_mat_initial', L_cls_matrix_init)
        A_cls_matrix_init = SimMetric(self.emb_size, self.emb_size, dropout=self.dropout)
        self.add_module('A_cls_mat_initial', A_cls_matrix_init)
        B_cls_matrix_init = SimMetric(self.emb_size, self.emb_size, dropout=self.dropout)
        self.add_module('B_cls_mat_initial', B_cls_matrix_init)
                
        for l in range(self.generation):
            L_clsm = SimMetric(self.emb_size, self.emb_size, dropout=self.dropout if l < self.generation-1 else 0.0)
            self.add_module('L_cls_mat_gen_{}'.format(l), L_clsm)
            A_clsm = SimMetric(self.emb_size, self.num_sample, dropout=self.dropout if l < self.generation-1 else 0.0)
            self.add_module('A_cls_mat_gen_{}'.format(l), A_clsm)
            B_clsm = SimMetric(self.emb_size, self.num_sample, dropout=self.dropout if l < self.generation-1 else 0.0)
            self.add_module('B_cls_mat_gen_{}'.format(l), B_clsm)
            L2A_emb = embed_update(self.num_sample+self.emb_size, self.emb_size)
            self.add_module('L2A_emb_gen_{}'.format(l), L2A_emb)
            L2B_emb = embed_update(self.num_sample+self.emb_size, self.emb_size)
            self.add_module('L2B_emb_gen_{}'.format(l), L2B_emb)
            A2L_emb = L_embed_update(self.emb_size*2, self.emb_size, dropout=self.dropout if l < self.generation-1 else 0.0)
            self.add_module('A2L_emb_gen_{}'.format(l), A2L_emb)
            B2L_emb = L_embed_update(self.emb_size*2, self.emb_size, dropout=self.dropout if l < self.generation-1 else 0.0)
            self.add_module('B2L_emb_gen_{}'.format(l), B2L_emb)

    def forward(self, L_layer_data, A_layer_data, B_layer_data, L_cls_matrix, A_cls_matrix, B_cls_matrix):
        init_L_emb, init_A_emb, init_B_emb = L_layer_data, A_layer_data, B_layer_data, 
        
        # Initialize classification matrix:
        init_L_cls_mat, _ = self._modules['L_cls_mat_initial'](init_L_emb, L_cls_matrix, self.dis_metric) # torch.Size([16, 10, 10]) for 1-shot
        init_A_cls_mat, _ = self._modules['A_cls_mat_initial'](init_A_emb, A_cls_matrix, self.dis_metric) # torch.Size([16, 10, 10]) for 1-shot
        init_B_cls_mat, _ = self._modules['B_cls_mat_initial'](init_B_emb, B_cls_matrix, self.dis_metric) # torch.Size([16, 10, 10]) for 1-shot

        # Initialize LAB feature embeddings and classification matrix:
        L_emb, A_emb, B_emb = init_L_emb, init_A_emb, init_B_emb
        L_clsm, A_clsm, B_clsm = init_L_cls_mat, init_A_cls_mat, init_B_cls_mat
        L_clsm_sims, L_emb_sims = [], []      
        A_clsm_sims, A_emb_sims = [], []      
        B_clsm_sims, B_emb_sims = [], []      
        
        for l in range(self.generation): 
            L_clsm, L_emb_sim = self._modules['L_cls_mat_gen_{}'.format(l)](L_emb, L_clsm, self.dis_metric)
            A_emb = self._modules['L2A_emb_gen_{}'.format(l)](L_clsm, A_emb)
            A_clsm, A_emb_sim = self._modules['A_cls_mat_gen_{}'.format(l)](A_emb, A_clsm, self.dis_metric)
            L_emb = self._modules['A2L_emb_gen_{}'.format(l)](A_clsm, L_emb)
            B_emb = self._modules['L2B_emb_gen_{}'.format(l)](L_clsm, B_emb)
            B_clsm, B_emb_sim = self._modules['B_cls_mat_gen_{}'.format(l)](B_emb, B_clsm, self.dis_metric)
            L_emb = self._modules['B2L_emb_gen_{}'.format(l)](B_clsm, L_emb)            
            L_clsm_sims.append(L_clsm * self.loss_indicator[0])    
            L_emb_sims.append(L_emb_sim * self.loss_indicator[1])  
            A_clsm_sims.append(A_clsm * self.loss_indicator[2])    
            B_clsm_sims.append(B_clsm * self.loss_indicator[2])    
            A_emb_sims.append(A_emb_sim * self.loss_indicator[3])  
            B_emb_sims.append(B_emb_sim * self.loss_indicator[3])
            
        return L_clsm_sims, L_emb_sims, A_clsm_sims, B_clsm_sims

##################################
# ColorSense Learner(CoSe Learner)
##################################
class CoSeLearner(nn.Module):
    def __init__(self, encoder_flag, emb_size, num_generations, dropout, num_support_sample, num_sample, loss_indicator, dis_metric,
                shunt_layer=480, proj_layer=128, **kwargs):
        """
        :param num_generations: number of total generations
        :param dropout: dropout rate
        :param num_support_sample: number of support sample
        :param num_sample: number of sample
        :param loss_indicator: indicator of what losses are using
        :param dis_metric: metric for distance
        """
        super(CoSeLearner, self).__init__()
        self.num_gen = num_generations
        self.num_sup, self.num_sample = num_support_sample, num_sample
        self.shunt_layer, self.proj_layer = shunt_layer, proj_layer
        
        self.FeatureEchelon = FeatureEchelon(shunt_layer=self.shunt_layer, proj_layer=self.proj_layer, encoder_flag=encoder_flag)
        self.ColorPattern = ColorPattern(
                                emb_size=emb_size, num_generations=self.num_gen, dropout=dropout,num_support_sample= self.num_sup, 
                                num_sample=self.num_sample, loss_indicator=loss_indicator, dis_metric=dis_metric)
        
    def forward(self, all_data, sub1_cls_matrix, sub2_cls_matrix, sub3_cls_matrix, color_mode='CIELab'):    # LAB Color Space
    # def forward(self, all_data, sub1_cls_matrix, sub2_cls_matrix, sub3_cls_matrix, color_mode='RGB'):     # RGB Color Space
    # def forward(self, all_data, sub1_cls_matrix, sub2_cls_matrix, sub3_cls_matrix, color_mode='HSV'):     # HSV Color Space
    # def forward(self, all_data, sub1_cls_matrix, sub2_cls_matrix, sub3_cls_matrix, color_mode='HSL'):     # HSL Color Space
    # def forward(self, all_data, sub1_cls_matrix, sub2_cls_matrix, sub3_cls_matrix, color_mode='YUV'):     # YUV Color Space
        
        # enbedding channels
        sub1_data_temp = []
        sub2_data_temp = []
        sub3_data_temp = []
        
        for data in all_data.chunk(all_data.size(1), dim=1):                   
            ################
            # 1.Color Shunt
            ################
            input = ColorShunt(data, color_mode)                               
            ####################
            # 2.Feature Echelon
            ####################
            color_embed = self.FeatureEchelon(input)                           
            dim = color_embed.size(1)
            sub1_data_temp.append(color_embed[:,0:int(dim//3)])               
            sub2_data_temp.append(color_embed[:,int(dim//3):int(2*dim//3)])   
            sub3_data_temp.append(color_embed[:,int(2*dim//3):])               

        color_embedding1 = torch.stack(sub1_data_temp, dim=1)                  
        color_embedding2 = torch.stack(sub2_data_temp, dim=1)                 
        color_embedding3 = torch.stack(sub3_data_temp, dim=1)                  

        ##################
        # 3.Color Pattern
        ##################
        sub1_clsm, sub1_emb_sims, sub2_clsm, sub3_clsm = self.ColorPattern(color_embedding1, color_embedding2, color_embedding3, 
                                                               sub1_cls_matrix, sub2_cls_matrix, sub3_cls_matrix)

        return sub1_clsm, sub1_emb_sims, sub2_clsm, sub3_clsm

def CoSeLearner_(encoder_flag=False, **kwargs):
    """constructs CoSeLearner network."""
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'<<CoSeLearner Architecture>>:')
    model = CoSeLearner(encoder_flag=encoder_flag, emb_size=128, num_generations=2, dropout=0.1, loss_indicator= [1, 1, 1, 0], 
            dis_metric= 'l1', num_support_sample=5, num_sample=10, shunt_layer=480, proj_layer=128, **kwargs)
    print(model)
    print(Fore.RED+'*********'* 10)
    return model


if __name__ == '__main__':
    #########################################################################
    # Instantiate the model:
    #########################################################################
    # 1)Generate pseudo-input:
    Input_tensor = torch.rand(2, 10, 3, 84, 84)
    encoder_flag = False
    L_cls_matrix = torch.ones(2, 10, 10)
    A_cls_matrix = torch.ones(2, 10, 10)
    B_cls_matrix = torch.ones(2, 10, 10)

    # 2)Network instantiation:
    model = CoSeLearner_(encoder_flag = encoder_flag).cuda()
    
    # 3)Network inference + output:
    s_time = time.time()
    for i in range(10): 
        L_clsm, L_emb_sims, A_clsm, B_clsm = model(Input_tensor.cuda(), L_cls_matrix.cuda(), A_cls_matrix.cuda(), B_cls_matrix.cuda(),)
    f_time = time.time()
    period = "{:.4f}".format(f_time - s_time)
    print('L_clsm:', L_clsm[0].shape)                         # CoSeLeaner: torch.Size([16, 10, 10])
