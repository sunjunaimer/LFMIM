import os
import time
import math
import collections
from collections import OrderedDict
import argparse
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, MultiStepLR, ExponentialLR
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch.distributed as dist
import torch.utils.data.distributed

from torchmetrics import Accuracy, ConfusionMatrix
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from data_prepare import MMSAATBaselineDataset
from modules.position_embedding import SinusoidalPositionalEmbedding


max_len = 50

labels_eng =  ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def pad_collate(batch):
    (x_t, x_a, x_v, y_t, y_a, y_v, y_m) = zip(*batch)
    x_t = torch.stack(x_t, dim=0)
    x_v = torch.stack(x_v, dim=0)
    y_t = torch.tensor(y_t)
    y_a = torch.tensor(y_a)
    y_v = torch.tensor(y_v)
    y_m = torch.tensor(y_m)
    x_a_pad = pad_sequence(x_a, batch_first=True, padding_value=0)
    len_trunc = min(x_a_pad.shape[1], max_len)
    x_a_pad = x_a_pad[:, 0:len_trunc, :]
    len_com = max_len - len_trunc
    zeros = torch.zeros([x_a_pad.shape[0], len_com, x_a_pad.shape[2]], device='cpu')
    x_a_pad = torch.cat([x_a_pad, zeros], dim=1)

    return x_t, x_a_pad, x_v, y_t, y_a, y_v, y_m
#######################################################

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=0.2, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.ln_12 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("relu", nn.ReLU()),
            ('dropout', nn.Dropout(p=0.1)),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.ln_22 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.dropout = nn.Dropout(p=0.1)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.ln_12(self.attention(self.ln_1(x)))
        x = x + self.ln_22(self.mlp(self.ln_2(x)))
        return x



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class MultiUniTransformer(nn.Module,):
    def __init__(self, width: list, layers: int, heads: int, embed_dim: list, args, attn_mask: torch.Tensor = None):
        super().__init__()
        self.layers = layers
        self.width_t = width[0]
        self.width_a = width[1]
        self.width_v = width[2]
        self.width_m = width[3]
        self.embed_dim_t = embed_dim[0]
        self.embed_dim_a = embed_dim[1]
        self.embed_dim_v = embed_dim[2]
        self.embed_dim_m = embed_dim[3]
        self.width_t_ori = 1024
        self.width_a_ori = 1024
        self.width_v_ori = 2048
        self.dropout_ft = 0.25
        self.dropout_fa = 0.25
        self.dropout_fv = 0.25
        self.dropout_fm = 0.25
        self.fea_len_t = args.fea_len_t
        self.fea_len_a = args.fea_len_a
        self.fea_len_v = args.fea_len_v
        self.fea_len_m = args.fea_len_m
        num_classes = args.num_classes


        self.transformer_t = nn.Sequential(*[ResidualAttentionBlock(self.width_t, heads[0], attn_mask) for _ in range(layers)])
        self.transformer_a = nn.Sequential(*[ResidualAttentionBlock(self.width_a, heads[1], attn_mask) for _ in range(layers)])
        self.transformer_v = nn.Sequential(*[ResidualAttentionBlock(self.width_v, heads[2], attn_mask) for _ in range(2 * layers)])
        self.transformer_m = nn.Sequential(*[ResidualAttentionBlock(self.width_m, heads[3], attn_mask) for _ in range(layers)])
        

        self.t2m = nn.Sequential(*[nn.Linear(self.fea_len_t, 1) for _ in range(layers)])
        self.v2m = nn.Sequential(*[nn.Linear(self.fea_len_v, 1) for _ in range(layers)])
        self.fc_vtom = nn.Linear(self.width_v, self.width_m)
        self.fc_tdimtr = nn.Linear(self.width_t_ori, self.width_t)
        self.fc_adimtr = nn.Linear(self.width_a_ori, self.width_a)
        self.fc_vdimtr = nn.Linear(self.width_v_ori, self.width_m)

        self.ln_pre_t = LayerNorm(self.width_t)
        self.ln_pre_a = LayerNorm(self.width_a)
        self.ln_pre_v = LayerNorm(self.width_v)
        self.ln_pre_m = LayerNorm(self.width_m)

        self.ln_post_t = LayerNorm(self.width_t)
        self.ln_post_a = LayerNorm(self.width_a)
        self.ln_post_v = LayerNorm(self.width_v)
        self.ln_post_m = LayerNorm(self.width_m)

        self.ln_prepe_t = LayerNorm(self.width_t)
        self.ln_prepe_a = LayerNorm(self.width_a)
        self.ln_prepe_v = LayerNorm(self.width_v)
        self.ln_prepe_m = LayerNorm(self.width_m)

        self.proj_a = nn.Parameter(torch.empty(self.width_a, self.embed_dim_a))
        self.proj_v = nn.Parameter(torch.empty(self.width_v, self.embed_dim_v))
        self.mix_t = nn.Parameter((self.fea_len_t ** -0.5) * torch.rand(self.fea_len_t))
        self.mix_a = nn.Parameter((self.fea_len_a ** -0.5) * torch.rand(self.fea_len_a))
        self.mix_v = nn.Parameter((self.fea_len_v ** -0.5) * torch.rand(self.fea_len_v))
        self.mix_m = nn.Parameter(((self.fea_len_m + self.fea_len_t + self.fea_len_a + self.fea_len_v) ** -0.5) * torch.rand(self.fea_len_m + self.fea_len_t + self.fea_len_a + self.fea_len_v))
        self.mlp_t = nn.Sequential(nn.Linear(self.embed_dim_t, int(self.embed_dim_t / 2)), nn.ReLU(), nn.Linear(int(self.embed_dim_t / 2), 32), nn.ReLU(), nn.Linear(32, num_classes))
        self.mlp_a = nn.Sequential(nn.Linear(self.embed_dim_a, int(self.embed_dim_a / 2)), nn.ReLU(), nn.Linear(int(self.embed_dim_a / 2), 32), nn.ReLU(), nn.Linear(32, num_classes))
        self.mlp_v = nn.Sequential(nn.Linear(self.embed_dim_v, int(self.embed_dim_v / 2)), nn.ReLU(), nn.Linear(int(self.embed_dim_v / 2), 32), nn.ReLU(), nn.Linear(32, num_classes))
        self.mlp_m = nn.Sequential(nn.Linear(self.embed_dim_m + self.embed_dim_t +self.embed_dim_a + self.embed_dim_v, int((self.embed_dim_t + self.embed_dim_a + self.embed_dim_v + self.embed_dim_m) / 2)), nn.ReLU(), \
            nn.Linear(int((self.embed_dim_t + self.embed_dim_a + self.embed_dim_v + self.embed_dim_m) / 2), 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))
   
        self.plus_p = nn.Parameter(torch.randn(6))

        scale_t = self.width_t ** -0.5
        self.pe_t = nn.Parameter(scale_t * torch.randn(self.fea_len_t, self.width_t))
        self.pe_t2m = nn.Parameter(scale_t * torch.randn(self.fea_len_t, self.width_t))
        scale_a = self.width_a ** -0.5
        self.pe_a = nn.Parameter(scale_a * torch.randn(self.fea_len_a, self.width_a))
        scale_v = self.width_v ** -0.5
        self.pe_v = nn.Parameter(scale_v * torch.randn(self.fea_len_v, self.width_v))
        self.pe_v2m = nn.Parameter(scale_v * torch.randn(self.fea_len_v, self.width_m))
        scale_m = self.width_m ** -0.5
        self.pe_m = nn.Parameter(scale_m * torch.randn(self.fea_len_m, self.width_m))

        self.me = nn.Parameter(scale_t * torch.randn(4))

        self.embed_scale = math.sqrt(self.width_m)
        self.embed_positions = SinusoidalPositionalEmbedding(self.width_m)


        scale_t = self.width_t ** -0.5
        self.embedding_t = nn.Parameter(scale_t * torch.randn(self.width_t))
        scale_a = self.width_a ** -0.5
        self.embedding_a = nn.Parameter(scale_a * torch.randn(self.width_a))
        scale_v = self.width_v ** -0.5
        self.embedding_v = nn.Parameter(scale_v * torch.randn(self.width_v))
        scale_m = self.width_m ** -0.5
        self.embedding_m = nn.Parameter(scale_m * torch.randn(self.fea_len_m, self.width_m))
    
    def initialize_parameters(self):
        nn.init.normal_(self.proj_a, std=self.width_a ** -0.5)
        nn.init.normal_(self.proj_v, std=self.width_v ** -0.5)
   
        proj_std = (self.width_t ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width_t ** -0.5
        fc_std = (2 * self.width_t) ** -0.5
        for branch in [self.transformer_t, self.transformer_a, self.transformer_m]:
            for block in branch:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        proj_std = (self.width_v ** -0.5) * ((2 * 2 * self.layers) ** -0.5)
        attn_std = self.width_v ** -0.5
        fc_std = (2 * self.width_v) ** -0.5
        for branch in [self.transformer_v]:
            for block in branch:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    
    def forward(self, x_t, x_a, x_v):
        x_t = x_t[:, 0:80, :]
        x_v = x_v.to(torch.float32)
        x_t = x_t.to(torch.float32)
        x_a = x_a.to(torch.float32)
        x_t = self.fc_tdimtr(x_t)
        x_a = self.fc_adimtr(x_a)
        x_v = self.fc_vdimtr(x_v)
        #x_t = x_t[:, 1:79, :]

        x_m = self.embedding_m.to(x_t.dtype) + torch.zeros(x_t.shape[0], self.fea_len_m, x_t.shape[-1], dtype=x_t.dtype, device=x_t.device)
        x_m = x_m.to(device)
        x_t = torch.cat([self.embedding_t.to(x_t.dtype) + torch.zeros(x_t.shape[0], 1, x_t.shape[-1], dtype=x_t.dtype, device=x_t.device), x_t], dim=1)
        x_a = torch.cat([self.embedding_a.to(x_a.dtype) + torch.zeros(x_a.shape[0], 1, x_a.shape[-1], dtype=x_a.dtype, device=x_a.device), x_a], dim=1)
        x_v = torch.cat([self.embedding_v.to(x_v.dtype) + torch.zeros(x_v.shape[0], 1, x_v.shape[-1], dtype=x_v.dtype, device=x_v.device), x_v], dim=1)

        x_t = self.embed_scale * x_t
        x_a = self.embed_scale * x_a
        x_v = self.embed_scale * x_v
        x_m = self.embed_scale * x_m

        x_t = self.ln_prepe_t(x_t)
        x_a = self.ln_prepe_a(x_a)
        x_v = self.ln_prepe_v(x_v)
        x_m = self.ln_prepe_m(x_m)
        
        x_t += self.embed_positions(x_t[:, :, 0])
        x_a += self.embed_positions(x_a[:, :, 0])
        x_v += self.embed_positions(x_v[:, :, 0])
        x_m += self.embed_positions(x_m[:, :, 0])

        x_t = F.dropout(x_t, p=self.dropout_ft, training=self.training)
        x_a = F.dropout(x_a, p=self.dropout_fa, training=self.training)
        x_v = F.dropout(x_v, p=self.dropout_fv, training=self.training)
        x_m = F.dropout(x_m, p=self.dropout_fm, training=self.training)

        x_t = self.ln_pre_t(x_t)
        x_a = self.ln_pre_a(x_a)
        x_v = self.ln_pre_v(x_v)
        x_m = self.ln_pre_m(x_m)

        for i in range(0, self.layers):

            x_t2m = self.t2m[i](x_t.permute(0, 2, 1))
            x_a2m = torch.mean(x_a, 1)
            x_v2m = self.v2m[i](x_v.permute(0, 2, 1))
            x_t2m = x_t2m.permute(0, 2, 1)
            x_a2m = x_a2m.unsqueeze(1)
            x_v2m = x_v2m.permute(0, 2, 1)

            if i == 0: 
                x_m = torch.cat([x_m[:, 0:self.fea_len_m, :], x_t, x_v, x_a], dim=1)
            else:
                x_tacc = x_m[:, self.fea_len_m: self.fea_len_m + self.fea_len_t, :] * self.plus_p[0] + x_t * self.plus_p[1]
                x_vacc = x_m[:, self.fea_len_m + self.fea_len_t : self.fea_len_m + self.fea_len_t + self.fea_len_v, :] * self.plus_p[2] + x_v * self.plus_p[3]
                x_aacc = x_m[:, self.fea_len_m + self.fea_len_t + self.fea_len_v : self.fea_len_m + self.fea_len_t + self.fea_len_v + self.fea_len_a, :] * self.plus_p[4] + x_a * self.plus_p[5]
                x_m = torch.cat([x_m[:, 0:self.fea_len_m, :], x_tacc, x_vacc, x_aacc], dim=1)
        
            x_t = self.transformer_t[i](x_t)
            x_a = self.transformer_a[i](x_a)
            x_v = self.transformer_v[2 * i](x_v)
            x_v = self.transformer_v[2 * i + 1](x_v)
            x_m = self.transformer_m[i](x_m)

        x_t = self.ln_post_t(x_t)
        x_a = self.ln_post_a(x_a)
        x_v = self.ln_post_v(x_v)
        x_m = self.ln_post_m(x_m)

        x_t = torch.matmul(self.mix_t, x_t) 
        x_a = torch.matmul(self.mix_a, x_a)
        x_v = torch.matmul(self.mix_v, x_v) 
        x_m = torch.matmul(self.mix_m, x_m) 
        
        x_all = torch.cat([x_t, x_a, x_v, x_m], dim=1)

        x_t = self.mlp_t(x_t)
        x_a = self.mlp_a(x_a)
        x_v = self.mlp_v(x_v)
        x_m = self.mlp_m(x_all)

        output_t = x_t
        output_a = x_a
        output_v = x_v
        output_m = x_m
        
        return [output_t, output_a, output_v, output_m]

class Trainer():
    def __init__(self, args):
        
        self.args = args
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.log_interval = args.log_interval
        self.local_rank  = args.local_rank
        num_classes = args.num_classes
        self.num_classes = args.num_classes
        self.beta = args.beta
 
        self.model = MultiUniTransformer([1024]*4, 4, [16]*4, [1024]*4, args)
        self.model = self.model.to(device)
        self.model.initialize_parameters()
        # print(self.model)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.005)
        self.scheduler_1r = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
        
        
        train_data = MMSAATBaselineDataset('train')
        traindata_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=traindata_sampler, batch_size=self.batch_size, num_workers=6, collate_fn=pad_collate)

        test_data = MMSAATBaselineDataset('test')
        testdata_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, num_workers=6, collate_fn=pad_collate)
        self.train_te_dataloader = DataLoader(train_data, batch_size=self.batch_size, num_workers=6, collate_fn=pad_collate)
        self.test_accuracy = Accuracy()
        self.test_confmat = ConfusionMatrix(num_classes=num_classes)

        self.test_pred = []
        self.test_label = []

    def train(self):
        self.model.train()
        loss_train_t, loss_train_a, loss_train_v, loss_train_m, loss_train_all = [], [], [], [], []
        loss_test_t, loss_test_a, loss_test_v, loss_test_m ,loss_test_all = [], [], [], [], []
        acc_test_t, acc_test_a, acc_test_v, acc_test_m = [], [], [], []

        #############################################################################################
    
        test_loss, test_acc, _ = self.test(self.test_dataloader)
        self.model.train()


        loss_test_t.append(test_loss[0])
        loss_test_a.append(test_loss[1])
        loss_test_v.append(test_loss[2])
        loss_test_m.append(test_loss[3])
        loss_test_all.append(test_loss[4]) 

        acc_test_t.append(test_acc[0])
        acc_test_a.append(test_acc[1])
        acc_test_v.append(test_acc[2])
        acc_test_m.append(test_acc[3])


        #############################################################################################


        for epoch in range(0, self.epoch):
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                text, audio, video, label_t, label_a, label_v, label_m = batch
                label_t, label_a, label_v, label_m = label_t.to(device), label_a.to(device), label_v.to(device), label_m.to(device)
                label_m_onehot = F.one_hot(label_m, self.num_classes)
                label_t_onehot = (1 - self.beta) * F.one_hot(label_t, self.num_classes) + self.beta * label_m_onehot
                label_a_onehot = (1 - self.beta) * F.one_hot(label_a, self.num_classes) + self.beta * label_m_onehot
                label_v_onehot = (1 - self.beta) * F.one_hot(label_v, self.num_classes) + self.beta * label_m_onehot

                text = text.to(device)
                audio = audio.to(device)
                video = video.to(device)

                output = self.model(text, audio, video)
                loss_t = F.cross_entropy(output[0], label_t)
                loss_a = F.cross_entropy(output[1], label_a)
                loss_v = F.cross_entropy(output[2], label_v)
                loss_m = F.cross_entropy(output[3], label_m)
                
                loss = loss_t + loss_a + loss_v + loss_m
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        epoch, batch_idx * self.batch_size, len(self.train_dataloader.dataset),
                            100. * batch_idx / len(self.train_dataloader)))
                    print('\n Train set: loss_t: {:.4f}, loss_a: {:.4f}, loss_v: {:.4f}, loss_m: {:.4f}, loss: {:.4f}\n'.format(
                        loss_t.item(), loss_a.item(), loss_v.item(), loss_m.item(), loss.item()))

            self.scheduler_1r.step()
            print("epoch %d learning rate: %f" % (epoch, self.optimizer.param_groups[0]['lr']))

            test_loss, test_acc, p = self.test(self.test_dataloader)
            save_name = 'output/ep' + str(epoch) + '.npy'
            np.save(save_name, p)
            save_name='LFMIM'
            torch.save(self.model.state_dict(), f'saved_models/{save_name}_{str(epoch)}.pth')
            
            self.model.train()

            loss_test_t.append(test_loss[0])
            loss_test_a.append(test_loss[1])
            loss_test_v.append(test_loss[2])
            loss_test_m.append(test_loss[3])
            loss_test_all.append(test_loss[4]) 

            acc_test_t.append(test_acc[0])
            acc_test_a.append(test_acc[1])
            acc_test_v.append(test_acc[2])
            acc_test_m.append(test_acc[3])

        loss_test = [loss_test_t, loss_test_a, loss_test_v, loss_test_m]
        acc_test = [acc_test_t, acc_test_a, acc_test_v, acc_test_m]
        return loss_test, acc_test
    
    def test(self, dataloader):
        self.model.eval()
        loss_t = 0
        loss_a = 0
        loss_v = 0
        loss_m = 0
        test_loss = 0
        cor_t, cor_a, cor_v, cor_m = 0, 0, 0, 0
        predicted = []
        all_label_m = []
        p = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                text, audio, video, label_t, label_a, label_v, label_m = batch
                label_t, label_a, label_v, label_m = label_t.to(device), label_a.to(device), label_v.to(device), label_m.to(device)

                text = text.to(device)
                audio = audio.to(device)
                video = video.to(device)

                output = self.model(text, audio, video)
                if batch_idx == 0:
                    p = np.array(output[3].cpu().numpy())

                else:
                    p = np.concatenate((p, output[3].cpu().numpy()), axis=0)
                
                
                loss_t += F.cross_entropy(output[0], label_t, reduction ='sum').item()
                loss_a += F.cross_entropy(output[1], label_a, reduction ='sum').item()
                loss_v += F.cross_entropy(output[2], label_v, reduction ='sum').item()
                loss_m += F.cross_entropy(output[3], label_m, reduction ='sum').item()

                pred = output[0].argmax(dim=1, keepdim=True) 
                cor_t += pred.eq(label_t.view_as(pred)).sum().item()
                pred = output[1].argmax(dim=1, keepdim=True)  
                cor_a += pred.eq(label_a.view_as(pred)).sum().item()
                pred = output[2].argmax(dim=1, keepdim=True)  
                cor_v += pred.eq(label_v.view_as(pred)).sum().item()
                pred = output[3].argmax(dim=1, keepdim=True)  
                cor_m += pred.eq(label_m.view_as(pred)).sum().item()

                predicted.extend(output[3].cpu().numpy().argmax(1))
                all_label_m.extend(label_m.cpu().numpy())
        
            self.test_pred.extend(pred.tolist())
            self.test_label.extend(label_m.tolist())


        print('accuracy: ', self.test_accuracy)
        print('confusion matrix: ', self.test_confmat)

        c_m = confusion_matrix(all_label_m, predicted)#,  normalize='true')
        c_m_n = confusion_matrix(all_label_m, predicted,  normalize='true')
        c_r = classification_report(all_label_m, predicted, target_names = labels_eng, digits = 4)

        print(c_m)
        print(c_r)

        disp = ConfusionMatrixDisplay(confusion_matrix=c_m_n, display_labels = labels_eng)

        test_len = len(dataloader.dataset)
        cor_t /= test_len
        cor_a /= test_len
        cor_v /= test_len
        cor_m /= test_len

        loss_t /= test_len
        loss_a /= test_len
        loss_v /= test_len
        loss_m /= test_len
        loss = loss_t + loss_a + loss_v + loss_m

        print('\nTest set: loss_t: {:.4f}, loss_a: {:.4f}, loss_v: {:.4f}, loss_m: {:.4f}, loss: {:.4f}  Acc_t: {:.4f}, Acc_a: {:.4f}, Acc_v: {:.4f}, Acc_m: {:.4f}\n'.format(
        loss_t, loss_a, loss_v, loss_m, loss, cor_t, cor_a, cor_v, cor_m))
        
        return [loss_t, loss_a, loss_v, loss_m, loss], [cor_t, cor_a, cor_v, cor_m], p



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LFMIM')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:23456')
    parser.add_argument('--world_size', type=int, default=64, metavar='N')
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')
    
    parser.add_argument('--max_len', default=50, type=int, help='maximum length for audio sequence')
    parser.add_argument('--num_classes', default=7, type=int, help='number of emotions')
    parser.add_argument('--fea_len_t', default=81, type=int, help='dimension of the feature vector of text')
    parser.add_argument('--fea_len_a', default=51, type=int, help='dimension of the feature vector of audio')
    parser.add_argument('--fea_len_v', default=17, type=int, help='dimension of the feature vector of visual')
    parser.add_argument('--fea_len_m', default=4, type=int, help='dimension of the feature vector of multi-modality')
    parser.add_argument('--beta', default=0, type=float, help='mixing rate of labels')
    parser.add_argument('--epoch', default=20, type=int, help='number of training epoches')
    parser.add_argument('--batch_size', default=24, type=int, help='batch_size for training')
    parser.add_argument('--log_interval', default=50, type=int)
    
     
    args = parser.parse_args()
    args.labels_eng = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    dist.init_process_group(backend='nccl', world_size=args.world_size, init_method=args.init_method)

    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local_rank)
    args.local_rank = local_rank


    tic =time.time()
    a = Trainer(args)
    print(a)
    loss_test, acc_test = a.train()
    print('test loss:', loss_test)
    print('test accuracy:', acc_test)

    toc = time.time()
    runtime = toc - tic
    print('running time: ', runtime)

        
