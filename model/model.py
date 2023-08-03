
import torch
from torch import nn
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

from .text_encoders import TextEncoder
from .img_encoders import VisionEncoder
from .user_encoders import User_Encoder_NextItNet, User_Encoder_GRU4Rec, User_Encoder_SASRec

from .fushion_module import SumFusion, ConcatFusion, FiLM, GatedFusion, Wighted_Sum_fushion
from .clip_loss import clip_loss

# from lxmert, two attention fusion methods of co- and merge-.
from .modeling import CoAttention, MergedAttention

import numpy as np
from torch.nn import functional


def alignment(x, y):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(2).mean()

def uniformity(x):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


def align_uniformity(log_mask, _input1, _input2): # _input1, _input1 -> (batch_size*max_seq_len, embedding_dim)
    indices = torch.where(log_mask.view(-1) != 0)

    _input1 = _input1[indices]
    _input2 = _input2[indices]
    
    align = alignment(_input1, _input2)
    
    uniform = (uniformity(_input1) + uniformity(_input2)) / 2
    
    return align, uniform
    

class metric_learning_8(nn.Module):
    '''
    pull similar pairs closer and different pairs apart
    go through a shared MLP (two layers: expand and restore), return metric loss and output embeddings
    '''
    def __init__(self, input_dim):
        super(metric_learning_8, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(input_dim, int(2*input_dim))
        self.linear2 = nn.Linear(int(2*input_dim), input_dim)
        self.dense = nn.Linear(input_dim, input_dim)
        # self.drop = nn.Dropout(0.1)
    def forward(self, args, _ids, log_mask, _input1, _input2, temperature=1.0): # _input1, _input1 -> (batch_size*max_seq_len, embedding_dim)
        indices = torch.where(log_mask.view(-1) != 0)

        seq_indexs = torch.arange(log_mask.shape[0]).unsqueeze(-1).expand_as(log_mask)[(log_mask!=0)]

        _input1 = _input1[indices]
        _input2 = _input2[indices]
        ids = _ids.view(-1)[indices]

        input1 = self.bn(self.linear2(self.relu(self.linear1(_input1))))
        input2 = self.bn(self.linear2(self.relu(self.linear1(_input2))))

        # input1 = self.relu(self.bn(self.dense(_input1)))
        # input2 = self.relu(self.bn(self.dense(_input2)))

        input1 = functional.normalize(input1, dim=1)
        input2 = functional.normalize(input2, dim=1)

        loss_i = 0.0
        calc_ids = set()
        for i in range(input1.shape[0]):
            if ids[i] not in calc_ids:
                anchor_emb = input1[i].unsqueeze(0)
                if i < input1.shape[0]-1 and seq_indexs[i] == seq_indexs[i+1]:
                    pos_emb = input2[[i, i+1]]
                    neg_index_1 = (ids != ids[i])
                    neg_index_2 = (ids != ids[i+1])
                    # cut other items in current user 
                    item_of_user_index_indices = seq_indexs.view(-1) == seq_indexs[i]
                    item_of_user_index_indices = ~item_of_user_index_indices.to(input2.get_device())
                    neg_emb = input2[neg_index_1*neg_index_2*item_of_user_index_indices]
                    all_embs = torch.cat([pos_emb, neg_emb], dim=0)
                else:
                    pos_emb = input2[i].unsqueeze(0)
                    neg_emb = input2[torch.where(ids != ids[i])]
                    all_embs = torch.cat([pos_emb, neg_emb], dim=0)

                inner_loss = torch.sum(self.cos(anchor_emb, pos_emb))/temperature
                latter_term = torch.sum(torch.exp(self.cos(anchor_emb, all_embs)/temperature))
                inner_loss -= 1*torch.log(latter_term)
                inner_loss = -1*inner_loss/pos_emb.shape[0]
                loss_i += inner_loss/input1.shape[0]

                calc_ids.add(ids[i])
            else:
                continue

        return loss_i



class Model(torch.nn.Module):
    def __init__(self, args, item_num, bert_model, image_net, pop_prob_list):
        super(Model, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len #修正
        self.pop_prob_list = torch.FloatTensor(pop_prob_list) # serve for debias

        # various benchmark
        if "sasrec" in args.benchmark:
            self.user_encoder = User_Encoder_SASRec(item_num=item_num, args=args)
        elif "nextit" in args.benchmark:
            self.user_encoder = User_Encoder_NextItNet(args=args)
        elif "grurec"  in args.benchmark:
            self.user_encoder = User_Encoder_GRU4Rec(args=args)

        # various encoders
        if "CV" in args.item_tower or "modal" in args.item_tower:
            self.cv_encoder = VisionEncoder(args=args, image_net=image_net)

        if "text" in args.item_tower or "modal" in args.item_tower:
            self.text_encoder = TextEncoder(args=args, bert_model=bert_model)

        if  "ID" in args.item_tower:
            self.id_encoder = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
            xavier_normal_(self.id_encoder.weight.data)

        if "modal" in args.item_tower:
            # various fusion methods
            if args.fusion_method == 'sum':
                self.fusion_module = SumFusion(args=args)
            elif args.fusion_method == 'concat':
                self.fusion_module = ConcatFusion(args=args)
            elif args.fusion_method == 'film':
                self.fusion_module = FiLM(args=args, x_film=True)
            elif args.fusion_method == 'gated':
                self.fusion_module = GatedFusion(args=args, x_gate=True)
            elif args.fusion_method == 'weight_sum':
                self.fusion_module = Wighted_Sum_fushion(args=args)
            elif args.fusion_method == 'co_att' :
                cofig_path = "/ceph-jd/pub/jupyter/yuanfj/notebooks/liyouhua/Recsys_Inbatch_OGM_MM/TextEncoders/bert-base-uncased"
                self.fusion_module = CoAttention.from_pretrained(cofig_path, args=args)
                # self.fusion_module = CoAttention.from_pretrained("bert-base-uncased", args=args)
            elif args.fusion_method == 'merge_attn':
                cofig_path = "/ceph-jd/pub/jupyter/yuanfj/notebooks/liyouhua/Recsys_Inbatch_OGM_MM/TextEncoders/bert-base-uncased"
                self.fusion_module = MergedAttention.from_pretrained(cofig_path, args=args)
                # self.fusion_module = MergedAttention.from_pretrained("bert-base-uncased",args=args)

        # loss
        self.criterion = nn.CrossEntropyLoss()
        
        # self.temp = torch.nn.Parameter(torch.tensor(args.UI_temp, dtype=torch.float32), requires_grad=True) # 设置初始值  learnable 
        
        self.temp = args.UI_temp
        
        self.metric_learning_8 = metric_learning_8(args.embedding_dim)


    def forward(self, sample_items_id, sample_items_text, sample_items_CV, log_mask, local_rank, args):
        
        loss, loss_i, align, uniform = torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0)
        
        # 计算de-bias
        self.pop_prob_list = self.pop_prob_list.to(local_rank)
        debias_logits = torch.log(self.pop_prob_list[sample_items_id.view(-1)])

        if "modal" in args.item_tower:
            # text mask
            batch_size, num_words = sample_items_text.shape
            num_words = num_words // 2
            text_mask = torch.narrow(sample_items_text, 1, num_words, num_words)

            # text and img last hidden states
            hidden_states_text = self.text_encoder(sample_items_text.long())
            # print("hidden_states_text", hidden_states_text.size())
            hidden_states_CV = self.cv_encoder(sample_items_CV)
 

            if args.fusion_method in ['sum', 'concat', 'film', 'gated', "weight_sum"]:
                text_mask_expanded = text_mask.unsqueeze(-1).expand(hidden_states_text.size()).float()
                hidden_states_text = torch.sum(hidden_states_text * text_mask_expanded, 1) / torch.clamp(text_mask_expanded.sum(1), min=1e-9)
                hidden_states_CV = torch.mean(hidden_states_CV, dim=1)  # mean
                score_embs = self.fusion_module(hidden_states_text, hidden_states_CV)
                
                loss_i = self.metric_learning_8(args, sample_items_id, log_mask, hidden_states_text, hidden_states_CV, temperature=self.temp)
                
            if args.fusion_method in ['co_att', 'merge_attn']:
                CV_mask = torch.ones(hidden_states_CV.size()[0], hidden_states_CV.size()[1]).to(local_rank)
                score_embs = self.fusion_module(hidden_states_text, text_mask, hidden_states_CV, CV_mask,local_rank)
                
                text_mask_expanded = text_mask.unsqueeze(-1).expand(hidden_states_text.size()).float()
                hidden_states_text = torch.sum(hidden_states_text * text_mask_expanded, 1) / torch.clamp(text_mask_expanded.sum(1), min=1e-9)
                hidden_states_CV = torch.mean(hidden_states_CV, dim=1)  # mean
                loss_i = self.metric_learning_8(args, sample_items_id, log_mask, hidden_states_text, hidden_states_CV, temperature=self.temp)
                
                align, uniform = align_uniformity(log_mask,hidden_states_text,hidden_states_CV)

        if "text-only" in args.item_tower:
            batch_size, num_words = sample_items_text.shape
            num_words = num_words // 2
            text_mask = torch.narrow(sample_items_text, 1, num_words, num_words)
            hidden_states_text = self.text_encoder(sample_items_text.long())
            text_mask_expanded = text_mask.unsqueeze(-1).expand(hidden_states_text.size()).float().to(local_rank)       
            score_embs = torch.sum(hidden_states_text * text_mask_expanded, 1) / torch.clamp(text_mask_expanded.sum(1), min=1e-9) # mean

        if "CV-only" in args.item_tower:
            hidden_states_CV = self.cv_encoder(sample_items_CV)
            score_embs = torch.mean(hidden_states_CV, dim=1)  
            

        if "ID" in args.item_tower:
            score_embs = self.id_encoder(sample_items_id)

        input_embs = score_embs.view(-1, self.max_seq_len+1, self.args.embedding_dim)

         # various benchmark
        if "sasrec" in args.benchmark:
             prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask, local_rank)
        elif "nextit" in args.benchmark:
            prec_vec = self.user_encoder(input_embs[:, :-1, :])
        elif "grurec"  in args.benchmark:
            prec_vec = self.user_encoder(input_embs[:, :-1, :])

        prec_vec = prec_vec.contiguous().view(-1, self.args.embedding_dim)  # (bs*max_seq_len, ed)

        

        ######################################  IN-BATCH CROSS-ENTROPY LOSS  ######################################
        

        bs = log_mask.size(0)
        ce_label = torch.tensor(
            [i * self.max_seq_len + i + j for i in range(bs) for j in range(1, self.max_seq_len + 1)],
            dtype=torch.long).to(local_rank)

        logits = torch.matmul(prec_vec, score_embs.t())  # (batch_size*max_seq_len, batch_size*(max_seq_len+1))

        # 增加debias
        logits = logits - debias_logits

        logits[:, torch.cat((log_mask, torch.ones(log_mask.size(0)).unsqueeze(-1).to(local_rank)), dim=1).view(-1) == 0] = -1e4
        logits = logits.view(bs, self.max_seq_len, -1)
        id_list = sample_items_id.view(bs, -1)  # sample_items_id (bs, max_seq_len)
        for i in range(bs):
            reject_list = id_list[i]  # reject_list (max_seq_len)
            u_ids = sample_items_id.repeat(self.max_seq_len).expand((len(reject_list), -1))
            reject_mat = reject_list.expand((u_ids.size(1), len(reject_list))).t()
            mask_mat = (u_ids == reject_mat).any(axis=0).reshape(logits[i].shape)
            for j in range(self.max_seq_len):
                mask_mat[j][i * (self.max_seq_len + 1) + j + 1] = False
            logits[i][mask_mat] = -1e4


        indices = torch.where(log_mask.view(-1) != 0)
        logits = logits.view(bs * self.max_seq_len, -1)
        loss = self.criterion(logits[indices], ce_label[indices])
        

        loss_all =  (1 - args.UI_alpha) * loss +  args.UI_alpha  * loss_i


        return loss_all, loss, loss_i, align, uniform

