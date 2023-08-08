
import torch
from torch import nn
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

from .text_encoders import TextEncoder
from .img_encoders import VisionEncoder
from .user_encoders import User_Encoder_NextItNet, User_Encoder_GRU4Rec, User_Encoder_SASRec

from .fushion_module import SumFusion, ConcatFusion, FiLM, GatedFusion

# from lxmert, two attention fusion methods of co- and merge-.
from .Attention_Fusion import CoAttention, MergedAttention

import numpy as np
from torch.nn import functional



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
            # if args.fusion_method == 'sum':
            #     self.fusion_module = SumFusion(args=args)
            # elif args.fusion_method == 'concat':
            #     self.fusion_module = ConcatFusion(args=args)
            # elif args.fusion_method == 'film':
            #     self.fusion_module = FiLM(args=args, x_film=True)
            # elif args.fusion_method == 'gated':
            #     self.fusion_module = GatedFusion(args=args, x_gate=True)
            # elif args.fusion_method == 'co_att' :
            #     self.fusion_module = CoAttention.from_pretrained("bert-base-uncased", args=args)
            # elif args.fusion_method == 'merge_attn':
            #     self.fusion_module = MergedAttention.from_pretrained("bert-base-uncased",args=args)
            self.fusion_module = MergedAttention.from_pretrained("bert-base-uncased",args=args)

        # loss
        self.criterion = nn.CrossEntropyLoss()



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
            hidden_states_CV = self.cv_encoder(sample_items_CV)
 

            if args.fusion_method in ['sum', 'concat', 'film', 'gated']:
                text_mask_expanded = text_mask.unsqueeze(-1).expand(hidden_states_text.size()).float()
                hidden_states_text = torch.sum(hidden_states_text * text_mask_expanded, 1) / torch.clamp(text_mask_expanded.sum(1), min=1e-9)
                hidden_states_CV = torch.mean(hidden_states_CV, dim=1)  # mean
                score_embs = self.fusion_module(hidden_states_text, hidden_states_CV)

            if args.fusion_method in ['co_att', 'merge_attn']:
                CV_mask = torch.ones(hidden_states_CV.size()[0], hidden_states_CV.size()[1]).to(local_rank)
                score_embs = self.fusion_module(hidden_states_text, text_mask, hidden_states_CV, CV_mask,local_rank)


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


        return loss

