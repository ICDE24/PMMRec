import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_


class SumFusion(nn.Module):
    def __init__(self, args):
        super(SumFusion, self).__init__()
        
        self.fc_x = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_y = nn.Linear(args.embedding_dim, args.embedding_dim)

    def forward(self, x, y):

        output = self.fc_x((x)) + self.fc_y((y))

        return output

class ConcatFusion(nn.Module):
    def __init__(self,args):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(args.embedding_dim * 2, args.embedding_dim)
        

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, args, x_film=True):
        super(FiLM, self).__init__()

        self.dim = args.embedding_dim
        self.fc = nn.Linear(args.embedding_dim, 2 * args.embedding_dim)
        self.fc_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.x_film = x_film
                                                            

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return output

class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, args, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_y = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
                                                            
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return  output

    
class Wighted_Sum_fushion(torch.nn.Module):
    def __init__(self, args):
        super(Wighted_Sum_fushion, self).__init__()
        
        self.cv_embed = nn.ReLU(nn.Linear(args.embedding_dim, args.embedding_dim)) 
        self.text_embed = nn.ReLU(nn.Linear(args.embedding_dim, args.embedding_dim)) 
        self.dropout = nn.Dropout(args.drop_rate)

        self.layer_norm = nn.LayerNorm(args.embedding_dim, eps=1e-6)
        self.dense = nn.Linear(2 * args.embedding_dim, 1)
        self.activate = nn.Sigmoid()

    def forward(self, input_embs_text, input_embs_CV):
        
        # 投影    
        input_embs_text = self.dropout(self.text_embed(input_embs_text))
        input_embs_CV = self.dropout(self.cv_embed(input_embs_CV))

        # # 归一化，抹平两种模态的数量上的差距
        # input_embs_text = self.layer_norm(input_embs_text)
        # input_embs_CV = self.layer_norm(input_embs_CV)

        # fushion
        input_embs_all_CV_text_concat = torch.cat([input_embs_text, input_embs_CV], 1)
        alpha = self.activate(self.dense(input_embs_all_CV_text_concat)) #加权
        input_embs_all_CV_text_concat = alpha * input_embs_text + (1 - alpha) * input_embs_CV  # 加权融合

        return input_embs_all_CV_text_concat

