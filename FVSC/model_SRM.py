"""
@Brief: SRM model
"""


import numpy as np
import random
import torch
import time
from dataset_FVSC import *
from torch.utils.data import DataLoader
from Quantization import *
from channel import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):

    def __init__(self, head, d_model, dropout=0.1, mask=None, side="tx"):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head  
        self.head = head  
        self.d_model = d_model  
        self.linear_query = nn.Linear(128, 128)
        self.linear_key = nn.Linear(128, 128)
        self.linear_value = nn.Linear(128, 128)
        if side == "tx":
            self.linear_out = nn.Sequential(nn.Linear(128, 32), nn.Linear(32, 8), nn.Linear(8, 2), nn.Linear(2, 1), nn.Sigmoid())
        else:
            self.linear_out = nn.Sequential(nn.Linear(128, 32), nn.Linear(32, 8), nn.Linear(8, 2), nn.Linear(2, 1))

        self.dropout = nn.Dropout(p=dropout)
        self.attn = None
        self.mask = mask

        self.updim1 = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 32), nn.Linear(32, 128))
        self.updim2 = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 32), nn.Linear(32, 128))
        self.updim3 = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 32), nn.Linear(32, 128))

    def self_attention(self, query, key, value, dropout=None, mask=None):
        d_k = query.size(-1)  
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  
        self_attn = F.softmax(scores, dim=-1)

        return torch.matmul(self_attn, value), self_attn

    def forward(self, query, key, value):
        n_batch = query.size(0)
        query = self.updim1(query)  
        key = self.updim2(key)
        value = self.updim3(value)
        positional_encoding = torch.unsqueeze(torch.unsqueeze(torch.linspace(0, 1, steps=64), dim=0), dim=2).repeat(n_batch, 1, 128).to(device)
        query += positional_encoding
        key += positional_encoding
        value += positional_encoding

        query = (self.linear_query(query)).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  
        x, self.attn = self.self_attention(query, key, value, mask=self.mask)  

        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k) 
        x = self.linear_out(x)   

        return x


class Semantic_Remapping(nn.Module):
    """
    :Brief: Semantic remapping module based on multi-head attention.
    """
    def __init__(self, batch):
        super(Semantic_Remapping, self).__init__()
        self.Q = QuantizationLayer(2)
        self.DQ = DequantizationLayer(2)
        self.batch = batch
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.attention_tx = MultiHeadAttention(8, 128, dropout=0.1, mask=None, side="tx")
        self.attention_rx = MultiHeadAttention(8, 128, dropout=0.1, mask=None, side="rx")
        self.ratio = 0

    def Remapping(self, x, H_f):
        zero_pad = torch.zeros(self.batch, 44).to(device)
        x = torch.cat((x, zero_pad), dim=1)  
        CdI = torch.tensor(1./np.linalg.cond(H_f), dtype=torch.float).to(device)
        CdI = CdI.repeat(self.batch, 71) 
        x_ = torch.stack((x, CdI), dim=1) 
        z = torch.zeros((self.batch, 4544)).to(device)

        a = 0
        for i in range(71):
            tmp = time.time()
            z[:, 64*i:(i+1)*64] = torch.squeeze(  # 64 is the number of subcarrier
                self.attention_tx(query=(x_[:, :, 64 * i:(i + 1) * 64].permute(0, 2, 1)),
                                  key=(x_[:, :, 64 * i:(i + 1) * 64].permute(0, 2, 1)),
                                  value=(x_[:, :, 64 * i:(i + 1) * 64].permute(0, 2, 1))), dim=2)
            a += time.time() - tmp
        print("encoding:", a / 71)
        return z

    def Deremapping(self, x, H_f):
        CdI = torch.tensor(1./np.linalg.cond(H_f), dtype=torch.float).to(device)
        CdI = CdI.repeat(self.batch, 71) 
        x_ = torch.stack((x, CdI), dim=1)  
        z = torch.zeros((self.batch, 4544)).to(device)
        b = 0
        for i in range(71):
            tmp = time.time()
            z[:, 64 * i:(i + 1) * 64] = torch.squeeze(
                self.attention_rx(query=(x_[:, :, 64 * i:(i + 1) * 64].permute(0, 2, 1)),
                                  key=(x_[:, :, 64 * i:(i + 1) * 64].permute(0, 2, 1)),
                                  value=(x_[:, :, 64 * i:(i + 1) * 64].permute(0, 2, 1))), dim=2)
            b += time.time()-tmp
        print("decoding:", b / 71)
        z = z[:, :-44]
        return z

    def forward(self, semantic_feature, H_f):
        semantic_feature_ = self.Remapping(semantic_feature, H_f) 
        semantic_feature_ = self.Q(semantic_feature_)  
        ratio = random.uniform(0.09, 0.27)  # flip bit randomly
        CdI = torch.tensor(np.linalg.cond(H_f), dtype=torch.float) 
        CdI = CdI / torch.sum(CdI)  
        bitErr_num_perSub = torch.clip(torch.round(CdI * (128 * ratio) * 1.1), 0, 2).int()  

        tmp = torch.zeros((self.batch, 9088)).to(device)  
        for i in range(71):
            tmp_ = semantic_feature_[:, 128 * i:(i + 1) * 128]
            for j in range(64):
                if bitErr_num_perSub[j] == 0:
                    pass
                elif bitErr_num_perSub[j] == 1:
                    if torch.randint(0, 2, (1, 1)).item() == 0:
                        tmp_[0][j * 2] += 1
                    else:
                        tmp_[0][j * 2 + 1] += 1
                else:
                    tmp_[0][j * 2] += 1
                    tmp_[0][j * 2 + 1] += 1
            tmp[:, 128 * i:(i + 1) * 128] = tmp_

        semantic_feature_ = torch.where(tmp < 1.5, tmp,
                                        torch.zeros(tmp.shape).to(device))

        semantic_feature_ = self.DQ(semantic_feature_)  
        semantic_feature__ = self.Deremapping(semantic_feature_, H_f)

        loss = self.cal_loss(semantic_feature, semantic_feature__)

        return semantic_feature__, loss

    def cal_loss(self, x, y):
        loss = self.loss_fn(x, y)

        return loss


if __name__ == "__main__":
    model = Semantic_Remapping(1).to(device)
    model.eval()
    with torch.no_grad():
        inpu = torch.randint(0, 255, (1, 4500)).to(device)
        chan = Channel()
        input_frame = chan.gen_ch_2_2_UMi_NLOS()
        result = model(inpu, input_frame)[0]

