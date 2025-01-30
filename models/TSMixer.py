import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi

from layers.SelfAttention_Family import AttentionLayer, ProbAttention, FullAttention
import torch.nn.init as init



class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(*normalized_shape))  # 可学习的缩放参数

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=(-2, -1), keepdim=True) + self.eps)
        return self.scale * x / rms


class ResBlock(nn.Module):
    def __init__(self, configs, seq_len=96):
        super(ResBlock, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = seq_len
        self.d_model = configs.d_model


        self.channel_function = configs.channel_function
        self.temporal_function = configs.temporal_function

        if self.temporal_function == 'patch':
            self.temporal = nn.Sequential(
                RMSNorm([self.enc_in,self.seq_len]),
                nn.Linear(self.seq_len, configs.d_model),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, self.seq_len),
                nn.Dropout(configs.dropout)
            )
            self.patch = configs.patch
            self.patch_num = [self.seq_len // i for i in self.patch]
            self.decomp = torch.nn.ModuleList([series_decomp(i+1) for i in self.patch])
            self.temporal1 = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.patch_num[i],self.patch[i]]),
                nn.Linear(self.patch[i], self.patch[i]*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch[i]*4, self.patch[i]),
                nn.Dropout(configs.dropout)
            ) for i in range(len(self.patch))])
            self.temporal2 = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.patch[i],self.patch_num[i]]),
                nn.Linear(self.patch_num[i], self.patch_num[i]*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch_num[i]*4, self.patch_num[i]),
                nn.Dropout(configs.dropout)
            )  for i in range(len(self.patch))])
            self.temporal1_season = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.patch_num[i],self.patch[i]]),
                nn.Linear(self.patch[i], self.patch[i]*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch[i]*4, self.patch[i]),
                nn.Dropout(configs.dropout)
            ) for i in range(len(self.patch))])
            self.temporal2_season = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.patch[i],self.patch_num[i]]),
                nn.Linear(self.patch_num[i], self.patch_num[i]*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch_num[i]*4, self.patch_num[i]),
                nn.Dropout(configs.dropout)
            )  for i in range(len(self.patch))])
            self.linear = torch.nn.ModuleList([nn.Linear(self.seq_len, self.seq_len) 
                    for i in range(len(self.patch))])

        if self.temporal_function == 'normal':
            self.temporal = nn.Sequential(
                RMSNorm([self.enc_in,self.seq_len]),
                nn.Linear(self.seq_len, configs.d_model),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, self.seq_len),
                nn.Dropout(configs.dropout)
            )
        if self.temporal_function == 'down':
            self.kernel = configs.patch
            self.layers = len(self.kernel)
            self.temporal = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.enc_in, self.seq_len//self.kernel[i]]),
                nn.Linear(self.seq_len//self.kernel[i], configs.d_model ),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model , self.seq_len//self.kernel[i] ),
                nn.Dropout(configs.dropout),
            ) for i in range(self.layers)])
            self.linear = torch.nn.ModuleList([nn.Linear(self.seq_len//self.kernel[i], self.seq_len) 
                    for i in range(self.layers)])


        if self.channel_function == 'RNN':
            self.norm = RMSNorm([self.enc_in,self.seq_len])
            self.linear1 = nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Dropout(configs.d2),
            )
            self.lstm = torch.nn.LSTM(input_size=self.seq_len,hidden_size=self.seq_len,
                                    num_layers=1,batch_first=True, bidirectional=True)
            self.pro = nn.Sequential( 
                torch.nn.Linear(self.seq_len*2, configs.seq_len),
                nn.SiLU(),
                nn.Dropout(configs.d2), 
            )

        if self.channel_function == 'MLP':
            self.final_linear = nn.Sequential(
                RMSNorm([self.seq_len,self.enc_in]),
                nn.Linear(self.enc_in, self.enc_in//4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.enc_in//4, self.enc_in),
                nn.Dropout(configs.dropout),
            )  


    def forward(self, x):
        B, L, D = x.shape

        if self.temporal_function == 'patch':
            add = torch.zeros([B, L, D], device=x.device)
            for i in range(len(self.patch)):
                if self.patch[i] == 1:
                    add = x + self.temporal((x).transpose(1, 2)).transpose(1, 2)
                else:
                    season, x_group = self.decomp[i](x)
                    x_group = x_group.permute(0,2,1)
                    x_group = x_group.reshape(B, D, self.patch_num[i], self.patch[i])
                    x_group = x_group + self.temporal1[i](x_group)
                    x_group = x_group.permute(0,1,3,2)
                    x_group = x_group + self.temporal2[i](x_group)
                    x_group = x_group.permute(0,1,3,2).reshape(B, D, -1).permute(0,2,1)
                    season = season.permute(0,2,1)
                    season = season.reshape(B, D, self.patch_num[i], self.patch[i])
                    season = season + self.temporal1_season[i](season)
                    season = season.permute(0,1,3,2)
                    season = season + self.temporal2_season[i](season)
                    season = season.permute(0,1,3,2).reshape(B, D, -1).permute(0,2,1)
                    add = add + self.linear[i]((x_group + season).permute(0,2,1)).permute(0,2,1) 
            x = add/(len(self.patch))

        if self.temporal_function == 'normal':
            x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)

        if self.temporal_function == 'down':
            add = torch.zeros([B, L, D], device=x.device)
            for i in range(self.layers):
                tmp =  torch.nn.AvgPool1d(kernel_size=self.kernel[i])(x.transpose(1, 2)) + torch.nn.MaxPool1d(kernel_size=self.kernel[i])(x.transpose(1, 2))  
                tmp = tmp + self.temporal[i](tmp)
                tmp = self.linear[i](tmp)
                add = add + tmp.permute(0,2,1)
            x = add/(self.layers)

        if self.channel_function == 'MLP':
            x = x + self.final_linear(x)
        if self.channel_function == 'RNN':
            x = x.permute(0,2,1)
            h0 = torch.randn(2, B, self.seq_len, device=x.device)
            c0 = torch.randn(2, B, self.seq_len, device=x.device)
            x = x + torch.mul(self.linear1(x), self.pro(self.lstm(self.norm(x), (h0,c0))[0]))
            x = x.permute(0,2,1)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        self.model = nn.ModuleList([ResBlock(configs, seq_len=self.seq_len)
                                    for _ in range(configs.e_layers)])

        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        self.use_norm = configs.use_norm

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch, seq, channel = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()       
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection((x_enc ).transpose(1, 2)).transpose(1, 2)
        
        if self.use_norm:
            enc_out = enc_out *stdev + means

        return enc_out 

    # def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
    #     # Normalization from Non-stationary Transformer
    #     means = x_enc.mean(1, keepdim=True).detach()
    #     x_enc = x_enc - means
    #     stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    #     x_enc /= stdev

    #     _, L, N = x_enc.shape

    #     for i in range(self.layer):
    #         x_enc = self.model[i](x_enc)
    #     enc_out = self.projection((x_enc ).transpose(1, 2)).transpose(1, 2)

    #     enc_out = enc_out *stdev + means

    #     return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out  = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]   # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
