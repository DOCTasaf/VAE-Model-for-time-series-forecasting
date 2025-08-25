

from torchinfo import summary
# from torchsummary import summary
from torch import nn, optim
import torch.nn.functional as F


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import scipy.io as sio
import matplotlib.pyplot as plt

device = torch.device('cpu') 

### temporal attention
class Time_att(nn.Module):
    def __init__(self, dim_input, dropout, num_head):
        super(Time_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x):
        x = x.transpose(-3, -1)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x)).transpose(-3, -1)
            k = self.dropout(self.key(x)).transpose(-3, -1)
            k = k.transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -1)
            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x.transpose(-3, -1) + result
        x = self.laynorm(x)
        return x


### space_attention
class space_att(nn.Module):
    def __init__(self, Input_len, dim_input, dropout, num_head):
        super(space_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x)).transpose(-3, -1)
            k = self.dropout(self.key(x)).transpose(-3, -1)
            k = k.transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -1)
            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)

            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.linear1(result)
        result = result.squeeze(-1)
        result = result.transpose(1, 3)
        return result


### space_attention2
class space_att2(nn.Module):
    def __init__(self, Input_len, dim_input, dropout, num_head):
        super(space_att2, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x.transpose(1, 3)
        result = 0.0
        q = self.dropout(self.query(x))
        k = self.dropout(self.key(x))
        k = k.transpose(-2, -1)
        v = self.dropout(self.value(x))
        kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)

        for i in range(self.num_head):

            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.linear1(result)
        result = result.squeeze(-1)
        result = result.transpose(1, 3)
        return result


### cross attention
class cross_att(nn.Module):
    def __init__(self, dim_input, dropout, num_head):
        super(cross_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x, x2):
        x = x.transpose(-3, -1)
        x2 = x2.transpose(-3, -1)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x2)).transpose(-3, -1)
            k = self.dropout(self.key(x)).transpose(-3, -1)
            k = k.transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -1)

            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x.transpose(-3, -1) + result
        x = self.laynorm(x)
        return x


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

def channel_attenstion(inputs, ratio):

    channel = int(inputs.shape[-1])  
    x_max = GlobalMaxPooling1D()(inputs)
    x_avg = GlobalAveragePooling1D()(inputs)

    # [None,c]==>[1,1,c]
    x_max = Reshape((1, channel))(x_max) 
    x_avg = Reshape((1, channel))(x_avg)  

 
    x_max = Conv1D(channel // ratio, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.1))(
        x_max)
    x_max = Activation('relu')(x_max)
    x_max = Conv1D(channel, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.1))(x_max)

    x_avg = Conv1D(channel // ratio, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.1))(
        x_avg)
    x_avg = Activation('relu')(x_avg)
    x_avg = Conv1D(channel, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.1))(x_avg)

    x = Add()([x_max, x_avg])


    x = Activation('sigmoid')(x)

    x = Multiply()([inputs, x])

    return x


# （2）空间注意力机制
def spatial_attention(inputs, kernel_size):
    # 在通道维度上做最大池化和平均池化[b,h,w,c]==>[b,h,w,1]
    x_max = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(inputs)  # 在通道维度求最大值
    x_avg = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(inputs)  # axis也可以为-1

    x = Concatenate(axis=-1)([x_max, x_avg])

    x = Conv1D(1, kernel_size=kernel_size, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.1))(x)

    x = Activation('sigmoid')(x)

    x = Multiply()([inputs, x])

    return x


# （3）CBAM注意力
def CBAM_attention(inputs, ratio, kernel_size):
    # 先经过通道注意力再经过空间注意力
    x = channel_attenstion(inputs, ratio)
    x = spatial_attention(x, kernel_size)
    return x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, int(gate_channels // reduction_ratio)),
            nn.ReLU(),
            nn.Linear(int(gate_channels // reduction_ratio), gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=int((kernel_size - 1) // 2), relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class cbam(nn.Module):
    def __init__(self, gate_channels, reduction=16, pool_types=['avg', 'max'], no_spatial=False):
        super(cbam, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    @staticmethod
    def get_module_name():
        return "cbam"

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)


def calculate_metrics(y_true, y_pred):

    metrics = {}

    # MAE: Mean Absolute Error
    metrics['MAE'] = np.mean(np.abs(y_true - y_pred))

    metrics['MSE'] = np.mean((y_true - y_pred) ** 2)

  
    metrics['RMSE'] = np.sqrt(metrics['MSE'])

 
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    metrics['R²'] = 1 - (ss_residual / ss_total)

    return metrics


class VAE(nn.Module):
    def __init__(self, Input_len, out_len, num_id, latent_dim):

        super(VAE, self).__init__()

        self.inputlen = Input_len
        self.out_len = out_len
        self.num_id = num_id

     
        # self.RevIN = RevIN(num_id)

        self.encoder_cnn = nn.Conv1d(in_channels=num_id, out_channels=64, kernel_size=3, padding=1)
        self.decoder_cnn = nn.ConvTranspose1d(in_channels=128, out_channels=num_id, kernel_size=3, padding=1)


        self.encoder_lstm = nn.LSTM(input_size=64, hidden_size=latent_dim, num_layers=1, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_size=latent_dim, hidden_size=128, num_layers=1, batch_first=True)


        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, latent_dim)

    def encode(self, x):

        x = x.transpose(1, 2)  
        x = F.relu(self.encoder_cnn(x))  
        x = x.transpose(1, 2)  

        _, (h_n, _) = self.encoder_lstm(x)  # [1, B, latent_dim]
        h_n = h_n.squeeze(0)  # [B, latent_dim]

        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
 
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        
        h = F.relu(self.fc_decode(z)).unsqueeze(1).repeat(1, self.out_len, 1)  # [B, L, latent_dim]

      
        h, _ = self.decoder_lstm(h)  # [B, L, 32]
        h = h.transpose(1, 2)  # [B, 32, L]
        h = self.decoder_cnn(h)  # [B, N, L]
        h = h.transpose(1, 2)  # [B, L, N]
        return h

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # [batch_size, input_len, num_id]
   
        # x = self.RevIN(x, 'norm')


        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        x_reconstructed = self.decode(z)
        # x_reconstructed = self.RevIN(x_reconstructed, 'denorm')

        # print(f"Reconstructed output shape: {x_reconstructed.shape}")  # [batch_size, out_len, num_id]

        return x_reconstructed, mu, logvar

def vae_loss(reconstructed_x, x, mu, logvar):
    
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='sum')

    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence

def create_sequences(data, H, L):
    sequences, targets = [], []
    for i in range(len(data) - H - L + 1):
        seq = data[i:i + H]
        target = data[i + H:i + H + L]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)




