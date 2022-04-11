import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable
import copy
import Linformer as LF

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=(1, 2, 2))
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=(1, 2, 2))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class EncoderLayer_inRes(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x1 ,x2):
        x1_ = self.norm_1(x1)
        x2 = self.norm_2(x2)
        x1 = x1 + self.dropout_1(self.attn(x1_, x2, x2))
        x1_ = self.norm_2(x1)
        x1 = x1 + self.dropout_2(self.ff(x1_))
        return x1

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Encoder_inRes(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer_inRes(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, x1, x2):
        x1 = self.pe(x1)
        x2 = self.pe(x2)
        for i in range(self.N):
            x1 = self.layers[i](x1, x2)
        return self.norm(x1)

class ResNet_Transformer(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=2,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1,
                 heads=4,
                 dim=512):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.dim = dim

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=(2, 2, 2))
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=(2, 2, 2))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc_1 = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.fc_1 = nn.Linear(64 + 6, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.transformer = Encoder_inRes(d_model=dim, heads=heads, dropout=0.8, N=1)
        self.dropout = nn.Dropout(p=0.8)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc_0 = nn.Linear(self.dim, 64)
        # self.fc_2 = nn.Linear(block_inplanes[3] * block.expansion, 1)
        self.fc_2 = nn.Linear(64 + 6, 1)
        self.fc_3 = nn.Linear(64 * 4, 1)
        self.fc_x1 = nn.Linear(self.dim + 6, n_classes)
        self.fc_x2 = nn.Linear(self.dim + 6, n_classes)
        self.fc_x3 = nn.Linear(self.dim + 6, n_classes)
        self.fc_x4 = nn.Linear(self.dim + 6, n_classes)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3, x4, info):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        if not self.no_max_pool:
            x1 = self.maxpool(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = x1.view(x1.size(0), x1.size(1), -1).transpose(1, 2)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        if not self.no_max_pool:
            x2 = self.maxpool(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        x2 = x2.view(x2.size(0), x2.size(1), -1).transpose(1, 2)

        x3 = self.conv1(x3)
        x3 = self.bn1(x3)
        x3 = self.relu(x3)
        if not self.no_max_pool:
            x3 = self.maxpool(x3)
        x3 = self.layer1(x3)
        x3 = self.layer2(x3)
        x3 = self.layer3(x3)
        x3 = self.layer4(x3)
        x3 = x3.view(x3.size(0), x3.size(1), -1).transpose(1, 2)

        x4 = self.conv1(x4)
        x4 = self.bn1(x4)
        x4 = self.relu(x4)
        if not self.no_max_pool:
            x4 = self.maxpool(x4)
        x4 = self.layer1(x4)
        x4 = self.layer2(x4)
        x4 = self.layer3(x4)
        x4 = self.layer4(x4)
        x4 = x4.view(x4.size(0), x4.size(1), -1).transpose(1, 2)

        cls_x1 = self.fc_x1(torch.cat((self.dropout(x1.mean(dim=1)), info), dim=1))
        cls_x2 = self.fc_x2(torch.cat((self.dropout(x2.mean(dim=1)), info), dim=1))
        cls_x3 = self.fc_x3(torch.cat((self.dropout(x3.mean(dim=1)), info), dim=1))
        cls_x4 = self.fc_x4(torch.cat((self.dropout(x4.mean(dim=1)), info), dim=1))

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.transformer(x, x)
        x = x.transpose(1, 2).reshape(-1, x.size(1))
        x = self.fc_3(x).view(-1, self.dim)

        x = self.dropout(x)
        x = self.fc_0(x)
        x = self.dropout2(x)
        x = torch.cat((x, info), dim=-1)
        cls = self.fc_1(x)
        # l = torch.exp(self.fc_2(x))
        l = self.fc_2(x)

        # x = F.sigmoid(x)

        return l, cls, cls_x1, cls_x2, cls_x3, cls_x4

class Linformer_Layer(nn.Module):
    """
    A wrapper function to accept LM tasks, inspired by https://github.com/lucidrains/sinkhorn-transformer
    """
    def __init__(self, num_tokens, input_size, channels,
                       dim_k=64, dim_ff=1024, dim_d=None,
                       dropout_ff=0.1, dropout_tokens=0.1, nhead=4, depth=1, ff_intermediate=None,
                       dropout=0.05, activation="gelu", checkpoint_level="C0",
                       parameter_sharing="layerwise", k_reduce_by_layer=0, full_attention=False,
                       include_ff=True, w_o_intermediate_dim=None, emb_dim=None,
                       return_emb=False, decoder_mode=False, causal=False, method="learnable"):
        super(Linformer_Layer, self).__init__()
        emb_dim = channels if emb_dim is None else emb_dim

        self.input_size = input_size

        self.to_token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = LF.PositionalEmbedding(emb_dim)
        self.linformer = LF.Linformer(input_size, channels, dim_k=dim_k,
                                   dim_ff=dim_ff, dim_d=dim_d, dropout_ff=dropout_ff,
                                   nhead=nhead, depth=depth, dropout=dropout, ff_intermediate=ff_intermediate,
                                   activation=activation, checkpoint_level=checkpoint_level, parameter_sharing=parameter_sharing,
                                   k_reduce_by_layer=k_reduce_by_layer, full_attention=full_attention, include_ff=include_ff,
                                   w_o_intermediate_dim=w_o_intermediate_dim, decoder_mode=decoder_mode, causal=causal, method=method)

        if emb_dim != channels:
            self.linformer = LF.ProjectInOut(self.linformer, emb_dim, channels)

        self.dropout_tokens = nn.Dropout(dropout_tokens)

    def forward(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len), and all items are ints from [0, num_tokens-1]
        """
        # tensor = self.to_token_emb(tensor)
        tensor = self.pos_emb(tensor).type(tensor.type()) + tensor
        tensor = self.dropout_tokens(tensor)
        tensor = self.linformer(tensor, **kwargs)
        return tensor

class ResNet_Linformer(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=2,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1,
                 heads=4,
                 dim=512):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.dim = dim

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=(2, 2, 2))
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=(2, 2, 2))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc_1 = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.fc_1 = nn.Linear(64 + 6, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        self.dropout = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc_0 = nn.Linear(self.dim, 64)
        self.lf1 = Linformer_Layer(num_tokens=1, input_size=32**3*4, channels=64)
        self.lf2 = Linformer_Layer(num_tokens=1, input_size=16**3*4, channels=128)
        self.lf3 = Linformer_Layer(num_tokens=1, input_size=8**3*4, channels=256)
        self.lf4 = Linformer_Layer(num_tokens=1, input_size=64*4, channels=512)

        self.fc_2 = nn.Linear(64 + 6, 1)
        self.fc_3 = nn.Linear(64 * 4, 1)
        self.fc_x1 = nn.Linear(self.dim + 6, n_classes)
        self.fc_x2 = nn.Linear(self.dim + 6, n_classes)
        self.fc_x3 = nn.Linear(self.dim + 6, n_classes)
        self.fc_x4 = nn.Linear(self.dim + 6, n_classes)

    def linformer_layer(self, x1, x2, x3, x4, lf):
        b, c, d, h, w = x1.size()
        x1 = x1.view(x1.size(0), x1.size(1), -1).transpose(1, 2)
        x2 = x2.view(x2.size(0), x2.size(1), -1).transpose(1, 2)
        x3 = x3.view(x3.size(0), x3.size(1), -1).transpose(1, 2)
        x4 = x4.view(x4.size(0), x4.size(1), -1).transpose(1, 2)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = lf(x).transpose(1, 2)
        x1 = x[:, :, 0:d * h * w].view(b, c, d, h, w)
        x2 = x[:, :, d * h * w:2 * d * h * w].view(b, c, d, h, w)
        x3 = x[:, :, 2 * d * h * w:3 * d * h * w].view(b, c, d, h, w)
        x4 = x[:, :, 3 * d * h * w:4 * d * h * w].view(b, c, d, h, w)
        return x1, x2, x3, x4, x


    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3, x4, info):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        if not self.no_max_pool:
            x1 = self.maxpool(x1)
        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        if not self.no_max_pool:
            x2 = self.maxpool(x2)
        x3 = self.conv1(x3)
        x3 = self.bn1(x3)
        x3 = self.relu(x3)
        if not self.no_max_pool:
            x3 = self.maxpool(x3)
        x4 = self.conv1(x4)
        x4 = self.bn1(x4)
        x4 = self.relu(x4)
        if not self.no_max_pool:
            x4 = self.maxpool(x4)

        x1 = self.layer1(x1)
        x2 = self.layer1(x2)
        x3 = self.layer1(x3)
        x4 = self.layer1(x4)
        _x1, _x2, _x3, _x4, _ = self.linformer_layer(x1, x2, x3, x4, self.lf1)
        x1, x2, x3, x4 = x1 + _x1, x2 + _x2, x3 + _x3, x4 + _x4

        x1 = self.layer2(x1)
        x2 = self.layer2(x2)
        x3 = self.layer2(x3)
        x4 = self.layer2(x4)
        _x1, _x2, _x3, _x4, _ = self.linformer_layer(x1, x2, x3, x4, self.lf2)
        x1, x2, x3, x4 = x1 + _x1, x2 + _x2, x3 + _x3, x4 + _x4

        x1 = self.layer3(x1)
        x2 = self.layer3(x2)
        x3 = self.layer3(x3)
        x4 = self.layer3(x4)
        _x1, _x2, _x3, _x4, _ = self.linformer_layer(x1, x2, x3, x4, self.lf3)
        x1, x2, x3, x4 = x1 + _x1, x2 + _x2, x3 + _x3, x4 + _x4

        x1 = self.layer4(x1)
        x2 = self.layer4(x2)
        x3 = self.layer4(x3)
        x4 = self.layer4(x4)

        cls_x1 = self.fc_x1(torch.cat((self.dropout(x1.mean(dim=(2, 3, 4))), info), dim=1))
        cls_x2 = self.fc_x2(torch.cat((self.dropout(x2.mean(dim=(2, 3, 4))), info), dim=1))
        cls_x3 = self.fc_x3(torch.cat((self.dropout(x3.mean(dim=(2, 3, 4))), info), dim=1))
        cls_x4 = self.fc_x4(torch.cat((self.dropout(x4.mean(dim=(2, 3, 4))), info), dim=1))

        x1, x2, x3, x4, x = self.linformer_layer(x1, x2, x3, x4, self.lf4)
        # x = x.reshape(-1, x.size(2))
        x = self.fc_3(x).view(-1, self.dim)

        x = self.dropout(x)
        x = self.fc_0(x)
        x = self.dropout2(x)
        x = torch.cat((x, info), dim=-1)
        cls = self.fc_1(x)
        # l = torch.exp(self.fc_2(x))
        l = self.fc_2(x)

        # x = F.sigmoid(x)

        return l, cls, cls_x1, cls_x2, cls_x3, cls_x4

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

def generate_model_BraTS(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet_Transformer(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet_Transformer(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet_Transformer(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet_Transformer(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet_Transformer(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet_Transformer(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet_Transformer(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

def generate_model_BraTS_LF(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet_Linformer(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet_Linformer(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet_Linformer(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet_Linformer(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet_Linformer(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet_Linformer(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet_Linformer(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model











