import torch
import torchvision
import torch.nn as nn
import numpy as np

# from utils.hrnet import hrnet_w32

class Encoder(nn.Module):
    def __init__(self, encoder='hrnet', pretrained=True):
        super(Encoder, self).__init__()

        if encoder == 'swin':
            '''Swin Transformer encoder'''
            self.encoder = torchvision.models.swin_b(weights='DEFAULT')
            self.encoder.head = nn.GELU()
        elif encoder == 'hrnet':
            '''HRNet encoder'''
            self.encoder = hrnet_w32(pretrained=pretrained)
        else:
            raise NotImplementedError('Encoder not implemented')

    def forward(self, x):
        out = self.encoder(x)
        return out

class Self_Attn(nn.Module):
    """ Self attention Layer for Feature Map dimension, outputs B x out_dim x N """
    def __init__(self, q_in_dim, k_in_dim, v_in_dim, out_dim):
        super(Self_Attn, self).__init__()
        self.out_dim = out_dim

        self.query_conv = nn.Conv1d(in_channels=q_in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_in_dim, out_channels=out_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """ q, k, v: B x in_dim x N """
        proj_query = self.query_conv(q.permute(0, 2, 1))
        proj_key = self.key_conv(k.permute(0, 2, 1))
        energy = torch.bmm(proj_query, proj_key.permute(0, 2, 1))
        attention = self.softmax(energy / (self.out_dim ** 0.5))
        proj_value = self.value_conv(v.permute(0, 2, 1))
        out = torch.bmm(attention, proj_value)
        out = out.permute(0, 2, 1)
        return out

class Cross_Att(nn.Module):
    def __init__(self, sem_in_dim, part_in_dim, out_dim):
        super(Cross_Att, self).__init__()

        self.cross_attn_1 = Self_Attn(q_in_dim=sem_in_dim, k_in_dim=part_in_dim, v_in_dim=part_in_dim, out_dim=out_dim)
        self.cross_attn_2 = Self_Attn(q_in_dim=part_in_dim, k_in_dim=sem_in_dim, v_in_dim=sem_in_dim, out_dim=out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, sem_seg, part_seg):
        cross1 = self.cross_attn_1(sem_seg, part_seg, part_seg)
        cross2 = self.cross_attn_2(part_seg, sem_seg, sem_seg)
        out = cross1 * cross2
        out = self.layer_norm(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, encoder='hrnet'):
        super(Decoder, self).__init__()
        self.out_dim = out_dim
        if encoder == 'swin':
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.Softmax(1)
            )
        elif encoder == 'hrnet':
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                # nn.ReLU(),
                # nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.BatchNorm2d(out_dim),
                nn.Softmax(1)
            )
        else:
            raise NotImplementedError('Decoder not implemented')

    def forward(self, x):
        out = self.upsample(x)
        return out


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim=6890):
        super(Classifier, self).__init__()

        self.out_dim = out_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 4096, True),
            nn.ReLU(),
            nn.Linear(4096, out_dim, True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.classifier(x)
        return out.reshape(-1, self.out_dim)