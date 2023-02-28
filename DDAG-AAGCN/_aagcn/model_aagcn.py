import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from resnet import resnet50, resnet18
import torch.nn.functional as F
import math
from attention import GraphAttentionLayer, IWPA

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class embed_net(nn.Module):
    def __init__(self,  low_dim,  class_num, drop=0.2, part = 3, alpha=0.2, nheads=4, arch='resnet50', wpa = False, adjacency_method='metric'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048
        self.dropout = drop
        self.part = part
        self.lpa = wpa

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.wpa = IWPA(pool_dim, part)


        self.attentions = [GraphAttentionLayer(pool_dim, low_dim, dropout=drop, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(low_dim * nheads, class_num, dropout=drop, alpha=alpha, concat=False)
        '''
        ###########################################################################################################
        '''
        from _aagcn.adjacency_matrix import Metric_Learning_based_Adjacency_Matrix,Attention_based_Adjacency_Matrix
        from _aagcn.GCN_layer import GCNNet

        num_classes = class_num
        in_planes = pool_dim
        gcn_out_dim = 512

        if adjacency_method == 'metric':
            self.get_adjacency = Metric_Learning_based_Adjacency_Matrix(in_features=in_planes, alpha=1.0)
        elif adjacency_method == 'attention':
            self.get_adjacency = Attention_based_Adjacency_Matrix(in_features=in_planes, alpha=1.0)
        self.gcn_layer = GCNNet(in_features=in_planes, out_features=gcn_out_dim)

        self.bottleneck_gcn = nn.BatchNorm1d(in_planes+gcn_out_dim)
        self.classifier_gcn = nn.Linear(in_planes+gcn_out_dim, num_classes, bias=False)
        self.bottleneck_gcn.bias.requires_grad_(False)  # no shift
        self.bottleneck_gcn.apply(weights_init_kaiming)
        self.classifier_gcn.apply(weights_init_classifier)

        ''''''

    def forward(self, x1, x2, modal=0, cpa = False, gcn=False):
        # domain specific block
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared four blocks
        x = self.base_resnet(x)     # x [64,64,72,36] -> [64,2048,18,9]
        x_pool = self.avgpool(x)    # x_pool [64,2048,1,1]
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))    # [64,2048]
        feat  = self.bottleneck(x_pool)                         # [64,2048]

        '''
        ###########################################################################################################
        '''
        x_independent = feat
        adjacency, adjacency_pre = self.get_adjacency(x_independent)
        x_related = self.gcn_layer(adjacency, x_independent)
        feat = self.bottleneck_gcn(torch.cat((x_independent, x_related), dim=1))
        if self.training:
            return self.classifier_gcn(feat), x_independent, x_related, adjacency_pre
        else:
            return feat  # , adjacency_pre

        '''
        ###########################################################################################################
        '''

        '''
        if self.lpa:
            # intra-modality weighted part attention
            feat_att = self.wpa(x, feat, 1, self.part)

        if self.training:
            # cross-modality graph attention
            x_g = F.dropout(x_pool, self.dropout, training=self.training)
            x_g = torch.cat([att(x_g, adj) for att in self.attentions], dim=1)
            x_g = F.dropout(x_g, self.dropout, training=self.training)
            x_g = F.elu(self.out_att(x_g, adj))
            return x_pool, self.classifier(feat), self.classifier(feat_att), F.log_softmax(x_g, dim=1)
        else:
            return self.l2norm(feat), self.l2norm(feat_att)
        '''




if __name__ == '__main__':
    net = embed_net( 512, 395, adjacency_method='metric')
    print(net)
