# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
import torch
# from triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from _aagcn.triplet_loss import TripletLoss, CrossEntropyLabelSmooth



def make_loss(num_classes):    # modified by gu
    sampler = 'softmax_triplet'
    triplet = TripletLoss(0.3)  # triplet loss
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # LabelSmooth

    def loss_func(score, feat_independent, feat_related, A_pre, target):
        one_hot = torch.index_select(torch.eye(num_classes), dim=0, index=target.cpu())
        A_gt = torch.mm(one_hot, torch.transpose(one_hot, 0, 1)).float().cuda()
        loss_A = torch.norm(A_pre - A_gt)
        # print('loss_A =====>>>>> ', loss_A)
        return xent(score, target) +  triplet(feat_independent, target)[0] +  triplet(feat_related, target)[0] +  0.05*loss_A,  loss_A


    '''
    else:
        if sampler == 'softmax':
            def loss_func(score, feat, target):
                return F.cross_entropy(score, target)
        elif cfg.DATALOADER.SAMPLER == 'triplet':
            def loss_func(score, feat, target):
                return triplet(feat, target)[0]
        elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
            def loss_func(score, feat, target):
                if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                    if cfg.MODEL.IF_LABELSMOOTH == 'on':
                        return xent(score, target) + triplet(feat, target)[0]
                    else:
                        return F.cross_entropy(score, target) + triplet(feat, target)[0]
                else:
                    print('expected METRIC_LOSS_TYPE should be triplet'
                          'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
        else:
            print('expected sampler should be softmax, triplet or softmax_triplet, '
                  'but got {}'.format(cfg.DATALOADER.SAMPLER))
    '''
    return loss_func
