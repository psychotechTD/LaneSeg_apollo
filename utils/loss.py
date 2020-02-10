import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.DiceLoss
        elif mode == 'diceplusce':
            return  self.DiceplusCE
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def BinaryDiceLoss(self, logit, target, smooth=1, p=2, reduction='mean'):
        logit = logit.contiguous().view(logit.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = 2*torch.sum(torch.mul(logit, target), dim=1) + smooth
        den = torch.sum(logit.pow(p) + target.pow(p), dim=1) + smooth

        loss = 1 - num / den

        return loss.mean()
        # if self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # elif self.reduction == 'none':
        #     return loss

    def DiceLoss(self, logit, target):
        total_loss = 0
        logit = F.softmax(logit, dim=1)
        for i in range(logit.shape[1]):
            if i != self.ignore_index:
                dice_loss = self.BinaryDiceLoss(logit[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shapep[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss
        return total_loss/target.shape[1]


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    input = input.unsqueeze(dim=1)
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    result = result.cuda()

    return result




if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(2, 8, 512, 512).cuda()
    b = torch.rand(2, 512, 512).cuda()
    print("ce:", loss.CrossEntropyLoss(a, b).item())
    print("fc:", loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print("fc:", loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())


