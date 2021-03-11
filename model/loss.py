import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable, Function



def cross_entropy_loss_age(output, target, age):
    loss = F.cross_entropy(output, target.long(), reduction='none')

    mean = 9.0
    standard_deviation = 5.0
    n = torch.distributions.normal.Normal(torch.tensor(mean), torch.tensor(standard_deviation))
    # print(age)
    weight = n.log_prob(age).exp()
    # print(weight)

    # weight = (weight - weight.min())/(weight.max()-weight.min())
    # print(weight)

    # print(loss.size())

    # print(output.size(), target.size(), age.size(), loss.size())
    loss = loss*weight
    # print(loss.size())

    return torch.mean(loss)


def cross_entropy_loss_weight(output, target, quantile):
    weights = torch.Tensor([1,0.9,0.8,0.8,0.7,0.5,0.4,0.3,0.2,0.1]).cuda()

    loss = F.cross_entropy(output, target.long(), reduction='none')

    weights_batch = torch.index_select(weights, 0, quantile)

    loss = loss*weights_batch

    # raise
    return torch.mean(loss)


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target.long())

def bce_loss_niki(output, target):
    loss = F.binary_cross_entropy_with_logits(output, target)
    return loss

def get_corr():
    import pickle
    result = pickle.load(open('adj_corr.pickle', 'rb'))
    return torch.Tensor(result).float()

def get_ccc():
    res = np.load('ccc.npy')
    return torch.Tensor(res).float()

class BceLossSpecial(nn.Module):
    def __init__(self):
        super(BceLossSpecial, self).__init__()
        self.threshold = nn.Parameter(torch.Tensor([0.5])).cuda()

    def forward(self, output, target):
        t = target.clone().detach()

        t[t >= 0.5] = 1  # threshold to get binary labels
        t[t < 0.5] = 0

        weights = torch.abs(target - self.threshold)

        loss = F.binary_cross_entropy_with_logits(output, t, reduction='none')

        return loss.mul(weights).mean()


def bce_loss_special(output, target):
    t = target.clone().detach()

    t[t >= 0.5] = 1  # threshold to get binary labels
    t[t < 0.5] = 0

    weights = torch.abs(target-threshold)

    loss = F.binary_cross_entropy_with_logits(output, t)


    return loss


def correlation_loss(output):
    import numpy as np

    c = torch.Tensor(np.load('corr.npy')).cuda()
    import audtorch.metrics
    out_corr = torch.zeros(8,8).cuda()
    for i in range(output.size(1)):
        for j in range(output.size(1)):
            out_corr[i,j] = audtorch.metrics.functional.pearsonr(output[:,i].squeeze(),output[:,j].squeeze())

    # print(out_corr)
    # print(torch.mean(out_corr-c))
    loss = F.l1_loss(out_corr, c)

    # print(loss.size())
    return loss

# def helper(x,y):
#     return audtorch.metrics.functional.concordance_cc(output_continuous, torch.matmul(output_categorical, corr[:26,26:]))


def intercorrelation_loss(output_continuous, output_categorical):
    # corr = get_corr()[:26,26:].cuda()
    # corr[corr>0.4] = 1
    # corr[corr<0.4] = 0
    corr = get_ccc().cuda()
    corr[corr>0.01] = 1
    corr[corr<0.01] = 0
    import audtorch
    # print(output_continuous.size(), torch.matmul(output_categorical, corr[:26,:26]).size())

    corr_predicted = torch.zeros(26,3).cuda()
    for i in range(output_continuous.size(1)):
        for j in range(output_categorical.size(1)):
            corr_predicted[j,i] = audtorch.metrics.functional.concordance_cc(output_continuous[:, i], output_categorical[:,j])

    # corr_predicted[corr_predicted>0.4] = 1
    # corr_predicted[corr_predicted<0.4] = 0

    loss = F.binary_cross_entropy_with_logits(corr_predicted, corr)
    # loss = audtorch.metrics.functional.concordance_cc(output_continuous, torch.matmul(output_categorical, corr[:26,26:]))

    return loss


def bce_loss(output, target):

    t = target.clone().detach()

    # t[t >= 0.5] = 1  # threshold to get binary labels
    # t[t < 0.5] = 0
    # weights = torch.softmax(torch.abs(target-0.5), dim=1)

    # pos_weight = torch.Tensor([6.672937771345875, 10.111842105263158, 10.056706652126499, 5.1838111298482294, 3.0946308724832217, 4.87936507936508, 7.342356687898089, 9.112648221343873, 9.626304801670146, 14.253477588871716, 19.961038961038962, 6.554371002132196, 9.576323987538942, 24.723860589812332, 52.39772727272727, 53.616279069767444, 8.421917808219177, 17.4, 6.836174944403262, 15.845360824742269, 17.237383177570095, 13.423580786026202, 7.389423076923077, 17.83752417794971, 88.67307692307692, 23.28787878787879])

    loss = F.binary_cross_entropy_with_logits(output, t)
    # print(loss.item())
    # loss = loss + correlation_loss(output)
    # print(loss.item())
    return loss

def combined_loss(output, target):
    l = F.mse_loss(output, target)

    l += bce_loss(output, target)
    # from model.bpmll_pytorch import BPMLLLoss
    # l += BPMLLLoss()(output, target)

    return l

def cross_entropy(output, target):
    t = target.clone().detach()
    output = torch.sigmoid(output)
    t[t >= 0.5] = 1  # threshold to get binary labels
    t[t < 0.5] = 0
    return F.cross_entropy(output, t.long())/torch.sum(t,dim=(0,1))



class ContinuousLoss_SL1(nn.Module):
    ''' Class to measure loss between continuous emotion dimension predictions and labels. Using smooth l1 loss as base. '''

    def __init__(self, margin=1):
        super(ContinuousLoss_SL1, self).__init__()
        self.margin = margin


def sl1_loss(output, target, margin=0.1):
    output = output
    target = target
    labs = torch.abs(output - target)
    loss = 0.5 * (labs ** 2)
    loss[(labs > margin)] = labs[(labs > margin)] - 0.05
    loss = loss.sum(dim=1)
    return loss.mean()

def multilabel_soft_margin_loss(output, target):
    return F.multilabel_soft_margin_loss(output, target)

def mse_loss(output, target):
    print(output.size(), target.size())
    return F.mse_loss(output, target)


def l1_loss(output, target):
	return F.l1_loss(output, target)
import torch
from torch import Tensor


def cosine_loss(output, target, labels=None):
    output = F.normalize(output, dim=1)
    target = F.normalize(target, dim=1)
    return torch.mean(1-torch.nn.CosineSimilarity(dim=1)(output, target))


# def hinge_rank_loss(output, target, labels, m=1):
#
#     t = labels.clone().detach()
#     t[t >= 0.5] = 1  # threshold to get binary labels
#     t[t < 0.5] = 0
#     target = target[:,:26]
#
#     lhinge = 0
#     # lsumexp = 0
#     for i in range(output.size(0)):
#         for j in range(target.size(1)): #positives
#             if t[i,j] == 0:
#                 continue
#             for k in range(target.size(1)): # negatives
#                 if t[i,k] == 1:
#                     continue
#
#                 lhinge += max(0, m + torch.norm(torch.dot(F.normalize(output[i],dim=0),F.normalize(target[i,j], dim=0))) - torch.norm(torch.dot(F.normalize(output[i],dim=0),F.normalize(target[i,k],dim=0))  ))
#
#                 # lsumexp += torch.exp(torch.norm(torch.dot(F.normalize(output[i],dim=0),F.normalize(target[i,j], dim=0))) - torch.norm(torch.dot(F.normalize(output[i],dim=0),F.normalize(target[i,k],dim=0))  ))
#                 # lsumexp += torch.exp(torch.norm(torch.dot(F.normalize(output[i],dim=0),F.normalize(target[i,j], dim=0))) - torch.norm(torch.dot(F.normalize(output[i],dim=0),F.normalize(target[i,k],dim=0))  ))
#
#     # return torch.log(1+lsumexp)/output.size(0)
#     return lhinge/output.size(0)



def mse_center_loss(output, target, labels, loss_fn=F.mse_loss):
    # output = F.normalize(output, dim=1)
    # target = F.normalize(target, dim=1)
    # print(output.size(), target.size(), labels.size())
    t = labels.clone().detach()
    # t[t >= 0.5] = 1  # threshold to get binary labels
    # t[t < 0.5] = 0
    target = target[0].squeeze()
    # target = target[:26]

    positive_centers = []
    negative_centers = []
    for i in range(output.size(0)):
        # print(target.size(), t.size(), t[i,:].size())
        p = target[t[i, :] == 1]
        if p.size(0) == 0:
            positive_center = torch.zeros(300).cuda()
        else:
            positive_center = torch.mean(p, dim=0)

        positive_centers.append(positive_center)

        negative_center = torch.mean(target[t[i, :] == 0], dim=0)
        negative_centers.append(negative_center)

    positive_centers = torch.stack(positive_centers,dim=0)
    negative_centers = torch.stack(negative_centers,dim=0)
    

    if loss_fn == "cosine":
        loss = torch.mean(1-F.cosine_similarity(output,positive_centers,dim=1))
        
        # loss += torch.mean(F.cosine_similarity(output,negative_centers,dim=1))
    else:
        loss = loss_fn(output, positive_centers)
    # loss = max(0, F.mse_loss(output, positive_centers) - F.mse_loss(output, negative_centers))



        # else:
            # positive_center = F.normalize(positive_center, dim=0)
        # negative_center = F.normalize(negative_center, dim=0)
            # loss += [max(0, 1 - F.cosine_similarity(output[i],positive_center,dim=0))]
        # loss += [max(0, F.cosine_similarity(output[i],negative_center,dim=0))]
        # print(loss)
        # print(loss)

    return loss


def cosine_embed(output, target, labels):
    l = torch.nn.CosineEmbeddingLoss()

    t = labels.clone().detach()
    t[t >= 0.5] = 1  # threshold to get binary labels
    t[t < 0.5] = 0

    loss = 0
    for i in range(output.size(0)):
        for j in range(target.size(1)):
            if t[i,j] == 1:
                loss += 1 - F.cosine_similarity(output[i], target[0,j], dim=0)
            else:
                loss += max(0, F.cosine_similarity(output[i], target[0,j], dim=0))

    return loss/output.size(0)

def myhinge(output, target, labels, norm=False):
    t = labels.clone().detach()
    t[t >= 0.5] = 1  # threshold to get binary labels
    t[t < 0.5] = 0

    target = target[:,:26]

    batch_size = target.size()[0]

    positive_indices = t.gt(0).float()
    negative_indices = t.eq(0).float()

    ## summing over all negatives and positives
    # print(positive_indices)
    loss = 0.
    for i in range(output.size()[0]):  # loop over examples
        pos = torch.Tensor([j for j, pos in enumerate(positive_indices[i]) if pos != 0]).long()
        neg = torch.Tensor([j for j, neg in enumerate(negative_indices[i]) if neg != 0]).long()
        if pos.size(0) == 0:
            continue
        for j, pj in enumerate(pos):
            for k, nj in enumerate(neg):
                if norm:
                    loss += max[0, 0.1 - torch.dot(output[i], F.normalize(target[i, pj], dim=0)) + torch.dot(output[i], F.normalize(target[i, nj],dim=0))]
                else:
                    loss += max[0, 0.1-torch.dot(output[i],target[i,pj]) + torch.dot(output[i],target[i,nj])]

    return loss/batch_size


def mylsep(output, target, labels):
    t = labels.clone().detach()
    t[t >= 0.5] = 1  # threshold to get binary labels
    t[t < 0.5] = 0

    # target = target[:,:26]

    batch_size = target.size()[0]

    positive_indices = t.gt(0).float()
    negative_indices = t.eq(0).float()

    ## summing over all negatives and positives
    # print(positive_indices)
    loss = 0.
    for i in range(output.size()[0]):  # loop over examples
        pos = torch.Tensor([j for j, pos in enumerate(positive_indices[i]) if pos != 0]).long()
        neg = torch.Tensor([j for j, neg in enumerate(negative_indices[i]) if neg != 0]).long()
        if pos.size(0) == 0:
            continue
        for j, pj in enumerate(pos):
            for k, nj in enumerate(neg):
                loss += torch.exp(torch.norm(torch.dot(output[i],target[i,pj])) - torch.norm(torch.dot(output[i],target[i,nj])))

    loss = torch.log(1 + loss)

    return loss/batch_size



def lsep_for_categ(output, target):
    t = target.clone().detach()
    t[t >= 0.5] = 1  # threshold to get binary labels
    t[t < 0.5] = 0

    batch_size = target.size()[0]

    positive_indices = t.gt(0).float()
    negative_indices = t.eq(0).float()

    ## summing over all negatives and positives
    # print(positive_indices)
    loss = 0.
    for i in range(output.size()[0]):  # loop over examples
        pos = torch.Tensor([j for j, pos in enumerate(positive_indices[i]) if pos != 0]).long()
        neg = torch.Tensor([j for j, neg in enumerate(negative_indices[i]) if neg != 0]).long()
        if pos.size(0) == 0:
            continue
        for j, pj in enumerate(pos):
            for k, nj in enumerate(neg):
                loss += torch.sum(torch.exp(output[i,pj]-output[i,nj]))

    loss = torch.log(1 + loss)

    return loss/batch_size



def my_cosine_loss(output, target, labels):
    # output = F.normalize(output, dim=1)
    # target = F.normalize(target, dim=1)

    t = labels.clone().detach()
    t[t >= 0.5] = 1  # threshold to get binary labels
    t[t < 0.5] = 0

    loss = 0
    for i in range(output.size(0)):
        for j in range(target.size(1)):
            cosdist = 1 - torch.nn.CosineSimilarity(dim=0)(output[i], target[0,j])
            if labels[i,j] == 0:
                loss += max(0, 1-cosdist)
            else:
                loss += cosdist

    return loss/output.size(0)


def hinge_embedding_loss(output, embeddings, labels):
    l = torch.nn.HingeEmbeddingLoss()

    output = F.normalize(output, dim=1)
    output = output.unsqueeze(1).repeat(1, 26, 1)
    embeddings = F.normalize(embeddings, dim=1)
    # print(output.size(), embeddings.size())
    cosine_distances = 1- torch.nn.CosineSimilarity(dim=2)(output, embeddings)
    return l(cosine_distances, labels)




from torch.autograd import Variable, Function



class DiscreteLoss(nn.Module):
    ''' Class to measure loss between categorical emotion predictions and labels.'''

    def __init__(self, weight_type='dynamic', device=torch.device('cpu')):
        super(DiscreteLoss, self).__init__()
        self.weight_type = weight_type
        self.device = device
        if self.weight_type == 'mean':
            self.weights = torch.ones((1, 26)) / 26.0
            self.weights = self.weights.to(self.device)
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                                              0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                                              0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520,
                                              0.1537]).unsqueeze(0)
            self.weights = self.weights.to(self.device)

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to(self.device)
        loss = (((pred - target) ** 2) * self.weights)
        return loss.sum()


def prepare_dynamic_weights(target):
    target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0)
    weights = torch.zeros((1, 26)).cuda()
    weights[target_stats != 0] = 1.0 / torch.log(target_stats[target_stats != 0].data + 1.2)
    weights[target_stats == 0] = 0.0001
    return weights

def discrete_loss(output, target):
    output = output
    t = target.clone().detach()

    t[t >= 0.5] = 1  # threshold to get binary labels
    t[t < 0.5] = 0

    weights = prepare_dynamic_weights(t)
    loss = (((output - t) ** 2) * weights)
    return loss.sum(dim=1).mean()
