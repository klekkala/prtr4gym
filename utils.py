
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
from IPython import embed
from pytorch_metric_learning import losses
from arguments import get_args

args = get_args()

def get_data_STL10(trainset, testset, transform, batch_size):
    
    if trainset != None:
        print("Loading trainset...")
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    if testset != None:
        print("Loading testset...")
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    print("Done!")
    if trainset == None:
        return None, testloader
    if testset == None:
        return trainloader, None


# Linear scaling the learning rate down
def lr_Linear(optimizer, epoch_max, epoch, lr):
    lr_adj = ((epoch_max - epoch) / epoch_max) * lr
    set_lr(optimizer, lr=lr_adj)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vae_loss(recon, x, mu, logvar, kl_weight):
    #recon_loss = F.binary_cross_entropy_with_logits(recon, x)
    #the below was what was used for the prev experiment that was working
    recon_loss = F.binary_cross_entropy_with_logits(recon, x, reduction='mean')
    #recon_loss = F.mse_loss(torch.sigmoid(recon), x, reduction='mean')
    KL_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl_weight*KL_loss
    return loss



def cont_loss(x1, x2, label, margin: float = 1.0):
    """
    Computes Contrastive Loss
    """
    #if d == 0:
    #  return T.mean(T.pow(euc_dist, 2))  # distance squared
    #else:  # d == 1
    #  delta = self.m - euc_dist  # sort of reverse distance
    #  delta = T.clamp(delta, min=0.0, max=None)
    #  return T.mean(T.pow(delta, 2))  # mean over all rows


    dist = torch.nn.functional.pairwise_distance(x1, x2)

    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)
    # >>> input1 = torch.randn(100, 128)
    # >>> input2 = torch.randn(100, 128)
    # >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # >>> output = cos(input1, input2)
    return loss


import torch
import inspect
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.module_with_records_and_reducer import ModuleWithRecordsReducerAndDistance
from pytorch_metric_learning.losses.mixins import EmbeddingRegularizerMixin
from pytorch_metric_learning.reducers import AvgNonZeroReducer


def get_vmatches_and_vdiffs(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels

    bsize = labels.shape[0]
    labels1 = labels.unsqueeze(0).repeat(bsize, 1)
    labels2 = ref_labels.unsqueeze(1).repeat(1, bsize)
    matches = (torch.isclose(labels1, labels2, atol=args.kl_weight, rtol=0.0)).byte()
    #embed()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs


def get_all_pairs_indices(labels, ref_labels=None):

    vlabels, elabels = torch.split(labels, 1, dim=1)
    #check vmatches. why are you getting false even when kl_weight is high?
    vmatches, vdiffs = get_vmatches_and_vdiffs(torch.squeeze(vlabels), ref_labels)
    ematches, ediffs = lmu.get_matches_and_diffs(torch.squeeze(elabels), ref_labels)

    matches = torch.logical_and(vmatches, ematches)
    diffs = torch.logical_or(vdiffs, ediffs)
    #embed()
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx



def convert_value_to_pairs(indices_tuple, labels, ref_labels):
    
    #get matches and diffs based on threshold values
    if indices_tuple is None:
        return get_all_pairs_indices(labels)
        


class BaseMetricLossFunction(
    EmbeddingRegularizerMixin, ModuleWithRecordsReducerAndDistance
):
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        """
        This has to be implemented and is what actually computes the loss.
        """
        raise NotImplementedError

    def forward(
        self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        ###WARNING!!!! commenting this might throw unexpected errors
        c_f.check_shapes(embeddings, labels)

        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
        ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)

    def zero_loss(self):
        return {"losses": 0, "indices": None, "reduction_type": "already_reduced"}

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}

    def _sub_loss_names(self):
        return ["loss"]

    def sub_loss_names(self):
        return self._sub_loss_names() + self.all_regularization_loss_names()

    def all_regularization_loss_names(self):
        reg_names = []
        for base_class in inspect.getmro(self.__class__):
            base_class_name = base_class.__name__
            mixin_keyword = "RegularizerMixin"
            if base_class_name.endswith(mixin_keyword):
                descriptor = base_class_name.replace(mixin_keyword, "").lower()
                if getattr(self, "{}_regularizer".format(descriptor)):
                    reg_names.extend(base_class.regularization_loss_names(self))
        return reg_names



class GenericPairLoss(BaseMetricLossFunction):
    def __init__(self, mat_based_loss, **kwargs):
        super().__init__(**kwargs)
        self.loss_method = (
            self.mat_based_loss if mat_based_loss else self.pair_based_loss
        )

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)

        ##convert_value_to_pairs
        ##generate anchor, positive, negative tuples from the value estimates.
        indices_tuple = convert_value_to_pairs(indices_tuple, labels, ref_labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        return self.loss_method(mat, indices_tuple)

    def _compute_loss(self):
        raise NotImplementedError

    def mat_based_loss(self, mat, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return self._compute_loss(mat, pos_mask, neg_mask)

    def pair_based_loss(self, mat, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, indices_tuple)


class ContrastiveLoss(GenericPairLoss):
    def __init__(self, pos_margin=0, neg_margin=1, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.add_to_recordable_attributes(
            list_of_names=["pos_margin", "neg_margin"], is_stat=False
        )

    def _compute_loss(self, pos_pair_dist, neg_pair_dist, indices_tuple):
        pos_loss, neg_loss = 0, 0
        if len(pos_pair_dist) > 0:
            pos_loss = self.get_per_pair_loss(pos_pair_dist, "pos")
        if len(neg_pair_dist) > 0:
            neg_loss = self.get_per_pair_loss(neg_pair_dist, "neg")
        pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
        neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)
        return {
            "pos_loss": {
                "losses": pos_loss,
                "indices": pos_pairs,
                "reduction_type": "pos_pair",
            },
            "neg_loss": {
                "losses": neg_loss,
                "indices": neg_pairs,
                "reduction_type": "neg_pair",
            },
        }

    def get_per_pair_loss(self, pair_dists, pos_or_neg):
        loss_calc_func = self.pos_calc if pos_or_neg == "pos" else self.neg_calc
        margin = self.pos_margin if pos_or_neg == "pos" else self.neg_margin
        per_pair_loss = loss_calc_func(pair_dists, margin)
        return per_pair_loss

    def pos_calc(self, pos_pair_dist, margin):
        return torch.nn.functional.relu(self.distance.margin(pos_pair_dist, margin))

    def neg_calc(self, neg_pair_dist, margin):
        return torch.nn.functional.relu(self.distance.margin(margin, neg_pair_dist))

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def _sub_loss_names(self):
        return ["pos_loss", "neg_loss"]