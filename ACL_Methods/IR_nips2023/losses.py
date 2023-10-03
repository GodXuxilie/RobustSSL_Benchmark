import torch
import torch.nn as nn
import torch.nn.functional as F

### Implementation is based on https://github.com/NightShade99/Self-Supervised-Vision/blob/42183d391f51c60383ff0cb044e2d71379aa7461/utils/losses.py#L154 ###
class ACL_IR_Loss(nn.Module):

    def __init__(self, normalize=True, temperature=0.5, lambda1=0.5,lambda2=0.5):
        super(ACL_IR_Loss, self).__init__()
        self.normalize = normalize
        self.temperature = temperature
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, zi, zj, z_orig, zi_adv, zj_adv, weight=0.0):

        bs = zi.shape[0]
        labels = torch.zeros((2*bs,)).long().to(zi.device)
        mask = torch.ones((bs, bs), dtype=bool).fill_diagonal_(0)

        if self.normalize:
            zi_norm = F.normalize(zi, p=2, dim=-1)
            zj_norm = F.normalize(zj, p=2, dim=-1)
            zo_norm = F.normalize(z_orig, p=2, dim=-1)
            zi_adv_norm = F.normalize(zi_adv, p=2, dim=-1)
            zj_adv_norm = F.normalize(zj_adv, p=2, dim=-1)
        else:
            zi_norm = zi
            zj_norm = zj
            zo_norm = z_orig
            zi_adv_norm = zi_adv
            zj_adv_norm = zj_adv

        ### Standard Contrastive Loss ### 
        logits_ii = torch.mm(zi_norm, zi_norm.t()) / self.temperature
        logits_ij = torch.mm(zi_norm, zj_norm.t()) / self.temperature
        logits_ji = torch.mm(zj_norm, zi_norm.t()) / self.temperature
        logits_jj = torch.mm(zj_norm, zj_norm.t()) / self.temperature

        logits_ij_pos = logits_ij[torch.logical_not(mask)]                                          # Shape (N,)
        logits_ji_pos = logits_ji[torch.logical_not(mask)]                                          # Shape (N,)
        logits_ii_neg = logits_ii[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_ij_neg = logits_ij[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_ji_neg = logits_ji[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_jj_neg = logits_jj[mask].reshape(bs, -1)                                             # Shape (N, N-1)

        pos = torch.cat((logits_ij_pos, logits_ji_pos), dim=0).unsqueeze(1)                         # Shape (2N, 1)
        neg_i = torch.cat((logits_ii_neg, logits_ij_neg), dim=1)                                    # Shape (N, 2N-2)
        neg_j = torch.cat((logits_ji_neg, logits_jj_neg), dim=1)                                    # Shape (N, 2N-2)
        neg = torch.cat((neg_i, neg_j), dim=0)                                                      # Shape (2N, 2N-2)

        logits = torch.cat((pos, neg), dim=1)                                                       # Shape (2N, 2N-1)
        nat_contrastive_loss = F.cross_entropy(logits, labels)

        ### Adversarial Contrastive Loss ### 
        logits_ii = torch.mm(zi_adv_norm, zi_adv_norm.t()) / self.temperature
        logits_ij = torch.mm(zi_adv_norm, zj_adv_norm.t()) / self.temperature
        logits_ji = torch.mm(zj_adv_norm, zi_adv_norm.t()) / self.temperature
        logits_jj = torch.mm(zj_adv_norm, zj_adv_norm.t()) / self.temperature

        logits_ij_pos = logits_ij[torch.logical_not(mask)]                                          # Shape (N,)
        logits_ji_pos = logits_ji[torch.logical_not(mask)]                                          # Shape (N,)
        logits_ii_neg = logits_ii[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_ij_neg = logits_ij[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_ji_neg = logits_ji[mask].reshape(bs, -1)                                             # Shape (N, N-1)
        logits_jj_neg = logits_jj[mask].reshape(bs, -1)                                             # Shape (N, N-1)

        pos = torch.cat((logits_ij_pos, logits_ji_pos), dim=0).unsqueeze(1)                         # Shape (2N, 1)
        neg_i = torch.cat((logits_ii_neg, logits_ij_neg), dim=1)                                    # Shape (N, 2N-2)
        neg_j = torch.cat((logits_ji_neg, logits_jj_neg), dim=1)                                    # Shape (N, 2N-2)
        neg = torch.cat((neg_i, neg_j), dim=0)                                                      # Shape (2N, 2N-2)

        logits = torch.cat((pos, neg), dim=1)                                                       # Shape (2N, 2N-1)
        adv_contrastive_loss = F.cross_entropy(logits, labels)


        ### Standard Invariant Regularization (Implementation follows https://github.com/NightShade99/Self-Supervised-Vision/blob/42183d391f51c60383ff0cb044e2d71379aa7461/utils/losses.py#L154) ###
        logits_io = torch.mm(zi_norm, zo_norm.t()) / self.temperature
        logits_jo = torch.mm(zj_norm, zo_norm.t()) / self.temperature
        probs_io_zi = F.softmax(logits_io[torch.logical_not(mask)], -1)
        probs_jo_zj = F.log_softmax(logits_jo[torch.logical_not(mask)], -1)
        SIR_loss = F.kl_div(probs_io_zi, probs_jo_zj, log_target=True, reduction="sum")


        ### Adversarial Invariant Regularization ###
        logits_io = torch.mm(zi_adv_norm, zi_norm.t()) / self.temperature
        logits_jo = torch.mm(zj_adv_norm, zj_norm.t()) / self.temperature
        probs_io_zi_adv_consis = F.softmax(logits_io[torch.logical_not(mask)], -1)
        probs_jo_zj_adv_consis = F.softmax(logits_jo[torch.logical_not(mask)], -1)

        logits_io = torch.mm(zi_adv_norm, zo_norm.t()) / self.temperature
        logits_jo = torch.mm(zj_adv_norm, zo_norm.t()) / self.temperature
        probs_io_zi_adv = F.softmax(logits_io[torch.logical_not(mask)], -1)
        probs_jo_zj_adv = F.softmax(logits_jo[torch.logical_not(mask)], -1)

        probs_io_zi_adv = torch.mul(probs_io_zi_adv, probs_io_zi_adv_consis)
        probs_jo_zj_adv = torch.mul(probs_jo_zj_adv, probs_jo_zj_adv_consis)
        AIR_loss = F.kl_div(probs_io_zi_adv, torch.log(probs_jo_zj_adv), log_target=True, reduction="sum")

        return nat_contrastive_loss * (1 - weight) + adv_contrastive_loss * (1 + weight) + self.lambda1 * SIR_loss + self.lambda2 * AIR_loss, \
                nat_contrastive_loss.detach().cpu(), adv_contrastive_loss.detach().cpu(),SIR_loss.detach().cpu(), AIR_loss.detach().cpu()
    