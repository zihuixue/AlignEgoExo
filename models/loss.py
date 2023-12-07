import torch
from torch import nn
from models.tcc_loss import compute_tcc_loss
from models.dtw_loss import compute_dtw_loss
from models.vava_loss import compute_vava_loss


class AlignmentLoss(nn.Module):
    def __init__(self, args):
        super(AlignmentLoss, self).__init__()
        self.args = args
        self.cyclic = False if self.args.dataset in ['break_eggs', 'tennis_forehand'] else True
        print(f'Cyclic action {self.cyclic}')

    def forward(self, embeddings, steps=None, seq_lens=None, pos_steps=None, global_step=0):
        if 'tcc' in self.args.loss:
            loss = compute_tcc_loss(embeddings, steps, seq_lens, self.args.tcc_temp)
            return loss, torch.zeros_like(loss)
        elif 'dtw' in self.args.loss:
            return compute_dtw_loss(self.args, embeddings, pos_steps,
                                    alignment_type=self.args.loss,
                                    cyclic_action=self.cyclic)
        elif 'vava' in self.args.loss:
            loss = compute_vava_loss(embeddings, global_step=global_step)
            return loss, torch.zeros_like(loss)
        else:
            raise NotImplementedError

