import torch
from torch import nn


class PatchAdjModel(nn.Module):
    def __init__(self, spatial_model):
        super(PatchAdjModel, self).__init__()

        self.spatial_model = spatial_model
        self.alpha_val = torch.Tensor([0.5])
        self.alpha = torch.nn.Parameter(self.alpha_val, requires_grad=True)

    def forward(self, batch):
        spatial_probs = self.spatial_model(batch)

        score_fn = self._calc_edge_adj_dissimilarity_score_a_above_b
        edge_probs = self._calc_edge_probs(batch, score_fn)

        final_probs = edge_probs  # self.alpha * spatial_probs + (1 - self.alpha) * edge_probs

        return final_probs

    def _calc_edge_probs(self, batch, score_fn):
        a = batch[:, :3, :, :]
        b = batch[:, 3:, :, :]

        a_above_b = score_fn(a, b)
        a_below_b = score_fn(b, a)
        a_left_of_b = score_fn(a.transpose(2, 3), b.transpose(2, 3))
        a_right_of_b = score_fn(b.transpose(2, 3), a.transpose(2, 3))

        grad_probs = torch.stack([a_above_b, a_left_of_b, a_below_b, a_right_of_b], dim=-1)
        adj_probs = grad_probs #** 8
        not_adj_prob = (1 - adj_probs).sum(dim=1) / 4

        probs = torch.concat([adj_probs, not_adj_prob.unsqueeze(dim=1)], dim=1)
        probs = torch.nn.functional.softmax(probs * 9, dim=1)

        return probs


    @classmethod
    def _calc_edge_adj_grad_score_a_above_b(cls, a: torch.Tensor, b: torch.Tensor):
        """
        Gets two tensors of shape[batch x channels x 2 x K], predicts whether they are adgacent
        Based on the paper "A fully automated greedy square jigsaw puzzle solver" by Pomeranz et. al.
        :param a:
        :param b:
        :param dim: adjacence dimension
        :return: float: logit
        """
        batch, channels, irrelevant, k = a.shape
        # assert two == 2
        assert a.shape == b.shape

        a_to_b_score = ((2 * a[:, :, -1, :] - a[:, :, -2, :]) - b[:, :, 0, :]).pow(2).pow(1/2)
        b_to_a_score = ((2 * b[:, :, 0, :] - b[:, :, 1, :]) - a[:, :, -1, :]).pow(2).pow(1/2)
        score = (a_to_b_score + b_to_a_score).sum(dim=[1, 2]) / (channels * k)
        score = 1 - torch.nn.functional.sigmoid((score - score.median()) * 12)
        return score

    def _calc_edge_adj_dissimilarity_score_a_above_b(cls, a: torch.Tensor, b: torch.Tensor):
        """
        Gets two tensors of shape[batch x channels x 2 x K], predicts whether they are adgacent
        Based on the paper "A fully automated greedy square jigsaw puzzle solver" by Pomeranz et. al.
        :param a:
        :param b:
        :param dim: adjacence dimension
        :return: float: logit
        """
        batch, channels, irrelevant, k = a.shape
        # assert two == 2
        assert a.shape == b.shape

        score = (a[:, :, -1, :] - b[:, :, 0, :]).pow(2).sum(dim=[1, 2]).sqrt()
        return 1 - score / 16



