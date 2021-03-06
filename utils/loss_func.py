import torch
import torch.nn as nn
import torch.nn.functional as F


class PermLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    cal avg loss, 平均每个节点的loss
    """
    def __init__(self):
        super(PermLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_perm[b, :pred_ns[b], :gt_ns[b]],
                gt_perm[b, :pred_ns[b], :gt_ns[b]],
                reduction='sum')
            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum


class RobustLoss(nn.Module):
    """
    RobustLoss Criterion computes a robust loss function.
    L = Sum(Phi(d_i - d_i^gt)),
        where Phi(x) = sqrt(x^T * x + epsilon)
    Parameter: a small number for numerical stability epsilon
               (optional) division taken to normalize the loss norm
    Input: displacement matrix d1
           displacement matrix d2
           (optional)dummy node mask mask
    Output: loss value
    """
    def __init__(self, epsilon=1e-5, norm=None):
        super(RobustLoss, self).__init__()
        self.epsilon = epsilon
        self.norm = norm

    def forward(self, d1, d2, mask=None):
        # Loss = Sum(Phi(d_i - d_i^gt))
        # Phi(x) = sqrt(x^T * x + epsilon)
        if mask is None:
            mask = torch.ones_like(mask)
        x = d1 - d2
        if self.norm is not None:
            x = x / self.norm

        xtx = torch.sum(x * x * mask, dim=-1)
        phi = torch.sqrt(xtx + self.epsilon)
        loss = torch.sum(phi) / d1.shape[0]

        return loss


class ClassifyLoss(nn.Module):
    """
    Cross entropy loss between two classify.
    cal avg loss, 平均每个节点的loss
    """
    def __init__(self):
        super(ClassifyLoss, self).__init__()

    def forward(self, pred_class, gt_class, gt_ns):
        batch_num = pred_class.shape[0]

        pred_class = pred_class.to(dtype=torch.float32)

        assert torch.all((pred_class >= 0) * (pred_class <= 1))
        assert torch.all((gt_class >= 0) * (gt_class <= 1))

        loss = torch.tensor(0.).to(pred_class.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_class[b, :gt_ns[b], :],
                gt_class[b, :gt_ns[b], :],
                reduction='sum')
            n_sum += gt_ns[b].to(n_sum.dtype).to(pred_class.device)

        return loss / n_sum


if __name__ == '__main__':
    d1 = torch.tensor([[1., 2.],
                       [2., 3.],
                       [3., 4.]], requires_grad=True)
    d2 = torch.tensor([[-1., -2.],
                       [-2., -3.],
                       [-3., -4.]], requires_grad=True)
    mask = torch.tensor([[1., 1.],
                         [1., 1.],
                         [0., 0.]])

    rl = RobustLoss()
    loss = rl(d1, d2, mask)
    loss.backward()
    print(d1.grad)
    print(d2.grad)
