import torch
import torch.nn as nn


class Sinkhorn(nn.Module):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=10, epsilon=1e-4, skaddcr=False, skaddcr_value=0.001):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.skaddcr = skaddcr
        self.skaddcr_value = skaddcr_value


    def forward(self, s, nrows=None, ncols=None, exp=False, exp_alpha=20, dummy_row=False, dtype=torch.float32):
        batch_size = s.shape[0]

        if self.skaddcr:
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(s[:, None, :, :])

            log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

            for i in range(self.max_iter):
                # Row normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                    dim=1)

                # Column normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                    dim=2)

            log_alpha_padded = torch.exp(log_alpha_padded)
            s = log_alpha_padded[:, :-1, :-1]

        else:
            if dummy_row:
                dummy_shape = list(s.shape)
                dummy_shape[1] = s.shape[2] - s.shape[1]
                s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
                new_nrows = ncols
                for b in range(batch_size):
                    s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
                nrows = new_nrows

            s += self.epsilon

            for i in range(self.max_iter):
                if exp:
                    s = torch.exp(exp_alpha * s)
                if i % 2 == 1:
                    # column norm
                    # sum1 = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
                    sum = torch.sum(s, dim=2, keepdim=True).expand(s.shape)
                else:
                    # row norm
                    # 原本的sum0和sum1极度消耗内存，中间会产生的最大变量所占的空间为4*1024*1024*1024*4bytes=16GB
                    # sum0 = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)
                    sum = torch.sum(s, dim=1, keepdim=True).expand(s.shape)

                tmp = torch.zeros_like(s)
                for b in range(batch_size):
                    row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                    col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                    tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
                s = s * tmp

            if dummy_row and dummy_shape[1] > 0:
                s = s[:, :-dummy_shape[1]]

        return s