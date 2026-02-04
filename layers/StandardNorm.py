import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-8, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def _create_padding_mask(self, x):
        """
        Create padding mask by detecting zero vectors along the feature dimension
        Assumes padding is at the beginning (left side) of the sequence
        :param x: input tensor of shape [B, T, C]
        :return: mask tensor of shape [B, T] where 1 indicates valid data, 0 indicates padding
        """
        # Check if all features are zero for each timestep (indicating padding)
        # Sum over the channel dimension and check if the result is zero
        non_zero_mask = (x.abs().sum(dim=-1) > self.eps)  # [B, T]
        return non_zero_mask.float()

    def forward(self, x, mode: str, mask=None, auto_mask=False):
        """
        :param x: input tensor of shape [B, T, C]
        :param mode: 'norm' or 'denorm'
        :param mask: optional mask tensor of shape [B, T] where 1 indicates valid data, 0 indicates padding
        :param auto_mask: if True, automatically create mask by detecting zero padding
        """
        if auto_mask and mask is None:
            mask = self._create_padding_mask(x)
        
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == 'denorm':
            x = self._denormalize(x, mask)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        """
        Calculate statistics only for non-padding parts when mask is provided
        :param x: input tensor of shape [B, T, C]
        :param mask: optional mask tensor of shape [B, T] where 1 indicates valid data, 0 indicates padding
        """
        if mask is not None:
            # Create mask for valid data: [B, T, 1]
            mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
            
            if self.subtract_last:
                self.last = x[:, -1, :].unsqueeze(1)
            else:
                # Calculate mean only for valid (non-padding) parts
                masked_x = x * mask_expanded  # Zero out padding parts
                valid_counts = mask_expanded.sum(dim=1, keepdim=True)  # [B, 1, 1]
                self.mean = (masked_x.sum(dim=1, keepdim=True) / (valid_counts + self.eps)).detach()
            
            # Calculate standard deviation for valid parts
            if self.subtract_last:
                diff = (x - self.last) * mask_expanded
            else:
                diff = (x - self.mean) * mask_expanded
            
            valid_counts = mask_expanded.sum(dim=1, keepdim=True)  # [B, 1, 1]
            variance = (diff ** 2).sum(dim=1, keepdim=True) / (valid_counts + self.eps)
            self.stdev = torch.sqrt(variance + self.eps).detach()
        else:
            # Original logic when no mask is provided
            dim2reduce = tuple(range(1, x.ndim - 1))
            if self.subtract_last:
                self.last = x[:, -1, :].unsqueeze(1)
                # diff = x - self.last
            else:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
                # diff = x - self.mean
            # self.stdev = torch.sqrt(torch.var(diff, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x, mask=None):
        if self.non_norm:
            return x
        
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        
        # Ensure padding parts remain zero when mask is provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
            x = x * mask_expanded
        
        return x

    def _denormalize(self, x, mask=None):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x_channels = x.shape[-1]
        if x_channels < self.mean.shape[-1] and x_channels == 1:
            self.mean = self.mean[:, :, -2:-1]
            self.stdev = self.stdev[:, :, -2:-1]
            if self.subtract_last:
                self.last = self.last[:, :, -2:-1]
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        
        # Ensure padding parts remain zero when mask is provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
            x = x * mask_expanded
        
        return x
