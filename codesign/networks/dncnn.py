import torch
import torch.nn as nn
from torch.nn.functional import conv2d
from .modules import GaussianNormalize

def normalize(tensor, eps=1e-12):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, eps)
    # if out is None:
    #     ret = tensor / norm
    # else:
        # ret = torch.div(tensor, norm, out=torch.jit._unwrap_optional(out))
    ans = tensor / norm
    return ans

class SpectralNorm(object):

    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        # out, In, kernel0, kernel1 = weight_mat.shape
        '''
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        '''
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                # v = normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                # u = normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
                # v = normalize(conv2d(u, weight_mat.permute(1,0,2,3), padding=1), dim=0, eps=self.eps)
                # u = normalize(conv2d(v, weight_mat, padding=1), dim=0, eps=self.eps)
                v = normalize(conv2d(u.flip(2,3), weight_mat.permute(1, 0, 2, 3), padding=2),
                              eps=self.eps).flip(2,3)[:,:,1:-1,1:-1]
                u = normalize(conv2d(v, weight_mat, padding=1), eps=self.eps)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()


        # sigma = torch.dot(u, conv2d(v, weight_mat, padding=1))
        # sigma = torch.sum(torch.matmul(u, conv2d(v, weight_mat, padding=1)))
        sigma = torch.sum(u * conv2d(v, weight_mat, padding=1))
        weight = weight / sigma
        weight = weight# * pow(0.3, 1.0/17.0)  # omit this (comment out) if lip constant = 1
        return weight, u

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        # delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)

        # u = normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        if module.weight.shape[0] == 2:
            C_out = 2
        elif module.weight.shape[0] == 3:
            C_out = 3
        else:
            C_out = 64
        # u = normalize(weight.new_empty(batch_size, C_out , 40, 40).normal_(0, 1), dim=0, eps=fn.eps)
        u = normalize(weight.new_empty(1, C_out, 40, 40).normal_(0, 1), eps=fn.eps)# input size
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # buffer, which will cause weight to be included in the state dict
        # and also supports nn.init due to shared storage.
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)

        module.register_forward_pre_hook(fn)
        return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    r"""Applies spectral normalization to a parameter in the given module.
    .. math::
         \mathbf{W} &= \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) &= \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
    Spectral normalization stabilizes the training of discriminators (critics)
    in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.
    See `Spectral Normalization for Generative Adversarial Networks`_ .
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectal norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is 0, except for modules that are instances of
            ConvTranspose1/2/3d, when it is 1
    Returns:
        The original module with the spectal norm hook
    Example::
        >>> m = spectral_norm(nn.Linear(20, 40))
        Linear (20 -> 40)
        >>> m.weight_u.size()
        torch.Size([20])
    """
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17, lip=1.0):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(spectral_norm(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        cgn = GaussianNormalize()
        x = cgn.input(x)
        out = self.dncnn(x)
        out = cgn.output(out)
        return out