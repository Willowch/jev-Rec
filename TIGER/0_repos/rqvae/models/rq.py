import torch
import torch.nn as nn

from .vq import VectorQuantizer

class ResidualVectorQuantizer(nn.Module):

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 0.25,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        beta=self.beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []

        quantized_sum = torch.zeros_like(x)
        residual = x

        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)

            quantized_value = x_res.detach()
            residual = residual - quantized_value
            quantized_sum = quantized_sum + quantized_value

            all_losses.append(loss)
            all_indices.append(indices)

        x_q = x + (quantized_sum - x).detach()

        rq_loss = torch.stack(all_losses).mean()
        indices = torch.stack(all_indices, dim=-1)

        return x_q, rq_loss, indices
