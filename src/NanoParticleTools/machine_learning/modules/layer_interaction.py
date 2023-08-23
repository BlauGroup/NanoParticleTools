import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class InteractionConv(MessagePassing):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 nsigma: int = 5,
                 **kwargs):
        super().__init__(node_dim=0, aggr='add', **kwargs)

        self.interaction_block = InteractionBlock(nsigma)
        # self.message_mlp = nn.Linear(2*input_dim+sigma.size(0), output_dim)
        message_in_dim = 2 * input_dim + self.interaction_block.output_dim
        message_mlp_layers = [
            nn.Linear(message_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, output_dim)
        ]
        self.message_mlp = nn.Sequential(*message_mlp_layers)

        # We use this NN to reduce the dimension of sigma to compute
        # the attention weights
        alpha_mlp_layers = [
            nn.Linear(self.interaction_block.output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1)
        ]
        self.alpha_mlp = nn.Sequential(*alpha_mlp_layers)

    def forward(self,
                x: Tensor | PairTensor,
                compositions: Tensor,
                edge_index: Adj,
                edge_attr: OptTensor = None):
        interaction_strength = self.interaction_block(edge_attr[..., 0],
                                                      edge_attr[..., 1],
                                                      edge_attr[..., 2],
                                                      edge_attr[..., 3])

        # Compute the integrated total of interaction.
        comps = compositions[edge_index]
        comps = comps[..., None, None].expand(*comps.shape, comps.size(-1),
                                              interaction_strength.size(-1))
        comps_j = comps[0]  # Source node compositions
        comps_i = comps[1]  # Target node compositions

        # indexed by: [edge_index, dopant_in_i, dopant_in_j, sigma]
        Iij_xi_xj = comps_i * comps_j * interaction_strength[:, None,
                                                             None, :].expand(
                                                                 comps_i.shape)
        Iij_xi_xj = Iij_xi_xj.abs().add(1).log10().float()

        out = self.propagate(edge_index,
                             x=x,
                             edge_attr=edge_attr,
                             Iij_xi_xj=Iij_xi_xj)
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor, Iij_xi_xj) -> Tensor:

        # concatenate embeddings of j to embeddings of i
        a = x_i.expand(3, -1, -1, -1).moveaxis(0, 1)
        b = x_j.expand(3, -1, -1, -1).moveaxis(0, 1).transpose(1, 2)

        embeddings_ij = torch.concat((a, b), dim=-1)

        # Concatenate embeddings to the interaction strength
        pre_mlp_message = torch.concat((embeddings_ij, Iij_xi_xj), dim=-1)

        # Compute the messages, which will depend on the state of dopant
        # in each layer and on the interaction strength
        mlp_messages = self.message_mlp(pre_mlp_message)

        message_weights = F.softmax(self.alpha_mlp(Iij_xi_xj).squeeze(-1),
                                    dim=1)
        # message_weights = F.normalize(Iij_xi_xj.reshape(-1, 9), p=1).reshape(Iij_xi_xj.shape)

        # Multiply by the weights to compute the message
        weighted_messages = torch.einsum('ijkl, ijk -> ijkl', mlp_messages,
                                         message_weights)

        # Sum over the contributions for dopant j by dopant i
        final_message = weighted_messages.sum(2)
        # print(weighted_messages, weighted_messages.shape)
        return final_message


class InteractionBlock(torch.nn.Module):

    def __init__(self,
                 sigma: torch.Tensor | None = None,
                 nsigma: int = 5,
                 tunable_sigma: bool = True,
                 tanh_approx: bool = False):
        super().__init__()

        if sigma is None:
            if nsigma is None:
                # defaulting to 5 sigmas
                nsigma = 5

            # Initialize the sigma values to a default
            sigma = torch.linspace(8, 100, nsigma)
        else:
            assert sigma.size(
                0) == nsigma, "Mismatch between nsigma and the size of sigma"
        self.sigma = nn.Parameter(sigma, requires_grad=tunable_sigma)
        self.output_dim = nsigma

        if tanh_approx:
            self.interaction_fn = integrated_gaussian_interaction_tanh
        else:
            self.interaction_fn = integrated_gaussian_interaction

    def forward(self, ri0, rif, rj0, rjf):
        ri0 = ri0.unsqueeze(-1)
        rif = rif.unsqueeze(-1)
        rj0 = rj0.unsqueeze(-1)
        rjf = rjf.unsqueeze(-1)

        out = self.interaction_fn(ri0, rif, rj0, rjf, self.sigma)
        return out


# @torch.jit.script
# Not sure why, but the jit script is slower than the python function here
def integrated_gaussian_interaction(ri0: torch.Tensor, rif: torch.Tensor,
                                    rj0: torch.Tensor, rjf: torch.Tensor,
                                    sigma: torch.Tensor) -> torch.Tensor:
    """
    This function computes the integrated interaction between two spherical
    layers as defined by their inner and outer radii.

    Args:
        ri0 (torch.Tensor): Inner radius of the first shell
        rif (torch.Tensor): Outer raidus of the first shell
        rj0 (torch.Tensor): Inner radius of the second shell
        rjf (torch.Tensor): Outer radius of the second shell
        sigma (torch.Tensor): Values for standard deviation in the gaussian

    Returns:
        torch.Tensor: The integrated interaction between the two shells
    """

    input_dtype = ri0.dtype
    # Make sure all tensors are in double precision.
    # The increased precision is required for small and large numbers
    ri0 = ri0.double()
    rif = rif.double()
    rj0 = rj0.double()
    rjf = rjf.double()
    sigma = sigma.double()

    # Compute some constants that are used many times
    sqrt2 = torch.sqrt(torch.tensor(2))
    sqrt2pi = torch.sqrt(torch.tensor(2) * torch.pi)
    sqrt2_pi = torch.sqrt(torch.tensor(2) / torch.pi)
    pi3_2 = torch.pow(torch.pi, torch.tensor(3 / 2))
    pi2 = torch.pow(torch.pi, torch.tensor(2))
    sigma2_2 = 2 * torch.pow(sigma, 2)
    sqrt2_pi3_2_sigma6 = sqrt2 * pi3_2 * torch.pow(sigma, 6)
    sqrt2_sigma = sqrt2 * sigma

    exp1 = torch.exp(-torch.pow(ri0 - rj0, 2) / sigma2_2)
    exp2 = torch.exp(-torch.pow(rif - rj0, 2) / sigma2_2)
    erf1 = torch.erf((ri0 - rj0) / (sqrt2_sigma))
    erf2 = torch.erf((rif - rj0) / (sqrt2_sigma))
    term1 = sqrt2_pi3_2_sigma6 * (2 * sigma * (exp1 - exp2) - sqrt2pi * rj0 *
                                  (erf1 - erf2))
    term2 = 2 / 3 * pi2 * torch.pow(sigma, 4) * \
        (-exp1 * sqrt2_pi * sigma * (ri0**2 + ri0 * rj0 + rj0**2 + 2 * sigma**2) +
         exp2 * sqrt2_pi * sigma * (rif**2 + rif * rj0 + rj0**2 + 2 * sigma**2) -
         (ri0**3 - rj0 * (rj0**2 + 3 * sigma**2)) * erf1 +
         (rif**3 - rj0 * (rj0**2 + 3 * sigma**2)) * erf2)

    exp1 = torch.exp(-torch.pow(ri0 + rj0, 2) / sigma2_2)
    exp2 = torch.exp(-torch.pow(rif + rj0, 2) / sigma2_2)
    erf1 = torch.erf((ri0 + rj0) / (sqrt2_sigma))
    erf2 = torch.erf((rif + rj0) / (sqrt2_sigma))
    term3 = sqrt2_pi3_2_sigma6 * (2 * sigma * (exp1 - exp2) + sqrt2pi * rj0 *
                                  (erf1 - erf2))
    term4 = 2 / 3 * pi2 * torch.pow(sigma, 4) * \
        (-exp1 * sqrt2_pi * sigma * (ri0**2 - ri0 * rj0 + rj0**2 + 2 * sigma**2) +
         exp2 * sqrt2_pi * sigma * (rif**2 - rif * rj0 + rj0**2 + 2 * sigma**2) -
         (ri0**3 + rj0**3 + 3 * rj0 * sigma**2) * erf1 +
         (rif**3 + rj0**3 + 3 * rj0 * sigma**2) * erf2)

    exp1 = torch.exp(-torch.pow(ri0 - rjf, 2) / sigma2_2)
    exp2 = torch.exp(-torch.pow(rif - rjf, 2) / sigma2_2)
    erf1 = torch.erf((ri0 - rjf) / (sqrt2_sigma))
    erf2 = torch.erf((rif - rjf) / (sqrt2_sigma))
    term5 = sqrt2_pi3_2_sigma6 * (2 * sigma * (exp1 - exp2) - sqrt2pi * rjf *
                                  (erf1 - erf2))
    term6 = 2 / 3 * pi2 * torch.pow(sigma, 4) * \
        (-exp1 * sqrt2_pi * sigma * (ri0**2 + ri0 * rjf + rjf**2 + 2 * sigma**2) +
         exp2 * sqrt2_pi * sigma * (rif**2 + rif * rjf + rjf**2 + 2 * sigma**2) -
         (ri0**3 - rjf * (rjf**2 + 3 * sigma**2)) * erf1 +
         (rif**3 - rjf * (rjf**2 + 3 * sigma**2)) * erf2)

    exp1 = torch.exp(-torch.pow(ri0 + rjf, 2) / sigma2_2)
    exp2 = torch.exp(-torch.pow(rif + rjf, 2) / sigma2_2)
    erf1 = torch.erf((ri0 + rjf) / (sqrt2_sigma))
    erf2 = torch.erf((rif + rjf) / (sqrt2_sigma))
    term7 = sqrt2_pi3_2_sigma6 * (2 * sigma * (exp1 - exp2) + sqrt2pi * rjf *
                                  (erf1 - erf2))
    term8 = 2 / 3 * pi2 * torch.pow(sigma, 4) * \
        (-exp1 * sqrt2_pi * sigma * (ri0**2 - ri0 * rjf + rjf**2 + 2 * sigma**2) +
         exp2 * sqrt2_pi * sigma * (rif**2 - rif * rjf + rjf**2 + 2 * sigma**2) -
         (ri0**3 + rjf**3 + 3 * rjf * sigma**2) * erf1 +
         (rif**3 + rjf**3 + 3 * rjf * sigma**2) * erf2)

    out = term1 + term2 - term3 - term4 - term5 - term6 + term7 + term8
    return out.to(input_dtype)


# @torch.jit.script
# Not sure why, but the jit script is slower than the python function here
def integrated_gaussian_interaction_tanh(ri0: torch.Tensor, rif: torch.Tensor,
                                         rj0: torch.Tensor, rjf: torch.Tensor,
                                         sigma: torch.Tensor) -> torch.Tensor:
    """
    This function computes the integrated interaction between two spherical layers as
    defined by their inner and outer radii.

    Individually, the tanh function is ~ 10x faster than the erf function, but in the full
    interaction calculation, it only yields a 10% improvement in speed since there are
    so many other operations.

    Args:
        ri0 (torch.Tensor): Inner radius of the first shell
        rif (torch.Tensor): Outer raidus of the first shell
        rj0 (torch.Tensor): Inner radius of the second shell
        rjf (torch.Tensor): Outer radius of the second shell
        sigma (torch.Tensor): Values for standard deviation in the gaussian

    Returns:
        torch.Tensor: The integrated interaction between the two shells
    """

    input_dtype = ri0.dtype
    # Make sure all tensors are in double precision.
    # The increased precision is required for small and large numbers
    ri0 = ri0.double()
    rif = rif.double()
    rj0 = rj0.double()
    rjf = rjf.double()
    sigma = sigma.double()

    # Compute some constants that are used many times
    sqrt2 = torch.sqrt(torch.tensor(2))
    sqrt2pi = torch.sqrt(torch.tensor(2) * torch.pi)
    sqrt2_pi = torch.sqrt(torch.tensor(2) / torch.pi)
    pi3_2 = torch.pow(torch.pi, torch.tensor(3 / 2))
    pi2 = torch.pow(torch.pi, torch.tensor(2))
    sigma2_2 = 2 * torch.pow(sigma, 2)
    sqrt2_pi3_2_sigma6 = sqrt2 * pi3_2 * torch.pow(sigma, 6)
    sqrt2_sigma = sqrt2 * sigma

    exp1 = torch.exp(-torch.pow(ri0 - rj0, 2) / sigma2_2)
    exp2 = torch.exp(-torch.pow(rif - rj0, 2) / sigma2_2)
    tanh1 = torch.tanh((ri0 - rj0) / (sqrt2_sigma))
    tanh2 = torch.tanh((rif - rj0) / (sqrt2_sigma))
    term1 = sqrt2_pi3_2_sigma6 * (2 * sigma * (exp1 - exp2) - sqrt2pi * rj0 *
                                  (tanh1 - tanh2))
    term2 = 2 / 3 * pi2 * torch.pow(sigma, 4) * \
        (-exp1 * sqrt2_pi * sigma * (ri0**2 + ri0 * rj0 + rj0**2 + 2 * sigma**2) +
         exp2 * sqrt2_pi * sigma * (rif**2 + rif * rj0 + rj0**2 + 2 * sigma**2) -
         (ri0**3 - rj0 * (rj0**2 + 3 * sigma**2)) * tanh1 +
         (rif**3 - rj0 * (rj0**2 + 3 * sigma**2)) * tanh2)

    exp1 = torch.exp(-torch.pow(ri0 + rj0, 2) / sigma2_2)
    exp2 = torch.exp(-torch.pow(rif + rj0, 2) / sigma2_2)
    tanh1 = torch.tanh((ri0 + rj0) / (sqrt2_sigma))
    tanh2 = torch.tanh((rif + rj0) / (sqrt2_sigma))
    term3 = sqrt2_pi3_2_sigma6 * (2 * sigma * (exp1 - exp2) + sqrt2pi * rj0 *
                                  (tanh1 - tanh2))
    term4 = 2 / 3 * pi2 * torch.pow(sigma, 4) * \
        (-exp1 * sqrt2_pi * sigma * (ri0**2 - ri0 * rj0 + rj0**2 + 2 * sigma**2) +
         exp2 * sqrt2_pi * sigma * (rif**2 - rif * rj0 + rj0**2 + 2 * sigma**2) -
         (ri0**3 + rj0**3 + 3 * rj0 * sigma**2) * tanh1 +
         (rif**3 + rj0**3 + 3 * rj0 * sigma**2) * tanh2)

    exp1 = torch.exp(-torch.pow(ri0 - rjf, 2) / sigma2_2)
    exp2 = torch.exp(-torch.pow(rif - rjf, 2) / sigma2_2)
    tanh1 = torch.tanh((ri0 - rjf) / (sqrt2_sigma))
    tanh2 = torch.tanh((rif - rjf) / (sqrt2_sigma))
    term5 = sqrt2_pi3_2_sigma6 * (2 * sigma * (exp1 - exp2) - sqrt2pi * rjf *
                                  (tanh1 - tanh2))
    term6 = 2 / 3 * pi2 * torch.pow(sigma, 4) * \
        (-exp1 * sqrt2_pi * sigma * (ri0**2 + ri0 * rjf + rjf**2 + 2 * sigma**2) +
         exp2 * sqrt2_pi * sigma * (rif**2 + rif * rjf + rjf**2 + 2 * sigma**2) -
         (ri0**3 - rjf * (rjf**2 + 3 * sigma**2)) * tanh1 +
         (rif**3 - rjf * (rjf**2 + 3 * sigma**2)) * tanh2)

    exp1 = torch.exp(-torch.pow(ri0 + rjf, 2) / sigma2_2)
    exp2 = torch.exp(-torch.pow(rif + rjf, 2) / sigma2_2)
    tanh1 = torch.tanh((ri0 + rjf) / (sqrt2_sigma))
    tanh2 = torch.tanh((rif + rjf) / (sqrt2_sigma))
    term7 = sqrt2_pi3_2_sigma6 * (2 * sigma * (exp1 - exp2) + sqrt2pi * rjf *
                                  (tanh1 - tanh2))
    term8 = 2 / 3 * pi2 * torch.pow(sigma, 4) * \
        (-exp1 * sqrt2_pi * sigma * (ri0**2 - ri0 * rjf + rjf**2 + 2 * sigma**2) +
         exp2 * sqrt2_pi * sigma * (rif**2 - rif * rjf + rjf**2 + 2 * sigma**2) -
         (ri0**3 + rjf**3 + 3 * rjf * sigma**2) * tanh1 +
         (rif**3 + rjf**3 + 3 * rjf * sigma**2) * tanh2)

    out = term1 + term2 - term3 - term4 - term5 - term6 + term7 + term8
    return out.to(input_dtype)
