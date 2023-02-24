from torch.optim import Optimizer
from torch_geometric.utils import scatter
import torch


class NanoParticleOptimizer():
    lr: float

    def __init__(self, lr, x_index):
        self.lr = lr
        self.x_index = x_index

    def optimize_dopant(self,
                        x_tensor: torch.Tensor,
                        algo: str = 'sgd') -> torch.Tensor:
        """
        We cannot allow the total dopant concentration in a layer to exceed 1, nor can we
        allow the concentration to be negative.

        Args:
            x_tensor (torch.tensor): _description_
            algo (str): _description_. Defaults to 'sgd'.

        Returns:
            _type_: _description_
        """
        #
        if algo == 'sgd':
            # Update according to the gradient, but ensuring that no layer exceeds 1
            x_tensor = constrained_composition_update(x_tensor, self.lr,
                                                      self.x_index)
        elif algo == 'adam':
            # We want to use adam, but clamp at x = 0 and x = 1. When at either of these points,
            # we will set the ema of the gradient to zero 
            raise NotImplementedError('Adam not implemented')
        return x_tensor

    def optimize_radii(self,
                       radii_tensor,
                       algo: str = 'sgd'):
        # We need to make sure here that our radii is always in ascending order.
        # If we have a violation, we will need to clamp the radii or remove it (if it is 0 width).
        # For the implementation with momentum, we also set the ema of the gradient at that
        # point to zero
        if algo == 'sgd':
            # Seems like gradients with respect to the radii are much smaller, so we
            # set the learning rate to be 10 times the learning rate of the dopant
            radii_tensor = constrainted_radius_update(radii_tensor, 10*self.lr)
        elif algo == 'adam':
            raise NotImplementedError('Adam not implemented')

        return radii_tensor


def constrainted_radius_update(radii: torch.Tensor,
                               lr: float):
    #TODO: make sure that the radii cannot have negative thickness 
    change = radii.grad

    # Calculate what the new radii would be
    new_radii = radii.add(lr * change).detach()

    # Clamp the smallest radii to be > 0
    new_radii[0, 0] = max(0, new_radii[0, 0])

    # The case in which both boundaries are decreasing, but the outer decreases more, causing an overlap
    new_radii[1:, 0] = torch.where(torch.logical_and(change[1:, 0] < 0, change[:-1, 1] < 0), new_radii[:-1, 1], new_radii[1:, 0])
    # The case in which both boundaries increase, but the inner increases more, causing an overlap
    new_radii[:-1, 1] = torch.where(torch.logical_and(change[1:, 0] > 0, change[:-1, 1] > 0), new_radii[1:, 0], new_radii[:-1, 1])

    # Now this is the condition in which the outer boundary decreases, but the inner boundary increases
    loc_overlap = torch.logical_and(change[1:, 0] < 0, change[:-1, 1] > 0)

    # This is the subcase in which the radii are already equal. In this case, we just keep
    # them as the original radii
    new_radii[1:, 0] = torch.where(torch.logical_and(loc_overlap, radii[1:, 0] == radii[:-1, 1]), radii[1:, 0], new_radii[1:, 0])
    new_radii[:-1, 1] = torch.where(torch.logical_and(loc_overlap, radii[1:, 0] == radii[:-1, 1]), radii[:-1, 1], new_radii[:-1, 1])
    # This is the subcase in which the radii are not equal. In this case, we need to scale the 
    # new_radii[1:, 0] = torch.where(, new_radii[:-1, 1], new_radii[1:, 0])
    _idx = torch.logical_and(loc_overlap, new_radii[1:, 0] != new_radii[:-1, 1])
    factor = (radii[1:, 0]-radii[:-1, 1]) / (change[:-1, 1]-change[1:, 0])
    new_radii[1:, 0] = torch.where(_idx, radii[1:, 0] + factor * change[1:, 0], new_radii[1:, 0])
    new_radii[:-1, 1] = torch.where(_idx, radii[:-1, 1] + factor * change[:-1, 1], new_radii[:-1, 1])

    return new_radii.detach().requires_grad_(True)


def constrained_composition_update(
        x_dopant: torch.Tensor,
        lr: float,
        x_layer_index: torch.Tensor) -> torch.Tensor:
    """
    This function performs a constrained update of the dopant composition.
    It will ensure that:
        1. There are no negative concentrations
        2. The total concentration of each layer is <= 1

    Args:
        x_dopant (torch.Tensor): The composition tensor to be optimized.
            Should have gradients attached.
        lr (float): The learning rate.
        x_layer_index (torch.Tensor): The mapping of each dopant to a layer.

    Returns:
        torch.Tensor: The updated composition tensor.
    """
    change = lr * x_dopant.grad
    # print(change)
    # Determine the original total layer concentration
    x_scatter_out = scatter(x_dopant.detach(),
                            x_layer_index,
                            dim=0,
                            reduce='sum')

    # Cap negative changes such that the final value is 0
    allowed_negative_change = 0 - x_dopant.detach()
    _change = torch.where(change < allowed_negative_change,
                          allowed_negative_change, change)

    # Use the negative changes to determine the maximum allowable increase
    # in composition for each layer
    negative_change = torch.where(_change < 0, _change,
                                  torch.zeros_like(_change))
    allowed_positive_change = 1 - x_scatter_out - scatter(
        negative_change, x_layer_index, dim=0, reduce='sum')

    # Determine the requested positive changes, then
    positive_change = torch.where(_change > 0, _change,
                                  torch.zeros_like(_change))
    requested_positive_change = scatter(positive_change,
                                        x_layer_index,
                                        dim=0,
                                        reduce='sum')
    # divide by the maximum allowable positive change previously determined
    # to get the scaling factor for each layer
    positive_change_factor = torch.min(
        torch.ones_like(allowed_positive_change),
        allowed_positive_change / requested_positive_change)

    # only apply the positive change factor if there is a positive change
    change_factor = torch.where(_change > 0,
                                positive_change_factor[x_layer_index],
                                torch.ones_like(_change))
    # print(change_factor * _change)
    x_dopant = (x_dopant.detach() + change_factor * _change)
    return x_dopant.detach().requires_grad_(True)
