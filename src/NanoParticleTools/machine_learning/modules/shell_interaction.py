import torch
import math
from e3nn import o3, io

class ShellInteractionBlock(torch.nn.Module):
    def __init__(self,
                 sigma,
                 lmax = 0,
                 npoints = 500,
                 interaction_type: str = 'gaussian'):
        """
        This interaction layer will approximate the integral of point-wise interactions between two layers
        
        The function used to describe the interaction can be chosen to be of the form:
        $\frac{\sqrt{2\pi}}{\sigma}*e^{-\frac{1}{2}(\frac{s}{\sigma{}})^2}$
        or
        $\frac{\pi*\gamma}{(1+\frac{s}{\gamma}^2)}$
        """
        
        super().__init__()
        
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma).double()
        
        self.sigma = torch.nn.Parameter(sigma)
        self.lmax = lmax
        self.npoints = npoints
        self.spherical_tensor = io.SphericalTensor(lmax, 1, 1)
        self.interaction_type = interaction_type
        
    def forward(self, sh_coeffs1, sh_coeffs2):
        out = numeric_shell_interaction(sh_coeffs1, sh_coeffs2, self.npoints, self.sigma, self.spherical_tensor, self.interaction_type)
        return out

def gaussian_interaction(s, sigma=10):
    return 1/(sigma*torch.sqrt(torch.pi*torch.tensor(2))) * torch.exp(-1/2*(s/sigma)**2)

def cauchy(s, gamma):
    return 1/((torch.pi * gamma) * ( 1 + ((s) / gamma)**2))

def numeric_shell_interaction(sh_1, sh_2, npoints, sigma, spherical_tensor, interaction='gaussian'):
    # We generate points on the sphere according to the fibonacci scheme.    
    sphere_points = fibonacci_sphere(npoints)
    
    # We assume that all the points are evenly distributed on the surface by this method
    # In the case which the number of points is large enough, this crude approximation is not bad
    dS = 4*torch.pi / npoints * (1)**2

    # Evaluate the SPHARM at these points
    r1 = spherical_tensor.signal_xyz(sh_1, sphere_points)
    r2 = spherical_tensor.signal_xyz(sh_2, sphere_points)
    
    # Create the mesgrid of both surface points
    points1 = torch.einsum('...ij, ...i -> ...ij', sphere_points, r1).unsqueeze(-2).expand(*r1.shape, sphere_points.size(0), 3)
    points2 = torch.einsum('...ij, ...i -> ...ij', sphere_points, r2).unsqueeze(-3).expand(*r2.shape, sphere_points.size(0), 3)

    # Calculate the distance matrix
    distance_matrix = (points2-points1).pow(2).sum(-1).sqrt()
    
    # Compute the value of the interaction for each dV
    if interaction == 'gaussian':
        dI = gaussian_interaction(distance_matrix, sigma)
    elif interaction == 'cauchy':
        dI = cauchy(distance_matrix, sigma)
    else:
        raise ValueError(f'Requested interaction type {interaction} is unknown')
        
    # Perform the numerical integration (analogous to a double riemann sum, but in 2d)
    I = (dI * dS * dS).sum((-1, -2))
    return I


def fibonacci_sphere(samples=1000):
    """
    adapted from: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    points = torch.tensor(points)
    return points[torch.randperm(points.size(0))]