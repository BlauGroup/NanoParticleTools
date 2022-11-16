import torch

class InteractionBlock(torch.nn.Module):
    def __init__(self, sigma=1):
        super().__init__()
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma)
        self.sigma = sigma
        
    def forward(self, ri0, rif, rj0, rjf, xi, xj, sigma = None):
        if sigma is None:
            sigma = self.sigma
            
        out = integrated_gaussian_interaction(ri0, rif, rj0, rjf, xi, xj, sigma)
        return torch.nn.functional.relu_(out)
    
def integrated_gaussian_interaction(ri0, rif, rj0, rjf, xi, xj, sigma):
    # Compute some constants that are used many times
    sqrt2 = torch.sqrt(torch.tensor(2))
    sqrt2pi = torch.sqrt(torch.tensor(2)*torch.pi)
    sqrt2_pi = torch.sqrt(torch.tensor(2)/torch.pi)
    pi3_2 = torch.pow(torch.pi, torch.tensor(3/2))
    pi2 = torch.pow(torch.pi, torch.tensor(2))


    exp1 = torch.exp(-torch.pow(ri0 - rj0, 2)/(2*torch.pow(sigma, 2)))
    exp2 = torch.exp(-torch.pow(rif - rj0, 2)/(2*torch.pow(sigma, 2)))
    erf1 = torch.erf((ri0-rj0)/(sqrt2*sigma))
    erf2 = torch.erf((rif-rj0)/(sqrt2*sigma))
    term1 = sqrt2 * pi3_2 * torch.pow(sigma, 6) * (2 * sigma * (exp1-exp2) - sqrt2pi*rj0*(erf1-erf2))

    exp1 = torch.exp(-torch.pow(ri0 - rj0, 2)/(2*torch.pow(sigma, 2)))
    exp2 = torch.exp(-torch.pow(rif - rj0, 2)/(2*torch.pow(sigma, 2)))
    erf1 = torch.erf((ri0-rj0)/(sqrt2*sigma))
    erf2 = torch.erf((rif-rj0)/(sqrt2*sigma))
    term2 = 2/3 * pi2 * torch.pow(sigma, 4) * \
        (-exp1*sqrt2_pi*sigma*(ri0**2+ri0*rj0+rj0**2+2*sigma**2) + \
         exp2*sqrt2_pi*sigma*(rif**2+rif*rj0+rj0**2+2*sigma**2) - \
         (ri0**3-rj0*(rj0**2+3*sigma**2))*erf1 + \
         (rif**3-rj0*(rj0**2+3*sigma**2))*erf2)

    exp1 = torch.exp(-torch.pow(ri0 + rj0, 2)/(2*torch.pow(sigma, 2)))
    exp2 = torch.exp(-torch.pow(rif + rj0, 2)/(2*torch.pow(sigma, 2)))
    erf1 = torch.erf((ri0+rj0)/(sqrt2*sigma))
    erf2 = torch.erf((rif+rj0)/(sqrt2*sigma))
    term3 = sqrt2 * pi3_2 * torch.pow(sigma, 6) * (2 * sigma * (exp1-exp2) + sqrt2pi*rj0*(erf1-erf2))

    exp1 = torch.exp(-torch.pow(ri0 + rj0, 2)/(2*torch.pow(sigma, 2)))
    exp2 = torch.exp(-torch.pow(rif + rj0, 2)/(2*torch.pow(sigma, 2)))
    erf1 = torch.erf((ri0+rj0)/(sqrt2*sigma))
    erf2 = torch.erf((rif+rj0)/(sqrt2*sigma))
    term4 = 2/3 * pi2 * torch.pow(sigma, 4) * \
        (-exp1*sqrt2_pi*sigma*(ri0**2-ri0*rj0+rj0**2+2*sigma**2) + \
         exp2*sqrt2_pi*sigma*(rif**2-rif*rj0+rj0**2+2*sigma**2) - \
         (ri0**3+rj0**3+3*rj0*sigma**2)*erf1 + \
         (rif**3+rj0**3+3*rj0*sigma**2)*erf2)

    exp1 = torch.exp(-torch.pow(ri0 - rjf, 2)/(2*torch.pow(sigma, 2)))
    exp2 = torch.exp(-torch.pow(rif - rjf, 2)/(2*torch.pow(sigma, 2)))
    erf1 = torch.erf((ri0-rjf)/(sqrt2*sigma))
    erf2 = torch.erf((rif-rjf)/(sqrt2*sigma))
    term5 = sqrt2 * pi3_2 * torch.pow(sigma, 6) * (2 * sigma * (exp1-exp2) - sqrt2pi*rjf*(erf1-erf2))

    exp1 = torch.exp(-torch.pow(ri0 - rjf, 2)/(2*torch.pow(sigma, 2)))
    exp2 = torch.exp(-torch.pow(rif - rjf, 2)/(2*torch.pow(sigma, 2)))
    erf1 = torch.erf((ri0-rjf)/(sqrt2*sigma))
    erf2 = torch.erf((rif-rjf)/(sqrt2*sigma))
    term6 = 2/3 * pi2 * torch.pow(sigma, 4) * \
        (-exp1*sqrt2_pi*sigma*(ri0**2+ri0*rjf+rjf**2+2*sigma**2) + \
         exp2*sqrt2_pi*sigma*(rif**2+rif*rjf+rjf**2+2*sigma**2) - \
         (ri0**3-rjf*(rjf**2+3*sigma**2))*erf1 + \
         (rif**3-rjf*(rjf**2+3*sigma**2))*erf2)

    exp1 = torch.exp(-torch.pow(ri0 + rjf, 2)/(2*torch.pow(sigma, 2)))
    exp2 = torch.exp(-torch.pow(rif + rjf, 2)/(2*torch.pow(sigma, 2)))
    erf1 = torch.erf((ri0+rjf)/(sqrt2*sigma))
    erf2 = torch.erf((rif+rjf)/(sqrt2*sigma))
    term7 = sqrt2 * pi3_2 * torch.pow(sigma, 6) * (2 * sigma * (exp1-exp2) + sqrt2pi*rjf*(erf1-erf2))

    exp1 = torch.exp(-torch.pow(ri0 + rjf, 2)/(2*torch.pow(sigma, 2)))
    exp2 = torch.exp(-torch.pow(rif + rjf, 2)/(2*torch.pow(sigma, 2)))
    erf1 = torch.erf((ri0+rjf)/(sqrt2*sigma))
    erf2 = torch.erf((rif+rjf)/(sqrt2*sigma))
    term8 = 2/3 * pi2 * torch.pow(sigma, 4) * \
        (-exp1*sqrt2_pi*sigma*(ri0**2-ri0*rjf+rjf**2+2*sigma**2) + \
         exp2*sqrt2_pi*sigma*(rif**2-rif*rjf+rjf**2+2*sigma**2) - \
         (ri0**3+rjf**3+3*rjf*sigma**2)*erf1 + \
         (rif**3+rjf**3+3*rjf*sigma**2)*erf2)


    return xi * xj * (term1+term2-term3-term4-term5-term6+term7+term8)