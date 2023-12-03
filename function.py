"""
Functions
"""

import numpy as np
import scipy as sp


def neutrino_prob(E, theta=(np.pi)/4, dm2=2.4, L=295):
    """
    Probability that the muon neutrino will be observed as a muon neutrino and will not have oscillated into a tau or electron neutrino
    
    Args:
    - E -- neutrino energy in GeV
    - theta -- 'mixing angle', the parameter that determines the amplitude of the neutrino oscillation probability
    - dm2 -- difference between the squared masses of the two neutrinos, which determines the frequency of the oscillations
    - L -- neutrino travels a distance L in km
    """
    
    sin1 = np.sin(2 * theta)
    term = (1.267 * (dm2*1e-3) * L) / E
    sin2 = np.sin(term)

    return 1 - ((sin1**2)*(sin2**2))


def NLL(lamb, m):
    """
    Calculate negative log likelihood

    Args:
    - lamb -- the expected average number
    - m -- observed number of neutrino events in bin i.
    """

    term2 = m * np.log(lamb)
    term3 = np.log(sp.special.factorial(m))
    sum = lamb - term2 + term3

    return 2*np.sum(sum)
