"""
Functions
"""

import numpy as np


def neutrino_prob(theta, del_m, E, L):
    """
    Probability that the muon neutrino will be observed as a muon neutrino and will not have oscillated into a tau or electron neutrino
    
    Args:
    - theta -- 'mixing angle', the parameter that determines the amplitude of the neutrino oscillation probability
    - del_m -- difference between the squared masses of the two neutrinos, which determines the frequency of the oscillations
    - E -- neutrino energy in GeV
    - L -- neutrino travels a distance L in km
    """
    
    sin1 = np.sin(2 * theta)
    term = (1.267 * del_m**2 * L) / E
    sin2 = np.sin(term)
    return 1 - ((sin1**2)*(sin2**2))