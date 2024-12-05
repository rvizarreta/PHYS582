import numpy as np

class OpticalModel:
    '''
    Class that provides the MINERvA optical model.
    '''
    BIRKS_CONSTANT = 0.0905 # mm/GeV
    LIGHT_YIELD = 8000 # Photons/MeV
    
    def birks_response(self, dE, dX):
        '''
        Function that calculates optical response.
        '''
        dEdX = -1*1e-3*dE/dX # GeV/mm
        photons = self.LIGHT_YIELD*dE/(1+self.BIRKS_CONSTANT*dEdX) # Photons
        photons_fluctuations = np.random.poisson(photons)
        return photons_fluctuations
    
    
    