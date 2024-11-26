class OpticalModel:
    '''
    Class that provides the MINERvA optical model.
    '''
    BIRKS_CONSTANT = 0.0905 # mm/MeV
    LIGHT_YIELD = 8000 # Photons/MeV
    REFLECTION_COEFF = 0.83
    FIBER_REF_INDEX = 1.923
    C = 299.792458 # mm/ns
    
    def birks_response(self, dE, dX):
        '''
        Function that calculates optical response.
        '''
        dEdX = dE/dX
        return self.LIGHT_YIELD*dE/(1+self.BIRKS_CONSTANT*dEdX) # Photons
    
    
    