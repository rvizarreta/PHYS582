import numpy as np

class Particle:
    '''
    Store particle properties and kinematics.
    '''
    def __init__(self,mass,momentum,energy):
        self.mass = mass
        self.momentum = momentum
        self.energy = energy
        
    def get_gamma(self):
        '''
        Relativistic gamma factor
        '''
        return self.energy/self.mass
    
    def get_beta(self):
        '''
        Relativistic beta factor
        '''
        return np.sqrt(1-1/self.get_gamma()**2)
        #return np.linalg.norm(self.momentum)/self.energy