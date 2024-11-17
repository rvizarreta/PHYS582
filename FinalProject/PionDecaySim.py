import numpy as np
from Particle import Particle

class PionDecaySim:
    '''
    Simulates π+ → μ+ + νμ decay
    '''
    # Particle masses in MeV/c^2
    PION_MASS = 139.57  # π+ mass
    MUON_MASS = 105.66  # μ+ mass
    
    def __init__(self, n_particles, pion_momentum_mean, pion_momentum_std):
        self.n_particles = n_particles
        self.pion_momentum_mean = pion_momentum_mean
        self.pion_momentum_std = pion_momentum_std
        
    def pion_decay_lab_frame(self):
        '''
        Generate distribution of pions resulting from decay in lab frame
        '''
        momentum = np.random.normal(self.pion_momentum_mean, self.pion_momentum_std, self.n_particles) # pion initial p magnitude
        phi = np.random.uniform(0, 2*np.pi, self.n_particles) # Polar angle distribution
        cos_theta = np.random.uniform(-1, 1, self.n_particles) # Azimuthal cos distribution
        sin_theta = np.sqrt(1 - cos_theta**2) # Azimuthal sin distribution
        px = momentum*sin_theta*np.cos(phi) # Momentum x component
        py = momentum*sin_theta*np.sin(phi) # Momentum y component
        pz = momentum*cos_theta # Momentum z component
        pions = [] # Array that will store pion objects
        for i in range(self.n_particles): # Compute particles momentum and energy in lab frame
            momentum = np.array([px[i], py[i], pz[i]])
            energy = np.sqrt(self.PION_MASS**2 + np.sum(momentum**2))
            pions.append(Particle(self.PION_MASS, momentum, energy))
        return pions
            
    def after_decay_particles_rest_frame(self):
        '''
        Generate muon and neutrino momenta in rest frame after pion decay
        '''
        momentum = (self.PION_MASS**2 - self.MUON_MASS**2) / (2*self.PION_MASS)
        phi = np.random.uniform(0, 2*np.pi, self.n_particles) # Polar angle distribution
        cos_theta = np.random.uniform(-1, 1, self.n_particles) # Azimuthal cos distribution
        sin_theta = np.sqrt(1 - cos_theta**2) # Azimuthal sin distribution
        px = momentum*sin_theta*np.cos(phi) # Momentum x component
        py = momentum*sin_theta*np.sin(phi) # Momentum y component
        pz = momentum*cos_theta # Momentum z component
        muons_momentum, nus_momentum = [], []
        for i in range(self.n_particles): # Save particles momentum in rest frame
            muons_momentum.append(np.array([px[i], py[i], pz[i]]))
            nus_momentum.append(np.array([-px[i], -py[i], -pz[i]]))
        return muons_momentum, nus_momentum
    
    def after_decay_particles_lab_frame(self, rest_momentum, mass, pion):
        '''
        Generate particle momentum in lab frame after pion decay
        '''
        E_rest = np.sqrt(mass**2 + np.sum(rest_momentum**2)) # Rest frame energy
        beta = pion.get_beta()
        gamma = pion.get_gamma()
        p_parallel = rest_momentum[0] # x-component of momentum
        p_parallel_vec = np.array([p_parallel,0,0])
        p_perp_vec = rest_momentum - p_parallel_vec
        p_lab_parallel = gamma*(p_parallel + beta*E_rest)
        p_lab = p_perp_vec + p_lab_parallel 
        E_lab = gamma*(E_rest + beta*p_parallel)
        return Particle(mass, p_lab, E_lab)