import numpy as np
from Particle import Particle

class PionDecaySim:
    '''
    Simulates π+ → μ+ + νμ decay
    '''
    # Particle masses in MeV/c^2
    PION_MASS = 139.57  # π+ mass MeV
    MUON_MASS = 105.66  # μ+ mass MeV
    C = 3e8 # m/s
    
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
        p_parallel_vec = np.array([p_parallel,0,0]) # This component will be modified due to gamma factor
        p_perp_vec = rest_momentum - p_parallel_vec # This component remains the same, particle does not move in this direction
        p_lab_parallel = gamma*(p_parallel + beta*E_rest) # New momentum component that moves on the beam direction, lab frame
        p_lab = p_perp_vec + np.array([p_lab_parallel,0,0]) 
        x, z = p_lab[0], p_lab[2]
        p_lab[0], p_lab[2] = z, x
        E_lab = gamma*(E_rest + beta*p_parallel)
        #if np.linalg.norm(p_lab) > E_lab:
        #    print(p_lab,E_lab)
        return Particle(mass, p_lab, E_lab) 
    
    def analyze_results(self, muons):
        """Analyze and plot results of the simulation"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Extract muon momentum and energies
        momentum = np.array([np.linalg.norm(muon.momentum) for muon in muons])
        energies = np.array([muon.energy for muon in muons])
        angles_x = np.array([np.arccos(muon.momentum[0]/np.linalg.norm(muon.momentum)) 
                          for muon in muons])
        angles_y = np.array([np.arccos(muon.momentum[1]/np.linalg.norm(muon.momentum)) 
                          for muon in muons])
        angles_z = np.array([np.arccos(muon.momentum[2]/np.linalg.norm(muon.momentum)) 
                          for muon in muons])

        print(np.max(np.rad2deg(angles_z)))
        # Print summary statistics
        print(f"Muon momentum: mean = {np.mean(momentum):.2f} MeV/c, "
              f"std = {np.std(momentum):.2f} MeV/c")
        print(f"Muon energy: mean = {np.mean(energies):.2f} MeV, "
              f"std = {np.std(energies):.2f} MeV")
        print(f"Muon angle x: mean = {np.rad2deg(np.mean(angles_x)):.2f}°, "
              f"std = {np.rad2deg(np.std(angles_x)):.2f}°")
        print(f"Muon angle y: mean = {np.rad2deg(np.mean(angles_y)):.2f}°, "
              f"std = {np.rad2deg(np.std(angles_y)):.2f}°")
        print(f"Muon angle z: mean = {np.rad2deg(np.mean(angles_z)):.2f}°, "
              f"std = {np.rad2deg(np.std(angles_z)):.2f}°")

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))

        # 1. Momentum distribution
        ax1 = fig.add_subplot(221)
        mean, std = np.mean(momentum), np.std(momentum)
        ax1.hist(momentum, bins=50, histtype='bar', color='darkblue', label=f"\n$\\mathbf{{Mean}}$ = {mean:.2f}\n$\\mathbf{{Std}}$ = {std:.2f}")
        ax1.set_xlabel('Momentum (MeV/c)', fontsize=20)
        ax1.set_ylabel('Counts', fontsize=20)
        ax1.set_title('Muon Momentum Distribution - lab frame', fontsize=20)
        ax1.grid(True)
        leg = ax1.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), 
                          fontsize='small', title=f'$\\mathbf{{Summary}}$', title_fontsize='medium')
        leg.get_title().set_fontweight('bold')
        leg._legend_box.align = "center"

        # 2. Energy distribution
        ax2 = fig.add_subplot(222)
        mean, std = np.mean(energies), np.std(energies)
        ax2.hist(energies, bins=50, histtype='bar', color='brown', label=f"\n$\\mathbf{{Mean}}$ = {mean:.2f}\n$\\mathbf{{Std}}$ = {std:.2f}")
        ax2.set_xlabel('Energy (MeV)', fontsize=20)
        ax2.set_ylabel('Counts', fontsize=20)
        ax2.set_title('Muon Energy Distribution - lab frame', fontsize=20)
        ax2.grid(True)
        leg = ax2.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), 
                          fontsize='small', title=f'$\\mathbf{{Summary}}$', title_fontsize='medium')
        leg.get_title().set_fontweight('bold')
        leg._legend_box.align = "center"

        # 3. Angular distribution x
        ax3 = fig.add_subplot(223)
        mean, std = np.rad2deg(np.mean(angles_z)), np.rad2deg(np.std(angles_z))
        ax3.hist(np.rad2deg(angles_z), bins=50, histtype='bar', color='darkgreen', label=f"\n$\\mathbf{{Mean}}$ = {mean:.2f}\n$\\mathbf{{Std}}$ = {std:.2f}")
        ax3.set_xlabel('Angle (degrees)', fontsize=20)
        ax3.set_ylabel('Counts', fontsize=20)
        ax3.set_title('Muon Angular Distribution in z - lab frame', fontsize=20)
        ax3.grid(True)
        leg = ax3.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), 
                          fontsize='small', title=f'$\\mathbf{{Summary}}$', title_fontsize='medium')
        leg.get_title().set_fontweight('bold')
        leg._legend_box.align = "center"

        # 4. Add momentum-energy correlation plot
        ax5 = fig.add_subplot(224)
        ax5.scatter(momentum, energies, color='black', alpha=0.1, s=1)
        ax5.set_xlabel('Momentum (MeV/c)', fontsize=20)
        ax5.set_ylabel('Energy (MeV)', fontsize=20)
        ax5.set_title('Muon Energy vs Momentum - lab frame', fontsize=20)
        ax5.grid(True)

        # Make plots tight and show
        plt.tight_layout()
        plt.show()