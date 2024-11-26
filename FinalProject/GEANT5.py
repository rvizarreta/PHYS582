import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from OpticalModel import OpticalModel
from PMTModel import PMTModel
from DiscriminatorModel import DiscriminatorModel

class GEANT5:
    '''
    Class simulating muon motion through detector.
    '''
    
    OpticalModel = OpticalModel()
    PMTModel = PMTModel()
    DiscriminatorModel = DiscriminatorModel()
    
    
    def __init__(self):
        '''
        Initialize interaction
        '''
        
    def calculate_bethe_bloch(self, particle):
        ELECTRON_MASS = 0.511 # MeV
        K = 0.307075
        z = 1
        avg_Z = 0.0759*1 + 0.919*6 + 0.0051*8 + 0.0077*22
        avg_A = 0.0759*1.00784 + 0.919*12.01 + 0.0051*32 + 0.0077*47.867
        beta, gamma = particle.get_beta(), particle.get_gamma()
        I_mean = avg_Z*12 + 7
        M = particle.mass
        delta = 0
        Tmax = (2*ELECTRON_MASS*(beta*gamma)**2)/(1 + 2*ELECTRON_MASS/M*np.sqrt(1+(beta*gamma)**2) + (ELECTRON_MASS/M)**2)
        dEdX = K*z**2*(avg_Z/avg_A)*(1/beta**2)*(0.5*np.log(2*ELECTRON_MASS*(beta*gamma)**2*Tmax/I_mean**2) - beta**2 - delta/2)
        return dEdX # cm

    def calculate_highland_angle(self, detector, particle):
        '''
        Calculate RMS scattering angle using Highland formula
        '''
        t = detector.strip_thickness/406.8 # Thickness of 20 mm and radiation length of 286 mm
        momentum = particle.momentum
        return (13.6/momentum)*np.sqrt(t)*(1 + 0.038*np.log(t))

    def create_orthonormal_basis(self, direction):
        '''
        Create orthonormal basis given a direction vector
        '''
        p = direction / np.linalg.norm(direction) # Normalize direction
        if abs(p[2]) < 1/np.sqrt(2): # Choose most stable axis to cross with
            u = np.cross(p, [0, 0, 1]) # If not too parallel to z, cross with z
        else:
            u = np.cross(p, [1, 0, 0]) # If too parallel to z, cross with x
        u = u / np.linalg.norm(u) # Normalize u
        v = np.cross(p, u) # v completes right-handed system
        return u, v

    def calculate_new_direction(self, detector, particle):
        '''
        Apply multiple scattering to momentum direction
        '''
        delta_theta = self.calculate_highland_angle(detector, particle)
        theta = np.random.normal(0, delta_theta)  # Polar angle
        phi = np.random.uniform(0, 2*np.pi)  # Azimuthal angle
        momentum_dir = particle.momentum/np.linalg.norm(particle.momentum)
        u, v = self.create_orthonormal_basis(momentum_dir) # Get orthonormal basis
        new_dir = (np.cos(theta)*momentum_dir + np.sin(theta)*(np.cos(phi)*u + np.sin(phi)*v))
        return new_dir/np.linalg.norm(new_dir)
    
    def track_muon(self, detector, particle):
        '''
        Track muon through detector planes and record hits.
        '''
        initial_momentum = particle.momentum
        initial_energy = particle.energy
        # Starting position 1m before detector in central axis
        start_pos = particle.position  # 1m = 1000mm before z=0
        # Calculate velocity direction (assuming relativistic particle)
        velocity_dir = initial_momentum / np.linalg.norm(initial_momentum)
        # Initialize results list
        hits = []
        # Loop through all modules
        for module_idx, module in enumerate(detector.modules):
            # Check intersection with each plane in the module
            for plane_name, plane in [('X', module.x_plane1), ('U', module.u_plane), ('X', module.x_plane2), ('V', module.v_plane)]:
                # Calculate intersection point with plane
                t = (plane.position[2] - start_pos[2]) / velocity_dir[2]
                intersection = start_pos + velocity_dir*t
                # Find which strips (if any) were hit
                hit_strips = plane.find_intersecting_strips(intersection)
                # Calculate new energy
                dEdX = self.calculate_bethe_bloch(particle)
                new_energy = initial_energy + dEdX*(detector.strip_thickness/10)
                if new_energy > particle.mass:
                    # If there was a hit, record it
                    particle.energy = new_energy
                    # Calculate new momentum
                    velocity_dir = self.calculate_new_direction(detector, particle)
                    new_momentum = np.sqrt(new_energy**2 - particle.mass**2)*velocity_dir
                    if hit_strips:
                        photons = -1*self.OpticalModel.birks_response(dEdX*(detector.strip_thickness/10), detector.strip_thickness/10)
                        charge = self.PMTModel.PhotonsToCharge(photons)
                        discriminator = self.DiscriminatorModel.check_discriminator(charge)
                        adc = self.PMTModel.getADC(charge)
                        hit_info = {
                            "module": module_idx,
                            "plane_orientation": plane_name,
                            "strip": hit_strips[0],  # Take first hit if multiple
                            "intersection_point": intersection.tolist(),
                            "energy": new_energy,  
                            "momentum_vector" : new_momentum,
                            "photons" : photons,
                            "charge" : charge,
                            "discriminator" : discriminator,
                            "adc" : adc
                        }
                        hits.append(hit_info)
                elif initial_energy > particle.mass and new_energy < particle.mass:
                    dE = initial_energy - particle.mass
                    if hit_strips:
                        particle.energy = particle.mass
                        new_momentum = 0*self.calculate_new_direction(detector,particle)
                        photons = self.OpticalModel.birks_response(dE, detector.strip_thickness/10)
                        charge = self.PMTModel.PhotonsToCharge(photons)
                        discriminator = self.DiscriminatorModel.check_discriminator(charge)
                        adc = self.PMTModel.getADC(charge)
                        hit_info = {
                            "module": module_idx,
                            "plane_orientation": plane_name,
                            "strip": hit_strips[0],  # Take first hit if multiple
                            "intersection_point": intersection.tolist(),
                            "energy": particle.mass,  # Muon stops in strip
                            "momentum_vector": new_momentum,
                            "photons" : photons,
                            "charge" : charge,
                            "discriminator" : discriminator,
                            "adc" : adc
                        }
                        hits.append(hit_info)
                initial_energy = new_energy
        return hits
    
    def plot_from_dict(self, data, x_key, y_key, x_label, y_label):
        '''
        Plots an x,y scatter plot using values from a list of dictionaries.
        '''
        x_values = [item[x_key][2] for item in data]
        y_values = [item[y_key] for item in data]

        plt.figure(figsize=(8, 6))
        plt.scatter(x_values, y_values, color='blue', alpha=0.7, label='Data points')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(f'{x_key.capitalize()} vs {y_key.capitalize()}', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        #plt.legend()
        plt.show()