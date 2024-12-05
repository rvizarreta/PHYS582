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
    highland_angles = []
    
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
        I_mean = 68.7 #avg_Z*12 + 7
        M = particle.mass
        delta = 0
        Tmax = (2*ELECTRON_MASS*(beta*gamma)**2)/(1 + 2*ELECTRON_MASS/M*np.sqrt(1+(beta*gamma)**2) + (ELECTRON_MASS/M)**2)
        dEdX = -1*1.060*K*z**2*(0.53768)*(1/beta**2)*(0.5*np.log(2*ELECTRON_MASS*(beta*gamma)**2*Tmax/I_mean**2) - beta**2 - delta/2) 
        return dEdX # cm

    def calculate_highland_angle(self, detector, particle):
        '''
        Calculate RMS scattering angle using Highland formula
        '''
        t = detector.strip_thickness/406.8 # Thickness of 20 mm and radiation length of 406.8 mm
        momentum = np.linalg.norm(particle.momentum)
        return np.sqrt(2)*(13.6/(momentum*particle.get_beta()))*np.sqrt(t)*(1 + 0.038*np.log(t))

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
        self.highland_angles.append(delta_theta)
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
        momentum_magnitude = np.linalg.norm(initial_momentum)
        initial_energy = particle.energy
        # Starting position 1m before detector in central axis
        start_pos = particle.position  # 1m = 1000mm before z=0
        # Calculate velocity direction (assuming relativistic particle)
        velocity_dir = initial_momentum / momentum_magnitude
        # Initialize results list
        hits = []
        # Loop through all modules
        for module_idx, module in enumerate(detector.modules):
            # Check intersection with each plane in the module
            for plane_name, plane in [('X', module.x_plane1), ('U', module.u_plane), ('X', module.x_plane2), ('V', module.v_plane)]:
                collected = False
                # Calculate intersection point with plane
                t = (plane.position[2] - start_pos[2]) / velocity_dir[2]
                intersection = start_pos + velocity_dir*t
                particle.position = intersection
                # Find which strips (if any) were hit
                hit_strips, pmt_distance = plane.find_intersecting_strips(intersection)
                # Calculate new energy
                dEdX = -1*self.calculate_bethe_bloch(particle)
                energy_deposit = abs(dEdX*3*(detector.strip_thickness/10))
                new_energy = initial_energy - energy_deposit
                if new_energy > particle.mass:
                    # If there was a hit, record it
                    particle.energy = new_energy
                    # Calculate new momentum
                    velocity_dir = self.calculate_new_direction(detector, particle)
                    new_momentum = np.sqrt(new_energy**2 - particle.mass**2)*velocity_dir
                    if hit_strips:
                        photons = self.OpticalModel.birks_response(energy_deposit, detector.strip_thickness/10)*detector.strip_thickness
                        if photons > 1e5:
                            collected = True
                        charge = self.PMTModel.PhotonsToCharge(photons, pmt_distance)
                        discriminator = self.DiscriminatorModel.check_discriminator(charge)
                        adc = self.PMTModel.getADC(charge, 'high', pmt_distance)
                        hit_info = {
                            "module": module_idx,
                            "plane_orientation": plane_name,
                            "strip": hit_strips[0],  # Take first hit if multiple
                            "hit_position" : particle.position,
                            "intersection_point": intersection.tolist(),
                            "pmt_distance" : pmt_distance,
                            "energy": new_energy,  
                            "dEdX" : dEdX,
                            "energy_deposit" : abs(energy_deposit),
                            "KE" : new_energy - particle.mass,
                            "momentum_vector" : new_momentum,
                            "momentum_magnitude" : np.linalg.norm(new_momentum),
                            "photons" : photons,
                            "collected" : collected,
                            "charge" : charge,
                            "discriminator" : discriminator,
                            "adc" : adc,
                            "contained" : False
                        }
                        hits.append(hit_info)
                elif initial_energy > particle.mass and new_energy <= particle.mass:
                    dE = initial_energy - particle.mass
                    if hit_strips:
                        particle.energy = particle.mass
                        new_momentum = 0*self.calculate_new_direction(detector,particle)
                        photons = self.OpticalModel.birks_response(dE, detector.strip_thickness/10)
                        charge = self.PMTModel.PhotonsToCharge(photons, pmt_distance)
                        discriminator = self.DiscriminatorModel.check_discriminator(charge)
                        adc = self.PMTModel.getADC(charge, 'low', pmt_distance)
                        hit_info = {
                            "module": module_idx,
                            "plane_orientation": plane_name,
                            "strip": hit_strips[0],  # Take first hit if multiple
                            "hit_position" : particle.position,
                            "intersection_point": intersection.tolist(),
                            "pmt_distance" : pmt_distance,
                            "energy": particle.mass,  # Muon stops in strip
                            "dEdX" : dEdX,
                            "energy_deposit" : dE,
                            "KE" : 0.0,
                            "momentum_vector": new_momentum,
                            "photons" : photons,
                            "collected" : collected,
                            "charge" : charge,
                            "discriminator" : discriminator,
                            "adc" : adc, 
                            "contained" : True
                        }
                        hits.append(hit_info)
                initial_energy = new_energy
        return hits
    
    def plot_from_dict(self, data, x_key, y_key, x_label, y_label):
        '''
        Plots an x,y scatter plot using values from a list of dictionaries.
        '''
        if x_key == 'intersection_point':
            x_values = [item[x_key][2] for item in data]
        else:
            x_values = [item[x_key] for item in data]
        y_values = [item[y_key] for item in data]

        plt.figure(figsize=(8, 6))
        plt.scatter(x_values, y_values, color='blue', alpha=0.7, label='Data points')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(f'{x_key.capitalize()} vs {y_key.capitalize()}', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        #plt.legend()
        plt.show()
        
    def extract_table_data(self):
        '''
        Extract data from the specified text file
        '''
        # Read the file content
        with open('table.txt', 'r') as f:
            lines = f.readlines()

        energies = []  # MeV
        momenta = []   # MeV/c  
        csda_ranges = []  # g/cm²

        for line in lines:
            # Skip header/empty lines
            if not line.strip() or 'T' in line or '[MeV]' in line or 'Muon critical energy' in line:
                continue

            try:
                parts = line.split()
                if len(parts) < 8:  # We need at least the basic columns
                    continue

                energy = float(parts[0])
                momentum = float(parts[1]) 
                csda = float(parts[8])

                # Add values to arrays if they make physical sense
                if energy > 0 and momentum > 0 and csda > 0:
                    energies.append(energy)
                    momenta.append(momentum)
                    csda_ranges.append(csda)

            except (ValueError, IndexError):
                continue

        return np.array(energies), np.array(momenta), np.array(csda_ranges)
    
    def interpolate_from_csda(self, csda_value, csda_ranges, values):
        """
        Interpolate a value for a given CSDA range using log-log interpolation.
        """
        # Convert to log space
        log_csda = np.log10(csda_value)
        log_csda_ranges = np.log10(csda_ranges)
        log_values = np.log10(values)

        # Perform linear interpolation in log space
        interpolated_log_value = np.interp(log_csda, log_csda_ranges, log_values)

        # Convert back to linear space
        return 10**interpolated_log_value

    def get_kinematics_from_csda(self, csda_value):
        """
        Get the interpolated momentum and kinetic energy for a given CSDA range.

        Args:
            csda_value: Input CSDA range in g/cm^2

        Returns:
            Tuple of (momentum in MeV/c, kinetic energy in MeV)
        """
        energies, momenta, csda_ranges = self.extract_table_data()

        if csda_value < np.min(csda_ranges) or csda_value > np.max(csda_ranges):
            raise ValueError(f"CSDA range {csda_value} g/cm^2 is outside the valid range " 
                           f"[{np.min(csda_ranges):.2e}, {np.max(csda_ranges):.2e}] g/cm^2")

        momentum = self.interpolate_from_csda(csda_value, csda_ranges, momenta)
        energy = self.interpolate_from_csda(csda_value, csda_ranges, energies)

        return momentum, energy
    
    def linear_fit(self, points):
        '''
        Make linear fit starting from start and end points
        '''
        centroid = np.mean(points, axis=0) # Recover data centroid to determine offset from origin.
        centered_points = points - centroid # Shift data points to origin for ease of calculation
        _, _, Vh = np.linalg.svd(centered_points, full_matrices=False) # Apply Single Value Decomposition and recover the direction that points to the larger spread
        direction = Vh[0] 
        direction = direction / np.linalg.norm(direction) # Our best fit line will point in this direction
        t_values = np.dot(centered_points, direction) # Project centered points to this direction
        length = np.abs(np.max(t_values) - np.min(t_values)) # The track length is the distance between max and min values of centered points
        return length
    
    def make_histogram_with_residuals(self, array1, array2, x_label, bins=50, title1="Truth", title2="Reco", color1="darkblue", color2="darkred"):
        import matplotlib.pyplot as plt
        '''
        Plot overlapping histograms with aligned residual subplot.
        Parameters:
        '''
        # Create figure with GridSpec for no spacing between subplots
        fig = plt.figure(figsize=(12, 8))
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

        # Define common binning for both histograms
        range_min = min(array1.min(), array2.min())
        range_max = max(array1.max(), array2.max())
        bin_edges = np.linspace(range_min, range_max, bins+1)

        # Top subplot: Histograms
        ax1 = fig.add_subplot(gs[0])
        mean1, std1 = np.mean(array1), np.std(array1)
        mean2, std2 = np.mean(array2), np.std(array2)

        # Plot histograms with same binning
        n1, _, _ = ax1.hist(array1, bins=bin_edges, alpha=0.7, 
                            label=f"{title1}: Mean = {mean1:.2f}, Std = {std1:.2f}", 
                            color=color1)
        n2, _, _ = ax1.hist(array2, bins=bin_edges, alpha=0.7,
                            label=f"{title2}: Mean = {mean2:.2f}, Std = {std2:.2f}", 
                            color=color2)

        ax1.set_ylabel('Counts', fontsize=20)
        ax1.grid(True)
        ax1.legend(loc='upper right', title="Summary")
        # Remove x-axis labels from top plot
        ax1.tick_params(labelbottom=False)

        # Bottom subplot: Residuals
        ax2 = fig.add_subplot(gs[1])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        residuals = n1 - n2  # Calculate bin-by-bin differences

        ax2.bar(bin_centers, residuals, width=np.diff(bin_edges), color='black')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)

        ax2.set_xlabel(x_label, fontsize=20)
        ax2.set_ylabel('Residuals')
        ax2.grid(True)

        # Align x-axes of both plots
        ax2.set_xlim(ax1.get_xlim())

        plt.show()