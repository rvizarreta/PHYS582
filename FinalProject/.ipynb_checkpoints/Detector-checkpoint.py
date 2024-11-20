from Plane import DetectorPlane
from Module import Module
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class Detector:
    '''
    Class representing the full detector with multiple XUVY modules
    '''
    
    def __init__(self,
                 n_modules = 15,
                 module_spacing = 20.0,  
                 module_gap = 40.0,      
                 n_strips = 127,
                 strip_width = 33,      
                 strip_thickness = 1.0    
                 ):
        '''
        Initialize detector with multiple XUVY modules
        '''
        self.n_modules = n_modules
        self.module_spacing = module_spacing # mm between planes in a module
        self.module_gap = module_gap # mm between modules
        self.plane_width = n_strips*strip_width # mm (x dimension)
        self.plane_height = 2*self.plane_width/np.sqrt(3) # mm (y dimension)
        self.outer_radius = self.plane_height/2 # radius of circumscribed circle of hexagon (mm)
        self.n_strips = n_strips
        self.strip_width = strip_width # mm
        self.strip_thickness = strip_thickness # mm
        # Calculate total length of detector
        self.length = (n_modules*(3*module_spacing + 4*strip_thickness) + 
                      (n_modules - 1)*module_gap)
        # Create modules
        self.modules = self.create_modules()
        
    def plot_detector_xy(self, point):
        '''
        Plot detector hexagonal boundary and point location
        '''
        # Create hexagon points
        angles = np.linspace(np.pi/6, 2*np.pi+np.pi/6, 7)
        outer_hex_x = self.outer_radius * np.cos(angles)
        outer_hex_y = self.outer_radius * np.sin(angles)

        # Create plot
        plt.figure(figsize=(8, 8))
        plt.plot(outer_hex_x, outer_hex_y, 'b-', label='Outer boundary')
        plt.plot(point[0], point[1], 'kx', markersize=10, label=f'Point ({point[0]:.1f}, {point[1]:.1f})')

        plt.axhline(y=0, color='gray', linestyle=':')
        plt.axvline(x=0, color='gray', linestyle=':')

        plt.xlabel('X [mm]')
        plt.ylabel('Y [mm]')
        plt.title('Detector XY View')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()

        return plt.gcf()
        
    def create_modules(self):
        '''
        Create all detector modules
        '''
        modules = []
        for i in range(self.n_modules):
            # Calculate z position for this module
            module_start_z = i*(3*self.module_spacing + 
                                4*self.strip_thickness + 
                                self.module_gap)
            # Positionate planes for this module
            x1_pos = module_start_z
            u_pos = x1_pos + self.strip_thickness + self.module_spacing
            x2_pos = u_pos + self.strip_thickness + self.module_spacing 
            v_pos = x2_pos + self.strip_thickness + self.module_spacing 
            # Create each plane in the module
            x_plane1 = DetectorPlane(
                orientation="X",
                position=np.array([0, 0, x1_pos]),
                n_strips = self.n_strips,
                strip_width=self.strip_width,
                center_strip_length=self.plane_height,
                strip_thickness=self.strip_thickness
            )
            u_plane = DetectorPlane(
                orientation="U",
                position=np.array([0, 0, u_pos]),
                n_strips = self.n_strips,
                strip_width=self.strip_width,
                center_strip_length=self.plane_height,
                strip_thickness=self.strip_thickness
            )
            
            x_plane2 = DetectorPlane(
                orientation="V",
                position=np.array([0, 0, x2_pos]),
                n_strips = self.n_strips,
                strip_width=self.strip_width,
                center_strip_length=self.plane_height,
                strip_thickness=self.strip_thickness
            )
            
            v_plane = DetectorPlane(
                orientation="X",
                position=np.array([0, 0, v_pos]),
                n_strips = self.n_strips,
                strip_width=self.strip_width,
                center_strip_length=self.plane_height,
                strip_thickness=self.strip_thickness
            )
            
            modules.append(Module(
                id=i,
                x_plane1=x_plane1,
                u_plane=u_plane,
                v_plane=v_plane,
                x_plane2=x_plane2,
                z_position=module_start_z
            ))        
        return modules
    
    def find_intersecting_strips(self, point):
        '''
        Find all strips intersecting with a given point
        '''
        intersections = []
        # Find which module contains this z position
        module_length = (3*self.module_spacing + 
                        4*self.strip_thickness)
        module_idx = int(point[2] / module_length)
        if 0 <= module_idx < self.n_modules:
            module = self.modules[module_idx]
            
        # Check each plane in the module
        for plane_name, plane in [('x1', module.x_plane1), 
                                ('u', module.u_plane),
                                ('x2', module.x_plane2),
                                ('v', module.v_plane)]:
            strips = plane.find_intersecting_strips(point)
            for strip in strips:
                intersections.append((module.id, plane_name, strip))
        self.plot_detector_xy(point)
        return intersections
    
    def get_detector_info(self):
        '''
        Return string with detector information
        '''
        info = [
            f"Detector Configuration:",
            f"Number of modules: {self.n_modules}",
            f"Total length: {self.length:.1f} mm",
            f"Plane dimensions: {self.plane_width:.1f} x {self.plane_height:.1f} mm",
            f"Module spacing: {self.module_spacing:.1f} mm",
            f"Module gap: {self.module_gap:.1f} mm",
            f"\nModule structure:",
            f"- X plane",
            f"- U plane (+60°)",
            f"- X plane",
            f"- V plane (-60°)"
        ]
        return "\n".join(info)

    def create_hexagonal_plane(self, center, outer_radius, thickness, angle=0):
        """Create vertices for a hexagonal plane"""
        # Create hexagon vertices
        angles = np.linspace(np.pi/6, 2*np.pi+np.pi/6, 7)[:-1]  # 6 points for hexagon
        
        # Base vertices for front and back faces
        front_vertices = []
        back_vertices = []
        
        for theta in angles:
            x = outer_radius * np.cos(theta)
            y = outer_radius * np.sin(theta)
            front_vertices.append([x, y, -thickness/2])
            back_vertices.append([x, y, thickness/2])
            
        vertices = np.array(front_vertices + back_vertices)
        
        # Rotate around z-axis if needed
        if angle != 0:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            vertices = np.dot(vertices, rot_matrix.T)
        
        # Translate to center position
        vertices += center
        
        # Create faces for plotly mesh3d
        n_points = len(angles)
        
        # Indices for front and back faces
        i = []
        j = []
        k = []
        
        # Front face
        for idx in range(1, n_points-1):
            i.append(0)
            j.append(idx)
            k.append(idx+1)
        
        # Back face
        for idx in range(1, n_points-1):
            i.append(n_points)
            j.append(n_points + idx)
            k.append(n_points + idx + 1)
        
        # Side faces
        for idx in range(n_points):
            next_idx = (idx + 1) % n_points
            i.extend([idx, idx, n_points + idx])
            j.extend([next_idx, n_points + idx, n_points + next_idx])
            k.extend([n_points + next_idx, next_idx, next_idx])
        
        return vertices[:, 0], vertices[:, 1], vertices[:, 2], np.array(i), np.array(j), np.array(k)

    def visualize(self, width=1500, height=1000):
        """Create interactive 3D visualization of the detector"""
        fig = go.Figure()
        
        plane_info = {
            'X': {
                'color': '#3366cc',
                'name': 'X Planes (0°)',
                'description': 'Vertical strips'
            },
            'U': {
                'color': '#2ecc71',
                'name': 'U Planes (+60°)',
                'description': 'Strips rotated +60°'
            },
            'V': {
                'color': '#e74c3c',
                'name': 'V Planes (-60°)',
                'description': 'Strips rotated -60°'
            }
        }
        
        # Create each module
        for i in range(self.n_modules):
            module_start_z = i * (3 * self.module_spacing + 
                                4 * self.strip_thickness + 
                                self.module_gap)
            
            positions = [
                (module_start_z, 'X', 0),
                (module_start_z + self.module_spacing + self.strip_thickness, 'U', np.pi/3),
                (module_start_z + 2*self.module_spacing + 2*self.strip_thickness, 'X', -np.pi/3),
                (module_start_z + 3*self.module_spacing + 3*self.strip_thickness, 'V', 0)
            ]
            
            # Create each plane
            for z_pos, plane_type, angle in positions:
                x, y, z, i, j, k = self.create_hexagonal_plane(
                    center=np.array([0, 0, z_pos]),
                    outer_radius=self.outer_radius,
                    thickness=self.strip_thickness,
                    angle=angle
                )
                
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    opacity=0.3,
                    color=plane_info[plane_type]['color'],
                    name=plane_info[plane_type]['name'],
                    hovertemplate=f"{plane_info[plane_type]['name']}<br>"
                                 f"{plane_info[plane_type]['description']}<br>"
                                 f"Module: {i}<br>"
                                 f"Z position: {z_pos:.1f} mm<extra></extra>"
                ))
        
        # Update layout
        fig.update_layout(
            width=width,
            height=height,
            scene=dict(
                xaxis=dict(range=[-self.plane_height, self.outer_radius*1.2], title='X [mm]'),
                yaxis=dict(range=[-self.outer_radius*1.2, self.outer_radius*1.2], title='Y [mm]'),
                zaxis=dict(range=[-10, self.length + 10], title='Z [mm]'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.5),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data'
            ),
            title=dict(
                text='MINERνA-like Detector Layout (Hexagonal)',
                x=0.5,
                y=0.95
            ),
            showlegend=True,
            legend=dict(
                title=dict(text='Plane Types', font=dict(size=14)),
                itemsizing='constant',
                itemwidth=30,
                font=dict(size=12),
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            annotations=[
                dict(
                    text=f"Total modules: {self.n_modules}<br>"
                         f"Module spacing: {self.module_spacing} mm<br>"
                         f"Module gap: {self.module_gap} mm<br>",
                    xref="paper", yref="paper",
                    x=0, y=1,
                    xanchor='left', yanchor='top',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    font=dict(size=12)
                )
            ]
        )
        
        return fig
    

