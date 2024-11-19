import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
from Strip import Strip

class DetectorPlane:
    '''
    Class representing a single detector plane
    '''
    
    def __init__(self, 
                 orientation,
                 position,  
                 n_strips,
                 strip_width,    
                 center_strip_length, 
                 strip_thickness 
                 ):
        self.orientation = orientation # X, U, V
        self.position = position # (x,y,z) center of plane
        self.n_strips = n_strips
        self.strip_width = strip_width # mm
        self.center_strip_length = center_strip_length # mm
        self.strip_thickness = strip_thickness # mm
        # Calculate rotation angle based on orientation
        if orientation == 'X':
            self.angle = 0
        elif orientation == 'U':
            self.angle = np.pi/3
        elif orientation == 'V':
            self.angle = -np.pi/3
        # Initialize strips
        self.strips = self.create_strips()
        
    def create_strips(self):
        '''
        Create and position all strips in the plane
        '''
        strips = []
        # Total width of plane
        plane_width = self.n_strips*self.strip_width
        # Starting position (leftmost strip)
        start_x = self.position[0] - plane_width/2 + self.strip_width/2
        
        for i in range(self.n_strips):
            # Base position before rotation (centered on strip)
            x = start_x + i*self.strip_width
            y = self.position[1]
            z = self.position[2]
            # Strip length based on plane location (before rotation if not X kind)
            length = 2*(self.center_strip_length/2 - np.abs(x)/np.sqrt(3))
            
            # If not X orientation, rotate position around z-axis
            if self.orientation != 'X':
                # Translate to origin, rotate, translate back
                x_rot = (x - self.position[0]) * np.cos(self.angle) - \
                       (y - self.position[1]) * np.sin(self.angle) + \
                       self.position[0]
                y_rot = (x - self.position[0]) * np.sin(self.angle) + \
                       (y - self.position[1]) * np.cos(self.angle) + \
                       self.position[1]
                x, y = x_rot, y_rot
            
            strips.append(Strip(
                id=i,
                width=self.strip_width,
                length=length,
                thickness=self.strip_thickness,
                position=np.array([x, y, z])
            ))
        
        return strips
    
    def to_local_coordinates(self, point, strip_pos, angle):
        '''
        Transform point to local strip coordinates (angle = 0°)
        '''
        # Translate to strip center
        translated = point - strip_pos
        # Rotate back as if they were parallel strips or X-like plane
        if self.orientation != "X":
            x = translated[0] * np.cos(-angle) - translated[1] * np.sin(-angle)
            y = translated[0] * np.sin(-angle) + translated[1] * np.cos(-angle)
            return np.array([x, y, translated[2]])
        return translated

    def find_intersecting_strips(self, point):
        '''
        Find strips that intersect with a given point
        '''
        intersecting_strips = []
        for strip in self.strips:
            # Transform point to strip's local coordinates
            local_point = self.to_local_coordinates(point, strip.position, self.angle)
            # Check if point is within strip boundaries
            if (abs(local_point[0]) <= self.strip_width/2 and
                abs(local_point[1]) <= strip.length/2 and
                abs(local_point[2]) <= self.strip_thickness):
                intersecting_strips.append(strip)
        return intersecting_strips
    
