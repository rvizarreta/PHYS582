class Module:
    '''
    Class to store detector single module properties
    '''
    def __init__(self, id, x_plane1, u_plane, x_plane2, v_plane, z_position):
        self.id = id
        self.x_plane1 = x_plane1
        self.u_plane = u_plane
        self.x_plane2 = x_plane2
        self.v_plane = v_plane
        self.z_position = z_position