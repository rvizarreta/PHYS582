class DiscriminatorModel():
    '''
    Modeling detector discriminator
    '''
    THRESHOLD = 1200
    
    def check_discriminator(self, charge):
        return True if charge > self.THRESHOLD else False
        