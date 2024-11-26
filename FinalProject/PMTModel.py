import numpy as np

class PMTModel:
    '''
    Photomultiplier tube model
    '''
    PMT_GAIN = 5e5
    QUANTUM_EFFICIENCY = 0.12 # Minimum quantum efficiency
    E_CHARGE = 1.602177e-4 # In fC
    ADC_GAIN = 0.0625
    ADC_PEDESTAL = 450 
    ADC_PEDESTAL_RMS = 8.0 # ADC Noise
    
    def PhotonsToPE(self, photons):
        mean_pe = photons*self.QUANTUM_EFFICIENCY
        actual_pe = np.random.default_rng().poisson(mean_pe)
        return actual_pe
        
    def PhotonsToCharge(self, photons):
        PE = self.PhotonsToPE(photons)
        charge = PE*self.E_CHARGE*self.PMT_GAIN
        return charge
    
    def getADC(self, charge):
        noise = np.random.default_rng().normal(self.ADC_PEDESTAL, self.ADC_PEDESTAL_RMS)
        adc = charge*self.ADC_GAIN + noise
        return int(adc)