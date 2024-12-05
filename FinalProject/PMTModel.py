import numpy as np

class PMTModel:
    '''
    Photomultiplier tube model
    '''
    PMT_GAIN = 500 # electrons / PE
    QUANTUM_EFFICIENCY = 0.12 # Minimum quantum efficiency
    E_CHARGE = 1.602177e-4 # In fC
    ADC_GAIN = 0.0625
    ADC_PEDESTAL = 450 
    ADC_PEDESTAL_RMS = 8.0 # ADC Noise
    MAX_ADC = 4095
    REFLECTION_COEFF = 0.83
    FIBER_REF_INDEX = 1.923
    BAGGIE_ATT = 0.9
    CLEAR_ATT = 0.95
    S2S_FACTOR = 1.0
    STRIP_ATT = 0.35
    DIRECT_DIFRACTION = 0.6
    SPARSIFICATION_THRESHOLD = 3  # In units of pedestal RMS
    GAINS = {
        "high" : 64,
        "medium" : 8,
        "low" : 1
    }
    
    def PhotonsToPE(self, photons, pmt_distance):
        total_attenuation = 1/(self.BAGGIE_ATT*self.CLEAR_ATT*self.STRIP_ATT)
        mean_pe = total_attenuation*photons
        direct_pe = np.random.RandomState(42).poisson(mean_pe*self.DIRECT_DIFRACTION)
        reflected_pe = np.random.RandomState(42).poisson(mean_pe*(1-self.DIRECT_DIFRACTION))
        return direct_pe + reflected_pe
        
    def PhotonsToCharge(self, photons, pmt_distance):
        PE = self.PhotonsToPE(photons, pmt_distance)
        charge = PE*self.E_CHARGE*self.PMT_GAIN
        return charge # in fC
    
    def getADC(self, charge, gain_range, pmt_distance):
        trip_charge = charge*self.GAINS[gain_range]
        pedestal_noise = np.random.default_rng().normal(self.ADC_PEDESTAL, self.ADC_PEDESTAL_RMS)
        adc_count = int(min(trip_charge*self.ADC_GAIN + pedestal_noise, self.MAX_ADC))
        if adc_count < (self.ADC_PEDESTAL + self.SPARSIFICATION_THRESHOLD*self.ADC_PEDESTAL_RMS):
            return 0
        return adc_count