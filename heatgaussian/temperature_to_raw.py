from numpy import exp, sqrt

def temperature_to_raw(
	temp,
	E = 1,
	OD = 1,
	RTemp = 20,
	ATemp = 20,
	IRWTemp = 20,
	IRT = 1,
	RH = 50,
	PR1 = 21106.77,
	PB = 1501,
	PF = 1,
	PO = -7340,
	PR2 = 0.012545258,
	ATA1 = 0.006569,
	ATA2 = 0.01262,
	ATB1 = -0.002276,
	ATB2 = -0.00667,
	ATX = 1.9,
):
	"""
	Ref:
	https://github.com/gtatters/Thermimage/blob/master/R/temp2raw.R

	temp: temperature in celsius
	E: emissivity
	OD: distance in meters
	RTemp: Refl. temperature in celsius
	ATemp: Atomspheric temperature in celsius
	IRWTemp: Ext. optics temperature in celsius
	IRT: Ext. optics transmission
	RH: relative humidity in percentage (between 0 and 100)

	// below is camera specific calibration parameters
	PR1: Planck R1
	PB: Planck B
	PF: Planck F
	PO: Planck O
	PR2: Planck R2
	ATA1: 
	ATA2:
	ATB1:
	ATB2:
	ATX:
	"""
	emiss_wind = 1-IRT
	refl_wind = 0 # anti-reflective coating on window
	h2o = (RH / 100) * exp(1.5587 + 0.06939 * (ATemp)-0.00027816*(ATemp) ** 2+0.00000068455*(ATemp) ** 3)
	# converts relative humidity into water vapour pressure (I think in units mmHg)
	tau1 = ATX*exp(-sqrt(OD/2)*(ATA1+ATB1*sqrt(h2o)))+(1-ATX)*exp(-sqrt(OD/2)*(ATA2+ATB2*sqrt(h2o)))
	tau2 = ATX*exp(-sqrt(OD/2)*(ATA1+ATB1*sqrt(h2o)))+(1-ATX)*exp(-sqrt(OD/2)*(ATA2+ATB2*sqrt(h2o)))
	# transmission through atmosphere - equations from Minkina and Dudzik's Infrared Thermography Book

	raw_obj = PR1/(PR2*(exp(PB/(temp+273.15))-PF))-PO

	raw_refl1 = PR1/(PR2*(exp(PB/(RTemp+273.15))-PF))-PO   # radiance reflecting off the object before the window
	raw_refl1_attn = (1-E)/E*raw_refl1   # attn = the attenuated radiance (in raw units) 

	raw_atm1 = PR1/(PR2*(exp(PB/(ATemp+273.15))-PF))-PO # radiance from the atmosphere (before the window)
	raw_atm1_attn = (1-tau1)/E/tau1*raw_atm1 # attn = the attenuated radiance (in raw units) 

	raw_wind = PR1/(PR2*(exp(PB/(IRWTemp+273.15))-PF))-PO
	raw_wind_attn = emiss_wind/E/tau1/IRT*raw_wind

	raw_refl2 = PR1/(PR2*(exp(PB/(RTemp+273.15))-PF))-PO   
	raw_refl2_attn = refl_wind/E/tau1/IRT*raw_refl2

	raw_atm2 = PR1/(PR2*(exp(PB/(ATemp+273.15))-PF))-PO
	raw_atm2_attn = (1-tau2)/E/tau1/IRT/tau2*raw_atm2

	raw = (
		 raw_obj
		+raw_atm1_attn
		+raw_atm2_attn
		+raw_wind_attn
		+raw_refl1_attn
		+raw_refl2_attn) * E * tau1 * IRT * tau2

	return raw

