from numpy import exp, sqrt, log

def raw_to_temperature(
		raw,
		E=1,
		OD=1,
		RTemp=20,
		ATemp=20,
		IRWTemp=20,
		IRT=1,
		RH=50,
		PR1=21106.77,
		PB=1501,
		PF=1,
		PO=-7340,
		PR2=0.012545258,
		ATA1=0.006569, 
		ATA2=0.01262, 
		ATB1=-0.002276, 
		ATB2=-0.00667, 
		ATX=1.9):

	"""
	convert flir raw data to temperature

	references:
	https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
	"""

  
  # E: Emissivity - default 1, should be ~0.95 to 0.97 depending on source
  # OD: Object distance in metres
  # RTemp: apparent reflected temperature - one value from FLIR file (oC), default 20C
  # ATemp: atmospheric temperature for tranmission loss - one value from FLIR file (oC) - default = RTemp
  # IRWinT: Infrared Window Temperature - default = RTemp (oC)
  # IRT: Infrared Window transmission - default 1.  likely ~0.95-0.96. Should be empirically determined.
  # RH: Relative humidity - default 50%
  
  # Note: PR1, PR2, PB, PF, and PO are specific to each camera and result from the calibration at factory
  # of the camera's Raw data signal recording from a blackbody radiation source
  # Calibration Constants                 (A FLIR SC660, A FLIR T300(25o), T300(telephoto), A Mikron 7515)
  # PR1: PlanckR1 calibration constant from FLIR file  21106.77       14364.633     14906.216       21106.77
  # PB: PlanckB calibration constant from FLIR file    1501           1385.4        1396.5          9758.743281
  # PF: PlanckF calibration constant from FLIR file    1              1             1               29.37648768
  # PO: PlanckO calibration constant from FLIR file    -7340          -5753         -7261           1278.907078
  # PR2: PlanckR2 calibration constant form FLIR file  0.012545258    0.010603162   0.010956882     0.0376637583528285
 
  # Set constants below. Comment out those that are variables in this function
  # Keep those that should remain constants.  
  # These are here to make troubleshooting calculations easier if not running this as a function call
  # raw = 19746; E = 0.95; OD = 20; IRT = 0.96
  # RTemp = 20; IRWTemp = RTemp; ATemp = 20; RH = 50
  # PR1 = 21106.77; PB = 1501; PF = 1; PO = -7340; PR2 = 0.012545258
  
  # These humidity parameters were extracted from the FLIR SC660 camera
  # ATA1: Atmospheric Trans Alpha 1  0.006569 constant for calculating humidity effects on transmission 
  # ATA2: Atmospheric Trans Alpha 2  0.012620 constant for calculating humidity effects on transmission
  # ATB1: Atmospheric Trans Beta 1  -0.002276 constant for calculating humidity effects on transmission
  # ATB2: Atmospheric Trans Beta 2  -0.006670 constant for calculating humidity effects on transmission
  # ATX:  Atmospheric Trans X        1.900000 constant for calculating humidity effects on transmission
 
  # Equations to convert to temperature
  # See http://130.15.24.88/exiftool/forum/index.php/topic,4898.60.html
  # Standard equation: temperature = PB/log(PR1/(PR2*(raw+PO))+PF)-273.15
  # Other source of information: Minkina and Dudzik's Infrared Thermography: Errors and Uncertainties

	emiss_wind = 1-IRT
	refl_wind = 0 # anti-reflective coating on window
	h2o = (RH/100)*exp(1.5587+0.06939*(ATemp)-0.00027816*(ATemp) ** 2+0.00000068455*(ATemp) ** 3)
	# converts relative humidity into water vapour pressure (I think in units mmHg)
	tau1 = ATX*exp(-sqrt(OD/2)*(ATA1+ATB1*sqrt(h2o)))+(1-ATX)*exp(-sqrt(OD/2)*(ATA2+ATB2*sqrt(h2o)))
	tau2 = ATX*exp(-sqrt(OD/2)*(ATA1+ATB1*sqrt(h2o)))+(1-ATX)*exp(-sqrt(OD/2)*(ATA2+ATB2*sqrt(h2o)))
	# transmission through atmosphere - equations from Minkina and Dudzik's Infrared Thermography Book
	# Note: for this script, we assume the thermal window is at the mid-point (OD/2) between the source
	# and the camera sensor

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

	raw_obj = (raw/E/tau1/IRT/tau2-raw_atm1_attn-raw_atm2_attn-raw_wind_attn-raw_refl1_attn-raw_refl2_attn)

	temp_C = PB/log(PR1/(PR2*(raw_obj+PO))+PF)-273.15

	return temp_C
