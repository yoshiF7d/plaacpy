import numpy as np
np.seterr(divide = 'ignore') 

table={}
table['letter'] = ['X','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','*']
table['name'] = ['???','Ala','Cys','Asp','Glu','Phe','Gly','His','Ile','Lys','Leu','Met','Asn','Pro','Gln','Arg','Ser','Thr','Val','Trp','Tyr','***']
table['charge'] = [0,0,0,1,1,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,0]
table['hydro'] = [0.0,1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3,0.0]
table['fgpapa'] = [0.0,0.042,0.033,0.014,0.009,0.075,0.038,0.059,0.102,0.009,0.059,0.038,0.096,0.038,0.024,0.054,0.125,0.069,0.102,0.024,0.054,0.0]
table['bgpapa'] = [0.0,0.072,0.022,0.051,0.017,0.032,0.040,0.078,0.045,0.045,0.061,0.020,0.089,0.127,0.022,0.081,0.109,0.078,0.045,0.012,0.025,0.0]
table['fgpapa2'] = [0.0,0.057,0.015,0.045,0.023,0.064,0.057,0.080,0.038,0.004,0.045,0.030,0.068,0.072,0.030,0.045,0.110,0.087,0.038,0.042,0.049,0.0]
table['bgpapa2'] = [0.0,0.064,0.012,0.067,0.024,0.021,0.046,0.070,0.030,0.021,0.052,0.021,0.040,0.095,0.037,0.076,0.119,0.095,0.037,0.009,0.064,0.0]
table['odpapa1'] = [0.0,0.67267686,1.5146198,0.27887323,0.5460614,2.313433,0.96153843,0.75686276,2.2562358,0.20664589,0.9607843,1.9615384,1.0836071,0.30196398,1.0716166,0.6664044,1.1432927,0.8917492,2.2562358,1.9478673,2.1785367,0.0]
table['odpapa2'] = [0.0,0.88066554,1.2461538,0.72039115,0.77220076,3.6936572,1.2570281,1.1460011,1.2519685,0.17436177,0.87114847,1.4330357,1.7729831,0.7429888,0.8229167,0.5531136,0.9144572,0.9143354,1.0367454,4.710145,0.75716186,0.0]
table['bgFreqScer'] = [0,0.0550,0.0126,0.0586,0.0655,0.0441,0.0498,0.0217,0.0655,0.0735,0.0950,0.0207,0.0615,0.0438,0.0396,0.0444,0.0899,0.0592,0.0556,0.0104,0.0337,0]
table['prdFreqScer04'] = [0,0.0488,0.0032,0.0202,0.0234,0.0276,0.1157,0.0149,0.0191,0.0329,0.0456,0.0149,0.1444,0.0308,0.2208,0.0202,0.1008,0.0297,0.0234,0.0064,0.0573,0]
table['prdFreqScer28'] = [0,0.04865,0.00219,0.01638,0.00783,0.02537,0.07603,0.0181,0.02018,0.01641,0.02639,0.02975,0.25885,0.05126,0.15178,0.025,0.10988,0.03841,0.01972,0.00157,0.05624,0]
table['mask'] = [0,0,0,0,0,0,0,0,0,0,0,0,1.0,0,1.0,0,0,0,0,0,0,0]

for k,v in table.items():
	table[k] = np.array(v)

table['hydro2'] = (1.0/9.0)*table['hydro'] + 0.5
table['lodpapa1'] = np.log(table['odpapa1'])
table['lodpapa1'][0] = table['lodpapa1'][-1] = 0
table['lodpapa2'] = np.log(table['odpapa2'])
table['lodpapa2'][0] = table['lodpapa2'][-1] = 0

length = len(table['letter'])
index = dict(zip(table['letter'],range(length)))

def stringToIndices(string):
	return np.array(list(map(lambda c:index.get(c,0),string.upper())))

def indicesToString(indices):
	return ''.join(table['letter'][indices])
	
def isValidProtein(aa):
	if np.where((aa[1:-1] == 0) | (aa[1:-1] == 21))[0].size > 1:
		return False
	if aa[-1] == 0:
		return False
	return True
	
ctable = np.array(['N','Q','Y','G','M','S','P','A','H','T','F','R','V','I','D','L','K','C','W','E'])
cdict = dict(zip(ctable,range(len(ctable))))

def aaToColorIndex(aa):
	return np.array(list(map(lambda c:cdict[c],table['letter'][aa])))

def stringToColorIndices(string):
	arr = np.array(list(map(cdict.get,string.upper())))
	#print(len(arr))
	return arr