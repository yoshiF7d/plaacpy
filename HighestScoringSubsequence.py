import numpy as np
def highestScoringSubsequence(seq,min=None,max=None):
	if min is None:
		min = 1
	if max is None:
		max = len(seq)
	
	rfseq = np.fft.rfft(seq)
	n = len(seq)
	
	bestStart = 0
	bestStop = min -1
	bestScore = np.sum(seq[:min])
	
	if n%2==0:
		rn = n//2
		rfrec = np.ndarray(rn+1,dtype='c16')
	else:
		rn = (n+1)//2
		rfrec = np.ndarray(rn,dtype='c16')
		
	for l in range(min,max+1):
		ff = lambda k: np.exp(-1.j*np.pi*(l-1)*k/n) * np.sin(np.pi*l*k/n) / np.sin(np.pi*k/n) 

		rfrec[0] = l
		rfrec[1:rn] = ff(np.arange(1,rn))
		if n%2==0:
			rfrec[rn] = l%2
		
		conv = np.round(np.fft.irfft(rfseq*rfrec,n),15)
		if conv[l-1:].size > 0:
			ind = np.argmax(conv[l-1:]) + l - 1
		else:
			ind = 0
		if conv[ind] >= bestScore:
			bestStart = ind - l + 1
			bestStop = ind
			bestScore = conv[ind]
				
	return [bestStart,bestStop,bestScore]
	