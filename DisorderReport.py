import numpy as np
np.seterr(divide = 'ignore')
import AA
from HighestScoringSubsequence import highestScoringSubsequence
from Colors import Colors

class DisorderReport():
	def __init__(self,aa,ww1,ww2,ww3,cc,plaacWeights,papaWeights):
		self.ww1 = ww1
		self.ww2 = ww2
		self.ww3 = ww3
		self.cc = cc
		self.aa = aa
		self.placcWeights = plaacWeights
		self.papaWeights = papaWeights
		self.length = len(aa)
		
		self.minLength = 80
		self.maxLength = 240
		self.minLength2 = 40
		
		maa1 = AA.table['hydro2'][aa]
		self.meanHydro = np.mean(maa1)
		self.hydro = movingAverage(maa1,ww1)
		
		maa2 = AA.table['charge'][aa]
		self.meanCharge = np.mean(maa2)
		self.charge = movingAverage(maa2,ww1)
		
		self.meanFI = self.cc[2] + self.cc[1]*np.abs(self.meanCharge) +self.cc[0]*self.meanHydro
		self.fi = self.cc[0]*self.hydro + self.cc[1]*np.abs(self.charge) + self.cc[2]
		self.maa3 = self.placcWeights[aa]
		self.plaacLLR = movingAverage(self.maa3,ww3)
		self.maa4 = self.papaWeights[aa]
		self.papa = movingAverage(self.maa4,ww2,mergeMe=13,seq=aa)
		self.papaX2 = movingAverage(self.papa,ww2,weight=True)
		self.plaacLLRX2 = movingAverage(self.plaacLLR,ww3,weight=True)
		self.fix2 = movingAverage(self.fi,ww1,weight=True)
		
		self.numDisordered = np.sum(self.fi[self.fi<0])
		
		halfw = (ww1-1)//2
		if halfw > self.length//2:
			halfw = self.length//2
			
		self.numDisorderedStrict = self.fi[self.fi<0][halfw+1:-(halfw+1)]
			
		if self.fi[halfw] < 0:
			self.numDisorderedStrict += halfw + 1
		if self.fi[self.length-halfw-1] < 0:
			self.numDisorderedStrict += halfw + 1
		
		temp = self.papaX2[(ww2-1)//2:-(ww2-1)//2]

		if temp.size > 0:
			self.papaMaxCenter = np.argmax(temp[self.fix2[(ww2-1)//2:-(ww2-1)//2] < 0])
		else:
			self.papaMaxCenter = -1

		print(Colors.YELLOW + str(temp) + Colors.RESET)
		#print(Colors.CYAN + str(self.papa) + Colors.RESET)
		#print(Colors.MAGENTA + str(self.aa) + Colors.RESET)
		#print(Colors.RED + str(self.papaMaxCenter) + Colors.RESET)

		self.papaMaxScore = self.papaX2[self.papaMaxCenter]		
		self.papaMaxProb = self.papaX2[self.papaMaxCenter]
		self.papaMaxDis = self.fix2[self.papaMaxCenter]
		self.papaMaxLLR2 = self.plaacLLRX2[self.papaMaxCenter]
		self.papaMaxLLR = self.plaacLLR[self.papaMaxCenter]
		
		self.hssr = highestScoringSubsequence(self.maa3)
		
		if(
			self.hssr[1] - self.hssr[0] + 1 < self.minLength or
			self.hssr[1] - self.hssr[0] + 1 > self.maxLength
		):
			self.hssr2 = highestScoringSubsequence(self.maa3,self.minLength,self.maxLength)
		else:
			self.hssr2 = self.hssr
				
		n = self.length//self.minLength
		
		self.startAA = np.zeros(n)
		self.stopAA = np.zeros(n)
		self.lengthAA = np.zeros(n)
		self.localMean = np.zeros(n)
		self.localSD = np.zeros(n)
		self.localHydro = np.zeros(n)
		self.localCharge = np.zeros(n)

		self.numSegment = 0
		self.numDisorderedStrict2 = 0
		if n > 0:
			self.localHydro[0] = 0.5

		i = 0 + halfw
		
		while i < self.length - halfw:
			if self.fi[i] < 0:
				sc = self.fi[i]
				lc = np.abs(self.charge[i])
				lh = self.hydro[i]
				scsc = self.fi[i]*self.fi[i]
				startIndex = i
				i += 1
				while i < self.length - halfw and self.fi[i] < 0:
					sc += self.fi[i]
					lh += self.hydro[i]
					lc += np.abs(self.charge[i])
					scsc += self.fi[i]*self.fi[i]
					i += 1
				stopIndex = i - 1
				segLength = stopIndex - startIndex + 1
				msc = sc/segLength
				sdsc = np.sqrt(scsc/segLength - msc*msc)
				if startIndex == halfw:
					startIndex = 0
				if stopIndex == self.length - halfw - 1:
					stopIndex = self.length - 1
				segLength = stopIndex - startIndex + 1
				
				if segLength >= self.minLength:
					self.startAA[self.numSegment] = startIndex
					self.stopAA[self.numSegment] = stopIndex
					self.lengthAA[self.numSegment] = segLength
					self.localMean[self.numSegment] = msc
					self.localSD[self.numSegment] = sdsc
					self.localHydro[self.numSegment] = lh/segLength
					self.localCharge[self.numSegment] = lc/segLength
					self.numSegment += 1
					self.numDisorderedStrict2 += segLength
			else:
				i+=1
		if len(self.lengthAA) > 0:
			self.maxLength = int(np.max(self.lengthAA))

		temp = self.localMean[self.lengthAA >= self.minLength2]
		if temp.size > 0:
			self.bestIndex = np.argmax(self.localMean[self.lengthAA >= self.minLength2])
			self.maxLong = self.localMean[self.bestIndex]
		else:
			self.bestIndex = 0
			self.maxLong = 0
	
	def print(id=''):
		print('### protein length:    ' + str(self.length))
		print('### num disordered(1): ' + str(self.numDisordered))
		print('### num disordered(2): ' + str(self.numDisorderedStrict))
		print('### num disordered(3): ' + str(self.numDisorderedStrict2))
		for i in range(self.numSeg):
			print(
				'### ' + str(self.startAA[i]) + 
				'-' + str(self.stopAA[i]) + 
				': ' + str(self.localMean[i]) + 
				' +- ' + str(self.localSD[i])
			)
		print(
			'### ' + str(self.maxLength) + 
			' ' + str(self.localMean[self.bestIndex]) + 
			' ' + str(self.lengthAA[self.bestIndex])
		)
		print(
			'# ' + str(self.length) + 
			'\t' + str(self.meanCharge) + 
			'\t' + str(self.meanHydro) +
			'\t' + str(self.meanFI) +
			'\t' + str(self.maxLong)
			)
		print(
			'# ' + str(self.hssr[0]) + 
			'\t' + str(self.hssr[1]) +
			'\t' + str(self.hssr[2]) +
			'\t' + str(self.hssr2[0]) +
			'\t' + str(self.hssr2[1]) +
			'\t' + str(self.hssr2[2])
		)
		for i in range(self.length):
			print(
				id + '\t' +
				str(self.aa[i]) + '\t' +
				str(self.charge[i]) + ' \t' +
				str(self.hydro[i]) + ' \t' +
				str(self.fi[i]) + '\t' +
				str(self.plaacLLR[i]) + ' \t' +
				str(self.papa[i]) + '\t # [' +
				str(i+1-self.ww1/2) + '-' +
				str(i+1+self.ww1/2) + ']'
			)
	
def movingAverage(arr,ww,*,weight=False,mergeMe=None,seq=None):
	w = ww//2
	pad = 2*w
	n0 = len(arr)
	n = n0 + pad
	l = pad + 1
	ff = lambda k: np.exp(-1.j*np.pi*(l-1)*k/n) * np.sin(np.pi*l*k/n) / np.sin(np.pi*k/n)
	
	if weight:
		mask = l*np.ones(n0)
		mask[:w] = 1 + w + np.arange(w)
		mask[-w:] = 1 + w + np.arange(w)[::-1]
	
	if mergeMe is not None:
		mask = np.ones(n0)
		for i in range(n0):
			if(
				(seq[i] == mergeMe and i >= 1 and seq[i-1] == mergeMe) or
				(seq[i] == mergeMe and i >= 2 and seq[i-2] == mergeMe)
			):
				mask[i] = 0
	
	if (mergeMe is not None) or weight:
		rfa = np.fft.rfft(np.concatenate((mask*arr,np.zeros(pad))))
	else:
		rfa = np.fft.rfft(np.concatenate((arr,np.zeros(pad))))
	
	if n%2==0:
		rn = n//2
		rfr = np.concatenate(([l],ff(np.arange(1,rn)),[l%2]))
	else:
		rn = (n+1)//2
		rfr = np.concatenate(([l],ff(np.arange(1,rn))))
	
	conv = np.fft.irfft(rfa*rfr,n)
	denom = l*np.ones(n)
	denom[:l] = 1 + np.arange(l)
	denom[-l:] = 1 + np.arange(l)[::-1]
	
	if weight:
		return (conv/denom)[w:-w]/mask
	else:
		return (conv/denom)[w:-w]
		