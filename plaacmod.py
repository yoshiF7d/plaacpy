import argparse
from Colors import Colors
import AA
import HiddenMarkovModel
import numpy as np
np.seterr(divide = 'ignore') 
import re
from tabulate import tabulate
from DisorderReport import DisorderReport
from HighestScoringSubsequence import highestScoringSubsequence
import time

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import os
os.system('')

class Sequence():
	def __init__(self,name,seq,fasta,oneLine=False):
		self.name = name
		self.seq = seq
		if self.seq[-1] == '*':
			self.seq = self.seq[:-1]
		
		self.length = len(self.seq)
		self.indices = AA.stringToIndices(seq)
		if len(self.indices) < 1:
			print(name + ' has no seqence')
			return
		
		self.hmm = fasta.hmm
		self.hmmRef = fasta.hmmRef
				
		self.hmm.decodeAll(self.indices)
		self.viterbiPath = self.hmm.viterbiPath
		
		if not oneLine:
			self.hmmRef.decodeAll(self.indices)
			self.hmmScore = self.hmm.lmarginalProb - self.hmmRef.lmarginalProb
			self.hmmScoreV = self.hmm.lviterbiProb - self.hmmRef.lviterbiProb
		
			#MW#
			self.indicesMW = AA.table['mask'][self.indices]
			self.sizeMW = 80
			if len(self.indicesMW) < 80:
				self.sizeMW = len(self.indicesMW)
			self.scoreMW = highestScoringSubsequence(self.indicesMW,self.sizeMW,self.sizeMW)
			##

			#LLR#
			self.indicesLLR = llr[self.indices]
			self.sizeLLR = fasta.coreLength
			self.scoreLLR = highestScoringSubsequence(self.indicesLLR,self.sizeLLR,self.sizeLLR)
			##
			self.disorderReport = DisorderReport(self.indices,fasta.ww1,fasta.ww2,fasta.ww3,np.array([2.785,-1,-1.151]),fasta.llr,AA.table['lodpapa1'])
			self.longestPrd = longestOnes(self.viterbiPath)

		self.indicesCORE = fasta.llr[self.indices]
		bigNeg = -1e+6
		self.indicesCORE[self.viterbiPath==0] = bigNeg
		
		self.scoreCORE = highestScoringSubsequence(self.indicesCORE,fasta.coreLength,fasta.coreLength)
		
		self.coreStart = self.scoreCORE[0]
		self.coreStop = self.scoreCORE[1]
		
		self.aaStart = self.coreStart
		self.aaStop = self.coreStop
		
		if(self.scoreCORE[2] > bigNeg/2):
			while self.aaStart >= 0 and self.viterbiPath[self.aaStart] == 1:
				self.aaStart -= 1
			self.aaStart += 1
			while self.aaStop < len(self.viterbiPath) and self.viterbiPath[self.aaStop] == 1:
				self.aaStop += 1
			self.aaStop -= 1
			self.indicesPRD = self.indices[self.aaStart:self.aaStop]
			self.scorePRD = np.sum(llr[self.indicesPRD])
		else:
			self.scoreCORE[2] = np.nan
			self.aaStart = -1
			self.aaStop = -2
			self.coreStart = -1
			self.coreStop = -2
			self.PRD = []
			self.PRDScore = 0
	
	def print(self):
		print(
			tabulate(
				[
					[
						Colors.CYAN + self.name + Colors.RESET,
						Colors.YELLOW + 'Score' + Colors.RESET,
						Colors.YELLOW  + 'Start' + Colors.RESET,
						Colors.YELLOW  + 'End' + Colors.RESET,
						Colors.YELLOW  + 'Length' + Colors.RESET
						
					],
					[
						Colors.YELLOW + 'MW' + Colors.RESET,
						int(self.scoreMW[2]),
						int(self.scoreMW[0]+1),
						int(self.scoreMW[1]+1),
						int(self.scoreMW[1]-self.scoreMW[0]+1)
					],
					[
						Colors.YELLOW + 'LLR' + Colors.RESET,
						self.scoreLLR[2],
						int(self.scoreLLR[0]+1),
						int(self.scoreLLR[1]+1),
						int(self.scoreLLR[1]-self.scoreLLR[0]+1)
					],
					[
						Colors.YELLOW + 'CORE' + Colors.RESET,
						self.scoreCORE[2],
						int(self.coreStart+1),
						int(self.coreStop+1),
						int(self.coreStop - self.coreStart + 1)
					
					],
					[
						Colors.YELLOW + 'PRD' + Colors.RESET,
						self.scorePRD,
						int(self.aaStart+1),
						int(self.aaStop+1),
						int(self.aaStop - self.aaStart + 1)
					]
				],
				headers="firstrow",
				floatfmt=".3f"
			)
		)
		print()
		print(
			tabulate(
				[
					[Colors.YELLOW + 'NLLR' + Colors.RESET,self.scoreLLR[2] / (self.scoreLLR[1]-self.scoreLLR[0]+1)],
					[Colors.YELLOW + 'VIT maxrun' + Colors.RESET,self.longestPrd],
					[Colors.YELLOW + 'PRDT len' + Colors.RESET,self.length],
					[Colors.YELLOW + 'HMM all' + Colors.RESET,self.hmmScore],
					[Colors.YELLOW + 'HMM vit' + Colors.RESET,self.hmmScoreV],
					[Colors.YELLOW + 'FI num' + Colors.RESET,self.disorderReport.numDisorderedStrict2],
					[Colors.YELLOW + 'FI hydro' + Colors.RESET,self.disorderReport.meanHydro],
					[Colors.YELLOW + 'FI charge' + Colors.RESET,self.disorderReport.meanCharge],
					[Colors.YELLOW + 'FI combo' + Colors.RESET,self.disorderReport.meanFI],
					[Colors.YELLOW + 'FI max run' + Colors.RESET,self.disorderReport.maxLengthFI],
					[Colors.YELLOW + 'PAPA combo' + Colors.RESET,self.disorderReport.papaMaxScore],
					[Colors.YELLOW + 'PAPA prop' + Colors.RESET,self.disorderReport.papaMaxProb],
					[Colors.YELLOW + 'PAPA FI' + Colors.RESET,self.disorderReport.papaMaxDis],
					[Colors.YELLOW + 'PAPA LLR' + Colors.RESET,self.disorderReport.papaMaxLLR,],
					[Colors.YELLOW + 'PAPA LLR2' + Colors.RESET,self.disorderReport.papaMaxLLR2],
					[Colors.YELLOW + 'PAPA cen' + Colors.RESET,self.disorderReport.papaMaxCenter + 1]
				],
				floatfmt=".2f"
			)
		)
		print()

	def printSequence(self):
			ist = 0
			flag = False
			slist = []
			for i in range(self.length):
				if self.hmm.mapPath[i] > 0 :
					if not flag:
						flag = True
						slist.append(self.seq[ist:i] + Colors.RED)
						ist = i
				elif flag:
					flag = False
					slist.append(self.seq[ist:i] + Colors.RESET)
					ist = i
					
			slist.append(self.seq[ist:])
			
			print(''.join(slist))
			print(Colors.RESET)

	def printScoreCore(self):
		print(self.name + '\t' + f'{self.scoreCORE[2]:.3f}')

	def plot(self,ax,length):
		ax.imshow(AA.stringToColorIndices(self.seq)[np.newaxis,:],cmap='jet',aspect=20)
		ax.axes.get_xaxis().set_ticks([])
		ax.axes.get_yaxis().set_ticks([])
		ax.yaxis.set_label_coords(-0.05,0)
		ax.set_xlim([0,length])
		ax.set_ylim([-0.7,0.7])
		ax.set_ylabel(self.name,rotation=0)
		
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.set_aspect('auto')
	
		cmap = ListedColormap(['k','r'])
		norm = BoundaryNorm([0,1],cmap.N)
		
		x = np.arange(self.length)
		y = 0.65 * np.ones(self.length)
		
		points = np.array([x,y]).T.reshape(-1,1,2)
		segments = np.concatenate([points[:-1],points[1:]],axis=1)
		lc = LineCollection(segments,cmap=cmap,norm=norm)
		lc.set_array(self.hmm.mapPath)
		lc.set_linewidth(6)
		ax.add_collection(lc)
		
		y = -0.65 * np.ones(len(self.seq))
		
		points = np.array([x,y]).T.reshape(-1,1,2)
		segments = np.concatenate([points[:-1],points[1:]],axis=1)
		lc = LineCollection(segments,cmap=cmap,norm=norm)
		lc.set_array(self.hmm.viterbiPath)
		lc.set_linewidth(6)
		ax.add_collection(lc)

class Fasta():
	def __init__(self,inputFile,coreLength,ww1,ww2,ww3,fg,bg,flim,visualize):
		self.hmm = HiddenMarkovModel.prionHMM1(fg,bg)
		self.hmmRef = HiddenMarkovModel.prionHMM0(bg)
		self.llr = np.log(fg/bg)
		self.coreLength = coreLength
		self.ww1 = ww1
		self.ww2 = ww2
		self.ww3 = ww3

		self.fasta = readFasta(inputFile)

		if flim is not None:
			self.fasta = self.fasta[flim[0]:flim[1]]

		if visualize:
			self.fastaLen = len(self.fasta)		
			self.seqLenMax = np.max([len(fs[1]) for fs in self.fasta])

			self.fig, self.ax = plt.subplots(nrows=self.fastaLen,sharex=True,figsize=(12,3))
			if len(self.fasta) == 1:
				self.ax = [self.ax]
			self.scalarMap = plt.cm.ScalarMappable(
				cmap = plt.get_cmap('jet',len(AA.ctable)),
				norm = matplotlib.colors.BoundaryNorm(np.arange(len(AA.ctable)+1)-0.5,len(AA.ctable))
			)
			self.scalarMap.set_array([])
			self.colorBar = self.fig.colorbar(self.scalarMap,ax=self.ax,
				ticks=np.arange(len(AA.ctable)),
				location='top',
				shrink=0.8,
				orientation='horizontal'
			)
			self.colorBar.set_ticklabels(AA.ctable)
		
	def print(self,plotDir,visualize,oneLine):
		if plotDir is not None:
			os.makedirs(plotDir,exist_ok=True)
		
		for i,fs in enumerate(self.fasta):
			seq = Sequence(*fs,self,oneLine)
			if oneLine:
				seq.printScoreCore()
			else:
				seq.print()
				seq.printSequence()
			if visualize:
				seq.plot(self.ax[i],self.seqLenMax)
			if plotDir is not None:
				figp,axp = plt.subplots(nrows=3,sharex=True,figsize=(12,6),gridspec_kw={'height_ratios': [1,0.2,1]})
				#axp[0].set_ylim([0,0.01])
				for i in range(seq.hmm.ns):
					axp[0].plot(normalize2(seq.hmm.postProb[i]),label=seq.hmm.names[i])
				
				axp[0].set_xlim([0,seq.length])
				axp[0].set_ylim([-0.01,1.01])
				axp[0].set_yticks([0,1])
				axp[0].set_yticklabels(seq.hmm.states)
				axp[0].legend(loc=(1.04,0.5))
				
				seq.plot(axp[1],seq.length)
				
				axp[2].plot(seq.disorderReport.fi,label='FoldIndex')
				axp[2].fill_between(np.arange(seq.length),seq.disorderReport.fi,alpha=0.7)
				axp[2].plot(-seq.disorderReport.plaacLLR,label='-PLAAC')
				axp[2].plot(- 4 * seq.disorderReport.papa,label='-4*PAPA')
				
				axp[2].set_xlim([0,seq.length])
				axp[2].set_ylim([-1,1])
				axp[2].legend(loc=(1.04,0.5))
				
				path = os.path.join(plotDir,seq.name + '.png')
				plt.subplots_adjust(right=0.85)
				figp.savefig(path)
				plt.close(figp)
		if visualize:
			plt.show()

class Plaac():
	def __init__(self):
		fgFreq = AA.table['prdFreqScer28']
		bgFreq = bgScer = normalize(AA.table['bgFreqScer'])

		fgFreq[0] = 0
		fgFreq[21] = 0
		fgFreq = normalize(fgFreq)

		bgFreq[0] = 0
		bgFreq[21] = 0
		bgFreq = normalize(bgFreq)

		#bgCombo = normalize(args.alpha * bgScer + (1-args.alpha) * bgFreq)
		bgCombo = bgFreq

		epsx = 0.00001
		fgFreq[0] = epsx
		fgFreq[21] = epsx
		bgCombo[0] = epsx
		bgCombo[21] = epsx

		fg = normalize(fgFreq)
		bg = normalize(bgCombo)

		coreLength = 60
		ww1 = 41
		ww2 = 41
		ww3 = 41

		self.hmm = HiddenMarkovModel.prionHMM1(fg,bg)
		self.hmmRef = HiddenMarkovModel.prionHMM0(bg)
		self.llr = np.log(fg/bg)
		self.coreLength = coreLength
		self.ww1 = ww1
		self.ww2 = ww2
		self.ww3 = ww3

	def score(self,seq):
		indices = AA.stringToIndices(seq)
		self.hmm.decodeAll(indices)
		indicesCORE = self.llr[indices]
		bigNeg = -1e+6
		indicesCORE[self.hmm.viterbiPath==0] = bigNeg
		scoreCORE = highestScoringSubsequence(indicesCORE,self.coreLength,self.coreLength)
	
		if scoreCORE[2] <= bigNeg/2:
			scoreCORE[2] = np.nan
		
		return scoreCORE[2]

def normalize(array):
	sum = array.sum()
	if sum == 0:
		sum = 1
	return array/sum

def normalize2(array):
	max = array.max()
	if max == 0:
		max = 1
	return array/max

def readFasta(file):
	with open(file,'r') as f:
		slist = re.split(r'(>.*\n)',f.read())
	
	slist = [s for s in slist if s]
	slist = [slist[i:i+2] for i in range(0,len(slist),2)]
	
	for i in range(len(slist)):
		slist[i][0] = slist[i][0][1:].split()[0]
		slist[i][1] = re.sub('\n','',slist[i][1])
	
	return slist

def readAAParams(file):
	with open(file,'r') as f:
		slist = f.read().splitlines()
	tlist = []
	for s in slist:
		tlist.append(s.split())
	letters=AA.table['letter']
	tlist = np.array(tlist)
	for i,t in enumerate(tlist):
		if t[2] != letters[i]:
			print('# warning: ' + file + 'does not have expected name in line ' + str(i+1))
	return np.asarray(tlist[:,0],dtype=float)

def computeAAFreq(file):
	slist = readFasta(file)
	res = np.zeros(AA.length)
	for s in slist:
		aa = AA.stringToIndices(s[1])
		if AA.isValidProtein(aa):
			res += np.bincount(aa,minlength=AA.length)
	
	return res

def longestOnes(seq):
	maxl = 0
	i = 0
	while i < len(seq):
		if seq[i] > 0:
			si = i
			while i < len(seq) and seq[i] > 0:
				i += 1
			if i - si >= maxl:
				maxl = i - si
		else:
			i += 1
	return maxl

def parselim(limstr):
	l0,l1 = limstr.split(",")
	l0 = l0.lstrip('([').lstrip()
	l1 = l1.rstrip(')]').rstrip()
	return [int(l0),int(l1)]