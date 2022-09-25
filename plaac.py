import argparse
from Colors import Colors
import AA
import HiddenMarkovModel
import numpy as np
np.seterr(divide = 'ignore') 
import re
from tabulate import tabulate
from DisorderReport import DisorderReport
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})
from HighestScoringSubsequence import highestScoringSubsequence
import time

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import os
os.system('')

class Sequence():
	def __init__(self,name,seq,fasta):
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
		
		self.indicesMW = AA.table['mask'][self.indices]
		self.sizeMW = 80
		if len(self.indicesMW) < 80:
			self.sizeMW = len(self.indicesMW)
		self.scoreMW = highestScoringSubsequence(self.indicesMW,self.sizeMW,self.sizeMW)
		
		self.indicesLLR = llr[self.indices]
		self.sizeLLR = fasta.coreLength
		self.scoreLLR = highestScoringSubsequence(self.indicesLLR,self.sizeLLR,self.sizeLLR)
		
		self.hmm.decodeAll(self.indices)
		self.hmmRef.decodeAll(self.indices)
		
		self.hmmScore = self.hmm.lmarginalProb - self.hmmRef.lmarginalProb
		self.hmmScoreV = self.hmm.lviterbiProb - self.hmmRef.lviterbiProb
		
		self.disorderReport = DisorderReport(self.indices,fasta.ww1,fasta.ww2,fasta.ww3,np.array([2.785,-1,-1.151]),fasta.llr,AA.table['lodpapa1'])
		self.viterbiPath = self.hmm.viterbiPath
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
			self.indicesPRD = []
			self.scorePRD = 0
	
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
						self.scoreMW[0]+1,
						self.scoreMW[1]+1,
						self.scoreMW[1]-self.scoreMW[0]+1
					],
					[
						Colors.YELLOW + 'LLR' + Colors.RESET,
						int(self.scoreLLR[2]),
						self.scoreLLR[0]+1,
						self.scoreLLR[1]+1,
						self.scoreLLR[1]-self.scoreLLR[0]+1
					],
					[
						Colors.YELLOW + 'CORE' + Colors.RESET,
						self.scoreCORE[2],
						self.coreStart+1,
						self.coreStop+1,
						self.coreStop - self.coreStart + 1
					
					],
					[
						Colors.YELLOW + 'PRD' + Colors.RESET,
						int(self.scorePRD),
						self.aaStart+1,
						self.aaStop+1,
						self.aaStop - self.aaStart + 1
					]
				],
				headers="firstrow",
				floatfmt=".2f"
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
	def __init__(self,inputFile,coreLength,ww1,ww2,ww3,fg,bg,flim):
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
		self.fastaLen = len(self.fasta)
		self.seqLenMax = np.max([len(fs[1]) for fs in self.fasta])
		
		self.fig, self.ax = plt.subplots(nrows=self.fastaLen,sharex=True,figsize=(12,3))
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
		
	def print(self,plotDir):
		if plotDir is not None:
			os.makedirs(plotDir,exist_ok=True)
		
		for i,fs in enumerate(self.fasta):
			seq = Sequence(*fs,self)
			seq.print()
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
		plt.show()
	
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
		slist[i][0] = slist[i][0][1:].strip()
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


parser = argparse.ArgumentParser(description='plaac')
parser.add_argument('-i','--inputFile')
parser.add_argument('-b','--bgFile',
	help='-b background.fa, where background.fa is the name of a protein fasta file used to\n'\
	'  compute background AA frequencies for the species.\n'\
	'  This option is ignored if -B is used, but otherwise if -b is not specified it defaults to the input.fa file.\n'
	'  If the sequences in input.fa have biased AA composition then a separate background.fa or bg_freqs.txt is recommended.\n'
	'  If -b is specified but -i is not, AA counts for background.fa will be written to standard output, and the program will exit.\n'
	'  These counts can be redirected to a file (e.g. with > bg_freqs.txt), in a format that can be read by the -B option.'
)
parser.add_argument('-B','--bgFreqFile',
	help='-B bg_freqs.txt specifying background AA freqs to use for the species, one per line, in the following order:\n'\
	'  X, A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y, *\n'\
	'(Values for X and * will be set to zero and other numbers normalized to add to 1)'
)
parser.add_argument('-F','--fgFreqFile',
	help='-F fg_freqs.txt, specifying prion-like AA freqs in same format as -B above. Defaults to freqs from 28 S. cerevisiae domains.'
)
parser.add_argument('-c','--coreLength',type=int,default=60,
	help='-c coreLength, where the integer coreLength is the minimal contiguous prion-like domain length\n  for the HMM parses. Default is 60.'
)
parser.add_argument('-w','--ww1',type=int,default=41,
	help='-w window_size, the window size for FoldIndex disorder predictions. Default is 41.'
)
parser.add_argument('-W','--ww2',type=int,default=41,
	help='-W Window_size, the window size for the PAPA algorithm. Default is 41.'
)
parser.add_argument('-a','--alpha',type=float,default=1,
	help='-a alpha, where alpha is a number between 0 and 1 (inclusive) that controls the degree to which the S. cerevisiae\n'\
	'  background AA frequencies are mixed with the background AA frequencies from -B, -b, or -i.\n'\
	'  If alpha = 0, just the AA frequencies from the -B, -b, or -i are used, and if alpha = 1 just the\n  S. cerevisiae AA frequencies are used. Default is 1.0.'
)
parser.add_argument('-m','--hmmType',type=int)
#parser.add_argument('-p','--plotList',
#	help='-p print_list.txt, where print_list.txt has the name of one fasta on each line, and specifies'\
#	'\n  which fastas in input.fa will be plotted\n'\
#	'  The names must exactly match those in input.fa, but do need need the > symbol before the name.\n'\
#	'  If no print_list.txt is specified the output from the program will be a table of summaries for each protein (one per line) in input.fa;\n'\
#	'  If a print_list.txt is specified the output from the program will be a table (one line per residue) that is used\n'\
#	'  for making plots for each of the proteins listed in print_list.txt.\n'\
#	'  If the option is given as -p all, then plots will be made for all of the proteins in input.fa, \n  which is not advised if input.fa is an entire proteome.\n'\
#	'  To make the plots from output that has been redirected to output.txt, at the command-line type type\n  Rscript plaac_plot.r output.txt plotname.pdf.'\
#	'  This requires that the program R be installed (see http://www.r-project.org/)\n  and will create a file named plotname.pdf, with one plot per page.'\
#	'  Calling Rscript plaac_plot.r with no file specified will list other options for plotting.'
#)
parser.add_argument('-p','--plotDir',help='-p plotDir')
parser.add_argument('-H','--hmmDotFile',
	help='-H hmm_filename.txt, writes parameters of HMM to hmm_filenmae.txt in dot format, which can be made into a figure with GraphViz.'
)
parser.add_argument('-d','--printHeaders',action='store_true',
	help='-d, print documentation for headers. If flag is not set, headers will not be printed.'
)

parser.add_argument('-f','--flim',help='-f [fmin,fmax]')
#parser.add_argument('-s','--printParameters',action='store_false',
#	help='-s, skip printing of run-time parameters at top of file. If flag is not set, run-time parameters will be printed.'
#)

#parser.add_argument('-s','--printParameters',action='store_false',
#	help='-s, skip printing of run-time parameters at top of file. If flag is not set, run-time parameters will be printed.'
#)


args = parser.parse_args()

if args.flim is not None:
	args.flim = parselim(args.flim)
#readFasta(args.inputFile)
#test(args.inputFile)
#print(readAAParams(args.inputFile))
#exit()
#print(AA.AminoAcid.header())
#for aa in AA.AA:
#	print(aa)

if args.bgFreqFile is not None:
	bgFreq = readAAParams(args.bgFreqFile)
elif args.bgFile is not None:
	bgFreq = computeAAFreq(args.bgFile)
elif args.inputFile is not None:
	bgFreq = computeAAFreq(args.inputFile)

if args.fgFreqFile is not None:
	fgFreq = readAAParams(args.fgFreqFile)

fgFreq = AA.table['prdFreqScer28']
bgScer = normalize(AA.table['bgFreqScer'])

fgFreq[0] = 0
fgFreq[21] = 0
fgFreq = normalize(fgFreq)

bgFreq[0] = 0
bgFreq[21] = 0
bgFreq = normalize(bgFreq)

bgCombo = normalize(args.alpha * bgScer + (1-args.alpha) * bgFreq)

epsx = 0.00001
fgFreq[0] = epsx
fgFreq[21] = epsx
bgCombo[0] = epsx
bgCombo[21] = epsx

fg = normalize(fgFreq)
bg = normalize(bgCombo)
llr = np.log(fg/bg)

#print(Colors.YELLOW + str(fg) + Colors.RESET)
#print(Colors.CYAN + str(bg) + Colors.RESET)
#print(Colors.BLUE + str(llr) + Colors.RESET)

#hmm1 = HiddenMarkovModel.prionHMM1(fg,bg)
#hmm0 = HiddenMarkovModel.prionHMM0(bg)

ww3 = 41

if (args.inputFile is not None):
	#scoreAllFastas(args.inputFile,args.coreLength,args.ww1,args.ww2,ww3,fg,bg,llr,hmm1,hmm0,args.plotDir)
	fasta = Fasta(args.inputFile,args.coreLength,args.ww1,args.ww2,ww3,fg,bg,args.flim)
	fasta.print(args.plotDir)
