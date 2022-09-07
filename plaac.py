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

import os
os.system('')

def normalize(array):
	sum = array.sum()
	if sum == 0:
		sum = 1
	return array/sum

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

#@profile

def scoreAllFastas(file,coreLength,ww1,ww2,ww3,fg,bg,llr,hmm1,hmm0):
	print(
		"SEQid\tMW\tMWstart\tMWend\tMWlen\tLLR\tLLRstart\tLLRend\t"\
		"LLRlen\tNLLR\tVITmaxrun\tCOREscore\tCOREstart\tCOREend\tCORElen\tPRDscore\tPRDstart\tPRDend\tPRDlen\tPROTlen\t"\
		"HMMall\tHMMvit\tCOREaa\tSTARTaa\tENDaa\tPRDaa\tFInumaa\tFImeanhydro\tFImeancharge\tFImeancombo\tFImaxrun\t"\
		"PAPAcombo\tPAPAprop\tPAPAfi\tPAPAllr\tPAPAllr2\tPAPAcen\tPAPAaa"
	)
	print()
	
	fasta = readFasta(file)
	fastalen = len(fasta)
	seqlenmax = 0
	for fs in fasta:
		if len(fs[1]) > seqlenmax:
			seqlenmax = len(fs[1])
	fig,ax = plt.subplots(nrows=fastalen,sharex=True,figsize=(12,3))
	
	sm = plt.cm.ScalarMappable(
		cmap = plt.get_cmap('jet',len(AA.ctable)),
		norm = matplotlib.colors.BoundaryNorm(np.arange(len(AA.ctable)+1)-0.5,len(AA.ctable))
	)
	sm.set_array([])
	cb = fig.colorbar(sm,ax=ax,
			ticks=np.arange(len(AA.ctable)),
			location='top',
			shrink=0.8,
			orientation='horizontal'
		)
	
	cb.set_ticklabels(AA.ctable)
	
	for i,fs in enumerate(fasta):
		name = fs[0]
		seq = fs[1]
		if seq[-1] == '*':
			seq = seq[:-1]
		
		aa = AA.stringToIndices(seq)

		if len(aa) < 1:
			continue		
		maa1 = AA.table["mask"][aa]
		mwsize = 80
		if len(maa1) < 80:
			mwsize = len(maa1)
		hs1 = highestScoringSubsequence(maa1,mwsize,mwsize)
		
		maa2 = llr[aa]
		hs2 = highestScoringSubsequence(maa2,coreLength,coreLength)
		
		hmm1.decodeAll(aa)
		hmm0.decodeAll(aa)
		
		hmmScore = hmm1.lmarginalProb - hmm0.lmarginalProb
		hmmScoreV = hmm1.lviterbiProb - hmm0.lviterbiProb
		
		dr = DisorderReport(aa,ww1,ww2,ww3,np.array([2.785,-1,-1.151]),llr,AA.table['lodpapa1'])
		mp = hmm1.viterbiPath
		#longestPrd = int(highestScoringSubsequence(mp)[2])
		longestPrd = longestOnes(mp)
		maa3 = llr[aa]
		big_neg = -1e+6
		maa3[mp==0] = big_neg
		hs3 = highestScoringSubsequence(maa3,coreLength,coreLength)
		
		coreStart = hs3[0]
		coreStop = hs3[1]
		
		aaStart = coreStart
		aaStop = coreStop
		
		if(hs3[2] > big_neg/2):
			while aaStart >= 0 and mp[aaStart] == 1:
				aaStart -= 1
			aaStart += 1
			while aaStop < len(mp) and mp[aaStop] == 1:
				aaStop += 1
			aaStop -= 1
			prd = aa[aaStart:aaStop+1]
			prdScore = np.sum(llr[prd])
		else:
			hs3[2] = np.nan
			aaStart = -1
			aaStop = -2
			coreStart = -1
			coreStop = -2
			prd = []
			prdScore = 0
		
		fmtstr = (
			'{}\t' + '{:d}\t'*4 +
			'{:.3f}\t' + '{:d}\t'*3 +
			'{:.3f}\t' +
			Colors.RED + '{:d}\t' + Colors.RESET + 
			'{:.3f}\t' + '{:d}\t'*3 +
			'{:.3f}\t' + '{:d}\t'*3 +
			Colors.RED + '{:d}\t' + Colors.RESET +
			'{:.3f}\t'*2
		)
		
		str1 = fmtstr.format(
			Colors.CYAN + name + Colors.RESET,
			int(hs1[2]),hs1[0] + 1,hs1[1] + 1,hs1[1] - hs1[0] + 1,
			hs2[2],hs2[0] + 1,hs2[1] + 1,hs2[1] - hs2[0] + 1,
			hs2[2] / (hs2[1] - hs2[0] + 1),
			longestPrd,
			hs3[2],coreStart + 1,coreStop + 1,coreStop - coreStart + 1,
			prdScore,aaStart + 1,aaStop + 1,aaStop - aaStart + 1,
			len(aa),
			hmmScore,
			hmmScoreV
		)
		
		if aaStop - aaStart + 1 >= coreLength:
			str2 = (
				AA.indicesToString(aa[coreStart:coreStop+1]) + '\t' +
				AA.indicesToString(aa[aaStart:aaStart+14]) + '\t' +
				AA.indicesToString(aa[aaStop-14:aaStop]) + '\t' +
				AA.indicesToString(prd)
			)
		else:
			str2 = '-\t-\t-\t-'
		
		fmtstr = (
			'\t{:d}' +
			'\t{:.3f}'*3 +
			'\t{:d}' + 
			'\t{:.3f}'*5 +
			'\t{:d}\t{}'
		)
		
		str3 = fmtstr.format(
			dr.numDisorderedStrict2,
			dr.meanHydro,dr.meanCharge,dr.meanFI,
			dr.maxLengthFI,
			dr.papaMaxScore,dr.papaMaxProb,dr.papaMaxDis,dr.papaMaxLLR,dr.papaMaxLLR2,
			dr.papaMaxCenter + 1,
			AA.indicesToString(aa[dr.papaMaxCenter-ww2//2:dr.papaMaxCenter+ww2//2+1])
		)
		print(str1+str2+str3)
		
		#ax[i].imshow(AA.stringToColorIndices(seq)[np.newaxis,:],cmap='jet',aspect=20)
		#ax[i].axes.get_xaxis().set_ticks([])
		#ax[i].axes.get_yaxis().set_ticks([])
		#ax[i].yaxis.set_label_coords(-0.05,0)
		#ax[i].set_xlim([0,seqlenmax])
		#ax[i].set_ylabel(name,rotation=0)
	
	#plt.show()

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
parser.add_argument('-p','--plotList',
	help='-p print_list.txt, where print_list.txt has the name of one fasta on each line, and specifies'\
	'\n  which fastas in input.fa will be plotted\n'\
	'  The names must exactly match those in input.fa, but do need need the > symbol before the name.\n'\
	'  If no print_list.txt is specified the output from the program will be a table of summaries for each protein (one per line) in input.fa;\n'\
	'  If a print_list.txt is specified the output from the program will be a table (one line per residue) that is used\n'\
	'  for making plots for each of the proteins listed in print_list.txt.\n'\
	'  If the option is given as -p all, then plots will be made for all of the proteins in input.fa, \n  which is not advised if input.fa is an entire proteome.\n'\
	'  To make the plots from output that has been redirected to output.txt, at the command-line type type\n  Rscript plaac_plot.r output.txt plotname.pdf.'\
	'  This requires that the program R be installed (see http://www.r-project.org/)\n  and will create a file named plotname.pdf, with one plot per page.'\
	'  Calling Rscript plaac_plot.r with no file specified will list other options for plotting.'
)
parser.add_argument('-H','--hmmDotFile',
	help='-H hmm_filename.txt, writes parameters of HMM to hmm_filenmae.txt in dot format, which can be made into a figure with GraphViz.'
)
parser.add_argument('-d','--printHeaders',action='store_true',
	help='-d, print documentation for headers. If flag is not set, headers will not be printed.'
)
parser.add_argument('-s','--printParameters',action='store_false',
	help='-s, skip printing of run-time parameters at top of file. If flag is not set, run-time parameters will be printed.'
)

args = parser.parse_args()

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

hmm1 = HiddenMarkovModel.prionHMM1(fg,bg)
hmm0 = HiddenMarkovModel.prionHMM0(bg)

ww3 = 41

if (args.inputFile is not None) and (args.plotList is None):
	scoreAllFastas(args.inputFile,args.coreLength,args.ww1,args.ww2,ww3,fg,bg,llr,hmm1,hmm0)
