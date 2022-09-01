#cumsum(mat,false) : np.cumsum(mat,axis=1)
#cumsum(mat,true) : np.cumsum(mat,axis=0)
#ns : len(iprob)
#no : len(eprob[0])
import numpy as np
from Colors import Colors
from functools import reduce
np.seterr(divide = 'ignore') 

class HiddenMarkovModel():
	def	__init__(self,tprob,eprob,iprob):
		self.tprob = tprob
		self.eprob = eprob
		self.iprob = iprob
		
		self.ctprob = np.cumsum(self.tprob,axis=0)
		self.ceprob = np.cumsum(self.eprob,axis=0)
		self.ciprob = np.cumsum(self.iprob,axis=0)
		
		self.ltprob = np.log(self.tprob)
		self.leprob = np.log(self.eprob)
		self.liprob = np.log(self.iprob)
		
		print(Colors.YELLOW + str(self.ltprob))
		print(Colors.CYAN + str(self.leprob))
		print(Colors.BLUE + str(self.liprob) + Colors.RESET)
				
		self.rs = np.sum(self.tprob,axis=1)
		self.ns = len(self.iprob)
		self.fprob = np.zeros(self.ns)
		
		freeEnd = True
		for i in range(self.ns):
			self.fprob[i] = max(0.0,1.0-self.rs[i])
			if self.fprob[i] > 0.0001:
				freeEnd = False
		
		if freeEnd:
			self.fprob.fill(1.0)
		
		self.lfprob = np.log(self.fprob)
		self.subTrellis = np.ones(self.ns)
		self.states = np.arange(self.ns)
		self.names = np.arange(self.ns)
		self.classes = np.arange(self.ns)
		
		self.numClasses = self.ns
		self.uStates = self.states
		self.uNames = self.names
		
		self.lviterbiProb = - float('inf')
		self.lmarginalProb = - float('inf')
		
		self.viterbiPath = None
		
		self.mapPath = None
		self.postProb = None
		self.margCollapse = None
		self.ltotProb = None
		self.lpseq = None
		
		self.etst = 0.0
		self.lpst = 0.0
	
	def setnames(self,names):
		self.names = names
		self.uNames = names
	
	def decodeAll(self,seq):
		self.viterbiDecodeL(seq)
		self.mapDecodeL(seq)
		self.margCollapse = np.zeros(len(seq))
		self.etst = 0.0
		
		if self.ns == self.numClasses:
			self.margCollapse = np.sum(self.postProb,axis=0)
			self.etst = np.sum(self.postProb[self.subTrellis == 0])
		else:
			self.margCollapse = self.postProb[0]
			self.etst = np.sum(self.postProb[0])
		
		self.lpst = self.logProbSubTrellis(seq)
	
	def viterbiDecodeL(self,seq):
		n = len(seq)
		vit = np.zeros(n,dtype='i8')
		s = np.zeros((self.ns,n))
		tb = np.zeros((self.ns,n))
		
		s[:,0] = self.liprob + self.leprob[:,seq[0]]
		
		for t in range(1,n):
			for i in range(self.ns):
				bestIndex = np.argmax(self.ltprob[:,i] + s[:,t-1])
				bestScore = self.ltprob[bestIndex,i] + s[bestIndex,t-1]
				s[i,t] = bestScore + self.leprob[i,seq[t]]
				tb[i,t] = bestIndex
		
		#print(Colors.YELLOW + str(self.ltprob[:,0] + s[:,0]) + Colors.RESET)
		#print(Colors.GRAY + str(s) + Colors.RESET)
		#print(Colors.MARINE + str(tb) + Colors.RESET)
		
		bestIndex = np.argmax(s[:,n-1] + self.lfprob)
		bestScore = s[bestIndex,n-1] + self.lfprob[bestIndex]
		
		vit[n-1] = bestIndex
		for t in reversed(range(n-1)):
			vit[t] = tb[vit[t+1],t+1]

		#print(Colors.DARKGREEN + str(vit) + Colors.RESET)

		self.lviterbiProb = bestScore
		if self.numClasses < self.ns:
			vit = self.classes[vit]
		
		self.viterbiPath = vit
		return vit
	
	def mapDecodeL(self,seq):
		map = np.zeros(len(seq),dtype='i8')
		pp = self.posteriorL(seq)
		for i in range(len(seq)):
			for j in range(len(pp)):
				if pp[j][i] > pp[map[i]][i]:
					map[i] = j
		self.postPorb = pp
		self.mapPath = map
		return map
		
	def posteriorL(self,seq):
		n = len(seq)
		a = np.zeros((self.ns,n))
		pp = np.zeros((self.ns,n))
		a[:,0] = self.liprob +self.leprob[:,seq[0]]
		#print(Colors.CYAN + str(a) + Colors.RESET)
		for t in range(1,n):
			for i in range(self.ns):
				a[i,t] = reduce(logeapeb,self.ltprob[:,i] + a[:,t-1]) + self.leprob[i,seq[t]]
				#a[i,t] = np.log(np.sum(np.exp(self.ltprob[:,i] + a[:,t-1]))) + self.leprob[i,seq[t]]
				#print(np.exp(self.ltprob[:,i] + a[:,t-1]))
		
		#print(Colors.RED + str(a) + Colors.RESET)
		
		self.ltotProb = reduce(logeapeb,self.lfprob + a[:,n-1])
		#self.ltotProb = np.log(np.sum(np.exp(self.lfprob + a[:,n-1])))
		self.lmarginalProb = self.ltotProb
		#print(Colors.DARKRED + str(self.leprob) + Colors.RESET)
		#print(Colors.YELLOW + str(self.liprob) + Colors.RESET)
		#print(Colors.CYAN + str(self.lfprob) + Colors.RESET)
		#print(Colors.BLUE + str(self.lTotProb) + Colors.RESET)
		#print(Colors.CYAN + str(np.exp(self.lfprob + a[:,n-1])) + Colors.RESET)
		#print(Colors.BLUE + str(self.lfprob + a[:,n-1]) + Colors.RESET)
		
		b = np.ndarray((self.ns,n),dtype='f8')
		b[:,n-1] = self.lfprob
		
		for t in reversed(range(n-1)):
			for i in range(self.ns):
				b[i,t] = reduce(logeapeb,self.ltprob[i,:] + b[:,t+1] + self.leprob[:,seq[t+1]]) 
				#b[i,t] = np.log(np.sum(np.exp(self.ltprob[i,:] + b[:,t+1] + self.leprob[:,seq[t+1]])))
		
		lpseq = reduce(logeapeb,a[:][0] + b[:][0])
		#print(Colors.MAGENTA + str(self.lfprob + a[:,n-1]) + Colors.RESET)
		pp = np.exp(a+b-lpseq)
		#print(Colors.GRAY + str(a) + Colors.RESET)
		#print(Colors.DARKCYAN + str(b) + Colors.RESET)
		#print(Colors.YELLOW + str(lpseq) + Colors.RESET)
		
		if(self.numClasses<self.ns):
			pp = collapsePosteriors(pp,self.classes,self.numClasses)
		
		self.postProb = pp
		return pp
		
	def posterior(self,seq):
		n = len(seq)
		a = np.zeros((self.ns,n))
		pp = np.zeros((self.ns,n))
		a[:,0] = self.iprob * self.eprob[:,seq[0]]

		#print(Colors.GREEN + str(self.iprob) + Colors.RESET)
		#print(Colors.DARKGREEN + str(np.sum(self.eprob)) + Colors.RESET)
		#print(Colors.CYAN + str(a[:,0]) + Colors.RESET)
		
		for t in range(1,n):
			for i in range(self.ns):
				a[i,t] = np.sum(self.tprob[:,i]*a[:,t-1]) * self.eprob[i,seq[t]]
		#print(Colors.RED + str(a) + Colors.RESET)
		
		self.ltotProb = np.log(np.sum(self.fprob*a[:,n-1]))
		self.lmarginalProb = self.ltotProb
		#print(Colors.DARKRED + str(self.leprob) + Colors.RESET)
		#print(Colors.YELLOW + str(self.lMarginalProb) + Colors.RESET)
		#print(Colors.CYAN + str(self.fprob) + Colors.RESET)
		#print(Colors.BLUE + str(np.sum(self.fprob*a[:,n-1])) + Colors.RESET)
		#print(Colors.CYAN + str(np.exp(self.lfprob + a[:,n-1])) + Colors.RESET)
		#print(Colors.BLUE + str(self.lfprob + a[:,n-1]) + Colors.RESET)
		
		b = np.ndarray((self.ns,n),dtype='f8')
		b[:,n-1] = self.fprob
		
		for t in reversed(range(n-1)):
			for i in range(self.ns):
				b[i,t] = np.sum(self.tprob[i,:]*b[:,t+1]*self.eprob[:,seq[t+1]])
		
		pseq = np.sum(a[:][0] * b[:][0])
		#print(Colors.MAGENTA + str(self.lfprob + a[:,n-1]) + Colors.RESET)
		pp = a*b/pseq
		print(Colors.GRAY + str(pseq) + Colors.RESET)
		#print(Colors.DARKCYAN + str(b) + Colors.RESET)
		#print(Colors.YELLOW + str(lpseq) + Colors.RESET)
		
		if(self.numClasses<self.ns):
			pp = collapsePosteriors(pp,self.classes,self.numClasses)
		
		self.postProb = pp
		return pp
	
	def collapsePosteriors(self,pp,mask,newns):
		npp = np.zeros((newns,len(pp)))
		for j in range(len(pp[0])):
			npp[mask,j] += pp[:,j]
		
		return npp
		
	def logProbSubTrellis(self,seq):	
		return self.logProbTrellis(seq,self.subTrellis) - self.logProbTrellis(seq,np.ones(self.ns))
	
	def logProbTrellis(self,seq,st):
		n = len(seq)
		sc = np.ones(n)
		a = np.zeros((self.ns,n))
		
		for i in range(self.ns):
			if st[i] == 1: a[i,0] =  self.iprob[i]*self.eprob[i,seq[0]]	
		
		for t in range(1,n):
			sf = 0
			for i in range(self.ns):
				if st[i]==1:
					a[i,t] = np.sum(self.tprob[:,i]*a[:,t-1])*self.eprob[i][seq[t]]
					sf += a[i,t]
			sc[t] = 1/sf
			a[:,t] = a[:,t]*sc[t]
		
		return - np.sum(np.log(sc))
		
def normalize(array):
	sum = array.sum()
	if sum == 0:
		sum = 1
	return array/sum

def prionHMM0(bgFreq):
	tmat = np.array([[1,0],[0,1]])
	imat = np.array([1,0])
	bg = normalize(bgFreq)
	emat = np.array([bg,bg])
	
	hmm = HiddenMarkovModel(tmat,emat,imat)
	hmm.subTrellis = np.array([1,0])
	hmm.states = np.array(['-','+'])
	hmm.setnames(['background','also.background'])
	return hmm
	
def prionHMM1(fgFreq,bgFreq):
	tmat = np.array([[99.9/100,0.1/100],[2.0/100,98.0/100]])
	imat = np.array([0.9524,0.0476])
	bg = normalize(bgFreq)
	fg = normalize(fgFreq)
	emat = np.array([bg,fg])
	
	hmm = HiddenMarkovModel(tmat,emat,imat)
	hmm.subTrellis = np.array([1,0])
	hmm.states = np.array(['-','+'])
	hmm.setnames(['background','PrD-like'])
	return hmm

def logeapeb(a,b):
	if a > b:
		return a + np.log(1+np.exp(b-a))
	elif b > a:
		return b + np.log(1+np.exp(a-b))
	else:
		return a + np.log(2)