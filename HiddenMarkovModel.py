#cumsum(mat,false) : np.cumsum(mat,axis=1)
#cumsum(mat,true) : np.cumsum(mat,axis=0)
#ns : len(iprob)
#no : len(eprob[0])
import numpy as np
from Colors import Colors
from functools import reduce
np.seterr(divide = 'ignore') 

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})
from logProbTrellis import logProbTrellis
from Decode import decode

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
	
	#SLOW 2
	@profile
	def decodeAll(self,seq):
		#self.viterbiDecodeL(seq)
		#self.mapDecodeL(seq)
		self.ltotProb,self.lviterbiProb,self.postProb,self.viterbiPath,self.mapPath = decode(seq,len(seq),self.ns,self.liprob,self.lfprob,self.ltprob,self.leprob)
		self.lmarginalProb = self.ltotProb
		#self.decode(seq)
		self.margCollapse = np.zeros(len(seq))
		self.etst = 0.0
		
		if self.ns == self.numClasses:
			self.margCollapse = np.sum(self.postProb,axis=0)
			self.etst = np.sum(self.postProb[self.subTrellis == 0])
		else:
			self.margCollapse = self.postProb[0]
			self.etst = np.sum(self.postProb[0])
		
		#self.lpst = self.logProbSubTrellis(seq)
		self.lpst = self.logProbSubTrellisFast(seq)
	
	# vitabiDecodeL and mapDecodeL are combined in one and it becomes slow why
	#@profile
	def decode(self,seq):
		n = len(seq)
		vit = np.zeros(n,dtype='i8')
		tb = np.zeros((self.ns,n),dtype='i8')
		
		s = np.zeros((self.ns,n),dtype='float64')
		a = np.zeros((self.ns,n),dtype='float64')
		b = np.zeros((self.ns,n),dtype='float64')
		pp = np.zeros((self.ns,n),dtype='float64')
		
		a[:,0] = self.liprob + self.leprob[:,seq[0]]
		s[:,0] = a[:,0]
		
		for t in range(1,n):
			for i in range(self.ns):
				tb[i,t] = bestIndex = np.argmax(self.ltprob[:,i] + s[:,t-1])
				bestScore = self.ltprob[bestIndex,i] + s[bestIndex,t-1]
				s[i,t] = bestScore + self.leprob[i,seq[t]]
				a[i,t] = reduce(logeapeb,self.ltprob[:,i] + a[:,t-1]) + self.leprob[i,seq[t]]
		
		self.ltotProb = reduce(logeapeb,self.lfprob + a[:,n-1])
		self.lmarginalProb = self.ltotProb
		b[:,n-1] = self.lfprob
		
		bestIndex = np.argmax(s[:,n-1] + self.lfprob)
		bestScore = s[bestIndex,n-1] + self.lfprob[bestIndex]
		vit[n-1] = bestIndex
		
		for t in reversed(range(n-1)):
			vit[t] = tb[vit[t+1],t+1]
			for i in range(self.ns):
				b[i,t] = reduce(logeapeb,self.ltprob[i,:] + b[:,t+1] + self.leprob[:,seq[t+1]]) 
		
		lpseq = reduce(logeapeb,a[:][0] + b[:][0])
		pp = np.exp(a+b-lpseq)
		
		self.lviterbiProb = bestScore
		if self.numClasses < self.ns:
			vit = self.classes[vit]
			pp = collapsePosteriors(pp,self.classes,self.numClasses)
		self.viterbiPath = vit
		self.postProb = pp
		
		map = np.zeros(len(seq),dtype='i8')
		for i in range(n):
			for j in range(self.ns):
				if pp[j,i] > pp[map[i]][i]:
					map[i] = j
		
		self.mapPath = map
		
	#@profile
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
				
		bestIndex = np.argmax(s[:,n-1] + self.lfprob)
		bestScore = s[bestIndex,n-1] + self.lfprob[bestIndex]
		
		vit[n-1] = bestIndex
		for t in reversed(range(n-1)):
			vit[t] = tb[vit[t+1],t+1]

		self.lviterbiProb = bestScore
		if self.numClasses < self.ns:
			vit = self.classes[vit]
		
		self.viterbiPath = vit
		return vit
	
	#@profile
	def mapDecodeL(self,seq):
		map = np.zeros(len(seq),dtype='i8')
		pp = self.posteriorL(seq)
		for i in range(len(seq)):
			for j in range(len(pp)):
				if pp[j,i] > pp[map[i]][i]:
					map[i] = j
		#self.postProb = pp
		self.mapPath = map
		return map
	
	#@profile
	def posteriorL(self,seq):
		n = len(seq)
		a = np.zeros((self.ns,n))
		pp = np.zeros((self.ns,n))
		a[:,0] = self.liprob + self.leprob[:,seq[0]]

		for t in range(1,n):
			for i in range(self.ns):
				a[i,t] = reduce(logeapeb,self.ltprob[:,i] + a[:,t-1]) + self.leprob[i,seq[t]]
				#a[i,t] = logExpSum(self.ltprob[:,i] + a[:,t-1]) + self.leprob[i,seq[t]]
		
		self.ltotProb = reduce(logeapeb,self.lfprob + a[:,n-1])
		#self.ltotProb = logExpSum(self.lfprob + a[:,n-1])
		self.lmarginalProb = self.ltotProb
		
		b = np.ndarray((self.ns,n),dtype='f8')
		b[:,n-1] = self.lfprob
		
		for t in reversed(range(n-1)):
			for i in range(self.ns):
				b[i,t] = reduce(logeapeb,self.ltprob[i,:] + b[:,t+1] + self.leprob[:,seq[t+1]]) 
				#b[i,t] = logExpSum(self.ltprob[i,:] + b[:,t+1] + self.leprob[:,seq[t+1]]) 

		lpseq = reduce(logeapeb,a[:][0] + b[:][0])
		#lpseq = logExpSum(a[:][0] + b[:][0])
		pp = np.exp(a+b-lpseq)
		
		if self.numClasses<self.ns:
			pp = collapsePosteriors(pp,self.classes,self.numClasses)
		
		self.postProb = pp
		return pp
	
	#@profile
	def posterior(self,seq):
		n = len(seq)
		a = np.zeros((self.ns,n))
		b = np.zeros((self.ns,n))
		pp = np.zeros((self.ns,n))
		scale = np.sqrt(np.linalg.det(self.tprob)*np.average(np.prod(self.eprob,0)))
		deta = np.ones(n)
		detb = np.ones(n)

		#print(scale)
		a[:,0] = self.iprob

		for t in range(1,n):
			te = self.tprob * self.eprob[:,seq[t-1]]
			#deta[t] = np.sqrt(dett * np.prod(self.eprob[:,seq[t-1]]))
			#print(deta[t]) 
			a[:,t] = np.dot(te,a[:,t-1]) / scale
		
		self.ltotProb = np.log(np.sum(a[:,n-1] * self.eprob[:,seq[n-1]] * self.fprob)) + n*np.log(scale)
		self.lmarginalProb = self.ltotProb
		
		b[:,n-1] = self.fprob
		
		for t in reversed(range(n-1)):
			te = self.tprob.transpose() * self.eprob[:,seq[t+1]]
			#detb[t] = np.sqrt(dett * np.prod(self.eprob[:,seq[t+1]]))
			b[:,t] = np.dot(te,b[:,t+1]) / scale
		
		pseq = np.sum(a[:,0]*b[:,0]*self.eprob[:,0])
		pp = a*b*self.eprob[:,seq]/pseq
		print(scale)
		
		if self.numClasses<self.ns:
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
	
	def logProbSubTrellisFast(self,seq):
		a = logProbTrellis(
			seq,
			self.subTrellis,
			len(seq),
			self.ns,
			self.iprob,self.tprob,self.eprob
		)
		b = logProbTrellis(
			seq,
			np.ones(self.ns,dtype=np.int),
			len(seq),
			self.ns,
			self.iprob,self.tprob,self.eprob
		)
		return a - b
		#return self.logProbTrellis(seq,self.subTrellis) - self.logProbTrellis(seq,np.ones(self.ns))
	
	#@profile
	def logProbTrellis(self,seq,st):
		n = len(seq)
		sc = np.ones(n)
		a = np.zeros((self.ns,n))
		
		for i in range(self.ns):
			if st[i] == 1:
				a[i,0] =  self.iprob[i]*self.eprob[i,seq[0]]
		
		for t in range(1,n):
			sf = 0
			for i in range(self.ns):
				if st[i]==1:
					a[i,t] = np.sum(self.tprob[:,i]*a[:,t-1])*self.eprob[i,seq[t]]
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
	tmat = np.array([[1.,0.],[0.,1.]])
	imat = np.array([1.,0.])
	bg = normalize(bgFreq)
	emat = np.array([bg,bg])
	
	hmm = HiddenMarkovModel(tmat,emat,imat)
	hmm.subTrellis = np.array([1,0])
	hmm.states = np.array(['-','+'])
	hmm.names = ['background','also.background']
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
	hmm.names = ['background','PrD-like']
	return hmm

def logeapeb(a,b):
	if a > b:
		return a + np.log(1+np.exp(b-a))
	elif b > a:
		return b + np.log(1+np.exp(a-b))
	else:
		return a + np.log(2)
		
def logExpSum(a):
	t = a[a!=-np.inf]
	#print(Colors.RED + str(t) + Colors.RESET)
	if len(t)>0:
		return np.log(np.sum(np.exp(t)))
	
	return -np.inf