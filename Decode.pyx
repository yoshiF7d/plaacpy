import numpy as np
cimport numpy as np

from libc.math cimport exp, log

ctypedef np.int_t INT
ctypedef np.float64_t DOUBLE

cpdef decode(
	np.ndarray[INT,ndim=1] seq,
	int n,
	int ns,
	np.ndarray[DOUBLE,ndim=1] liprob,
	np.ndarray[DOUBLE,ndim=1] lfprob,
	np.ndarray[DOUBLE,ndim=2] ltprob,
	np.ndarray[DOUBLE,ndim=2] leprob
):

	cdef np.ndarray[INT,ndim=1] vit = np.zeros(n,dtype=np.int)
	cdef np.ndarray[INT,ndim=1] map = np.zeros(n,dtype=np.int)
	
	cdef np.ndarray[INT,ndim=2] tb = np.zeros((ns,n),dtype=np.int)
	
	cdef np.ndarray[DOUBLE,ndim=2] s = np.zeros((ns,n),dtype=np.float)
	cdef np.ndarray[DOUBLE,ndim=2] a = np.zeros((ns,n),dtype=np.float)
	cdef np.ndarray[DOUBLE,ndim=2] b = np.zeros((ns,n),dtype=np.float)
	cdef np.ndarray[DOUBLE,ndim=2] pp = np.zeros((ns,n),dtype=np.float)
	
	cdef np.ndarray[DOUBLE,ndim=1] temp = np.zeros(ns,dtype=np.float)
	
	cdef int i,j,t,bestIndex
	cdef double bestScore,ltotProb,lpseq
	
	#print(liprob)
	#print(lfprob)
	#print(ltprob)
	#print(leprob)
	
	for i in range(ns):
		a[i,0] = s[i,0] = liprob[i] + leprob[i,seq[0]]
	
	for t in range(n):
		for i in range(ns):
			for j in range(ns):
				temp[j] = ltprob[j,i] + s[j,t-1]
			tb[i,t] = bestIndex = argmax(temp,ns)
			bestScore = ltprob[bestIndex,i] + s[bestIndex,t-1]
			s[i,t] = bestScore + leprob[i,seq[i]]
			for j in range(ns):
				temp[j] = ltprob[j,i] + a[j,t-1]
			a[i,t] = reduce(temp,ns) + leprob[i,seq[t]]

	for j in range(ns):
		temp[j] = lfprob[j] + a[j,t-1]
	ltotProb = reduce(temp,ns)
	
	for j in range(ns):
		b[j,n-1] = lfprob[j]
		temp[j] = s[j,n-1] + lfprob[j]
		
	bestIndex = argmax(temp,ns)
	bestScore = s[bestIndex,n-1] + lfprob[bestIndex]
	vit[n-1] = bestIndex
	
	for t in reversed(range(n-1)):
		vit[t] = tb[vit[t+1],t+1]
		for i in range(ns):
			for j in range(ns):
				temp[j] = ltprob[i,j] + b[j,t+1] + leprob[j,seq[t+1]]
			b[i,t] = reduce(temp,ns)
			
	for j in range(ns):
		temp[j] = a[j,0] + b[j,0]
	lpseq = reduce(temp,ns)
	
	for i in range(n):
		for j in range(ns):
			pp[j,i] = exp(a[j,i] +b[j,i] - lpseq)

	for i in range(n):
		for j in range(ns):
			if pp[j,i] > pp[map[i],i]:
				map[i] = j

	# ltotProb,lviterbiProb,postProb,viterbiPath,mapPath
	return [ltotProb,bestScore,pp,vit,map]

cdef int argmax(
	np.ndarray[DOUBLE,ndim=1] seq,
	int n
):
	cdef double smax = seq[0]
	cdef int i
	cdef int ind = 0
	
	for i in range(n):
		if smax < seq[i]:
			smax = seq[i]
			ind = i
	
	return ind 

cdef double reduce(
	np.ndarray[DOUBLE,ndim=1] arr,
	int n
):
	cdef double tot = 0
	cdef int i
	
	for i in range(n):
		if tot == 0:
			tot = arr[i]
		else:
			if tot > arr[i]:
				tot += log(1+exp(arr[i]-tot))
			elif tot < arr[i]:
				tot = arr[i] + log(1+exp(tot-arr[i]))
			else:
				tot += log(2)
				
	#print(tot)
	return tot
