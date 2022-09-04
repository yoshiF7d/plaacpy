import numpy as np
cimport numpy as np

ctypedef np.int_t INT
ctypedef np.float64_t DOUBLE

cpdef DOUBLE logProbTrellis(
	np.ndarray[INT,ndim=1] seq, 
	np.ndarray[INT,ndim=1] st, 
	int n, 
	int ns, 
	np.ndarray[DOUBLE,ndim=1] iprob,
	np.ndarray[DOUBLE,ndim=2] tprob,
	np.ndarray[DOUBLE,ndim=2] eprob
):
	cdef np.ndarray[DOUBLE,ndim=1] sc = np.ones(n,dtype=np.float)
	cdef np.ndarray[DOUBLE,ndim=2] a = np.zeros((ns,n),dtype=np.float)
	cdef int i,j,t
	cdef double sf
	
	for i in range(ns):
		if st[i] == 1:
			a[i,0] = iprob[i]*eprob[i,seq[0]]
		
	for t in range(1,n):
		sf = 0
		for i in range(ns):
			if st[i] == 1:
				a[i,t] = 0
				for j in range(ns):
					a[i,t] += tprob[j,i]*a[j,t-1]
				a[i,t] *= eprob[i,seq[t]]
				#a[i,t] = np.sum(tprob[:,i]*a[:,t-1])*eprob[i,seq[t]]
				sf += a[i,t]
		sc[t] = 1/sf
		for i in range(ns):
			a[i,t] = a[i,t]*sc[t]
		
	sf = 0
	for i in range(n):
		sf += sc[i]
	
	return - sf
	#return - np.sum(np.log(sc))