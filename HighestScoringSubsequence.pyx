# distutils: extra_compile_args = ["-O3"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np

ctypedef np.int_t INT
ctypedef np.float64_t DOUBLE
ctypedef np.complex128_t COMPLEX

from libc.math cimport sin, cos, M_PI
cdef extern from "complex.h":
	float complex I
	
cdef double sum(
	np.ndarray[DOUBLE,ndim=1] seq,
	int n
):
	cdef double s = 0
	cdef int i
	for i in range(n):
		s += seq[i]
	return s
	
cdef int argmax(
	np.ndarray[DOUBLE,ndim=1] seq,
	int ist,
	int n
):
	cdef double smax = seq[0]
	cdef int i
	cdef int ind = 0
	
	for i in range(ist,n):
		if smax < seq[i]:
			smax = seq[i]
			ind = i
	
	return ind
	
def highestScoringSubsequence(seq,min=None,max=None):
	if min is None:
		min = 1
	if max is None:
		max = len(seq)
	return highestScoringSubsequenceC(seq,len(seq),min,max)
	
cdef highestScoringSubsequenceC(
	np.ndarray[DOUBLE,ndim=1] seq,
	int n,
	int min,
	int max
):
	cdef np.ndarray[COMPLEX,ndim=1] rfseq = np.fft.rfft(seq)
	cdef np.ndarray[COMPLEX,ndim=1] rfrec
	cdef np.ndarray[DOUBLE,ndim=1] conv
	
	cdef int bestStart = 0
	cdef int bestStop = min -1
	cdef double bestScore = sum(seq,min)
	
	cdef int l,k,rn,ind
	
	if n%2==0:
		rn = n//2
		rfrec = np.ndarray(rn+1,dtype='c16')
	else:
		rn = (n+1)//2
		rfrec = np.ndarray(rn,dtype='c16')
		
	for l in range(min,max+1):
		rfrec[0] = l
		for k in range(1,rn):
			rfrec[k] = (cos(M_PI*(l-1)*k/n) - I * sin(M_PI*(l-1)*k/n)) * sin(M_PI*l*k/n) / sin(M_PI*k/n)
		if n%2==0:
			rfrec[rn] = l%2
		
		conv = np.fft.irfft(rfseq*rfrec,n)
		
		if l-1 < n:
			ind = argmax(conv,l-1,n)
		else:
			ind = 0
		
		if conv[ind] >= bestScore:
			bestStart = ind - l + 1
			bestStop = ind
			bestScore = conv[ind]
				
	return [bestStart,bestStop,bestScore]

