
import math
import numpy as np
import pandas as pd
import json
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def getGenericSize( x ) : 
	try : float(x) 
	except TypeError : 
		try : N = len(x) 
		except Exception as e : 
			raise ValueError( 'Cannot parse a generic size for object (%s)' % e )
		else : return N
	else : return 1
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# is this a reasonable import? 
from scipy.ndimage.interpolation import shift as scipyshift

# internal implementation
def stupidMemShift( x , i , j , n , usenp=False , fill=0.0 ) : 
	
	""" naive shift of elements in memory... x[i:j) -> x[i+n:j+n) """
	
	if n == 0 : return
	if n > 0 : # shift to the left
		if j+n > len( x ) : raise IndexError( 'requested shift goes out-of-bounds for array passed' )
		if usenp : 
			t = x[i:j].copy()
			x[i+n:j+n] = t
			x[i:i+n] = fill
		else : 
			k = j-1
			while k >= i : 
				x[k+n] = x[k]
				k -= 1
			# here k == i-1; increment and count up with fill
			k += 1
			while k < i+n : 
				x[k] = fill
				k += 1
	else : # n < 0, shift to the right not left
		if i+n < 0 : raise IndexError( 'requested shift goes out-of-bounds for array passed' )
		if usenp : 
			t = x[i:j].copy()
			x[i+n:j+n] = t
			x[j+n:j] = fill
		else : 
			k = i
			while k < j : 
				x[k+n] = x[k]
				k += 1
			# here k == j; back up and count up with fill
			k = j+n
			while k < j : 
				x[k] = fill
				k += 1
	
	return 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def r_int_asr( f , a , b , tol , w , fa , fb , m , cluster=False , sfreq=-1.0 ):
	
	""" Recursive implementation of adaptive Simpson's rule for a callable function 
	
		test log: 
	"""
	
	# NOTE: even fewer function calls needed if we pass down fa, fb, fm, and m
	
	if m is None : m  = ( a + b ) / 2.0 # midpoint
	ml = ( a + m ) / 2.0
	mr = ( m + b ) / 2.0
	hl = ( m - a ) / 6.0
	hr = ( b - m ) / 6.0
	
	# cluster function calls for additional efficiency? 
	if cluster : 
		if fa is None and fb is None : 
			t = [a,ml,m,mr,b]
			F = f(t)
			fa = F[0]
			fl = F[1]
			fm = F[2]
			fr = F[3]
			fb = F[4]
		elif fa is None : 
			t = [a,ml,m,mr]
			F = f(t)
			fa = F[0]
			fl = F[1]
			fm = F[2]
			fr = F[3]
		elif fb is None : 
			t = [ml,m,mr,b]
			F = f(t)
			fl = F[0]
			fm = F[1]
			fr = F[2]
			fb = F[3]
		else : # fa and fb already defined as passed in
			t = [ml,m,mr]
			F = f(t)
			fl = F[0]
			fm = F[1]
			fr = F[2]
		# print( F )
	else : 
		if fa is None : fa = f(a)
		fl = f(ml)
		fm = f(m)
		fr = f(mr)
		if fb is None : fb = f(b)
	
	# left and right simpson's rules
	l = ( fa + 4.0 * fl + fm ) * hl
	r = ( fm + 4.0 * fr + fb ) * hr
	
	# test and return/recursion
	if abs( l + r - w ) <= 15 * tol : 
		# print( 'returning (%0.2f,%0.2f): %0.6f' % ( a , b , l + r + ( l + r - w ) / 15.0 ) )
		return l + r + ( l + r - w ) / 15.0
	else : 
		# print( 'recursing...' , abs( l + r - w ) , ' vs ' , 15*tol )
		
		if 3.0 * hl <= 1.0 / sfreq : rl = l # don't recurse below twice the sampling rate
		else : rl = r_int_asr( f , a , m , tol/2.0 , l , fa , fm , ml , cluster )
		
		if 3.0 * hr <= 1.0 / sfreq : rr = r # don't recurse below twice the sampling rate
		else : rr = r_int_asr( f , m , b , tol/2.0 , r , fm , fb , mr , cluster )
		
		return rl + rr

def int_asr( f , a , b , tol=1.0e-4 , cluster=False , sfreq=-1 ):
	
	""" Calculate integral of f from a to b with max error of t using recursive adaptive simpsons rule 
		
		test log: 
	"""
	
	if a == b : return 0 
	
	tol = 1.0e-16 if tol <= 1.0e-16 else tol # safegaurd tol spec
	
	m  = (a+b) / 2.0
	h  = abs(b-a) / 6.0
	fa = f(a)
	fm = f(m)
	fb = f(b)
	w = h * ( fa + 4.0 * fm + fb )
	
	print( 'initial guess: ' , w )
	
	return r_int_asr( f , a , b , tol , w , fa , fb , m , cluster , sfreq ) if b > a else r_int_asr( f , b , a , tol , w , fb , fa , m , cluster , sfreq )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class OXCTSO :
	
	""" # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	
	TSO: A class for Time Series Objects. 
	
	This object encapsulates three things: 
	
	  _t: a type ('b': boolean, 'c': categorical, 'f': float)
	  _T: a vector of time increments from start (in seconds)
	  _V: a vector of data values (in various units)
	  
	We may also package in other meta data. vectors _T and _V are of
	the same length (with _T[0] = 0?). 
	
	While Pandas has "time series" representations, these are 
	basically just tables whose row indices are time. This is a 
	different form, where we store a set of time-value pairs
	(but in different arrays to facilitate high-performance
	analysis over value vectors)
	
	Pandas also has sparse objects, which also could be used. 
	This is a type of sparse object. 
	
	Append routines are provided, but it is more efficient to 
	have pre-allocations. We implement blocking allocations. 
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # """ 
	
	_blocksize = 100 # class variable: block size (in elements)
	_timetype  = np.float64 # default time type for these objects
	
	def __init__( self , type='f' , size=0 ) : 
		
		""" Initialize data structures 
		
			test log: 
		"""
		
		self._S = None # start time, in datetime format
		
		# data type; boolean, categorical, or float
		if   type == 'b' : self._t = np.bool_
		elif type == 'c' : self._t = np.uint8
		elif type == 'f' : self._t = np.float64
		else : raise ValueError( 'Unknown Type Code' )
		
		# allocate for value array... note we use blocking, so we allocate mB bytes where
		# m is the smallest int such that m B >= N, i.e. m = ceil( N / B )
		# also allocate for time array
		self._N = size
		if self._N > 0 : 
			self._A = OXCTSO._blocksize * math.ceil( self._N / OXCTSO._blocksize )
			self._V = np.empty( (self._A,) , dtype=self._t          )
			self._T = np.empty( (self._A,) , dtype=OXCTSO._timetype )
		else : 
			self._A = 0
			self._V = None
			self._T = None
		
		# value mapping, for categorical data
		self._M  = None
		self._iM = None
		
		# until we load data, don't say we have any
		self._N = 0
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def makeValueMapping( self , D ) : 
		
		""" value mapping, if categorical, from a list of string. That is, "str" is replaced in the data by _M["str"] 
			
			test log: 
		"""
		
		self._M = {} # reset as an empty dict
		c = 1 # coding starts at 1, by convention
		for k in D : 
			self._M[k] = c
			c += 1
			
		# inverse lookup
		self._iM = dict( (v, k) for k, v in self._M.items() )
		
		return
		
	def mapValue( self , s ) : return self._M[s]
	def valueMap( self , v ) : return self._iM[v]
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		
	def isBoolean( self ) 	  : return ( True if self._t is np.bool_   else False )
	def isCategorical( self ) : return ( True if self._t is np.uint8   else False )
	def isFloat( self ) 	  : return ( True if self._t is np.float64 else False )
	
	def typeMatch( self , v ) : 
		""" match input value (v) to this TSO's type, and coerce value to return
			
			test log: 
		"""
		if self.isBoolean() : # this is a bit naive, but hey
			if v : return (True,True)
			else : return (False,False)
		elif self.isCategorical() : 
			if v in self._M : return (True,self._M[v]) # passed something from the dictionary
			try : V = int( v )
			except Exception : return (False,v)
			else : 
				if V in self._iM : return (True,V) # passed int is in inverse dictionary
				else : return (False,v)
		elif self.isFloat() :  
			try : V = float( v )
			except Exception : return (False,v)
			else : return (True,V)
		else : raise ValueError( 'TSO has an unknown code' )
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def findTimes( self , t , sorted=False , orderout=False ) : 
		
		""" find if time(s) exist, passing back bool flags and indices if they do 
			
			note this returns a *sorted* result, sorted such that:
			
			1. all assignments come first (if any)
			2. followed by insertions (if any) ordered last-to-first in insertion index
			
			2. is important because the insertion (and assignment) indices will
			change with rearrangement, and going last-to-first (back-to-front) 
			avoids this problem
			
			test log: 
		"""
		
		if t is None : return (None,None,None)
		if self._T is None : 
			try : float(t) 
			except TypeError : 
				try : N = len(t)
				except Exception as e : 
					raise ValueError( "can't assess length of the times passed to findTimes" )
				else : 
					if sorted : return ( [ False for i in iter(t) ] , list(range(0,N)) , None )
					else : 
						si = np.argsort( t )
						return ( [ False for i in iter(t) ] , si , si )
			else : # we have a float, basically
				return ( [False] , [0] , [0] )
		
		if sorted : si = None
		else : 
			si = np.argsort( t )
			t = np.array( t ).flatten()[si]
		
		# find insertion points with search sorted
		I  = np.searchsorted( self._T[0:self._N] , t , side='left' )
		
		# compare times passed to returned searchsorted indices, and append 
		# True/False flags for "assignments" instead of "insertions"
		iT = []
		for i in range(0,len(I)) :
			if self._T[I[i]] == t[i] : iT.append( True )
			else : iT.append( False ) # self._T[I[i]] > t[i]
		
		# ok, so two more steps. really, one, but here it is: if there are points 
		# to be inserted after array ends, or before array starts, we modify. Examples
		# might help: 		
		# 
		#   _N = 2, _T = ( 5 , 6 ), t = ( 7 , 8 , 9 )
		#    
		# implies 
		# 
		# 	R = ([F,F,F],[2,2,2],None) -> ([F,F,F],[2,3,4],None)
		# 
		# Also, 
		# 
		#   _N = 2, _T = ( 2 , 3 ), t = ( 0 , 1 )
		#    
		# implies R = ([F,F],[0,0],None); doing a backward search/shift/insert does
		# 
		# 	(_T,_V) -> ((2,3),(v2,v3)) -> ((?,2,3),(?,v2,v3)) -> ((1,2,3),(v[1],v2,v3))
		# 	(_T,_V) -> ((1,2,3),(v[1],v2,v3)) -> ((?,1,2,3),(?,v[1],v2,v3)) -> ((0,1,2,3),(v[0],v[1],v2,v3))
		# 
		# so we don't need to do anything if 1. the times are ascending and 2. we shift 
		# one at a time. If we block shift, however, we can restart from 0 to get
		# 
		# 	(_T,_V) -> ((2,3),(v2,v3)) -> ((?,?,2,3),(?,?,v2,v3)) 
		# 							   -> ((0,?,2,3),(v[0],?,v2,v3))
		# 							   -> ((0,1,2,3),(v[0],v[1],v2,v3))
		# 
		if orderout : 
			c = 1
			try : 
				while I[-c] >= self._N : c += 1
			except IndexError : # ALL elements are after array
				for i in range(1,len(I)) : 
					I[i] = I[i-1] + 1
			else : 
				# c == -1, none; c == -2, 1; c == -3, 2 etc
				# that is, count c is c - 1
				# I[-c] < self._N, I[-c-1] >= self._N
				c = -c-2
				while c > 0 : 
					I[c] = I[c-1] + 1
					c -= 1
		
		return (iT,I,si)
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		
	def setByTime( self , t , v , add=True , accumulate=False , sorted=False ) : 
		
		""" set value(s) by time(s) 
			
			add: add elements (in order) if they don't exist
			accumulate: add into existing values, instead of assigning
			sorted: True if the times passed are already sorted
			
			test log: 
		"""
		
		if t is None or v is None : return
		if ( self._T is None ) and add: 
			try : float(t) 
			except TypeError : 
				try : len(t) 
				except Exception as e : 
					raise ValueError( 'cannot parse times passed as an array (%s)' % e )
				else : 
					if sorted : return self.append( t , v )
					else : return self.append( np.sort( t ) , v )
			
			else : return self.append( t , v )
		
		N = getGenericSize( t )
		if getGenericSize( v ) != N : 
			raise ValueError( 'time and value arguments inconsistently sized' )
		
		# get assignment or insertion indices, as a tuple ([assign],[_T indices],[t indices])
		
		if add : # include insertions in time, so search BACKWARD over results
			
			# find times 
			R = self.findTimes( t , sorted=sorted , orderout=True )
			
			if N == 1 : 
				
				if self._N + 1 >= self._A : self.extend( )
				
				if R[0][0] : self._V[0] = self._V[0] + v if accumulate else v
				else : 
					if R[1][0] >= self._N : # just insert at end
						self._T[self._N] = t # assign new value at the emptied location
						self._V[self._N] = v # assign new value at the emptied location
						self._N += 1 # don't forget to increment size with each insert
					else : 
						stupidMemShift( self._T , R[1][0] , self._N , 1 , usenp=True )
						stupidMemShift( self._V , R[1][0] , self._N , 1 , usenp=True )
						# scipyshift( self._T[R[1][0]:self._N] , 1 , mode='constant' , cval=np.nan )
						# scipyshift( self._V[R[1][0]:self._N] , 1 , mode='constant' , cval=np.nan )
						self._T[R[1][0]] = t # assign new value at the emptied location
						self._V[R[1][0]] = v # assign new value at the emptied location
						self._N += 1 # don't forget to increment size with each insert
			
			else : 
				
				# extend array to fit number of insertions required, if needed
				Ni = len(R[0]) - np.sum(R[0]) # should be number of False's
				if self._N + Ni >= self._A : 
					self.extend( A=(OXCTSO._blocksize * math.ceil( (self._N+Ni) / OXCTSO._blocksize ) ) )
				
				if sorted : 
					
					i = len( R[0] )-1 # start at end, search backward
					try : 
						while R[1][i] >= self._N : # just insert
							self._V[ R[1][i] ] = v[i]
							self._T[ R[1][i] ] = t[i]
							i -= 1
					except IndexError : pass # no more indices left
					else : # ok, so now we are inserting into the actual _T,_V arrays until index is zero
						try : 
							while R[1][i] > 0 : 
								if R[0][i] : self._V[R[1][i]] = self._V[R[1][i]] + v[i] if accumulate else v[i]
								else : # shift and insert... t[i] occurs BEFORE _T[ R[1][i] ]
									stupidMemShift( self._T , R[1][i] , self._N , 1 , usenp=True )
									stupidMemShift( self._V , R[1][i] , self._N , 1 , usenp=True )
									# scipyshift( self._T[R[1][i]:self._N] , 1 , mode='constant' , cval=np.nan )
									# scipyshift( self._V[R[1][i]:self._N] , 1 , mode='constant' , cval=np.nan )
									self._T[R[1][i]] = t[i] # assign new value at the emptied location
									self._V[R[1][i]] = v[i] # assign new value at the emptied location
									self._N += 1 # don't forget to increment size with each insert
								i -= 1 # could maybe get smarter with block inserts... but more search
						except IndexError : pass # no more indices left
						else : 
							# are there any elements with t[i] EQUAL to _T[0]? 
							if accumulate : 
								try : 
									while R[0][i] : 
										self._V[0] = self._V[0] + v[i]
										i -= 1
								except IndexError : pass
							else : 
								try : 
									while R[0][i] : i -= 1
								except IndexError : self._V[0] = v[0]
								else : self._V[0] = v[i]
							# if we have elements left then block shift and restart a forward search at zero
							if i >= 0 : # still indices left
								# there are i+1 elements left (e.g., R[0][0,1,...,i])
								# so shift _T,_V[0:_N] -> _T,_V[i+1:_N+i+1]
								stupidMemShift( self._T , 0 , self._N , i+1 , usenp=True )
								stupidMemShift( self._V , 0 , self._N , i+1 , usenp=True )
								# scipyshift( self._T[0:self._N] , i+1 , mode='constant' , cval=np.nan )
								# scipyshift( self._V[0:self._N] , i+1 , mode='constant' , cval=np.nan )
								# increment _N
								self._N += i+1
								# assign values in forward order
								self._V[0:i+1] = v[0:i+1]
								self._T[0:i+1] = t[0:i+1]
					# whew, done here
					
				else : 
					
					i = len( R[0] )-1 # start at end, search backward
					try : 
						while R[1][i] >= self._N : # just insert
							self._V[ R[1][i] ] = v[R[2][i]]
							self._T[ R[1][i] ] = t[R[2][i]]
							i -= 1
					except IndexError : pass # no more indices left
					else : # ok, so now we are inserting into the actual _T,_V arrays until index is zero
						
						try : 
							while R[1][i] > 0 : 
								if R[0][i] : self._V[R[1][i]] = self._V[R[1][i]] + v[R[2][i]] if accumulate else v[R[2][i]]
								else : # shift and insert... t[i] occurs BEFORE _T[ R[1][i] ]
									stupidMemShift( self._T , R[1][i] , self._N , 1 , usenp=True )
									stupidMemShift( self._V , R[1][i] , self._N , 1 , usenp=True )
									# scipyshift( self._T[R[1][i]:self._N] , 1 , mode='constant' , cval=np.nan )
									# scipyshift( self._V[R[1][i]:self._N] , 1 , mode='constant' , cval=np.nan )
									self._T[R[1][i]] = t[R[2][i]] # assign new value at the emptied location
									self._V[R[1][i]] = v[R[2][i]] # assign new value at the emptied location
									self._N += 1 # don't forget to increment size with each insert
								i -= 1 # could maybe get smarter with block inserts... but more search
						except IndexError : pass # no more indices left
						else : 
							
							# are there any elements with t[i] EQUAL to _T[0]? 
							if accumulate : 
								try : 
									while R[0][i] : 
										self._V[0] = self._V[0] + v[R[2][i]]
										i -= 1
								except IndexError : pass
							else : 
								try : 
									while R[0][i] : i -= 1
								except IndexError : self._V[0] = v[R[2][0]]
								else : self._V[0] = v[R[2][i]]
							# if we have elements left then block shift and restart a forward search at zero
							if i >= 0 : # still indices left
								# there are i+1 elements left (e.g., R[0][0,1,...,i])
								# so shift _T,_V[0:_N] -> _T,_V[i+1:_N+i+1]
								stupidMemShift( self._T , 0 , self._N , i+1 , usenp=True )
								stupidMemShift( self._V , 0 , self._N , i+1 , usenp=True )
								# scipyshift( self._T[0:self._N] , i+1 , mode='constant' , cval=np.nan )
								# scipyshift( self._V[0:self._N] , i+1 , mode='constant' , cval=np.nan )
								# increment _N
								self._N += i+1
								# assign values in forward order
								self._V[0:i+1] = np.array(v)[R[2][0:i+1]] # need numpy cast to understand fancy indexing
								self._T[0:i+1] = np.array(t)[R[2][0:i+1]] # need numpy cast to understand fancy indexing
								
					# whew, done here
			
		else : # assignments only, search forward ok
			
			R = self.findTimes( t , sorted=sorted , orderout=False )
			
			if N == 1 : 
				if R[0][0] : self._V[R[1][0]] = self._V[R[1][0]] + v if accumulate else v
			else : 
				if sorted : 
					for i in range(len(R[0])) :
						if R[0][i] : self._V[R[1][i]] = self._V[R[1][i]] + v[i] if accumulate else v[i]
				else : 
					for i in range(len(R[0])) : 
						if R[0][i] : self._V[R[1][i]] = self._V[R[1][i]] + v[R[2][i]] if accumulate else v[R[2][i]]
		
		return
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def extend( self , b=1 , A=None ) : 
		
		""" expand arrays, ala realloc, by b blocks or otherwise 
			
			this doesn't set any values, rather simply manages the allocated array size
			
			test log: 
		"""
		
		if A is None : self._A += b * OXCTSO._blocksize
		elif A <= self._A : return
		elif A % OXCTSO._blocksize != 0 : self._A = OXCTSO._blocksize * math.ceil( A / OXCTSO._blocksize )
		else : self._A = A
		
		if self._V is None : self._V = np.empty( (self._A,) , dtype=self._t )
		else : self._V.resize( (self._A,) )
		
		if self._T is None : self._T = np.empty( (self._A,) , dtype=OXCTSO._timetype )
		else : self._T.resize( (self._A,) )
		
		return
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		
	def append( self , t , v ) :
	
		""" append elements; don't use too often, as this is slow. blocking memory speeds this up a bit 
			
			test log: 
		"""
		 
		if t is None or v is None : return
		
		try : float(t) 
		except TypeError : 
			try : N = len( t )
			except Exception as e : 
				raise ValueError( 'cannot parse length of times passed (%s)' % e )
			else : 
				try : M = len(v)
				except Exception : 
					raise ValueError( 't and v incompatible' )
				else : 
					if N != M : 
						raise ValueError( 't and v incompatible' )
		else : 
			try : float(v) 
			except TypeError : 
				raise ValueError( 't and v incompatible' )
			else : N = 1 # t doesn't have a shape attribute, presumably is a scalar
		
		if N == 1 : 
			if self._N + 1 >= self._A : self.extend(  ) # default extension by one block
		else : 
			if self._N + N >= self._A : self.extend( A=(OXCTSO._blocksize * math.ceil( (self._N+N) / OXCTSO._blocksize ) ) )
		
		self._V[self._N:self._N+N] = v 
		self._T[self._N:self._N+N] = t
		self._N += N
		
		return
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		
	def extractByTime( self , s , e ) :
		
		""" extract subset of data using time indices
			
			test log: 
		"""
		
		if s is None and e is None : return None
		
		if s is None : s = self._T[0]
		elif s > self._T[self._N-1] : return None
		
		if e is None : e = self._T[self._N-1]
		elif e < self._T[0] : return None
		
		# cool print format: [ ... ( ... ) ... ] , ( ... [ ... ) ... ] , [ ... ( ... ] ... ) , ( ... ) [ ... ] , [ ... ] ( ... )
		
		if s > e : raise ValueError( 'start time (%0.2f) must be less than end time (%0.2f) (%i)' % (s,e,self._N) )
		
		# find valid indices
		I = np.where( np.logical_and( self._T >= s , self._T <= e ) )[0]
		N = len(I)
		
		if N == 0 : return None
		
		# initialize new OXCTSO and copy data
		if self.isBoolean() : R = OXCTSO( type='b' , size=N )
		elif self.isFloat() : R = OXCTSO( type='f' , size=N )
		elif self.isCategorical() : 
			R = OXCTSO( type='c' , size=N )
			R._M  = dict( self._M  )
			R._iM = dict( self._iM )
		else : raise ValueError( 'unknown type code' )
		R._N = N
		R._V = self._V[I]
		R._T = self._T[I]
		
		return R
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def setIRIMethod( self , m ) :
		
		""" set method for interpolation/reverse interpolation
			e.g., round up, round down, linear interp, quadratic interp
			NOTE : NOT YET IMPLEMENTED
			
			test log: 
		"""
		
		return
	
	def interp( self , t , sorted=False ) : 
		
		""" interpolate values at time(s) 
		
			sorted declares whether the times given are sorted already
			
			test log: 
		"""
		
		# assert times sorted to be increasing in time
		T = np.array( t , dtype=np.float64 ).flatten()
		if not sorted : T = np.sort( T , axis=0 )
		
		# set size (if we're here)
		N = T.shape[0]
		
		# initialize result storage and indexers used below
		v = np.zeros( (N,) , dtype=self._t )
		i = 0 
		j = i
		
		# start with times ahead of this TSO
		try : 
			while T[i] <= self._T[j] : 
				v[i] = self._V[j]
				i += 1
		except IndexError : pass # ran out of times in arg, time to return
		else : # times in or after this TSO
			
			try : # now iterate over i AND j
				# on entry, self._T[0] == self._T[j] < T[i]
				
				while T[i] <= self._T[self._N-1] : 
				
					try: 
						while self._T[j] < T[i] : j += 1 # throws IndexError when j too large for this TSO
					except IndexError : pass # j >= _N so _T[_N-1] < T[i], so (outer) loop will end if we pass
					else : # (inner) while loop terminated without failure
						# if we're here, self._T[j-1] < T[i] <= self._T[j]
						if self._T[j] == T[i] : v[i] = self._V[i]
						else : # interpolate, self._T[j-1] < T[i] < self._T[j]
							# (NOTE: this conditional should probably be replaced with a virtual method implemented in subclasses)
							# (NOTE: methods should also be generalizable
							if   self.isBoolean()     : v[i] = self._V[j-1] # feed forward
							elif self.isCategorical() : v[i] = self._V[j-1] # feed forward
							elif self.isFloat()       : # linear interpolation
								v[i] = self._V[j-1] + (self._V[j]-self._V[j-1]) * (t[i]-self._T[j-1]) / (self._T[j]-self._T[j-1])
								
							else : raise ValueError( 'Unknown data type' )
						i += 1 # push i forward
				
			except IndexError : pass # ran out of times in arg (i got too large; j error caught in above)
			else : # remaining times after this TSO, which exist because (outer) loop above terminated before i got out of bounds
				try : 
					while T[i] >= self._T[self.N-1] : 
						v[i] = self._V[self._N-1]
						i += 1
				except IndexError : pass # ran out of times in arg
				# else : done
		
		# return scalars as scalars, arrays as arrays
		return v[0] if N == 1 else v
	
	def rinterp( self , v , time=None ) : 
		
		""" reverse interpolate as a set of values starting from time t 
			more specifically, return as t[i] the FIRST occurence of v[i]
			starting from time (according to the implied model). 
			
			test log: 
		"""
		
		# check that values are of same type? might want a way to avoid
		# lots of conditionals if we're evaluating this alot
		
		# if t == None, start from internal time "pointer" from last use
		# this internal pointer, _t, is stored in index form not 
		
		return
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def integrate( self , s=None , e=None , tol=1.0e-4 , freqlimit=True ) :
	
		""" integrate signal from time max(s,start()) to time min(e,end()) 
		
			Our approach is (currently) to just run adaptive quadrature with interpolation
		
			test log: 
		"""
		
		if s is None and e is None : 
			i = np.trapz( self._V[0:self._N] , x=self._T[0:self._N] )
			
		else : 
			# if s is None : s = self._T[0]
			# if e is None : e = self._T[self._N-1]
			I = self.findTimes( [s,e] )
			j = I[0]
			k = I[1]
			i = np.trapz( self._V[I[0]:I[1]+1] , x=self._T[I[0]:I[1]+1] )
			if s < self._T[j] : pass # finish integral with interpolant
			if e > self._T[k] : pass # finish integral with interpolant
		
		# d = self._T[0]
		# self._T = self._T - d
		
		# if freqlimit : i = int_asr( self.interp , 0 , e-d , tol , cluster=True , sfreq=10.0 )
		# else : i = int_asr( self.interp , 0 , e-d , tol , cluster=True )
		
		return i
		
		# return int_asr( self.interp , s , e , tol , cluster=True )
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		
	def sort( self , time=True , ascending=True ) : 
	
		""" sort data ascending/descending in time or value
		
			test log: 
		"""
		
		if time : 
			if ascending : i = np.argsort( self._T[0:self._N] ).tolist()
			else : i = np.argsort( self._T[0:self._N] )[::-1].tolist()
		else : 
			if ascending : i = np.argsort( self._V[0:self._N] ).tolist()
			else : i = np.argsort( self._V[0:self._N] )[::-1].tolist()
		
		self._V[0:self._N] = self._V[i]
		self._T[0:self._N] = self._T[i]
		
		return 
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	# value statistics
	def mean( self ) 	 : return np.mean( self._V )
	def std( self ) 	 : return np.std( self._V )
	def median( self ) 	 : return np.median( self._V )
	def max( self ) 	 : return np.max( self._V )
	def min( self ) 	 : return np.min( self._V )
	
	# time "statistics"
	def start( self )  	 : return self._T[0]
	def end( self ) 	 : return self._T[self._N-1]
	def duration( self ) : return ( self.end() - self.start() )
	
	def iat( self ) : return self._T[1:self._N] - self._T[0:self._N-1]
	
	def stats( self ) : 
		""" placeholder for a function to return multiple statistics
		"""
		# s = np.zeros( (5,) , dtype=np.float64 )
		s = [ 0 , 0 , 0 , 0 , 0 , 0 ]
		i = 0
		
		s[i] = self._N # count
		i += 1
		
		s[i] = self.start() # start time, here as seconds
		i += 1
		
		s[i] = self.end() # end time, here as seconds
		i += 1
		
		s[i] = s[i-1] - s[i-2] # duration (seconds)
		i += 1
		
		s[i] = float(self._N)/s[i-1] # signals per second (i.e., Hz)
		i += 1
		
		d = self.iat()
		s[i] = np.mean( d ) # mean interarrival time, in seconds
		i += 1
		
		s[1] = pd.to_datetime( 10**9 * s[1] ) # convert start time to a datatime object
		s[2] = pd.to_datetime( 10**9 * s[2] ) # convert  end  time to a datatime object
		return s
	
	def hist( self , B=10 ) : 
		""" placeholder to build and return a histogram of the values in this TSO use B bins
		"""
		return
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def plot( self , fromzero=False , toone=False ) : 
		""" (simple) time series plot wrapper """
		# plot( self._T , self._V )
		if fromzero : 
			if toone : return ( ( self._T[0:self._N] - self._T[0] ) / ( self._T[self._N-1] - self._T[0] ) , self._V[0:self._N] )
			else : return ( self._T[0:self._N] - self._T[0] , self._V[0:self._N] )
		else : return ( self._T[0:self._N] , self._V[0:self._N] )
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		
	def print( self ) : 
		
		""" basic print functionality; print information about this TSO 
		
			test log: 
		"""
		
		if   self.isBoolean()     : s = 'boolean    ' 
		elif self.isCategorical() : s = 'categorical' 
		elif self.isFloat()       : s = 'float      ' 
		else : pass
		
		s = '%s\t%i' % (s,self._N) # add number of data
		s = '%s\t%0.2f' % (s,self.start()) # add start time
		s = '%s\t%0.2f' % (s,self.end()) # add end time
		s = '%s\t%0.2f' % (s,self.duration()) # add duration
		s = '%s\t%0.6f' % (s,float(self._N)/self.duration()) # aggregate signal rate
		d = self._T[1:self._N] - self._T[0:self._N-1]
		s = '%s\t%0.6f' % (s,np.mean(d)) # avg interarrival time
		
		return s
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
class OXCTSC : 
	
	""" # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	
	OXCTSC: A class for Time Series Collections; ie sets of OpenXC TSO's
	
	This is implemented basically a dictionary whose values are TSO's
	
	Methods are provided to access and evaluate the underlying series 
	based on keys.  
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # """ 
	
	def __init__( self , f=None , token=None ) : 
		"""  """
		
		self._token = token
		
		if f is not None : 
			self.ImportFromJSON( self , f )
		else : 
			self._F = {}
			
	def __getitem__( self , name ) : 
		""" slicing syntax implementation : return TSO corresponding to name"""
		return self._F[name]
		
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		
	def addField( self , n ) :
		
		if n in self._F :
			raise ValueError( '%s already a field in this TSC' % n )
		else : 
			self._F[n] = OXCTSO(  ) # blank initialization
	 
	def isField( self , n ) : 
		return True if n in self._F else False
		
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		
	def importFromJSON( self , f ) : 
		
		""" import OXC-like streaming data from JSON 
		
			test log: 
		"""
		
		fp = open( f , mode='r' )
		try:
			
			# get name value dictionary and occurence counts
			D = {}
			for line in fp :
				j = json.loads( line )
				if 'name' in j and 'value' in j and 'timestamp' in j : 
					j['value'].lower()
					if j['name'] in D : 
						D[j['name']][0] += 1
						try : f = float( j['value'] ) 
						except Exception : # not a float, so categorical (by default)
							if j['value'] in D[j['name']][1] : D[j['name']][1][j['value']] += 1
							else : D[j['name']][1][j['value']] = 1
					else : 
						D[j['name']] = [ 1 , {} ]
						try : f = float( j['value'] ) 
						except Exception : # not a float, so categorical
							D[j['name']][1][j['value']] = 1
				else : pass
				
			# rewind file now that we have counts
			fp.seek(0,0)
			
			# allocate space for internal dictionary of TSO's
			self._F = {}
			for n in D : 
				if len( D[n][1] ) == 0 : self._F[n] = OXCTSO( type='f' , size=D[n][0] ) # 0 , np.empty( (D[n][0],) , dtype=np.float64 ) ,  np.empty( (D[n][0],) , dtype=np.float64 )
				else : 
					if len( D[n][1] ) <= 2 : # possibly boolean
						if 'true' in D[n][1] or 'false' in D[n][1] : self._F[n] = OXCTSO( type='b' , size=D[n][0] ) 
						else : self._F[n] = OXCTSO( type='c' , size=D[n][0] )
					else : self._F[n] = OXCTSO( type='c' , size=D[n][0] )
				if self._F[n].isCategorical() : # make value mapping (declared empty above)
					self._F[n].makeValueMapping( D[n][1] )
			
			# re-read and parse
			for line in fp : 
				j = json.loads( line )
				if 'name' in j and 'value' in j and 'timestamp' in j : 
					if self._F[j['name']].isBoolean() : v = True if j['value'].lower()[0] == 't' else False
					elif self._F[j['name']].isCategorical() : 
						try : v = self._F[j['name']].mapValue( j['value'] )
						except KeyError : print( line )
					elif self._F[j['name']].isFloat() : v = float( j['value'] )
					else : 
						print( self._F[j['name']]._t )
						raise ValueError( 'unknown type (%s)' % line )
					# using append here, sorting later. That's faster in bulk than OXCTSO.set( ... )
					self._F[j['name']].append( float( j['timestamp'] ) , v )
					
			# now that we have read everything, and we simply appended, enforce sort (ascending in time by default)
			self.sort()
				
		finally:
			# always close your files
			fp.close() 
		
		return
		
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def importFromCSV( self , f ) : 
		
		""" import data from a csv file; not yet implemented
		
			test log: 
		"""
		
		raise NotImplementedError( 'sorry, TBD' )
		
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		
	def importFromDataFrame( self , DF , time=None ) : 
		
		""" import data from a DataFrame 
			
			if time is None, expects index to be timestamp. otherwise, time should be the name of
			the column of DF containing times
			
			also expects column headers to be appended with types like
			
				data_element -> data_element(x)
				
			where "x" can be b/B (boolean), c/C (categorical), f/F (float)
		
			test log: 
		"""
		
		if DF is None or not isinstance( DF , pd.DataFrame ) : 
			raise ValueError( 'import expects a DataFrame object' )
		
		if time is not None : 
			try : DF[time]
			except Expection as e : 
				raise ValueError( 'time column name passed but not in DataFrame (%s)' % e )
		
		# get column names
		H = list( DF )
		self._F = {}
		for n in H : 
			
			c = DF[n].dropna() # get column, only non nan values
			N = c.shape[0] # assess size
			
			if n[-3] == '(' and n[-1] == ')' : 
				
				# real name
				s = n[0:-3]
				# initialize and allocate
				if   n[-2] == 'b' or [-2] == 'B' : self._F[s] = OXCTSO( type='b' , size=N ) 
				elif n[-2] == 'c' or [-2] == 'C' : self._F[s] = OXCTSO( type='c' , size=N ) 
				elif n[-2] == 'f' or [-2] == 'F' : self._F[s] = OXCTSO( type='f' , size=N ) 
				else : raise ValueError( 'unknown data type code (%c) in column %s' % (n[-2],s) )
				self._F[s]._N = N # assign size
				self._F[s]._V = c.values.copy() # assign values
			
				# now assign times
				t = c.index if time is None else DF[time].values
				try : self._F[s]._T = t.copy().astype( np.float64 )
				except Exception : self._F[s]._T = t.copy().astype( np.int64 ) / 10**9
				
			else : 
				
				print( 'WARNING: no (parseable) type code in DataFrame column name (%s), defaulting to float' % n )
			
				self._F[n] = OXCTSO( type='f' , size=N ) # initialize and allocate
				self._F[n]._N = N # assign size
				self._F[n]._V = c.values.copy() # assign values
			
				# now assign times
				t = c.index if time is None else DF[time].values
				try : self._F[n]._T = t.copy().astype( np.float64 )
				except Exception : self._F[n]._T = t.copy().astype( np.int64 ) / 10**9
			
		return
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def appendDataFrame( self , F ) :
		
		""" append elements from a DataFrame; not yet implemented
		
			test log: 
		"""
		
		raise NotImplementedError( 'sorry, TBD' )
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def extractByTime( self , s , e ) :
		
		""" extract a OXCTSC with data in between two times 
		
			test log: 
		"""
		
		R = OXCTSC(  ) 
		R._F = {}
		for n in self._F : 
			R._F[n] = self._F[n].extractByTime( s , e )
		
		return R
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		
	def sort( self ) : 
		""" TSO sort wrapper 
		
			test log: 
		"""
		for n in self._F : self._F[n].sort()
		return
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def tripize( self , 		 # self object...
				 sort=False , 	 # force sort, ascending in time (in place)
			     indices=False , # return indices only if true, not list of split collections
			     thresh=0 , 	 # provide this to generate an initial guess (or result) based on a seconds threshold
			     stats=True 	 # return stats object 
				) :
		
		""" attempt to turn this object into trips using interarrival times 
			
			test log: 
		"""
		
		# create full time (index) vector
		N = 0 
		for n in self._F : N += self._F[n]._N
	
		t = np.zeros( (N,) , dtype='float64' )
		N = 0 
		for n in self._F : 
			t[ N : N + self._F[n]._N ] = self._F[n]._T[0:self._F[n]._N]
			N += self._F[n]._N
		
		# unique ones only
		t = np.unique( t )
		
		# get (successive) time differences
		d = t[1:] - t[0:-1]
		
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		# In here you could generalize the method used; possibly an optional method arg # # # #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
		
		# look for "extraordinary" time differences
		if thresh > 0 : I = np.where( d > thresh )[0] # index element 0 as a tuple is returned
		else : pass
		
		# do anything else here? metrics and merging? further splitting?
		
		# stats for time differences in between splits
		if stats : 
			l = 0
			S = []
			for i in I : 
				S.append( { 'start' : pd.to_datetime( 10**9 * t[l] ) , 'end' : pd.to_datetime( 10**9 * t[i] ) , 'duration' : t[i]-t[l] , 'count' : i-l , 'min' : d[l:i].min() , 'mean' : d[l:i].mean() , 'std' : d[l:i].std() , 'med' : np.median(d[l:i]) , 'max' : d[l:i].max() } )
				l = i+1
			if l < t.shape[0]-1 :
				S.append( { 'start' : pd.to_datetime( 10**9 * t[l] ) , 'end' : pd.to_datetime( 10**9 * t[-1] ) , 'duration' : t[-1]-t[l] , 'count' : t.shape[0]-l ,  'min' : d[l:].min() , 'mean' : d[l:].mean() , 'std' : d[l:].std() , 'med' : np.median(d[l:]) , 'max' : d[l:].max() } )
		
		# finish up
		if indices : 
			if stats : return ( I , S )
			else : return I
		else :
			Fp = []
			l = min( 0 , I[0] )
			for i in I : 
				if i > l : 
					Fp.append( self.extractByTime( t[l] , t[i] ) )
					l = i+1
			if l < t.shape[0]-1 :
				Fp.append( self.extractByTime( t[l] , None ) )
			
			if stats : return ( Fp , S )
			else : return Fp
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def exportToDataFrame( self ) : 
		
		""" turn openXC streaming-style JSON into a time series DataFrame, indices are timestamp
			
			test log: 
		"""
		
		# create full time (index) vector
		N = 0 
		for n in self._F : N += self._F[n]._N
	
		t = np.zeros( (N,) , dtype='float64' )
		N = 0 
		for n in self._F : 
			t[ N : N + self._F[n]._N ] = self._F[n]._T[0:self._F[n]._N]
			N += self._F[n]._N
		
		i = np.unique( t ) # indices are unique times
		c = []
		for n in self._F : 
			c.append( '%s(%c)' % ( n , 'b' if self._F[n].isBoolean() else ( 'c' if self._F[n].isCategorical() else 'f' ) ) )
		N = i.shape[0]
		d = np.nan * np.ones( (N,len(self._F)) , dtype=np.float64 )
		
		DF = pd.DataFrame( data=d , index=i , columns=c ) 
	
		i = 0
		for n in self._F : 
			if   self._F[n].isBoolean()     : DF.loc[ self._F[n]._T[0:self._F[n]._N] , c[i] ] = self._F[n]._V[0:self._F[n]._N].astype( np.float64 )
			elif self._F[n].isCategorical() : DF.loc[ self._F[n]._T[0:self._F[n]._N] , c[i] ] = self._F[n]._V[0:self._F[n]._N].astype( np.float64 )
			elif self._F[n].isFloat() 	    : DF.loc[ self._F[n]._T[0:self._F[n]._N] , c[i] ] = self._F[n]._V[0:self._F[n]._N]
			else : raise ValueError( 'Unknown data type code' )
			i += 1
		
		DF = DF.set_index( pd.to_datetime( 10**9 * DF.index ) )
		DF.sort_index( axis=0 , inplace=True , na_position='first' )
		
		return DF
		
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def signalStats( self ) :
	
		""" create (and return) a dataframe with signal stats 
			
			test log: 
		"""
		
		i = [ 'count','start','end','duration','sig freq','avg iat' ]
		E = len(i)
		
		DF = pd.DataFrame( data=np.zeros((len(self._F),E)) , 
							index=list(self._F.keys()) , 
							columns=i )
							# data=np.zeros((E,len(self._F))) , 
							# index=i , 
							# columns=list(self._F.keys()) )
		for n in self._F : 
			if self._F[n] is None : s = np.nan * np.ones( (E,) )
			else : s = self._F[n].stats()
			DF.loc[n,:] = s
		
		# DF.transpose()
	
		return DF
		
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def print( self ) : 
		
		for n in self._F : 
			print( n , '\n\t%s' % self._F[n].print() )
			
		
		return
		
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	def plot( self , col=None , fromzero=False , toone=False ) : 
		
		""" plot functionality 
			
			test log: 
		"""
		
		L = []
		if col is None: 
			for n in self._F : 
				if self._F[n] is not None : 
					L.append( self._F[n].plot( fromzero , toone ) )
		else : 
			for n in iter(col) : 
				if n in self._F : 
					if self._F[n] is not None : 
						L.append( self._F[n].plot( fromzero , toone ) )
					else :
						L.append( ([0],[0]) )
		
		return L
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
			 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def tripizeDataFrame( F , # dataframe to split
			   sort=True , # force sort, not in place, ascending in index (time)
			   indices=False , # return indices only if true, not list of split dataframes
			   thresh=0 , # provide this to generate an initial guess (or result) based on a seconds threshold
			   stats=True # return stats object 
			 ) : 
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	""" 
		"split" time series dataframes based on statistics of the data
		interarrival times. That is, the 1-row differences in times when sorted in
		ascending order (by time)
		
		expects a dataframe, F, whose indices are valid times (in the sense that
		casting the indices as datetime indices and then int64's gives UNIX time-
		stamps in nanoseconds). 
		
	"""
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	# make sure (or try to) that the dataframe is sorted increasing in index
	if sort : f = F.sort_index( axis=0 , inplace=False , na_position='first' )
	else : f = F
	
	# get (successive) time differences
	t = pd.to_datetime( f.index ).astype(np.int64) / 10 ** 9 # UNIX times, as seconds
	d = t[1:] - t[0:-1] # successive time differences (lags)
	
	# hack validity check: >>> print( pd.to_datetime( 10 ** 9 * t ) )
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# In here you could generalize the method used; possibly an optional method arg # # # #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	# look for "extraordinary" time differences
	if thresh > 0 : I = np.where( d > thresh )[0] # index element 0 as a tuple is returned
	else : pass # what to do with no threshold?
	
	# do anything else here? metrics and merging? further splitting?
	
	# stats for time differences in between splits
	if stats : 
		l = 0
		S = []
		for i in I : 
			S.append( { 'start' : pd.to_datetime( 10**9 * t[l] ) , 'end' : pd.to_datetime( 10**9 * t[i] ) , 'duration' : t[i]-t[l] , 'count' : i-l , 'min' : d[l:i].min() , 'mean' : d[l:i].mean() , 'std' : d[l:i].std() , 'med' : np.median(d[l:i]) , 'max' : d[l:i].max() } )
			l = i+1
		if l < f.shape[0]-1 :
			S.append( { 'start' : pd.to_datetime( 10**9 * t[l] ) , 'end' : pd.to_datetime( 10**9 * t[-1] ) , 'duration' : t[-1]-t[l] , 'count' : f.shape[0]-l ,  'min' : d[l:].min() , 'mean' : d[l:].mean() , 'std' : d[l:].std() , 'med' : np.median(d[l:]) , 'max' : d[l:].max() } )
		
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	# finish up
	if indices : 
		if stats : return ( I , S )
		else : return I
	else :
		Fp = []
		l = min( 0 , I[0] )
		for i in I : 
			if i > l : 
				Fp.append( f.iloc[l:i] )
				l = i+1
		if l < f.shape[0]-1 :
			Fp.append( f.iloc[l:] )
		if stats : return ( Fp , S )
		else : return Fp
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
