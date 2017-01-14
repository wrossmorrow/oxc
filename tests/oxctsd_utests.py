import unittest
import numpy as np
import pandas as pd

# these import commands enable interactive work while developing the module itself
import sys
# sys.path.append( "C:\\Users\\William\\Documents\\Work\\Software\\python\\oxc\\src" ) # for straight windows
sys.path.append( "/mnt/c/Users/William/Documents/Work/Software/python/oxc/src" ) # for ubuntu bash on windows
import oxctsd as oxc
import importlib

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def passmein( func ) :
	def wrapper( *args , **kwargs ) : 
		return func( func , *args , **kwargs )
	return wrapper
	
class OXCTSDTestCase( unittest.TestCase ) :
	
	""" Tests for oxctsd module """
	
	@passmein
	def test_00( me , self ) : 
		
		""" testing custom version of a memory shift routine... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		x = np.array( list(range(10)) , dtype=np.float_ )
		
		try : 
			
			print( 'mem shift test...   original' , x )
			
			i , j , n = 2 , 5 , 1
			y = x.copy()
			oxc.stupidMemShift( y , i , j , n ) # , usenp=True )
			print( 'shift [%i,%i) by %i to the right' % (i,j,n) , y )
			
			y = x.copy()
			oxc.stupidMemShift( y , i , j , -n ) # , usenp=True )
			print( 'shift [%i,%i) by %i to the right' % (i,j,n) , y )
		
		except Exception as e : self.assertTrue( False )
		else : self.assertTrue( True )
		
	@passmein
	def test_01( me , self ) :
		
		""" testing that OXCTSC empty constructor runs... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		D = oxc.OXCTSC()
		self.assertTrue( True )
	
	@passmein
	def test_02( me , self ) :
		
		""" testing OXCTSC add field spec works... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		D = oxc.OXCTSC()
		n = 'a field'
		D.addField(n)
		self.assertTrue( D.isField(n) )
		
	@passmein
	def test_03( me , self ) :
		
		""" testing OXCTSC field slice syntax... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		D = oxc.OXCTSC()
		n = 'a field'
		D.addField(n)
		f = D[n] # how to evaluate? is the below appropriate?
		self.assertTrue( f == D._F[n] )
	
	@passmein
	def test_04( me , self ) :
		
		""" testing OXCTSC add field spec and subsequent population works (single element)... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		D = oxc.OXCTSC()
		n = 'a field'
		D.addField(n)
		try : D[n].setByTime( 1.0 , 0.0 , add=True )
		except Exception : self.assertTrue( False )
		else : self.assertTrue( True )
	
	@passmein
	def test_05( me , self ) :
		
		""" testing OXCTSC add field spec and subsequent population fails (mismatched v/t sizes)... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		D = oxc.OXCTSC()
		n = 'a field'
		D.addField(n)
		try : D[n].setByTime( [2.0,3.0] , 0.0 , add=True )
		except Exception as e : self.assertTrue( True )
		else : self.assertTrue( False )
	
	@passmein
	def test_06( me , self ) :
		
		""" testing OXCTSC add field spec and subsequent population works (lists)... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		D = oxc.OXCTSC()
		n = 'a field'
		D.addField(n)
		try : D[n].setByTime( [2.0,3.0] , [0.0,0.0] , add=True )
		except Exception as e : self.assertTrue( False )
		else : self.assertTrue( True )
	
	@passmein
	def test_07( me , self ) :
		
		""" testing OXCTSC add field spec and subsequent population works (numpy arrays)... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		D = oxc.OXCTSC()
		n = 'a field'
		D.addField(n)
		try : D[n].setByTime( np.array( [2.0,3.0] ) , np.array( [0.0,0.0] ) , add=True )
		except Exception as e : self.assertTrue( False )
		else : self.assertTrue( True )
	
	@passmein
	def test_08( me , self ) :
	
		""" testing OXCTSC add field spec and subsequent population works (two adds, insert before)... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		D = oxc.OXCTSC()
		n = 'a field'
		D.addField(n)
		try : 
			D[n].setByTime( [2.0,3.0] , [2.0,3.0] , add=True )
			print( np.array( [D[n]._T[0:D[n]._N],D[n]._V[0:D[n]._N]] ) )
			D[n].setByTime( 1.0 , 1.0 , add=True )
			print( np.array( [D[n]._T[0:D[n]._N],D[n]._V[0:D[n]._N]] ) )
		except Exception as e : 
			self.assertTrue( False )
		else : self.assertTrue( True )
		
	@passmein
	def test_09( me , self ) :
	
		""" testing OXCTSC add field spec and subsequent population works (two adds, insert after)... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		D = oxc.OXCTSC()
		n = 'a field'
		D.addField(n)
		try : 
			D[n].setByTime( [2.0,3.0] , [2.0,3.0] , add=True )
			print( np.array( [D[n]._T[0:D[n]._N],D[n]._V[0:D[n]._N]] ) )
			D[n].setByTime( 4.0 , 4.0 , add=True )
			print( np.array( [D[n]._T[0:D[n]._N],D[n]._V[0:D[n]._N]] ) )
		except Exception as e : 
			self.assertTrue( False )
		else : self.assertTrue( True )
	
	@passmein
	def test_10( me , self ) :
	
		""" testing OXCTSC add field spec and subsequent population works (two adds, mixed values)... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		D = oxc.OXCTSC()
		n = 'a field'
		D.addField(n)
		try : 
			D[n].setByTime( [2.0,3.0] , [0.0,0.0] , add=True )
			print( np.array( [D[n]._T[0:D[n]._N],D[n]._V[0:D[n]._N]] ) )
			D[n].setByTime( [1.0,2.0,2.5,4.0] , [1.0,1.0,1.0,1.0] , add=True )
			print( np.array( [D[n]._T[0:D[n]._N],D[n]._V[0:D[n]._N]] ) )
		except Exception as e : 
			print( e )
			self.assertTrue( False )
		else : self.assertTrue( True )
	
	@passmein
	def test_11( me , self ) :
	
		""" testing OXCTSC add field spec and subsequent population works (assign only, mixed values)... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		D = oxc.OXCTSC()
		n = 'a field'
		D.addField(n)
		try : 
			D[n].setByTime( [2.0,3.0] , [0.0,0.0] , add=True )
			print( np.array( [D[n]._T[0:D[n]._N],D[n]._V[0:D[n]._N]] ) )
			D[n].setByTime( [1.0,2.0,2.5,4.0] , [1.0,1.0,1.0,1.0] , add=False )
			print( np.array( [D[n]._T[0:D[n]._N],D[n]._V[0:D[n]._N]] ) )
		except Exception as e : 
			print( e )
			self.assertTrue( False )
		else : self.assertTrue( True )
		
	@passmein
	def test_11( me , self ) :
	
		""" testing OXCTSC integration capability... """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		# set up linear values : 
		# 	
		# 	v( t ) = a t + b, int_s^e v(t)dt = a/2 ( e^2 - s^2 ) + b ( e - s )
		# 
		# 
		
		a , b = 1.0 , 12.0
		D = oxc.OXCTSC()
		n = 'a field'
		D.addField(n)
		try : 
			t = np.random.rand( 10 )
			t = np.sort( t , axis=0 )
			s , e = np.min(t) , np.max(t)
			v = a * t + b
			D[n].setByTime( t , v , add=True )
			i = D[n].integrate()
			I = 0.5 * a * ( e**2 - s**2 ) + b * ( e - s )
		except Exception as e : 
			print( e )
			self.assertTrue( False )
		else : self.assertTrue( abs( i - I ) <= 1.0e-6 )
		
	
# 	@passmein
#	def test_DESCRIPTIVE_NAME_HERE( me , self ) :
#		""" DESCRIPTIVE DOCSTRING HERE """
#		print( me.__doc__ )
#		self.assertTrue( True )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		
if __name__ == '__main__':
	unittest.main()