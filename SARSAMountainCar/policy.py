'''
Created on Apr 2, 2016

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)
'''
import numpy as np
import scipy.stats

class BoltzmannPolicy( object ):

    def __init__( self, temperature ):
        self.__temperature = temperature

    def selectionProbabilities( self, actionValues ):
        actionValues = np.array( actionValues, copy=True )
        actionValues /= self.__temperature
        actionValues = np.exp( actionValues )
        mass = np.sum( actionValues )
        if mass > 0:
            actionValues /= mass
        else:
            actionValues = np.ones( len( actionValues ) ) / float( len( actionValues ) )
        return actionValues

    def sampleAction( self, actionValues=None, actionSpace=None, actionProb=None ):
        if actionProb is None and actionValues is None:
            raise Exception( 'Cannot sample action without action values or action probabilities' )
        elif actionProb is None and not actionValues is None:
            actionProb = self.selectionProbabilities( actionValues )
        if actionSpace is None:
            actionSpace = np.array( range( len( actionProb ) ) )
        rv = scipy.stats.rv_discrete( name='act_dist', values=( range( len( actionProb ) ), actionProb ) )
        actionInd = rv.rvs( size=1 )[0]
        return actionSpace[actionInd]

