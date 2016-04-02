'''
Created on Apr 2, 2016

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)
'''
import numpy as np

class SARSA( object ):

    def __init__( self, alpha, traceLambda, pi, phi, initTheta, actionSpace, gamma ):
        self.__alpha = alpha
        self.__lambda = traceLambda
        self.__pi = pi
        self.__phi = phi
        
        self.__theta = np.array( initTheta, copy=True )
        self.__e = np.zeros( len( self.__theta ) )
        
        self.__nextAction = None
        self.__actionSpace = actionSpace
        self.__gamma = gamma


    def __sampleAction( self, state ):
        qAr = np.array( [ np.dot( self.__theta, self.__phi( state, a ) ) for a in self.__actionSpace ] )
        self.__nextAction = self.__pi.sampleAction( actionValues=qAr, actionSpace=self.__actionSpace )

    def getNextAction( self, state ):
        if self.__nextAction is None:
            self.__sampleAction( state )
        return self.__nextAction

    def update( self, state, action, reward, stateNext ):
        self.__sampleAction( stateNext )
        phi_t = self.__phi( state, action )
        phi_tn = self.__phi( stateNext, self.__nextAction )
        
        delta_t = reward + self.__gamma * np.dot( self.__theta, phi_tn ) - np.dot( self.__theta, phi_t )
        
        self.__e = self.__e + phi_t
        self.__theta += self.__alpha * delta_t * self.__e
        self.__e = self.__gamma * self.__lambda * self.__e

    def getTheta( self ):
        return np.array( self.__theta, copy=True )
