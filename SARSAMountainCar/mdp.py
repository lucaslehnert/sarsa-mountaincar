'''
Created on Apr 2, 2016

@author: Lucas Lehnert (lucas.lehner@mail.mcgill.ca)
'''

class MDP( object ):
    '''
    Implements a small MDP class.
    '''

    def __init__( self, actionSpace, nextStateSampler, rewardFunction, gamma, startState, isGoalState ):
        self.__actionSpace = actionSpace
        self.__nextStateSampler = nextStateSampler
        self.__rewardFunction = rewardFunction
        self.__gamma = gamma
        self.__startState = startState
        self.__isGoalState = isGoalState

    def getStartState( self ):
        return self.__startState

    def sampleNextState( self, state, action ):
        return self.__nextStateSampler( state, action )

    def getReward( self, state, action, nextState ):
        return self.__rewardFunction( state, action, nextState )

    def isGoalState( self, state ):
        return self.__isGoalState( state )

    def getStateSpace( self ):
        return self.__discretizedStateSpace

    def getActionSpace( self ):
        return self.__actionSpace

    def getGamma( self ):
        return self.__gamma