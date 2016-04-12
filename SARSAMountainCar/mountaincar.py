'''
Created on Apr 2, 2016

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)
'''
import numpy as np
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
import json

from mdp import MDP
from policy import BoltzmannPolicy
from simulator import simulateEpisode
from sarsa import SARSA

def bound( x, m, M ):
    return min( max( x, m ), M )

positionMin = -1.2
positionMax = 0.6
velocityMin = -0.07
velocityMax = 0.07

def createMountainCarMDP():
    global positionMin, positionMax, velocityMin, velocityMax

    positionGoal = 0.5

    ''' backward, neutral, forward '''
    actionSpace = np.array( [-1, 0, 1] )

    def transitionSampler( state, action ):
        pos = state[0]
        vel = state[1]

        pos = pos + vel
        pos = bound( pos, positionMin, positionMax )
        vel = vel + action * 0.001 - 0.0025 * np.cos( 3 * pos )
        vel = bound( vel, velocityMin, velocityMax )

        staten = np.array( [ pos, vel ] )
        return staten

    isGoalState = lambda s : s[0] >= positionGoal

    def rewardFunction( state, action, staten ):
        if isGoalState( staten ):
            return 0.0
        else:
            return -1.0

    startState = np.array( [ -0.5, 0.0 ] )

    gamma = 1.0

    statePosRange = np.linspace( positionMin, positionMax, 19 )
    stateVelRange = np.linspace( velocityMin, velocityMax, 11 )
    statePos, stateVel = np.meshgrid( statePosRange, stateVelRange )
    discretizedStateSpace = np.array( [statePos.flatten(), stateVel.flatten()], dtype=np.double ).T

    discretizedStartStateDistribution = np.zeros( len( discretizedStateSpace ) )
    startInd = np.where( np.all( discretizedStateSpace == np.array( [-0.5, 0.0] ), axis=1 ) )[0][0]
    discretizedStartStateDistribution[startInd] = 1.0

    return MDP( actionSpace, transitionSampler, rewardFunction, gamma, startState, isGoalState )

# def getBasisFunctionOld( mdp ):
#     return getTiledStateActionBasisFunction( mdp, np.array( [18, 18] ) )

def getBasisFunction( mdp, nmbrPositionTiles, nmbrVelocityTiles ):
    global positionMin, positionMax, velocityMin, velocityMax

    positionTileSize = ( positionMax - positionMin ) / nmbrPositionTiles
    velocityTileSize = ( velocityMax - velocityMin ) / nmbrVelocityTiles

    def phi( s, a ):
        actionTiles = np.zeros( len( mdp.getActionSpace() ) )
        actionInd = np.where( a == mdp.getActionSpace() )[0][0]
        actionTiles[actionInd] = 1.0

        positionTiles = np.zeros( nmbrPositionTiles )
        positionInd = int( np.floor( ( s[0] - positionMin ) / positionTileSize ) )
        positionTiles[positionInd] = 1.0

        velocityTiles = np.zeros( nmbrVelocityTiles )
        velocityInd = int( np.floor( ( s[1] - velocityMin ) / velocityTileSize ) )
        velocityTiles[velocityInd] = 1.0

        phiVector = np.outer( np.outer( positionTiles, velocityTiles ).flatten(), actionTiles ).flatten()
        return phiVector

    phiDim = nmbrPositionTiles * nmbrVelocityTiles * len( mdp.getActionSpace() )
    return phi, phiDim

def runExperiment( alpha=0.2, traceLambda=0.9, maxTransitions=10000, episodes=100, temperature=1.0, \
                   experimentJSON='experiment_results.json', experimentPlot='experiment_length.pdf' ):

    print 'Running SARSA on Mountain Car'
    print 'alpha= ' + str( alpha )
    print 'lambda= ' + str( traceLambda )
    print 'maxTransitions= ' + str( maxTransitions )
    print 'episodes= ' + str( episodes )
    print 'temperature= ' + str( temperature )

    mdp = createMountainCarMDP()
    phi, phiDim = getBasisFunction( mdp, 18, 18 )
    initTheta = np.zeros( phiDim )
    pi = BoltzmannPolicy( temperature=temperature )

    agent = SARSA( alpha, traceLambda, pi, phi, initTheta, mdp.getActionSpace(), mdp.getGamma() )

    episodeLengths = []
    for epInd in range( episodes ):
        print 'Running episode ' + str( epInd )
        epLen = simulateEpisode( mdp, pi, agent, maxTransitions=maxTransitions )
        episodeLengths.append( epLen )
        print '\tLength: ' + str( epLen )

    print 'Done experiment'
    print 'Episode length: ' + str( episodeLengths )

    experimentResults = { 'alpha' : alpha, 'lambda' : traceLambda, 'maxTransitions' : maxTransitions, \
                         'mdp' : 'Mountain Car', 'algorithm' : 'SARSA', 'episode_length' : episodeLengths }
    experimentStr = json.dumps( experimentResults )

    resultFile = open( experimentJSON, 'w' )
    resultFile.write( experimentStr )
    resultFile.close()
    print 'Saving json file to ' + str(experimentJSON)

    plt.plot( range( 1, episodes + 1 ), episodeLengths )
    plt.xlabel( 'Episode' )
    plt.ylabel( 'Episode Length' )
    plt.xlim( [0, episodes + 1] )
    plt.ylim( [0, 1.05 * maxTransitions] )
    plt.savefig( experimentPlot )
    print 'Saving plot file to ' + str(experimentPlot)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser( description='Mountaincar Experiments', \
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '-a', '--alpha', type=float, default=0.1, help='SARSA learning rate.' )
    parser.add_argument( '-l', '--trace-lambda', type=float, default=0.9, help='SARSA learning rate.' )
    parser.add_argument( '-t', '--temperature', type=float, default=0.2, help='Boltzmann temperature.' )
    parser.add_argument( '-e', '--episodes', type=int, default=200, help='Number of episodes.' )
    parser.add_argument( '-i', '--max-iterations', type=int, default=1000, \
                         help='Maximum number of iterations per episode.' )
    
    
    parser.add_argument( '-p', '--plot', type=str, default='experiment_length.pdf', \
                         help='Experiment plot file.' )
    parser.add_argument( '-r', '--results', type=str, default='experiment_results.json', \
                         help='Experiment result JSON file.' )
    args = parser.parse_args()

    alpha = args.alpha
    traceLambda = args.trace_lambda
    temp = args.temperature
    episodes = args.episodes
    iterations = args.max_iterations
    resultFile = args.results
    plotFile = args.plot

    runExperiment( alpha=alpha, traceLambda=traceLambda, maxTransitions=iterations, episodes=episodes, \
                   temperature=temp, experimentJSON=resultFile, experimentPlot=plotFile )
    pass
