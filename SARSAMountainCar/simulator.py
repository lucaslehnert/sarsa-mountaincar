'''
Created on Apr 2, 2016

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)
'''

def simulateEpisode( mdp, pi, agent, maxTransitions=10000 ):
    '''
    Simulates an episode and returns the number of update steps the agent needs.
    '''
    s = mdp.getStartState()
    episodeEnd = False
    updateStep = 0

    while not episodeEnd:
        a = agent.getNextAction( s )
        snext = mdp.sampleNextState( s, a )
        reward = mdp.getReward( s, a, snext )

        agent.update( s, a, reward, snext )

        s = snext
        updateStep += 1
        if mdp.isGoalState( s ) or updateStep == maxTransitions:
            episodeEnd = True

    return updateStep

