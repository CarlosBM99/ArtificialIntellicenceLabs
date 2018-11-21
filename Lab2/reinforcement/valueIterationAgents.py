# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #Assign the values in a new variable
        self.oldValues = self.values.copy()
        states = mdp.getStates()
        #For every iteration we check every state and the possible actions
        for iteration in range(iterations):
            for state in states:
                actions = mdp.getPossibleActions(state)
                #While this state is not the last we look for the best action value
                if not mdp.isTerminal(state):
                    #Set Qvalue from the current state
                    self.oldValues[state] = self.getQValue(state, self.getAction(state))
            self.oldValues = self.values.copy()



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #Initial value to 0 (Has not started, no rewards, no points)
        q_value = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            #For every transition we upgrade the q_value. We store there the cost of staying alive one more time per the
            #cost of the new movement plus geting the reward. All of this multiplied by the probability of this event to 
            #happen.
            q_value += prob * (self.discount * self.oldValues[nextState] + self.mdp.getReward(state, action, nextState))
        #Return the total q_value    
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #We first take the possibles actions that we can do
        actions = self.mdp.getPossibleActions(state)
        #Set the value to a negative number so it can be easily updated
        maxReward = -999999 
        #In the beginning there is no decision made
        decision = None
        #We check the possible actions and then compare it with the maxReward
        for action in actions:
            actionReward = self.computeQValueFromValues(state, action)
            if actionReward > maxReward:
                #If the reward is bigger than the former maximum, now this is the new maximum and the
                #final decision is to take this action
                maxReward = actionReward
                decision = action
        #We return the actions with the best reward        
        return decision

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
