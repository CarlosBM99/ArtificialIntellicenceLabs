# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #Initially the score is the score of the next position
        #we will modify this score depending the distance between
        #pac-man and the food/ghost 
        score = successorGameState.getScore()
        newGhostPos = successorGameState.getGhostPositions()
        #Checking ghosts
        for pos in newGhostPos:
          distToGhost = manhattanDistance(newPos, pos)
          if distToGhost != 0.0 and distToGhost <= 5:
            #Check if we have eaten something (killer mode)
            if newScaredTimes[0] == 0:
              #Using the inverse of the distance we can have high
              #values when the ghost is to close
              score = score - 1.0/distToGhost
            else:
              score = score + 1.0/distToGhost
        #Checking food
        closestFood = float("inf")
        for food in newFood.asList():
          distanceFood = manhattanDistance(newPos,food)
          #We only care about the closest food
          if closestFood > distanceFood:
            closestFood = distanceFood
        #Same strategy as before using the inverse    
        score = score + 1.0/closestFood

        return score    

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):

        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
    
    # Function in charge of decide who plays (it decides if we have to use min or max)
    def dispatch(self, gameState, action, turn, alpha, beta):

        # We get the current turn
        turn = (turn + 1) % (gameState.getNumAgents() * self.depth)   
        # First we check if the game is over or if it is the first turn
        if gameState.isLose() or gameState.isWin() or (turn == 0):
    		return self.evaluationFunction(gameState)
        # Get who plays now
        agentIndex = turn % gameState.getNumAgents()  
        # If 0 -> is pacman
        if agentIndex == 0:
            return self.value(gameState, turn, alpha, beta, 'pacman')
        # Otherwise -> ghost
        else:
            return self.value(gameState, turn, alpha, beta, 'ghost')

    def generalGetAction(self, gameState):

        # Init the valAction 
        valAction = util.Counter()
        # Init alpha as minus infinity
        alpha = float("-inf")
        # Init beta as plus infinity
        beta = float("inf")
        # Get all the possible (legal) actions that we can carry out in a pacman state
        # and iterate for every action
        for action in gameState.getLegalActions(self.index):
            # Assign the successor given an action
            successor = gameState.generateSuccessor(self.index, action)
            valAction[action] = self.dispatch( successor, action, self.index, alpha, beta )
            # Assign alpha as a max between the current value of alpha and the current val of that action
            alpha = max(alpha, valAction[action])
        # Return the key with the highest value
        return valAction.argMax()

class MinimaxAgent(MultiAgentSearchAgent):

    # Function that returns the value of pacman turn
    def value(self, gameState, turn, alpha, beta, element):
        
        # Get the agent index depending of the turn
        agentIndex = turn % gameState.getNumAgents()

        if element == 'pacman':
            # Init the maximum as minus infinity
            maximum = -float('inf')
            # Get all the possible (legal) actions that we can carry out at the position of the agent,
            # and iterate for every action
            for action in gameState.getLegalActions(agentIndex):
                # Get a candidate -> val
                val = self.dispatch( gameState.generateSuccessor(agentIndex, action), action, turn, 0, 0 )
                # Check if it is a maximum
                if val > maximum:
                    # Assing the val as max
                    maximum = val
            return maximum
        else: # else is a ghost
            # Init the minimum as plus infinity
            minimum = float('inf')
            # Get all the possible (legal) actions that we can carry out at the position of the agent,
            # and iterate for every action
            for action in gameState.getLegalActions(agentIndex):
                # Get a candidate -> val
                val = self.dispatch( gameState.generateSuccessor(agentIndex, action), action, turn, 0, 0 )
                # Check if it is a maximum
                if val < minimum:
                    # Assing the val as max
                    minimum = val 
            return minimum

    # Function that returns de action of the actual GameState
    def getAction(self, gameState):
        return self.generalGetAction(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):

    def value(self, gameState, turn, alpha, beta, element):

        # Get the agent index depending of the turn
        agentIndex = turn % gameState.getNumAgents()
        
        if element == 'pacman':
            # Init the val as minus infinity
            val = -float("inf")
            # Get all the possible (legal) actions that we can carry out at the position of the agent,
            # and iterate for every action
            for action in gameState.getLegalActions(agentIndex):
                # Assign the successor given an action
                successor = gameState.generateSuccessor(agentIndex, action)
                # and the val as a max of the previous val and the new one
                val = max(val, self.dispatch( successor, action, turn, alpha, beta ))
                # If it is bigger than beta, we have to return the val, and end with this action
                # It means that we do not continue with that branch (pruned)
                if val > beta:
                    return val
                # Otherwise, alpha is the max between alpha and the value
                alpha = max(alpha, val)
            return val
        else:
            # Init the val as plus infinity
            val = float("inf")
            # Get all the possible (legal) actions that we can carry out at the position of the agent,
            # and iterate for every action
            for action in gameState.getLegalActions(agentIndex):
                # Assign the successor given an action
                successor = gameState.generateSuccessor(agentIndex, action)
                # and the val as a min of the previous val and the new one
                val = min(val, self.dispatch( successor, action, turn, alpha, beta ))
                # If it is smaller than alpha, we have to return the val, and end with this action
                # It means that we do not continue with that branch (pruned)
                if val < alpha:
                    return val
                # Otherwise, beta is the max between beta and the value
                beta = min(beta, val)
            return val

    # Function that returns de action of the actual GameState
    def getAction(self, gameState):
        return self.generalGetAction(gameState)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

