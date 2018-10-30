# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def solver(problem, route):

    # Inite first values that we will modify
    visited = []
    route.push((problem.getStartState(), [], 0))

    # It will only be empty when it visits all the nodes
    while not route.isEmpty():
        # First, we take the information doing the pop
        state, actions, weight = route.pop()
        
        # If the state is visited then continue with the next iteration
        if state in visited:
            continue

        # Add the current state to the visited list
        visited.append(state)
        
        # If the current state is the goal state, all right -> return the path
        if problem.isGoalState(state):
            return actions
        
        # Get all succesors and take its information
        for successor, action, stepCost in problem.getSuccessors(state):
            # Check all the successors that are not visited
            if successor not in visited:
                # Then add to the route the current information
                route.push((successor, actions + [action], stepCost + weight))

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # command: 
    #   python pacman.py -l tinyMaze -p SearchAgent --frameTime 0
    #   python pacman.py -l mediumMaze -p SearchAgent --frameTime 0
    #   python pacman.py -l bigMaze -z .5 -p SearchAgent --frameTime 0
  
    #DFS uses stack bc when visit a node we add it in front, so, when we
    #are evaluating, we do first this new discovered node
    route = util.Stack()
    return solver(problem, route)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #BFS uses Queue bc when visit a node we add it in the back, so, when we
    #are evaluating, we do first the nodes that we had before adding
    route = util.Queue()
    return solver(problem,route)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #UCS uses a priority queue bc we are taking into account which node has
    #reached less distance. When we arrive to the Goal we check if any other
    #node can arrive easily.
    route = util.PriorityQueueWithFunction(lambda n: n[-1])
    return solver(problem,route)

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    route = util.PriorityQueueWithFunction(lambda n: n[-1] + heuristic(n[0],problem))
    return solver(problem,route)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
