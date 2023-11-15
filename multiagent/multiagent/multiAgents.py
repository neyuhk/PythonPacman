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
from pacman import GameState

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        foodList = newFood.asList()


        if successorGameState.isWin():
            return 10000000

        heurictic = successorGameState.getScore()

        closestFoodDistance = 100000

        if len(foodList) == 0:
            return 0

        for foodPosition in foodList:
            distance = manhattanDistance(newPos, foodPosition)
            if distance < closestFoodDistance:
                closestFoodDistance = distance


        closestGostDistance = 1000000
        idxGostClosest  = -1

        if len(newGhostStates) == 0:
            return successorGameState.getScore()
        for idx,ghostState in enumerate(newGhostStates):
            distance = manhattanDistance(newPos, ghostState.getPosition())
            if distance < closestGostDistance:
                closestGostDistance = distance
                idxGostClosest = idx

        if newGhostStates[idxGostClosest].scaredTimer <= 0 and closestGostDistance == 1:
            heurictic -= 1000000
        
        if newGhostStates[idxGostClosest].scaredTimer != 0 and closestGostDistance == 0:
            heurictic += 1000000


        if action == Directions.STOP:
            heurictic -= 10


        foodScore = 100 if currentGameState.getNumFood() > successorGameState.getNumFood() else 0

        return heurictic - closestFoodDistance + foodScore

#         return successorGameState.getScore()

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def maxValue(self, gameState, agent, depth):
        bestValue = float("-inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = self.minimax(successor, agent+1, depth)
            bestValue = max(bestValue, v)
            # Save the action that we will finally take, which is the
            # best action that we perform at depth 1
            if depth == 1 and bestValue == v: self.action = action
        return bestValue

    def minValue(self, gameState, agent, depth):
        bestValue = float("inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = self.minimax(successor, agent+1, depth)
            bestValue = min(bestValue, v)
        return bestValue

    def minimax(self, gameState, agent=0, depth=0):

        # To simplify the code, we increment by one the agent index thus
        # we must perform a modulus operation to keep the agent index in range
        agent = agent % gameState.getNumAgents()

        if self.isTerminalState(gameState):
            return self.evaluationFunction(gameState)

        if self.isPacman(agent):
            # Since the depth is measured in plies, we must increment the depth each
            # time a search ply is performed (in a ply, all agents each take one action)
            if depth < self.depth:
                return self.maxValue(gameState, agent, depth+1)
            else:
                return self.evaluationFunction(gameState)
        else:
            return self.minValue(gameState, agent, depth)


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # This function returns the best action's utility, but we can
        # ignore that value as we are interested in the action, not the value
        self.minimax(gameState)
        return self.action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, gameState, agent, depth, alpha, beta):
        bestValue = float("-inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = self.minimax(successor, agent + 1, depth, alpha, beta)
            bestValue = max(bestValue, v)
            if depth == 1 and bestValue == v: self.action = action
            if bestValue > beta: return bestValue
            alpha = max(alpha, bestValue)
        return bestValue

    def minValue(self, gameState, agent, depth, alpha, beta):
        bestValue = float("inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = self.minimax(successor, agent + 1, depth, alpha, beta)
            bestValue = min(bestValue, v)
            if bestValue < alpha: return bestValue
            beta = min(beta, bestValue)
        return bestValue

    def minimax(self, gameState, agent=0, depth=0,
                alpha=float("-inf"), beta=float("inf")):

        agent = agent % gameState.getNumAgents()

        if self.isTerminalState(gameState):
            return self.evaluationFunction(gameState)

        if self.isPacman(agent):
            if depth < self.depth:
                return self.maxValue(gameState, agent, depth+1, alpha, beta)
            else:
                return self.evaluationFunction(gameState)
        else:
            return self.minValue(gameState, agent, depth, alpha, beta)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.minimax(gameState)
        return self.action

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
