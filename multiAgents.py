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
        current_pacman_pos = currentGameState.getPacmanPosition()
        current_score = currentGameState.getScore()
        foodList = newFood.asList()

        min_food = 9999
        for food in foodList:
            distance = manhattanDistance(food, newPos)
            if distance < min_food:
                min_food = distance
        ghostPosition = successorGameState.getGhostPosition(1)
        ghostDistance = manhattanDistance(ghostPosition, newPos)
        if ghostDistance > 4:
            current_score += 200
        else:
            current_score -= 15
        
        if current_pacman_pos != newPos:
            current_score += 10

        current_score += 100 / min_food
        if len(currentGameState.getFood().asList()) > len(foodList):
            current_score += 100

        return current_score

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

        def min_value(state, agent, depth):
            minV = [99999, ""]

            # go through every action this agent can take, evaluate the minimum value that will propogate up
            for move in state.getLegalActions(agent):
                val = minimax(state.generateSuccessor(agent, move), agent + 1, depth)
                if val[0] < minV[0]:
                    minV = [val[0], move]
            return minV

        def max_value(state, agent, depth):
            maxV = [-99999, ""]
            
            # go through every action this agent can take, evaluate the maximum value that will propogate up
            for move in state.getLegalActions(agent):
                val = minimax(state.generateSuccessor(agent, move), agent + 1, depth)
                if val[0] > maxV[0]:
                    maxV = [val[0], move]
            return maxV
    
        def minimax(state, agent, depth):
            # BASE CASE: if we are at depth 0 or the game is over
            if depth == 0 or state.isWin() or state.isLose():
                return (self.evaluationFunction(state), "FINISHED")
            
            # computing which agent we are looking at
            # using % modulus operator deal with agent values that exceed the number of total agents (ex. 10 % 9 = 1)
            agent %= state.getNumAgents()

            # we go to the next level once we're done looking at all the agents at the current level
            if agent == state.getNumAgents() - 1:
                depth -= 1
            if agent == 0: # if the agent is Pacman
                return max_value(state, agent, depth)
            else: # if the agent is a ghost
                return min_value(state, agent, depth)
            
        # initial call, starting with agent = Pacman
        bestMove = minimax(gameState, 0, self.depth)
        return bestMove[1]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def min_value(state, agent, depth, alpha, beta):
            minV = [99999, ""]

            # go through every action this agent can take, evaluate the minimum value that will propogate up
            for move in state.getLegalActions(agent):
                val = minimax(state.generateSuccessor(agent, move), agent + 1, depth, alpha, beta)
                if val[0] < minV[0]:
                    minV = [val[0], move]

                # if the min value we computed is less than MAX's best option, prune
                if minV[0] < alpha:
                    return minV
                beta = min(beta, minV[0])
            return minV

        def max_value(state, agent, depth, alpha, beta):
            maxV = [-99999, ""]
            
            # go through every action this agent can take, evaluate the maximum value that will propogate up
            for move in state.getLegalActions(agent):
                val = minimax(state.generateSuccessor(agent, move), agent + 1, depth, alpha, beta)
                if val[0] > maxV[0]:
                    maxV = [val[0], move]
                
                # if the max value we computed is greater than MIN's best option, prune
                if maxV[0] > beta:
                    return maxV
                alpha = max(alpha, maxV[0])
            return maxV
    
        def minimax(state, agent, depth, alpha, beta):
            # BASE CASE: if we are at depth 0 or the game is over
            if depth == 0 or state.isWin() or state.isLose():
                return (self.evaluationFunction(state), "FINISHED")
            
            # computing which agent we are looking at
            # using % modulus operator deal with agent values that exceed the number of total agents (ex. 10 % 9 = 1)
            agent %= state.getNumAgents()

            # we go to the next level once we're done looking at all the agents at the current level
            if agent == state.getNumAgents() - 1:
                depth -= 1
            if agent == 0: # if the agent is Pacman
                return max_value(state, agent, depth, alpha, beta)
            else: # if the agent is a ghost
                return min_value(state, agent, depth, alpha, beta)
            
        # initial call, starting with agent = Pacman
        bestMove = minimax(gameState, 0, self.depth, -99999, 99999)
        return bestMove[1]
        util.raiseNotDefined()

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
        def exp_value(state, agent, depth):
            # minV = [99999, ""]
            actions = state.getLegalActions(agent)
            len_actions = len(actions)
            val = [0, ""]

            for move in actions:
                # computing the 
                temp = (minimax(state.generateSuccessor(agent, move), agent + 1, depth)[0]) / len_actions
                val[0] = val[0] + temp
            return val

        def max_value(state, agent, depth):
            maxV = [-99999, ""]
            
            # go through every action this agent can take, evaluate the maximum value that will propogate up
            for move in state.getLegalActions(agent):
                val = minimax(state.generateSuccessor(agent, move), agent + 1, depth)
                if val[0] > maxV[0]:
                    maxV = [val[0], move]
            return maxV
    
        def minimax(state, agent, depth):
            # BASE CASE: if we are at depth 0 or the game is over
            if depth == 0 or state.isWin() or state.isLose():
                return (self.evaluationFunction(state), "FINISHED")
            
            # computing which agent we are looking at
            # using % modulus operator deal with agent values that exceed the number of total agents (ex. 10 % 9 = 1)
            agent %= state.getNumAgents()

            # we go to the next level once we're done looking at all the agents at the current level
            if agent == state.getNumAgents() - 1:
                depth -= 1
            if agent == 0: # if the agent is Pacman
                return max_value(state, agent, depth)
            else: # if the agent is a ghost
                return exp_value(state, agent, depth)
            
        # initial call, starting with agent = Pacman
        bestMove = minimax(gameState, 0, self.depth)
        return bestMove[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    print("STATE ", type(currentGameState))
    successorGameState = currentGameState.generatePacmanSuccessor()
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    current_pacman_pos = currentGameState.getPacmanPosition()
    current_score = currentGameState.getScore()
    foodList = newFood.asList()

    min_food = 9999
    for food in foodList:
        distance = manhattanDistance(food, newPos)
        if distance < min_food:
            min_food = distance
    ghostPosition = successorGameState.getGhostPosition(1)
    ghostDistance = manhattanDistance(ghostPosition, newPos)
    if ghostDistance > 4:
        current_score += 200
    else:
        current_score -= 15
    
    if current_pacman_pos != newPos:
        current_score += 10

    current_score += 100 / min_food
    if len(currentGameState.getFood().asList()) > len(foodList):
        current_score += 100

    return current_score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
