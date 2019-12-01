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
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # https://stackoverflow.com/questions/7781260/how-can-i-represent-an-infinite-number-in-python
        from decimal import Decimal # infinity values
        pos_inf = Decimal('Infinity')
        neg_inf = Decimal('-Infinity')

        if successorGameState.isWin():    #found winning state
          return pos_inf
        # get it to move all the time
        if action==Directions.STOP:
          return neg_inf

        map_food = newFood.asList()
        all_ghosts = successorGameState.getGhostPositions()
        food_distance = list()
        ghost_distance = list()
        distance_cost = 0 #extra cost to return

        if newPos in currentGameState.getCapsules():
          distance_cost += 2
        #manhattan of examining pos from food
        for ifood in map_food:
          food_distance.append( util.manhattanDistance(newPos, ifood) )
        #manhattan of examining pos from ghosts
        for ighost in all_ghosts:
          ghost_distance.append( util.manhattanDistance(newPos, ighost) )

        # my cost for food distances
        # good if food is close
        for ifood in food_distance:
          if ifood>3 and ifood<=13:
            distance_cost += 0.25
          elif ifood<=3:
            distance_cost +=1
          else:
            distance_cost += 0.2

        for ighost in ghost_distance:
          if ighost<2:
            return neg_inf #ghost too close

        # bad extra cost if ghost is close
        for ighost in all_ghosts:
          if ighost==newPos:  # fell on ghost
            return neg_inf
          elif util.manhattanDistance(newPos, ighost) <= 3.5: #ghost is a little close
            distance_cost += 0.8

        return successorGameState.getScore() + distance_cost

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
    def minimaxFunc(self, gameState, agent, depth):
      from decimal import Decimal # infinity values
      pos_inf = Decimal('Infinity')
      neg_inf = Decimal('-Infinity')
      
      # game won/lost
      if gameState.isLose() or gameState.isWin() or depth==0:
        # return the heuristic value of node
        return self.evaluationFunction(gameState)
      
      # minimum for ghosts
      if agent!=0:
        newAgent = agent + 1
        # change agent and depth
        if gameState.getNumAgents() == newAgent:
          # went through all agents
          newAgent = 0
        if newAgent == 0:
          depth -= 1

        min_return=pos_inf
        for legal_action in gameState.getLegalActions(agent):
          temp=self.minimaxFunc( gameState.generateSuccessor(agent, legal_action), newAgent, depth )
          if temp<min_return:
            min_return=temp
        return min_return

      # maximum for pacman
      else: 
        max_return=neg_inf
        for legal_action in gameState.getLegalActions(agent):
          temp=self.minimaxFunc( gameState.generateSuccessor(0, legal_action), 1, depth )
          if temp>max_return:
            max_return=temp
        return max_return

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
        """
        "*** YOUR CODE HERE ***"
        from decimal import Decimal # infinity values
        neg_inf = Decimal('-Infinity')
        # perform maximum action for pacman
        maximizingPlayer = neg_inf
        # random first action
        act = Directions.STOP
        # Pacman is always agent 0
        for legal_action in gameState.getLegalActions(0):
          temp = self.minimaxFunc( gameState.generateSuccessor(0, legal_action), 1, self.depth )
          if temp>maximizingPlayer:
            maximizingPlayer = temp
            act = legal_action
        return act
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):

    def min_value(self, gameState, agent, depth, a, b):
      from decimal import Decimal # infinity values
      pos_inf = Decimal('Infinity')
      neg_inf = Decimal('-Infinity')

      ghost_actions = gameState.getLegalActions(agent)
      if not ghost_actions:
        return self.evaluationFunction(gameState)
      
      # minimazing = 


    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
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

