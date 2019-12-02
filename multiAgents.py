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
      from decimal import Decimal                 # infinity values
      pos_inf = Decimal('Infinity')

      ret_action = Directions.STOP              # random first action
      if not gameState.getLegalActions(agent):  # list is empty
        return self.evaluationFunction(gameState), ret_action
      
      minimizing = pos_inf
      for legal_action in gameState.getLegalActions(agent):
        if agent==(gameState.getNumAgents()-1): #-1 because start from pacman is 0
          ret_value, action_unused = self.max_value( gameState.generateSuccessor(agent, legal_action), depth-1, a, b)
        else:
          ret_value, action_unused = self.min_value( gameState.generateSuccessor(agent, legal_action), agent+1, depth, a, b)
        if ret_value<minimizing:
          minimizing = ret_value
          ret_action = legal_action             # get action for minimum value
        if minimizing<a:                        #
          return minimizing, ret_action
        if minimizing<b:
          b = minimizing
      return minimizing, ret_action

    def max_value(self, gameState, depth, a, b):
      from decimal import Decimal               # infinity values
      neg_inf = Decimal('-Infinity')

      ret_action = Directions.STOP
      if gameState.isWin() or gameState.isLose() or depth==0:
        return self.evaluationFunction(gameState), ret_action
      
      maximizing = neg_inf
      for legal_action in gameState.getLegalActions(0):
        ret_value, act_unused = self.min_value( gameState.generateSuccessor(0, legal_action), 1, depth, a, b )
        if maximizing<ret_value:
          maximizing = ret_value
          ret_action = legal_action
        if maximizing>b:
          return maximizing, ret_action
        if maximizing>a:
          a = maximizing
      return maximizing, ret_action
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        from decimal import Decimal # infinity values
        pos_inf = Decimal('Infinity')
        neg_inf = Decimal('-Infinity')
        a = neg_inf
        b = pos_inf
        val_unused, ret_action = self.max_value(gameState, self.depth, a, b)  #0 is for pacman
        return ret_action
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState, agent, depth):
      from decimal import Decimal # infinity values
      neg_inf = Decimal('-Infinity')

      if agent==gameState.getNumAgents():             #checked all agents
        return self.expectimax(gameState, 0, depth-1) #start procedure for pacman and lower depth

      if gameState.isWin() or gameState.isLose() or depth==0:
        return self.evaluationFunction(gameState)
      
      if agent!=0:  #ghost -> make calculation
        ret_sum = 0.0
        ret_size = 0
        for legal_action in gameState.getLegalActions(agent):
          ret_value = float(self.expectimax( gameState.generateSuccessor(agent, legal_action), agent+1, depth ))
          ret_sum += ret_value
          ret_size += 1
        return float(ret_sum)/float(ret_size) #average
      else: #pacman
        ret = neg_inf
        for legal_action in gameState.getLegalActions(0):
          ret_value = self.expectimax( gameState.generateSuccessor(0, legal_action), 1, depth )#1=pacman+1
          if ret_value>ret:
            ret = ret_value
        return ret

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        import random
        from decimal import Decimal # infinity values
        neg_inf = Decimal('-Infinity')
        compared = neg_inf
        legal_actions = list()

        for legal_action in gameState.getLegalActions(0):
          temp = self.expectimax(gameState.generateSuccessor(0, legal_action), 1, self.depth) #1=pacman+1
          legal_actions.append(legal_action)
          if temp>compared:
            compared = temp
        
        final_legal_actions = list()
        for legal_action in gameState.getLegalActions(0):
          temp = self.expectimax(gameState.generateSuccessor(0, legal_action), 1, self.depth) #1=pacman+1
          if temp==compared:
            final_legal_actions.append(legal_action)
        random_pos = random.randrange(len(final_legal_actions)) #random list index

        return final_legal_actions[random_pos]
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    from decimal import Decimal # infinity values
    pos_inf = Decimal('Infinity')
    # neg_inf = Decimal('-Infinity')
    
    pacman_pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_list = currentGameState.getGhostStates()
    capsule_list = currentGameState.getCapsules()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in ghost_list]

    if currentGameState.isWin():
      return pos_inf
    min_food = pos_inf
    for food in food_list:
      temp = util.manhattanDistance( food, pacman_pos )
      if temp<min_food:
        min_food = temp
    min_caps = pos_inf
    for caps in capsule_list:
      temp = util.manhattanDistance( caps, pacman_pos )        
      if temp<min_caps:
        min_caps = temp
    ghost_val = 0
    m_gh_list = list()
    for gh in ghost_list:
      min_ghost = util.manhattanDistance( gh.getPosition(), pacman_pos )
      m_gh_list.append(min_ghost)
      if min_ghost>0:
        if gh.scaredTimer!=0:
          ghost_val += 100
        else:
          ghost_val -= 10
        if min_ghost<4:
          ghost_val -= 50
        elif min_ghost<8:
          ghost_val -= 2
    if min(m_gh_list)>6:
      return (min_food + min_caps)*2
    return currentGameState.getScore() + 10.0/min_food + ghost_val + 10.0/float(min_caps)

# Abbreviation
better = betterEvaluationFunction

