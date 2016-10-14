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
        """
        metric = util.manhattanDistance
        score = 0  # = successorGameState.getScore()
        punishGhostLambdas = {0: -7000, 1: -1000, 2: -30, 3: -10, 4: -4, 5: -2}
        nearFoodBonusDict = {0: 30, 1: 20, 2: 12, 3: 7, 4: 4}
        foodRemPunishK = -20
        foodCount = newFood.count(True)
        if (foodCount == 0):
            return 9999
        nearFoodDist = 100
        for i, item in enumerate(newFood):
            for j, foodItem in enumerate(item):
                nearFoodDist = min(nearFoodDist, metric(newPos, (i, j)) if foodItem else 100)
        nearFoodBonus = nearFoodBonusDict[nearFoodDist] if nearFoodDist in nearFoodBonusDict else (3 + 1 / nearFoodDist)
        foodRemPunish = foodRemPunishK * foodCount
        #print foodCount, nearFoodDist
        ghostDistances = [metric(newPos, hh.getPosition()) for hh in newGhostStates]
        ghostK = sum([punishGhostLambdas[dist] for dist in ghostDistances if dist in punishGhostLambdas])

        score = score + nearFoodBonus + ghostK + foodRemPunish * foodCount
        #print "score: ", score, ghostK
        return score
        """
        '''
        ghostdists = []
        fooddists = []
        for ghostState in newGhostStates:
            if ghostState.scaredTimer == 0:
                ghostdists.append(manhattanDistance(newPos, ghostState.getPosition()))

        ghostdists.sort()

        for food in newFood.asList():
            fooddists.append(manhattanDistance(newPos, food))

        fooddists.sort()

        if len(fooddists) > 0:
            closestFoodManhattan = fooddists[0]
        else:
            closestFoodManhattan = 0

        numNewFood = successorGameState.getNumFood()

        ghostEvalFunc = 0
        for ghost in newGhostStates:
            ghostdist = manhattanDistance(newPos, ghost.getPosition())
            if ghost.scaredTimer > ghostdist:
                ghostEvalFunc += ghost.scaredTimer - ghostdist

        # if there is a ghost in play that isn't scared, stay away from the nearest one.
        if len(ghostdists) > 0:
            ghostEvalFunc += ghostdists[0]
        return ghostEvalFunc - 10 * numNewFood - closestFoodManhattan
        '''


        #calculating minimum food distance
        fooddists = []
        for food in newFood.asList():
            fooddists.append(manhattanDistance(newPos, food))

        if len(fooddists) <= 0:
            minfoodDistance = 0
        else:
            minfoodDistance = min(fooddists)

        #Get the Food Count
        foodcount = successorGameState.getNumFood()


        #calculate minimum Ghost distance
        gost_distance_list=[]
        for ghost in newGhostStates:
            gost_distance_list.append(manhattanDistance(newPos, ghost.getPosition()))

        ghostdist=min(gost_distance_list)


        return ghostdist - 20 * foodcount - minfoodDistance


        #return successorGameState.getScore()

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
        """
        "*** YOUR CODE HERE ***"
        """
        def search_depth(state, depth, agent):
            if agent == state.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return search_depth(state, depth + 1, 0)
            else:
                actions = state.getLegalActions(agent)

                if len(actions) == 0:
                    return self.evaluationFunction(state)

                next_states = (
                    search_depth(state.generateSuccessor(agent, action),
                                 depth, agent + 1)
                    for action in actions
                )

                return (max if agent == 0 else min)(next_states)

        return max(
            gameState.getLegalActions(0),
            key=lambda x: search_depth(gameState.generateSuccessor(0, x), 1, 1)
        )
        """
        '''
        # Code of sing King -original
        actions = gameState.getLegalActions(0)
        v = float('-inf')
        nextAction = Directions.STOP
        for action in actions:
            temp = self.minValue(0, 1, gameState.generateSuccessor(0, action))
            if temp > v:
                v = temp
                nextAction = action
        return nextAction

    def maxValue(self, depth, agent, gameState):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        v = float('-inf')
        for action in actions:
            v = max(v, self.minValue(depth, agent + 1, gameState.generateSuccessor(agent, action)))
        return v

    def minValue(self, depth, agent, gameState):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        v = float('inf')
        for action in actions:
            if agent == gameState.getNumAgents() - 1:
                temp = self.maxValue(depth + 1, 0, gameState.generateSuccessor(agent, action))
            else:
                temp = self.minValue(depth, agent + 1, gameState.generateSuccessor(agent, action))
            v = min(v, temp)
        return v
    '''



        actions = gameState.getLegalActions()
        value = -float('inf')
        next_action=0

        for action in actions:

            child = self.minimiser(1, 0, gameState.generateSuccessor(0, action))
            if value< child:
                value = child
                next_action = action

        return next_action


    def maximiser(self,  agent, depth,gameState):

        actions = gameState.getLegalActions(agent)
        #basecase for the recursive function
        if not actions:
            return self.evaluationFunction(gameState)

        if gameState.isWin()==True:
            return self.evaluationFunction(gameState)
        elif gameState.isLose()==True:
            return self.evaluationFunction(gameState)
        elif depth == self.depth:
            return self.evaluationFunction(gameState)


        value=-float('inf')
        for action in actions:
            aggressor_agents=agent+1
            value = max(value, self.minimiser(aggressor_agents, depth, gameState.generateSuccessor(agent, action)))
        return value

    def minimiser(self, agent, depth, gameState):


        actions = gameState.getLegalActions(agent)
        # basecase for the recursive function
        if not actions:
            return self.evaluationFunction(gameState)

        if gameState.isWin() == True:
            return self.evaluationFunction(gameState)
        elif gameState.isLose() == True:
            return self.evaluationFunction(gameState)
        elif depth == self.depth:
            return self.evaluationFunction(gameState)

        value=float('inf')

        for action in actions:
            if agent < gameState.getNumAgents() - 1:
                new_aggressor_agent=agent + 1
                value= min(value, self.minimiser(new_aggressor_agent, depth, gameState.generateSuccessor(agent, action)))
            else:
                next_depth=depth + 1
                value = min(value, self.maximiser(0, next_depth, gameState.generateSuccessor(agent, action)))

        return value


        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        '''
        # sing king code ORIGINAL

        actions = gameState.getLegalActions(0)
        v = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        nextAction = Directions.STOP
        for action in actions:
            temp = self.minValue(0, 1, gameState.generateSuccessor(0, action), alpha, beta)
            if temp > v:
                v = temp
                nextAction = action
            if v > beta:
                return nextAction
            alpha = max(alpha, v)
        return nextAction

    def maxValue(self, depth, agent, gameState, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        v = float('-inf')
        for action in actions:
            v = max(v, self.minValue(depth, agent + 1, gameState.generateSuccessor(agent, action), alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, depth, agent, gameState, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        v = float('inf')
        for action in actions:
            if agent == gameState.getNumAgents() - 1:
                temp = self.maxValue(depth + 1, 0, gameState.generateSuccessor(agent, action), alpha, beta)
            else:
                temp = self.minValue(depth, agent + 1, gameState.generateSuccessor(agent, action), alpha, beta)
            v = min(v, temp)
            if v < alpha:
                return v
            beta = min(beta, v)
        return v
        '''



        actions = gameState.getLegalActions()
        value = -float('inf')
        #assign initial Alpha and beta
        Alpha=-float('inf')
        Beta=float('inf')
        next_action = 0

        for action in actions:

            child = self.minimiser(1, 0, gameState.generateSuccessor(0, action),Alpha,Beta)
            if value < child:
                value = child
                next_action = action

            if value>Beta :  #equality not included as per the instruction
                return next_action
            Alpha=max(Alpha,value)

        return next_action

    def maximiser(self, agent, depth, gameState,Alpha,Beta):

        actions = gameState.getLegalActions(agent)
        # basecase for the recursive function
        if not actions:
            return self.evaluationFunction(gameState)

        if gameState.isWin() == True:
            return self.evaluationFunction(gameState)
        elif gameState.isLose() == True:
            return self.evaluationFunction(gameState)
        elif depth == self.depth:
            return self.evaluationFunction(gameState)

        value = -float('inf')

        for action in actions:
            aggressor_agents = agent + 1
            value = max(value, self.minimiser(aggressor_agents, depth, gameState.generateSuccessor(agent, action),Alpha,Beta))
            if value > Beta:  #equality not included as per the instruction
                return value
            Alpha = max(Alpha, value)
        return value

    def minimiser(self, agent, depth, gameState,Alpha,Beta):

        actions = gameState.getLegalActions(agent)
        # basecase for the recursive function
        if not actions:
            return self.evaluationFunction(gameState)

        if gameState.isWin() == True:
            return self.evaluationFunction(gameState)
        elif gameState.isLose() == True:
            return self.evaluationFunction(gameState)
        elif depth == self.depth:
            return self.evaluationFunction(gameState)

        value = float('inf')
        for action in actions:
            if agent < gameState.getNumAgents() - 1:
                new_aggressor_agent = agent + 1
                value = min(value,
                            self.minimiser(new_aggressor_agent, depth, gameState.generateSuccessor(agent, action),Alpha,Beta))
            else:
                next_depth = depth + 1
                value = min(value, self.maximiser(0, next_depth, gameState.generateSuccessor(agent, action),Alpha,Beta))
            if value < Alpha:  #equality not included as per the instruction
                return value
            Beta = min(Beta, value)

        return value









        #util.raiseNotDefined()

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

        '''

        #sing kings code ORIGINAL

        actions = gameState.getLegalActions(0)
        v = float('-inf')
        nextAction = Directions.STOP
        for action in actions:
            temp = self.expValue(0, 1, gameState.generateSuccessor(0, action))
            if temp > v:
                v = temp
                nextAction = action
        return nextAction

    def maxValue(self, depth, agent, gameState):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        v = float('-inf')
        for action in actions:
            v = max(v, self.expValue(depth, agent + 1, gameState.generateSuccessor(agent, action)))
        return v


    def expValue(self, depth, agent, gameState):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agent)
        numActions = len(actions)
        if numActions == 0:
            return self.evaluationFunction(gameState)
        v = 0
        for action in actions:
            if agent == gameState.getNumAgents() - 1:
                v += self.maxValue(depth + 1, 0, gameState.generateSuccessor(agent, action))
            else:
                v += self.expValue(depth, agent + 1, gameState.generateSuccessor(agent, action))
        v = float(v) / float(len(actions))
        return v

        '''

        actions = gameState.getLegalActions()
        value = -float('inf')
        next_action = 0

        for action in actions:

            child = self.expectimax(1, 0, gameState.generateSuccessor(0, action))
            if value < child:
                value = child
                next_action = action

        return next_action

    def maximiser(self, agent, depth, gameState):

        actions = gameState.getLegalActions(agent)
        # basecase for the recursive function
        if not actions:
            return self.evaluationFunction(gameState)

        if gameState.isWin() == True:
            return self.evaluationFunction(gameState)
        elif gameState.isLose() == True:
            return self.evaluationFunction(gameState)
        elif depth == self.depth:
            return self.evaluationFunction(gameState)

        value = -float('inf')
        for action in actions:
            aggressor_agents = agent + 1
            value = max(value, self.expectimax(aggressor_agents, depth, gameState.generateSuccessor(agent, action)))
        return value

    def expectimax(self, agent, depth, gameState):

        actions = gameState.getLegalActions(agent)
        # basecase for the recursive function
        if not actions:
            return self.evaluationFunction(gameState)

        if gameState.isWin() == True:
            return self.evaluationFunction(gameState)
        elif gameState.isLose() == True:
            return self.evaluationFunction(gameState)
        elif depth == self.depth:
            return self.evaluationFunction(gameState)

        value = 0

        for action in actions:
            if agent < gameState.getNumAgents() - 1:
                new_aggressor_agent = agent + 1
                value = value+ self.expectimax(new_aggressor_agent, depth, gameState.generateSuccessor(agent, action))
            else:
                next_depth = depth + 1
                value = value+ self.maximiser(0, next_depth, gameState.generateSuccessor(agent, action))

        value= value/len(actions)

        return value













        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    "*** YOUR CODE HERE ***"

    import math

    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    currentPosition = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score1=currentGameState.getScore()

    fooddists = []
    for food in newFood.asList():
        fooddists.append(manhattanDistance(currentPosition, food))

    if len(fooddists) <= 0:
        minfoodDistance = 0
    else:
        minfoodDistance = min(fooddists)

    # Get the Food Count
    foodcount = currentGameState.getNumFood()

    # calculate minimum Ghost distance
    gost_distance_list = []
    for ghost in newGhostStates:
        gost_distance_list.append(manhattanDistance(currentPosition, ghost.getPosition()))

    ghostdist = min(gost_distance_list)
    if ghostdist==0: ghostdist+=1
    maxScaredTime= max(newScaredTimes)

    if maxScaredTime>0:
        eff_ghost=50*maxScaredTime
    else:
        eff_ghost=30*ghostdist



    Bullets=currentGameState.getCapsules()

    #numBullets=len(Bullets)
    numBullets=10
    if currentPosition in Bullets:
        numBullets=20*numBullets
    score=0
    if foodcount > 0:
        for food in newFood.asList():
            minfoodDistance = min(minfoodDistance, util.manhattanDistance(food, currentPosition))
        score += 1.0 / minfoodDistance

    if ghostdist ==0: ghostdist=1
    #return 50*maxScaredTime + 30*ghostdist - 500 * foodcount - 10*minfoodDistance
    #return eff_ghost + 30*numBullets - 500 * foodcount - 10 * minfoodDistance

    return -minfoodDistance + math.log(ghostdist) + currentGameState.getScore()+score1



    '''



    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    metric = util.manhattanDistance
    ghostDistances = [metric(newPos, hh.getPosition()) for hh in newGhostStates]
    if any([gh == 0 for gh in ghostDistances]):
        return -999

    score = currentGameState.getScore()  # = successorGameState.getScore()

    foodCount = currentGameState.getNumFood()
    if (foodCount == 0):
        return 9999
    nearFoodDist = 100
    for i, item in enumerate(newFood):
        for j, foodItem in enumerate(item):
            nearFoodDist = min(nearFoodDist, metric(newPos, (i, j)) if foodItem else 100)
    ghostFun = lambda d: -20 + d ** 4 if d < 3 else -1.0 / d
    # nearFoodBonus = nearFoodBonusDict[nearFoodDist] if nearFoodDist in nearFoodBonusDict else (30 - 2*nearFoodDist)
    ghostK = sum([ghostFun(ghostDistances[i]) if newScaredTimes[i] < 1 else 0 for i in range(len(ghostDistances))])
    nearFoodBonus = 1.0 / nearFoodDist
    foodRemPunish = -1.5
    peleteRemPunish = -8 if all((t == 0 for t in newScaredTimes)) else 0
    if all((t > 0 for t in newScaredTimes)):
        ghostK *= (-1)

    pelets = currentGameState.getCapsules()
    pelets.sort()
    # print pelets
    nearPeletDist = 100
    nearPeletDist = min(nearPeletDist, [metric(newPos, pelet) for pelet in pelets])
    nearPeletBonus = 1.0 / nearPeletDist
    peleteRemaining = len(pelets)

    # print foodCount, nearFoodDist

    score = score + nearFoodBonus + 2 * ghostK + 3 * nearPeletBonus + foodRemPunish * foodCount + peleteRemaining * peleteRemPunish
    # print "score:", score, "nearFoodBonus:",nearFoodBonus, "ghostK:",ghostK, "peleteRemPunish:", peleteRemPunish, newScaredTimes
    return score
    '''
    '''
    import math
    import sys
    x, y = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    # print capsules

    newGhostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    if ghostDist ==0: ghostDist=1

    for ghost in newGhostStates:
        ghostPositions.append(ghost.getPosition())
        #ghostPositions = [ghost.getPosition() for ghost in newGhostStates]

    for x1, y1 in ghostPositions:
        ghostDist = min([abs(x1 - x) + abs(y1 - y))



    if any(scaredTimes):
        ghostDist = sys.maxint

    try:
        foodDist = min([abs(x1 - x) + abs(y1 - y) for x1, y1 in foodGrid])
    except:
        foodDist = 0

    return -foodDist + math.log(ghostDist + 1) + currentGameState.getScore()


    #util.raiseNotDefined()
    '''
    '''
      import math
    x, y = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    #print capsules

    newGhostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    try:
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
        ghostDist = min([abs(x1-x)+abs(y1-y) for x1, y1 in ghostPositions])
    except:
        ghostDist = 0

    if any(scaredTimes):
        ghostDist = sys.maxint

    try:
        foodDist = min([abs(x1-x)+abs(y1-y) for x1, y1 in foodGrid])
    except:
        foodDist = 0

    return -foodDist + math.log(ghostDist+1) + currentGameState.getScore()

    '''

# Abbreviation
better = betterEvaluationFunction

