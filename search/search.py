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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """
    #print "Start:", problem.getStartState()
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    stack = util.Stack()
    stack.push((problem.getStartState(),[],[]))
    visited_node=[]
    #print "Stack is :", stack.pop()
    #node, actions, visited = stack.pop()
    #print "line 2 :",node, actions, visited
    while not stack.isEmpty():
        #node, actions, cost = stack.pop()
        position, actions, visited_node = stack.pop()
        #visited_node = list.append(set(visited_node))

        if problem.isGoalState(position):
            return actions
        for state_new, action_new, child_cost in problem.getSuccessors(position):
            if state_new not in visited_node:
                #print "succesor function now:", state_new, action_new, child_cost
                #print "action beffore push",actions
                #actions= actions + [direction]
                action_new=actions+[action_new];
                visited_node=visited_node+[position]
                #stack.push((state_new, actions+ [action_new], visited_node + [position]))
                stack.push((state_new, action_new , visited_node ))
                #print " cord, action and visited :",state_new, actions+ [child_cost],visited_node + [position]
                #print "actions after push", actions

                """
    



    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    stack.push((problem.getStartState(),[]))
    visited_node = set([])
    #print "Stack is :", stack.pop()
    #node, actions, visited = stack.pop()
    #print "line 2 :",node, actions, visited
    while not stack.isEmpty():
        #node, actions, cost = stack.pop()
        position, actions = stack.pop()
        #visited_node = list.append(set(visited_node))


        if problem.isGoalState(position):
            return actions

        successor_list = problem.getSuccessors(position)

        for state_new, action_new, child_cost in successor_list:
            if state_new not in visited_node:
                #print "succesor function now:", state_new, action_new, child_cost
                #print "action beffore push",actions
                #actions= actions + [direction]
                action_new=actions+[action_new];
                visited_node = set(list(visited_node) + [position])
                #stack.push((state_new, actions+ [action_new], visited_node + [position]))
                stack.push((state_new, action_new  ))
                #print " cord, action and visited :",state_new, actions+ [child_cost],visited_node + [position]
                #print "actions after push", actions

    return []
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    queue.push((problem.getStartState(), []))
    visited_node = set()
    # print "Stack is :", stack.pop()
    # node, actions, visited = stack.pop()
    # print "line 2 :",node, actions, visited
    while not queue.isEmpty():
        # node, actions, cost = stack.pop()
        position, actions = queue.pop()
        # visited_node = list.append(set(visited_node))

        if problem.isGoalState(position):
            return actions

        successor_list = problem.getSuccessors(position)

        visited_node.add(position)
        for state_new, action_new, child_cost in successor_list:
            if state_new not in visited_node:
                # print "succesor function now:", state_new, action_new, child_cost
                # print "action beffore push",actions
                # actions= actions + [direction]
                action_new = actions + [action_new];
                if not problem.isGoalState(state_new):
                    visited_node.add(state_new)
                # stack.push((state_new, actions+ [action_new], visited_node + [position]))
                queue.push((state_new, action_new))
                # print " cord, action and visited :",state_new, actions+ [child_cost],visited_node + [position]
                # print "actions after push", actions

    return []


    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pqueue = util.PriorityQueue()
    pqueue.push((problem.getStartState(), []),0)
    visited_node = set()
    # print "Stack is :", stack.pop()
    # node, actions, visited = stack.pop()
    # print "line 2 :",node, actions, visited
    while not pqueue.isEmpty():
        # node, actions, cost = stack.pop()
        position, actions = pqueue.pop()
        # visited_node = list.append(set(visited_node))

        if problem.isGoalState(position):
            return actions

        successor_list = problem.getSuccessors(position)

        visited_node.add(position)
        for state_new, action_new, child_cost in successor_list:
            if state_new not in visited_node:
                # print "succesor function now:", state_new, action_new, child_cost
                # print "action beffore push",actions
                # actions= actions + [direction]
                action_new = actions + [action_new];
                if not problem.isGoalState(state_new):
                   visited_node.add(state_new)
                # stack.push((state_new, actions+ [action_new], visited_node + [position]))
                pqueue.push((state_new, action_new), problem.getCostOfActions(action_new))
                # print " cord, action and visited :",state_new, actions+ [child_cost],visited_node + [position]
                # print "actions after push", actions

    return []

    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pqueue = util.PriorityQueue()
    pqueue.push((problem.getStartState(), []),0)
    visited_node = set()
    # print "Stack is :", stack.pop()
    # node, actions, visited = stack.pop()
    # print "line 2 :",node, actions, visited
    while not pqueue.isEmpty():
        # node, actions, cost = stack.pop()
        position, actions = pqueue.pop()
        # visited_node = list.append(set(visited_node))

        if problem.isGoalState(position):
            return actions

        successor_list = problem.getSuccessors(position)

        visited_node.add(position)
        for state_new, action_new, child_cost in successor_list:
            if state_new not in visited_node:
                # print "succesor function now:", state_new, action_new, child_cost
                # print "action beffore push",actions
                # actions= actions + [direction]
                action_new = actions + [action_new];
                if not problem.isGoalState(state_new):
                   visited_node.add(state_new)
                # stack.push((state_new, actions+ [action_new], visited_node + [position]))
                pqueue.push((state_new, action_new), problem.getCostOfActions(action_new)+heuristic(state_new,problem))
                # print " cord, action and visited :",state_new, actions+ [child_cost],visited_node + [position]
                # print "actions after push", actions

    return []

    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
