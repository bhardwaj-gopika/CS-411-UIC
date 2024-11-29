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
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            new_values = self.values.copy()
            for s in self.mdp.getStates():
                if self.mdp.isTerminal(s):
                    continue
                actions = self.mdp.getPossibleActions(s)
                highest_expected = max([self.getQValue(s,action) for action in actions])
                new_values[s] = highest_expected
            self.values = new_values


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
        q_value = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            q_value += prob*(self.mdp.getReward(state, action, nextState) + self.discount*self.getValue(nextState))
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
        "make a dictionary for pi"
        pi = util.Counter()
        for a in self.mdp.getPossibleActions(state):
            pi[a] = self.getQValue(state, a)
        return pi.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        num_states = len(self.mdp.getStates())
        counter = 0
        for i in range(self.iterations):
            if counter == num_states:
                counter = 0

            new_values = self.values.copy()
            states = self.mdp.getStates()
            s = states[counter]
            if self.mdp.isTerminal(s):
                counter += 1
                continue

            actions = self.mdp.getPossibleActions(s)
            highest_expected = max([self.getQValue(s, action) for action in actions])
            new_values[s] = highest_expected
            self.values = new_values
            counter += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        # dictionary with key: state and value: set of predecessors
        states_dict = {}

        # add all states to dictionary with empty sets
        for s in states:
            empty_predecessor_set = set()
            states_dict[s] = empty_predecessor_set

        # fill sets with predecessors
        for s in states:
            actions = self.mdp.getPossibleActions(s)
            for a in actions:
                transitions = self.mdp.getTransitionStatesAndProbs(s, a)
                for next_state, prob in transitions:
                    if prob != 0:
                        predecessors = states_dict[next_state]
                        predecessors.add(s)
                        states_dict[next_state] = predecessors

        q = util.PriorityQueue()

        # push non terminal states into the queue with priority negative diff
        for s in states:
            if (self.mdp.isTerminal(s)):
                continue
            actions = self.mdp.getPossibleActions(s)
            highest_expected = max([self.getQValue(s, action) for action in actions])
            curr_val = self.values[s]
            diff = abs(curr_val - highest_expected)
            q.push(s, -diff)

        for i in range(self.iterations):
            if (q.isEmpty()):
                break
            s = q.pop()
            if (self.mdp.isTerminal(s)):
                continue

            new_values = self.values.copy()
            actions = self.mdp.getPossibleActions(s)
            new_values[s] = max([self.getQValue(s, action) for action in actions])
            self.values = new_values

            predecessors = states_dict[s]
            for p in predecessors:
                actions = self.mdp.getPossibleActions(p)
                highest_expected = max([self.getQValue(p, action) for action in actions])
                curr_val = self.values[p]
                diff = abs(curr_val - highest_expected)
                if (diff > self.theta):
                    q.update(p, -diff)





