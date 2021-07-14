from laser_tank import LaserTankMap, DotDict
import numpy as np
import time
import random

"""
Template file for you to implement your solution to Assignment 4. You should implement your solution by filling in the
following method stubs:
    train_q_learning()
    train_sarsa()
    get_policy()

You may add to the __init__ method if required, and can add additional helper methods and classes if you wish.

To ensure your code is handled correctly by the autograder, you should avoid using any try-except blocks in your
implementation of the above methods (as this can interfere with our time-out handling).

COMP3702 2020 Assignment 4 Support Code
"""
MOVE_FORWARD = 'f'
TURN_LEFT = 'l'
TURN_RIGHT = 'r'
SHOOT_LASER = 's'
MOVES = [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, SHOOT_LASER]
INIT_DICT = {'f': 0, 'l': 0, 'r': 0, 's': 0}


def dict_argmax(d):
    return max(d, key=d.get)


# epsilon-greedy way to select next move
def choose_a_move(state, epsilon, q_values):
    d = q_values.get(hash(state))
    if d is None:
        return np.random.choice(MOVES)

    if np.random.random() < epsilon:
        return max(q_values[hash(state)], key=q_values[hash(state)].get)
    else:
        return np.random.choice(MOVES)


class Solver:

    def __init__(self):
        """
        Initialise solver without a Q-value table.
        """

        #
        # TODO
        # You may add code here if you wish (e.g. define constants used by both methods).
        #
        # The allowed time for this method is 1 second.
        #

        self.q_values = None

    def train_q_learning(self, simulator):
        # initialize Q(s,a) arbitrarily
        q_values = {}
        q_values[hash(simulator)] = {'f': 0, 'l': 0, 'r': 0, 's': 0}
        alpha = 0.01  # also tried high value like 0.9
        epsilon = 0.8
        gamma = simulator.gamma
        time_limit = simulator.time_limit
        start = time.time()
        end = start + time_limit
        while time.time() < end:
            counter = 0
            # initialize s
            simulator.reset_to_start()
            done = False
            while not done and counter < 500:
                # choose a from s using e-greedy policy (80% chance best move)
                if random.random() > 0.8 or q_values.get(hash(simulator)) is None:
                    move = np.random.choice(MOVES)
                else:
                    move = max(q_values[hash(simulator)], key=q_values[hash(simulator)].get)

                if q_values.get(hash(simulator)) is None:
                    q_values[hash(simulator)] = {'f': 0, 'l': 0, 'r': 0, 's': 0}

                old_value = q_values[hash(simulator)][move]
                # store old state, also tried cloning previous simulator
                old_hash = hash(simulator)

                # apply move, state has changed
                reward, done = simulator.apply_move(move)
                counter += 1
                if q_values.get(hash(simulator)) is None:
                    q_values[hash(simulator)] = {'f': 0, 'l': 0, 'r': 0, 's': 0}
                td = reward + (gamma * (max(q_values[hash(simulator)].values()))) - old_value

                new_value = old_value + (alpha * td)
                q_values[old_hash][move] = new_value

        # store the computed Q-values
        self.q_values = q_values
        # print(self.q_values)

    def train_sarsa(self, simulator):
        # initialize Q(s,a) arbitrarily
        q_values = {}
        q_values[hash(simulator)] = {'f': 0, 'l': 0, 'r': 0, 's': 0}
        alpha = 0.01  # also tried high value like 0.9
        epsilon = 0.8
        gamma = simulator.gamma
        time_limit = simulator.time_limit
        start = time.time()
        end = start + time_limit
        while time.time() < end:
            counter = 0
            # initialize s
            simulator.reset_to_start()
            done = False
            # choose a from s using e-greedy policy (80% chance best move)
            if random.random() > 0.8 or q_values.get(hash(simulator)) is None:
                move = np.random.choice(MOVES)
            else:
                move = max(q_values[hash(simulator)], key=q_values[hash(simulator)].get)

            while not done and counter < 500:
                if q_values.get(hash(simulator)) is None:
                    q_values[hash(simulator)] = {'f': 0, 'l': 0, 'r': 0, 's': 0}

                old_value = q_values[hash(simulator)][move]
                # store old state, also tried cloning previous simulator
                old_hash = hash(simulator)

                # apply move, state has changed
                reward, done = simulator.apply_move(move)

                if random.random() > 0.8 or q_values.get(hash(simulator)) is None:
                    next_move = np.random.choice(MOVES)
                else:
                    next_move = max(q_values[hash(simulator)], key=q_values[hash(simulator)].get)

                counter += 1
                if q_values.get(hash(simulator)) is None:
                    q_values[hash(simulator)] = {'f': 0, 'l': 0, 'r': 0, 's': 0}

                td = reward + (gamma * (q_values[hash(simulator)][next_move])) - old_value

                new_value = old_value + (alpha * td)
                q_values[old_hash][move] = new_value
                move = next_move

        # store the computed Q-values
        self.q_values = q_values
        # print(self.q_values)

    def get_policy(self, state):
        """
        Get the policy for this state (i.e. the action that should be performed at this state).
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """

        #
        # TODO
        # Write code to return the optimal action to be performed at this state based on the stored Q-values.
        #
        # You can assume that either train_q_learning( ) or train_sarsa( ) has been called before this
        # method is called.
        #
        # When this method is called, you are allowed up to 1 second of compute time.
        #

        # {'f': 1, 'l': 2, 'r': 5, 's': 0}

        # print(self.q_values)

        return dict_argmax(self.q_values[hash(state)])
