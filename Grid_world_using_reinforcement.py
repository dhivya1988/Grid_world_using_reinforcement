import numpy as np

BOARD_ROWS = 5
BOARD_COLS = 5
WIN_STATE = (4, 4)
JUMP_STATE = (3, 3)
START = (1, 0)
BLACK_BOX = [(3, 2), (2, 2), (2, 3), (2, 4)]
DETERMINISTIC = False


class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        # for box in BLACK_BOX:
        #     self.board[box[0], box[1]] = -5
        # for i in range(BOARD_ROWS):
        #     for j in range(BOARD_COLS):
        #         self.board[i,j] = -1
        # self.board[3, 3] = 5
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self, flag):
        if self.state == WIN_STATE and flag == "win":
            return 10
        elif self.state == WIN_STATE and flag == "jump":
            return 15
        else:
            return 0

    def isEndFunc(self):
        if (self.state == WIN_STATE):
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == "north":
            return np.random.choice(["north", "west", "east"], p=[0.8, 0.1, 0.1])
        if action == "south":
            return np.random.choice(["south", "west", "east"], p=[0.8, 0.1, 0.1])
        if action == "west":
            return np.random.choice(["west", "north", "south"], p=[0.8, 0.1, 0.1])
        if action == "east":
            return np.random.choice(["east", "north", "south"], p=[0.8, 0.1, 0.1])

    def nextPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position on board
        """
        if self.determine:
            if action == "north":
                nextState = (self.state[0] - 1, self.state[1])
            elif action == "south":
                nextState = (self.state[0] + 1, self.state[1])
            elif action == "west":
                nextState = (self.state[0], self.state[1] - 1)
            else:
                nextState = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            # non-deterministic
            action = self._chooseActionProb(action)
            self.determine = True
            nextState = self.nextPosition(action)

        # if next state is legal
        if (nextState[0] >= 0) and (nextState[0] <= 4):
            if (nextState[1] >= 0) and (nextState[1] <= 4):
                if nextState not in BLACK_BOX:
                    return nextState
                if nextState == (2, 3) and action == "south":
                    self.state = (3, 3)
        return self.state


class Agent:

    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["north", "south", "west", "east"]
        self.State = State()
        self.isEnd = self.State.isEnd
        self.lr = 0.2
        self.exp_rate = 0.3
        self.decay_gamma = 0.9

        # initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = -1  # Q value is a dict of dict
                    # if (i,j) in BLACK_BOX:
                    #     self.Q_values[(i, j)][a] = -1
                    # elif (i,j) == (3,3):
                    #     self.Q_values[(i, j)][a] = 5
                    # else:
                    #     self.Q_values[(i, j)][a] = -1  # Q value is a dict of dict

    def chooseAction(self):
        # choose action with most expected value
        mx_next_reward = 0
        action = ''

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.State.state
                next_reward = self.Q_values[current_position][a]
                if next_reward >= mx_next_reward:
                    action = a
                    mx_next_reward = next_reward
            # print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        return action

    def takeAction(self, action):
        position = self.State.nextPosition(action)
        # update State
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd

    def play(self, rounds=100):
        i = 0
        cum_itr_list = []
        cum_itr = False
        while (i < rounds and cum_itr == False):
            print("-----------------------------------------------> episode ", i)
            # to the end of game back propagate reward
            print(self.Q_values[self.State.state])
            if self.State.isEnd:
                # back propagate
                print("___________________ executed_________________")
                if [(1, 3), "south"] in (self.states):
                    print(self.states)
                    reward = self.State.giveReward("jump")
                else:
                    reward = self.State.giveReward("win")
                cum_itr_list.append(reward)
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                print("Game End Reward", reward)
                for s in reversed(self.states):
                    try:
                        current_q_value = self.Q_values[s[0]][s[1]]
                    except:
                        current_q_value = -1
                    print(current_q_value)
                    if s[0] == (1, 3) and s[1] == "south":
                        reward = 5
                        reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    else:
                        reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)

                if cum_itr_list[-30:].count(15) == 30:
                    print(cum_itr_list[-30:])
                    cum_itr = True
                    break

                self.reset()
                i += 1

            else:
                action = self.chooseAction()
                # append trace
                self.states.append([(self.State.state), action])
                print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
                print("next state", self.State.state)
                print("---------------------")
                self.isEnd = self.State.isEnd

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(max(self.Q_values[(i, j)].values())).ljust(6) + ' | '
                # out += str(self.Q_values[i, j]['up']).ljust(6) + ' | '
                # out += str(self.Q_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')


if __name__ == "__main__":
    ag = Agent()
    print("initial Q-values ... \n")
    print(ag.Q_values)

    ag.play(500)
    print("latest Q-values ... \n")
    print(ag.Q_values)
    ag.showValues()
