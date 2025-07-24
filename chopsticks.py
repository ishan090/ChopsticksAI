
import json
from math import ceil, pow
import random
import time
from copy import deepcopy

# random.seed(0)

class Chopsticks:
    other_player = [1, 0]

    def __init__(self, initial=[[1, 1], [1, 1]], rand=False):
        if rand:
            self.state = self.randState()
        else:
            self.state = deepcopy(initial)
        self.player = 0
        self.winner = None
    
    def getKeyAble(self):
        return tuple([tuple(i) for i in self.state])
    
    def getState(self):
        return self.state.copy()
    
    def randState(self, k=2):
        """
        returns a random state as a list of hands
        """
        hands1 = []
        x = 0
        while x < 2:
            n = random.randint(0, 4)
            if n == 0 and 0 in hands1:
                continue
            hands1.append(n)
            x += 1
        k -= 1
        if k != 0:
            return [list(sorted(hands1, reverse=True))] + self.randState(k)
        return [list(sorted(hands1, reverse=True))]
    
    @classmethod
    def available_actions(self, state):
        return Chopsticks([list(i) for i in state]).get_moves()
        # p, o = state
        # initial, total = p, sum(p)
        # p, o = sorted(list(set(p)), reverse=True), sorted(list(set(o)), reverse=True)
        # # print("state, players:", s, p, o)
        # inter_player = [(i, j) for i in range(len(p)) if p[i] != 0 for j in range(len(o)) if o[j] != 0]
        # intra_player = []
        # for k in range(ceil((total+1)/2)):
        #     if k < 5 and total - k < 5 and (k, total-k) != initial:
        #         intra_player.append((2, (k, total-k)))
        # if (2, tuple(reversed(initial))) not in intra_player + [initial] and 0 not in initial:
        #     intra_player.append((2, tuple(reversed(initial))))
        # return inter_player + intra_player
    
    @classmethod
    def rev_moves(self, state: tuple):
        """
        state is in its tuple form
        """
        p, o = state
        p, o = sorted(list(set(p)), reverse=True), sorted(list(set(o)), reverse=True)
        # print("state, players:", s, p, o)
        inter_player = [(j, i) for j in range(len(o)) for i in range(len(p)) if (p[i]-o[j])%5 != 0]
        # intra player moves
        if state[0] == (0, 0):
            return inter_player
        initial = list(state[1])
        total = sum(initial)
        intra_player = []
        for k in range(ceil((total+1)/2)):
            if k < 5 and total - k < 5 and [total-k, k] != initial:
                intra_player.append((2, (total-k, k)))
        if (2, tuple(reversed(initial))) not in intra_player and 0 not in initial and list(reversed(initial)) != initial:
            intra_player.append((2, tuple(reversed(initial))))
        return inter_player + intra_player

    @classmethod
    def rev_action(self, state, action):
        new_state = [list(i) for i in state]
        if action[0] == 2:
            new_state[1] = list(sorted(action[1], reverse=True))   #HERE
        else:
            new_state[0][action[1]] = (new_state[0][action[1]] - new_state[0][action[0]]) % 5
            new_state[0] = list(sorted(new_state[0], reverse=True))
        new_state = list(reversed(new_state))
        # print("new state:", self.state)
        return [tuple(i) for i in new_state]

    def get_moves(self):
        """
        Assumes the game doesn't already have a winner
        """
        s = self.getState()
        initial = self.getState()[0]
        total = sum(initial)
        p, o = s.copy()
        p, o = sorted(list(set(p)), reverse=True), sorted(list(set(o)), reverse=True)
        # print("state, players:", s, p, o)
        inter_player = [(i, j) for i in range(len(p)) if p[i] != 0 for j in range(len(o)) if o[j] != 0]
        # print("interplayer actions chosen:", inter_player)
        intra_player = []
        for k in range(ceil((total+1)/2)):
            if k < 5 and total - k < 5 and [total-k, k] != initial:
                intra_player.append((2, (total-k, k)))
        if (2, tuple(reversed(initial))) not in intra_player and 0 not in initial and list(reversed(initial)) != initial:
            intra_player.append((2, tuple(reversed(initial))))
        # if total == 1:
        #     intra_player = []
        return inter_player + intra_player
    
    def interMove(self, action):
        self.state[1][action[1]] = (self.state[1][action[1]] + self.state[0][action[0]]) % 5
        self.state[1] = list(sorted(self.state[1], reverse=True))

    
    def move(self, action):
        """
        After making every move, the lists (perspective) in the state should flip
        """
        # print("old state:", self.state)
        if action[0] == 2:
            self.state[0] = list(sorted(action[1], reverse=True))   #HERE
        else:
            self.interMove(action)    # have a look at this
        if self.state[1] == [0, 0]:
            self.winner = self.player
        self.state = list(reversed(self.getState()))
        # print("new state:", self.state)
        self.player = self.other_player[self.player]


class ChopsticksAI:
    verbose = False
    def __init__(self, alpha=0.8, epsilon_pow=1, epsilon_thresh=-1):
        self.q = {}
        self.alpha = alpha
        self.epsilon = epsilon_pow
        self.epsi_thresh = epsilon_thresh
    
    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        if (best_future != 0 or reward != 0) and old == 0:
            if ChopsticksAI.verbose:
                print("approved for update", "old_state, action, old, reward, best_future", old_state, action, old, reward, best_future)
            self.update_q_value(old_state, action, old, reward, best_future)
    
    def get_q_value(self, state, action):
        """
        Returns the reward associated with performing an action on a state
        """
        assert type(state) == tuple and type(action) == tuple, "Invalid keys"
        # if (state, action) not in self.q:
        #     self.q[state, action] = 0.0
        return self.q.get((state, action), 0.0)
    
    def choose_action(self, state: Chopsticks, epsilon=False, not_in=()):
        """
        Tries to choose an action which leads to the exploration of a new key
        """
        if not epsilon and random.random() > self.epsi_thresh:
            if ChopsticksAI.verbose:
                print("choosing randomly...   ", end="")
            moves = []
            best_futs = []
            for i in state.get_moves():
                if self.get_q_value(state.getKeyAble(), i) == 0:
                    s = Chopsticks(state.getState())
                    s.move(i)
                    if self.best_future_reward(s.getKeyAble()) == 0 and s.getKeyAble() not in not_in:
                        moves.append(i)
                        best_futs.append(self.best_future_reward(s.getKeyAble()))
            if ChopsticksAI.verbose:
                print("possible moves and best future states from there~", moves, best_futs)
            if moves:
                return random.choice(moves)
            return random.choice(state.get_moves())
        # otherwise, find the best
        actions = state.get_moves()
        if epsilon:
            print("choosing from", actions)
            print("with rewards", [self.get_q_value(state.getKeyAble(), i) for i in actions])
        action = actions[random.randint(0, len(actions)-1)]
        best = self.get_q_value(state.getKeyAble(), action)
        if epsilon:
            print("choosing action")
        for act in actions:
            if epsilon:
                print("considering", act)
            if self.get_q_value(state.getKeyAble(), act) > best:
                if epsilon:
                    print("found new best", act)
                best = self.get_q_value(state.getKeyAble(), act)
                action = act
        return action
    
    def choose_action2(self, state: Chopsticks, epsilon=False):
        """
        Chooses the best action given a state
        """
        if not epsilon and random.random() < self.epsi_thresh:
            # print("choosing randomly...   ", end="")
            moves = [i for i in state.get_moves() if self.get_q_value(state.getKeyAble(), i) >= 0]
            return random.choice(state.get_moves())
        # otherwise, find the best
        actions = state.get_moves()
        # if epsilon:
        # print("choosing from", actions)
        # print("with rewards", [self.get_q_value(state.getKeyAble(), i) for i in actions])
        action = actions[random.randint(0, len(actions)-1)]
        best = self.get_q_value(state.getKeyAble(), action)
        # if epsilon:
        #     print("choosing action")
        for act in actions:
            # if epsilon:
            #     print("considering", act)
            if self.get_q_value(state.getKeyAble(), act) > best:
                # if epsilon:
                #     print("found new best", act)
                best = self.get_q_value(state.getKeyAble(), act)
                action = act
        return action
    
    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estiamte of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        self.q[tuple(state), action] = round(old_q + self.alpha * (reward + future_rewards - old_q), 3)
    
    def best_future_reward(self, state, k=6):
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `state`, return 0.
        """
        # best = 0
        # for action in Chopsticks.available_actions(state):
        #     f = self.get_q_value(state, action)
        #     if f == 0 and k != 1:
        #         c = Chopsticks([list(i) for i in state])
        #         c.move(action)
        #         f = self.best_future_reward(c.getKeyAble(), k-1)
        #     elif f > best:
        #         best = f
        # return best
        actions = Chopsticks.available_actions(state)
        if len(actions) == 0:
            return 0
        best = max([self.get_q_value(state, action) for action in Chopsticks.available_actions(state)])
        # for action in Chopsticks.available_actions(state):
        #     if self.get_q_value(state, action) > best:
        #         best = self.get_q_value(state, action)
        return -best
        
    

def train(n, player=None, rand=False, js_out="model.json", verbose=False):
    """
    Train an AI by playing `n` games against itself.
    """

    if player is None:
        player = ChopsticksAI()

    # Play n games
    for i in range(n):
        if verbose:
            print("current len keys:", len(player.q))
        if (i+1) % 10 == 0:
            print(f"\tPlaying training game {i + 1}")
        game = Chopsticks(rand=rand)

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }
        explored = [game.getKeyAble()]

        # Game loop
        while True:

            # Keep track of current state and action
            state = game.getKeyAble()
            action = player.choose_action(game, not_in=explored)
            if verbose:
                print("current state", state)
                print("chose:", action, "\n")


            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make move
            # print("state", state, "before action", action)
            game.move(action)
            new_state = game.getKeyAble()
            explored.append(new_state)
            if any([len(i) != 2 for i in new_state]):
                print(new_state)
                raise ValueError("This is where the error took place")

            # When game is over, update Q values with rewards
            if game.winner is not None:
                player.update(state, action, new_state, 1)
                try:
                    player.update(
                        last[game.player]["state"],
                        last[game.player]["action"],
                        new_state,
                        -1
                    )
                    if verbose:
                        print("game ended... updated q:", player.q, "\n")
                except:
                    pass
                break

            # If game is continuing, no rewards yet
            else:

                # print("game hasn't ended, updating", last[game.player]["state"], last[game.player])
                # player.update(
                #     last[game.player]["state"],
                #     last[game.player]["action"],
                #     new_state,
                #     0
                # )
                player.update(state, action, new_state, 0)
    if verbose:
        print("Done training")
        print("here's what things look like")
        print("explored", len(player.q), "keys")


    with open(js_out, "w") as f:
        json.dump({str(i):str(player.q[i]) for i in sorted(player.q)}, f, indent=2)

    # Return the trained AI
    return player


def play(ai: ChopsticksAI, human_player=None):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """

    # If no player order set, choose human's order randomly
    if human_player is None:
        human_player = random.randint(0, 1)

    # Create new game
    game = Chopsticks()

    # Game loop
    while True:

        # Print contents of hands
        print("Current player:", game.player)
        print("Hands:", game.state)
        

        # Compute available actions
        available_actions = Chopsticks.available_actions(game.getState())
        time.sleep(1)

        # Let human make a move
        if game.player == human_player:
            print("Your Turn")
            while True:
                x = input("Enter your move: should be either 2 or 3 digits\n")
                try:
                    x = tuple([int(i) for i in x])
                    if len(x) == 3:
                        x = (x[0], x[1:])
                    if x in available_actions:
                        break
                except ValueError:
                    print("Invalid move, try again.")
                else:
                    print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn")
            x = ai.choose_action(game, epsilon=True)
            print(f"AI's action: {x}")

        # Make move
        game.move(x)

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return    


def numStates(initial: list):
    frontier = [initial]
    explored = []
    win_states = 0
    while frontier:
        state = frontier.pop()
        explored.append(state)
        state = Chopsticks(state)
        for action in state.get_moves():
            other_state = Chopsticks(state.getState())
            other_state.move(action)
            if other_state.getState() not in explored and other_state.getState() not in frontier:
                explored.append(other_state.getState())
                if other_state.winner is not None:
                    win_states += 1
                    continue
                frontier.append(other_state.getState())
    return explored, win_states

class Frontier:
    def __init__(self):
        self.frontier = []
    def remove(self):
        assert not self.empty()
        out = self.frontier[0]
        self.frontier = self.frontier[1:]
        return out
    def empty(self):
        return len(self.frontier) == 0

def mapQ(depth=3):
    win_states = [((0, 0), (i, j)) for i in range(1, 5) for j in range(i+1)]
    print(win_states)
    print(len(win_states))
    frontiers = [[win_states[i]] for i in range(len(win_states))]
    explored = [[] for i in range(len(frontiers))]
    while depth > 0:
        # go through each of the frontiers one by one
        for i in range(len(frontiers)):
            state = frontiers[i].remove()
            print("actions from", state)
            for action in Chopsticks.rev_moves(state):
                new_state = Chopsticks.rev_action(state, action)
                print("new state", new_state)
                if new_state not in explored:
                    pass
                    # Continue from here!

    
# print(Chopsticks.rev_moves(((2, 0), (2, 1))))
# mapQ()

# x = Chopsticks([[2, 0], [2, 1]])
# print(x.get_moves())
# for i in range(10):
#     moves = x.get_moves()
#     print("move~", moves[0])
#     x.move(moves[0])
#     print(x.state)
#     if x.winner is not None:
#         print("player", x.winner, "won!")
#         break

keys = 0
states, wins = numStates([[1, 1], [1, 1]])
print(len(states), "number of states")
print("and", wins, "states result in a win")
for s in states:
    keys += len(Chopsticks(s).get_moves())
print("number of keys for the q_dict", keys)
print("*"*30)

trained = train(200)
play(trained)
# trained = train(1000)
# trained = train(2000)
# trained = train(4000)

# print("first batch", len(trained.q))
# trained = train(100, trained, True)
# print("second batch", len(trained.q))
# print(trained.q)
# print("contains info on", len(trained.q), "states")
# play(trained)

# s = Chopsticks()
# for i in range(10):
#     print(s.randState())
