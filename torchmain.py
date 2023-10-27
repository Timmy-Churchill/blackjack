import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import time
import pickle

class BlackjackEnv:
    def __init__(self, num_decks=6):
        self.num_decks = num_decks
        self.cards = np.array([])
        self.reset()

    def reset(self):
        # Initialize the deck with num_decks of cards
        self.cards = np.array(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * 4 * self.num_decks)
        np.random.shuffle(self.cards)  # Use numpy's shuffle for numpy arrays
        self.percent = random.randint(25, 76)
        self.game_deck = self.cards[:(self.num_decks-1)*52]
        self.seen_split = random.randint(0, len(self.game_deck)-15)
        self.seen_cards_strings = self.game_deck[0:self.seen_split]
        self.seen_cards =(self.cards_to_values(self.seen_cards_strings))
        self.cards_left = self.game_deck[self.seen_split:len(self.game_deck)]
        self.player_hand = np.array([])
        self.dealer_hand = np.array([])
        self.player_sum = 0
        self.dealer_sum = 0

        # Deal initial cards
        for _ in range(2):
            self.new_player_card = self.draw_card(True)
            self.player_hand = np.append(self.player_hand, self.new_player_card)
        self.dealer_hand = np.append(self.dealer_hand, self.draw_card(True))
        self.dealer_hand = np.append(self.dealer_hand, self.draw_card(False))
        self.update_sums()

        return self.get_state()

    def random_pop(self, arr):
        if arr.size == 0:
            raise ValueError("The array is empty")
        
        random_index = np.random.randint(0, arr.size)
        random_value = arr[random_index]
        
        # Remove the element at the selected index
        arr = np.delete(arr, random_index)
        
        return random_value, arr

    def draw_card(self, seen):
        card, self.cards_left = self.random_pop(self.cards_left)
        if seen:
            self.seen_cards = np.append(self.seen_cards, self.card_to_value(card))
        return card

    def cards_to_values(self, cards):
        card_values = {
            'A': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5': 5,
            '6': 6,
            '7': 7,
            '8': 8,
            '9': 9,
            '10': 10,
            'J': 11,
            'Q': 12,
            'K': 13,
        }
        return np.array([card_values[card] for card in cards])

    def card_to_value(self, card):
        card_values = {
            'A': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5': 5,
            '6': 6,
            '7': 7,
            '8': 8,
            '9': 9,
            '10': 10,
            'J': 11,
            'Q': 12,
            'K': 13,
        }
        return card_values[card]

    def update_sums(self):
        self.player_sum = self.calculate_hand_sum(self.player_hand)
        self.dealer_sum = self.calculate_hand_sum(self.dealer_hand)


    def calculate_hand_sum(self, hand):
        sum_hand = 0
        num_aces = 0

        for card in hand:
            if card.isdigit():
                sum_hand += int(card)
            elif card in ('K', 'Q', 'J'):
                sum_hand += 10
            elif card == 'A':
                num_aces += 1
                sum_hand += 11

        while sum_hand > 21 and num_aces > 0:
            sum_hand -= 10
            num_aces -= 1

        return sum_hand

    def step(self, action):
        if action == 0:
            self.player_hand = np.append(self.player_hand, self.draw_card(False))
            self.update_sums()
            if self.player_sum > 21:
                #print(f"ACTION: HIT,  HAND: {self.player_hand}={self.player_sum}, DEALER: {self.dealer_hand} = {self.dealer_sum}, REWARD: -1")
                return self.get_state(), -1, True, {}
            else:
                #print(f"ACTION: HIT,  HAND: {self.player_hand}={self.player_sum}, DEALER: {self.dealer_hand} = {self.dealer_sum}, REWARD: 0")
                return self.get_state(), 0, False, {}

        elif action == 1:

            while self.dealer_sum < 17:
                self.dealer_hand = np.append(self.dealer_hand, self.draw_card(False))
                self.update_sums()

            if self.dealer_sum > 21 or self.dealer_sum < self.player_sum:
                #print(f"ACTION: STAND,  HAND: {self.player_hand}={self.player_sum}, DEALER: {self.dealer_hand} = {self.dealer_sum}, REWARD: 1")
                return self.get_state(), 1, True, {}
            elif self.dealer_sum > self.player_sum:
                #print(f"ACTION: STAND,  HAND: {self.player_hand}={self.player_sum}, DEALER: {self.dealer_hand} = {self.dealer_sum}, REWARD: -1")
                return self.get_state(), -1, True, {}
            else:
                #print(f"ACTION: STAND,  HAND: {self.player_hand}={self.player_sum}, DEALER: {self.dealer_hand} = {self.dealer_sum}, REWARD: 0")
                return self.get_state(), 0, True, {}
        
        elif action == 2:
            self.player_hand = np.append(self.player_hand, self.draw_card(True))
            self.update_sums()
            if self.player_sum > 21:
                #print(f"ACTION: DOUBLE,  HAND: {self.player_hand}={self.player_sum}, DEALER: {self.dealer_hand} = {self.dealer_sum}, REWARD: -2")
                return self.get_state(), -2, True, {}  # player loses double the bet
            
            else:
                while self.dealer_sum < 17:
                    self.dealer_hand = np.append(self.dealer_hand, self.draw_card(False))
                    self.update_sums()

                if self.dealer_sum > 21 or self.dealer_sum < self.player_sum:
                    #print(f"ACTION: DOUBLE,  HAND: {self.player_hand}={self.player_sum}, DEALER: {self.dealer_hand} = {self.dealer_sum}, REWARD: 2")
                    return self.get_state(), 2, True, {}  # player wins double the bet
                elif self.dealer_sum > self.player_sum:
                    #print(f"ACTION: DOUBLE,  HAND: {self.player_hand}={self.player_sum}, DEALER: {self.dealer_hand} = {self.dealer_sum}, REWARD: -2")
                    return self.get_state(), -2, True, {}  # player loses double the bet
                else:
                    #print(f"ACTION: DOUBLE,  HAND: {self.player_hand}={self.player_sum}, DEALER: {self.dealer_hand} = {self.dealer_sum}, REWARD: 0")
                    return self.get_state(), 0, True, {}  # push, but the player has risked double

    def get_state(self):
        self.tuple_of_tuples= (tuple(self.cards_to_values(self.player_hand)), (self.card_to_value(self.dealer_hand[0]),), tuple(sorted(self.seen_cards)))
        self.flat_list = [item for subtuple in self.tuple_of_tuples for item in subtuple]
        self.flat_array = np.array(self.flat_list)
        self.zero_array = np.zeros(270-len(self.flat_list))
        self.flat_array = np.concatenate((self.flat_array, self.zero_array))
        return self.flat_array


    def render(self):
        print(f"Player's Hand: {self.player_hand} ({self.player_sum})")
        print(f"Dealer's Card: {self.dealer_hand[0]}")


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


env = BlackjackEnv(num_decks=6)
state_size = 270
action_size = 3
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32
EPISODES = 10000

if input("Train new bot? (y/n)") == "y":
    iteration_number = input("iteration number: ")
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):   #should this be while not done?
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10   #why is this -10?
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    agent.save(f"blackjack-dqn{iteration_number}.h5")
else:
    iteration_number = input("iteration number: ")
    agent.load(f"blackjack-dqn{iteration_number}.h5")

# Testing the agent
wins = 0
reps=100
for e in range(reps):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = np.reshape(next_state, [1, state_size])
        if done:
            wins += reward

print(f"PROFIT: {wins/reps}")
