import random
import numpy as np
import pickle
import time
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque

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
        self.seen_cards = np.concatenate((self.seen_cards, np.zeros(len(self.game_deck) - 15 - len(self.seen_cards))))
        self.cards_left = self.game_deck[self.seen_split:len(self.game_deck)]
        self.player_hand = np.array([])
        self.dealer_hand = np.array([])
        self.player_sum = 0
        self.dealer_sum = 0

        # Deal initial cards
        for _ in range(2):
            self.new_player_card = self.draw_card(True)
            self.player_hand = np.append(self.player_hand, self.new_player_card)
        self.new_dealer_card = self.draw_card(True)
        self.dealer_hand = np.append(self.dealer_hand, self.new_dealer_card)
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
                return self.get_state(), -1, True, {}
            else:
                return self.get_state(), 0, False, {}

        elif action == 1:

            while self.dealer_sum < 17:
                self.dealer_hand = np.append(self.dealer_hand, self.draw_card(False))
                self.update_sums()

            if self.dealer_sum > 21 or self.dealer_sum < self.player_sum:
                return self.get_state(), 1, True, {}
            elif self.dealer_sum > self.player_sum:
                return self.get_state(), -1, True, {}
            else:
                return self.get_state(), 0, True, {}
        
        elif action == 2:
            self.player_hand = np.append(self.player_hand, self.draw_card(True))
            self.update_sums()
            if self.player_sum > 21:
                return self.get_state(), -2, True, {}  # player loses double the bet
            else:
                while self.dealer_sum < 17:
                    self.dealer_hand = np.append(self.dealer_hand, self.draw_card(False))
                    self.update_sums()

                if self.dealer_sum > 21 or self.dealer_sum < self.player_sum:
                    return self.get_state(), 2, True, {}  # player wins double the bet
                elif self.dealer_sum > self.player_sum:
                    return self.get_state(), -2, True, {}  # player loses double the bet
                else:
                    return self.get_state(), 0, True, {}  # push, but the player has risked double
        else:
            print(action)
            raise ValueError("Invalid action")


    def get_state(self):
        max_hand_size = 11  # Max number of cards in a hand without busting
        player_hand_values = self.pad_hand(self.player_hand, max_hand_size)
        dealer_hand_values = self.pad_hand(self.dealer_hand, max_hand_size)
        seen_cards_padded = np.concatenate((self.seen_cards, np.zeros(max_hand_size * 2 - len(self.seen_cards))))
        return (player_hand_values, dealer_hand_values, seen_cards_padded)

    def pad_hand(self, hand, max_size):
        hand_values = np.array([self.card_to_value(x) for x in hand] + [0] * (max_size - len(hand)))
        return hand_values




    def render(self):
        print(f"Player's Hand: {self.player_hand} ({self.player_sum})")
        print(f"Dealer's Card: {self.dealer_hand[0]}")


class DQNAgent:
    def __init__(self, state_size, action_size, alpha=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model = self._build_model()
        self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=alpha))

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.vstack([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.vstack([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        targets = self.model.predict(states)
        next_Q_values = self.model.predict(next_states)
        targets[range(self.batch_size), actions] = rewards + self.gamma * np.max(next_Q_values, axis=1) * (1 - dones)

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Create the Blackjack environment
env = BlackjackEnv(num_decks=6)
state_size = 4
action_size = 3

# Create the DQN agent
agent = DQNAgent(state_size, action_size)

# Training the DQN agent
NUM_EPISODES = 1000
for episode in range(NUM_EPISODES):
    state = env.reset()
    print("State before reshaping:", state)
    flattened_state = [item for sublist in state for item in sublist]
    # Convert the flattened list to a numpy array
    flattened_state = np.array(flattened_state, dtype=np.float32)
    # Reshape the numpy array
    state_size = len(flattened_state)
    state = np.reshape(flattened_state, [1, state_size])

    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        flattened_next_state = [item for sublist in next_state for item in sublist]  # Flatten next_state
        flattened_next_state = np.array(flattened_next_state, dtype=np.float32)  # Convert the list to a NumPy array
        next_state_size = len(flattened_next_state)  # Get the size of the flattened array
        next_state = np.reshape(flattened_next_state, [1, next_state_size])  # Reshape the array
        agent.remember(state, action, reward, next_state, done)
        state = next_state

    agent.replay()

# Testing the DQN agent
def test_agent(env, agent, episodes=100):
    wins = 0
    for episode in range(NUM_EPISODES):
        state = env.reset()
        state = np.concatenate(state)  # Flatten the state tuple into a 1D array
        state = np.reshape(state, [1, -1])
        done=False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            if done:
                wins += reward
    win_rate = wins / episodes
    return win_rate

win_rate = test_agent(env, agent, episodes=100)
print(f"Win rate: {win_rate:.2f}")
