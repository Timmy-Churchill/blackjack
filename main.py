import random
import numpy as np
import pickle
import time


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
            self.seen_cards = np.append(self.seen_cards, card)
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
        if action == 'hit':

            self.player_hand = np.append(self.player_hand, self.draw_card(False))
            self.update_sums()
            if self.player_sum > 21:
                return self.get_state(), -1, True, {}
            else:
                return self.get_state(), 0, False, {}

        elif action == 'stand':

            while self.dealer_sum < 17:
                self.dealer_hand = np.append(self.dealer_hand, self.draw_card(False))
                self.update_sums()

            if self.dealer_sum > 21 or self.dealer_sum < self.player_sum:
                return self.get_state(), 1, True, {}
            elif self.dealer_sum > self.player_sum:
                return self.get_state(), -1, True, {}
            else:
                return self.get_state(), 0, True, {}
        
        elif action == 'double':
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

    def get_state(self):

        return (tuple(self.player_hand), tuple(self.dealer_hand), tuple(sorted(self.seen_cards)))


    def render(self):
        print(f"Player's Hand: {self.player_hand} ({self.player_sum})")
        print(f"Dealer's Card: {self.dealer_hand[0]}")


class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.3):
        self.Q = {}
        self.actions = actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration factor

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_values = [self.get_q_value(state, action) for action in self.actions]
            return self.actions[np.argmax(q_values)]

    def get_q_value(self, state, action):
        return self.Q.get((state, action), 0.0)

    def learn(self, state, action, reward, next_state, done):
        q_value = self.get_q_value(state, action)
        if done:
            target = reward
        else:
            next_max_q_value = max([self.get_q_value(next_state, a) for a in self.actions])
            target = reward + self.gamma * next_max_q_value
        self.Q[(state, action)] = q_value + self.alpha * (target - q_value)

# Create the Blackjack environment
env = BlackjackEnv(num_decks=6)

# Create the Q-learning agent
agent = QLearningAgent(actions=['hit', 'stand', 'double'])

if input("Train new bot? (y/n)") == "y":
    iterationNumber = input("Iteration Number:  ")
    NUM_EPISODES = int(input("Number of episodes: "))
    startTime = time.time()
    timeCalculated = False
    for episode in range(NUM_EPISODES+1):
        if not(timeCalculated):
            if episode == 10000:
                elapsed_time = time.time() - startTime
                print((int(elapsed_time))*NUM_EPISODES/10000)
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state


    with open(f"agent_q_values{iterationNumber}.pkl", "wb") as file:
        pickle.dump(agent.Q, file)
else:
    iterationNumber = input("Iteration Number:  ")
    file_path = f"agent_q_values{iterationNumber}.pkl"
    try:
        with open(file_path, "rb") as file:
            print("before loaded q values")
            loaded_q_values = pickle.load(file)
            print("after loaded q values")
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"Error loading Q-values: {e}")

# Now, you can initialize your agent with these Q-values.
    agent.Q = loaded_q_values

max_allowed_steps = 100  # Or another appropriate value


def test_agent(env, agent, episodes=1000):
    print("test agent")
    wins = 0
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0  # Track the number of steps taken in this episode
        while not done:
            print("Testing episode:", episode, "Step:", steps)  # Debug print
            action = agent.get_action(state) 
            state, reward, done, _ = env.step(action)
            print(f"Reward: {reward}")
            steps += 1
            if steps > max_allowed_steps:  # Prevent infinite loops
                print("Max steps reached, ending episode")
                done = True
            if done:
                wins += reward
    win_rate = wins / episodes
    return win_rate


# Make sure to set epsilon=0 for testing to ensure deterministic behavior
agent.epsilon = 0
win_rate = test_agent(env, agent, episodes=1000)
print(f"winings: {win_rate:.2f}")


