# Paddle Game

# Import required libraries.
import turtle

import random

import numpy as np

import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras import backend as K

class PaddleGame:

	def __init__(self, turtle):
		
		self.turtle = turtle

		self.window = self.create_window()

		self.paddle = self.create_paddle()

		self.ball = self.create_ball()

		self.scoreboard = self.create_scoreboard()

		# Add keyboard controls.
		self.window.listen()

		self.window.onkey(self.move_paddle_left, 'Left')

		self.window.onkey(self.move_paddle_right, 'Right')

		# Add ball movement.
		self.ball.dx = 2

		self.ball.dy = -2

		self.hit, self.miss, self.reward, self.done, self.episodes = 0, 0, 0, False, 0

	# Create display.
	def create_window(self):

		window = self.turtle.Screen()

		window.title('Paddle')

		window.bgcolor('black')

		window.tracer(0)

		window.setup(width = 600, height = 600)

		return window

	# Render window.
	def render_window(self):
		# Update screen continuously.
		while self.episodes < self.max_episodes:

			while self.done == False:
				
				self.run_frame()

			self.episodes += 1

			self.reset()

	# Add ball collision.
	def ball_collision(self):
		# Right wall.
		if self.ball.xcor() > 290:

			self.ball.setx(290)

			self.ball.dx *= -1

		# Left wall.
		if self.ball.xcor() < -290:

			self.ball.setx(-290)

			self.ball.dx *= -1

		# Top wall.
		if self.ball.ycor() > 290:

			self.ball.sety(290)

			self.ball.dy *= -1

		# Bottom wall.
		if self.ball.ycor() < -290:

			self.ball.goto(0, 100)

			self.miss += 1

			self.reward -= 3

			self.ball.goto(random.randint(-10, 50), random.randint(-150, 150))

			self.paddle.goto(random.randint(-150, 150), -275)

		# Paddle collision.
		if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 22:

			self.ball.dy *= -1

			self.hit += 1

			self.reward += 3

		if self.hit == 50:

			self.done = True

			self.reward = 50

		elif self.miss == 50:

			self.done = True

			self.reward = -50

						
		
	# Create paddle object.
	def create_paddle(self):
	
		paddle = self.turtle.Turtle()

		paddle.shape('square')

		paddle.speed(0)

		paddle.shapesize(stretch_wid = 1, stretch_len = 2)

		paddle.penup()

		paddle.color('white')

		paddle.goto(0, -275)

		return paddle
		
	# Show scoreboard.
	def show_scoreboard(self):

		self.scoreboard.clear()

		self.scoreboard.write(f"Hit:  {self.hit}   Miss:  {self.miss}", align = 'center', font = ('courier', 24, 'normal'))

	# Create scoreboard object.
	def create_scoreboard(self):

		scoreboard = self.turtle.Turtle()

		scoreboard.speed(0)

		scoreboard.color('white')

		scoreboard.hideturtle()

		scoreboard.goto(0, 250)

		scoreboard.penup()

		return scoreboard

	# Create ball object.
	def create_ball(self):
		
		ball = self.turtle.Turtle()

		ball.shape('circle')

		ball.color('red')

		ball.penup()

		ball.goto(0, 100)

		return ball

	# Create movement/ action methods.
	def move_paddle_right(self):

		x = self.paddle.xcor()

		if x < 230:
			self.paddle.setx(x + 25)

	def move_paddle_left(self):
	
		x = self.paddle.xcor()

		if x > -230:
			self.paddle.setx(x - 25)

	def reset(self):

		self.hit, self.miss, self.done = 0, 0, False

		self.ball.goto(random.randint(-10, 50), random.randint(-150, 150))

		self.paddle.goto(random.randint(-150, 150), -275)

		return [self.paddle.xcor(), self.paddle.ycor(), self.ball.xcor(), self.ball.ycor(), self.ball.dx, self.ball.dy]


	def step(self, action):

		self.reward = 0

		# Move left.
		if action == 0:
			self.move_paddle_left()

			self.reward = -.1

		# Move right
		elif action == 2:
			self.move_paddle_right()

			self.reward = -.1

		self.run_frame()

		# create the state vector
		state = [self.paddle.xcor(), self.paddle.ycor(), self.ball.xcor(), self.ball.ycor(), self.ball.dx, self.ball.dy]

		return state, self.reward, self.done, ["action: " + str(action), "reward: " + str(self.reward), "done: " + str(self.done)]

	# Runs the game for one frame.
	def run_frame(self):
		
		self.window.update()

		self.ball_collision()

		self.show_scoreboard()

		self.ball.setx(self.ball.xcor() + self.ball.dx)

		self.ball.sety(self.ball.ycor() + self.ball.dy)


	def play(self, max_episodes):

		self.max_episodes = max_episodes

		try:
			self.render_window()
		except Exception as e:

			if 'invalid command name' in str(e):
				print("Game stopped by user!. ")

			else:
				print("Exception: ", e)

	def agent_play(self, max_episodes):

		self.max_episodes = max_episodes

		try:

			while self.episodes < self.max_episodes:

				while self.done == False:
				
					print("observation: ", self.step(2))

				self.episodes += 1

				self.reset()

		except Exception as e:

			if 'invalid command name' in str(e):
				print("Game stopped by user!. ")

			else:
				print("Exception: ", e)


def build_networks(state_shape, action_size, learning_rate, critic_weight, hidden_neurons, entropy):
	"""Creates Actor Critic Neural Networks.

	Creates a two hidden-layer Policy Gradient Neural Network. The loss
	unction is altered to be a log-likelihood function weighted
	by an action's advantage.

	Args:
		space_shape: a tuple of ints representing the observation space.
		action_size (int): the number of possible actions.
		learning_rate (float): the nueral network's learning rate.
		critic_weight (float): how much to weigh the critic's training loss.
		hidden_neurons (int): the number of neurons to use per hidden layer.
		entropy (float): how much to enourage exploration versus exploitation.
	"""
	state_input = layers.Input(state_shape, name='frames')
	advantages = layers.Input((1,), name='advantages')  # PG, A instead of G

	# PG
	actor_1 = layers.Dense(hidden_neurons, activation='relu')(state_input)
	actor_2 = layers.Dense(hidden_neurons, activation='relu')(actor_1)
	probabilities = layers.Dense(action_size, activation='softmax')(actor_2)

	# DQN
	critic_1 = layers.Dense(hidden_neurons, activation='relu')(state_input)
	critic_2 = layers.Dense(hidden_neurons, activation='relu')(critic_1)
	values = layers.Dense(1, activation='linear')(critic_2)

	def actor_loss(y_true, y_pred):  # PG
		y_pred_clipped = K.clip(y_pred, CLIP_EDGE, 1-CLIP_EDGE)
		log_lik = y_true*K.log(y_pred_clipped)
		entropy_loss = y_pred * K.log(K.clip(y_pred, CLIP_EDGE, 1-CLIP_EDGE))  # New

		return K.sum(-log_lik * advantages) - (entropy * K.sum(entropy_loss))

	# Train both actor and critic at the same time.
	actor = models.Model( inputs=[state_input, advantages], outputs=[probabilities, values])

	actor.compile( loss=[actor_loss, 'mean_squared_error'],  # [PG, DQN]
	loss_weights=[1, critic_weight],  # [PG, DQN]
	optimizer=tf.keras.optimizers.Adam(lr=learning_rate))

	critic = models.Model(inputs=[state_input], outputs=[values])
	policy = models.Model(inputs=[state_input], outputs=[probabilities])

	return actor, critic, policy
		
class Memory():
	"""Sets up a memory replay for actor-critic training.

	Args:
		gamma (float): The "discount rate" used to assess state values.
		batch_size (int): The number of elements to include in the buffer.
	"""
	def __init__(self, gamma, batch_size):
		self.buffer = []
		self.gamma = gamma
		self.batch_size = batch_size

	def add(self, experience):
		"""Adds an experience into the memory buffer.

		Args:
			experience: (state, action, reward, state_prime_value, done) tuple.
		"""
		self.buffer.append(experience)

	def check_full(self):
		return len(self.buffer) >= self.batch_size

	def sample(self):
		"""Returns formated experiences and clears the buffer.

		Returns:
			(list): A tuple of lists with structure [
				[states], [actions], [rewards], [state_prime_values], [dones]
			]
		"""
		# Columns have different data types, so numpy array would be awkward.
		batch = np.array(self.buffer).T.tolist()

		states_mb = np.array(batch[0], dtype=np.float32)
		actions_mb = np.array(batch[1], dtype=np.int8)
		rewards_mb = np.array(batch[2], dtype=np.float32)
		dones_mb = np.array(batch[3], dtype=np.int8)
		value_mb = np.squeeze(np.array(batch[4], dtype=np.float32))
		self.buffer = []
		return states_mb, actions_mb, rewards_mb, dones_mb, value_mb

class Agent():
	"""Sets up a reinforcement learning agent to play in a game environment."""
	def __init__(self, actor, critic, policy, memory, action_size):
		"""Initializes the agent with DQN and memory sub-classes.

		Args:
			network: A neural network created from deep_q_network().
			memory: A Memory class object.
			epsilon_decay (float): The rate at which to decay random actions.
			action_size (int): The number of possible actions to take.
		"""
		self.actor = actor
		self.critic = critic
		self.policy = policy
		self.action_size = action_size
		self.memory = memory

	def act(self, state):
		"""Selects an action for the agent to take given a game state.

		Args:
			state (list of numbers): The state of the environment to act on.
			traning (bool): True if the agent is training.

		Returns:
			(int) The index of the action to take.
		"""
		# If not acting randomly, take action with highest predicted value.
		state_batch = np.expand_dims(state, axis=0)
		probabilities = self.policy.predict(state_batch)[0]
		action = np.random.choice(self.action_size, p=probabilities)
		return action

	def learn(self, print_variables=False):
		"""Trains the Deep Q Network based on stored experiences."""
		gamma = self.memory.gamma
		experiences = self.memory.sample()
		state_mb, action_mb, reward_mb, dones_mb, next_value = experiences

		# One hot enocde actions
		actions = np.zeros([len(action_mb), self.action_size])
		actions[np.arange(len(action_mb)), action_mb] = 1

		#Apply TD(0)
		discount_mb = reward_mb + next_value * gamma * (1 - dones_mb)
		state_values = self.critic.predict([state_mb])
		advantages = discount_mb - np.squeeze(state_values)

		if print_variables:
			print("discount_mb", discount_mb)
			print("next_value", next_value)
			print("state_values", state_values)
			print("advantages", advantages)
		else:
			self.actor.train_on_batch( [state_mb, advantages], [actions, discount_mb])


env = PaddleGame(turtle)

env.reset()

CLIP_EDGE = 1e-8

# Change me please.
test_gamma = .7
test_batch_size = 32
test_learning_rate = .1
test_hidden_neurons = 1000
test_critic_weight = 0.5
test_entropy = 0.0001

space_shape = (6,) # Number of state variables(turple)
action_size = 3
test_gamma = .5

# Feel free to play with these
test_learning_rate = .2
test_hidden_neurons = 10

test_memory = Memory(test_gamma, test_batch_size)

test_actor, test_critic, test_policy = build_networks(
	space_shape, action_size,
	test_learning_rate, test_critic_weight,
	test_hidden_neurons, test_entropy)

test_agent = Agent(test_actor, test_critic, test_policy, test_memory, action_size)

state = env.reset()
episode_reward = 0
done = False

while done == False:
	action = test_agent.act(state)
	state_prime, reward, done, _ = env.step(action)
	episode_reward += reward

	next_value = test_agent.critic.predict([[state_prime]])
	test_agent.memory.add((state, action, reward, done, next_value))
	state = state_prime

test_agent.learn(print_variables=True)

with tf.Graph().as_default():
	test_memory = Memory(test_gamma, test_batch_size)
	test_actor, test_critic, test_policy = build_networks(
		space_shape, action_size,
		test_learning_rate, test_critic_weight,
		test_hidden_neurons, test_entropy)

	test_agent = Agent(
		test_actor, test_critic, test_policy, test_memory, action_size)
	for episode in range(200):  
		state = env.reset()
		episode_reward = 0
		done = False

		while done == False:
			action = test_agent.act(state)
			state_prime, reward, done, _ = env.step(action)
			episode_reward += reward

			next_value = test_agent.critic.predict([[state_prime]]) 
			test_agent.memory.add((state, action, reward, done, next_value))

			#if test_agent.memory.check_full():
			#test_agent.learn(print_variables=True)
			state = state_prime

		test_agent.learn(print_variables=True)
		print("Episode", episode, "Score =", episode_reward)


# main console

"""paddleGame = PaddleGame(turtle)

paddleGame.reset()

paddleGame.agent_play(5)"""

