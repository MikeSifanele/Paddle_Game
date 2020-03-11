# Paddle Game

# Import required libraries.
import turtle

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

		self.window.onkey(self.step, 's')

		# Add ball movement.
		self.ball.dx = 1

		self.ball.dy = -1

		self.hit, self.miss, self.reward, self.done = 0, 0, 0, 0
		
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
		while True:

			self.run_frame()

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

		# Paddle collision.
		if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:

			self.ball.dy *= -1

			self.hit += 1

			self.reward += 3
		
	# Create paddle object.
	def create_paddle(self):
	
		paddle = self.turtle.Turtle()

		paddle.shape('square')

		paddle.speed(0)

		paddle.shapesize(stretch_wid = 1, stretch_len = 5)

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

		self.hit, self.miss, self.done = 0, 0, 0

	def step(self, action):

		self.reward, self.done = 0, 0

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
		state = [self.paddle.xcor(), self.ball.xcor(), self.ball.ycor(), self.ball.dx, self.ball.dy]

		return self.reward, state, self.done

	# Runs the game for one frame.
	def run_frame(self):
		
		self.window.update()

		self.ball_collision()

		self.show_scoreboard()

		self.ball.setx(self.ball.xcor() + self.ball.dx)

		self.ball.sety(self.ball.ycor() + self.ball.dy)

	def play(self):

		try:
			self.render_window()
		except Exception as e:

			if 'invalid command name' in str(e):
				print("Game stopped by user!. ")

			else:
				print("Exception: ", e)

# main console

paddleGame = PaddleGame(turtle)

paddleGame.play()
