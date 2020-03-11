# Paddle Game

# Import required libraries.
import turtle

class PaddleGame:

	def __init__(self, turtle):
		
		self.turtle = turtle
		
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

			self.window.update()

			self.ball_collision()

			self.show_scoreboard()

			self.ball.setx(self.ball.xcor() + self.ball.dx)

			self.ball.sety(self.ball.ycor() + self.ball.dy)

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

		# Paddle collision.
		if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:

			self.ball.dy *= -1

			self.hit += 1
		
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
	def move_ball_right(self):

		x = self.paddle.xcor()

		if x < 230:
			self.paddle.setx(x + 25)

	def move_ball_left(self):
	
		x = self.paddle.xcor()

		if x > -230:
			self.paddle.setx(x - 25)

	def reset(self):

		self.hit, self.miss = 0, 0

	def play(self):

		self.window = self.create_window()

		self.paddle = self.create_paddle()

		self.ball = self.create_ball()

		self.scoreboard = self.create_scoreboard()

		# Add keyboard controls.
		self.window.listen()

		self.window.onkey(self.move_ball_left, 'Left')

		self.window.onkey(self.move_ball_right, 'Right')

		# Add ball movement.
		self.ball.dx = 1

		self.ball.dy = -1

		self.hit, self.miss = 0, 0

		try:
			self.render_window()
		except Exception as e:
			print("Game stopped by user!.")
		

paddleGame = PaddleGame(turtle)

paddleGame.play()
