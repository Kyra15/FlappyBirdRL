import random
from collections import defaultdict
from abc import ABC, abstractmethod
import pygame
from game import load_images, Bird, PipePair, msec_to_frames, frames_to_msec
from collections import deque


class BirdEnv:
    """
    env!!
    need to make:
    init
    reset
    render (for pygame)
    step
    get state
    """

    def __init__(self):
        pygame.init()
        self.width = 568
        self.height = 512
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Pygame Flappy Bird')
        images = load_images()
        print("make sure loading", images.keys())

        self.score = 0

        self.clock = pygame.time.Clock()
        self.FPS = 60

        self.images = load_images()

        self.bird = Bird(
            50,
            self.height // 2,
            0,
            (self.images['bird-wingup'], self.images['bird-wingdown'])
        )

        self.pipes = deque()

        self.frame_clock = 0
        self.done = False
        self.score = 0

        self.add_pipe()

    def add_pipe(self):
        pipe = PipePair(self.images['pipe-end'], self.images['pipe-body'])
        self.pipes.append(pipe)

    def reset(self):
        self.bird = Bird(50,
                         self.height // 2,
                         0,
                         (self.images['bird-wingup'],
                          self.images['bird-wingdown']))
        self.score = 0
        self.pipes.clear()

        self.add_pipe()
        self.done = False
        self.frame_clock = 0

        return self.get_state()

    def render(self):

        for x in (0, self.width / 2):
            self.display.blit(self.images['background'], (x, 0))

        self.display.blit(self.bird.image, self.bird.rect)

        for pipe in self.pipes:
            self.display.blit(pipe.image, pipe.rect)

        font = pygame.font.SysFont('Arial', 24)
        score_surf = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(score_surf, (10, 10))

    def step(self, action):
        if action == 1:
            self.bird.msec_to_climb = self.bird.CLIMB_DURATION

        self.bird.update()

        for pipe in self.pipes:
            pipe.update()

        while self.pipes and not self.pipes[0].visible:
            self.pipes.popleft()

        pipe_collision = any(p.collides_with(self.bird) for p in self.pipes)
        if pipe_collision or 0 >= self.bird.y or self.bird.y >= 512 - Bird.HEIGHT:
            self.done = True

        reward = 0
        for p in self.pipes:
            if p.x + PipePair.WIDTH < self.bird.x and not p.score_counted:
                self.score += 1
                p.score_counted = True
                reward += 1
        # reward += frames_to_msec(1)

        state = self.get_state()
        info = {}

        self.frame_clock += 1
        if self.frame_clock % int(msec_to_frames(PipePair.ADD_INTERVAL)) == 0:
            self.pipes.append(PipePair(self.images['pipe-end'], self.images['pipe-body']))

        self.render()
        pygame.display.update()

        return state, reward, self.done, info

    def get_state(self):
        bird_pos_x = self.bird.x / self.width
        bird_pos_y = self.bird.y / self.height
        # if self.bird.msec_to_climb > 0:
        #     bird_vel = -self.bird.CLIMB_SPEED
        # else:
        #     bird_vel = self.bird.SINK_SPEED

        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + PipePair.WIDTH > self.bird.x:
                next_pipe = pipe
                break

        if next_pipe is None:
            dist_to_pipe = 1.0
            height_of_pipe = 0.5
        else:
            dist_to_pipe = (next_pipe.x + PipePair.WIDTH - self.bird.x) / self.width

            height_of_pipe = next_pipe.bottom_height_px / self.height

        # idk good params uhfuhdufhu
        return bird_pos_x, bird_pos_y, dist_to_pipe, height_of_pipe


class FlappyAgent(ABC):
    @abstractmethod
    def policy(self, state, env):
        """
        Given the state and the environment, return the action
        corresponding to the agent's learned best policy.
        Used when evaluating the agent after training.
        """
        pass

    @abstractmethod
    def training_policy(self, state, env):
        """
        Given the state and the environment, return the action
        corresponding to the agent's policy.
        Used when training the agent.
        """
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        """
        Carry out any policy updates/other internal state updates.
        Does not need to return anything.
        """
        pass


class RandomFlappyAgent(FlappyAgent):
    def policy(self, state, env):
        # always 0 or 1 (flap or not)
        return random.choice([0, 1])

    def training_policy(self, state, env):
        return random.choice([0, 1])

    def learn(self, *args, **kwargs):
        pass


scores = []


def train(agent, env, episodes=1000, max_steps=10000):
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        for step in range(max_steps):
            action = agent.training_policy(state, env)
            next_state, reward, done, _ = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"ep: {ep + 1} total: {total_reward}")
        scores.append(total_reward)


class QLearningFlappyAgent:
    """Tabular Q-Learning with fixed ε, trained via self-play and reward shaping."""

    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(float)

    def normalizing_stuff(self, state):
        bird_x, bird_y, pipe_dx, pipe_height = state
        return (
            round(bird_x / 10),
            round(bird_y / 10),
            round(pipe_dx / 10),
            round(pipe_height / 10)
        )

    def policy(self, state, env):
        """Greedy policy: always make the best move"""
        state_normal = self.normalizing_stuff(state)
        q0 = self.Q[(state_normal, 0)]
        q1 = self.Q[(state_normal, 1)]
        return 0 if q0 > q1 else 1

    def training_policy(self, state, env):
        """ε-greedy: explore with probability ε."""
        state_normal = self.normalizing_stuff(state)
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        else:
            q0 = self.Q[(state_normal, 0)]
            q1 = self.Q[(state_normal, 1)]
            return 0 if q0 > q1 else 1

    def learn(self, state, action, reward, next_state, done):
        """Update Q-table with shaping penalties/bonuses."""
        state_d = self.normalizing_stuff(state)
        next_state_d = self.normalizing_stuff(next_state)

        max_next_q = max(self.Q[(next_state_d, a)] for a in [0, 1]) if not done else 0
        self.Q[(state_d, action)] = self.Q[(state_d, action)] + self.alpha * (reward + self.gamma * max_next_q - self.Q[(state_d, action)])


class QLearningDecayAgent(QLearningFlappyAgent):
    """Q-Learning with ε decay."""

    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.5, decay=0.995, min_eps=0.05):
        super().__init__(alpha, gamma, epsilon)
        self.decay = decay
        self.min_eps = min_eps

    def decay_epsilon(self):
        """
        Decay epsilon by multiplying it by self.decay
        but don't go below min_eps.
        """
        if self.epsilon * self.decay > self.min_eps:
            self.epsilon *= self.decay

    def learn(self, state, action, reward, next_state, done):
        super().learn(state, action, reward, next_state, done)
        self.decay_epsilon()


env = BirdEnv()
agent = QLearningDecayAgent()
# agent = QLearningFlappyAgent()
train(agent, env, episodes=100)
print(f"max: {max(scores)}")
