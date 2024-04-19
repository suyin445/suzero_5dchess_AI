import numpy
import chessin5d


class Game:
    def __init__(self):
        self.game = chessin5d.State()

    def step(self, action):
        if not isinstance(action, list):
            action = list(action)
        reward = 0
        self.game.onemove(action)
        done = self.game.end
        if done:
            if self.game.winner == self.game.turn + 2:
                reward = 1
            elif self.game.winner == 1:
                reward = 0
            else:
                reward = -1
        return self.get_observation(), reward, done

    def get_observation(self):
        observation = []
        for first_axis in self.game.state:
            for second_axis, board in enumerate(self.game.state[first_axis]):
                if board is not None:
                    data_board = numpy.array(board)
                    data_board = data_board.flatten()
                    data_board = numpy.append(data_board, [first_axis, second_axis] * 32)
                    observation.append(data_board)
        return numpy.array(observation)

    def to_play(self):
        return 1 if self.game.turn == 1 else 0

    def reset(self):
        self.game = chessin5d.State()
        return self.get_observation()

    def close(self):
        pass

    def get_all_legal(self):
        return self.game.get_all_legal()
