import torch
import numpy

import test_tools
import models
import Game
import self_play

class Analyzer:
    def __init__(self):
        self.config = test_tools.SuzeroConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.SuZeroNetwork(*self.config.net_config).to(self.device)
        self.model.set_weights(torch.load(r'D:\Project\AI\SuZero\save\suzero.sv')["weights"])
        self.game = Game.Game
        self.present_game = self.game()
        self.done = False

    def start_analyze(self, num_action):
        with torch.no_grad():
            if not self.done:
                root = self_play.MCTS(self.config).run(
                    self.model,
                    self.present_game
                )
                result = self.get_best_action(root, num_action)
                return result
            return False

    def step(self, action):
        _, _, self.done = self.present_game.step(action)

    @staticmethod
    def get_best_action(node, num_action):
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        visit_counts = visit_counts / numpy.sum(visit_counts)
        actions = [action for action in node.children.keys()]
        num_action = min(len(actions), num_action)
        index_list = numpy.flip(numpy.argsort(visit_counts))[:num_action]
        action = [(actions[index], visit_counts[index]) for index in index_list]
        return action
