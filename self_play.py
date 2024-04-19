import torch
import numpy
import math
import copy
import time
from torch.multiprocessing import Process

import models
import replay_buffer


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior.item() if isinstance(prior, torch.Tensor) else prior
        self.value_sum = 0
        self.children = {}
        self.game_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, game_state):
        self.to_play = to_play
        self.reward = reward
        self.game_state = game_state

        policy_values = torch.softmax(
            policy_logits, dim=0
        ).tolist()
        policy = {tuple(a): policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)


class MCTS:
    def __init__(self, config):
        self.config = config

    def run(self, model, game_state, root_node=None):
        device = next(model.parameters()).device
        if not root_node:
            root = Node(0)
            game_state = copy.deepcopy(game_state)
            observation = game_state.get_observation()
            observation = torch.tensor(observation).float().unsqueeze(0).to(device)
            hidden_state = model.representation(observation)
            actions = game_state.get_all_legal()
            num_legal = len(actions)
            action_tensor = torch.unsqueeze(torch.tensor(actions, dtype=torch.float, device=device), dim=0)
            value, policy_logits = model.prediction_network(hidden_state, action_tensor)
            root.expand(actions, game_state.to_play(), 0, torch.squeeze(policy_logits, dim=0).cpu(), game_state)
        else:
            root = root_node
            if not root.expanded():
                root = Node(0)
                game_state = copy.deepcopy(game_state)
                observation = game_state.get_observation()
                observation = torch.tensor(observation).float().unsqueeze(0).to(device)
                hidden_state = model.representation(observation)
                actions = game_state.get_all_legal()
                num_legal = len(actions)
                action_tensor = torch.unsqueeze(torch.tensor(actions, dtype=torch.float, device=device), dim=0)
                value, policy_logits = model.prediction_network(hidden_state, action_tensor)
                root.expand(actions, game_state.to_play(), 0, torch.squeeze(policy_logits, dim=0).cpu(), game_state)
            else:
                num_legal = len(list(root.children.keys()))
        min_max_stats = MinMaxStats()
        max_tree_depth = 0
        now_num_simulations = max(self.config.num_simulations, int(num_legal / self.config.simulations_ratio))
        now_num_simulations = min(now_num_simulations, self.config.max_simulations)
        print('现在蒙树数为: ', now_num_simulations)
        print('现在合法步为: ', num_legal)
        for _ in range(now_num_simulations):
            node = root
            search_path = [node]
            current_tree_depth = 0
            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)
            parent = search_path[-2]
            game_state = copy.deepcopy(parent.game_state)
            observation, reward, done = game_state.step(action)
            virtual_to_play = game_state.to_play()
            observation = torch.tensor(observation).float().unsqueeze(0).to(device)
            hidden_state = model.representation(observation)
            actions = game_state.get_all_legal()
            action_tensor = torch.unsqueeze(torch.tensor(actions, dtype=torch.float, device=device), dim=0)
            value, policy_logits = model.prediction_network(hidden_state, action_tensor)

            node.expand(actions, virtual_to_play, reward, torch.squeeze(policy_logits, dim=0).cpu(), game_state)
            self.backpropagate(search_path, torch.squeeze(value, dim=0).cpu(), virtual_to_play, min_max_stats)
            max_tree_depth = max(max_tree_depth, current_tree_depth)
        return root

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        value = value.item()
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.reward + self.config.discount * -node.value())

            value = (
                        -node.reward if node.to_play == to_play else node.reward
                    ) + self.config.discount * value

    def select_child(self, node, min_max_stats):
        ucb = numpy.array([self.ucb_score(node, child, min_max_stats) for action, child in node.children.items()])
        ucb = torch.from_numpy(ucb)
        ucb = torch.softmax(ucb, dim=0)
        ucb = ucb.numpy()
        index = numpy.random.choice(range(len(ucb)), p=ucb)
        action = list(node.children.keys())[index]
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        pb_c = math.log(
            (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
        ) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            value_score = child.reward
            if child.to_play != -1:
                if child.to_play == parent.to_play:
                    value_score += self.config.discount * child.value()
                else:
                    value_score -= self.config.discount * child.value()
            value_score = min_max_stats.normalize(value_score)
        else:
            value_score = 0

        return prior_score + value_score


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class GameHistory:
    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.legal_history = []
        self.child_visits = []
        self.root_values = []

    def store_search_statistics(self, root):
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            action_space = list(root.children.keys())
            complete_child_visits = numpy.array([
                root.children[a].visit_count / sum_visits
                for a in action_space
            ])
            self.child_visits.append(complete_child_visits)

            self.legal_history.append(action_space)
            self.root_values.append(root.value())
        else:
            self.root_values.append(None)
            self.legal_history.append(None)

    def get_stacked_observations(self, index):
        index = index % len(self.observation_history)
        stacked_observations = self.observation_history[index].copy()
        return stacked_observations


class Selfplay(Process):
    def __init__(self, Game, config, shared_dict, id_):
        super().__init__(daemon=True)
        self.config = config
        self.game = Game()
        self.model = models.SuZeroNetwork(*self.config.net_config)
        self.model.set_weights(shared_dict["weights"])
        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()
        self.shared_dict = shared_dict
        self.id_ = id_

    def run(self):
        self.continuous_self_play(self.shared_dict)

    def continuous_self_play(self, shared_dict):
        while shared_dict['checkpoint']["training_step"] < self.config.training_steps:
            self.model.set_weights(shared_dict["weights"])
            game_history = self.play_game()

            checkpoint = shared_dict['checkpoint']
            checkpoint["episode_length"] = len(game_history.action_history) - 1
            checkpoint["total_reward"] = sum(game_history.reward_history)
            checkpoint["mean_value"] = numpy.mean(
                        [value for value in game_history.root_values if value]
                    )
            shared_dict['checkpoint'] = checkpoint

            replay_buffer.save_game(shared_dict, game_history, self.config)
            print('进行了一次selfplay, id:', self.id_)

            if self.config.ratio:
                while (shared_dict['checkpoint']["num_played_steps"]
                        / max(1, shared_dict['checkpoint']["training_step"])
                        > self.config.ratio) and \
                        shared_dict['checkpoint']["training_step"] < self.config.training_steps:
                    time.sleep(0.5)

    def play_game(self):
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())
        game_state_list = [copy.deepcopy(self.game)]

        done = False
        next_node = None

        with torch.no_grad():
            while (
                    not done and len(game_history.action_history) <= self.config.max_moves
            ):
                game_state = copy.deepcopy(game_state_list[-1])
                root = MCTS(self.config).run(
                    self.model,
                    game_state,
                    next_node
                )
                action = self.select_action(root)
                next_node = root.children[action]
                observation, reward, done = game_state.step(action)
                print('进行了一步', action, ' id:', self.id_)
                game_state_list.append(copy.deepcopy(game_state))
                game_history.store_search_statistics(root)
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(game_state.to_play())
        return game_history

    def close_game(self):
        self.game.close()

    @staticmethod
    def select_action(node):
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        p = visit_counts / numpy.sum(visit_counts)
        index = numpy.random.choice(range(len(actions)), p=p)
        action = actions[index]

        return action
