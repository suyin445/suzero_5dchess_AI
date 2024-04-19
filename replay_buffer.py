import copy

import numpy
import ray


@ray.remote
class ReplayBuffer:
    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )

    def save_game(self, game_history, shared_storage=None):
        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            value_batch,
            policy_batch

        ) = ([], [], [], [], [])
        for game_id, game_history in self.sample_n_games(self.config.batch_size):
            game_pos = numpy.random.choice(len(game_history.root_values))
            values, policies, actions = self.make_target(game_history, game_pos)
            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos
                )
            )
            action_batch.append(actions)
            value_batch.append(values)
            policy_batch.append(policies)
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                policy_batch
            ),
        )

    def sample_n_games(self, n_games: int):
        buffer_list = list(self.buffer.keys())
        size = len(buffer_list)
        size = min(size, n_games)
        selected_games = numpy.random.choice(buffer_list, size=size, replace=False)
        ret = [
            (game_id, self.buffer[game_id])
            for game_id in selected_games
        ]
        return ret

    def sample_game(self):
        game_index = numpy.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index
        return game_id, self.buffer[game_id]

    def compute_target_value(self, game_history, index):
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = game_history.root_values
            last_step_value = (
                root_values[bootstrap_index]
                if game_history.to_play_history[bootstrap_index] == game_history.to_play_history[index]
                else -root_values[bootstrap_index]
            )
            value = last_step_value * self.config.discount ** self.config.td_steps
        else:
            value = 0
        for i, reward in enumerate(
                game_history.reward_history[index + 1: bootstrap_index + 1]
        ):
            value += (
                         reward
                         if game_history.to_play_history[index] == game_history.to_play_history[index + i] else -reward
                     ) * self.config.discount ** i

        return value

    def make_target(self, game_history, state_index):
        value = self.compute_target_value(game_history, state_index)
        target_policies = game_history.child_visits[state_index]
        actions = game_history.legal_history[state_index]
        return value, target_policies, actions

    def get_buffer(self):
        return self.buffer
