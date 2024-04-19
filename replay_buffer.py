import numpy


def save_game(shared_dict, game_history, config):
    checkpoint = shared_dict['checkpoint']
    num_played_games = checkpoint["num_played_games"]
    important_ratio = shared_dict['important_ratio']
    if game_history.reward_history[-1] == 0:
        important_ratio[0] += 1
        buffer = shared_dict['buffer']
        buffer[num_played_games] = game_history
        if config.replay_buffer_size < len(buffer):
            del_id = min(list(buffer.keys()))
            del buffer[del_id]
        shared_dict['buffer'] = buffer
    else:
        important_ratio[1] += 1
        important_buffer = shared_dict['important_buffer']
        important_buffer[num_played_games] = game_history
        if config.replay_buffer_size < len(important_buffer):
            del_id = min(list(important_buffer.keys()))
            del important_buffer[del_id]
        shared_dict['important_buffer'] = important_buffer
    shared_dict['important_ratio'] = important_ratio
    checkpoint["num_played_games"] = num_played_games + 1
    checkpoint["num_played_steps"] += len(game_history.root_values)

    shared_dict['checkpoint'] = checkpoint

def get_batch(shared_dict, config):
    (
        index_batch,
        observation_batch,
        action_batch,
        value_batch,
        policy_batch

    ) = ([], [], [], [], [])
    important_ratio = shared_dict['important_ratio']
    if important_ratio[0] == 0:
        buffer = shared_dict['important_buffer']
    elif important_ratio[1] == 0:
        buffer = shared_dict['buffer']
    else:
        important_ratio[1] = max(important_ratio) if important_ratio[1] else 0
        p = [important_ratio[i] / sum(important_ratio) for i in range(2)]
        if numpy.random.choice(2, p=p) == 1:
            buffer = shared_dict['important_buffer']
        else:
            buffer = shared_dict['buffer']
    for game_id, game_history in sample_n_games(buffer, config.batch_size[0]):
        lens = len(game_history.root_values)
        p = [(config.a * i + config.b) for i in range(len(game_history.root_values))]
        sum_ = sum(p)
        p = [i / sum_ for i in p]
        game_pos = numpy.random.choice(
            lens,
            size=min(config.batch_size[1], lens),
            replace=False,
            p=p
        )
        for each_index in game_pos:
            values, policies, actions = make_target(config, game_history, each_index)
            index_batch.append([game_id, each_index])
            observation_batch.append(
                game_history.get_stacked_observations(
                    each_index
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

def sample_n_games(buffer, n_games: int):
    buffer_list = list(buffer.keys())
    size = len(buffer_list)
    size = min(size, n_games)
    selected_games = numpy.random.choice(buffer_list, size=size, replace=False)
    ret = [
        (game_id, buffer[game_id])
        for game_id in selected_games
    ]
    return ret

def compute_target_value(config, game_history, index):
    bootstrap_index = index + config.td_steps
    if bootstrap_index < len(game_history.root_values):
        root_values = game_history.root_values
        last_step_value = (
            root_values[bootstrap_index]
            if game_history.to_play_history[bootstrap_index] == game_history.to_play_history[index]
            else -root_values[bootstrap_index]
        )
        value = last_step_value * config.discount ** config.td_steps
    else:
        value = 0
    for i, reward in enumerate(
            game_history.reward_history[index + 1: bootstrap_index + 1]
    ):
        value += (
                     reward
                     if game_history.to_play_history[index] == game_history.to_play_history[index + i] else -reward
                 ) * config.discount ** i

    return value

def make_target(config, game_history, state_index):
    value = compute_target_value(config, game_history, state_index)
    target_policies = game_history.child_visits[state_index]
    actions = game_history.legal_history[state_index]
    return value, target_policies, actions
