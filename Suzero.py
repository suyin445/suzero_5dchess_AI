import torch
import os
import json
import platform
import time
from torch.multiprocessing import Manager
from shutil import copyfile

import models
import Game
import self_play
import trainer

class SuzeroConfig:
    def __init__(self):
        # Net
        self.nhead = 32
        self.d_model_times = 32
        self.transformer_layers = 5
        self.max_len_representation = 300
        self.num_blocks = 6
        self.num_channels = 8
        self.net_config = [
            self.nhead,
            self.d_model_times,
            self.transformer_layers,
            self.max_len_representation,
            self.num_blocks,
            self.num_channels
        ]

        # Self_play
        self.num_simulations = 100
        self.max_simulations = 1000
        self.simulations_ratio = 0.5
        self.max_moves = 75
        self.discount = 0.9
        self.num_self_play = 2
        self.num_concurrent = 3
        self.selfplay_on_gpu = True

        # Replay_buffer
        self.replay_buffer_size = 200 * self.num_self_play
        self.batch_size = [1, 16]
        self.td_steps = 50
        self.discount = 0.9
        self.ratio = 2
        self.a = 100
        self.b = 1

        # Trainer
        self.lr_init = 0.002
        self.weight_decay = 1e-4
        self.training_steps = 50000
        self.num_trainer = 1
        self.lr_decay_rate = 0.9
        self.lr_decay_steps = 10000
        self.checkpoint_interval = 2
        self.policy_loss_weight = 1

        # UCB
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Save
        self.num_save = 2


class SuZero:
    def __init__(self, shared_dict):
        torch.multiprocessing.set_start_method('spawn', force=True)
        self.config = SuzeroConfig()
        cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda_available else 'cpu')
        self.game = Game.Game
        print(f'device为{self.device}')
        temp_model = models.SuZeroNetwork(*self.config.net_config).to(self.device)
        if platform.system() == 'Windows':
            self.sv_path = r'./save/'
        elif platform.system() == 'Linux':
            self.sv_path = r'/root/SuZero/save/'
        else:
            raise SystemError('系统不支持。')
        if os.path.isfile(self.sv_path + 'suzero.sv'):
            try:
                checkpoint = torch.load(self.sv_path + 'suzero.sv')
                temp_model.set_weights(checkpoint["weights"])
            except (RuntimeError, EOFError):
                print('模型加载失败，可能是不兼容。')
                while True:
                    inp = input('请输入选择:\n0表示新建模型\n1表示使用备份\n2表示退出\n')
                    try:
                        inp = int(inp)
                    except ValueError:
                        pass
                    else:
                        if inp == 0:
                            checkpoint = {
                                "weights": temp_model.get_weights(),
                                "optimizer_state": None,
                                "total_reward": 0,
                                "muzero_reward": 0,
                                "opponent_reward": 0,
                                "episode_length": 0,
                                "mean_value": 0,
                                "training_step": 0,
                                "lr": 0,
                                "loss": 0,
                                "value_loss": 0,
                                "policy_loss": 0,
                                "num_played_games": 0,
                                "num_played_steps": 0,
                            }
                            break
                        elif inp == 1:
                            if os.path.isfile(self.sv_path + 'suzero_back_up.sv'):
                                try:
                                    checkpoint = torch.load(self.sv_path + 'suzero_back_up.sv')
                                    temp_model.set_weights(checkpoint["weights"])
                                except (RuntimeError, EOFError):
                                    print('备份模型加载失败，可能是不兼容。')
                                else:
                                    print('load完成。')
                                    break
                        elif inp == 2:
                            import sys
                            print('程序即将退出。')
                            sys.exit()

            else:
                print('load完成。')
        else:
            print('sv不存在。')
            os.makedirs(self.sv_path, exist_ok=True)
            open(self.sv_path + 'suzero.sv', 'w')
            checkpoint = {
                "weights": temp_model.get_weights(),
                "optimizer_state": None,
                "total_reward": 0,
                "muzero_reward": 0,
                "opponent_reward": 0,
                "episode_length": 0,
                "mean_value": 0,
                "training_step": 0,
                "lr": 0,
                "loss": 0,
                "value_loss": 0,
                "policy_loss": 0,
                "num_played_games": 0,
                "num_played_steps": 0,
            }
        self.checkpoint = checkpoint
        self.shared_dict = shared_dict
        checkpoint["num_played_games"] = 0
        self.shared_dict['weights'] = checkpoint.pop('weights')
        self.shared_dict['checkpoint'] = checkpoint
        self.shared_dict['buffer'] = {}
        self.shared_dict['important_buffer'] = {}
        self.shared_dict['important_ratio'] = [0, 0]
        self.self_play = [
            self_play.Selfplay(self.game, self.config, self.shared_dict, id_)
            for id_ in range(self.config.num_self_play)
        ]
        self.trainer = [
            trainer.Trainer(self.checkpoint, self.config, self.shared_dict)
            for _ in range(self.config.num_trainer)
        ]

    def save_model(self):
        checkpoint = self.shared_dict['checkpoint']
        checkpoint['weights'] = self.shared_dict['weights']
        copyfile(self.sv_path + 'suzero.sv', self.sv_path + 'suzero_back_up.sv')
        torch.save(checkpoint, self.sv_path + 'suzero.sv')
        print('save完成。')

    def train(self):
        for i in range(self.config.num_self_play):
            self.self_play[i].start()
        for i in range(self.config.num_trainer):
            self.trainer[i].start()
        now_training_step = self.shared_dict['checkpoint']["training_step"]
        while True:
            new_training_step = self.shared_dict['checkpoint']["training_step"]
            if new_training_step > now_training_step + 1:
                now_training_step = new_training_step
                value_loss = self.shared_dict['checkpoint']["value_loss"]
                policy_loss = self.shared_dict['checkpoint']["policy_loss"]
                print(f"value_loss为：{value_loss}, policy_loss为{policy_loss}")
                with open(self.sv_path + 'suzero_loss.json', 'a+') as f_obj:
                    f_obj.seek(0)  # 移动到文件开头
                    try:
                        loss_dic = json.load(f_obj)  # 读取已有的数据
                    except json.decoder.JSONDecodeError:  # 如果文件为空或者格式错误
                        loss_dic = {'value_loss': [], 'policy_loss': []}
                    loss_dic.update({"value_loss": value_loss, "policy_loss": policy_loss})
                    f_obj.seek(0)  # 移动到文件开头
                    json.dump(loss_dic, f_obj)  # 写入修改后的数据
                    print('loss写入完成。')
                self.save_model()
            else:
                time.sleep(1)


if __name__ == '__main__':
    main_ = SuZero(Manager().dict())
    main_.train()
