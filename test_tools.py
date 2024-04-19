import ray
import torch
import os
import time

import models
import Game
import self_play
import replay_buffer
import shared_storage
import trainer
import analyzer
import json


class SuzeroConfig:
    def __init__(self):
        # Net
        self.nhead = 32
        self.d_model_times = 4
        self.representation_transformer_layers = 20
        self.prediction_transformer_layers = 10
        self.max_len_representation = 300
        self.max_len_prediction = 50000
        self.net_config = [
            self.nhead,
            self.d_model_times,
            self.representation_transformer_layers,
            self.prediction_transformer_layers,
            self.max_len_representation,
            self.max_len_prediction
        ]

        # Self_play
        self.num_simulations = 10
        self.max_moves = 50
        self.discount = 0.5
        self.num_self_play = 1
        self.selfplay_on_gpu = True

        # Replay_buffer
        self.replay_buffer_size = 150
        self.batch_size = 2
        self.td_steps = 6
        self.discount = 0.5

        # Trainer
        self.lr_init = 0.002
        self.weight_decay = 1e-4
        self.training_steps = 50000
        self.checkpoint_interval = 10
        self.ratio = 2
        self.lr_decay_rate = 0.9
        self.lr_decay_steps = 10000

        # UCB
        self.pb_c_base = 19652
        self.pb_c_init = 1.25


class TestTool:
    def __init__(self):
        self.config = SuzeroConfig()
        cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda_available else 'cpu')
        self.model = models.SuZeroNetwork(*self.config.net_config).to(self.device)
        self.game = Game.Game
        if cuda_available:
            ray.init(num_gpus=2)
        else:
            ray.init()
        print(f'device为{self.device}')
        self.checkpoint = {
            "weights": self.model.get_weights(),
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
        self.replay_buffer = replay_buffer.ReplayBuffer.remote(self.checkpoint, {}, self.config)
        self.shared_storage = shared_storage.SharedStorage.remote(self.checkpoint, self.config)
        self.self_play = [
            self_play.Selfplay.remote(self.checkpoint, self.game, self.config)
            for _ in range(self.config.num_self_play)
        ]
        self.trainer = trainer.Trainer.remote(self.checkpoint, self.config, self.device)

    def save_model(self):
        torch.save(self.checkpoint, r'./save/suzero.sv')
        print('save完成。')

    def load_model(self):
        if os.path.isfile(r'./save/suzero.sv'):
            self.model.set_weights(torch.load(r'./save/suzero.sv')["weights"])
            print('load完成。')
        else:
            print('sv不存在。')
            os.makedirs('./save', exist_ok=True)
            open(r'./save/suzero.sv', 'w')

    def test_analyzer(self):
        analyzer_ = analyzer.Analyzer()
        analyzer_.start_analyze()

    def test_share(self):
        sharedstorage = shared_storage.SharedStorage.remote(self.checkpoint, self.config)
        aaa = sharedstorage.get_info.remote('weights')
        print(ray.get(aaa))

    def test_MCTS(self):
        pass

    def total_test(self):
        self.load_model()
        start_train_step = ray.get(self.shared_storage.get_info.remote("training_step"))
        for i in range(self.config.num_self_play):
            self.self_play[i].continuous_self_play.remote(self.shared_storage, self.replay_buffer)
        self.trainer.continuous_update_weights.remote(self.replay_buffer, self.shared_storage)
        while True:
            if ray.get(self.shared_storage.get_info.remote("training_step")) > start_train_step + 1:
                start_train_step = ray.get(self.shared_storage.get_info.remote("training_step"))
                loss = ray.get(self.shared_storage.get_info.remote("loss"))
                print(f'最新loss为：{loss}')
                with open(r'./save/suzero_loss.json', 'a+') as f_obj:
                    f_obj.seek(0)  # 移动到文件开头
                    try:
                        loss_list = json.load(f_obj)  # 读取已有的数据
                    except json.decoder.JSONDecodeError:  # 如果文件为空或者格式错误
                        loss_list = []  # 创建一个空列表
                    loss_list.append(loss)  # 把最新的损失值添加到列表中
                    f_obj.seek(0)  # 移动到文件开头
                    json.dump(loss_list, f_obj)  # 写入修改后的数据
                self.save_model()
            time.sleep(10)


if __name__ == '__main__':
    testtool = TestTool()
    while True:
        choose = input('0: 保存模型\n'
                       '1: 读取模型\n'
                       '2: 测试shared_storage\n'
                       '3: 测试分析\n'
                       '4: 测试selfplay\n'
                       '5: 全流程测试\n'
                       '9: 退出\n')
        if choose == '9':
            break
        elif choose == '0':
            testtool.save_model()
        elif choose == '1':
            testtool.load_model()
        elif choose == '2':
            testtool.test_share()
        elif choose == '3':
            testtool.test_analyzer()
        elif choose == '4':
            testtool.test_MCTS()
        elif choose == '5':
            testtool.total_test()
            break

ray.shutdown()
print('成功退出。')
