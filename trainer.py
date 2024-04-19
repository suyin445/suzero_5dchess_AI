import copy
import time
import torch
from torch.multiprocessing import Process
from torch.nn.utils.rnn import pad_sequence

import models
import replay_buffer


class Trainer(Process):
    def __init__(self, initial_checkpoint, config, shared_dict):
        super().__init__(daemon=True)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.SuZeroNetwork(*config.net_config)
        self.model.set_weights(copy.deepcopy(shared_dict["weights"]))
        self.model.to(self.device)
        self.model.train()
        self.training_step = initial_checkpoint["training_step"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr_init,
            weight_decay=self.config.weight_decay,
        )
        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )
        self.shared_dict = shared_dict

    def run(self):
        print('加载了一个trainer')
        self.continuous_update_weights(self.shared_dict)

    def update_weights(self, batch):
        (
            observation_batch,
            action_batch,
            target_value,
            target_policy,
        ) = batch

        device = next(self.model.parameters()).device
        observation_batch = [torch.tensor(each) for each in observation_batch]
        ob_ = pad_sequence(observation_batch, batch_first=True)
        key_padding_mask = torch.tensor(
            [[False if i < len(each) else True for i in range(ob_.size(1))] for each in observation_batch]
        ).to(device)
        observation_batch = ob_.float().to(device)
        action_batch = [torch.tensor(each) for each in action_batch]
        action_batch = pad_sequence(action_batch, batch_first=True).float().to(device)
        target_value = torch.tensor(target_value).float().to(device)
        target_policy = [torch.tensor(each).float().to(device) for each in target_policy]
        policy_mask = [torch.zeros_like(each, dtype=torch.float, device=device) for each in target_policy]
        target_policy = pad_sequence(target_policy, batch_first=True)
        policy_mask = pad_sequence(policy_mask, batch_first=True, padding_value=1)
        hidden_state = self.model.representation(observation_batch, key_padding_mask)
        value, policy_logits = self.model.prediction(hidden_state, action_batch)
        # policy_logits.data.masked_fill_(policy_mask.bool(), -float('inf'))
        value_loss, policy_loss = self.loss_function(
            value,
            policy_logits,
            target_value,
            target_policy
        )

        loss = value_loss + policy_loss * self.config.policy_loss_weight
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        self.training_step += 1

        return (
            loss.item(),
            value_loss.mean().item(),
            policy_loss.mean().item()
        )

    def continuous_update_weights(self, shared_dict):
        # Wait for the replay buffer to be filled
        while shared_dict['checkpoint']["num_played_games"] < 1:
            time.sleep(1)
        next_batch = replay_buffer.get_batch(shared_dict, self.config)
        # Training loop
        while self.training_step < self.config.training_steps:
            index_batch, batch = next_batch
            next_batch = replay_buffer.get_batch(shared_dict, self.config)
            self.update_lr()
            loss, value_loss, policy_loss = self.update_weights(batch)
            print('进行了一次训练')
            checkpoint = shared_dict['checkpoint']
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_dict["weights"] = copy.deepcopy(self.model.get_weights())
                checkpoint["optimizer_state"] = copy.deepcopy(models.dict_to_cpu(self.optimizer.state_dict()))
            checkpoint["training_step"] = self.training_step
            checkpoint["lr"] = self.optimizer.param_groups[0]["lr"]
            checkpoint["loss"] = loss
            checkpoint["value_loss"] = value_loss
            checkpoint["policy_loss"] = policy_loss
            shared_dict['checkpoint'] = checkpoint

            if self.config.ratio:
                while (
                        self.training_step
                        / max(1, shared_dict['checkpoint']["num_played_steps"])
                ) > self.config.ratio and self.training_step < self.config.training_steps:
                    time.sleep(0.5)

    def update_lr(self):
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
                self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(
            value,
            policy_logits,
            target_value,
            target_policy
    ):
        target_value = torch.unsqueeze(target_value, dim=1)
        target_value.clamp_(min=-0.999, max=0.999)
        value.clamp_(min=-0.999, max=0.999)
        target_value = (target_value + 1) / 2
        value = (value + 1) / 2
        BCEloss = torch.nn.BCELoss(reduction='none')
        value_loss = BCEloss(value, target_value).sum(1)
        value_loss = value_loss - BCEloss(target_value, target_value).sum(1)
        policy_loss = (- target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(1)
        policy_loss = policy_loss + (target_policy * torch.nn.LogSoftmax(dim=1)(target_policy)).sum(1)
        return value_loss, policy_loss
