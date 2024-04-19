import copy
import time

import ray
import torch

import models
from torch.nn.utils.rnn import pad_sequence

@ray.remote(num_gpus=0.5)
class Trainer:
    def __init__(self, initial_checkpoint, config, device):
        self.config = config
        self.model = models.SuZeroNetwork(*config.net_config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(device)
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
        ac = pad_sequence(action_batch, batch_first=True)
        action_key_padding_mask = torch.tensor(
            [[False if i < len(each) else True for i in range(ac.size(1))] for each in action_batch]
        ).to(device)
        action_batch = ac.float().to(device)
        target_value = torch.tensor(target_value).float().to(device)
        target_policy = [torch.tensor(each).float().to(device) for each in target_policy]
        target_policy = pad_sequence(target_policy, batch_first=True)
        hidden_state = self.model.representation(observation_batch, key_padding_mask)
        value, policy_logits = self.model.prediction(hidden_state, action_batch, action_key_padding_mask)
        value_loss, policy_loss = self.loss_function(
            value,
            policy_logits,
            target_value,
            target_policy
        )

        loss = value_loss + policy_loss
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            loss.item(),
            value_loss.mean().item(),
            policy_loss.mean().item()
        )

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)
        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps:
            index_batch, batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()
            loss, value_loss, policy_loss = self.update_weights(batch)
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(self.optimizer.state_dict())
                    }
                )
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "loss": loss,
                    "value_loss": value_loss,
                    "policy_loss": policy_loss
                }
            )
            if self.config.ratio:
                while (
                        self.training_step
                        / max(1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                              )
                        > self.config.ratio
                        and self.training_step < self.config.training_steps
                ):
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
        value_loss = (torch.square(target_value - value)).sum(1)
        policy_loss = (torch.square(target_policy - policy_logits)).sum(1)
        return value_loss, policy_loss
