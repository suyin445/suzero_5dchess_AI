import torch

def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict

class SwitchNorm2d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = torch.nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = torch.nn.Parameter(torch.ones(3))
            self.var_weight = torch.nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = torch.nn.Parameter(torch.ones(2))
            self.var_weight = torch.nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    @staticmethod
    def _check_input_dim(_input):
        if _input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(_input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = torch.nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.sn1 = SwitchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.sn2 = SwitchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.sn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.sn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


class RepresentationNetwork(torch.nn.Module):
    def __init__(
            self,
            nhead,
            d_model_times,
            num_layer,
            max_len,
            num_channels,
            num_blocks
    ):
        super().__init__()
        d_model = d_model_times * nhead
        self.transformer = transformer(d_model, nhead, 512, num_layer)
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, max_len, d_model))
        self.conv = conv3x3(1, num_channels)
        self.sn = SwitchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x, key_padding_mask=None):
        x = x.repeat(1, 1, 8)
        x = x + self.pos_embedding.data[:, :x.shape[1], :]
        if key_padding_mask is None:
            x = self.transformer(src=x)
        else:
            x = self.transformer(src=x, src_key_padding_mask=key_padding_mask)
        x = torch.sum(x, dim=1)
        x = x.view(-1, 1, 32, 32)
        x = self.conv(x)
        x = self.sn(x)
        for block in self.resblocks:
            x = block(x)
        return x


class PredictionNetwork(torch.nn.Module):
    def __init__(self,
                 num_channels,
                 d_model):
        super(PredictionNetwork, self).__init__()
        self.conv_value = conv3x3(num_channels, 1)
        self.conv_policy = conv3x3(num_channels, 1)
        self.d_model = d_model
        self.policy_mlp = mlp(self.d_model + 8, [64], 1)
        self.value_mlp = mlp(self.d_model, [64], 1)

    def forward(self, encoded_state, action: torch.Tensor):
        value = self.conv_value(encoded_state)
        value = value.view(-1, self.d_model)
        value = self.value_mlp(value)
        policy_logits = self.conv_policy(encoded_state)
        policy_logits = policy_logits.view(-1, 1, self.d_model)
        policy_logits = policy_logits.repeat(1, action.shape[1], 1)
        policy_logits = torch.cat([policy_logits, action], dim=2)
        policy_logits = self.policy_mlp(policy_logits)
        policy_logits = torch.squeeze(policy_logits, dim=2)
        value = torch.sigmoid(value)
        value = 3 * (value - 0.5)
        return value, policy_logits


class SuZeroNetwork(torch.nn.Module):
    def __init__(
        self,
        nhead,
        d_model_times,
        transformer_layers,
        max_len_representation,
        num_blocks,
        num_channels
    ):
        super().__init__()
        self.representation_network = torch.nn.DataParallel(
            RepresentationNetwork(nhead,
                                  d_model_times,
                                  transformer_layers,
                                  max_len_representation,
                                  num_channels,
                                  num_blocks)
        )
        self.prediction_network = torch.nn.DataParallel(
            PredictionNetwork(num_channels,
                              d_model_times * nhead)
        )

    def representation(self, observation, key_padding_mask = None):
        encoded_state = self.representation_network(observation, key_padding_mask = key_padding_mask)
        return encoded_state

    def prediction(self, encoded_state, action: torch.Tensor):
        return self.prediction_network(encoded_state, action)

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


def transformer(d_model, nhead, dim_feedforward, num_layers):
    layer_norm = torch.nn.LayerNorm(d_model)
    layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                             batch_first=True)
    return torch.nn.TransformerEncoder(layer, num_layers, layer_norm)


def mlp(
        input_size,
        layer_sizes,
        output_size,
        output_activation=torch.nn.Identity,
        activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)

def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )
