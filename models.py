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

class RepresentationNetwork(torch.nn.Module):
    def __init__(
            self,
            nhead,
            d_model_times,
            num_layer,
            max_len
    ):
        super().__init__()
        d_model = d_model_times * nhead
        self.transformer = transformer(d_model, nhead, 512, num_layer)
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x, key_padding_mask=None):
        x = x + self.pos_embedding.data[:, :x.shape[1], :]
        if key_padding_mask is None:
            x = self.transformer(src=x)
        else:
            x = self.transformer(src=x, src_key_padding_mask=key_padding_mask)
        x = torch.sum(x, dim=1)
        return x


class PredictionNetwork(torch.nn.Module):
    def __init__(
            self,
            nhead,
            d_model_times,
            num_layer,
            max_len
    ):
        super(PredictionNetwork, self).__init__()
        d_model = d_model_times * nhead
        self.transformer = transformer(d_model, nhead, 512, num_layer)
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, max_len, d_model))
        self.value_mlp = mlp(128, [16], 1)

    def forward(self, encoded_state, action: torch.Tensor, key_padding_mask=None):
        action = action.repeat(1, 1, 16)
        encoded_state = torch.unsqueeze(encoded_state, dim=1)
        x = action + encoded_state
        x = x + self.pos_embedding.data[:, :x.shape[1], :]
        if key_padding_mask is None:
            x = self.transformer(src=x)
        else:
            x = self.transformer(src=x, src_key_padding_mask=key_padding_mask)
        value = x[:, 0, :]
        value = self.value_mlp(value)
        policy_logits = torch.sum(x, dim=2)
        return value, policy_logits


class SuZeroNetwork(torch.nn.Module):
    def __init__(
        self,
        nhead,
        d_model_times,
        representation_transformer_layers,
        prediction_transformer_layers,
        max_len_representation,
        max_len_prediction
    ):
        super().__init__()
        self.representation_network = torch.nn.DataParallel(
            RepresentationNetwork(nhead, d_model_times, representation_transformer_layers, max_len_representation)
        )
        self.prediction_network = torch.nn.DataParallel(
            PredictionNetwork(nhead, d_model_times, prediction_transformer_layers, max_len_prediction)
        )

    def representation(self, observation, key_padding_mask = None):
        encoded_state = self.representation_network(observation, key_padding_mask = key_padding_mask)
        return encoded_state

    def prediction(self, encoded_state, action: torch.Tensor, key_padding_mask=None):
        return self.prediction_network(encoded_state, action, key_padding_mask)

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
