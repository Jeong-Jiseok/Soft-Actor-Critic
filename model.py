import torch
import torch.nn as nn

# Xavier initialization for linear layers
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


# Gaussian policy network for SAC
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, args, action_space=None):
        super().__init__()
        self.args = args

        layers = [nn.Linear(state_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.num_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        self.feature_layers = nn.Sequential(*layers)

        self.mean_layer = nn.Linear(args.hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(args.hidden_dim, action_dim)
        self.apply(weights_init_)

        # Action rescaling (tanh output → env action space)
        if action_space is not None:
            action_high = torch.FloatTensor(action_space.high)
            action_low = torch.FloatTensor(action_space.low)
            self.register_buffer("action_scale", (action_high - action_low) / 2.0)
            self.register_buffer("action_bias", (action_high + action_low) / 2.0)
        else:
            self.register_buffer("action_scale", torch.tensor(1.0))
            self.register_buffer("action_bias", torch.tensor(0.0))

    # Forward pass to compute mean and std
    def forward(self, state):
        x = self.feature_layers(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("x: ", x.size(), x)
        if torch.isnan(mean).any():
            print("forward x: ", x.size(), x)
            print("forward mean: ", mean.size(), mean)
        if torch.isnan(log_std).any():
            print("forward x: ", x.size(), x)
            print("forward log_std: ", log_std.size(), log_std)

        log_std = torch.clamp(log_std, self.args.LOG_STD_MIN, self.args.LOG_STD_MAX)
        std = torch.exp(log_std)
        
        return mean, std

    # Reparameterization trick to sample actions + log_prob
    def sample(self, state):
        mean, std = self.forward(state)

        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("state: ", state.size(), state)
            print("mean: ", mean.size(), mean)
            print("std: ", std.size(), std)

        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample() # Reparameterization trick: mean + std * N(0,1)
        y = torch.tanh(x)

        action = y * self.action_scale + self.action_bias
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, mean
    

# Twin Q-networks
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super().__init__()
        self.args = args

        input_dim = state_dim + action_dim
        layers = [nn.Linear(input_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.num_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [(nn.Linear(args.hidden_dim, 1))]
        self.q1 = nn.Sequential(*layers)

        layers = [nn.Linear(input_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.num_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [(nn.Linear(args.hidden_dim, 1))]
        self.q2 = nn.Sequential(*layers)

        self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)