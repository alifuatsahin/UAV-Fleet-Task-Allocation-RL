import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CriticNetwork, self).__init__()

        # Q1 architecture
        self.Q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 architecture
        self.Q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weights_init_)

    def forward(self, state, action):
        sa = th.cat([state, action], 1)

        q1 = self.Q1(sa)
        q2 = self.Q2(sa)

        return q1, q2
    
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        # policy architecture
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_space.shape[0])
        self.log_std = nn.Linear(hidden_dim, action_space.shape[0])

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = th.tensor(1.)
            self.action_bias = th.tensor(0.)
        else:
            self.action_scale = th.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = th.FloatTensor(
                (action_space.high + action_space.low) / 2.)
            
    def forward(self, state):
        x = self.layers(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        y = th.tanh(z)
        action = y * self.action_scale + self.action_bias

        log_pi = normal.log_prob(z) - th.log(self.action_scale * (1 - y.pow(2)) + EPS)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
    
class DirichletPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DirichletPolicy, self).__init__()

        # policy architecture
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()
        )

        self.apply(weights_init_)

    def forward(self, state):
        alpha = self.policy(state) + 1 + EPS
        # alpha = th.clamp(alpha, min=EPS, max=1 - EPS)
        return alpha

    def sample(self, state):
        alpha = self.forward(state)
        dist = Dirichlet(alpha)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions).sum()

        return actions, log_probs