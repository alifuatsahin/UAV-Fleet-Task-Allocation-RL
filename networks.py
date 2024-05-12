import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.distributions.dirichlet as dirichlet

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=1)
        m.bias.data.fill_(0)

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, lr, fc1_dims=256, fc2_dims=256, name = 'critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = (input_dims[0] + n_actions, )
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')

        self.modelQ1 = nn.Sequential(
            nn.Linear(*self.input_dims, self.fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(self.fc2_dims, 1)
        )

        self.modelQ1.apply(init_weights)

        self.modelQ2 = nn.Sequential(
            nn.Linear(*self.input_dims, self.fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(self.fc2_dims, 1)
        )

        self.modelQ2.apply(init_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        q1 = self.modelQ1(T.cat([state, action], dim=1))
        q2 = self.modelQ2(T.cat([state, action], dim=1))

        return q1, q2
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class DirichletPolicy(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name = 'actor', chkpt_dir='tmp/sac'):
        super(DirichletPolicy, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')
        self.noise = 1e-16

        self.model = nn.Sequential(
            nn.Linear(*self.input_dims, self.fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.Tanh(),
            nn.Linear(self.fc2_dims, self.n_actions),
            nn.Softplus()
        )

        self.apply(init_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        alpha = self.model(state)

        return alpha
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def sample(self, state):
        alpha = self.forward(state) + 1 
        alpha = T.clamp(alpha, self.noise, 1-self.noise)
        dist = dirichlet.Dirichlet(alpha)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)
        log_probs = T.sum(log_probs, dim=0)

        return actions, log_probs
    
class GaussianPolicy(nn.Module):
    def __init__(self, lr, input_dims, n_actions, action_space, fc1_dims=256, fc2_dims=256, name = 'actor', chkpt_dir='tmp/sac'):
        super(GaussianPolicy, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')
        self.noise = 1e-16

        self.model = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
        )

        self.mean_layer = nn.Linear(fc2_dims, n_actions)
        self.log_std_layer = nn.Linear(fc2_dims, n_actions)

        self.apply(init_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # action rescaling
        if action_space is None:
            self.action_scale = T.tensor(1.)
            self.action_bias = T.tensor(0.)
        else:
            self.action_scale = T.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = T.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = self.model(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = T.clamp(log_std, -20, 2)

        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = T.exp(log_std)
        normal = T.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = T.tanh(x_t)
        actions = y_t * self.action_scale + self.action_bias
        log_probs = normal.log_prob(x_t)
        log_probs -= T.log(self.action_scale * (1 - y_t.pow(2)) + self.noise)
        log_probs = T.sum(log_probs)
        mean = T.tanh(mean) * self.action_scale + self.action_bias

        return actions, log_probs, mean