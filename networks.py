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

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name = 'actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
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

        self.model.apply(init_weights)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, state):
        alpha = self.model(state)

        return alpha
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def sample_dirichlet(self, state):
        alpha = self.forward(state) + 1 
        alpha = T.clamp(alpha, self.noise, 1-self.noise)
        dist = dirichlet.Dirichlet(alpha)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        log_probs = T.sum(log_probs, dim=0)

        return actions, log_probs