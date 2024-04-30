import torch as T
import torch.nn.functional as F

from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], tau=0.005, scale=2, env=None, gamma=0.99, n_actions=2, max_size=1000000, layer1_size=256, layer2_size=256, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.scale = scale
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(alpha, input_dims, n_actions, layer1_size, layer2_size, name='actor')
        self.critic1 = CriticNetwork(beta, input_dims, n_actions, layer1_size, layer2_size, name='critic_1')
        self.critic2 = CriticNetwork(beta, input_dims, n_actions,layer1_size, layer2_size, name='critic_2')
        self.value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, name='value')
        self.target_value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, name='target_value')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions, _ = self.actor.sample_dirichlet(state)
        action = actions.cpu().detach().numpy()[0]
        return action
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_state_dict = dict(target_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()

    def learn(self):
        if self.memory.m_count < self.batch_size:
            return
        
        state, new_state, action, reward, done = self.memory.sample(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.value.device)
        done = T.tensor(done).to(self.value.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.value.device)
        state = T.tensor(state, dtype=T.float).to(self.value.device)
        action = T.tensor(action, dtype=T.float).to(self.value.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_dirichlet(state)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state, actions)
        q2_new_policy = self.critic2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_dirichlet(state)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state, actions)
        q2_new_policy = self.critic2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma*value_
        q1_old_policy = self.critic1.forward(state, action)
        q2_old_policy = self.critic2.forward(state, action)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_network_parameters()