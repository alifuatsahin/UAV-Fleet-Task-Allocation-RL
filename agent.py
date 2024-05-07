import torch as T
import torch.nn.functional as F

from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
from utils import soft_update, hard_update

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, 
                 input_dims=[8], tau=0.005, 
                 scale=2, env=None, 
                 gamma=0.99, n_actions=2, 
                 max_size=1000000, layer1_size=256, 
                 layer2_size=256, batch_size=256,
                 auto_entropy=True
                 ):
        self.gamma = gamma
        self.tau = tau
        self.scale = scale
        self.beta = beta
        self.auto_entropy = auto_entropy
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        if self.auto_entropy:
            self.target_entropy = -T.prod(T.tensor(n_actions).to(self.device)).item()
            self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=alpha)
        else:
            self.alpha = alpha

        self.actor = ActorNetwork(self.alpha, input_dims, n_actions, layer1_size, layer2_size, name='actor').to(self.device)

        self.critic = CriticNetwork(self.beta, input_dims, n_actions, layer1_size, layer2_size, name='critic_1').to(self.device)

        self.critic_target = CriticNetwork(self.beta, input_dims, n_actions,layer1_size, layer2_size, name='critic_2').to(self.device)

        hard_update(self.critic_target, self.critic)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        actions, _ = self.actor.sample_dirichlet(state)
        return actions.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store(state, action, reward, new_state, done)

    def update(self):
        if self.memory.m_count < self.batch_size:
            return
        
        state, new_state, action, reward, done = self.memory.sample(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)

        with T.no_grad():
            actions, log_probs = self.actor.sample_dirichlet(state)
            q1_new, q2_new = self.critic_target.forward(new_state, actions)
            q_new = T.min(q1_new, q2_new) - self.alpha * log_probs
            q_new = q_new.view(-1)
            next_q = reward + done * self.gamma * q_new

        qf1, qf2 = self.critic.forward(state, action) # Two Q functions to eliminate positive bias
        qf1 = qf1.view(-1)
        qf2 = qf2.view(-1)
        qf1_loss = F.mse_loss(qf1, next_q) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q)
        critic_loss = qf1_loss + qf2_loss

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        actions, log_probs = self.actor.sample_dirichlet(state)

        q1, q2 = self.critic.forward(state, actions)
        q = T.min(q1, q2)

        actor_loss = T.mean(self.alpha * log_probs - q)

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        if self.auto_entropy:
            alpha_loss = -T.mean(self.alpha * (log_probs + self.target_entropy))

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = T.tensor(0.).to(self.device)
            alpha_tlogs = self.alpha

        soft_update(self.critic_target, self.critic, self.tau)

        return actor_loss.item(), critic_loss.item(), alpha_loss.item(), alpha_tlogs.item()

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