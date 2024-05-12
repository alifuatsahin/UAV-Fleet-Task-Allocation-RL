import torch as th
import torch.nn.functional as F
import torch.optim as optim
from networks import CriticNetwork, GaussianPolicy, DirichletPolicy
from utils import soft_update, hard_update

class Agent:
    def __init__(self, env,
                hidden_dim,
                batch_size,
                alpha, gamma, tau, lr,
                update_interval,
                auto_entropy = True, 
                policy = "Gaussian"): # "Gaussian" or "Dirichlet"
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy = auto_entropy
        self.update_interval = update_interval

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.policy_type = policy

        self.critic = CriticNetwork(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target = CriticNetwork(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            self.policy = GaussianPolicy(env.observation_space.shape[0], hidden_dim, env.action_space).to(self.device)
        else:
            self.policy = DirichletPolicy(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim).to(self.device)

        if auto_entropy:
            self.target_entropy = -th.prod(th.Tensor(env.action_space.shape[0]).to(self.device)).item() #-0.1 * np.log(1/env.action_space.shape[0]) 
            self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr, eps=1e-4)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-4)

    def get_action(self, state):
        state = th.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.policy.sample(state)

        return action.detach().cpu().numpy()[0]
    
    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)

        state_batch = th.FloatTensor(state_batch).to(self.device)
        action_batch = th.FloatTensor(action_batch).to(self.device)
        reward_batch = th.FloatTensor(reward_batch).to(self.device)
        next_state_batch = th.FloatTensor(next_state_batch).to(self.device)
        mask_batch = th.FloatTensor(mask_batch).to(self.device)

        with th.no_grad():
            next_action, next_log_pi = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_action)
            min_qf_next_target = th.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
            min_qf_next_target = min_qf_next_target.view(-1)
            next_q_value = reward_batch + self.gamma * (1-mask_batch) * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1 = qf1.view(-1)
        qf2 = qf2.view(-1)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        action, log_pi = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, action)
        min_qf_pi = th.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()

        else:
            alpha_loss = th.tensor(0.).to(self.device)
            alpha_tlogs = self.alpha

        if updates % self.update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
