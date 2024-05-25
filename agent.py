import os

import torch as th
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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
            self.target_entropy = -(np.log(env.action_space.shape[0]) - 1/(2*env.action_space.shape[0])) * th.lgamma(th.Tensor([env.action_space.shape[0]])).item() - 0.25 # -0.98 * np.log(1/env.action_space.shape[0]) -th.prod(th.Tensor(env.action_space.shape[0]).to(self.device)).item()
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
        reward_batch = th.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = th.FloatTensor(next_state_batch).to(self.device)
        mask_batch = th.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with th.no_grad():
            next_action, next_log_pi = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_action)
            min_qf_next_target = th.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
            next_q_value = reward_batch + self.gamma * (1-mask_batch) * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
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
            # self.target_entropy = self.target_entropy_schedule(self.target_entropy, action, log_pi)
            
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()

        else:
            alpha_loss = th.tensor(0.).to(self.device)
            alpha_tlogs = th.tensor(self.alpha).to(self.device)

        if updates % self.update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    def target_entropy_schedule(self, target_entropy, actions, log_pi):
        exp_discount = 0.99
        avg_threshold = 0.05*(np.log(actions.shape[0]) - 1/(2*actions.shape[0]))
        std_threshold = 0.1*(np.log(actions.shape[0]) - 1/(2*actions.shape[0]))
        discount_factor = 1.1
        max_iter = 1000

        actions = actions.clone().detach().to(self.device).numpy()
        log_pi = log_pi.clone().detach().to(self.device).numpy()

        mu_hat = target_entropy
        # policy_entropy = -(actions*log_pi).sum()
        policy_entropy = -log_pi.mean()
        sigma = 0
        for i in range(max_iter):
            delta = policy_entropy - mu_hat
            mu_hat = mu_hat + delta * (1-exp_discount)
            sigma = np.sqrt(exp_discount * (sigma**2 + (1-exp_discount) * delta**2))
            if not (target_entropy - avg_threshold < mu_hat and target_entropy + avg_threshold > mu_hat) or std_threshold > sigma:
                return target_entropy

        return target_entropy*discount_factor
    
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = "logs/checkpoint_{}_{}".format(env_name, suffix)
        save_path = os.path.join(ckpt_path, 'model.pt')
        print('Saving models to {}'.format(ckpt_path))
        th.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, save_path)

    def load_checkpoint(self, ckpt_path):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = th.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

