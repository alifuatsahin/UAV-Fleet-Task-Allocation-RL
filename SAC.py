import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from network import ActorNetwork, CriticNetwork, ValueNetwork


class SAC:
    def __init__(self, alpha = 0.0003, beta = 0.0003, input_dim = [56], env = None, gamma = 0.99, n_actions = 5,
    max_buffer_size = 30000, tau = 0.005, layer1_size = 256, layer2_size = 256, batch_size = 256, reward_scale = 2):

        self._gamma  = gamma
        self._tau = tau
        self._batch_size = batch_size
        self._reward_scale = reward_scale
        self._env = env

        self.buffer = ReplayBuffer(max_buffer_size, input_dim, n_actions)

        self.actor = ActorNetwork(n_actions = n_actions, name = 'actor', max_action = env.action_space.high)
        self.critic_1 = CriticNetwork(name = 'critic_1', n_actions=n_actions)
        self.critic_2 = CriticNetwork(name = 'critic_2', n_actions=n_actions)
        self.value = ValueNetwork(name = 'value')
        self.target_value = ValueNetwork(name = 'target_value')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)


    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions[0]

    def remember(self, state, action, reward, new_state, done):
        self.buffer.store(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self._tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.value.save_weights(self.value.checkpoint_file)
        self.target_value.save_weights(self.target_value.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.value.load_weights(self.value.checkpoint_file)
        self.target_value.load_weights(self.target_value.checkpoint_file)

    def learn(self):

        if self.buffer._m_count < self._batch_size:
            return

        states, new_states, actions, rewards, dones = self.buffer.sample(self._batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states))
            target_value = tf.squeeze(self.target_value(new_states))
            expected_value = rewards + self._gamma * target_value * (1 - dones)
            value_loss = self.value.loss(expected_value, value)

        value_network_gradient = tape.gradient(value_loss, self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(value_network_gradient, self.value.trainable_variables))

        with tf.GradientTape(persistent=True) as tape:
            new_policy_actions, log_probs = self.actor.sample_normal(states, reparameterize=True)
            log_probs = tf.squeeze(log_probs)

            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)

            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy))
            actor_loss = self.actor.loss(log_probs, critic_value)

            q_hat = self.reward_scale * log_probs - critic_value
            value = tf.squeeze(self.value(states))

            target_value = tf.squeeze(self.target_value(states))
            
            expected_value = rewards + self._gamma * target_value * (1 - dones)

            critic_1_value = tf.squeeze(self.critic_1(states, actions))
            critic_2_value = tf.squeeze(self.critic_2(states, actions))

            critic_1_loss = self.critic_1.loss(expected_value, critic_1_value)
            critic_2_loss = self.critic_2.loss(expected_value, critic_2_value)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_1_network_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
