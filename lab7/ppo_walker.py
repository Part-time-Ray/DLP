#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh
import os
os.environ["MUJOCO_GL"] = "egl"

import random
from collections import deque
from typing import Deque, List, Tuple

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 0,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu_head = nn.Linear(128, out_dim)
        self.log_sigma_head = nn.Linear(128, out_dim)
        
        self.mu_head = init_layer_uniform(self.mu_head)
        self.log_sigma_head = init_layer_uniform(self.log_sigma_head)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc2(F.relu(self.fc1(state))))
        mu = torch.tanh(self.mu_head(x))
        log_sigma = (torch.tanh(self.log_sigma_head(x)) + 1) / 2
        log_sigma = self.log_std_min + log_sigma * (self.log_std_max - self.log_std_min)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value_head = nn.Linear(256, 1)
        
        self.value_head = init_layer_uniform(self.value_head)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc2(F.relu(self.fc1(state))))
        value = self.value_head(x)
        #############################

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    # gae_returns = list()
    # values.append(next_value)
    # sum = 0
    # for i in range(len(rewards)-1, -1, -1):
    #     td_error = rewards[i] + gamma * values[i+1] * masks[i] - values[i]
    #     sum = sum * (gamma * tau) * masks[i] + td_error
    #     gae_returns.append(sum + values[i])
    # values.pop()
    # gae_returns.reverse()
    values = values + [next_value]
    gae = 0
    returns: Deque[float] = deque()

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.appendleft(gae + values[step])

    return list(returns)
    #############################
    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, random_seed, args):
        """Initialize."""
        self.env = env
        self.args = args
        self.random_seed = random_seed
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.update_epoch = args.update_epoch
        
        # device: cpu / gpu
        self.device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.005)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1
        self.test_episode = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))

        return selected_action.cpu().detach().numpy().reshape(self.action_dim)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            ############TODO#############
            # actor_loss = ?
            surr1 = adv * ratio
            surr2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()
            actor_loss = policy_loss - self.entropy_weight * entropy
            #############################

            # critic_loss
            ############TODO#############
            # critic_loss = ?
            value = self.critic(state)
            critic_loss = F.mse_loss(value, return_)
            #############################
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        # state, _ = self.env.reset(seed=random.sample(self.random_seed, 1)[0])
        state, _ = self.env.reset(seed=self.random_seed[0])
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        eval_scores = []
        best_eval_score = -np.inf
        score = 0
        episode_count = 0
        ep_step = 0
        os.makedirs("v3", exist_ok=True)
        #  1M, 1.5M, 2M, 2.5M, and 3M
        snapshots_step = [(3000000, '3m'), (2500000, '2.5m'), (2000000, '2m'), (1500000, '1.5m'), (1000000, '1m'), (500000, '0.5m'), (100000, '0.1m')]
        tq = tqdm(total=3000000)
        for ep in range(1, self.num_episodes+1):
            for _ in range(self.rollout_len):
                self.total_step += 1
                ep_step += 1
                tq.update(1)
                action = self.select_action(state)
                action = action.reshape(self.action_dim,)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward[0][0]
                wandb.log({"step": self.total_step})
                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    # state, _ = self.env.reset(seed=random.sample(self.random_seed, 1)[0])
                    state, _ = self.env.reset(seed=self.random_seed[episode_count % len(self.random_seed)])
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    wandb.log({"episode": episode_count, "episode_length": ep_step, "return": score})
                    ep_step = 0
                    score = 0
                    # print(f"Episode {episode_count}: Total Reward = {score}")


            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            eval_score = self.test()
            eval_scores.append(eval_score)
            avg_eval_score = np.mean(eval_scores[-20:])
            tq.set_description(f"Score = {eval_score:.2f}, Avg Score = {avg_eval_score:.2f}")
            if avg_eval_score > best_eval_score:
                best_eval_score = avg_eval_score
                for snapshots in snapshots_step:
                    # save the model
                    self.save_model(os.path.join("v3", f"LAB7_StudentID_task3_ppo_{snapshots[1]}.pt"))
            wandb.log({"avg_score": avg_eval_score})  
            wandb.log({"best_avg_score": best_eval_score})
            if self.total_step > snapshots_step[-1][0]:
                snapshots_step.pop()
                if len(snapshots_step) == 0:
                    break
        torch.save({'actor': self.actor.state_dict(),'critic': self.critic.state_dict()}, "v3_final.pt")

        # termination
        self.env.close()
    def load_model(self):
        """Load the model."""
        state_dict = torch.load(self.args.model_path)
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.actor.eval()
        self.critic.eval()
    def gen_video(self, video_folder: str):
        os.makedirs(video_folder, exist_ok=True)
        [os.remove(os.path.join(video_folder, f)) for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]
        try:
            self.load_model()
            print(f"Model loaded from {self.args.model_path}")
        except Exception as e:
            print(f"Failed to load model from {self.args.model_path}: {e}")
            return None

        self.is_test = True
        scores = []
        for seed in self.random_seed:
            self.env = gym.make("Walker2d-v4", render_mode="rgb_array")
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder, name_prefix=f"seed-{seed}")
            state, _ = self.env.reset(seed=seed)
            done = False
            score = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += float(reward[0][0])
            scores.append(score)
            print("seed: ", seed, " score: ", score)
            self.env.close()
        
        avg_score = np.mean(scores)
        print("Average score: ", avg_score)
        return avg_score
        


    def test(self, video_folder = None):
        """Test the agent."""
        self.is_test = True

        tmp_env = self.env
        if video_folder:
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        # state, _ = self.env.reset(seed=random.sample(self.random_seed, 1)[0])
        state, _ = self.env.reset(seed=self.random_seed[self.test_episode % len(self.random_seed)])
        self.test_episode += 1
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward[0][0]

        self.env.close()

        self.env = tmp_env
        self.is_test = False
        return score
    def save_model(self, path: str):
        """Save the model."""
        state_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        torch.save(state_dict, path)
 
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-run")
    parser.add_argument("--gpu", "-g", type=str, default="0")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.995)
    parser.add_argument("--num-episodes", type=float, default=2000)
    # parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=int, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=8192)  
    parser.add_argument("--update-epoch", "-u", type=int, default=20)
    parser.add_argument("--model-path", "-p", type=str, default="")
    args = parser.parse_args()
 
    # environment
    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    random_seed = random.sample(range(0, 10000), 20)
    print("Random Seed: ", random_seed)
    
    agent = PPOAgent(env, random_seed, args)
    if len(args.model_path):
        agent.gen_video("video_3")
    else:
        wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True)
        agent.train()