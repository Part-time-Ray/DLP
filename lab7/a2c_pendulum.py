#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
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
from typing import Tuple
import os

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu_head = nn.Linear(128, out_dim)
        self.log_sigma_head = nn.Linear(128, out_dim)
        
        # initialize_uniformly(self.fc1)
        # initialize_uniformly(self.fc2)
        initialize_uniformly(self.mu_head)
        initialize_uniformly(self.log_sigma_head)
        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        x = F.relu(self.fc2(F.relu(self.fc1(state))))

        mu = torch.tanh(self.mu_head(x)) * 2
        log_sigma = F.softplus(self.log_sigma_head(x))
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        action = dist.sample().clamp(-2.0, 2.0)
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_head = nn.Linear(128, 1)
        # initialize_uniformly(self.fc1)
        # initialize_uniformly(self.fc2)
        initialize_uniformly(self.value_head)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        #############################

        return value
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, random_seed, args=None):
        """Initialize."""
        self.env = env
        self.args = args
        self.random_seed = random_seed
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        
        # device: cpu / gpu
        self.device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob, dist]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, done])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        state, log_prob, dist, next_state, reward, done = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise

        # state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        mask = torch.tensor(1 - done, dtype=torch.float32, device=self.device)
        value = self.critic(state)
        next_value = self.critic(next_state)
        td_target = reward + self.gamma * next_value * mask

        ############TODO#############
        # value_loss = ?
        value_loss = F.smooth_l1_loss(value, td_target.detach())
        #############################

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        ############TODO#############
        # policy_loss = ?
        advantage = (td_target - value).detach()
        policy_loss = -(log_prob * advantage) - self.entropy_weight * dist.entropy()
        #############################
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.transition = []
        return policy_loss.item(), value_loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        best_score = -np.inf
        total_score = []
        os.makedirs("v1", exist_ok=True)
        best_model_path = os.path.join("v1", f"LAB7_313551176_task1_a2c_pendulum.pt")
        tq = tqdm(range(1, self.num_episodes))
        for ep in tq: 
            actor_losses, critic_losses, scores = [], [], []
            seed = random.sample(self.random_seed, 1)[0]
            state, _ = self.env.reset(seed=seed)
            score = 0
            done = False
            self.transition = []
            while not done:
                # self.env.render()  # Render the environment
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                state = next_state
                score += reward
                step_count += 1
                # W&B logging
                wandb.log({
                    "step": step_count,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    }) 
                # if episode ends
                if done:
                    scores.append(score)
                    # print(f"Episode {ep}: Total Reward = {score}")
                    # W&B logging
                    wandb.log({
                        "episode": ep,
                        "return": score,
                        })  
            eval_score = self.test()
            total_score.append(eval_score)
            avg_score = np.mean(total_score[-20:])
            tq.set_description(f"Score = {eval_score:.2f}, Avg Score = {avg_score:.2f}")
            if avg_score > best_score:
                best_score = avg_score
                self.save_model(best_model_path)
                # print(f"Saved best model at episode {ep} with score {avg_score} to {best_model_path}")
            wandb.log({"avg_score": avg_score})
            wandb.log({"best_avg_score": best_score})
            if (ep+1) % 100 == 0:
                self.save_model(os.path.join("v1", f"check_point_{ep+1}.pt"))
        torch.save({'actor': self.actor.state_dict(),'critic': self.critic.state_dict()}, "v1_final.pt")
        self.test(video_folder="video_1")
        self.env.close()

    def load_model(self):
        state_dict = torch.load(self.args.model_path)
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.actor.eval()
        self.critic.eval()

    def gen_video(self, video_folder: str):
        """Generate videos for each seed using the trained model."""
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
            self.env = gym.make("Pendulum-v1", render_mode="rgb_array")
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder, name_prefix=f"pendulum-seed-{seed}")
            state, _ = self.env.reset(seed=seed)
            done = False
            score = 0
            with torch.no_grad():
                while not done:
                    action = self.select_action(state)
                    assert action.shape == (1,) and action[0] <= 2.0 and action[0] >= -2.0
                    next_state, reward, done = self.step(action)
                    state = next_state
                    score += float(reward)
            scores.append(score)
            print(f"seed: {seed}, score: {score}")
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

        state, _ = self.env.reset(seed=random.sample(self.random_seed, 1)[0])
        done = False
        score = 0
        with torch.no_grad():
            while not done:
                action = self.select_action(state)
                assert action.shape == (1,) and action[0] <= 2.0 and action[0] >= -2.0
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward


        # print("score: ", score)
        self.env.close()

        self.env = tmp_env
        self.is_test = False
        return score
    
    def save_model(self, path: str):
        """Save the model."""
        state_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }
        torch.save(state_dict, path)

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--gpu", "-g", type=str, default="0")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--entropy-weight", type=float, default=0.01) # entropy can be disabled by setting this to 0
    parser.add_argument("--model-path", "-p", type=str, default="")
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    random_seed = random.sample(range(0, 10000), 20)
    print("Random Seed: ", random_seed)
    # [4140, 5339, 3232, 3940, 3164, 1885, 4789, 7802, 9140, 3896, 2383, 9107, 8202, 39, 4586, 464, 8145, 2829, 3133, 8311]
    
    agent = A2CAgent(env, random_seed, args)
    if len(args.model_path):
        agent.gen_video(video_folder="video_1")
    else:
        wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
        agent.train()