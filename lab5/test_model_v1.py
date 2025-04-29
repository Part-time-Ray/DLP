import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse
import warnings
warnings.filterwarnings("ignore")

class DQN(nn.Module):
    def __init__(self, num_actions, input_dim=4):
        super(DQN, self).__init__()      
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_actions)
        )       
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x)  
    
class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized / 255

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked
    

        
def evaluate(args):
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    args.seed = 9330
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    model = DQN(env.action_space.n).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    rewards_history = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            frames.append(frame)

            obs_tenser = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(obs_tenser).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs
            frame_idx += 1
        rewards_history.append(total_reward)
        if args.video:
            out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
            with imageio.get_writer(out_path, fps=30) as video:
                for f in frames:
                    height, width = f.shape[:2]
                    new_width = (width // 16) * 16
                    new_height = (height // 16) * 16
                    video.append_data(cv2.resize(f, (new_width, new_height)))
            print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")
        # else:
        #     print(f"Episode {ep} with total reward {total_reward}")
    print(f"Average reward over {args.episodes} episodes: {np.mean(rewards_history)}")
    return np.mean(rewards_history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LAB5_313551176_task1_cartpole.pt", help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos_v1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--video", type=int, default=True, help="Save evaluation videos")
    parser.add_argument("--seed", type=int, default=9330, help="Random seed for evaluation")
    parser.add_argument("--gpu", "-g", type=str, default="0", help="GPU ID to use")
    args = parser.parse_args()
    reward = evaluate(args)
