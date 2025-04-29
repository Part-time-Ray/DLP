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
    def __init__(self, input_channel, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            # Layer 1: 4x84x84 -> 32x84x84
            nn.Conv2d(input_channel, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 32x42x42
            # Layer 2: 32x42x42 -> 64x42x42
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 64x21x21
            # Layer 3: 64x21x21 -> 64x21x21
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # Flatten to 64*21*21

            nn.Linear(64 * 21 * 21, 512),
            nn.ReLU(),

            nn.Linear(512, num_actions)
        )
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    model = DQN(4, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    rewards_history = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)
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
            # print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")
        # else:
        #     print(f"Episode {ep} with total reward {total_reward}")
    print(f"\tMax reward over {args.episodes} episodes: {max(rewards_history)}, seed: {args.seed}")
    return max(rewards_history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="results_v3_dqn_smooth_loss_multi_step_1/best_model.pt", help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--video", type=int, default=True, help="Save evaluation videos")
    parser.add_argument("--seed", type=int, default=313551176, help="Random seed for evaluation")
    parser.add_argument("--gpu", "-g", type=str, default='0', help="GPU index to use")
    args = parser.parse_args()
    mp = [
        ("LAB5_313551176_task3_pong200000.pt", 13175, "eval_videos_v3_200000"),
        ("LAB5_313551176_task3_pong400000.pt", 86704, "eval_videos_v3_400000"),
        ("LAB5_313551176_task3_pong600000.pt", 69834, "eval_videos_v3_600000"),
        ("LAB5_313551176_task3_pong800000.pt", 25619, "eval_videos_v3_800000"),
        ("LAB5_313551176_task3_pong1000000.pt", 69685, "eval_videos_v3_1000000")
    ]
    for model_path, seed, output_dir in mp:
        args.model_path = model_path
        args.seed = seed
        args.output_dir = output_dir
        print(f"Evaluating model {model_path} with seed {seed}")
        evaluate(args)
