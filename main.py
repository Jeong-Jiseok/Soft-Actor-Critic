import os
import time
import torch
import numpy as np
import gymnasium as gym
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from sac_agent import SACAgent
from replay_buffer import ReplayBuffer
from utils import set_seed, load_config


def evaluate_policy(agent, eval_env, eval_episodes, seed):
    returns = []

    for i in range(eval_episodes):
        state, _ = eval_env.reset(seed = seed+100+i)
        done = False
        total_reward = 0
        
        while not done:
            with torch.no_grad():
                action = agent.select_action(state, eval=True)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        returns.append(total_reward)

    return np.mean(returns), np.std(returns)

# ====================== main ======================
args = load_config("sac_config.yaml")

# Create environment
env = gym.make(args.env)
eval_env = gym.make(args.env)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Set random seeds
set_seed(args.seed)

# Initialize replay buffer, agent, tensorboard writer
buffer = ReplayBuffer(capacity=args.capacity, state_dim=state_dim, action_dim=action_dim)
agent = SACAgent(state_dim, action_dim, args.sac, action_space=env.action_space)

now = datetime.now().strftime("%m%d_%H%M")
log_dir = f"{args.log_dir}_{args.env}/seed_{args.seed}_{now}"
model_save_dir = f"{args.model_save_dir}_{args.env}/seed_{args.seed}_{now}"
writer = SummaryWriter(log_dir = log_dir)

# Environment state tracking
state, _ = env.reset(seed=args.seed)
total_steps = 0
episode_steps = 0
episode_reward = 0
episode = 1

# ========== Main training loop ==========
while total_steps < args.total_steps:
    # Select action
    if total_steps < args.start_steps:
        action = env.action_space.sample() # Random exploration before learning starts
    else:
        action = agent.select_action(state, eval=False)

    # Interact with environment
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    buffer.add(state, action, reward, next_state, terminated)

    # Update episode stats
    state = next_state
    episode_reward += reward
    episode_steps += 1
    total_steps += 1

    # Start training after enough initial data
    if len(buffer) >= args.sac.batch_size:
        for _ in range(args.updates_per_step):
            s, a, r, ns, d = buffer.sample(args.sac.batch_size)
            actor_loss, critic_loss, alpha_loss, alpha_clone = agent.update(s, a, r, ns, d)
        writer.add_scalar("loss/actor_loss", actor_loss, agent.total_it)
        writer.add_scalar("loss/critic_loss", critic_loss, agent.total_it)
        writer.add_scalar("loss/alpha_loss", alpha_loss, agent.total_it)
        writer.add_scalar("entropy_temprature/alpha_clone", alpha_clone, agent.total_it)

    # If episode ends, reset environment
    if done:
        writer.add_scalar("train/return", episode_reward, total_steps)
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode += 1

    # Evaluation
    if total_steps % args.eval_interval == 0:
        start_time = time.time()
        eval_return, std = evaluate_policy(agent, eval_env, args.eval_episodes, args.seed)
        elapsed = time.time() - start_time

        writer.add_scalar("eval/return", eval_return, total_steps)
        writer.add_scalar("eval/return_std", std, total_steps)
        
        print(f"✅ [Seed {args.seed}] Step {total_steps} | Eval: {eval_return:.1f} ± {std:.2f} | ⏱️ {elapsed:.2f}s")

    # Model save
    if total_steps % args.save_interval == 0:
        agent.save(os.path.join(model_save_dir, f"sac_step{total_steps}.pt"))
        print(f"💾 Saved models at step {total_steps} to {args.model_save_dir}")

# Cleanup
env.close()
eval_env.close()
writer.close()