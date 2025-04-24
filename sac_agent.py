import os
import torch
import torch.nn.functional as F
from model import Actor, Critic

# =======================
# Soft Actor-Critic Agent with entropy tuning
# =======================
class SACAgent:
    def __init__(self, state_dim, action_dim, args, action_space=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, args, action_space).to(self.device)
        self.critic = Critic(state_dim, action_dim, args).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.target_update_interval = args.target_update_interval
        self.total_it = 0

        if self.automatic_entropy_tuning:
            if args.target_entropy is not None:
                self.target_entropy = args.target_entropy
            else:
                if action_space is not None:
                    self.target_entropy = -torch.prod(torch.tensor(action_space.shape).to(self.device)).item()
                else:
                    self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr)
            
    # Action selection
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if eval:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
            
        action = action.detach().cpu().numpy()
        return action[0] if action.shape[0] == 1 else action

    # Agent update
    def update(self, state, action, reward, next_state, done):
        self.total_it += 1

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # ===== Critic update =====
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ===== Actor update =====
        pi, log_pi, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        actor_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ===== Alpha update =====
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = torch.exp(self.log_alpha)
            alpha_clone = self.alpha.clone() # For Tensorboard logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_clone = torch.tensor(self.alpha) # For TensorboardX logs

        # ===== Target network soft update =====
        if self.total_it % self.target_update_interval == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item(), alpha_loss.item(), alpha_clone.item()

    # Save the entire agent model (networks + optimizers + alpha)
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {"actor": self.actor.state_dict(), 
                      "critic": self.critic.state_dict(), 
                      "critic_target": self.critic_target.state_dict(), 
                      "actor_optimizer": self.actor_optimizer.state_dict(), 
                      "critic_optimizer": self.critic_optimizer.state_dict(), 
                      "alpha": self.alpha
                      }
        
        if self.automatic_entropy_tuning:
            checkpoint["log_alpha"] = self.log_alpha
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()

        torch.save(checkpoint, path)

    # Load the saved agent model (supports automatic entropy tuning)
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha = checkpoint["alpha"]

        if self.automatic_entropy_tuning and "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"].to(self.device)
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])