"""
PPO implementation based on CleanRL's PPO implementation.
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py
"""
import os
import time
import multiprocessing
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import NormalizeReward
import minigrid
from minigrid.wrappers import ImgObsWrapper
import argparse
from tqdm import tqdm

from models.ctm_rl import ContinuousThoughtMachineRL
from models.lstm_rl import LSTMBaseline
from utils.housekeeping import set_seed
from tasks.rl.envs import MaskVelocityWrapper
from tasks.rl.utils import combine_tracking_data
from tasks.rl.plotting import make_rl_gif
from tasks.image_classification.plotting import plot_neural_dynamics


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTM with RL.")

    # Model Architecture 
    parser.add_argument('--model_type', type=str, default="ctm", choices=['ctm', 'lstm'], help='Sequence length for parity task.')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=64, help='Dimension of the input projection.')
    parser.add_argument('--synapse_depth', type=int, default=1, help='Depth of U-NET model for synapse. 1=linear.')
    parser.add_argument('--n_synch_out', type=int, default=16, help='Number of neurons for output sync.')
    parser.add_argument('--neuron_select_type', type=str, default='random', choices=['first-last', 'random', 'random-pairing'], help='Protocol for selecting neuron subset.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of internal ticks.')
    parser.add_argument('--memory_length', type=int, default=5, help='Length of pre-activation history for NLMs.')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep NLMs.')
    parser.add_argument('--memory_hidden_dims', type=int, default=2, help='Hidden dimensions for deep NLMs.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs.')
    parser.add_argument('--continuous_state_trace', action=argparse.BooleanOptionalAction, default=True, help='Flag to carry over state trace between environment steps.')

    # Environment Configuration 
    parser.add_argument('--env_id', type=str, default="Acrobot-v1", help='Environment ID.')
    parser.add_argument('--mask_velocity', action=argparse.BooleanOptionalAction, default=True, help='Mask the velocity components of the observation.')
    parser.add_argument('--max_environment_steps', type=int, default=500, help='The maximum number of environment steps.')

    # Training Configuration
    parser.add_argument('--num_steps', type=int, default=100, help='The number of environment steps to run in each environment per policy rollout.')
    parser.add_argument('--total_timesteps', type=int, default=1_000_000, help='The combined total of all environment steps (across all batches).')
    parser.add_argument('--num_envs', type=int, default=8, help='The number of parallel game environments.')
    parser.add_argument('--anneal_lr', action=argparse.BooleanOptionalAction, default=True, help='Use learning rate annealing.')
    parser.add_argument('--discount_gamma', type=float, default=0.99, help='The discount factor gamma.')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='The lambda for the Generalized Advantage Estimation (GAE).')
    parser.add_argument('--num_minibatches', type=int, default=4, help='The number of mini-batches.')
    parser.add_argument('--update_epochs', type=int, default=1, help='The number of epochs to update the policy.')
    parser.add_argument('--norm_adv', action=argparse.BooleanOptionalAction, default=True, help='Toggle advantages normalization.')
    parser.add_argument('--clip_coef', type=float, default=0.1, help='The surrogate clipping coefficient.')
    parser.add_argument('--clip_vloss', action=argparse.BooleanOptionalAction, default=False, help='Use clipped loss for the value function (as per the PPO paper).')
    parser.add_argument('--ent_coef', type=float, default=0.1, help='Entropy coefficient.')
    parser.add_argument('--vf_coef', type=float, default=0.25, help='Value function coefficient.')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='The maximum norm for gradient clipping.')
    parser.add_argument('--target_kl', type=float, default=None, help='Target KL divergence threshold.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--num_validation_envs', type=int, default=1, help='Number of environments to evaluate with during training.')

    # Housekeeping 
    parser.add_argument('--log_dir', type=str, default='logs/rl/acrobot', help='Directory for logging.')
    parser.add_argument('--tb_log_dir', type=str, default='logs/runs', help='Directory for tensorboard logging.')
    parser.add_argument('--run_name', type=str, default='default_run', help='Name of the run for logging and tracking.')
    parser.add_argument('--save_every', type=int, default=100, help='Save checkpoint frequency.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=True, help='Reload checkpoint from log_dir?')
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics frequency.')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='GPU(s) or -1 for CPU.')

    # Early stopping and best checkpoint (optional, disabled by default)
    parser.add_argument('--early_stopping_patience', type=int, default=-1, help='Stop training if mean episode reward does not improve for this many save intervals. -1 to disable.')
    parser.add_argument('--save_best_checkpoint', action=argparse.BooleanOptionalAction, default=False, help='Save best model checkpoint separately as best_checkpoint.pt.')

    args = parser.parse_args()
    return args

def make_env_classic_control(env_id, max_environment_steps, mask_velocity=True, render_mode=None):
    def thunk():
        env = gym.make(env_id, render_mode=render_mode)
        if mask_velocity: env = MaskVelocityWrapper(env)
        env = NormalizeReward(env, gamma=0.99, epsilon=1e-8)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_environment_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def make_env_minigrid(env_id, max_environment_steps):
    def thunk():
        env = gym.make(env_id, max_steps=max_environment_steps, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        env = NormalizeReward(env, gamma=0.99, epsilon=1e-8)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_environment_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, size_action_space, args, device):
        super().__init__()

        self.continious_state_trace = args.continuous_state_trace
        self.device = device
        self.model_type = args.model_type

        if "MiniGrid" in args.env_id:
            backbone_type='navigation-backbone'
        else:
            backbone_type='classic-control-backbone'

        if args.model_type == "ctm":
            self.recurrent_model = ContinuousThoughtMachineRL(
                iterations=args.iterations,
                d_model=args.d_model,
                d_input=args.d_input,  
                n_synch_out=args.n_synch_out,
                synapse_depth=args.synapse_depth,
                memory_length=args.memory_length,  
                deep_nlms=args.deep_memory,
                memory_hidden_dims=args.memory_hidden_dims,  
                do_layernorm_nlm=args.do_normalisation,  
                backbone_type=backbone_type,
                prediction_reshaper=[-1],
                dropout=args.dropout,          
                neuron_select_type=args.neuron_select_type,
            )
            actor_input_dim = critic_input_dim = self.recurrent_model.synch_representation_size_out
        else:
            self.recurrent_model = LSTMBaseline(
                iterations=args.iterations,
                d_model=args.d_model,
                d_input=args.d_input,
                backbone_type=backbone_type,
            )
            actor_input_dim = critic_input_dim = args.d_model

        self.actor = nn.Sequential(
            layer_init(nn.Linear(actor_input_dim, 64), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(64, size_action_space), std=1)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_input_dim, 64), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1)
        )

    def get_initial_state(self, num_envs):
        if self.model_type == "ctm":
            return self.get_initial_ctm_state(num_envs)
        elif self.model_type == "lstm":
            return self.get_initial_lstm_state(num_envs)
        else:
            raise ValueError("Model type not supported.")

    def get_initial_ctm_state(self, num_envs):
        initial_state_trace = torch.repeat_interleave(self.recurrent_model.start_trace.unsqueeze(0), num_envs, 0)
        initial_activated_state_trace = torch.repeat_interleave(self.recurrent_model.start_activated_trace.unsqueeze(0), num_envs, 0)
        return initial_state_trace, initial_activated_state_trace
    
    def get_initial_lstm_state(self, num_envs):
        initial_hidden_state = torch.repeat_interleave(self.recurrent_model.start_hidden_state.unsqueeze(0), num_envs, 0)
        initial_cell_state = torch.repeat_interleave(self.recurrent_model.start_cell_state.unsqueeze(0), num_envs, 0)
        return initial_hidden_state, initial_cell_state

    def _get_hidden_states(self, state, done, num_envs):
        if self.model_type == "ctm":
            return self._get_ctm_hidden_states(state, done, num_envs)
        elif self.model_type == "lstm":
            return self._get_lstm_hidden_states(state, done, num_envs)
        else:
            raise ValueError("Model type not supported.")
    

    def _get_lstm_hidden_states(self, lstm_state, done, num_envs):
        initial_hidden_state, initial_cell_state = self.get_initial_lstm_state(num_envs)
        # Assuming continious hidden states
        masked_previous_hidden_state = (1.0 - done).view(-1, 1) * lstm_state[0]
        masked_previous_cell_state_state = (1.0 - done).view(-1, 1) * lstm_state[1]
        masked_initial_hidden_state = done.view(-1, 1) * initial_hidden_state
        masked_initial_cell_state = done.view(-1, 1) * initial_cell_state
        return (masked_previous_hidden_state + masked_initial_hidden_state), (masked_previous_cell_state_state + masked_initial_cell_state)


    def _get_ctm_hidden_states(self, ctm_state, done, num_envs):
        initial_state_trace, initial_activated_state_trace = self.get_initial_ctm_state(num_envs)
        if self.continious_state_trace:
            masked_previous_state_trace = (1.0 - done).view(-1, 1, 1) * ctm_state[0]
            masked_previous_activated_state_trace = (1.0 - done).view(-1, 1, 1) * ctm_state[1]
            masked_initial_state_trace = done.view(-1, 1, 1) * initial_state_trace
            masked_initial_activated_state_trace = done.view(-1, 1, 1) * initial_activated_state_trace
            return (masked_previous_state_trace + masked_initial_state_trace), (masked_previous_activated_state_trace + masked_initial_activated_state_trace)
        else:
            return (initial_state_trace, initial_activated_state_trace)

    def get_states(self, x, ctm_state, done, track=False):
        num_envs = ctm_state[0].shape[0]

        if len(x.shape) == 4:
            _, C, H, W = x.shape
            xs = x.reshape((-1, num_envs, C, H, W))
        elif len(x.shape) == 2:
            _, C = x.shape
            xs = x.reshape((-1, num_envs, C))
        else:
            raise ValueError("Input shape not supported.")
        
        done = done.reshape((-1, num_envs))
        new_hidden = []
        for x, d in zip(xs, done):
            if not track:
                synchronisation, ctm_state = self.recurrent_model(x, self._get_hidden_states(ctm_state, d, num_envs))
                tracking_data = None
                new_hidden += [synchronisation]
            else:
                synchronisation, ctm_state, pre_activations, post_activations = self.recurrent_model(x, self._get_hidden_states(ctm_state, d, num_envs), track=True)
                tracking_data = {
                    'pre_activations': pre_activations,
                    'post_activations': post_activations,
                    'synchronisation': synchronisation.detach().cpu().numpy(),
                }
                new_hidden += [synchronisation]
        
        return torch.cat(new_hidden), ctm_state, tracking_data

    def get_value(self, x, ctm_state, done):
        hidden, _, _ = self.get_states(x, ctm_state, done)
        return self.critic(hidden)
    
    def get_action_and_value(self, x, ctm_state, done, action=None, track=False):
        hidden, ctm_state, tracking_data = self.get_states(x, ctm_state, done, track=track)
        action_logits = self.actor(hidden)
        action_probs = Categorical(logits=action_logits)

        if action is None:
            action = action_probs.sample()
        
        value = self.critic(hidden)
        
        return action, action_probs.log_prob(action), action_probs.entropy(), value, ctm_state, tracking_data, action_logits, action_probs.probs

def save_model(agent, optimizer, global_step, training_iteration, episode_rewards_tracking, episode_lengths_tracking, global_steps_tracking, args, save_path, best_test_metric=-float('inf'), early_stopping_counter=0):
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'training_iteration': training_iteration,
        'episode_rewards_tracking': episode_rewards_tracking,
        'episode_lengths_tracking': episode_lengths_tracking,
        'global_steps_tracking': global_steps_tracking,
        'args': args,
        'best_test_metric': best_test_metric,
        'early_stopping_counter': early_stopping_counter,
    }, save_path)

def load_model(agent, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    global_step = checkpoint.get('global_step', 0)
    training_iteration = checkpoint.get('training_iteration', 0)
    episode_rewards_tracking = checkpoint.get('episode_rewards_tracking', [])
    episode_lengths_tracking = checkpoint.get('episode_lengths_tracking', [])
    global_steps_tracking = checkpoint.get('global_steps_tracking', [])
    model_args = checkpoint.get('args', None)

    print(f"Loaded checkpoint from {checkpoint_path} at iteration {training_iteration} and step {global_step}")
    return global_step, training_iteration, episode_rewards_tracking, episode_lengths_tracking, global_steps_tracking, model_args


def plot_activations(agent, device, args):
    agent.eval()
    with torch.no_grad():
        for idx in range(args.num_validation_envs):
            if args.env_id in ("CartPole-v1", "Acrobot-v1"):
                eval_env = make_env_classic_control(args.env_id, args.max_environment_steps, mask_velocity=args.mask_velocity, render_mode="rgb_array")()
            elif "MiniGrid" in args.env_id:
                eval_env = make_env_minigrid(args.env_id, args.max_environment_steps)()
            else:
                raise NotImplementedError(f"Environment {args.env_id} not supported.")

            eval_next_obs, _ = eval_env.reset()
            episode_reward = 0
            eval_next_done = False
            eval_state = agent.get_initial_state(1)
            tracking_data_by_world_step = []
            for _ in range(args.max_environment_steps):
                with torch.no_grad():
                    action, _, _, value, eval_state, tracking_data, action_logits, action_probs = agent.get_action_and_value(
                        torch.Tensor(eval_next_obs).to(device).unsqueeze(0),
                        eval_state,
                        torch.Tensor([eval_next_done]).to(device),
                        track=True
                    )
                eval_next_obs, reward, termination, truncation, _ = eval_env.step(action.cpu().numpy()[0])
                eval_next_done = termination or truncation

                tracking_data['actions'] = np.tile(action.detach().cpu().numpy(), (args.iterations)) # Shape T
                tracking_data['values'] = np.tile(value.squeeze(-1).detach().cpu().numpy(), (args.iterations)) # Shape T
                tracking_data['action_logits'] = np.tile(action_logits.detach().cpu().numpy(), (args.iterations, 1)) # Shape T, A
                tracking_data['action_probs'] = np.tile(action_probs.detach().cpu().numpy(), (args.iterations, 1)) # Shape T, A
                tracking_data['rewards'] = np.tile(np.array(reward), (args.iterations)) # Shape T
                tracking_data['inputs'] = np.tile(np.array(eval_env.render()), (args.iterations, 1, 1, 1)) # Shape T, H, W, C

                tracking_data_by_world_step.append(tracking_data)

                episode_reward += reward

                if eval_next_done:
                    break

            eval_env.close()

            combined_tracking_data = combine_tracking_data(tracking_data_by_world_step)

            n_to_plot = 80 if combined_tracking_data['post_activations'].shape[-1] < 100 else 100
            plot_neural_dynamics(combined_tracking_data['post_activations'], n_to_plot, args.log_dir, axis_snap=True)

            process = multiprocessing.Process(
                target=make_rl_gif,
                args=(
                    combined_tracking_data['action_logits'],
                    combined_tracking_data['action_probs'],
                    combined_tracking_data['actions'],
                    combined_tracking_data['values'],
                    combined_tracking_data['rewards'],
                    combined_tracking_data['pre_activations'],
                    combined_tracking_data['post_activations'],
                    combined_tracking_data['inputs'],
                    f"{args.log_dir}/eval_output_val_{idx}.gif"
                )
            )
            process.start()
        agent.train()

def initialize_args():
    args = parse_args()
    args = initialise_dynamic_args(args)
    return args

def initialise_dynamic_args(args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_training_iterations = args.total_timesteps // args.batch_size
    return args

if __name__ == "__main__":

    args = initialize_args()

    set_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"Tensorboard logging to {args.tb_log_dir}/{args.run_name}")

    writer = SummaryWriter(f"{args.tb_log_dir}/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment Setup
    if args.env_id in ("CartPole-v1", "Acrobot-v1"):
        envs = gym.vector.SyncVectorEnv([make_env_classic_control(args.env_id, args.max_environment_steps, args.mask_velocity) for _ in range(args.num_envs)])
    elif "MiniGrid" in args.env_id:
        envs = gym.vector.SyncVectorEnv([make_env_minigrid(args.env_id, args.max_environment_steps) for _ in range(args.num_envs)])

    agent = Agent(envs.single_action_space.n, args, device).to(device)
    plot_activations(agent, device, args)
    print(f'Total params: {sum(p.numel() for p in agent.parameters())}')
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    checkpoint_path = f"{args.log_dir}/checkpoint.pt"
    if os.path.exists(checkpoint_path) and args.reload:
        global_step, training_iteration, episode_rewards_tracking, episode_lengths_tracking, global_steps_tracking, _ = load_model(agent, optimizer, checkpoint_path, device)
        best_test_metric = checkpoint.get('best_test_metric', -float('inf'))
        early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
    else:
        global_step, training_iteration, episode_rewards_tracking, episode_lengths_tracking, global_steps_tracking = 0, 0, [], [], []
        best_test_metric = -float('inf')
        early_stopping_counter = 0

    # Rollout buffer
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_state = agent.get_initial_state(args.num_envs)

    progress_bar = tqdm(range(training_iteration, args.num_training_iterations + 1), total=args.num_training_iterations + 1, initial=training_iteration, desc="Training", dynamic_ncols=True,)

    for training_iteration in progress_bar:
        initial_state = (next_state[0].clone(), next_state[1].clone())

        if args.anneal_lr:
            frac = 1.0 - (training_iteration - 1.0) / args.num_training_iterations
            lrnow = frac * args.lr
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            next_obs = torch.Tensor(next_obs).to(device)
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, next_state, _, _, _ = agent.get_action_and_value(next_obs, next_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info[0]:
                        progress_bar.set_postfix({
                            "step": global_step,
                            "train. iter": training_iteration,
                            "return": round(info[0]["episode"]["r"], 2),
                            "length": info[0]["episode"]["l"]
                        })
                        writer.add_scalar("charts/episodic_return", info[0]["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info[0]["episode"]["l"], global_step)
                        episode_rewards_tracking.append(info[0]["episode"]["r"])
                        episode_lengths_tracking.append(info[0]["episode"]["l"])
                        global_steps_tracking.append(global_step)

            elif "episode" in infos:
                if infos["episode"]:
                    episode_rewards = infos["episode"]["r"]
                    episode_lengths = infos["episode"]["l"]
                    completed_episodes = infos["episode"]["_r"]
                    for env_idx in range(len(completed_episodes)):
                        if completed_episodes[env_idx]:
                            progress_bar.set_postfix({
                                "step": global_step,
                                "train. iter": training_iteration,
                                "return": round(episode_rewards[env_idx], 2),
                                "length": episode_lengths[env_idx]
                            })
                            writer.add_scalar("charts/episodic_return", episode_rewards[env_idx], global_step)
                            writer.add_scalar("charts/episodic_length", episode_lengths[env_idx], global_step)
                            episode_rewards_tracking.append(episode_rewards[env_idx])
                            episode_lengths_tracking.append(episode_lengths[env_idx])
                            global_steps_tracking.append(global_step)

        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.discount_gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.discount_gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                if args.model_type == "ctm":
                    selected_hidden_state = (initial_state[0][mbenvinds,:,:], initial_state[1][mbenvinds,:,:])
                elif args.model_type == "lstm":
                    selected_hidden_state = (initial_state[0][mbenvinds,:], initial_state[1][mbenvinds,:])


                _, newlogprob, entropy, newvalue, _, _, _, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    selected_hidden_state,
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()


                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()

                # Log gradient norms
                total_norm = 0
                for name, param in agent.named_parameters():
                    if param.grad is None:
                        print(f"Warning: Gradient for {name} is None!")
                    else:
                        param_norm = param.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                        writer.add_scalar(f"grad_norms/{name}", param_norm, global_step)
                total_norm = total_norm ** 0.5
                writer.add_scalar("grad_norms/total", total_norm, global_step)


                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        if training_iteration % args.track_every == 0 or training_iteration == 1:
            plot_activations(agent, device, args)

        if training_iteration % args.save_every == 0 or training_iteration == 1 or global_step == args.total_timesteps-1:
            save_model(agent, optimizer, global_step, training_iteration, episode_rewards_tracking, episode_lengths_tracking, global_steps_tracking, args, f"{args.log_dir}/checkpoint.pt", best_test_metric, early_stopping_counter)

            # Best model checkpoint and early stopping (based on recent episode rewards)
            if len(episode_rewards_tracking) > 0:
                current_metric = np.mean(episode_rewards_tracking[-100:])
                if current_metric > best_test_metric:
                    best_test_metric = current_metric
                    early_stopping_counter = 0
                    if args.save_best_checkpoint:
                        save_model(agent, optimizer, global_step, training_iteration, episode_rewards_tracking, episode_lengths_tracking, global_steps_tracking, args, f"{args.log_dir}/best_checkpoint.pt", best_test_metric, early_stopping_counter)
                        print(f'New best model saved at iteration {training_iteration} with mean reward {best_test_metric:.4f}')
                else:
                    early_stopping_counter += 1
                    if args.early_stopping_patience != -1 and early_stopping_counter >= args.early_stopping_patience:
                        print(f'Early stopping triggered at iteration {training_iteration}. '
                              f'No improvement for {args.early_stopping_patience} save intervals.')
                        break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/lr", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()