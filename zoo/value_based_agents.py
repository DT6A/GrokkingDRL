import os.path
import tempfile
import random
import glob
import time
import json
import sys
import os
import gc

from IPython.display import HTML

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *


class ValueBasedAgent:
    def __init__(self):
        pass

    def optimize_model(self, experiences):
        pass

    def interaction_step(self, state, env):
        pass

    def train(self, make_env_fn, make_env_kargs, seed, gamma,
              max_minutes, max_episodes, goal_mean_100_reward):
        pass

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        try:
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]): path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1
        checkpoint_idxs = np.linspace(1, last_ep + 1, n_checkpoints, endpoint=True, dtype=np.int) - 1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def demo_last(self, title='Fully-trained {} Agent', n_episodes=3, max_n_videos=3):
        env = self.make_env_fn(monitor_mode='evaluation', render=True, record=True, **self.make_env_kargs)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.online_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.online_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos,
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def demo_progression(self, title='{} Agent progression', max_n_videos=5):
        env = self.make_env_fn(monitor_mode='evaluation', render=True, record=True, **self.make_env_kargs)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.online_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.online_model, env, n_episodes=1)

        env.close()
        data = get_gif_html(env_videos=env.videos,
                            title=title.format(self.__class__.__name__),
                            subtitle_eps=sorted(checkpoint_paths.keys()),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(),
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))


class FCQ(nn.Module):
    def __init__(self,
                 input_dim, output_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(FCQ, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)

        return x

    def numpy_float_to_device(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable

    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals

class NFQ(ValueBasedAgent):
    def __init__(self, value_model_fn, value_optimizer_fn, value_optimizer_lr, training_strategy_fn,
                 evaluation_strategy_fn, batch_size, epochs):
        ValueBasedAgent.__init__(self)
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.batch_size = batch_size
        self.epochs = epochs

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = self.batch_size

        q_sp = self.online_model(next_states).detach()
        max_a_q_sp = q_sp.max(1)[0].unsqueeze(1)
        target_q_s = rewards + self.gamma * (1 - is_terminals) * max_a_q_sp
        q_sa = self.online_model(states).gather(1, actions)

        td_errors = q_sa - target_q_s
        value_loss = td_errors.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, info = env.step(action)

        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        is_failure = is_terminal and not is_truncated

        experience = (state, action, reward, new_state, float(is_failure))

        self.experiences.append(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)

        return new_state, is_terminal

    def train(self, make_env_fn, make_env_kargs, seed, gamma,
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = tempfile.mkdtemp()
        print(self.checkpoint_dir)
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma

        env = self.make_env_fn(seed=self.seed, **self.make_env_kargs)
        torch.manual_seed(self.seed); np.random.seed(self.seed); random.seed(self.seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        self.online_model = self.value_model_fn(nS, nA)
        self.value_optimizer = self.value_optimizer_fn(self.online_model,
                                                       self.value_optimizer_lr)

        self.training_strategy = self.training_strategy_fn()
        self.evaluation_strategy = self.evaluation_strategy_fn()
        self.experiences = []

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()

            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                state, is_terminal = self.interaction_step(state, env)

                if len(self.experiences) >= self.batch_size:
                    experiences = np.array(self.experiences)
                    batches = [np.vstack(sars) for sars in experiences.T]
                    experiences = self.online_model.load(batches)
                    for _ in range(self.epochs):
                        self.optimize_model(experiences)
                    self.experiences.clear()

                if is_terminal:
                    gc.collect()
                    break

            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_model, env)
            print(f"Epidode {episode} ended, score {evaluation_score}", end='\r', flush=True)
            # total_step = int(np.sum(self.episode_timestep))
            # self.evaluation_scores.append(evaluation_score)
            #
            # mean_10_reward = np.mean(self.episode_reward[-10:])
            # std_10_reward = np.std(self.episode_reward[-10:])
            # mean_100_reward = np.mean(self.episode_reward[-100:])
            # std_100_reward = np.std(self.episode_reward[-100:])
            # mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            # std_100_eval_score = np.std(self.evaluation_scores[-100:])
            # lst_100_exp_rat = np.array(
            #     self.episode_exploration[-100:]) / np.array(self.episode_timestep[-100:])
            # mean_100_exp_rat = np.mean(lst_100_exp_rat)
            # std_100_exp_rat = np.std(lst_100_exp_rat)
            #
            # wallclock_elapsed = time.time() - training_start
            # result[episode - 1] = total_step, mean_100_reward, \
            #                       mean_100_eval_score, training_time, wallclock_elapsed
            #
            # reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            # reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            # reached_max_episodes = episode >= max_episodes
            # reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            # training_is_over = reached_max_minutes or \
            #                    reached_max_episodes or \
            #                    reached_goal_mean_reward
            #
            # elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            # debug_message = 'el {}, ep {:04}, ts {:06}, '
            # debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            # debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            # debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            # debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            # debug_message = debug_message.format(
            #     elapsed_str, episode - 1, total_step, mean_10_reward, std_10_reward,
            #     mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
            #     mean_100_eval_score, std_100_eval_score)
            # print(debug_message, end='\r', flush=True)
            # if reached_debug_time or training_is_over:
            #     print(ERASE_LINE + debug_message, flush=True)
            #     last_debug_time = time.time()
            # if training_is_over:
            #     if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
            #     if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
            #     if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
            #     break

        self.save_checkpoint(max_episodes, self.online_model)
        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
            final_eval_score, score_std, training_time, wallclock_time))
        env.close();
        del env
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time


class DQN(ValueBasedAgent):
    def __init__(self, replay_buffer_fn, value_model_fn, value_optimizer_fn, value_optimizer_lr, training_strategy_fn,
                 evaluation_strategy_fn, n_warmup_batches, update_target_every_steps):
        ValueBasedAgent.__init__(self)
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp.max(1)[0].unsqueeze(1)
        target_q_s = rewards + self.gamma * (1 - is_terminals) * max_a_q_sp
        q_sa = self.online_model(states).gather(1, actions)

        td_errors = q_sa - target_q_s
        value_loss = td_errors.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        is_failure = is_terminal and not is_truncated
        experience = (state, action, reward, new_state, float(is_failure))

        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)
        return new_state, is_terminal

    def update_network(self):
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target.data.copy_(online.data)

    def train(self, make_env_fn, make_env_kargs, seed, gamma,
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma

        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        torch.manual_seed(self.seed);
        np.random.seed(self.seed);
        random.seed(self.seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        self.target_model = self.value_model_fn(nS, nA)
        self.online_model = self.value_model_fn(nS, nA)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model,
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()
        self.evaluation_strategy = self.evaluation_strategy_fn()

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()

            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                state, is_terminal = self.interaction_step(state, env)

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)

                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_network()

                if is_terminal:
                    gc.collect()
                    break

            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_model, env)
            if episode % 100 == 0:
                self.save_checkpoint(episode - 1, self.online_model)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)

            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:]) / np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start
            result[episode - 1] = total_step, mean_100_reward, \
                                  mean_100_eval_score, training_time, wallclock_elapsed

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode - 1, total_step, mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break

        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
            final_eval_score, score_std, training_time, wallclock_time))
        env.close();
        del env
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time


class DDQN(DQN):
    def __init__(self, replay_buffer_fn, value_model_fn, value_optimizer_fn, value_optimizer_lr, max_gradient_norm,
                 training_strategy_fn, evaluation_strategy_fn, n_warmup_batches, update_target_every_steps):
        DQN.__init__(self, replay_buffer_fn, value_model_fn, value_optimizer_fn, value_optimizer_lr,
                     training_strategy_fn, evaluation_strategy_fn, n_warmup_batches, update_target_every_steps)
        self.max_gradient_norm = max_gradient_norm

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        argmax_a_q_qs = self.online_model(next_states).max(1)[1]
        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp[np.arange(batch_size), argmax_a_q_qs].unsqueeze(1)
        target_q_s = rewards + self.gamma * (1 - is_terminals) * max_a_q_sp
        q_sa = self.online_model(states).gather(1, actions)

        td_errors = q_sa - target_q_s
        value_loss = td_errors.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(),
                                       self.max_gradient_norm)
        self.value_optimizer.step()

