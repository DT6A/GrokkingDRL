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

import torch.optim as optim
import torch.multiprocessing as mp
import threading

from .utils import *


class A3C():
    def __init__(self,
                 policy_model_fn,
                 policy_model_max_grad_norm,
                 policy_optimizer_fn,
                 policy_optimizer_lr,
                 value_model_fn,
                 value_model_max_grad_norm,
                 value_optimizer_fn,
                 value_optimizer_lr,
                 entropy_loss_weight,
                 max_n_steps,
                 n_workers):
        self.policy_model_fn = policy_model_fn
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr

        self.value_model_fn = value_model_fn
        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr

        self.entropy_loss_weight = entropy_loss_weight
        self.max_n_steps = max_n_steps
        self.n_workers = n_workers

    def optimize_model(self, logpas, entropies, rewards, values,
                       local_policy_model, local_value_model):
        T = len(rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T - t] * rewards[t:]) for t in range(T)])
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)

        logpas = torch.cat(logpas)
        entropies = torch.cat(entropies)
        values = torch.cat(values)

        value_error = returns - values
        policy_loss = -(discounts * value_error.detach() * logpas).mean()
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss
        self.shared_policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_policy_model.parameters(),
                                       self.policy_model_max_grad_norm)
        for param, shared_param in zip(local_policy_model.parameters(),
                                       self.shared_policy_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_policy_optimizer.step()
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())

        value_loss = value_error.pow(2).mul(0.5).mean()
        self.shared_value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_value_model.parameters(),
                                       self.value_model_max_grad_norm)
        for param, shared_param in zip(local_value_model.parameters(),
                                       self.shared_value_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_value_optimizer.step()
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

    @staticmethod
    def interaction_step(state, env, local_policy_model, local_value_model,
                         logpas, entropies, rewards, values):
        action, is_exploratory, logpa, entropy = local_policy_model.full_pass(state)
        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']

        logpas.append(logpa)
        entropies.append(entropy)
        rewards.append(reward)
        values.append(local_value_model(state))

        return new_state, reward, is_terminal, is_truncated, is_exploratory

    def work(self, rank):
        last_debug_time = float('-inf')
        self.stats['n_active_workers'].add_(1)

        local_seed = self.seed + rank
        env = self.make_env_fn(**self.make_env_kargs, seed=local_seed)
        torch.manual_seed(local_seed);
        np.random.seed(local_seed);
        random.seed(local_seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        local_policy_model = self.policy_model_fn(nS, nA)
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())
        local_value_model = self.value_model_fn(nS)
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

        global_episode_idx = self.stats['episode'].add_(1).item() - 1
        while not self.get_out_signal:
            episode_start = time.time()
            state, is_terminal = env.reset(), False

            # collect n_steps rollout
            n_steps_start, total_episode_rewards = 0, 0
            total_episode_steps, total_episode_exploration = 0, 0
            logpas, entropies, rewards, values = [], [], [], []

            for step in count(start=1):
                state, reward, is_terminal, is_truncated, is_exploratory = self.interaction_step(
                    state, env, local_policy_model, local_value_model,
                    logpas, entropies, rewards, values)

                total_episode_steps += 1
                total_episode_rewards += reward
                total_episode_exploration += int(is_exploratory)

                if is_terminal or step - n_steps_start == self.max_n_steps:
                    is_failure = is_terminal and not is_truncated
                    next_value = 0 if is_failure else local_value_model(state).detach().item()
                    rewards.append(next_value)

                    self.optimize_model(logpas, entropies, rewards, values,
                                        local_policy_model, local_value_model)
                    logpas, entropies, rewards, values = [], [], [], []
                    n_steps_start = step

                if is_terminal:
                    gc.collect()
                    break

            # save global stats
            episode_elapsed = time.time() - episode_start
            evaluation_score, _ = self.evaluate(local_policy_model, env)
            self.save_checkpoint(global_episode_idx, local_policy_model)

            self.stats['episode_elapsed'][global_episode_idx].add_(episode_elapsed)
            self.stats['episode_timestep'][global_episode_idx].add_(total_episode_steps)
            self.stats['episode_reward'][global_episode_idx].add_(total_episode_rewards)
            self.stats['episode_exploration'][global_episode_idx].add_(total_episode_exploration / total_episode_steps)
            self.stats['evaluation_scores'][global_episode_idx].add_(evaluation_score)

            mean_10_reward = self.stats[
                                 'episode_reward'][:global_episode_idx + 1][-10:].mean().item()
            mean_100_reward = self.stats[
                                  'episode_reward'][:global_episode_idx + 1][-100:].mean().item()
            mean_100_eval_score = self.stats[
                                      'evaluation_scores'][:global_episode_idx + 1][-100:].mean().item()
            mean_100_exp_rat = self.stats[
                                   'episode_exploration'][:global_episode_idx + 1][-100:].mean().item()
            std_10_reward = self.stats[
                                'episode_reward'][:global_episode_idx + 1][-10:].std().item()
            std_100_reward = self.stats[
                                 'episode_reward'][:global_episode_idx + 1][-100:].std().item()
            std_100_eval_score = self.stats[
                                     'evaluation_scores'][:global_episode_idx + 1][-100:].std().item()
            std_100_exp_rat = self.stats[
                                  'episode_exploration'][:global_episode_idx + 1][-100:].std().item()
            if std_10_reward != std_10_reward: std_10_reward = 0
            if std_100_reward != std_100_reward: std_100_reward = 0
            if std_100_eval_score != std_100_eval_score: std_100_eval_score = 0
            if std_100_exp_rat != std_100_exp_rat: std_100_exp_rat = 0
            global_n_steps = self.stats[
                                 'episode_timestep'][:global_episode_idx + 1].sum().item()
            global_training_elapsed = self.stats[
                                          'episode_elapsed'][:global_episode_idx + 1].sum().item()
            wallclock_elapsed = time.time() - self.training_start

            self.stats['result'][global_episode_idx][0].add_(global_n_steps)
            self.stats['result'][global_episode_idx][1].add_(mean_100_reward)
            self.stats['result'][global_episode_idx][2].add_(mean_100_eval_score)
            self.stats['result'][global_episode_idx][3].add_(global_training_elapsed)
            self.stats['result'][global_episode_idx][4].add_(wallclock_elapsed)

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, global_episode_idx, global_n_steps, mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)

            if rank == 0:
                print(debug_message, end='\r', flush=True)
                if time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS:
                    print(ERASE_LINE + debug_message, flush=True)
                    last_debug_time = time.time()

            with self.get_out_lock:
                potential_next_global_episode_idx = self.stats['episode'].item()
                self.reached_goal_mean_reward.add_(
                    mean_100_eval_score >= self.goal_mean_100_reward)
                self.reached_max_minutes.add_(
                    time.time() - self.training_start >= self.max_minutes * 60)
                self.reached_max_episodes.add_(
                    potential_next_global_episode_idx >= self.max_episodes)
                if self.reached_max_episodes or \
                        self.reached_max_minutes or \
                        self.reached_goal_mean_reward:
                    self.get_out_signal.add_(1)
                    break
                # else go work on another episode
                global_episode_idx = self.stats['episode'].add_(1).item() - 1

        while rank == 0 and self.stats['n_active_workers'].item() > 1:
            pass

        if rank == 0:
            print(ERASE_LINE + debug_message)
            if self.reached_max_minutes: print(u'--> reached_max_minutes \u2715')
            if self.reached_max_episodes: print(u'--> reached_max_episodes \u2715')
            if self.reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')

        env.close();
        del env
        self.stats['n_active_workers'].sub_(1)

    def train(self, make_env_fn, make_env_kargs, seed, gamma,
              max_minutes, max_episodes, goal_mean_100_reward):

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        self.max_minutes = max_minutes
        self.max_episodes = max_episodes
        self.goal_mean_100_reward = goal_mean_100_reward

        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        nS, nA = env.observation_space.shape[0], env.action_space.n
        torch.manual_seed(self.seed);
        np.random.seed(self.seed);
        random.seed(self.seed)

        self.stats = {}
        self.stats['episode'] = torch.zeros(1, dtype=torch.int).share_memory_()
        self.stats['result'] = torch.zeros([max_episodes, 5]).share_memory_()
        self.stats['evaluation_scores'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_reward'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_timestep'] = torch.zeros([max_episodes], dtype=torch.int).share_memory_()
        self.stats['episode_exploration'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_elapsed'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['n_active_workers'] = torch.zeros(1, dtype=torch.int).share_memory_()

        self.shared_policy_model = self.policy_model_fn(nS, nA).share_memory()
        self.shared_policy_optimizer = self.policy_optimizer_fn(self.shared_policy_model,
                                                                self.policy_optimizer_lr)
        self.shared_value_model = self.value_model_fn(nS).share_memory()
        self.shared_value_optimizer = self.value_optimizer_fn(self.shared_value_model,
                                                              self.value_optimizer_lr)

        self.get_out_lock = mp.Lock()
        self.get_out_signal = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_max_minutes = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_max_episodes = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_goal_mean_reward = torch.zeros(1, dtype=torch.int).share_memory_()
        self.training_start = time.time()
        workers = [mp.Process(target=self.work, args=(rank,)) for rank in range(self.n_workers)]
        [w.start() for w in workers];
        [w.join() for w in workers]
        wallclock_time = time.time() - self.training_start

        final_eval_score, score_std = self.evaluate(self.shared_policy_model, env, n_episodes=100)
        env.close();
        del env

        final_episode = self.stats['episode'].item()
        training_time = self.stats['episode_elapsed'][:final_episode + 1].sum().item()

        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
            final_eval_score, score_std, training_time, wallclock_time))

        self.stats['result'] = self.stats['result'].numpy()
        self.stats['result'][final_episode:, ...] = np.nan
        self.get_cleaned_checkpoints()
        return self.stats['result'], final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1, greedy=True):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                if greedy:
                    a = eval_policy_model.select_greedy_action(s)
                else:
                    a = eval_policy_model.select_action(s)
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
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.shared_policy_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.shared_policy_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos,
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def demo_progression(self, title='{} Agent progression', max_n_videos=5):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.shared_policy_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.shared_policy_model, env, n_episodes=1)

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


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        super().step(closure)


class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super(SharedRMSprop, self).__init__(
            params, lr=lr, alpha=alpha,
            eps=eps, weight_decay=weight_decay,
            momentum=momentum, centered=centered)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['square_avg'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if momentum > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data).share_memory_()
                if centered:
                    state['grad_avg'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        super().step(closure)


class GAE(A3C):
    def __init__(self, policy_model_fn, policy_model_max_grad_norm, policy_optimizer_fn, policy_optimizer_lr,
                 value_model_fn, value_model_max_grad_norm, value_optimizer_fn, value_optimizer_lr, entropy_loss_weight,
                 max_n_steps, n_workers, tau):
        A3C.__init__(self, policy_model_fn, policy_model_max_grad_norm, policy_optimizer_fn, policy_optimizer_lr,
                     value_model_fn, value_model_max_grad_norm, value_optimizer_fn, value_optimizer_lr,
                     entropy_loss_weight, max_n_steps, n_workers)

        self.tau = tau

    def optimize_model(self, logpas, entropies, rewards, values,
                       local_policy_model, local_value_model):
        T = len(rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T - t] * rewards[t:]) for t in range(T)])

        logpas = torch.cat(logpas)
        entropies = torch.cat(entropies)
        values = torch.cat(values)

        np_values = values.view(-1).data.numpy()
        tau_discounts = np.logspace(0, T - 1, num=T - 1, base=self.gamma * self.tau, endpoint=False)
        advs = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]
        gaes = np.array([np.sum(tau_discounts[:T - 1 - t] * advs[t:]) for t in range(T - 1)])

        values = values[:-1, ...]
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)
        gaes = torch.FloatTensor(gaes).unsqueeze(1)

        policy_loss = -(discounts * gaes.detach() * logpas).mean()
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss
        self.shared_policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_policy_model.parameters(),
                                       self.policy_model_max_grad_norm)
        for param, shared_param in zip(local_policy_model.parameters(),
                                       self.shared_policy_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_policy_optimizer.step()
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())

        value_error = returns - values
        value_loss = value_error.pow(2).mul(0.5).mean()
        self.shared_value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_value_model.parameters(),
                                       self.value_model_max_grad_norm)
        for param, shared_param in zip(local_value_model.parameters(),
                                       self.shared_value_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_value_optimizer.step()
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

    @staticmethod
    def interaction_step(state, env, local_policy_model, local_value_model,
                         logpas, entropies, rewards, values):
        action, is_exploratory, logpa, entropy = local_policy_model.full_pass(state)
        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']

        logpas.append(logpa)
        entropies.append(entropy)
        rewards.append(reward)
        values.append(local_value_model(state))

        return new_state, reward, is_terminal, is_truncated, is_exploratory

    def work(self, rank):
        last_debug_time = float('-inf')
        self.stats['n_active_workers'].add_(1)

        local_seed = self.seed + rank
        env = self.make_env_fn(**self.make_env_kargs, seed=local_seed)
        torch.manual_seed(local_seed);
        np.random.seed(local_seed);
        random.seed(local_seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        local_policy_model = self.policy_model_fn(nS, nA)
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())
        local_value_model = self.value_model_fn(nS)
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

        global_episode_idx = self.stats['episode'].add_(1).item() - 1
        while not self.get_out_signal:
            episode_start = time.time()
            state, is_terminal = env.reset(), False

            # collect n_steps rollout
            n_steps_start, total_episode_rewards = 0, 0
            total_episode_steps, total_episode_exploration = 0, 0
            logpas, entropies, rewards, values = [], [], [], []

            for step in count(start=1):
                state, reward, is_terminal, is_truncated, is_exploratory = self.interaction_step(
                    state, env, local_policy_model, local_value_model,
                    logpas, entropies, rewards, values)

                total_episode_steps += 1
                total_episode_rewards += reward
                total_episode_exploration += int(is_exploratory)

                if is_terminal or step - n_steps_start == self.max_n_steps:
                    is_failure = is_terminal and not is_truncated
                    next_value = 0 if is_failure else local_value_model(state).detach().item()
                    rewards.append(next_value)
                    values.append(torch.FloatTensor([[next_value, ], ]))

                    self.optimize_model(logpas, entropies, rewards, values,
                                        local_policy_model, local_value_model)
                    logpas, entropies, rewards, values = [], [], [], []
                    n_steps_start = step

                if is_terminal:
                    gc.collect()
                    break

            # save global stats
            episode_elapsed = time.time() - episode_start
            evaluation_score, _ = self.evaluate(local_policy_model, env)
            self.save_checkpoint(global_episode_idx, local_policy_model)

            self.stats['episode_elapsed'][global_episode_idx].add_(episode_elapsed)
            self.stats['episode_timestep'][global_episode_idx].add_(total_episode_steps)
            self.stats['episode_reward'][global_episode_idx].add_(total_episode_rewards)
            self.stats['episode_exploration'][global_episode_idx].add_(total_episode_exploration / total_episode_steps)
            self.stats['evaluation_scores'][global_episode_idx].add_(evaluation_score)

            mean_10_reward = self.stats[
                                 'episode_reward'][:global_episode_idx + 1][-10:].mean().item()
            mean_100_reward = self.stats[
                                  'episode_reward'][:global_episode_idx + 1][-100:].mean().item()
            mean_100_eval_score = self.stats[
                                      'evaluation_scores'][:global_episode_idx + 1][-100:].mean().item()
            mean_100_exp_rat = self.stats[
                                   'episode_exploration'][:global_episode_idx + 1][-100:].mean().item()
            std_10_reward = self.stats[
                                'episode_reward'][:global_episode_idx + 1][-10:].std().item()
            std_100_reward = self.stats[
                                 'episode_reward'][:global_episode_idx + 1][-100:].std().item()
            std_100_eval_score = self.stats[
                                     'evaluation_scores'][:global_episode_idx + 1][-100:].std().item()
            std_100_exp_rat = self.stats[
                                  'episode_exploration'][:global_episode_idx + 1][-100:].std().item()
            if std_10_reward != std_10_reward: std_10_reward = 0
            if std_100_reward != std_100_reward: std_100_reward = 0
            if std_100_eval_score != std_100_eval_score: std_100_eval_score = 0
            if std_100_exp_rat != std_100_exp_rat: std_100_exp_rat = 0
            global_n_steps = self.stats[
                                 'episode_timestep'][:global_episode_idx + 1].sum().item()
            global_training_elapsed = self.stats[
                                          'episode_elapsed'][:global_episode_idx + 1].sum().item()
            wallclock_elapsed = time.time() - self.training_start

            self.stats['result'][global_episode_idx][0].add_(global_n_steps)
            self.stats['result'][global_episode_idx][1].add_(mean_100_reward)
            self.stats['result'][global_episode_idx][2].add_(mean_100_eval_score)
            self.stats['result'][global_episode_idx][3].add_(global_training_elapsed)
            self.stats['result'][global_episode_idx][4].add_(wallclock_elapsed)

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, global_episode_idx, global_n_steps, mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)

            if rank == 0:
                print(debug_message, end='\r', flush=True)
                if time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS:
                    print(ERASE_LINE + debug_message, flush=True)
                    last_debug_time = time.time()

            with self.get_out_lock:
                potential_next_global_episode_idx = self.stats['episode'].item()
                self.reached_goal_mean_reward.add_(
                    mean_100_eval_score >= self.goal_mean_100_reward)
                self.reached_max_minutes.add_(
                    time.time() - self.training_start >= self.max_minutes * 60)
                self.reached_max_episodes.add_(
                    potential_next_global_episode_idx >= self.max_episodes)
                if self.reached_max_episodes or \
                        self.reached_max_minutes or \
                        self.reached_goal_mean_reward:
                    self.get_out_signal.add_(1)
                    break
                # else go work on another episode
                global_episode_idx = self.stats['episode'].add_(1).item() - 1

        while rank == 0 and self.stats['n_active_workers'].item() > 1:
            pass

        if rank == 0:
            print(ERASE_LINE + debug_message)
            if self.reached_max_minutes: print(u'--> reached_max_minutes \u2715')
            if self.reached_max_episodes: print(u'--> reached_max_episodes \u2715')
            if self.reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')

        env.close();
        del env
        self.stats['n_active_workers'].sub_(1)

    def train(self, make_env_fn, make_env_kargs, seed, gamma,
              max_minutes, max_episodes, goal_mean_100_reward):

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        self.max_minutes = max_minutes
        self.max_episodes = max_episodes
        self.goal_mean_100_reward = goal_mean_100_reward

        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        nS, nA = env.observation_space.shape[0], env.action_space.n
        torch.manual_seed(self.seed);
        np.random.seed(self.seed);
        random.seed(self.seed)

        self.stats = {}
        self.stats['episode'] = torch.zeros(1, dtype=torch.int).share_memory_()
        self.stats['result'] = torch.zeros([max_episodes, 5]).share_memory_()
        self.stats['evaluation_scores'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_reward'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_timestep'] = torch.zeros([max_episodes], dtype=torch.int).share_memory_()
        self.stats['episode_exploration'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_elapsed'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['n_active_workers'] = torch.zeros(1, dtype=torch.int).share_memory_()

        self.shared_policy_model = self.policy_model_fn(nS, nA).share_memory()
        self.shared_policy_optimizer = self.policy_optimizer_fn(self.shared_policy_model,
                                                                self.policy_optimizer_lr)
        self.shared_value_model = self.value_model_fn(nS).share_memory()
        self.shared_value_optimizer = self.value_optimizer_fn(self.shared_value_model,
                                                              self.value_optimizer_lr)

        self.get_out_lock = mp.Lock()
        self.get_out_signal = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_max_minutes = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_max_episodes = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_goal_mean_reward = torch.zeros(1, dtype=torch.int).share_memory_()
        self.training_start = time.time()
        workers = [mp.Process(target=self.work, args=(rank,)) for rank in range(self.n_workers)]
        [w.start() for w in workers];
        [w.join() for w in workers]
        wallclock_time = time.time() - self.training_start

        final_eval_score, score_std = self.evaluate(self.shared_policy_model, env, n_episodes=100)
        env.close();
        del env

        final_episode = self.stats['episode'].item()
        training_time = self.stats['episode_elapsed'][:final_episode + 1].sum().item()

        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
            final_eval_score, score_std, training_time, wallclock_time))

        self.stats['result'] = self.stats['result'].numpy()
        self.stats['result'][final_episode:, ...] = np.nan
        self.get_cleaned_checkpoints()
        return self.stats['result'], final_eval_score, training_time, wallclock_time


class FCAC(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(FCAC, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.value_output_layer = nn.Linear(hidden_dims[-1], 1)
        self.policy_output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.policy_output_layer(x), self.value_output_layer(x)

    def full_pass(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logpa = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        action = action.item() if len(action) == 1 else action.data.numpy()
        is_exploratory = action != np.argmax(logits.detach().numpy(), axis=int(len(state) != 1))
        return action, is_exploratory, logpa, entropy, value

    def select_action(self, state):
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action = action.item() if len(action) == 1 else action.data.numpy()
        return action

    def select_greedy_action(self, state):
        logits, _ = self.forward(state)
        return np.argmax(logits.detach().numpy())

    def evaluate_state(self, state):
        _, value = self.forward(state)
        return value


class MultiprocessEnv(object):
    def __init__(self, make_env_fn, make_env_kargs, seed, n_workers):
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.n_workers = n_workers
        self.pipes = [mp.Pipe() for rank in range(self.n_workers)]
        self.workers = [
            mp.Process(
                target=self.work,
                args=(rank, self.pipes[rank][1])) for rank in range(self.n_workers)]
        [w.start() for w in self.workers]
        self.dones = {rank: False for rank in range(self.n_workers)}

    def reset(self, rank=None, **kwargs):
        if rank is not None:
            parent_end, _ = self.pipes[rank]
            self.send_msg(('reset', {}), rank)
            o = parent_end.recv()
            return o

        self.broadcast_msg(('reset', kwargs))
        return np.vstack([parent_end.recv() for parent_end, _ in self.pipes])

    def step(self, actions):
        assert len(actions) == self.n_workers
        [self.send_msg(
            ('step', {'action': actions[rank]}),
            rank) for rank in range(self.n_workers)]
        results = []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            o, r, d, i = parent_end.recv()
            results.append((o,
                            np.array(r, dtype=np.float),
                            np.array(d, dtype=np.float),
                            i))
        return [np.vstack(block) for block in np.array(results).T]

    def close(self, **kwargs):
        self.broadcast_msg(('close', kwargs))
        [w.join() for w in self.workers]

    def _past_limit(self, **kwargs):
        self.broadcast_msg(('_past_limit', kwargs))
        return np.vstack([parent_end.recv() for parent_end, _ in self.pipes])

    def work(self, rank, worker_end):
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed + rank)
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == 'reset':
                worker_end.send(env.reset(**kwargs))
            elif cmd == 'step':
                worker_end.send(env.step(**kwargs))
            elif cmd == '_past_limit':
                worker_end.send(env._elapsed_steps >= env._max_episode_steps)
            else:
                # including close command
                env.close(**kwargs);
                del env;
                worker_end.close()
                break

    def send_msg(self, msg, rank):
        parent_end, _ = self.pipes[rank]
        parent_end.send(msg)

    def broadcast_msg(self, msg):
        [parent_end.send(msg) for parent_end, _ in self.pipes]


class A2C():
    def __init__(self,
                 ac_model_fn,
                 ac_model_max_grad_norm,
                 ac_optimizer_fn,
                 ac_optimizer_lr,
                 policy_loss_weight,
                 value_loss_weight,
                 entropy_loss_weight,
                 max_n_steps,
                 n_workers,
                 tau):
        assert n_workers > 1
        self.ac_model_fn = ac_model_fn
        self.ac_model_max_grad_norm = ac_model_max_grad_norm
        self.ac_optimizer_fn = ac_optimizer_fn
        self.ac_optimizer_lr = ac_optimizer_lr

        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

        self.max_n_steps = max_n_steps
        self.n_workers = n_workers
        self.tau = tau

    def optimize_model(self):
        logpas = torch.stack(self.logpas).squeeze()
        entropies = torch.stack(self.entropies).squeeze()
        values = torch.stack(self.values).squeeze()

        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        rewards = np.array(self.rewards).squeeze()
        returns = np.array([[np.sum(discounts[:T - t] * rewards[t:, w]) for t in range(T)]
                            for w in range(self.n_workers)])

        np_values = values.data.numpy()
        tau_discounts = np.logspace(0, T - 1, num=T - 1, base=self.gamma * self.tau, endpoint=False)
        advs = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]
        gaes = np.array([[np.sum(tau_discounts[:T - 1 - t] * advs[t:, w]) for t in range(T - 1)]
                         for w in range(self.n_workers)])
        discounted_gaes = discounts[:-1] * gaes

        values = values[:-1, ...].view(-1).unsqueeze(1)
        logpas = logpas.view(-1).unsqueeze(1)
        entropies = entropies.view(-1).unsqueeze(1)
        returns = torch.FloatTensor(returns.T[:-1]).view(-1).unsqueeze(1)
        discounted_gaes = torch.FloatTensor(discounted_gaes.T).reshape(-1).unsqueeze(1)

        T -= 1
        T *= self.n_workers
        assert returns.size() == (T, 1)
        assert values.size() == (T, 1)
        assert logpas.size() == (T, 1)
        assert entropies.size() == (T, 1)

        value_error = returns.detach() - values
        value_loss = value_error.pow(2).mul(0.5).mean()
        policy_loss = -(discounted_gaes.detach() * logpas).mean()
        entropy_loss = -entropies.mean()
        loss = self.policy_loss_weight * policy_loss + \
               self.value_loss_weight * value_loss + \
               self.entropy_loss_weight * entropy_loss

        self.ac_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(),
                                       self.ac_model_max_grad_norm)
        self.ac_optimizer.step()

    def interaction_step(self, states, envs):
        actions, is_exploratory, logpas, entropies, values = self.ac_model.full_pass(states)
        new_states, rewards, is_terminals, _ = envs.step(actions)

        self.logpas.append(logpas);
        self.entropies.append(entropies)
        self.rewards.append(rewards);
        self.values.append(values)

        self.running_reward += rewards
        self.running_timestep += 1
        self.running_exploration += is_exploratory[:, np.newaxis].astype(np.int)

        return new_states, is_terminals

    def train(self, make_envs_fn, make_env_fn, make_env_kargs, seed, gamma,
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_envs_fn = make_envs_fn
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma

        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        envs = self.make_envs_fn(make_env_fn, make_env_kargs, self.seed, self.n_workers)
        torch.manual_seed(self.seed);
        np.random.seed(self.seed);
        random.seed(self.seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.running_timestep = np.array([[0.], ] * self.n_workers)
        self.running_reward = np.array([[0.], ] * self.n_workers)
        self.running_exploration = np.array([[0.], ] * self.n_workers)
        self.running_seconds = np.array([[time.time()], ] * self.n_workers)
        self.episode_timestep, self.episode_reward = [], []
        self.episode_seconds, self.evaluation_scores = [], []
        self.episode_exploration = []

        self.ac_model = self.ac_model_fn(nS, nA)
        self.ac_optimizer = self.ac_optimizer_fn(self.ac_model,
                                                 self.ac_optimizer_lr)

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        states = envs.reset()

        # collect n_steps rollout
        episode, n_steps_start = 0, 0
        self.logpas, self.entropies, self.rewards, self.values = [], [], [], []
        for step in count(start=1):
            states, is_terminals = self.interaction_step(states, envs)

            if is_terminals.sum() or step - n_steps_start == self.max_n_steps:
                past_limits_enforced = envs._past_limit()
                is_failure = np.logical_and(is_terminals, np.logical_not(past_limits_enforced))
                next_values = self.ac_model.evaluate_state(
                    states).detach().numpy() * (1 - is_failure)
                self.rewards.append(next_values);
                self.values.append(torch.Tensor(next_values))
                self.optimize_model()
                self.logpas, self.entropies, self.rewards, self.values = [], [], [], []
                n_steps_start = step

            # stats
            if is_terminals.sum():
                episode_done = time.time()
                evaluation_score, _ = self.evaluate(self.ac_model, env)
                self.save_checkpoint(episode, self.ac_model)

                for i in range(self.n_workers):
                    if is_terminals[i]:
                        states[i] = envs.reset(rank=i)
                        self.episode_timestep.append(self.running_timestep[i][0])
                        self.episode_reward.append(self.running_reward[i][0])
                        self.episode_exploration.append(self.running_exploration[i][0] / self.running_timestep[i][0])
                        self.episode_seconds.append(episode_done - self.running_seconds[i][0])
                        training_time += self.episode_seconds[-1]
                        self.evaluation_scores.append(evaluation_score)
                        episode += 1

                        mean_10_reward = np.mean(self.episode_reward[-10:])
                        std_10_reward = np.std(self.episode_reward[-10:])
                        mean_100_reward = np.mean(self.episode_reward[-100:])
                        std_100_reward = np.std(self.episode_reward[-100:])
                        mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
                        std_100_eval_score = np.std(self.evaluation_scores[-100:])
                        mean_100_exp_rat = np.mean(self.episode_exploration[-100:])
                        std_100_exp_rat = np.std(self.episode_exploration[-100:])

                        total_step = int(np.sum(self.episode_timestep))
                        wallclock_elapsed = time.time() - training_start
                        result[episode - 1] = total_step, mean_100_reward, \
                                              mean_100_eval_score, training_time, wallclock_elapsed

                # debug stuff
                reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
                reached_max_minutes = wallclock_elapsed >= max_minutes * 60
                reached_max_episodes = episode + self.n_workers >= max_episodes
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

                # reset running variables for next time around
                self.running_timestep *= 1 - is_terminals
                self.running_reward *= 1 - is_terminals
                self.running_exploration *= 1 - is_terminals
                self.running_seconds[is_terminals.astype(np.bool)] = time.time()

        final_eval_score, score_std = self.evaluate(self.ac_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
            final_eval_score, score_std, training_time, wallclock_time))
        env.close();
        del env
        envs.close();
        del envs
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1, greedy=True):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                if greedy:
                    a = eval_policy_model.select_greedy_action(s)
                else:
                    a = eval_policy_model.select_action(s)
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
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.ac_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.ac_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos,
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def demo_progression(self, title='{} Agent progression', max_n_videos=5):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.ac_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.ac_model, env, n_episodes=1)

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
