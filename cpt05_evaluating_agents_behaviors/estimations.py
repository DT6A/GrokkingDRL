from itertools import cycle, count

import numpy as np
from tqdm import tqdm_notebook as tqdm


def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')

    return values


def generate_trajectory(pi, env, max_steps=20):
    done, trajectory = False, []
    while not done:
        state = env.reset()
        for t in count():
            action = pi(state)
            next_state, reward, done, _ = env.step(action)
            experience = (state, action, reward, next_state, done)
            trajectory.append(experience)
            if done:
                break
            if t >= max_steps - 1:
                trajectory = []
                break
            state = next_state
    return np.array(trajectory, np.object)


def mc_prediction(pi, env, gamma=1.0,
                  init_alpha=0.1, min_alpha=0.01, alpha_decay_ratio=0.3,
                  n_episodes=500, max_steps=100, first_visit=True):
    nS = env.observation_space.n
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))

    for e in tqdm(range(n_episodes), leave=False):
        trajectory = generate_trajectory(pi, env, max_steps)

        visited = np.zeros(nS, dtype=np.bool)
        for t, (state, _, reward, _, _) in enumerate(trajectory):
            if visited[state] and first_visit:
                continue
            visited[state] = True

            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            V[state] = V[state] + alphas[e] * (G - V[state])

        V_track[e] = V
    return V.copy(), V_track


def td(pi, env, gamma=1.0,
       init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.3,
       n_episodes=500):
    nS = env.observation_space.n
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, max_steps=n_episodes)

    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        while not done:
            action = pi(state)
            next_state, reward, done, _ = env.step(action)

            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]
            V[state] = V[state] + alphas[e] * td_error

            state = next_state
        V_track[e] = V
    return V, V_track


def ntd(pi, env, gamma=1.0,
       init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.3,
       n_step=3,
       n_episodes=500):
    nS = env.observation_space.n
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, max_steps=n_episodes)

    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))

    discounts = np.logspace(0, n_step + 1, num=n_step + 1, base=gamma, endpoint=False)

    for e in tqdm(range(n_episodes), leave=False):
        state, done, path = env.reset(), False, []
        while not done or path is not None:
            path = path[1:]
            while not done and len(path) < n_step:
                action = pi(state)
                next_state, reward, done, _ = env.step(action)

                experience = (state, reward, next_state, done)
                path.append(experience)
                state = next_state

                if done:
                    break
            n = len(path)
            est_state = path[0][0]

            rewards = np.array(path)[:, 1]
            partial_return = discounts[:n] * rewards
            bs_val = discounts[-1] * V[next_state] * (not done)
            ntd_target = np.sum(np.append(partial_return, bs_val))
            ntd_error = ntd_target - V[est_state]
            V[est_state] = V[est_state] + alphas[e] * ntd_error

            if len(path) == 1 and path[0][3]:
                path = None
        V_track[e] = V
    return V, V_track


def td_lambda(pi, env, gamma=1.0,
              init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.3,
              lambda_=0.3,
              n_episodes=500):
    nS = env.observation_space.n
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, max_steps=n_episodes)

    E = np.zeros(nS)
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))

    for e in tqdm(range(n_episodes), leave=False):
        E.fill(0)

        state, done = env.reset(), False

        while not done:
            action = pi(state)
            next_state, reward, done, _ = env.step(action)

            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]

            E[state] = E[state] + 1
            V = V + alphas[e] * td_error * E
            E = gamma * lambda_ * E

            state = next_state
        V_track[e] = V
    return V, V_track
