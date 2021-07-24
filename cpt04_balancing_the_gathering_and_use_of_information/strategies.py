import numpy as np
from tqdm import tqdm_notebook as tqdm


def pure_exploitation(env, n_episodes=5000):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    name = 'Pure exploitation'
    for e in tqdm(range(n_episodes), desc='Episodes for: ' + name, leave=False):
        action = np.argmax(Q)
        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action

    return name, returns, Qe, actions


def pure_exploration(env, n_episodes=5000):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    name = 'Pure exploration'
    for e in tqdm(range(n_episodes), desc='Episodes for: ' + name, leave=False):
        action = np.random.randint(len(Q))
        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action

    return name, returns, Qe, actions


def epsilon_greedy(env, epsilon=0.01, n_episodes=5000):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    name = 'E-greedy {}'.format(epsilon)
    for e in tqdm(range(n_episodes), desc='Episodes for: ' + name, leave=False):
        if np.random.random() > epsilon:
            action = np.argmax(Q)
        else:
            action = np.random.randint(len(Q))
        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action

    return name, returns, Qe, actions


def lin_dec_epsilon_greedy(env,
                           init_epsilon=0.01,
                           min_epsilon=0.01,
                           decay_ratio=0.05,
                           n_episodes=5000):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    name = 'Linear e-greedy {} {} {}'.format(init_epsilon, min_epsilon, decay_ratio)
    for e in tqdm(range(n_episodes), desc='Episodes for: ' + name, leave=False):
        decay_episode = n_episodes * decay_ratio

        epsilon = 1 - e / decay_episode
        epsilon *= init_epsilon - min_epsilon
        epsilon += min_epsilon
        epsilon = np.clip(epsilon, min_epsilon, init_epsilon)

        if np.random.random() > epsilon:
            action = np.argmax(Q)
        else:
            action = np.random.randint(len(Q))
        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action

    return name, returns, Qe, actions


def exp_dec_epsilon_greedy(env,
                           init_epsilon=0.01,
                           min_epsilon=0.01,
                           decay_ratio=0.1,
                           n_episodes=5000):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    decay_episodes = int(n_episodes * decay_ratio)
    rem_episodes = n_episodes - decay_episodes
    epsilons = 0.01
    epsilons /= np.logspace(-2, 0, decay_episodes)
    epsilons *= init_epsilon - min_epsilon
    epsilons += min_epsilon
    epsilons = np.pad(epsilons, (0, rem_episodes), 'edge')

    name = 'Exp e-greedy {} {} {}'.format(init_epsilon, min_epsilon, decay_ratio)
    for e in tqdm(range(n_episodes), desc='Episodes for: ' + name, leave=False):
        if np.random.random() > epsilons[e] :
            action = np.argmax(Q)
        else:
            action = np.random.randint(len(Q))
        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action

    return name, returns, Qe, actions


def optimistic_initialization(env,
                              optimistic_estimate=1.0,
                              initial_count=100,
                              n_episodes=5000):
    Q = np.full((env.action_space.n), optimistic_estimate, dtype=np.float64)
    N = np.full((env.action_space.n), initial_count, dtype=np.float64)

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    name = 'Optimistic {} {}'.format(optimistic_estimate, initial_count)
    for e in tqdm(range(n_episodes), desc='Episodes for: ' + name, leave=False):
        action = np.argmax(Q)
        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action

    return name, returns, Qe, actions


def softmax(env,
           init_temp=0.01,
           min_temp=0.01,
           decay_ratio=0.04,
           n_episodes=5000):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    name = 'SoftMax {} {} {}'.format(init_temp, min_temp, decay_ratio)
    for e in tqdm(range(n_episodes), desc='Episodes for: ' + name, leave=False):
        decay_episodes = n_episodes * decay_ratio
        temp = 1 - e / decay_episodes
        temp *= init_temp - min_temp
        temp += min_temp
        temp = np.clip(temp, min_temp, init_temp)

        scaled_Q = Q / temp
        norm_Q = scaled_Q - np.max(scaled_Q)
        exp_Q = np.exp(norm_Q)
        probs = exp_Q / np.sum(exp_Q)

        assert np.isclose(probs.sum(), 1.0)

        action = np.random.choice(np.arange(len(probs)), size=1, p=probs)[0]

        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action

    return name, returns, Qe, actions


def upper_confidence_bound(env,
                           c=2,
                           n_episodes=5000):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    name = 'UCB {}'.format(c)
    for e in tqdm(range(n_episodes), desc='Episodes for: ' + name, leave=False):
        if e < len(Q):
            action = e
        else:
            U = np.sqrt(c * np.log(e) / N)
            action = np.argmax(Q + U)

        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action

    return name, returns, Qe, actions


def thompson_sampling(env,
                      alpha=1,
                      beta=0,
                      n_episodes=5000):
    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    name = 'Thompson Sampling {} {}'.format(alpha, beta)
    for e in tqdm(range(n_episodes), desc='Episodes for: ' + name, leave=False):
        samples = np.random.normal(loc=Q, scale=alpha/(np.sqrt(N) + beta))
        action = np.argmax(samples)

        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action

    return name, returns, Qe, actions