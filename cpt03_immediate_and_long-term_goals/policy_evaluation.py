import numpy as np


def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P))

    while True:
        V = np.zeros(len(P))

        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))

        if np.max(np.abs(prev_V - V)) < theta:
            break

        prev_V = V.copy()

    return V
