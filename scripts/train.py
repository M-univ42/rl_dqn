from environment import CartPoleEnv
from dqn import MLP_DQN

def train(n_steps = 1000000, lr = 0.0001, epsilon_max = 1.0, epsilon_min=0.01,gamma=0.99):
    """Trains DQN agent"""

    env = CartPoleEnv(render_mode="rgb_array",verbose=True,seed = 42)

    dqn_agent = MLP_DQN(lr=lr,epsilon_max=epsilon_max,epsilon_min=epsilon_min)

    state = env.reset()

    ep_reward = 0
    rewards = []
    i = 0

    epsilon = epsilon_max
    while i <n_steps:

        epsilon = i*(epsilon_max -epsilon_min)/n_steps if epsilon > epsilon_min else epsilon_min

        a     = dqn_agent.select_action(state, policy="e-greedy", epsilon=epsilon)
        next_state, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        ep_reward += reward
        i += 1
        dqn_agent.update(state, a, reward, next_state, done, gamma=gamma)

        if done:
            state = env.reset()
            rewards.append(ep_reward)
            ep_reward = 0
            # log output dynamically for long runs
            avg = sum(rewards[-10:]) / min(len(rewards),10)
            print(f"\rstep {i} out of {n_steps} | episode no. {len(rewards)} |"
                  f"last return {rewards[-1]:.0f}|avg(10) {avg:.1f} | "
                  f"epsilon {epsilon:.3f}   ", end="", flush=True)
        else:
            state = next_state
    return rewards

if __name__ == "__main__":
    print(train())