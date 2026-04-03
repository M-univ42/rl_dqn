from environment import CartPoleEnv
from dqn import MLP_DQN

def train(n_steps=1000000, lr=0.0001, epsilon_max=1.0, epsilon_min=0.01, gamma=0.99,
          replay_buffer_size=-1, batch_size=-1, seed=42,
          update_freq=1, network_size="medium", target_update_freq=-1):
    """Trains DQN agent.

    update_freq: number of env steps between gradient updates (update-to-data ratio).
                 1 = update every step, 4 = update every 4 steps, etc.
    network_size: "small" | "medium" | "large"
    target_update_freq: number of env steps between target network syncs.
                        -1 disables the target network.
    """

    env = CartPoleEnv(render_mode="rgb_array", verbose=True, seed=seed)

    use_target_network = target_update_freq > 0
    dqn_agent = MLP_DQN(lr=lr, epsilon_max=epsilon_max, epsilon_min=epsilon_min,
                        replay_buffer_size=replay_buffer_size, batch_size=batch_size,
                        network_size=network_size, target_network=use_target_network)

    state = env.reset()

    ep_reward = 0
    ep_rewards = []
    ep_steps = []  # cumulative env step at which each episode ended
    i = 0

    epsilon = epsilon_max
    while i < n_steps:

        # epsilon = i*(epsilon_max - epsilon_min)/n_steps if epsilon > epsilon_min else epsilon_min
        epsilon = max(epsilon_min, epsilon_max - (epsilon_max - epsilon_min) * i / n_steps)

        
        a = dqn_agent.select_action(state, policy="e-greedy", epsilon=epsilon)
        next_state, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        ep_reward += reward
        i += 1

        dqn_agent.add_to_replay_buffer(state, a, reward, next_state, done)
        if use_target_network and i % target_update_freq == 0:
            dqn_agent.sync_target_network()

        if i % update_freq == 0:
            replay_batch = dqn_agent.sample_replay_buffer()
            if replay_batch is not None:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_batch
                dqn_agent.update(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, gamma=gamma)
            else:
                # wrap in lists so update() can handle them as a batch
                dqn_agent.update([state], [a], [reward], [next_state], [done], gamma=gamma)

        if done:
            state = env.reset()
            ep_rewards.append(ep_reward)
            ep_steps.append(i)
            ep_reward = 0
            avg = sum(ep_rewards[-10:]) / min(len(ep_rewards), 10)
            print(f"\rstep {i} out of {n_steps} | episode no. {len(ep_rewards)} | "
                  f"last return {ep_rewards[-1]:.0f} | avg(10) {avg:.1f} | "
                  f"epsilon {epsilon:.3f}   ", end="", flush=True)
        else:
            state = next_state

    return ep_rewards, ep_steps

if __name__ == "__main__":
    print(train())