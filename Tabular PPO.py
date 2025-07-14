import numpy as np
import pandas as pd

# Simple online PPO example: 2 states (Joy, Sad), 2 actions (Happy, Sad)
states = ["Joy", "Sad"]
actions = ["Happy", "Sad"]
num_states = len(states)
num_actions = len(actions)

# Initialize policy logits (state x action)
theta = np.zeros((num_states, num_actions))
epsilon = 0.2       # PPO clip parameter
alpha = 0.9         # learning rate
batch_size = 2      # update after 2 samples
total_steps = 10    # total interactions

# Softmax helper
def softmax(logits):
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()

# Buffer for storing (state_idx, action_idx, reward, logprob_old)
buffer = []

# Store the old theta used for sampling
theta_old = np.copy(theta)

s = 0

for step in range(1, total_steps + 1):
    # Cycle through states: Joy for odd steps, Sad for even
    # s = 0 if step % 2 == 1 else 1
    probs = softmax(theta_old[s])  # Use old theta to sample
    print(f"Theta (current):\n{theta}\n")
    a = np.random.choice(num_actions, p=probs)

    # Display state and action
    print(f"Step {step}: State = {states[s]}, Action = {actions[a]}")

    # Get user rating
    while True:
        try:
            r = float(input("Rate this from 1 to 5: "))
            if 1 <= r <= 5:
                break
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Enter a number between 1 and 5.")

    r_norm = (r - 1) / 4  # normalize to [0,1]
    logp_old = np.log(probs[a])  # from old theta
    buffer.append((s, a, r_norm, logp_old))
    s += 1


    # Perform PPO update every batch_size samples
    if len(buffer) == batch_size:
        s_batch, a_batch, r_batch, logp_old_batch = zip(*buffer)
        r_batch = np.array(r_batch)

        # Compute baseline and advantages
        baseline = r_batch.mean()
        advantages = r_batch - baseline

        print("\n--- PPO Update ---")
        grads = np.zeros_like(theta)
        for idx, ((s_i, a_i, r_i, logp_old_i), A_i) in enumerate(zip(buffer, advantages)):
            probs_current = softmax(theta[s_i])  # current theta
            logp_current = np.log(probs_current[a_i])
            ratio = np.exp(logp_current - logp_old_i)
            clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
            weight = clipped_ratio if (clipped_ratio * A_i) < (ratio * A_i) else ratio  ## The value can also be minus, that is why i didnt use min

            # grad log pi
            grad_log = -probs_current
            grad_log[a_i] += 1
            grads[s_i] += A_i * grad_log * weight

            print(f"Sample {idx+1}: State={states[s_i]}, Action={actions[a_i]}, "
                  f"Reward={r_i:.2f}, Old LogP={logp_old_i:.3f}, "
                  f"New LogP={logp_current:.3f}, Ratio={ratio:.3f}, Clipped={clipped_ratio:.3f}")


        # Update theta_old for next batch
        theta_old = np.copy(theta)
        buffer.clear()
        s = 0
    
        # Average and update
        grads /= batch_size
        theta += alpha * grads
        print(f"Updated Theta:\n{theta}\n------------------\n")

        

# After all updates, print final policy
policy = np.array([softmax(theta[s]) for s in range(num_states)])
df = pd.DataFrame(policy, columns=actions, index=states)
print("Final Policy Probabilities:")
print(df)
