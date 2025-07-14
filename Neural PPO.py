import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

# Setup
states = ["Joy", "Sad"]
actions = ["Happy", "Sad"]
num_states = len(states)
num_actions = len(actions)
epsilon = 0.2
alpha = 0.1  # Smaller learning rate for stability
batch_size = 2
total_steps = 10

# One-hot encode discrete state index
def encode_state(idx):
    vec = np.zeros(num_states)
    vec[idx] = 1.0
    return vec

# Policy network: input = one-hot state, output = action logits
class PolicyNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(32, activation='relu')
        self.logits = layers.Dense(num_actions)

    def call(self, x):
        x = self.dense1(x)
        return self.logits(x)  # raw logits

# Initialize networks
policy_net = PolicyNet()
policy_net_old = PolicyNet()

# Build both models (they now get random initial weights)
dummy = tf.zeros((1, num_states), dtype=tf.float32)
_ = policy_net(dummy)
_ = policy_net_old(dummy)

policy_net_old.set_weights(policy_net.get_weights())  # sync at start
optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

# Experience buffer
buffer = []
s = 0  # Start with Joy

for step in range(1, total_steps + 1):
    state_vec = encode_state(s).reshape(1, -1)

    # Sample from old policy
    logits_old = policy_net_old(tf.convert_to_tensor(state_vec, dtype=tf.float32))
    probs_old = tf.nn.softmax(logits_old).numpy().flatten()
    a = np.random.choice(num_actions, p=probs_old)

    print(f"\nStep {step}: State = {states[s]}, Action = {actions[a]}")
    while True:
        try:
            r = float(input("Rate this from 1 to 5: "))
            if 1 <= r <= 5:
                break
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Try again.")

    r_norm = (r - 1) / 4
    buffer.append((s, a, r_norm))  # no need to store logp_old — we'll recompute using policy_net_old

    s += 1  # next state

    if len(buffer) == batch_size:
        s_batch, a_batch, r_batch = zip(*buffer)
        s_encoded = np.array([encode_state(s_i) for s_i in s_batch])
        a_batch = np.array(a_batch)
        r_batch = np.array(r_batch)
        baseline = r_batch.mean()
        advantages = r_batch - baseline

        # PPO update
        with tf.GradientTape() as tape:
            s_tensor = tf.convert_to_tensor(s_encoded, dtype=tf.float32)

            # Log probs from old and new policy
            logits_old = policy_net_old(s_tensor)
            probs_old = tf.nn.softmax(logits_old)
            logp_old = tf.math.log(tf.reduce_sum(probs_old * tf.one_hot(a_batch, num_actions), axis=1))

            logits_new = policy_net(s_tensor)
            probs_new = tf.nn.softmax(logits_new)
            logp_new = tf.math.log(tf.reduce_sum(probs_new * tf.one_hot(a_batch, num_actions), axis=1))

            ratio = tf.exp(logp_new - logp_old)
            clipped = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
            weighted = tf.where((clipped * advantages) < (ratio * advantages), clipped, ratio)
            loss = -tf.reduce_mean(weighted * advantages)

        policy_net_old.set_weights(policy_net.get_weights())
        grads = tape.gradient(loss, policy_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

        # Logging
        print("\n--- PPO Update ---")
        for i in range(batch_size):
            print(f"Sample {i+1}: State={states[s_batch[i]]}, Action={actions[a_batch[i]]}, "
                  f"Reward={r_batch[i]:.2f}, LogP_old={logp_old[i].numpy():.3f}, "
                  f"LogP_new={logp_new[i].numpy():.3f}, Ratio={ratio[i].numpy():.3f}, "
                  f"Clipped={clipped[i].numpy():.3f}")

        # Sync old policy
        buffer.clear()
        s = 0

# Final policy
print("\nFinal Policy Probabilities:")
for i, state in enumerate(states):
    vec = encode_state(i).reshape(1, -1)
    probs = tf.nn.softmax(policy_net(tf.convert_to_tensor(vec, dtype=tf.float32))).numpy().flatten()
    print(f"{state}: Happy = {probs[0]:.3f}, Sad = {probs[1]:.3f}")
