import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from time import time
from env import FootballEnv
import psutil
import os
import gc

tf.config.threading.set_intra_op_parallelism_threads(10)


env = FootballEnv()
state_dim = len(env.reset())  
num_actions = env.action_space.n


gamma = 0.99
epsilon = 0.3
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
episodes = 3_000 #216
target_update_freq = 3




model_dir = 'saved_model'

weights_path_a = os.path.join(model_dir, "football_dqn_a.weights.h5")
weights_path_b = os.path.join(model_dir, "football_dqn_b.weights.h5")
buffer_path_a = os.path.join(model_dir,"replay_buffer_a.npz")
buffer_path_b = os.path.join(model_dir,"replay_buffer_b.npz")


os.makedirs(model_dir, exist_ok=True)

class DQN(tf.keras.Model):
    def __init__(self, num_actions, state_dim):
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(256, activation='relu', input_shape=(state_dim,))
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(128, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.dense3 = layers.Dense(64, activation='relu')
        self.out = layers.Dense(4, activation='linear') 
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dense3(x)
        return self.out(x)
    


class ReplayBuffer:
    def __init__(self, max_size=70_000):
        self.buffer = []
        self.max_size = max_size
    
    def add(self, state, action, reward, next_state, terminated):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, terminated))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in idx]
        
        states = np.vstack([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.vstack([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])
        
        return states, actions, rewards, next_states, dones


def save_buffer(buffer, path):
    np.savez(
        path,
        states=np.vstack([s[0] for s in buffer.buffer]),
        actions=np.array([s[1] for s in buffer.buffer]),
        rewards=np.array([s[2] for s in buffer.buffer]),
        next_states=np.vstack([s[3] for s in buffer.buffer]),
        dones=np.array([s[4] for s in buffer.buffer]),
    )

def load_buffer(buffer, path):
    data = np.load(path)
    buffer.buffer = list(zip(
        data["states"],
        data["actions"],
        data["rewards"],
        data["next_states"],
        data["dones"],
    ))



model_a = DQN(num_actions, state_dim)
target_model_a = DQN(num_actions, state_dim)
optimizer_a = tf.keras.optimizers.Adam(learning_rate=0.001)

model_b = DQN(num_actions, state_dim)
target_model_b = DQN(num_actions, state_dim)
optimizer_b = tf.keras.optimizers.Adam(learning_rate=0.001)

dummy_input = tf.expand_dims(tf.zeros(state_dim), 0)
_ = model_a(dummy_input)
_ = target_model_a(dummy_input)
_ = model_b(dummy_input)
_ = target_model_b(dummy_input)




if os.path.exists(weights_path_a):
    model_a.load_weights(weights_path_a)
    target_model_a.set_weights(model_a.get_weights())
    print("Модель A загружена из", weights_path_a)

if os.path.exists(weights_path_b):
    model_b.load_weights(weights_path_b)
    target_model_b.set_weights(model_b.get_weights())
    print("Модель B загружена из", weights_path_b)



buffer_a = ReplayBuffer()
buffer_b = ReplayBuffer()



if os.path.exists(buffer_path_a):
    load_buffer(buffer_a,buffer_path_a)

if os.path.exists(buffer_path_b):
    load_buffer(buffer_b,buffer_path_b)



def select_action_a(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    q_values = model_a.predict(np.array([state]), verbose=0)
    return np.argmax(q_values[0])

def select_action_b(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    q_values = model_b.predict(np.array([state]), verbose=0)
    return np.argmax(q_values[0])




def train():
    global gamma,epsilon,epsilon_min,epsilon_decay ,batch_size,episodes,target_update_freq 
    for episode in range(episodes):
        state = env.reset()
        total_reward_a = 0
        total_reward_b = 0
        terminated_a = False
        terminated_b = False

        time_e = time()
        
        while not (terminated_a or terminated_b):
    #------------------------A--------------------------------
            state_a = env._get_obs(player=1)
            action_a = select_action_a(state_a, epsilon)
            next_state_a, reward_a, terminated_a, info_a = env.step(action_a,1)
                
            

    #-------------------------B----------------------------------------
            state_b = env._get_obs(player=0)
            action_b = select_action_b(state_b, epsilon)
            next_state_b, reward_b, terminated_b, info_b = env.step(action_b,0)



            if info_a['win'] == 'a' or info_b['win'] == 'a':
                reward_b+=-20
                reward_a+=100
                
            if info_a['win'] == 'b' or info_b['win'] == 'b':
                reward_a+=-20
                reward_b+=100

            reward_a /= 20
            reward_b/=20

            buffer_a.add(state_a, action_a, reward_a, next_state_a, terminated_a)
            state_a = next_state_a
            total_reward_a += reward_a
                
                
            batch = buffer_a.sample(batch_size)
            if batch is not None:
                states, actions, rewards, next_states, dones = batch
                    
                # Целевые Q-значения
                next_q_values = target_model_a.predict(next_states, verbose=0)
                target_q = rewards + gamma * np.max(next_q_values, axis=1) * (1 - dones.astype(float))
                    
                # Обновление модели
                with tf.GradientTape() as tape:
                    current_q = tf.reduce_sum(model_a(states) * tf.one_hot(actions, num_actions), axis=1)
                    loss = tf.keras.losses.MSE(target_q, current_q)
                    
                grads = tape.gradient(loss, model_a.trainable_variables)
                optimizer_a.apply_gradients(zip(grads, model_a.trainable_variables))






                
            buffer_b.add(state_b, action_b, reward_b, next_state_b, terminated_b)
            state_b = next_state_b
            total_reward_b += reward_b





                
                
            batch = buffer_b.sample(batch_size)
            if batch is not None:
                states, actions, rewards, next_states, dones = batch
                    
                # Целевые Q-значения
                next_q_values = target_model_b.predict(next_states, verbose=0)
                target_q = rewards + gamma * np.max(next_q_values, axis=1) * (1 - dones.astype(float))
                    
                # Обновление модели
                with tf.GradientTape() as tape:
                    current_q = tf.reduce_sum(model_b(states) * tf.one_hot(actions, num_actions), axis=1)
                    loss = tf.keras.losses.MSE(target_q, current_q)
                    
                grads = tape.gradient(loss, model_b.trainable_variables)
                optimizer_b.apply_gradients(zip(grads, model_b.trainable_variables))

    #---------------------------------------------------------------------
            
            env.render()

        
        if episode % target_update_freq == 0:
            target_model_a.set_weights(model_a.get_weights())
            target_model_b.set_weights(model_b.get_weights())
            target_model_a.save_weights(weights_path_a)
            target_model_b.save_weights(weights_path_b)
            save_buffer(buffer_a,buffer_path_a)
            save_buffer(buffer_b,buffer_path_b)
            tf.keras.backend.clear_session()
            gc.collect()



        
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Эпизод: {episode}, A Награда: {round(total_reward_a,3)},B Награда: {round(total_reward_b,3)}, Epsilon: {epsilon:.3f} , time: {round(time()-time_e,1)} {info_a} {info_b}")
        print(f"Память: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB, буфер: {str(len(buffer_a.buffer)+len(buffer_b.buffer))}/{str(buffer_a.max_size+buffer_b.max_size)}")


try:
    print('start')
    train()
except KeyboardInterrupt:
    print('stop')
    target_model_a.save_weights(weights_path_a)
    target_model_b.save_weights(weights_path_b)
    save_buffer(buffer_a,buffer_path_a)
    save_buffer(buffer_b,buffer_path_b)
