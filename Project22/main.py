import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
from time import time
from env import FootballEnv

# Инициализация среды
env = FootballEnv()
state_dim = len(env.reset())  
num_actions = env.action_space.n

# Гиперпараметры
gamma = 0.99
epsilon = 0.33
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
episodes = 3_000
target_update_freq = 7

# Пути для сохранения моделей
model_dir = 'saved_model'
os.makedirs(model_dir, exist_ok=True)
team1_models = ['a', 'c', 'f']
team2_models = ['b', 'd', 'e']
model_paths = {name: os.path.join(model_dir, f'football_dqn_weights_{name}.h5') for name in team1_models + team2_models}

class DQN(tf.keras.Model):
    def __init__(self, num_actions, state_dim):
        super(DQN, self).__init__()
        
        self.input_norm = layers.BatchNormalization()
        
        
        self.dense1 = layers.Dense(512, kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        
        self.dense2 = layers.Dense(256, kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)
        
        self.dense3 = layers.Dense(128, kernel_initializer='he_normal')
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.2)
        
        
        self.out = layers.Dense(num_actions, activation='linear',
                              kernel_initializer='glorot_uniform')

    def call(self, inputs, training=False):
        
        x = self.input_norm(inputs)
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.dropout3(x, training=training)
        
        return self.out(x)  

class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
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

# Инициализация моделей и буферов
models = {}
target_models = {}
optimizers = {}
buffers = {}
dummy_input = tf.expand_dims(tf.zeros(state_dim), 0)
for name in team1_models + team2_models:
    models[name] = DQN(num_actions, state_dim)
    target_models[name] = DQN(num_actions, state_dim)
    optimizers[name] = tf.keras.optimizers.Adam(learning_rate=0.001)
    buffers[name] = ReplayBuffer()
    
    _ = models[name](dummy_input)
    _ = target_models[name](dummy_input)

    # Загрузка весов если существуют
    if os.path.exists(model_paths[name]):
        models[name].load_weights(model_paths[name])
        target_models[name].set_weights(models[name].get_weights())
        print(f"Модель {name.upper()} загружена из {model_paths[name]}")

def select_action(state, epsilon, model):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    q_values = model.predict(np.array([state]), verbose=0)
    return np.argmax(q_values[0])

def train_step(model, target_model, optimizer, states, actions, rewards, next_states, dones):
    next_q_values = target_model.predict(next_states, verbose=0)
    target_q = rewards + gamma * np.max(next_q_values, axis=1) * (1 - dones.astype(float))
    
    with tf.GradientTape() as tape:
        current_q = tf.reduce_sum(model(states) * tf.one_hot(actions, num_actions), axis=1)
        loss = tf.keras.losses.MSE(target_q, current_q)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def train():
    global epsilon
    time_e = time()
    
    for episode in range(episodes):
        state = env.reset()
        rewards = {name: 0 for name in team1_models + team2_models}
        terminated = False
        
        while not terminated:
            # Действия для всех игроков
            actions = {}
            next_states = {}
            dones = {}
            
            # Команда 1 (игроки a, c, f)
            for name in team1_models:
                player_id = name
                state_player = env._get_obs(player_name=player_id)
                actions[name] = select_action(state_player, epsilon, models[name])
                next_state, reward, done, _ = env.step(actions[name], player_id)
                
                buffers[name].add(state_player, actions[name], reward, next_state, done)
                next_states[name] = next_state
                dones[name] = done
                rewards[name] += reward
            
            # Команда 2 (игроки b, d, e)
            for name in team2_models:
                player_id = name
                state_player = env._get_obs(player_name=player_id)
                actions[name] = select_action(state_player, epsilon, models[name])
                next_state, reward, done, _ = env.step(actions[name], player_id)
                
                buffers[name].add(state_player, actions[name], reward, next_state, done)
                next_states[name] = next_state
                dones[name] = done
                rewards[name] += reward
            
            # Обучение всех моделей
            for name in team1_models + team2_models:
                batch = buffers[name].sample(batch_size)
                if batch:
                    train_step(models[name], target_models[name], optimizers[name], *batch)
            
            terminated = any(dones.values())
            env.render()
        
        # Обновление целевых сетей
        if episode % target_update_freq == 0:
            for name in team1_models + team2_models:
                target_models[name].set_weights(models[name].get_weights())
                models[name].save_weights(model_paths[name])
        
        # Уменьшение epsilon и логирование
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Эпизод: {episode}, Epsilon: {epsilon:.3f}, Time: {time()-time_e:.1f}с")
        print(f"Команда 1: A={rewards['a']:.1f}, C={rewards['c']:.1f}, F={rewards['f']:.1f}")
        print(f"Команда 2: B={rewards['b']:.1f}, D={rewards['d']:.1f}, E={rewards['e']:.1f}")
        time_e = time()

try:
    print('Старт обучения...')
    train()
except KeyboardInterrupt:
    print('Сохранение моделей перед выходом...')
    for name in team1_models + team2_models:
        models[name].save_weights(model_paths[name])