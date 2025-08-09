import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import psutil
print(f"Память: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")
from env import FootballEnv
import os




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
    






model_a = DQN(num_actions, state_dim)
target_model_a = DQN(num_actions, state_dim)
optimizer_a = tf.keras.optimizers.Adam(learning_rate=0.001)



dummy_input = tf.expand_dims(tf.zeros(state_dim), 0)
_ = model_a(dummy_input)
_ = target_model_a(dummy_input)




if os.path.exists(weights_path_a):
    model_a.load_weights(weights_path_a)
    target_model_a.set_weights(model_a.get_weights())
    print("Модель A загружена из", weights_path_a)






def select_action_a(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    q_values = model_a.predict(np.array([state]), verbose=0)
    return np.argmax(q_values[0])





rewards_history_a = []

def train():

    for episode in range(episodes):
        state = env.reset()


        terminated_a = False

        
        while not (terminated_a):
    #------------------------A--------------------------------
            state_a = env._get_obs(player=1)
            action_a = select_action_a(state_a, epsilon)
            next_state_a, reward_a, terminated_a, info_a = env.step(action_a,1)







                






                
                

            
            env.render()



try:
    print('start')
    train()
except KeyboardInterrupt:
    print('stop')
    target_model_a.save_weights(weights_path_a)

