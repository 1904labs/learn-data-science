from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
import numpy as np
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import time
from tf_agents.networks.q_network import QNetwork
import tensorflow.keras as keras
import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function
from tf_agents.eval.metric_utils import log_metrics
import logging
from tf_agents.policies.policy_saver import PolicySaver
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import PIL
import sys
from tf_agents.replay_buffers.py_hashed_replay_buffer import PyHashedReplayBuffer

from tensorflow.keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def load_policy_from_disk(policy_num):
    saved_policy = tf.compat.v2.saved_model.load(f'backupPol/policy_{policy_num}')
    return saved_policy


initial_policy = 0
if len(sys.argv) == 2:
    initial_policy = int(sys.argv[1])

policy = None    
if initial_policy != 0:
    policy = load_policy_from_disk(initial_policy)

logging.getLogger().setLevel(logging.INFO)


max_episode_steps = 27000
environment_name = "BreakoutNoFrameskip-v4"

env = suite_atari.load(environment_name, max_episode_steps = max_episode_steps, 
                gym_env_wrappers=[AtariPreprocessing,FrameStack4 ])
tf.random.set_seed(42)
np.random.seed(42)
env.seed(42)

tf_env = TFPyEnvironment(env)

# for i in range(max_episode_steps):
#     timestep = tf_env.step(np.array(np.random.choice([0,1,2,3]), dtype= np.int32))
#     if(timestep.is_last()):
#         print("game over", i)
#         break
#     tf_env.render(mode = "human")
#     time.sleep(0.2)

preprocessing_layers = keras.layers.Lambda(
    lambda obs: tf.cast(obs, np.float32) / 255.
)

print("after preprocessing layer")
conv_layer_params = [(32, (8,8),4), (64,(4,4),2), (64, (3,3),1)]

fc_layer_params = [512]

q_net = QNetwork(
                tf_env.observation_spec(),
                tf_env.action_spec(),
                preprocessing_layers = preprocessing_layers,
                conv_layer_params = conv_layer_params,
                fc_layer_params = fc_layer_params
                )

train_step = tf.Variable(0)
update_period = 4
optimizer = keras.optimizers.RMSprop(lr = 2.5e-4, rho = 0.95, momentum= 0.0, epsilon= 0.00001, centered=True)

print("Before Epsilon function")
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate = 1.0,
    decay_steps = 250000,
    end_learning_rate= 0.01
)
print("Before Agent")
agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network = q_net,
    optimizer = optimizer,
    target_update_period = 2000,
    td_errors_loss_fn = keras.losses.Huber(reduction = "none"),
    gamma = 0.99,
    train_step_counter = train_step,
    epsilon_greedy = lambda: epsilon_fn(train_step))

agent.initialize()
if policy != None:
    agent.policy = policy
    

print("After  Agent.initialize()")


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = agent.collect_data_spec,
    batch_size = tf_env.batch_size,
    max_length = 100000
)

"""
replay_buffer = PyHashedReplayBuffer(
    data_spec = agent.collect_data_spec,
    #batch_size = tf_env.batch_size,
    capacity = 1000000
)
"""
print("After  replay_buffer")
replay_buffer_observer = replay_buffer.add_batch

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]
print("Before  DynamicStepDriver")

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps= update_period
)
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")

initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

init_driver = DynamicStepDriver(tf_env, initial_collect_policy, observers = [replay_buffer.add_batch, ShowProgress(20000)], num_steps = 20000)

final_time_step , final_policy_state = init_driver.run()

dataset = replay_buffer.as_dataset(sample_batch_size=64, num_steps = 2,num_parallel_calls = 3).prefetch(3)

#collect_driver.run = function(collect_driver.run)
#agent.train = function(agent.train)



my_policy = agent.collect_policy
saver = PolicySaver(my_policy, batch_size=None)



def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(initial_policy, n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f} done:{:.5f}".format(iteration, train_loss.loss.numpy(), iteration / n_iterations * 100.0), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
        if iteration % 10000 == 0 and iteration > 0:
            #keras.saved_model.saved_model(my_policy, 'policy_' + str(iteration))
            #tf.saved_model.save(agent, 'policy_' + str(iteration))
            my_policy = agent.policy
            saver = PolicySaver(my_policy)
            saver.save('policy_' + str(iteration))
        #    pass
    
train_agent(10000000)

#train_agent(10000000)
#train_agent(10000)

#tf.saved_model.save(agent, "full_run")
# frames = []
# def save_frames(trajectory):
#     global frames
#     frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))


# ### Run Portion
# prev_lives = tf_env.pyenv.envs[0].ale.lives()
# def reset_and_fire_on_life_lost(trajectory):
#     global prev_lives
#     lives = tf_env.pyenv.envs[0].ale.lives()
#     if prev_lives != lives:
#         tf_env.reset()
#         tf_env.pyenv.envs[0].step(np.array(1, dtype = np.int32))
#         prev_lives = lives


# watch_driver = DynamicStepDriver(
#     tf_env,
#     agent.policy,
#     observers=[save_frames, reset_and_fire_on_life_lost, ShowProgress(1000)],
#     num_steps=1000)
# final_time_step, final_policy_state = watch_driver.run()

# def update_scene(num, frames, patch):
#     patch.set_data(frames[num])
#     return patch,

# def plot_animation(frames, repeat=False, interval=40):
#     fig = plt.figure()
#     patch = plt.imshow(frames[0])
#     plt.axis('off')
#     anim = animation.FuncAnimation(
#         fig, update_scene, fargs=(frames, patch),
#         frames=len(frames), repeat=repeat, interval=interval)
#     plt.close()
#     return anim

# plot_animation(frames)


# image_path = os.path.join("images", "rl", "breakout.gif")
# frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
# frame_images[0].save(image_path, format='GIF',
#                      append_images=frame_images[1:],
#                      save_all=True,
#                      duration=30,
#                      loop=0)



#K.get_session().close();
