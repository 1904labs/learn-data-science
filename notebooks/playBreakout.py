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
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import PIL
import matplotlib.animation as animation
import os
from tf_agents.trajectories.time_step import StepType
from tf_agents.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
#import tf_agents.policies.gaussian_policy.GaussianPolicy

max_episode_steps = 3000
environment_name = "BreakoutNoFrameskip-v4"
import sys

env = suite_atari.load(environment_name, max_episode_steps = max_episode_steps, 
                gym_env_wrappers=[AtariPreprocessing,FrameStack4 ])

tf_env = TFPyEnvironment(env)
#tf.random.set_seed(42)
#np.random.seed(42)
#env.seed(42)


first_episode_done = False
#agent = keras.models.load_model('policy_900')
total_reward = tf.constant([0.], dtype = tf.float32)
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")
gameIndex = 0
rewards_per_game= {}

frames=[]
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))



def flush_frame(gameIndex,total_reward):
    global frames
    total_reward = total_reward.numpy().item()
    image_path = os.path.join("images", "rl", f"breakout_seed_{total_reward}_{gameIndex}_{policy_num}.gif")
    frame_images = [PIL.Image.fromarray(frame) for frame in frames]
    frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)
    frames=[]

    
"""
def debug_trajectory(trajectory):
    global total_reward
    global first_episode_done
    if not first_episode_done:
        total_reward = total_reward + trajectory.reward
        if trajectory.step_type == StepType.LAST:
            first_episode_done = True
    #print(trajectory.reward)
"""

def debug_trajectory(trajectory):
    global total_reward
    global first_episode_done
    global gameIndex
    #if not first_episode_done:
    #    total_reward = total_reward + trajectory.reward
    if trajectory.step_type == StepType.LAST:
        #first_episode_done = True
        rewards_per_game.update({gameIndex:total_reward})
        flush_frame(gameIndex,total_reward)
        total_reward = 0
        gameIndex = gameIndex + 1
    else:
        total_reward = total_reward + trajectory.reward
        
    #print(trajectory.reward)
    
prev_lives = tf_env.pyenv.envs[0].ale.lives()
def reset_and_fire_on_life_lost(trajectory):
    global prev_lives
    lives = tf_env.pyenv.envs[0].ale.lives()
    if prev_lives != lives:
        #tf_env.reset()
        tf_env.pyenv.envs[0].step(np.array(1, dtype=np.int32))
        prev_lives = lives

policy_num = sys.argv[1]

#print(type(agent))
saved_policy = tf.compat.v2.saved_model.load(f'policy_{policy_num}')
saved_policy.time_step_spec =  tf_env.time_step_spec()
saved_policy.action_spec = tf_env.action_spec()
saved_policy.policy_state_spec = () # tf_env.policy_state_spec
saved_policy.info_spec = ()
saved_policy.emit_log_probability = True

saved_policy = EpsilonGreedyPolicy(saved_policy, epsilon=0.005)

#saved_policy = tf_agents.policies.gaussian_policy.GaussianPolicy(saved_policy)

#agent = tf.saved_model.load('policy_100')
#agent = tf.keras.models.load_model('policy_100')
#agent = tf.keras.models.load_model('policy_100')
#policy = tf.saved_model.load('')
#print(type(agent))
tf_env.pyenv.envs[0].step(np.array(1, dtype=np.int32))
watch_driver = DynamicStepDriver(
    tf_env,
    saved_policy,
    observers=[save_frames, reset_and_fire_on_life_lost, ShowProgress(max_episode_steps), debug_trajectory],
    num_steps=max_episode_steps)
#tf_env.pyenv.envs[0].step(np.array(1, dtype=np.int32))
final_time_step, final_policy_state = watch_driver.run()
#obs, reward, done, info = tf_env.pyenv.envs[0].step(np.array(1, dtype=np.int32))
print('rewards earned', rewards_per_game)

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

#plot_animation(frames)

float_reward = total_reward.numpy().item()

image_path = os.path.join("images", "rl", f"breakout_seed_{float_reward}_{policy_num}.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)
"""
def flush_frame(gameIndex,total_reward):
    global frames
    image_path = os.path.join("images", "rl", f"breakout_seed_{total_reward}_{gameIndex}_{policy_num}.gif")
    frame_images = [PIL.Image.fromarray(frame) for frame in frames]
    frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)
    frames=[]
"""    
    

# for idx, frame in enumerate(frame_images):
#     image_path =  os.path.join("images", "rl", f"{idx}.gif")
#     frame.save(image_path, format = 'GIF')
