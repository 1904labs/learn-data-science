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


logging.getLogger().setLevel(logging.INFO)


max_episode_steps = 27000
environment_name = "BreakoutNoFrameskip-v4"

env = suite_atari.load(environment_name, max_episode_steps = max_episode_steps, 
                gym_env_wrappers=[AtariPreprocessing,FrameStack4 ])

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

print("After  Agent.initialize()")

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = agent.collect_data_spec,
    batch_size = tf_env.batch_size,
    max_length = 1000000
)

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
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f} done:{:.5f}".format(iteration, train_loss.loss.numpy(), iteration / n_iterations * 100.0), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
        if iteration % 100 == 0 and iteration > 0:
            #keras.saved_model.saved_model(my_policy, 'policy_' + str(iteration))
            #tf.saved_model.save(agent, 'policy_' + str(iteration))
            my_policy = agent.collect_policy
            saver = PolicySaver(my_policy)
            saver.save('policy_' + str(iteration))
    
#train_agent(10000000)

train_agent(10000000)

tf.saved_model.save(agent, "full_run")
