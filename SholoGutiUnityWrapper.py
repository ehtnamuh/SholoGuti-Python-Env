import mlagents
import numpy
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


class SholoGutiUnityWrapper:

    def __init__(self, path='C:/Users/samin/OneDrive/Desktop/KamlaGutiBuild/KamlaGuti', no_graphics=False,
                 time_out_wait=60):
        self.env_parameters = EnvironmentParametersChannel()
        self.env = UnityEnvironment(file_name=path, seed=1, side_channels=[], no_graphics=no_graphics,
                                    timeout_wait=time_out_wait)
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        self.episode_rewards = 0
        self.tracked_agent = -1

    def get_observation(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        if(self.is_episode_over()):
            return terminal_steps.obs[0][0][:-1]
        return decision_steps.obs[0][0][:-1]

    # function returns episode_done, step_done, reward, next_obs
    def env_step_partial(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        tracked_agent = -1
        # Track the first agent we see if not tracking
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]
        self.env.set_actions(self.behavior_name, self.action)
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        step_holder = decision_steps
        if tracked_agent in decision_steps:  # The agent requested a decision
            self.episode_rewards += decision_steps[tracked_agent].reward
        elif tracked_agent in terminal_steps:
            step_holder = terminal_steps
            self.episode_rewards += terminal_steps[tracked_agent].reward
        reward = step_holder[tracked_agent].reward
        episode_done =  self.is_episode_over()
        step_done = self.is_step_ended(obs=step_holder.obs[0][0])
        next_obs = self.get_observation()
        return episode_done, step_done, reward, next_obs


    def is_step_ended(self, obs):
        return (obs[-1] == -2)

    def is_episode_over(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        tracked_agent = -1
        if(len(decision_steps) > 0):
            tracked_agent = decision_steps.agent_id[0]
        else:
            tracked_agent = terminal_steps.agent_id[0]
        return tracked_agent in terminal_steps

    def set_actions(self, action=None):
        if action is None:
            action = self.generate_random_action()
        else:
            action = numpy.array([action])
        self.action = action
        d, t = self.env.get_steps(self.behavior_name)
        print(len(d))
        self.env.set_actions(self.behavior_name, self.action)

    def generate_random_action(self,  decision_steps = None):
        if decision_steps is None:
            decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        return self.spec.create_random_action(len(decision_steps))

    def get_max_observation(self):
        raise NotImplemented

    # side channel communication
    def set_env_parameters(self, parameter_name="default", val=0.0):
        self.env_parameters.set_float_parameter(parameter_name, val)

    def reset_env(self):
        self.env.reset()

    def close_env(self):
        self.env.close()

    def make(self, path):
        return self.env
