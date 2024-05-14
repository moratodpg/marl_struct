""" Wrapper for struct_env respecting the interface of PyMARL. """

import numpy as np
import torch

from imp_marl.environments.struct_small import StructSmall
from imp_marl.imp_wrappers.pymarl_wrapper.MultiAgentEnv import MultiAgentEnv


class PymarlMASmall(MultiAgentEnv):
    """
    Wrapper for Struct_Simp respecting the interface of PyMARL.

    It manipulates an imp_env to create all inputs for PyMARL agents.
    """

    def __init__(self,
                 config_file: str = "small_k2n2",
                 state_obs: bool = True,
                 obs_multiple: bool = False,
                 seed=None):
        """
        Initialise based on the full configuration.

        Args:
            config_file: (str) Name of the configuration file
            state_obs: (bool) State contains the concatenation of obs
            obs_multiple: (bool) Obs contains the concatenation of all obs
            seed: (int) seed for the random number generator
        """
        # Check struct type and default values
      
        assert isinstance(state_obs, bool) \
               and isinstance(obs_multiple, bool) \
        
        self.config_file = config_file
        self.state_obs = state_obs
        self.obs_multiple = obs_multiple
        self._seed = seed

        self.struct_env = StructSmall(self.config_file)

        self.n_agents = self.struct_env.n_comp
        self.episode_limit = self.struct_env.ep_length
        self.agent_list = self.struct_env.agent_list
        self.n_actions = self.struct_env.actions_per_agent

        self.action_histogram = {"action_" + str(k): 0 for k in
                                 range(self.n_actions)}

        self.unit_dim = self.get_unit_dim()  # Qplex requirement

    def update_action_histogram(self, actions):
        """
        Update the action histogram for logging.

        Args:
            actions: list of actions
        """
        for k, action in zip(self.struct_env.agent_list, actions):
            if type(action) is torch.Tensor:
                action_str = str(action.cpu().numpy())
            else:
                action_str = str(action)
            self.action_histogram["action_" + action_str] += 1

    def step(self, actions):
        """
        Ask to run a step in the environment.

        Args:
            actions: list of actions

        Returns:
            rewards: list of rewards
            done: True if the episode is finished
            info: dict of info for logging
        """

        self.update_action_histogram(actions)
        action_dict = {k: action
                       for k, action in
                       zip(self.struct_env.agent_list, actions)}
        _, rewards, done, _ = self.struct_env.step(action_dict)
        info = {}
        if done:
            for k in self.action_histogram:
                self.action_histogram[k] /= self.episode_limit * self.n_agents
            info = self.action_histogram
        return rewards[self.struct_env.agent_list[0]], done, info

    def get_obs(self):
        """ Returns all agent observations in a list. """
        return [self.get_obs_agent(i) for i in
                range(self.n_agents)]

    def get_unit_dim(self):
        """ Returns the dimension of the unit observation used by QPLEX. """
        return len(self.all_obs_from_struct_env()) // self.n_agents

    def get_obs_agent(self, agent_id: int):
        """
        Returns observation for agent_id

        Args:
            agent_id: id of the agent (int in range(self.n_agents)).
        """
        agent_name = self.struct_env.agent_list[agent_id]

        if self.obs_multiple:
            obs = self.all_obs_from_struct_env()
        else:
            obs = self.struct_env.observations[agent_name]

        return obs

    def get_obs_size(self):
        """ Returns the size of the observation. """
        return len(self.get_obs_agent(0))

    def all_obs_from_struct_env(self):
        """ Returns all observations concatenated in a single vector. """
        # Concatenate all obs with a single time.
        idx = 0
        obs = None
        for k, v in self.struct_env.observations.items():
            if idx == 0:
                obs = v
                idx = 1
            else:
                obs = np.append(obs, v)
        return obs

    def get_state(self):
        """ Returns the state of the environment. """
        state = []
        if self.state_obs:
            state = np.append(state, self.all_obs_from_struct_env())
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.get_state())

    def get_avail_actions(self):
        """ Returns the available actions of all agents in a list. """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """
        Returns the available actions for agent_id.

        Args:
            agent_id: id of the agent (int in range(self.n_agents)).
        """
        return [1] * self.n_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take. """
        return self.struct_env.actions_per_agent

    def reset(self):
        """ Returns initial observations and states. """
        self.action_histogram = {"action_" + str(k): 0 for k in
                                 range(self.n_actions)}
        self.struct_env.reset()
        return self.get_obs(), self.get_state()

    def render(self):
        """ See base class. """
        pass

    def close(self):
        """ See base class. """
        pass

    def seed(self):
        """ Returns the random seed """
        return self._seed

    def save_replay(self):
        """ See base class. """
        pass

    def get_stats(self):
        """ See base class. """
        return {}
