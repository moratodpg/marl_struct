""" Interface for creating IMP environments. """

import numpy as np
import os
import importlib
from imp_marl.environments.imp_env import ImpEnv

class StructSmall(ImpEnv):
    """ k-out-of-n system (struct) class. 

    Attributes:
        n_comp: Integer indicating the number of components.
        discount_reward: Float indicating the discount factor.
        k_comp: Integer indicating the number 'k' (out of n) components in the system.
        campaign_cost: Boolean indicating whether a global campaign cost is considered in the reward model.
        ep_length: Integer indicating the number of time steps in the finite horizon.
        proba_size: Integer indicating the number of bins considered in the discretisation of the damage probability.
        n_obs_inspection: Integer indicating the number of potential outcomes resulting from an inspection.
        actions_per_agent: Integer indicating the number of actions that an agent can take.
        initial_damage_proba: Numpy array containing the initial damage probability.
        transition_model: Numpy array containing the transition model that drives the environment dynamics.
        inspection_model: Numpy array containing the inspection model.
        agent_list: Dictionary categorising the number of agents.
        time_step: Integer indicating the current time step.
        damage_proba: Numpy array contatining the current damage probability.
        observations: Dictionary listing the observations received by the agents in the Dec-POMDP.

    Methods: 
        reset
        step
        pf_sys
        immediate_cost
        belief_update_uncorrelated
    """
    def __init__(self, config_file=None):
        """ Initialises the class according to the provided config instructions.

        Args:
            config: Dictionary containing config parameters.
                Keys:
                    n_comp: Number of components.
                    discount_reward: Discount factor. 
                    k_comp: Number of components required to not fail.
                    campaign_cost: Whether to include campaign cost in reward.
        """
        if config_file is None:
            print("No config file provided.")

        self.config_file = config_file
        module_path = f"imp_marl.environments.pomdp_models.{self.config_file}"
        module = importlib.import_module(module_path)
        config = getattr(module, 'config', None)  # None is the default if 'config' is not found

        self.n_comp = config["n_comp"]
        self.discount_reward = config["discount_reward"]
        self.k_comp = self.n_comp - 1 if config["k_comp"] is None \
            else config["k_comp"]
        self.campaign_cost = config["campaign_cost"]
        self.ep_length = config["ep_length"] 
        self.proba_size = config["proba_size"]
        self.n_obs_inspection = config["n_obs_inspection"]
        self.actions_per_agent = config["actions_per_agent"]

        # (ncomp components, proba_size damage)
        self.initial_damage_proba = np.array(config["initial_damage_prob"])

        # (actions, components, damage states, damage states)
        self.transition_model = np.array(config["transition_model"])

        insp_accuracy = config["inspection_model"]
        O_inspect = np.zeros((self.n_comp, self.proba_size, self.n_obs_inspection))
        for i in range(self.n_comp):
            O_inspect[i, 0] = [insp_accuracy[i], 1-insp_accuracy[i], 0.0, 0.0]
            O_inspect[i, 1] = [(1-insp_accuracy[i])/2, insp_accuracy[i], (1-insp_accuracy[i])/2, 0.0]
            O_inspect[i, 2] = [0.0, (1-insp_accuracy[i])/2, insp_accuracy[i], (1-insp_accuracy[i])/2]
            O_inspect[i, 3] = [0.0, 0.0, 0.0, 1.0]
        
        O_noinspect = np.zeros((self.n_comp, self.proba_size, self.n_obs_inspection))
        for i in range(self.n_comp):
            O_noinspect[i, 0] = [1/4, 1/4, 1/4, 1/4]
            O_noinspect[i, 1] = [1/4, 1/4, 1/4, 1/4]
            O_noinspect[i, 2] = [1/4, 1/4, 1/4, 1/4]
            O_noinspect[i, 3] = [1/4, 1/4, 1/4, 1/4]

        # (3 actions, 5 components, 4 damage states, 4 inspections)
        self.inspection_model = O_inspect
        self.no_inspection_model = O_noinspect
        
        self.cost_inspection = config["cost_inspection"]
        self.cost_repair = config["cost_repair"]
        self.cost_campaign = config["cost_campaign"]

        self.failure_cost_factor = config["failure_cost_factor"]
        cost_rep = np.array(self.cost_repair)
        self.cost_failure = (np.sum(cost_rep)) * self.failure_cost_factor

        self.agent_list = ["agent_" + str(i) for i in range(self.n_comp)]

        self.time_step = 0
        self.damage_proba = self.initial_damage_proba
        self.observations = None  
        
        self.reset()

    def reset(self):
        """ Resets the environment to its initial step.

        Returns:
            observations: Dictionary with the damage probability received by the agents.
        """
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        self.time_step = 0
        self.damage_proba = self.initial_damage_proba
        self.observations = {}
        for i in range(self.n_comp):
            self.observations[self.agent_list[i]] = np.concatenate(
                (self.damage_proba[i], [self.time_step / self.ep_length]))
        return self.observations

    def step(self, action: dict):
        """ Transitions the environment by one time step based on the selected actions. 

        Args:
            action: Dictionary containing the actions assigned by each agent.

        Returns:
            observations: Dictionary with the damage probability received by the agents.
            rewards: Dictionary with the rewards received by the agents.
            done: Boolean indicating whether the final time step in the horizon has been reached.
            inspection: Integers indicating which inspection outcomes have been collected.
        """
        action_list = np.zeros(self.n_comp, dtype=int)
        for i in range(self.n_comp):
            action_list[i] = action[self.agent_list[i]]

        inspection, next_proba = \
            self.belief_update_uncorrelated(self.damage_proba, action_list)

        reward_ = self.immediate_cost(self.damage_proba, action_list)
        reward = self.discount_reward ** self.time_step * reward_.item()  # Convert float64 to float

        rewards = {}
        for i in range(self.n_comp):
            rewards[self.agent_list[i]] = reward

        self.time_step += 1

        self.observations = {}
        for i in range(self.n_comp):
            self.observations[self.agent_list[i]] = np.concatenate(
                (next_proba[i], [self.time_step / self.ep_length]))

        self.damage_proba = next_proba

        # An episode is done if the agent has reached the target
        done = self.time_step >= self.ep_length

        return self.observations, rewards, done, inspection

    def pf_sys(self, pf, k):
        """ Computes the system failure probability pf_sys for k-out-of-n components
        
        Args:
            pf: Numpy array with components' failure probability.
            k: Integer indicating k (out of n) components.
        
        Returns:
            PF_sys: Numpy array with the system failure probability.
        """
        n = pf.size
        nk = n - k
        m = k + 1
        A = np.zeros(m + 1)
        A[1] = 1
        L = 1
        for j in range(1, n + 1):
            h = j + 1
            Rel = 1 - pf[j - 1]
            if nk < j:
                L = h - nk
            if k < j:
                A[m] = A[m] + A[k] * Rel
                h = k
            for i in range(h, L - 1, -1):
                A[i] = A[i] + (A[i - 1] - A[i]) * Rel
        PF_sys = 1 - A[m]
        return PF_sys

    def immediate_cost(self, B, a):
        """ Computes the immediate reward (negative cost) based on current (and next) damage probability and action selected
        
            Args:
                B: Numpy array with current damage probability.
                a: Numpy array with actions selected.
                B_: Numpy array with the next time step damage probability.
                d_rate: Numpy array with current deterioration rates.
            
            Returns:
                cost_system: Float indicating the reward received.
        """
        cost_system = 0
        PF = B[:, -1]

        campaign_executed = False
        for i in range(self.n_comp):
            if a[i] == 1:
                cost_system += self.cost_inspection[i] # Individual inspection costs 
                if self.campaign_cost and not campaign_executed:
                    campaign_executed = True # Campaign executed
            elif a[i] == 2:
                cost_system += self.cost_repair[i]
                if self.campaign_cost and not campaign_executed:
                    campaign_executed = True # Campaign executed
        if self.n_comp < 2:  # single component setting
            PfSyS = PF
        else:
            PfSyS = self.pf_sys(PF, self.k_comp)

        cost_system += PfSyS * (self.cost_failure)
        if campaign_executed: 
            cost_system += self.cost_campaign
        return cost_system

    def belief_update_uncorrelated(self, proba, action):
        """ Transitions the environment based on the current damage prob, actions selected, and current deterioration rate
            In this case, the initial damage prob are not correlated among components.
        
        Args:
            proba: Numpy array with current damage probability.
            action: Numpy array with actions selected.
            drate: Numpy array with current deterioration rates.

        Returns:
            inspection: Integers indicating which inspection outcomes have been collected.
            new_proba: Numpy array with the next time step damage probability.
            new_drate: Numpy array with the next time step deterioration rate.
        """
        new_proba = np.zeros((self.n_comp, self.proba_size))
        new_proba[:] = proba
        inspection = np.zeros(self.n_comp)
        for i in range(self.n_comp):
            p1 = self.transition_model[action[i], i].T.dot(new_proba[i])  
            # environment transition
            new_proba[i] = p1
            

            if action[i] == 1:
                ins_dist = self.inspection_model[i].T.dot(p1)
                inspection[i] = np.random.choice(range(0, self.n_obs_inspection), size=None,
                                                     replace=True, p=ins_dist)
                new_proba[i, :] = p1 * self.inspection_model[i, :, int(inspection[i])] / (
                    p1.dot(self.inspection_model[i, :, int(inspection[i])]))  # belief update
            elif action[i] == 0 or action[i] == 2:
                ins_dist = self.no_inspection_model[i].T.dot(p1)
                inspection[i] = np.random.choice(range(0, self.n_obs_inspection), size=None,
                                                     replace=True, p=ins_dist)
                new_proba[i, :] = p1 * self.no_inspection_model[i, :, int(inspection[i])] / (
                    p1.dot(self.no_inspection_model[i, :, int(inspection[i])]))  # belief update
            
            #if action[i] == 2:
                # action in b_prime has already
                # been accounted in the env transition
        return inspection, new_proba
