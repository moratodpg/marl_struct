config = {}

config["n_comp"] = 2
config["discount_reward"] = 0.99
config["k_comp"] = 2
config["campaign_cost"] = True

config["ep_length"] = 20
config["proba_size"] = 4
config["n_obs_inspection"] = 4
config["actions_per_agent"] = 3

config["initial_damage_prob"] = [
    [1, 0.0, 0.0, 0.0],
    [1, 0.0, 0.0, 0.0],
]

# Action, component, state, state
config["transition_model"] = [
    [
        [[0.82, 0.13, 0.05, 0.0],
        [0.0, 0.87, 0.09, 0.04],
        [0.0, 0.0, 0.91, 0.09],
        [0.0, 0.0, 0.0, 1]],

        [[0.72, 0.19, 0.09, 0.0],
        [0.0, 0.78, 0.18, 0.04],
        [0.0, 0.0, 0.85, 0.15],
        [0.0, 0.0, 0.0, 1]],
    ],
    [
        [[0.82, 0.13, 0.05, 0.0],
        [0.0, 0.87, 0.09, 0.04],
        [0.0, 0.0, 0.91, 0.09],
        [0.0, 0.0, 0.0, 1]],

        [[0.72, 0.19, 0.09, 0.0],
        [0.0, 0.78, 0.18, 0.04],
        [0.0, 0.0, 0.85, 0.15],
        [0.0, 0.0, 0.0, 1]],
    ],
    [
        [[0.82, 0.13, 0.05, 0.0],
        [0.0, 0.87, 0.09, 0.04],
        [0.0, 0.0, 0.91, 0.09],
        [0.0, 0.0, 0.0, 1]],

        [[0.72, 0.19, 0.09, 0.0],
        [0.0, 0.78, 0.18, 0.04],
        [0.0, 0.0, 0.85, 0.15],
        [0.0, 0.0, 0.0, 1]],
    ],
]

config["inspection_model"] = [0.8, 0.85]

config["cost_inspection"] = [-20, -40]
config["cost_repair"] = [-30, -90]
config["cost_campaign"] = -40
config["failure_cost_factor"] = 3