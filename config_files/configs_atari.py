from yacs.config import CfgNode as CN

cfg = CN(
    {
        "exp_name": "test",
        'logdir': '/project/TransDreamer',
        "resume": True,
        "resume_ckpt": "",
        "debug": False,
        "seed": 0,
        "run_id": "run_0",
        "model": "dreamer_transformer",
        "total_steps": 1e7,
        "arch": {
            "use_pcont": True, # not used
            "mem_size": 100000, # checked
            "prefill": 50000, # checked
            "H": 15, # checked
            "world_model": {
                "reward_layer": 0, # checked
                "q_emb_action": False, # checked
                "act_after_emb": True, # not used
                "rec_sigma": 0.3, # checked
                "transformer": {
                    "max_time": 2000, # not used
                    "num_heads": 8, # checked
                    "d_model": 600, # checked
                    "d_inner": 64, # checked
                    "d_ff_inner": 1024, # checked
                    "dropout": 0.1, # checked
                    "dropatt": 0.1, # checked
                    "activation": "relu", # not used
                    "pos_enc": "temporal", # not used
                    "embedding_type": "linear", # not used
                    "n_layers": 6, # checked
                    "pre_lnorm": True, # checked
                    "deter_type": "concat_o", # checked
                    "gating": False, # checked
                },
                "q_transformer": {
                    "max_time": 2000,
                    "num_heads": 8,
                    "d_model": 600,
                    "d_inner": 64,
                    "d_ff_inner": 1024,
                    "dropout": 0.1,
                    "dropatt": 0.1,
                    "activation": "relu",
                    "pos_enc": "temporal",
                    "embedding_type": "linear",
                    "n_layers": 2,
                    "pre_lnorm": True,
                    "deter_type": "concat_o",
                    "q_emb_action": False,
                    "gating": False,
                },
                "RSSM": {
                    "act": "elu", # checked
                    "weight_init": "xavier", # checked
                    "stoch_size": 32, # checked
                    "stoch_discrete": 32, # checked
                    "deter_size": 600, # not used
                    "hidden_size": 600, # checked
                    "rnn_type": "LayerNormGRUV2", # not used
                    "ST": True, # not used
                },
                "reward": {
                    "num_units": 400, # checked
                    "act": "elu", # checked
                    "dist": "normal", # checked
                    "layers": 4, # checked
                },
                "pcont": {
                    "num_units": 400, # checked
                    "dist": "binary", # checked
                    "act": "elu", # checked
                    "layers": 4, # checked
                },
            },
            "actor": {
                "num_units": 400, # checked
                "act": "elu", # checked
                "init_std": 5.0, # checked
                "dist": "onehot", # checked
                "layers": 4, # checked
                "actor_loss_type": "reinforce", # checked
            },
            "value": {
                "num_units": 400, # checked
                "act": "elu", # checked
                "dist": "normal", # checked
                "layers": 4, # checked
            },
            "decoder": {
                "dec_type": "conv", # checked
            },
        },
        "loss": {
            "pcont_scale": 5.0, # checked
            "kl_scale": 0.1, # checked
            "free_nats": 0.0, # checked
            "kl_balance": 0.8, # checked
        },
        "env": {
            "action_size": 18, # checked
            "name": "atari_boxing", # checked
            "action_repeat": 4, # checked
            "max_steps": 1000, # checked
            "life_done": False, # checked
            "precision": 32, # checked
            "time_limit": 108000, # checked
            "grayscale": True, # checked
            "all_actions": True, # checked
            "time_penalty": 0.0, # checked
        },
        "rl": {
            "discount": 0.999, # checked
            "lambda_": 0.95, # checked
            "expl_amount": 0.0, # not used
            "expl_decay": 200000.0, # not used
            "expl_min": 0.0, # not used
            "expl_type": "epsilon_greedy", # not used
            "r_transform": "tanh", # checked
        },
        "data": {
            'datadir': '/project/TransDreamer',
        },
        "train": {
            "batch_length": 50, # checked
            "batch_size": 50, # checked
            "train_steps": 100, # not used
            "train_every": 16, # checked
            "print_every_step": 2000,
            "log_every_step": 1e3,
            "checkpoint_every_step": 1e4,
            "eval_every_step": 1e5,
            "n_sample": 10,
            "imag_last_T": False,
        },
        "optimize": {
            "model_lr": 2e-4, # checked
            "value_lr": 1e-4, # checked
            "actor_lr": 4e-5, # checked
            "optimizer": "adamW", # checked
            "grad_clip": 100.0, # checked
            "weight_decay": 1e-6, # checked
            "eps": 1e-5, # checked
            "reward_scale": 1.0, # not used
            "discount_scale": 5.0, # not used
        },
        "checkpoint": {
            'checkpoint_dir': '/project/TransDreamer',
            "max_num": 10,
        },
    }
)
