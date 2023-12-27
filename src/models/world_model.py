from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions import Independent, Normal, kl_divergence
import wandb

from .distributions import MyRelaxedOneHotCategorical
from .encoder_decoder import ImgEncoder, ImgDecoder, DenseDecoder, ActionDecoder
from .transformer import Transformer
from .utils import (
    Conv2DBlock,
    ConvTranspose2DBlock,
    Linear,
    MLP,
    GRUCell,
    LayerNormGRUCell,
    LayerNormGRUCellV2,
)
import pdb


@dataclass
class TransformerDynamicConfig:
    input_type: str
    action_size: int
    hidden_size: int
    stoch_size: int
    stoch_discrete: int
    discrete_type: str
    q_emb_action: bool
    reward_layer: int
    act: str = "elu"
    weight_init: str = "xavier"

class TransformerDynamic(nn.Module):
    def __init__(self, img_enc: ImgEncoder, cell: Transformer, q_transformer: Transformer, config: TransformerDynamicConfig):
        super().__init__()

        self.img_enc = img_enc
        self.cell = cell

        action_size = config.action_size
        hidden_size = config.hidden_size
        self.stoch_size = config.stoch_size
        self.stoch_discrete = config.stoch_discrete

        # self.ST = cfg.arch.world_model.RSSM.ST
        # self.pre_lnorm = cfg.arch.world_model.transformer.pre_lnorm
        # self.act_after_emb = cfg.arch.world_model.act_after_emb

        self.d_model = self.cell.d_model
        if self.stoch_discrete:
            latent_dim = self.stoch_size * self.stoch_discrete
            latent_dim_out = latent_dim
        else:
            latent_dim = self.stoch_size
            latent_dim_out = latent_dim * 2
        self.act_stoch_mlp = Linear(action_size + latent_dim, self.d_model, weight_init=config.weight_init)

        q_emb_size = 1536
        if config.q_emb_action:
            q_emb_size = q_emb_size + action_size

        self.q_trans_model = q_transformer
        self.q_emb = nn.Sequential(Linear(q_emb_size, self.d_model), nn.ELU())

        self.q_trans_deter_type = self.q_trans_model.deter_type
        self.q_trans_layers = self.q_trans_model.n_layers
        if self.q_trans_deter_type == "concat_o":
            d_model = self.q_trans_layers * self.d_model
        else:
            d_model = self.d_model
        self.post_stoch_mlp = MLP([d_model, hidden_size, latent_dim_out], act=config.act, weight_init=config.weight_init)

        self.deter_type = self.cell.deter_type
        self.n_layers = self.cell.n_layers
        if self.deter_type == "concat_o":
            d_model = self.n_layers * self.d_model
        else:
            d_model = self.d_model
        self.prior_stoch_mlp = MLP([d_model, hidden_size, latent_dim_out], act=config.act, weight_init=config.weight_init)

        self.input_type = config.input_type
        self.discrete_type = config.discrete_type
        self.reward_layer = config.reward_layer

    def forward(self, traj, prev_state, temp):
        """
        traj:
          observations: embedding of observed images, B, T, C
          actions: (one-hot) vector in action space, B, T, d_act
          dones: scalar, B, T

        prev_state:
          deter: GRU hidden state, B, h1
          stoch: RSSM stochastic state, B, h2
        """

        obs = traj[self.input_type]
        obs = obs / 255.0 - 0.5
        obs_emb = self.img_enc(obs)  # B, T, C

        actions = traj["action"]
        dones = traj["done"]

        # q(s_t | o_t)
        post = self.infer_post_stoch(obs_emb, temp, action=None)
        s_t = post["stoch"][:, :-1]
        # p(s_(t+1) | s_t, a_t)
        prior = self.infer_prior_stoch(s_t, temp, actions[:, 1:])

        post["deter"] = prior["deter"]
        post["o_t"] = prior["o_t"]

        if self.stoch_discrete:
            prior["stoch_int"] = prior["stoch"].argmax(-1).float()
            post["stoch_int"] = post["stoch"].argmax(-1).float()

        return prior, post

    def get_feature(self, state, layer=None):
        if self.stoch_discrete:
            shape = state["stoch"].shape
            stoch = state["stoch"].reshape([*shape[:-2]] + [self.stoch_size * self.stoch_discrete])

            if layer:
                o_t = state["o_t"]  # B, T, L, D
                deter = o_t[:, :, layer]
            else:
                # assert self.deter_type == 'concat_o'
                deter = state["deter"]

            return torch.cat([stoch, deter], dim=-1)  # B, T, 2H

        else:
            stoch = state["stoch"]

            if layer:
                o_t = state["o_t"]  # B, T, L, D
                deter = o_t[:, :, layer]
            else:
                assert self.deter_type == "concat_o"
                deter = state["deter"]

            return torch.cat([stoch, deter], dim=-1)  # B, T, 2H

    def get_dist(self, state, temp, detach=False):
        if self.stoch_discrete:
            return self.get_discrete_dist(state, temp, detach)
        else:
            return self.get_normal_dist(state, detach)

    def get_normal_dist(self, state, detach):
        mean = state["mean"]
        std = state["std"]

        if detach:
            mean = mean.detach()
            std = std.detach()

        return Independent(Normal(mean, std), 1)

    def get_discrete_dist(self, state, temp, detach):
        logits = state["logits"]

        if detach:
            logits = logits.detach()

        if self.discrete_type == "discrete":
            dist = Independent(OneHotCategorical(logits=logits), 1)

        if self.discrete_type == "gumbel":
            try:
                dist = Independent(MyRelaxedOneHotCategorical(temp, logits=logits), 1)
            except:
                pdb.set_trace()

        return dist

    def encode_s(self, prev_stoch, action):
        B, T, N, C = prev_stoch.shape
        prev_stoch = prev_stoch.reshape(B, T, N * C)
        act_sto_emb = self.act_stoch_mlp(torch.cat([action, prev_stoch], dim=-1))

        act_sto_emb = F.elu(act_sto_emb)

        return act_sto_emb

    def infer_prior_stoch(self, prev_stoch, temp, actions):
        B, T = prev_stoch.shape[:2]
        if self.stoch_discrete:
            B, T, N, C = prev_stoch.shape

            act_sto_emb = self.encode_s(prev_stoch, actions)
        else:
            act_sto_emb = self.act_stoch_mlp(torch.cat([prev_stoch, actions], dim=-1))
            act_sto_emb = F.elu(act_sto_emb)

        s_t_reshape = act_sto_emb.reshape(B, T, -1, 1, 1)
        o_t = self.cell(s_t_reshape, None)  # B, T, L, D, H, W

        o_t = o_t.reshape(B, T, self.n_layers, -1)
        if self.deter_type == "concat_o":
            deter = o_t.reshape(B, T, -1)
        else:
            deter = o_t[:, :, -1]
        pred_logits = self.prior_stoch_mlp(deter).float()

        if self.stoch_discrete:
            B, T, N, C = prev_stoch.shape
            pred_logits = pred_logits.reshape(B, T, N, C)

        prior_state = self.stat_layer(pred_logits, temp)
        prior_state.update(
            {
                "deter": deter,
                "o_t": o_t,
            }
        )

        return prior_state

    def infer_post_stoch(self, observation, temp, action=None):
        if action is not None:
            observation = torch.cat([observation, action], dim=-1)
        B, T, C = observation.shape
        q_emb = self.q_emb(observation)
        e = self.q_trans_model(q_emb.reshape(B, T, self.d_model, 1, 1), None)

        e = e.reshape(B, T, self.q_trans_layers, -1)
        if self.q_trans_deter_type == "concat_o":
            e = e.reshape(B, T, -1)
        else:
            e = e[:, :, -1]
        logits = self.post_stoch_mlp(e).float()

        if self.stoch_discrete:
            logits = logits.reshape(B, T, self.stoch_discrete, self.stoch_size).float()
        post_state = self.stat_layer(logits, temp)

        return post_state

    def stat_layer(self, logits, temp):
        if self.stoch_discrete:
            if self.discrete_type == "discrete":
                # print(f'logits min: {logits.min()}')
                # print(f'logits mean: {logits.mean()}')
                # print(f'logits max: {logits.max()}')
                dist = Independent(OneHotCategorical(logits=logits), 1)
                stoch = dist.sample()
                stoch = stoch + dist.mean - dist.mean.detach()

            if self.discrete_type == "gumbel":
                try:
                    dist = Independent(MyRelaxedOneHotCategorical(temp, logits=logits), 1)
                except:
                    pdb.set_trace()
                stoch = dist.rsample()

            state = {
                "logits": logits,
                "stoch": stoch,
            }

        else:
            mean, std = logits.float().chunk(2, dim=-1)
            std = 2.0 * (std / 2.0).sigmoid()
            dist = Normal(mean, std)
            stoch = dist.rsample()

            state = {
                "mean": mean,
                "std": std,
                "stoch": stoch,
            }

        return state


class TransformerWorldModel(nn.Module):
    def __init__(self, dynamic: TransformerDynamic, img_dec: ImgDecoder, reward: DenseDecoder, pcont: DenseDecoder,
                 H: int, sequence_length: int, r_transform: Optional[str] = "tanh", discount: Optional[float] = 0.95):
        super().__init__()

        self.dynamic = dynamic
        self.img_dec = img_dec
        self.reward = reward
        self.pcont = pcont

        # self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
        # self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
        # self.discrete_type = cfg.arch.world_model.discrete_type
        # d_model = cfg.arch.world_model.transformer.d_model
        # self.d_model = d_model
        # deter_type = cfg.arch.world_model.transformer.deter_type
        # n_layers = cfg.arch.world_model.transformer.n_layers
        # if deter_type == "concat_o":
        #     d_model = n_layers * d_model

        # if self.stoch_discrete:
        #     dense_input_size = d_model + self.stoch_size * self.stoch_discrete
        # else:
        #     dense_input_size = d_model + self.stoch_size

        self.H = H
        self.sequence_length = sequence_length
        self.input_type = self.dynamic.input_type

        self.reward_layer = self.dynamic.reward_layer
        self.r_transform = dict(
            tanh=torch.tanh,
            sigmoid=torch.sigmoid,
            none=nn.Identity(),
        )[r_transform]
        self.discount = discount
        # self.lambda_ = cfg.rl.lambda_

        self.pcont_scale = None # override with TransDreamer
        self.kl_scale = None # override with TransDreamer
        self.kl_balance = None # override with TransDreamer
        self.free_nats = None # override with TransDreamer

        # self.grad_clip = cfg.optimize.grad_clip
        # self.action_size = cfg.env.action_size
        # self.log_every_step = cfg.train.log_every_step
        # self.grayscale = cfg.env.grayscale
        # self.slow_update = 0
        # self.n_sample = cfg.train.n_sample
        # self.imag_last_T = cfg.train.imag_last_T
        # self.slow_update_step = cfg.slow_update_step
        # self.log_grad = cfg.train.log_grad

    def forward(self, traj):
        raise NotImplementedError

    def compute_loss(self, traj, temp):
        self.train()
        self.requires_grad_(True)

        # world model rollout to obtain state representation
        prior_state, post_state = self.dynamic(traj, None, temp)

        # compute world model loss given state representation
        model_loss, model_logs = self.world_model_loss(traj, prior_state, post_state, temp)

        return model_loss, model_logs, prior_state, post_state

    def world_model_loss(self, traj, prior_state, post_state, temp):
        obs = traj[self.input_type]
        obs = obs / 255.0 - 0.5
        reward = traj["reward"]
        reward = self.r_transform(reward).float()

        post_state_trimed = {}
        for k, v in post_state.items():
            if k in ["stoch", "logits", "mean", "std"]:
                post_state_trimed[k] = v[:, 1:]
            else:
                post_state_trimed[k] = v

        rnn_feature = self.dynamic.get_feature(post_state_trimed, layer=self.reward_layer)
        seq_len = self.H

        image_pred_pdf = self.img_dec(rnn_feature)  # B, T-1, 3, 64, 64
        reward_pred_pdf = self.reward(rnn_feature)  # B, T-1, 1

        pred_pcont = self.pcont(rnn_feature)  # B, T, 1
        pcont_target = self.discount * (1.0 - traj["done"][:, 1:].float())  # B, T
        pcont_loss = (-(pred_pcont.log_prob((pcont_target.unsqueeze(2) > 0.5).float())).sum(-1)/ seq_len)  #
        pcont_loss = self.pcont_scale * pcont_loss.mean()
        discount_acc = ((pred_pcont.mean == pcont_target.unsqueeze(2)).float().squeeze(-1)).sum(-1) / seq_len
        discount_acc = discount_acc.mean()

        image_pred_loss = (-(image_pred_pdf.log_prob(obs[:, 1:])).sum(-1).float() / seq_len)  # B
        image_pred_loss = image_pred_loss.mean()
        mse_loss = (F.mse_loss(image_pred_pdf.mean, obs[:, 1:], reduction="none").flatten(start_dim=-3).sum(-1)).sum(-1) / seq_len
        mse_loss = mse_loss.mean()
        reward_pred_loss = (-(reward_pred_pdf.log_prob(reward[:, 1:].unsqueeze(2))).sum(-1) / seq_len)  # B
        reward_pred_loss = reward_pred_loss.mean()
        pred_reward = reward_pred_pdf.mean

        prior_dist = self.dynamic.get_dist(prior_state, temp)
        post_dist = self.dynamic.get_dist(post_state_trimed, temp)

        value_lhs = kl_divergence(post_dist, self.dynamic.get_dist(prior_state, temp, detach=True))  # B, T
        value_rhs = kl_divergence(self.dynamic.get_dist(post_state_trimed, temp, detach=True), prior_dist)
        value_lhs = value_lhs.sum(-1) / seq_len
        value_rhs = value_rhs.sum(-1) / seq_len
        loss_lhs = torch.maximum(value_lhs.mean(), value_lhs.new_ones(value_lhs.mean().shape) * self.free_nats)
        loss_rhs = torch.maximum(value_rhs.mean(), value_rhs.new_ones(value_rhs.mean().shape) * self.free_nats)

        kl_loss = (1.0 - self.kl_balance) * loss_lhs + self.kl_balance * loss_rhs
        kl_scale = self.kl_scale
        kl_loss = kl_scale * kl_loss

        model_loss = image_pred_loss + reward_pred_loss + kl_loss + pcont_loss

        post_dist = Independent(OneHotCategorical(logits=post_state_trimed["logits"]), 1)
        prior_dist = Independent(OneHotCategorical(logits=prior_state["logits"]), 1)

        video = self._get_video(image_pred_pdf, obs)

        logs = {
            "model_loss": model_loss.detach().item(),
            "model_kl_loss": kl_loss.detach().item(),
            "model_reward_logprob_loss": reward_pred_loss.detach().item(),
            "model_image_loss": image_pred_loss.detach().item(),
            "model_mse_loss": mse_loss.detach(),
            # "ACT_prior_state": {k: v.detach() for k, v in prior_state.items()},
            "ACT_prior_entropy": prior_dist.entropy().mean().detach().item(),
            # "ACT_post_state": {k: v.detach() for k, v in post_state.items()},
            "ACT_post_entropy": post_dist.entropy().mean().detach().item(),
            "ACT_gt_reward": wandb.Histogram(reward[:, 1:].detach().cpu().numpy()),
            # "dec_img": (image_pred_pdf.mean.detach() + 0.5),  # B, T, 3, 64, 64
            # "gt_img": obs[:, 1:] + 0.5,
            "reward_input": rnn_feature.detach(),
            "model_discount_logprob_loss": pcont_loss.detach().item(),
            "discount_acc": discount_acc.detach(),
            "pred_reward": pred_reward.detach().squeeze(-1),
            "pred_discount": pred_pcont.mean.detach().squeeze(-1),
            "hp_kl_scale": kl_scale,
        }

        return model_loss, logs

    def imagine_ahead(self, actor, post_state, traj, sample_len, temp):
        """
        post_state:
          mean: mean of q(s_t | h_t, o_t), (B*T, H)
          std: std of q(s_t | h_t, o_t), (B*T, H)
          stoch: s_t sampled from q(s_t | h_t, o_t), (B*T, H)
          deter: h_t, (B*T, H)
        """

        self.eval()
        self.requires_grad_(False)

        action = traj["action"]

        # randomly choose a state to start imagination
        min_idx = (self.H - 2)  # trimed the last step, at least imagine 2 steps for TD target
        perm = torch.randperm(min_idx, device=action.device)
        min_idx = perm[0] + 1

        pred_state = defaultdict(list)

        # pred_prior = {k: v.detach()[:, :min_idx] for k, v in post_state_trimed.items()}
        post_stoch = post_state["stoch"][:, :min_idx]
        action = action[:, 1 : min_idx + 1]
        imag_rnn_feat_list = []
        imag_action_list = []

        for t in range(self.sequence_length - min_idx):
            pred_prior = self.dynamic.infer_prior_stoch(post_stoch[:, -sample_len:], temp, action[:, -sample_len:])
            rnn_feature = self.dynamic.get_feature(pred_prior, layer=self.reward_layer)

            pred_action_pdf = actor(rnn_feature[:, -1:].detach())
            imag_action = pred_action_pdf.sample()
            imag_action = (imag_action + pred_action_pdf.mean - pred_action_pdf.mean.detach())  # straight through
            action = torch.cat([action, imag_action], dim=1)

            for k, v in pred_prior.items():
                pred_state[k].append(v[:, -1:])
            post_stoch = torch.cat([post_stoch, pred_prior["stoch"][:, -1:]], dim=1)

            imag_rnn_feat_list.append(rnn_feature[:, -1:])
            imag_action_list.append(imag_action)

        for k, v in pred_state.items():
            pred_state[k] = torch.cat(v, dim=1)
        actions = torch.cat(imag_action_list, dim=1)
        rnn_features = torch.cat(imag_rnn_feat_list, dim=1)

        reward = self.reward(rnn_features).mean
        discount = self.discount * self.pcont(rnn_features).mean

        return rnn_features, pred_state, actions, reward, discount, min_idx