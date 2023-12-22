from collections import defaultdict
from dataclasses import dataclass
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions import Independent
import torch
import torch.nn as nn
import wandb
import pdb

from .world_model import TransformerWorldModel, DenseDecoder, ActionDecoder


class TransDreamer(nn.Module):
    def __init__(self, world_model: TransformerWorldModel, actor: ActionDecoder, value: DenseDecoder, slow_value: DenseDecoder, 
                 batch_length: int = 50, lambda_: int = 0.95):
        super().__init__()

        self.world_model = world_model
        self.actor = actor
        self.value = value
        self.slow_value = slow_value

        # self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
        # self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
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
        # self.aggregator = cfg.arch.actor.aggregator
        # if self.aggregator == "attn":
        #     dense_input_size = dense_input_size + self.d_model

        self.action_dist = self.actor.dist
        self.reward_layer = self.world_model.reward_layer

        self.batch_length = batch_length
        self.lambda_ = lambda_

        self.slow_update = 0

    def _setup_loss_config(self, config) -> None:
        self.actor_loss_type = config.actor_loss_type
        self.ent_scale = config.ent_scale
        self.world_model.pcont_scale = config.pcont_scale
        self.world_model.kl_scale = config.kl_scale
        self.world_model.kl_balance = config.kl_balance
        self.world_model.free_nats = config.free_nats

    def forward(self):
        raise NotImplementedError

    def compute_world_model_loss(self, traj, temp):
        model_loss, logs, prior_state, post_state = self.world_model.compute_loss(traj, temp)
        return model_loss, logs, post_state
    
    def compute_actor_and_value_loss(self, traj, post_state, temp):
        return self.actor_and_value_loss(traj, post_state, temp)

    def actor_and_value_loss(self, traj, post_state, temp):
        self.update_slow_target()
        self.value.eval()
        self.value.requires_grad_(False)

        (imagine_feat, imagine_state, imagine_action, imagine_reward, imagine_disc, imagine_idx) = self.world_model.imagine_ahead(
            self.actor, post_state, traj, self.batch_length - 1, temp
        )

        target, weights = self.compute_target(
            imagine_feat, imagine_reward, imagine_disc
        )  # B*T, H-1, 1

        slice_idx = -1

        actor_dist = self.actor(imagine_feat.detach())  # B*T, H
        if self.action_dist == "onehot":
            indices = imagine_action.max(-1)[1]
            actor_logprob = actor_dist._categorical.log_prob(indices)
        else:
            actor_logprob = actor_dist.log_prob(imagine_action)

        if self.actor_loss_type == "dynamic":
            actor_loss = target

        elif self.actor_loss_type == "reinforce":
            baseline = self.value(imagine_feat[:, :slice_idx]).mean
            advantage = (target - baseline).detach()
            actor_loss = actor_logprob[:, :slice_idx].unsqueeze(2) * advantage

        elif self.actor_loss_type == "both":
            raise NotImplementedError

        actor_entropy = actor_dist.entropy()
        ent_scale = self.ent_scale
        actor_loss = ent_scale * actor_entropy[:, :slice_idx].unsqueeze(2) + actor_loss
        actor_loss = -(weights[:, :slice_idx] * actor_loss).mean()

        self.value.train()
        self.value.requires_grad_(True)
        imagine_value_dist = self.value(imagine_feat[:, :slice_idx].detach())
        log_prob = -imagine_value_dist.log_prob(target.detach())
        value_loss = weights[:, :slice_idx] * log_prob.unsqueeze(2)
        value_loss = value_loss.mean()
        imagine_value = imagine_value_dist.mean

        imagine_dist = Independent(OneHotCategorical(logits=imagine_state["logits"]), 1)
        if self.action_dist == "onehot":
            action_samples = imagine_action.argmax(dim=-1).float().detach()
        else:
            action_samples = imagine_action.detach()
        logs = {
            "value_loss": value_loss.detach().item(),
            "actor_loss": actor_loss.detach().item(),
            # "ACT_imag_state": {k: v.detach() for k, v in imagine_state.items()},
            "ACT_imag_entropy": imagine_dist.entropy().mean().detach().item(),
            "ACT_actor_entropy": actor_entropy.mean().item(),
            "ACT_action_prob": wandb.Histogram(actor_dist.mean.detach().cpu().numpy()), #"ACT_action_prob": actor_dist.mean.detach(),
            "ACT_actor_logprob": actor_logprob.mean().item(),
            "ACT_action_samples": wandb.Histogram(action_samples.cpu().numpy()), # action_samples,
            "ACT_image_discount": wandb.Histogram(imagine_disc.detach().cpu().numpy()), # imagine_disc.detach(),
            "ACT_imag_value": wandb.Histogram(imagine_value.squeeze(-1).detach().cpu().numpy()), # imagine_value.squeeze(-1).detach(),
            "ACT_actor_target": wandb.Histogram(target.mean().detach().cpu().numpy()), # target.mean().detach(),
            "ACT_target": wandb.Histogram(target.squeeze(-1).detach().cpu().numpy()), # target.squeeze(-1).detach(),
            "ACT_actor_baseline": wandb.Histogram(baseline.mean().detach().cpu().numpy()), # baseline.mean().detach(),
            "ACT_imag_reward": wandb.Histogram(imagine_reward.detach().cpu().numpy()), # imagine_reward.detach(),
            "ACT_imagine_idx": imagine_idx.float(),
        }

        return actor_loss, value_loss, logs

    def compute_target(self, imag_feat, reward, discount_arr):
        self.slow_value.eval()
        self.slow_value.requires_grad_(False)

        value = self.slow_value(imag_feat).mean  # B*T, H, 1

        # v_t = R_{t+1} + v_{t+1}
        target = self.lambda_return(
            reward[:, 1:], value[:, :-1], discount_arr[:, 1:], value[:, -1], self.lambda_,
        )

        discount_arr = torch.cat([torch.ones_like(discount_arr[:, :1]), discount_arr[:, :-1]], dim=1)
        weights = torch.cumprod(discount_arr, 1).detach()  # B, T 1
        return target, weights

    def policy(self, batch, gradient_step, temp, state=None, training=True, context_len=49):
        """

        :param obs: B, C, H, W
        :param action: B, T, C
        :param gradient_step:
        :param state: B, T, C
        :param training:
        :param prior:
        :return:
        """
        prev_obs = batch["prev_image"]
        obs = batch["next_image"]
        action = batch["action_list"]

        obs = obs.unsqueeze(1) / 255.0 - 0.5  # B, T, C, H, W
        obs_emb = self.world_model.dynamic.img_enc(obs)  # B, T, C
        post = self.world_model.dynamic.infer_post_stoch(obs_emb, temp, action=None)  # B, T, N, C

        if state is None:
            state = post
            prev_obs = prev_obs.unsqueeze(1) / 255.0 - 0.5  # B, T, C, H, W
            prev_obs_emb = self.world_model.dynamic.img_enc(prev_obs)  # B, T, C
            prev_post = self.world_model.dynamic.infer_post_stoch(prev_obs_emb, temp, action=None)  # B, T, N, C

            for k, v in post.items():
                state[k] = torch.cat([prev_post[k], v], dim=1)
            s_t = state["stoch"]

        else:
            s_t = torch.cat([state["stoch"], post["stoch"][:, -1:]], dim=1)[:, -context_len:]
            for k, v in post.items():
                state[k] = torch.cat([state[k], v], dim=1)[:, -context_len:]

        pred_prior = self.world_model.dynamic.infer_prior_stoch(s_t[:, :-1], temp, action)

        post_state_trimed = {}
        for k, v in state.items():
            if k in ["stoch", "logits", "pos", "mean", "std"]:
                post_state_trimed[k] = v[:, 1:]
            else:
                post_state_trimed[k] = v
        post_state_trimed["deter"] = pred_prior["deter"]
        post_state_trimed["o_t"] = pred_prior["o_t"]

        rnn_feature = self.world_model.dynamic.get_feature(post_state_trimed, layer=self.reward_layer)
        pred_action_pdf = self.actor(rnn_feature[:, -1:].detach())

        if training:
            pred_action = pred_action_pdf.sample()  # B, 1, C
        else:
            if self.action_dist == "onehot":
                pred_action = pred_action_pdf.mean
                index = pred_action.argmax(dim=-1)[0]
                pred_action = torch.zeros_like(pred_action)
                pred_action[..., index] = 1
            else:
                pred_action = pred_action_pdf.mode

        action = torch.cat([action, pred_action], dim=1)[:, -(context_len - 1):]  # B, T, C

        return action, state

    def lambda_return(self, imagine_reward, imagine_value, discount, bootstrap, lambda_):
        """
        https://github.com/juliusfrost/dreamer-pytorch/blob/47bd509ab5cffa95ec613fd788d7ae1fe664ecd5/dreamer/algos/dreamer_algo.py
        """
        # Setting lambda=1 gives a discounted Monte Carlo return.
        # Setting lambda=0 gives a fixed 1-step return.
        next_values = torch.cat([imagine_value[:, 1:], bootstrap[:, None]], 1)
        target = imagine_reward + discount * next_values * (1 - lambda_)
        timesteps = list(range(imagine_reward.shape[1] - 1, -1, -1))

        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:
            inp = target[:, t]
            discount_factor = discount[:, t]

            accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
            outputs.append(accumulated_reward)

        returns = torch.flip(torch.stack(outputs, dim=1), [1])
        return returns

    @torch.no_grad()
    def update_slow_target(self):
        self.slow_value.load_state_dict(self.value.state_dict())
        self.slow_update += 1

    # def write_logs(self, logs, traj, global_step, writer, tag="train", min_idx=None):
    #     rec_img = logs["dec_img"]
    #     gt_img = logs["gt_img"]  # B, {1:T}, C, H, W

    #     writer.add_video("train/rec - gt", torch.cat([gt_img[:4], rec_img[:4]], dim=-2).clamp(0.0, 1.0).cpu(), global_step=global_step,)

    #     for k, v in logs.items():
    #         if "loss" in k:
    #             writer.add_scalar(tag + "_loss/" + k, v, global_step=global_step)
    #         if "grad_norm" in k:
    #             writer.add_scalar(tag + "_grad_norm/" + k, v, global_step=global_step)
    #         if "hp" in k:
    #             writer.add_scalar(tag + "_hp/" + k, v, global_step=global_step)
    #         if "ACT" in k:
    #             if isinstance(v, dict):
    #                 for kk, vv in v.items():
    #                     if isinstance(vv, torch.Tensor):
    #                         writer.add_histogram(tag + "_ACT/" + k + "-" + kk, vv, global_step=global_step)
    #                         writer.add_scalar(tag + "_mean_ACT/" + k + "-" + kk, vv.mean(), global_step=global_step)
    #                     if isinstance(vv, float):
    #                         writer.add_scalar(tag + "_ACT/" + k + "-" + kk, vv, global_step=global_step)
    #             else:
    #                 if isinstance(v, torch.Tensor):
    #                     writer.add_histogram(tag + "_ACT/" + k, v, global_step=global_step)
    #                     writer.add_scalar(tag + "_mean_ACT/" + k, v.mean(), global_step=global_step)
    #                 if isinstance(v, float):
    #                     writer.add_scalar(tag + "_ACT/" + k, v, global_step=global_step)
    #         if "imag_value" in k:
    #             writer.add_scalar(tag + "_values/" + k, v.mean(), global_step=global_step)
    #             writer.add_histogram(tag + "_ACT/" + k, v, global_step=global_step)
    #         if "actor_target" in k:
    #             writer.add_scalar(tag + "actor_target/" + k, v, global_step=global_step)