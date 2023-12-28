'''
This code from the following repository: https://github.com/LeapLabTHU/Agent-Attention

@article{han2023agent,
  title={Agent Attention: On the Integration of Softmax and Linear Attention},
  author={Han, Dongchen and Ye, Tianzhu and Han, Yizeng and Xia, Zhuofan and Song, Shiji and Huang, Gao},
  journal={arXiv preprint arXiv:2312.08874},
  year={2023}
}
'''
import torch
import math
from typing import Type, Dict, Any, Tuple, Callable

from . import merge
from .utils import isinstance_str, init_generator
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction


def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args["ratio"])
        agent_r = int(x.shape[1] * args["agent_ratio"])

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])

        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, agent_r,
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing_2, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"] else (merge.do_nothing_2, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing_2, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"] else (merge.do_nothing_2, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good







def make_tome_block(block_class: Type[torch.nn.Module], old_forward) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies AgentSD and ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class
        _old_forward = old_forward

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            y = self.norm1(x)
            feature, agent = m_a(y)
            x = u_a(self.attn1(feature, agent=agent, context=context if self.disable_self_attn else None)) + x
            y = self.norm2(x)
            feature, agent = m_c(y)
            x = u_c(self.attn2(feature, agent=agent, context=context)) + x
            y = self.norm3(x)
            feature, _ = m_m(y)
            x = u_m(self.ff(feature)) + x

            return x
    
    return ToMeBlock


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def make_agent_attn(block_class: Type[torch.nn.Module], k_scale2, k_shortcut, attn_precision=None) -> Type[torch.nn.Module]:
    """
    This patch applies AgentSD to the forward function of the block.
    """

    class AgentAttention(block_class):
        # Save for unpatching later
        _parent = block_class

        def set_new_params(self):
            self.k_scale2 = k_scale2
            self.k_shortcut = k_shortcut
            self.attn_precision = attn_precision

        def forward(self, x, agent=None, context=None, mask=None, *args, **kwargs):
            if agent is not None:
                if agent.shape[1] * 2 < x.shape[1]:
                    k_scale2 = self.k_scale2
                    k_shortcut = self.k_shortcut

                    h = self.heads

                    q = self.to_q(x)
                    context = default(context, x)
                    k = self.to_k(context)
                    v = self.to_v(context)
                    agent = self.to_q(agent)

                    q, k, v, agent = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v, agent))
                    if exists(mask):
                        print('Mask not supported yet!')

                    # force cast to fp32 to avoid overflowing
                    if self.attn_precision == "fp32":
                        with torch.autocast(enabled=False, device_type='cuda'):
                            agent, k = agent.float(), k.float()
                            sim1 = einsum('b i d, b j d -> b i j', agent, k) * self.scale
                        del k
                    else:
                        sim1 = einsum('b i d, b j d -> b i j', agent, k) * self.scale

                    # attention, what we cannot get enough of
                    attn1 = sim1.softmax(dim=-1)
                    agent_feature = einsum('b i j, b j d -> b i d', attn1, v)

                    # force cast to fp32 to avoid overflowing
                    if self.attn_precision == "fp32":
                        with torch.autocast(enabled=False, device_type='cuda'):
                            q = q.float()
                            sim2 = einsum('b i d, b j d -> b i j', q, agent) * self.scale ** k_scale2
                        del q, agent
                    else:
                        sim2 = einsum('b i d, b j d -> b i j', q, agent) * self.scale ** k_scale2

                    # attention, what we cannot get enough of
                    attn2 = sim2.softmax(dim=-1)
                    out = einsum('b i j, b j d -> b i d', attn2, agent_feature)

                    out = out * 1.0 + v * k_shortcut

                    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                    return self.to_out(out)

            h = self.heads

            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)

    return AgentAttention






def make_diffusers_tome_block(block_class: Type[torch.nn.Module], old_forward) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class
        _old_forward = old_forward

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # (1) ToMe
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return ToMeBlock






def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))








def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        agent_ratio: float = 0.8,
        k_scale2=0.3,
        k_shortcut=0.075,
        attn_precision=None,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False):
    """
    Patches a stable diffusion model with AgentSD.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
     - agent_ratio: The ratio of tokens to merge when producing agent tokens.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply AgentSD to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - k_scale2: The scale used for the second attention is head_dim ** (-0.5 * k_scale2)
     - k_shortcut: The ratio used in O = sigma(QA^T) sigma(AK^T) V + k * V.
     - attn_precision: Set attn_precision="fp32" to avoid numerical instabilities on SD v2.1 model.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, "unet") else model

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "agent_ratio": agent_ratio,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp
        }
    }
    hook_tome_model(diffusion_model)

    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            module._old_class_= [module.__class__]
            _old_forward = None
            make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            module.__class__ = make_tome_block_fn(module.__class__, _old_forward)
            module._tome_info = diffusion_model._tome_info
            module._old_attn1 = [module.attn1.__class__]
            module._old_attn2 = [module.attn2.__class__]
            module.attn1.__class__ = make_agent_attn(module.attn1.__class__, k_scale2=k_scale2, k_shortcut=k_shortcut, attn_precision=attn_precision)
            module.attn2.__class__ = make_agent_attn(module.attn2.__class__, k_scale2=k_scale2, k_shortcut=k_shortcut, attn_precision=attn_precision)
            module.attn1.set_new_params()
            module.attn2.set_new_params()

            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn") and not is_diffusers:
                module.disable_self_attn = False

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model





def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a AgentSD Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            if hasattr(module, "_old__class__"):
                module.__class__ = module._old__class__[0]
            else:
                module.__class__ = module._parent
        if hasattr(module, "_old_attn1"):
            module.attn1.__class__ = module._old_attn1[0]
        if hasattr(module, "_old_attn2"):
            module.attn2.__class__ = module._old_attn2[0]
    
    return model
