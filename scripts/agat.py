import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import numpy as np
from collections import OrderedDict
from typing import Union
import agentsd

from modules import script_callbacks, rng, shared
from modules.script_callbacks import CFGDenoiserParams

import torch

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

"""
An implementation of Agent Attention for stable-diffusion-webui: https://github.com/LeapLabTHU/Agent-Attention

@misc{han2023agent,
      title={Agent Attention: On the Integration of Softmax and Linear Attention}, 
      author={Dongchen Han and Tianzhu Ye and Yizeng Han and Zhuofan Xia and Shiji Song and Gao Huang},
      year={2023},
      eprint={2312.08874},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-agentattention

"""

class PassSettings:
        def __init__(self, sx, sy, ratio, agent_ratio):
                self.sx = sx
                self.sy = sy
                self.ratio = ratio
                self.agent_ratio = agent_ratio

class AgentAttentionExtensionScript(scripts.Script):
        # Extension title in menu UI
        def title(self):
                return "Agent Attention"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def ui(self, is_img2img):
                with gr.Accordion('AgentAttention', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='aa_active')
                        use_sp = gr.Checkbox(value=True, default=True, label="Use Second Pass", elem_id = 'aa_use_sp')
                        sp_step = gr.Slider(value = 20, minimum = 0, maximum = 100, step = 1, label="Second Pass Step", elem_id = 'aa_sp_step')
                        with gr.Accordion('First Pass', open=False):
                                sx = gr.Slider(value = 0.2, minimum = 0.0, maximum = 1.0, step = 0.05, label="sx", elem_id = 'aa_sx')
                                sy = gr.Slider(value = 0.2, minimum = 0.0, maximum = 1.0, step = 0.05, label="sy", elem_id = 'aa_sy')
                                ratio = gr.Slider(value = 0.4, minimum = 0.0, maximum = 1.0, step = 0.05, label="Ratio", elem_id = 'aa_ratio')
                                agent_ratio = gr.Slider(value = 0.95, minimum = 0.0, maximum = 1.0, step = 0.05, label="Agent Ratio", elem_id = 'aa_agent_ratio')
                        with gr.Accordion('Second Pass', open=False):
                                sp_sx = gr.Slider(value = 0.2, minimum = 0.0, maximum = 1.0, step = 0.05, label="sx", elem_id = 'aa_sp_sx')
                                sp_sy = gr.Slider(value = 0.2, minimum = 0.0, maximum = 1.0, step = 0.05, label="sy", elem_id = 'aa_sp_sy')
                                sp_ratio = gr.Slider(value = 0.4, minimum = 0.0, maximum = 1.0, step = 0.05, label="Ratio", elem_id = 'aa_sp_ratio')
                                sp_agent_ratio = gr.Slider(value = 0.95, minimum = 0.0, maximum = 1.0, step = 0.05, label="Agent Ratio", elem_id = 'aa_sp_agent_ratio')
                active.do_not_save_to_config = True
                use_sp.do_not_save_to_config = True
                sp_step.do_not_save_to_config = True
                sx.do_not_save_to_config = True
                sy.do_not_save_to_config = True
                ratio.do_not_save_to_config = True
                agent_ratio.do_not_save_to_config = True
                sp_sx.do_not_save_to_config = True
                sp_sy.do_not_save_to_config = True
                sp_ratio.do_not_save_to_config = True
                sp_agent_ratio.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='AgAt Active' in d)),
                        (use_sp, 'AgAt Use Second Pass'),
                        (sp_step, 'AgAt Second Pass Step'),
                        (sx, 'AgAt First Pass sx'),
                        (sy, 'AgAt First Pass sy'),
                        (ratio, 'AgAt First Pass Ratio'),
                        (agent_ratio, 'AgAt First Pass Agent Ratio'),
                        (sp_sx, 'AgAt Second Pass sx'),
                        (sp_sy, 'AgAt Second Pass sy'),
                        (sp_ratio, 'AgAt Second Pass Ratio'),
                        (sp_agent_ratio, 'AgAt Second Pass Agent Ratio'),
                ]
                return [active, use_sp, sp_step, sx, sy, ratio, agent_ratio, sp_sx, sp_sy, sp_ratio, sp_agent_ratio]

        def before_process_batch(self, p, active, use_sp, sp_step, sx, sy, ratio, agent_ratio, sp_sx, sp_sy, sp_ratio, sp_agent_ratio, *args, **kwargs):
                active = getattr(p, "aa_active", active)
                if active is False:
                        return
                use_sp = getattr(p, "aa_use_sp", use_sp)
                sp_step = getattr(p, "aa_sp_step", sp_step)
                sx = getattr(p, "aa_sx", sx)
                sy = getattr(p, "aa_sy", sy)
                ratio = getattr(p, "aa_ratio", ratio)
                agent_ratio = getattr(p, "aa_agent_ratio", agent_ratio)
                sp_sx = getattr(p, "aa_sp_sx", sp_sx)
                sp_sy = getattr(p, "aa_sp_sy", sp_sy)
                sp_ratio = getattr(p, "aa_sp_ratio", sp_ratio)
                sp_agent_ratio = getattr(p, "aa_sp_agent_ratio", sp_agent_ratio)

                p.extra_generation_params = {
                        "AgentAttention Active": active,
                        "AgentAttention Use Second Pass": use_sp,
                        "AgentAttention Second Pass Step": sp_step,
                        "AgentAttention First Pass sx": sx,
                        "AgentAttention First Pass sy": sy,
                        "AgentAttention First Pass Ratio": ratio,
                        "AgentAttention First Pass Agent Ratio": agent_ratio,
                        "AgentAttention Second Pass sx": sp_sx,
                        "AgentAttention Second Pass sy": sp_sy,
                        "AgentAttention Second Pass Ratio": sp_ratio,
                        "AgentAttention Second Pass Agent Ratio": sp_agent_ratio,
                }
                self.create_hook(p, active, use_sp, sp_step, sx, sy, ratio, agent_ratio, sp_sx, sp_sy, sp_ratio, sp_agent_ratio)
        
        def create_hook(self, p, active, use_sp, sp_step, sx, sy, ratio, agent_ratio, sp_sx, sp_sy, sp_ratio, sp_agent_ratio):
                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, active=active, use_sp=use_sp, sp_step=sp_step, sx=sx, sy=sy, ratio=ratio, agent_ratio=agent_ratio, sp_sx=sp_sx, sp_sy=sp_sy, sp_ratio=sp_ratio, sp_agent_ratio=sp_agent_ratio)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)

        def postprocess_batch(self, p, active, use_sp, sp_step, sx, sy, ratio, agent_ratio, sp_sx, sp_sy, sp_ratio, sp_agent_ratio, *args, **kwargs):
                self.unhook_callbacks()

        def unhook_callbacks(self):
                logger.debug('Unhooked callbacks')
                self.remove_patch()
                script_callbacks.remove_current_script_callbacks()

        def apply_patch(self, sx=2, sy=2, ratio=0.4, agent_ratio=0.95):
                logger.debug(f'Applied patch with sx: {sx}, sy: {sy}, ratio: {ratio}, agent_ratio: {agent_ratio}')
                agentsd.apply_patch(shared.sd_model, sx=sx, sy=sy, ratio=ratio, agent_ratio=agent_ratio)

        def remove_patch(self):
                logger.debug('Removed patch')
                agentsd.remove_patch(shared.sd_model)

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, active, use_sp, sp_step, sx, sy, ratio, agent_ratio, sp_sx, sp_sy, sp_ratio, sp_agent_ratio, *args, **kwargs):
                sampling_step = params.sampling_step

                if sampling_step == 0:
                        self.apply_patch(sx=sx, sy=sy, ratio=ratio, agent_ratio=agent_ratio)

                if sampling_step == sp_step:
                        self.remove_patch()
                        if use_sp:
                                self.apply_patch(shared.sd_model, sx=sp_sx, sy=sp_sy, ratio = sp_ratio, agent_ratio=sp_agent_ratio)


# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def aa_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
    return fun

def aa_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "aa_active"):
                setattr(p, "aa_active", True)
        setattr(p, field, x)

    return fun

def make_axis_options():
        xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
        extra_axis_options = {
                xyz_grid.AxisOption("[AgentAttention] Active", str, aa_apply_override('aa_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[AgentAttention] Use Second Pass", str, aa_apply_override('aa_use_sp', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[AgentAttention] Second Pass Step", int, aa_apply_field("aa_sp_step")),
                xyz_grid.AxisOption("[AgentAttention] First Pass sx", float, aa_apply_field("aa_sx")),
                xyz_grid.AxisOption("[AgentAttention] First Pass sy", float, aa_apply_field("aa_sy")),
                xyz_grid.AxisOption("[AgentAttention] First Pass Ratio", float, aa_apply_field("aa_ratio")),
                xyz_grid.AxisOption("[AgentAttention] First Pass Agent Ratio", float, aa_apply_field("aa_agent_ratio")),
                xyz_grid.AxisOption("[AgentAttention] Second Pass sx", float, aa_apply_field("aa_sp_sx")),
                xyz_grid.AxisOption("[AgentAttention] Second Pass sy", float, aa_apply_field("aa_sp_sy")),
                xyz_grid.AxisOption("[AgentAttention] Second Pass Ratio", float, aa_apply_field("aa_sp_ratio")),
                xyz_grid.AxisOption("[AgentAttention] Second Pass Agent Ratio", float, aa_apply_field("aa_sp_agent_ratio")),
        }
        if not any("[AgentAttention]" in x.label for x in xyz_grid.axis_options):
                xyz_grid.axis_options.extend(extra_axis_options)

def callback_before_ui():
        try:
                make_axis_options()
        except:
                logger.exception("AgentAttention: Error while making axis options")

script_callbacks.on_before_ui(callback_before_ui)
