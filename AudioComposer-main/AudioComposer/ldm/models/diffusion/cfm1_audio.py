from pytorch_memlab import LineProfiler,profile
import torch
import torch.nn as nn
from torchdyn.core import NeuralODE
from ldm.models.diffusion.ddpm_audio import LatentDiffusion_audio, disabled_train

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


class CFM(LatentDiffusion_audio):

    def __init__(self, **kwargs):

        super(CFM, self).__init__(**kwargs)
        self.sigma_min = 1e-4
        
    @torch.no_grad()
    def sample(self, cond, batch_size=16, timesteps=None, shape=None, x_latent=None, t_start=None, **kwargs):
        if shape is None:
            if self.channels > 0:
                shape = (batch_size, self.channels, self.mel_dim, self.mel_length)
            else:
                shape = (batch_size, self.mel_dim, self.mel_length)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        neural_ode = NeuralODE(self.ode_wrapper(cond), solver='euler', sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps)
        if t_start is not None:
            t_span = t_span[t_start:]

        x0 = torch.randn(shape, device=self.device) if x_latent is None else x_latent
        eval_points, traj = neural_ode(x0, t_span)

        return traj[-1], traj

    def ode_wrapper(self, cond):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper(self, cond)

    @torch.no_grad()
    def sample_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning, batch_size=16, timesteps=None, shape=None, x_latent=None, t_start=None, **kwargs):
        if shape is None:
            if self.channels > 0:
                shape = (batch_size, self.channels, self.mel_dim, self.mel_length)
            else:
                shape = (batch_size, self.mel_dim, self.mel_length)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        neural_ode = NeuralODE(self.ode_wrapper_cfg(cond, unconditional_guidance_scale, unconditional_conditioning), solver='euler', sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps)

        if t_start is not None:
            t_span = t_span[t_start:]

        x0 = torch.randn(shape, device=self.device) if x_latent is None else x_latent
        eval_points, traj = neural_ode(x0, t_span)

        return traj[-1], traj

    def ode_wrapper_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning)


class Wrapper(nn.Module):
    def __init__(self, net, cond):
        super(Wrapper, self).__init__()
        self.net = net
        self.cond = cond

    def forward(self, t, x, args):
        t = torch.tensor([t * 1000] * x.shape[0], device=t.device).long()
        return self.net.apply_model(x, t, self.cond)


class Wrapper_cfg(nn.Module):

    def __init__(self, net, cond, unconditional_guidance_scale, unconditional_conditioning):
        super(Wrapper_cfg, self).__init__()
        self.net = net
        self.cond = cond
        self.unconditional_conditioning = unconditional_conditioning
        self.unconditional_guidance_scale = unconditional_guidance_scale

    def forward(self, t, x, args):
        x_in = torch.cat([x] * 2)
        t = torch.tensor([t * 1000] * x.shape[0], device=t.device).long()
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([self.unconditional_conditioning, self.cond])  # c/uc shape [b,seq_len=77,dim=1024],c_in shape [b*2,seq_len,dim]
        e_t_uncond, e_t = self.net.apply_model(x_in, t_in, c_in).chunk(2)
        e_t = e_t_uncond + self.unconditional_guidance_scale * (e_t - e_t_uncond)
        return e_t