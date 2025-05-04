import torch
import os
from diffusers import DDPMScheduler

def inference(model, beta_scheduler, label, image_size=(3, 64, 64), device='cuda'):
    model.eval()
    with torch.no_grad():
        batch_size = label.shape[0]
        x_t = torch.randn(batch_size, *image_size, device=device)
        for t in range(beta_scheduler.num_diffusion_timesteps - 1, -1, -1):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = model(x_t, label, t_tensor)
            x_t = beta_scheduler.reverse(x_t, t_tensor, pred_noise)
        # print("Before clamp:", torch.max(x_t), torch.min(x_t))
        x_t = x_t.clamp(-1, 1)
        # print("After clamp:", torch.max(x_t), torch.min(x_t))
        return x_t

class BetaScheduler:
    def __init__(self, num_diffusion_timesteps=2000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.device = device
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.beta_schedular = DDPMScheduler(num_train_timesteps=self.num_diffusion_timesteps, beta_start=beta_start, beta_end=beta_end, beta_schedule='linear') # squaredcos_cap_v2


    def make_noise(self, x_start, t):
        x_start = x_start.to(self.device)
        noise = torch.randn_like(x_start, device=self.device)
        t = torch.tensor([t], device=self.device) if isinstance(t, int) else t
        return self.beta_schedular.add_noise(x_start, noise, t), noise

    def reverse(self, x_t, t, noise):
        # all value in t is same
        assert all(t == t[0])
        return self.beta_schedular.step(noise, t[0].cpu(), x_t).prev_sample

# class BetaScheduler:
#     def __init__(self, num_diffusion_timesteps=2000, beta_start=1e-4, beta_end=0.02, device='cuda'):
#         self.device = device
#         self.num_diffusion_timesteps = num_diffusion_timesteps
#         self.betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, device=self.device)
#         self.alphas = 1. - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
#         self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device),self.alphas_cumprod[:-1]])
#         self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
#         self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
#         self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))

#     def make_noise(self, x_start, t):
#         x_start = x_start.to(self.device)
#         noise = torch.randn_like(x_start, device=self.device)
#         t = torch.tensor([t], device=self.device) if isinstance(t, int) else t
#         sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, *[1] * (x_start.dim() - 1))
#         sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *[1] * (x_start.dim() - 1))
#         return (sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise), noise

#     def reverse(self, x_t, t, noise):
#         t = torch.tensor([t], device=self.device) if isinstance(t, int) else t
#         betas_t = self.betas[t].view(-1, 1, 1, 1)
#         alphas_t = self.alphas[t].view(-1, 1, 1, 1)
#         alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
#         posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
#         mu = (1. / torch.sqrt(alphas_t)) * (x_t - (betas_t / torch.sqrt(1. - alphas_cumprod_t)) * noise)
#         sigma = torch.sqrt(posterior_variance_t)
#         scale = (1 - t / self.num_diffusion_timesteps).view(-1, 1, 1, 1)
#         z = torch.randn_like(x_t, device=self.device) * (t > 0).float().view(-1, 1, 1, 1) * scale
#         return mu + sigma * z


class SMARTSave:
    def __init__(self, postfix = ''):
        self.save_path = 'result'
        os.makedirs(self.save_path, exist_ok=True)
        self.best_loss = float('inf')
        self.postfix = postfix
    def __call__(self, model, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(model.state_dict(), os.path.join(self.save_path, f'best_model{('_' if self.postfix != '' else '') + self.postfix}.pt'))
            print(f"=====Model saved with loss: {loss:.4f}=====")


if __name__ == '__main__':
    beta_scheduler = BetaScheduler(num_diffusion_timesteps=100, beta_start=1e-4, beta_end=0.02)
    for t in beta_scheduler.beta_schedular.timesteps:
        print(t)