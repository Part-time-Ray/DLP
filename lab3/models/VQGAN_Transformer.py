import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import torchvision.utils as vutils


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
        self.vqgan = self.vqgan.eval()

        # for param in self.vqgan.parameters():
        #     param.requires_grad = False
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        print(f"Loading checkpoint from {load_ckpt_path}")
        self.transformer.load_state_dict(torch.load(load_ckpt_path, weights_only=True))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path'], weights_only=True), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, _ = self.vqgan.encode(x)
        return codebook_mapping, codebook_indices.reshape(codebook_mapping.shape[0], -1)
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        def linear_gamma(r):
            return 1.0 - r

        def cosine_gamma(r):
            return 0.5 * (1 + math.cos(math.pi * r))

        def square_gamma(r):
            return 1 - (r) ** 2
        
        def stair_gamma(r):
            return 1 - (r // 0.1) * 0.1
        
        def god_gamma(r):
            return 0
        
        if mode == "linear":
            return linear_gamma
        elif mode == "cosine":
            return cosine_gamma
        elif mode == "square":
            return square_gamma
        elif mode == "stair":
            return stair_gamma
        elif mode == "god":
            return god_gamma
        else:
            raise NotImplementedError("Unsupported gamma mode: ", mode)

##TODO2 step1-3:            
    def forward(self, image):
        _, z_indices = self.encode_to_z(image)
        batch_size, num_tokens = z_indices.size()

        mask_ratio = 0.4 + torch.rand(1).item() * 0.1
        num_masked = int(mask_ratio * num_tokens)
        mask_token_id = self.mask_token_id
        mask = torch.ones(batch_size, num_tokens, device=image.device).long()
        for i in range(batch_size):
            indices = torch.randperm(num_tokens)[:num_masked]
            mask[i, indices] = 0
        mask_z_indices = mask * z_indices + (1 - mask) * mask_token_id
        logits = self.transformer(mask_z_indices)
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask, mask_num, ratio):
        ismask =  mask == True
        nomask = mask == False
        z_indices_mask = ismask * self.mask_token_id + (~ismask) * z_indices
        logits = self.transformer(z_indices_mask)
        probs = torch.softmax(logits[:, :, :-1], dim=-1)
        z_indices_predict_prob, z_indices_max = probs.max(dim=-1)
        z_indices_predict_prob[nomask] = torch.inf
        g = -torch.log(-torch.log(torch.rand_like(probs)))
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g.max(dim=-1)[0]

        num_to_mask = int(mask_num * ratio)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        threshold = sorted_confidence[:, num_to_mask].unsqueeze(-1)
        new_mask = (confidence < threshold).bool()
        # print("old mask number: ", mask.sum())
        # print("new mask number: ", new_mask.sum())
        return z_indices_max, new_mask
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        ##At the end of the decoding process, add back the original(non-masked) token values

    @torch.no_grad()
    def generate(self, z_indices, gt_image, loss):
        assert torch.min(z_indices) >= 0 and torch.max(z_indices) < self.mask_token_id, f"z_indices out of range: min {z_indices.min()}, max {z_indices.max()}"
        
        shape = (1, 16, 16, 256)
        z_q = self.vqgan.codebook.embedding(z_indices).view(shape)
        z_q = z_q.permute(0, 3, 1, 2)
        
        decoded_img = self.vqgan.decode(z_q)
        # print(f"decoded_img min: {decoded_img.min()}, max: {decoded_img.max()}")
        
        decoded_img = torch.clamp(decoded_img, -1, 1)
        decoded_img = (decoded_img + 1) / 2
        
        mean = torch.tensor([0.4868, 0.4341, 0.3844], device='cuda').view(3, 1, 1)
        std = torch.tensor([0.2620, 0.2527, 0.2543], device='cuda').view(3, 1, 1)
        dec_img_ori = (decoded_img[0] * std) + mean
        # print(f"dec_img_ori min: {dec_img_ori.min()}, max: {dec_img_ori.max()}")
        
        if isinstance(gt_image, torch.Tensor):
            gt_img = (gt_image + 1) / 2
            gt_img = (gt_img * std) + mean
        else:
            gt_img = gt_image
        
        combined_img = torch.cat([gt_img, dec_img_ori], dim=2)
        vutils.save_image(combined_img, f"test.png", nrow=2, normalize=False)
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
