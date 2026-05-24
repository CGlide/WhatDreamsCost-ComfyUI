# --- START OF FILE latent_slice.py ---

import torch

class CleanLatentSlice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "length": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1, "tooltip": "The target length in latent frames."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "slice_latent"
    CATEGORY = "WhatDreamsCost"
    DESCRIPTION = "Safely slices a video latent to a specific length. Uses torch.narrow to bypass PyTorch NestedTensor slicing bugs."

    def slice_latent(self, latent, length):
        new_latent = latent.copy()
        
        def safe_slice(tensor, target_len):
            dims = tensor.ndim if hasattr(tensor, "ndim") else len(tensor.shape)
            
            try:
                # Bypasses NestedTensor slicing bugs
                if dims == 5:
                    return torch.narrow(tensor, 2, 0, target_len)
                elif dims == 4:
                    return torch.narrow(tensor, 0, 0, target_len)
                elif dims == 3:
                    return torch.narrow(tensor, 0, 0, target_len)
            except Exception as e:
                # Extreme fallback if narrow fails
                if dims == 5:
                    return tensor[:, :, :target_len]
                elif dims == 4:
                    return tensor[:target_len]
                elif dims == 3:
                    return tensor[:target_len]
                    
            return tensor

        if "samples" in new_latent:
            new_latent["samples"] = safe_slice(new_latent["samples"], length)

        if "noise_mask" in new_latent:
            new_latent["noise_mask"] = safe_slice(new_latent["noise_mask"], length)

        return (new_latent,)