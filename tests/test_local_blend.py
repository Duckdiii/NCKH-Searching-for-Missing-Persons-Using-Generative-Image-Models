import numpy as np
import pytest
import torch
from transformers import CLIPTokenizer

from src.fading.editing import get_word_inds, local_blend
from src.utils.prompts import build_prompt_tau





def test_get_word_inds():
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    prompt = "photo of a 30 year old man, subtle fine lines, mature skin"
    inds = get_word_inds(prompt, "man", tokenizer)
    assert len(inds) > 0
    # Decoded token should contain 'man'
    decoded = tokenizer.decode([tokenizer.encode(prompt)[inds[0]]]).strip().lower()
    assert "man" in decoded


def test_local_blend_shapes_and_blending():
    recon_latent = torch.zeros(1, 4, 32, 32, dtype=torch.float32)
    edit_latent = torch.ones(1, 4, 32, 32, dtype=torch.float32)

    # 2 fake cross-attention maps for recon and edit: [heads=8, spatial_tokens=256, seq_len=77]
    # Simulate high attention in the center for subject tokens
    cross_recon = [torch.zeros(8, 256, 77, dtype=torch.float32)]
    cross_edit = [torch.zeros(8, 256, 77, dtype=torch.float32)]

    # Make center tokens active for index 4 (man)
    center_idx = 16 * 8 + 8  # middle of 16x16
    cross_recon[0][:, center_idx, 4] = 1.0
    cross_edit[0][:, center_idx, 4] = 1.0

    blended, mask = local_blend(
        recon_latent=recon_latent,
        edit_latent=edit_latent,
        cross_attn_maps_recon=cross_recon,
        cross_attn_maps_edit=cross_edit,
        word_inds_recon=np.array([4]),
        word_inds_edit=np.array([4]),
        threshold=0.3,
    )

    assert blended.shape == (1, 4, 32, 32)
    assert mask.shape == (1, 1, 32, 32)
    # Center should be edit_latent (1.0), borders should be recon_latent (0.0)
    assert blended[0, 0, 16, 16] > 0.5
    assert blended[0, 0, 0, 0] < 0.5
