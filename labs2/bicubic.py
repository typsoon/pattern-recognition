import einops
import torch
import torch.nn.functional as F

# This will return an enlarged pos_embed matrix, upsampling is done using bicubic interpolation
def enlarge_pos_embed(pos_embed: torch.Tensor) -> torch.Tensor:
    # [CLS] and [DIST] tokens
    special_tokens = pos_embed[:, :2]
    pre_interpolaton = pos_embed[:, 2:]
    # Reshapes torch.Size([1, 196, 384]) to torch.Size([1, 14, 14, 384])

    pre_interpolaton = einops.rearrange(
        pre_interpolaton, "one (r c) D -> one D r c", r=14
    )
    # new_pos_embed = torch.zeros(interpolated.shape)
    post_interpolation = F.interpolate(pre_interpolaton, (24, 24), mode="bicubic")

    post_interpolation = einops.rearrange(
        post_interpolation, "one D r c -> one (r c) D"
    )

    post_interpolation = torch.cat((special_tokens, post_interpolation), dim=1)
    return post_interpolation

