def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)
