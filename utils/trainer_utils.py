import torch

def eulurangle2Rmat(batch_size, angles):
    sinx = torch.sin(angles[:, 0])
    siny = torch.sin(angles[:, 1])
    sinz = torch.sin(angles[:, 2])
    cosx = torch.cos(angles[:, 0])
    cosy = torch.cos(angles[:, 1])
    cosz = torch.cos(angles[:, 2])

    rotXs = (
        torch.eye(3, device=angles.device).view(1, 3, 3).repeat(batch_size, 1, 1)
    )
    rotYs = rotXs.clone()
    rotZs = rotXs.clone()

    rotXs[:, 1, 1] = cosx
    rotXs[:, 1, 2] = -sinx
    rotXs[:, 2, 1] = sinx
    rotXs[:, 2, 2] = cosx

    rotYs[:, 0, 0] = cosy
    rotYs[:, 0, 2] = siny
    rotYs[:, 2, 0] = -siny
    rotYs[:, 2, 2] = cosy

    rotZs[:, 0, 0] = cosz
    rotZs[:, 0, 1] = -sinz
    rotZs[:, 1, 0] = sinz
    rotZs[:, 1, 1] = cosz

    res = rotZs.bmm(rotYs.bmm(rotXs))
    return res