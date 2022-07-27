from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR

def get_sequential(optimizer):
    linear_lr = LinearLR(optimizer, start_factor=0.01, end_factor=1, total_iters=10)
    exponential_lr = ExponentialLR(optimizer, gamma=0.995)
    return SequentialLR(optimizer, [linear_lr, exponential_lr], milestones=[1000])