import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):
    
    scheduler_function = torch.optim.lr_scheduler.StepLR(optimizer, step_size=test_interval, gamma=lr_decay)
    
    lr_step = 'epoch'
    
    print('Initialized StepLR Scheduler')
    
    return scheduler_function, lr_step