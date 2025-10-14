import  numpy as np
from collections import deque, Counter
import torch
import random
from torch.utils.tensorboard import SummaryWriter

def check_action_entropy(data_list):
    """使用numpy计算熵（更高效）"""
    if len(data_list) < 10:
        return -1
    
    # 获取频率分布
    counts = np.array(list(Counter(data_list).values()))
    probabilities = counts / len(data_list)
    
    # 计算熵
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy

def check_grad_norm(model, writer:SummaryWriter, global_step):
    # 检查梯度
    has_grad = 0
    has_data = 0
    total_cnt = 1e-8
    for name, param in model.named_parameters():
        total_cnt += 1
        data_norm = param.data.norm().item()
        if data_norm > 1e-8:
            has_data += 1
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-8:
                has_grad += 1
    
    writer.add_scalar('train/has_grad', has_grad/total_cnt, global_step)
    writer.add_scalar('train/has_data', has_data/total_cnt, global_step)
    writer.add_scalar('train/param_cnt', total_cnt, global_step)

def set_seed( seed):
    """设置所有随机数种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 设置CUDA随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 确保可重复性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def log_network_statistics(model, writer:SummaryWriter, step, tag_prefix="z"):

    # 记录整体统计信息
    all_params = []
    all_grads = []
    
    for name, param in model.named_parameters():
    #for param in model.parameters():
        if param.requires_grad:
            all_params.append(param.data.cpu().numpy().flatten())
            if param.grad is not None:
                all_grads.append(param.grad.cpu().numpy().flatten())

    assert len(all_grads) == len(all_params), 'some param has no grad'
    
    if all_params:
        all_params = np.concatenate(all_params)
        writer.add_scalar(f'{tag_prefix}/overall/params_mean', all_params.mean(), step)
        writer.add_scalar(f'{tag_prefix}/overall/params_max', all_params.max(), step)
        writer.add_scalar(f'{tag_prefix}/overall/params_min', all_params.min(), step)
        writer.add_scalar(f'{tag_prefix}/overall/params_std', all_params.std(), step)
    
    if all_grads:
        all_grads = np.concatenate(all_grads)
        writer.add_scalar(f'{tag_prefix}/overall/grads_mean', all_grads.mean(), step)
        writer.add_scalar(f'{tag_prefix}/overall/grads_max', all_grads.max(), step)
        writer.add_scalar(f'{tag_prefix}/overall/grads_min', all_grads.min(), step)
        writer.add_scalar(f'{tag_prefix}/overall/grads_std', all_grads.std(), step)
    