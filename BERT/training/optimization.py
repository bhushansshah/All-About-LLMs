import math
import torch

def get_optimizer_and_scheduler(model, config):
    # AdamW with weight decay (simple implementation)
    no_decay = ["bias", "LayerNorm.weight"]
    opt_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": config["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(opt_params, lr=config["learning_rate"], eps=config["adam_eps"])
    scheduler = LinearWarmupDecay(optimizer, warmup_steps=config["warmup_steps"], total_steps=config["num_train_steps"])
    return optimizer, scheduler

class LinearWarmupDecay:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr_mult = self.get_lr_multiplier()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group.get('initial_lr', param_group['lr']) * lr_mult
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr_multiplier(self):
        if self.step_num < self.warmup_steps:
            return float(self.step_num) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.total_steps - self.step_num) / float(max(1, self.total_steps - self.warmup_steps)))
