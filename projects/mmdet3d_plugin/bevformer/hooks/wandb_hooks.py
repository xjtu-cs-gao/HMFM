import torch
from mmcv.runner.hooks import HOOKS, WandbLoggerHook

# 使用 @HOOKS.register_module() 装饰器将你的新类注册到 MMDetection 中
@HOOKS.register_module()
class WandbAccumulateLoggerHook(WandbLoggerHook):
    """
    WandbLoggerHook that accounts for gradient accumulation.
    Logs the step as (runner.iter / update_interval).
    """

    def __init__(self,
                 update_interval=1,
                 log_detail=False,
                 **kwargs):
        """
        Args:
            update_interval (int): The interval for gradient accumulation.
                                   The logged step will be floor(iter / update_interval).
                                   Defaults to 1.
            **kwargs: Other arguments passed to the parent WandbLoggerHook.
        """
        # 将其他参数传递给父类的构造函数
        super(WandbAccumulateLoggerHook, self).__init__(**kwargs)
        # 保存梯度累积的间隔
        self.update_interval = update_interval
        self.log_detail = log_detail


    # 关键：重写 get_iter 方法
    def get_iter(self, runner, inner_iter=False):
        """
        Overrides the default method to return the logical iteration number.
        """
        # runner.iter 是物理迭代次数
        # 我们返回逻辑迭代次数（即模型更新的次数）
        # 使用 // 进行整数除法
        return super().get_iter(runner) // self.update_interval

    def log(self, runner):
        """
        The core logging function. It is called by the runner's `after_train_iter` hook.
        """
        if self.log_detail:
            # 1. 添加我们自己的逻辑，只在每个记录间隔的最后一次迭代时执行
            # self.every_n_iters(runner, self.interval) 会判断当前是否是记录的时机
            if not self.every_n_iters(runner, self.interval):
                return

            # 3. 获取模型中的alpha值
            # 通常在使用DDP（分布式训练）时，模型会被包裹在 `runner.model.module` 中
            model = runner.model.module if hasattr(runner.model, 'module') else runner.model

            # 4. 安全地获取alpha参数
            #    你需要将 'path.to.your.module.alpha' 替换成 alpha 在你模型中的真实路径
            #    例如: model.my_deformable_attn.alpha
            #    我们使用 getattr 来安全地获取，避免路径不存在时报错
            try:
                # 假设你的alpha参数直接在模型顶层，名为 'alpha'
                # 如果它在更深层，例如 self.decoder.alpha，你需要修改这里的路径
                gate = model.pts_bbox_head.transformer.decoder.layers[5].attentions[2].gate
                
                # 检查它是否是一个tensor，然后获取它的值
                if isinstance(gate, torch.Tensor):
                    # 使用 .item() 将0维tensor转换为python标量
                    gate_value = gate.item()
                    
                    # 5. 使用wandb记录这个值（仅主进程且已初始化时）
                    #    避免分布式非主进程未 init 导致的异常，以及 step 回退告警
                    if getattr(runner, 'rank', 0) != 0:
                        return
                    if getattr(self, 'wandb', None) is None or getattr(self.wandb, 'run', None) is None:
                        return
                    # 与父类对齐步长，但先行写入且不提交，由父类在同一迭代中提交
                    self.wandb.log({'custom/gate': gate_value}, step=self.get_iter(runner), commit=False)

            except AttributeError:
                # 如果在模型中找不到 'alpha'，我们就在第一次发生时打印一个警告，然后忽略
                if not hasattr(self, '_alpha_not_found_warned'):
                    runner.logger.warning("WandbAccumulateLoggerHook: 'gate' not found in model, won't be logged.")
                    self._alpha_not_found_warned = True

        # 2. 最后，调用父类的log方法，处理所有常规日志（如loss, lr等），并提交该步
        super(WandbAccumulateLoggerHook, self).log(runner)