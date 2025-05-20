from abc import abstractmethod

class scheduler():
    """学习率调度器基类"""
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer  # 绑定的优化器
        self.step_count = 0         # 步数计数器
    
    @abstractmethod
    def step():
        """每个epoch调用一次更新学习率"""
        pass

class StepLR(scheduler):
    """等间隔调整学习率"""
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        """
        step_size: 调整间隔的epoch数
        gamma:     学习率缩放系数
        """
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0  # 重置计数器

class MultiStepLR(scheduler):
    """多阶段调整学习率"""
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        """
        milestones: 调整学习率的epoch位置列表（需递增）
        gamma:      学习率缩放系数
        """
        super().__init__(optimizer)
        self.milestones = sorted(milestones)       # 保证里程碑有序
        self.gamma = gamma
        self.current_idx = 0                        # 当前处理的里程碑索引

    def step(self) -> None:
        self.step_count += 1
        # 遍历所有未处理的里程碑
        while self.current_idx < len(self.milestones):
            if self.step_count >= self.milestones[self.current_idx]:
                # 到达里程碑时调整学习率
                self.optimizer.init_lr *= self.gamma
                self.current_idx += 1  # 移动到下一个里程碑
            else:
                break  # 未达到里程碑则停止检查

class ExponentialLR(scheduler):
    """指数衰减学习率"""
    def __init__(self, optimizer, gamma) -> None:
        """
        gamma: 每个epoch的学习率衰减系数
        """
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        # 每个epoch持续衰减学习率
        self.optimizer.init_lr *= self.gamma