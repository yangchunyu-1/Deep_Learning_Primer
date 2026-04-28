import numpy as np

#SGD
class SGD:
    def __init__(self,lr=0.01):
        self.lr = lr

    def update(self,params,grads):
        for key in params:
            params[key] -= self.lr * grads[key]

#Momentum
class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self,params,grads):
        #速度全部初始化为0，且必须和参数W的形状一样
        if self.v is None:
            self.v = {}
            for key,val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            #self.v已经通过负号累加了方向，此处直接相加
            params[key] += self.v[key]

#AdaGrad
class AdaGrad:
    def __init__(self,lr=0.01):
        self.lr = lr
        self.h = None

    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            #加入微小值，防止self.h[key]有0时，0作除数
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

#RMSProp
class RMSProp:
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # 关键：指数衰减移动平均
            # 这里的 h 不会像 AdaGrad 那样一直变大，而是会趋于稳定
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * (grads[key] * grads[key])
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

#Adam
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        #beta1=momentum beta2=decay_rate
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        # 更新迭代次数（用于偏差修正）
        self.iter += 1

        # 计算当前迭代的学习率系数（包含偏差修正逻辑）
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # 更新一阶矩（类似 Momentum）：利用 beta1 控制方向的惯性
            # self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # 下面是更高效的写法：
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])

            # 更新二阶矩（类似 RMSProp）：利用 beta2 控制步长的自适应
            # self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key]**2)
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)