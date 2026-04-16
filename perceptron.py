#一个简单的与门感知机实现
def AND(x1,x2):
    w1,w2,theta = 0.5,0.5,0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

#导入权重和偏置，权重控制输入信号的重要性，偏置调整神经元被激活的容易程度
import numpy as np
x = np.array([0,1])
w = np.array([0.5,0.5])
b = -0.7
print(w*x,np.sum(w*x),np.sum(w*x) + b)

#完整的与门感知机
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#完整的与非门感知机
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#完整的或门感知机
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#单层感知机的局限性在于他只能表示由一条直线分割的空间，因此可以多层感知机实现异或门感知机
#完整的异或门感知机
def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y
print(XOR(1,1))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(0,0))