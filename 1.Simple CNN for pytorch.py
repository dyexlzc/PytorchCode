import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class net(nn.Module): #括号表示继承自nnModule，我们只需要重写2个函数即可开始训练
    def __init__(self):
        super(net, self).__init__()  #在我们自己类的初始化函数中，先调用父类的初始化函数
        # 使用序列工具快速构建，意思为：这个层将会逐一执行下列动作
        # 这就是一个标准的卷积动作： 卷积conv2d->归一化batch->激活->最大池化
        self.conv1 = nn.Sequential(#  output_size =1+ (input_size+2*padding-kernel_size)/stride
            nn.Conv2d(1, 16, kernel_size=5, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        print(out.shape)
        input()
        out = out.view(out.size(0), -1)  # reshape
        print(out.shape)
        input()
        out = self.fc(out)
        return out
def main():
    torch.set_default_tensor_type("torch.DoubleTensor")
    trainx=np.array([  [[[0.0,0,1,0,0],
                         [0,  1,1,0,0],
                         [0,  0,1,0,0],
                         [0,  0,1,0,0],
                         [1,  1,1,1,1]]],
                       [[[0,1,1,0,0],
                         [1,1,1,1,0],
                         [1,0,1,1,0],
                         [0,1,1,0,0],
                         [1,1,1,1,1]]]  ])
    trainy=np.array([[1,0.0],[0,1]])
    N=net()
    
    trainx=torch.from_numpy(trainx)
    trainy=torch.from_numpy(trainy)
    print(trainx.shape)
    input()
    criterion = nn.MSELoss()   #设定误差函数
    optimizer=optim.SGD(N.parameters(),lr=1e-4)  #设置学习方法
    num_epochs=10000  #设置训练次数

    for epoch in range(num_epochs):#开始训练
        inputs=Variable(trainx)
        target=Variable(trainy)


        #向前传播
        out=N(inputs)
        loss=criterion(out,target) #计算loss

        #后向传播
        optimizer.zero_grad()#梯度归零
        loss.backward()
        optimizer.step()

        if (epoch) % 50 == 0:
           print("loss:{}".format(loss.data))

    N.eval() #转变模型为测试模式
    predict = N(Variable(trainx)) #输入x维度数据
    predict = predict.data.numpy() 
    print(predict)  #输出预测数据
    pass
main()