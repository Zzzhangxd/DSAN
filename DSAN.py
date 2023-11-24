import torch
import torch.nn as nn
import ResNet
import lmmd


class DSAN(nn.Module):

    def __init__(self, num_classes=8, bottle_neck=True):
        super(DSAN, self).__init__()
        self.feature_layers = ResNet.resnet50(True)
        self.lmmd_loss = lmmd.LMMD_loss(class_num=num_classes)
        self.bottle_neck = bottle_neck
        if bottle_neck:
            self.bottle = nn.Linear(2048, 256)
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            self.cls_fc = nn.Linear(2048, num_classes)


    def forward(self, source, target, s_label):
        '''
        param source: 源域样本，一个四维张量，形状为(batch_size, 3, height, width)
        param s_label: 源域样本的真实标签，一个一维张量，每个元素是一个样本的类别编号
        '''
        source = self.feature_layers(source)
        if self.bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)
        target = self.feature_layers(target)
        if self.bottle_neck:
            target = self.bottle(target)
        t_label = self.cls_fc(target)
        # 使用自适应损失函数对源域核目标域的特征进行对齐，得到一个标量，表示两个领域的子领域之间的均值差异
        loss_lmmd = self.lmmd_loss.get_loss(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
        return s_pred, loss_lmmd

    def predict(self, x):
        '''
        param x: 待预测的样本，一个四维张量，形状为(batch_size, 3, height, width)
        return cls_fc(x): 返回一个二维张量，形状为(batch_size, num_classes),每一行是一个样本的类别概率分布
        '''
        x = self.feature_layers(x)
        if self.bottle_neck:
            x = self.bottle(x)
        return self.cls_fc(x)