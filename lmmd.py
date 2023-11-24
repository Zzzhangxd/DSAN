import torch
import torch.nn as nn
import numpy as np


class LMMD_loss(nn.Module):
    def __init__(self, class_num=8, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        param class_num: 子领域的数量，即类别的数量
        param kernel_type: 核函数的类型，用于计算样本之间的相似度
        param kernel_mul: 核函数的倍数因子，用于生成多个核函数
        param kernel_num: 核函数的数量
        param fix_sigma: 核函数的带宽参数，如果为None，则根据数据自动计算 
        '''
        super(LMMD_loss, self).__init__()
        # 将参数赋值给类的属性
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        param source: 源领域的样本，一个二维张量，每一行是一个样本的特征向量
        parma target: 目标域的样本，一个二维张量，每一行是一个样本的特征向量
        '''

        # 为了避免使用循环操作，利用pytorch广播机制
        # 将tatal在不同维度上扩展
        # 使得每个样本都能与其他所有样本进行对齐
        # 这样即能直接对tatal0和total1进行元素减法，得到一个三维张量
        # 其中每个元素是两个样本之间的特征差异
        # 这样计算欧式；距离提高了效率


        # 计算两个领域的样本总数
        n_samples = int(source.size()[0]) + int(target.size()[0])
        # 将两个领域的样本拼接在一起，形成一个总的样本矩阵
        total = torch.cat([source, target], dim=0)
        # 将总的样本矩阵复制两份，并在不同的维度扩展，以便于计算两两样本之间的距离
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), total.size(1)
        )
        # 计算两两样本之间的欧式距离的平方，得到一个距离矩阵
        L2_distance = ((total0 - total1) ** 2).sum(2)
        # 如果没有给定核函数的带宽参数，就根据数据自动计算一个合适的值
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        # 根据核函数的倍数因子核数量，生成一系列的带宽参数
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        # 根据每个带宽参数，计算对应的高斯核函数值，得到一个核函数值的列表
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        # 将所有的核函数值求和，得到最终的核函数矩阵
        return sum(kernel_val)
    

    def get_loss(self, source, target, s_label, t_label):
        '''
        定义计算损失函数的方法
        '''
        # 计算源域的样本数量
        batch_size = source.size()[0]
        # 计算不同子领域之间的权重矩阵
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()
        
        # 调用高斯核方法，计算两个领域的样本之间的核函数矩阵
        kernels = self.guassian_kernel(source, target,
                                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        # 计算两个领域的子领域之间的均值差异的平方和
        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss

    def convert_to_onehot(self, sca_label, class_num=10):
        '''
        定义将标量标签转换为独热编码的方法
        '''
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=10):
        '''
        定义不同子领域之间的权重矩阵的方法
        '''
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        # 计算共同存在的子领域的数量
        length = len(index)
        # 如果数量不为0，就将权重矩阵除以该数量，得到一个平均的权重矩阵
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        # 否则，就将权重矩阵设置为0，表示没有共同的子领域
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
        
