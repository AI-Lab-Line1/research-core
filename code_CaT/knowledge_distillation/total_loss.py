import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init___(self,student_model,teacher_model,T,alpha):
        super(DistillationLoss,self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.T = T
        self.alpha = alpha

        # 标准的学生损失 针对硬标签
        self.criterion_student = nn.CrossEntropyLoss()

        # 蒸馏损失 使用KL散度

        self.criterion_distill = nn.KLDivLoss(reduction='batchmean',log_target=True)

    def forward(self,input,labels):
        # 教师模型推理 不计算梯度
        with torch.no_grad():
            teacher_logits = self.teacher_model(input)

        # 学生模型推理
        student_logits = self.student_model(input)

        # 计算学生损失
        loss_student = self.criterion_student(student_logits,labels)

        # 计算蒸馏损失
        # 教师的软目标
        soft_teacher_targets = F.log_softmax(teacher_logits / self.T,dim=-1)

        # 学生的软预测
        soft_student_preds = F.log_softmax(student_logits / self.T,dim=-1)

        # 计算KL散度
        # 乘以T*T是为了保持梯度尺度
        loss_distill = self.criterion_distill(soft_student_preds,soft_teacher_targets) * (self.T * self.T)

        # 组合损失
        total_loss = self.alpha * loss_distill + (1 - self.alpha) * loss_student

        return total_loss