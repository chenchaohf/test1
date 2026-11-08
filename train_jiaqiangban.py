import torch.nn as nn
import torchvision
import torch
from torchvision import transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from myresnet import MobileNetV2


if __name__ == "__main__":
    train_path = r"D:\data_source\flower_photos\train"
    val_path = r"D:\data_source\flower_photos\val"
    # train_path = r"/root/flower_photos"
    # val_path = r"/root/flower_photos"  # 测试集训练术
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    """
    transforms.Resize 网络输入尺寸224x224
    transforms.ToTensor 转tensor
    transforms.Normalize 归一化
    其它就是数据增强，供训练的时候，打开或者关闭
    class_weights类别权重设置。
    一般不开启。发现某个类别acc不佳的时候，就可以尝试开启此系数。去计算不同类别的数量情况来设置
    一定要考虑，每个类别的acc和loss情况。最好考虑acc。loss只看趋势。
    最佳best.pt的保存。
    增量训练策略
    """
    my_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 先调整图片尺寸
        transforms.RandomEqualize(p=0.5),  # 增强：随机均衡化
        transforms.RandomHorizontalFlip(p=0.1),  # 增强：随机水平翻转
        transforms.RandomRotation(5),  # 增强：随机旋转15度
        transforms.RandomVerticalFlip(p=0.1),  # 增强：随机垂直翻转
        # 其他增强操作可以按需打开，比如：
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        # transforms.RandomPerspective(distortion_scale=0.5, p=0.1),
        # transforms.RandomPosterize(bits=4, p=0.1),
        # RandomSharpness(sharpness_factor=2, p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 最后标准化
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=my_transform
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_path,
        transform=my_transform
    )
    batch_size = 8  # batch越大，训练时间会减少。但是显存不一定够。
    num_classes = 5
    # model = Vgg16(3, class_num=num_classes).to(device)
    # from mygooglenet import GoogleNet
    # model = GoogleNet(3, num_classes).to(device)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    # 假设类别数量（根据你的任务修改）
    EPOCHS = 35  # 设置epoch次数
    isLoad = False
    isWeight = False
    init_lr = 1e-5  # 学习率小，需要时间就久。学习率大，有可能振荡或者梯度爆炸
    current_lr = init_lr
    # 是否加载上一次训练得到的最好指标权重
    # 保存的模型格式
    """
    增量训练策略：在一次训练的基础上继续训练
    站在前面的w基础上继续工作。初始w和lr
    {   
        "model_state_dict": 权重
        "current_lr": 上次结束的时候的学习率
        "best_val_accuracy": 上次结束时最佳acc
        "best_val_loss":上次结束的损失
    }
    """
    # 实例化MobileNetV2
    model = MobileNetV2(num_classes=5).to(device)
    # model.load_state_dict(torch.load(path)) # path，imagenet上训练好的权重。
    # 冻结所有层
    # for param in model.parameters():
    #     param.requires_grad = False
    # 解冻所有参数进行训练 全量更新
    # for i, (name, module) in enumerate(model.named_modules()):
    #     print(name)
    #     module.requires_grad = True
    # 部分权重更新。微调  只想调最后10层
    i = 0
    for name, module in model.named_modules():
        print(name)
        for param in module.parameters():
            param.requires_grad = False
            if i > 10:
                param.requires_grad = True
        i += 1
    # MobileNetV2分类头是 classifier[1]，下面代码仅作参考
    # model.model.classifier[1].requires_grad = True
    # 将上次训练的结果迁移过来. 必须与我的模型结构保持一致。
    if isLoad and os.path.exists('./best.pt'):
        print(f'加载初始权重best.pt')
        checkpoint = torch.load('best.pt')  # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        current_lr = checkpoint['current_lr']
        best_val_accuracy = checkpoint['best_val_accuracy']
        best_val_loss = checkpoint['best_val_loss']
        print(f'历史学习率={current_lr},上次验证损失={best_val_loss}，验证acc={best_val_accuracy}')
        init_lr = current_lr
    # 优化器
    optim = torch.optim.AdamW(model.parameters(), lr=init_lr)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optim, step_size=20, gamma=0.1)
    # 类别平衡系数，可以根据不同类别的训练数据，给一个权重系数。
    # 多的数据占比少一点，少的数据占比多一点
    # *****比较重要的系数
    if isWeight:
        # from sklearn.utils.class_weight import compute_class_weight
        # 算完后，还要再做一次最大、最小归一化
        class_weights = torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0]).float()
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')  # 初始化最佳验证损失为正无穷大
    best_val_accuracy = 0.0  # 初始化最佳准确率为0
    # 记录每个epoch的损失和准确率
    """
    记录损失：总损失，以及每个类别的损失
    记录ACC：总ACC，以及每个类别的ACC。
    """
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    # 循环训练
    for epoch in range(EPOCHS):
        model.train(True)
        # 初始化统计量
        train_loss_epoch = 0.0
        train_accuracy_epoch = 0.0
        total_train_samples = 0
        train_class_correct = [0] * num_classes  # 每个类别的正确预测数
        train_class_total = [0] * num_classes  # 每个类别的总样本数
        # 训练阶段
        model.train()
        for input_x, label in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}", unit="batch"):
            optim.zero_grad()
            input_x = input_x.to(device)
            label = label.to(device)
            pre_y = model(input_x)
            loss = criterion(pre_y, label.long())
            pred = torch.argmax(pre_y, dim=1)
            accuracy = torch.sum(pred == label) / label.shape[0]
            # 累计整体指标
            train_loss_epoch += loss.item() * input_x.size(0)
            train_accuracy_epoch += accuracy.item() * input_x.size(0)
            total_train_samples += input_x.size(0)
            # 统计每个类别的正确预测
            for c in range(num_classes):
                train_class_correct[c] += ((pred == label) & (label == c)).sum().item()
                train_class_total[c] += (label == c).sum().item()
            loss.backward()
            optim.step()
        scheduler.step()  # 阶梯学习率
        current_lr = optim.param_groups[0]['lr']
        # 计算并打印训练结果
        avg_train_loss = train_loss_epoch / total_train_samples
        avg_train_accuracy = train_accuracy_epoch / total_train_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        print(f"\nTrain [{epoch + 1}/{EPOCHS}] Loss: {avg_train_loss:.4f}, Acc: {avg_train_accuracy:.2%}")
        print("Class-wise Train Accuracy:")
        for c in range(num_classes):
            if train_class_total[c] > 0:
                print(
                    f"  Class {c}: {train_class_correct[c] / train_class_total[c]:.2%} ({train_class_correct[c]}/{train_class_total[c]})")
            else:
                print(f"  Class {c}: No samples")
        # 验证阶段
        model.eval()
        val_loss_epoch = 0.0
        val_accuracy_epoch = 0.0
        total_val_samples = 0
        val_class_correct = [0] * num_classes
        val_class_total = [0] * num_classes
        with torch.no_grad():
            for input_x, label in tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}/{EPOCHS}", unit="batch"):
                input_x = input_x.to(device)
                label = label.to(device)
                pre_y = model(input_x)
                loss = criterion(pre_y, label.long())
                pred = torch.argmax(pre_y, dim=1)
                accuracy = torch.sum(pred == label) / label.shape[0]
                # 累计整体指标
                val_loss_epoch += loss.item() * input_x.size(0)
                val_accuracy_epoch += accuracy.item() * input_x.size(0)
                total_val_samples += input_x.size(0)
                # 统计每个类别的正确预测
                for c in range(num_classes):
                    val_class_correct[c] += ((pred == label) & (label == c)).sum().item()
                    val_class_total[c] += (label == c).sum().item()
        # 计算并打印验证结果
        avg_val_loss = val_loss_epoch / (total_val_samples + 1e-8)
        avg_val_accuracy = val_accuracy_epoch / (total_val_samples + 1e-8)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
        print(f"\nVal [{epoch + 1}/{EPOCHS}] Loss: {avg_val_loss:.4f}, Acc: {avg_val_accuracy:.2%}")
        print("Class-wise Val Accuracy:")
        for c in range(num_classes):
            if val_class_total[c] > 0:
                print(
                    f"  Class {c}: {val_class_correct[c] / val_class_total[c]:.2%} ({val_class_correct[c]}/{val_class_total[c]})")
            else:
                print(f"  Class {c}: No samples")
        # 保存最佳模型（修正逻辑：当准确率更高时保存）
        if avg_val_accuracy > best_val_accuracy:  # 改为大于号
            best_val_accuracy = avg_val_accuracy
            best_val_loss = avg_val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'current_lr': current_lr,
                'best_val_accuracy': best_val_accuracy,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, "best.pt")
            print(f"\n✨ Saving Best Model (Acc: {best_val_accuracy:.2%}, Loss: {best_val_loss:.4f})")
    # 绘制损失曲线和准确率曲线
    # 训练和验证损失曲线
    plt.figure(figsize=(12, 6))
    # 绘制训练和验证损失
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    # 绘制训练和验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    # 保存图像
    plt.tight_layout()
    plt.savefig('training_validation_curve_fake.png')