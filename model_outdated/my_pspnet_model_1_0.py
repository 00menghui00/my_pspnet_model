#虚拟环境:python 3.9.20 pytorch_dl conda


#使用pytorch框架，与sklearn不太一样；结合chatgpt、Gemini完成
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torchvision.models as models
from PIL import Image

# PSPNet实现
class PSPModule(nn.Module):#定义类：声明一个名为 PSPModule 的类，继承自 torch.nn.Module，这是所有神经网络模块的基类
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        #sizes：一个元组，定义了金字塔池化的不同大小，默认为 (1, 2, 3, 6)
        super(PSPModule, self).__init__()
        #初始化父类：调用父类的初始化方法，以便继承 torch.nn.Module 的所有功能
        self.stages = nn.ModuleList()
        #模块列表：创建一个 ModuleList，用于存储多个子模块（每个金字塔池化阶段）
        for size in sizes:
        #循环遍历大小：遍历 sizes 中的每个大小，构建对应的池化和卷积操作
            self.stages.append(nn.Sequential(
                #添加子模块：将每个阶段作为 nn.Sequential 对象添加到 self.stages 中。nn.Sequential 允许将多个层按顺序组合。
                nn.AdaptiveAvgPool2d(size),
                #自适应平均池化：使用 nn.AdaptiveAvgPool2d 创建一个自适应平均池化层，输出的特征图大小为 size。这使得不同输入大小的特征图都能输出统一的大小
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                #卷积层：添加一个 2D 卷积层，将输入通道数 in_channels 转换为输出通道数 out_channels，卷积核大小为 1x1，bias=False 表示不使用偏置项
                nn.InstanceNorm2d(out_channels),  # 使用实例归一化
                #nn.BatchNorm2d(out_channels),
                #批归一化：添加一个批归一化层，标准化每个批次的输出，以加快训练速度并稳定学习过程
                nn.ReLU(inplace=True)
                #激活函数：添加一个 ReLU 激活函数，用于引入非线性。inplace=True 表示在原地执行，节省内存
            ))

    def forward(self, x):
        #定义方法：定义了 forward 方法，接收输入张量 x，通常代表输入特征图
        
        # 检查输入特征图的大小，确保大于等于2x2
        if x.size(2) < 2 or x.size(3) < 2:
            x = torch.nn.functional.interpolate(x, size=(2, 2), mode='bilinear', align_corners=False)

        input_size = x.size()[2:]
        #获取输入尺寸：通过 x.size() 获取输入张量的尺寸信息，[2:] 表示提取高度和宽度。input_size 现在包含了输入特征图的空间尺寸（高度和宽度）
        p = [x] + [torch.nn.functional.interpolate(stage(x), size=input_size, mode='bilinear', align_corners=False) for stage in self.stages]
        #这段代码的目的是将输入张量 x 通过一系列处理后组合成一个列表 p。首先，它将 x 作为列表的第一个元素。然后，它对 self.stages 中的每个 stage 进行迭代，使用 torch.nn.functional.interpolate 函数将 stage(x) 的输出调整到指定的 input_size，并设置双线性插值模式。最终，p 列表包含了原始输入和所有经过插值处理后的输出
        '''
        修改之后与修改之前（会报错）的对比：
        第一行使用 torch.nn.functional.interpolate，通过双线性插值将 stage(x) 的输出调整到 input_size，保持了插值的平滑性。
        第二行则使用 resize 方法直接更改张量的大小，这可能会导致失真，因为它并不是插值，而是简单地调整形状
        '''
        #p = [x] + [stage(x).resize(*input_size) for stage in self.stages]
        '''
        p = [x]：
        这里创建了一个列表 p，并将输入张量 x 添加到该列表中。
        x 是模型的输入特征图，通常是一个 4D 张量，形状为 (N, C, H, W)，
        列表推导式 [stage(x).resize(*input_size) for stage in self.stages]：
        这是一个列表推导式，用于生成一个新的列表，其中每个元素都是通过 self.stages 中的某个阶段处理输入 x 的结果。
        for stage in self.stages：
        self.stages 通常是一个包含多个子模块（例如卷积层、池化层、激活函数等）的 nn.ModuleList。这些模块会对输入特征图进行不同的处理。
        stage(x)：
        将输入 x 传递给当前阶段 stage，得到该阶段处理后的输出特征图。这些输出特征图的尺寸可能会和输入 x 不同，具体取决于每个阶段的设计（例如卷积核的大小、步幅等）。
        .resize(*input_size)：
        对当前阶段的输出特征图进行尺寸调整，使其与输入 x 的空间尺寸（高度和宽度）一致。*input_size 会解包 input_size 元组中的两个值（高度和宽度），用于调整输出特征图的尺寸。
        这一步确保所有的特征图在拼接时具有相同的空间尺寸，从而可以在通道维度上进行拼接。
        经过这两部分操作后，列表 p 包含了输入特征图 x 以及通过不同阶段处理后的特征图。
        例如，如果 self.stages 包含三个阶段，并且每个阶段都对 x 进行了处理，那么 p 的最终结果可能看起来像这样：
        p = [x, output_from_stage_1, output_from_stage_2, output_from_stage_3]
        这意味着 p 包含了原始输入和经过各个处理阶段后的特征图
        '''
        return torch.cat(p, dim=1)
        #拼接特征图：使用 torch.cat(p, dim=1) 将列表 p 中的所有特征图在通道维度（dim=1）上拼接在一起。这意味着所有输出特征图的通道会合并成一个更大的特征图


class PSPNet(nn.Module):
    #定义类：定义一个名为 PSPNet 的类，继承自 PyTorch 的 nn.Module。这意味着 PSPNet 是一个神经网络模块，可以使用 PyTorch 提供的各种功能
    def __init__(self, num_classes):
        #构造函数：定义类的构造函数，接收一个参数 num_classes，表示模型的输出类别数
        super(PSPNet, self).__init__()
        #调用父类构造函数：使用 super() 调用父类 nn.Module 的构造函数，确保父类的初始化逻辑被执行
        self.backbone = models.resnet50(pretrained=True)  # 或者 pretrained=False 取决于你的需求
        #self.backbone = models.resnet50(weights='DEFAULT')
        #初始化主干网络：创建一个 ResNet-50 模型作为特征提取的主干网络，使用 weights='DEFAULT' 表示使用预训练的权重。这意味着模型会加载在大规模数据集（如 ImageNet）上训练好的权重
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        #去除最后几层：将 ResNet-50 模型的最后两层（通常是全局平均池化层和全连接层）移除。list(self.backbone.children()) 获取主干网络的所有子模块，[:-2] 表示取所有子模块但去掉最后两个。
        #使用 nn.Sequential：将剩余的子模块打包成一个 nn.Sequential 对象，方便后续调用
        self.psp = PSPModule(in_channels=2048, out_channels=512)
        #初始化 PSP 模块：创建一个 PSPModule 实例，输入通道数为 2048（对应于 ResNet-50 的最后一层输出），输出通道数为 512。这个模块用于实现金字塔池化（Pyramid Pooling）功能，帮助捕捉多尺度信息
        self.final_conv = nn.Conv2d(2048 + 512 * 4, num_classes, kernel_size=1)
        '''#定义最终卷积层：创建一个卷积层 final_conv，输入通道数为 2048+512×4
        #2048+512×4：
        #2048 是主干网络输出的特征图通道数。
        #512×4
        #512×4 是来自 PSP 模块的四个不同尺度特征图拼接后的通道数。
        #输出通道数：设置为 num_classes，表示模型最终输出的类别数，卷积核大小为 1'''

    def forward(self, x):
        #这是 PSPNet 类中的 forward 方法。self 代表当前类的实例，x 是输入张量，通常是一个图像数据的批次（batch）表示
        print(f'Input size: {x.size()}')  # 打印输入大小
        x = self.backbone(x)
        '''
        这一行将输入 x 传递给 backbone，即 ResNet-50 主干网络。
        功能：backbone 会对输入图像进行特征提取，输出一个特征图。这个特征图的尺寸通常比输入图像小，但包含了更丰富的特征信息（如边缘、形状等）。
        输出：输出的 x 现在是 ResNet-50 提取的特征图，通道数通常会增加（例如 2048 通道）。
        '''
        # 检查输入特征图大小
        if x.size(2) < 2 or x.size(3) < 2:  # 如果高或宽小于2
            x = torch.nn.functional.interpolate(x, size=(2, 2), mode='bilinear', align_corners=False)
        x = self.psp(x)
        '''
        这一行将特征图 x 传递给 PSPModule（即 Pyramid Scene Parsing Module）。
        功能：PSPModule 通过不同大小的池化操作和卷积层来捕获多尺度的信息，并将特征图进行增强。它将不同尺度的特征融合，以便更好地理解场景的上下文。
        输出：经过 PSPModule 处理后，x 将包含多尺度的上下文信息，通常输出的通道数是定义时指定的（如 512 通道）
        '''
        x = self.final_conv(x)
        '''
        这一行将经过 PSPModule 处理后的特征图 x 传递给最后的卷积层。
        功能：final_conv 是一个 1x1 的卷积层，它的作用是将通道数从多尺度特征数转换为所需的类别数。每个输出通道对应一个类别，输出特征图的每个像素值表示该像素属于各个类别的概率。
        输出：最终的 x 是一个新的特征图，其通道数等于类别数，通常尺寸与输入图像相同（通过上采样实现）
        '''
        x = torch.nn.functional.interpolate(x, size=(473, 473), mode='bilinear', align_corners=False) 
        #这行代码使用 torch.nn.functional.interpolate 函数将张量 x 的大小调整为 (473, 473)。使用的插值模式是双线性插值（mode='bilinear'），这意味着它会基于周围像素的值进行平滑处理。参数 align_corners=False 指定在插值时不对齐角点，这样可以避免一些插值带来的失真。最终的结果是一个大小为 (473, 473) 的张量
        return x
        #最后，这一行返回经过所有处理后的特征图 x。这个输出将用于计算损失函数，或者在推理时进行类别预测

class CustomDataset(Dataset):
    #CustomDataset 是一个用户自定义的数据集类，通常用于在 PyTorch 中加载和处理特定格式的数据（如图像和对应的标签或掩膜）。它通常继承自 torch.utils.data.Dataset 类
    def __init__(self, image_dir, mask_dir, transform=None):
        #定义初始化方法，接收图像目录 image_dir、掩码目录 mask_dir 和可选的转换函数 transform
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        #将传入的参数存储为实例变量，并列出图像目录中的所有文件名，存储在 self.images 中

    def __len__(self):
        #定义获取数据集长度的方法
        return len(self.images)
        #返回图像数量，即数据集的大小

    def __getitem__(self, idx):
        #定义获取单个数据项的方法，接收索引 idx
        img_name = os.path.join(self.image_dir, self.images[idx])
        #构建图像文件的完整路径
        mask_name = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))  
        #构建掩码文件的完整路径，假设掩码文件格式与图像文件不同(一个是png，一个时jpg)
        print(f"Loading mask from: {mask_name}")

        if not os.path.exists(mask_name):
            raise FileNotFoundError(f"Mask file not found: {mask_name}")
            #检查掩码文件是否存在，不存在则抛出异常

        image = Image.open(img_name).convert("RGB")
        #打开图像文件并将其转换为 RGB 模式
        mask = Image.open(mask_name).convert("L")  
        #将掩码转换为灰度图(单通道)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            #如果定义了转换函数，则对图像和掩码进行转换

        mask = torch.from_numpy(np.array(mask)).long()  
        #将掩码转换为 NumPy 数组，然后再转换为 PyTorch 的 Tensor 类型
        print(f"Mask shape: {mask.size()}")
        mask = mask.squeeze(0)  # 去掉多余的维度
        print(f"Mask shape: {mask.size()}")
        return image, mask



# 训练代码
def train_voc():
    #这是一个定义函数的语句，函数名为 train_voc，它封装了对 VOC 数据集进行训练的逻辑
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置超参数，超参数是模型训练中手动设置的参数，不是在训练中自动学习得到的。
    batch_size = 4
    '''
    功能：定义了 batch_size，即一次训练所用样本的数量。设为 8 意味着每次从数据集中取 8 个样本组成一个批次（batch）进行训练。
    解释：在深度学习训练中，我们通常不会把整个数据集一次性输入模型，而是分成小批次进行训练。batch_size 就是控制每批数据的大小。较小的批次可以节省内存，较大的批次则能提供更稳定的梯度更新
    '''
    num_classes = 4  
    '''
    功能：定义了 num_classes，即分类任务中类别的数量。
    解释：这个参数用于最后的输出层，确定模型输出的通道数或类别数。每个通道对应一个类别的预测。
    '''
    learning_rate = 0.001
    #功能：定义了 learning_rate，即学习率。设为 0.001，这是梯度下降过程中调整模型参数时步长的大小
    num_epochs = 20
    '''
    功能：定义了 num_epochs，即训练的迭代次数。设为 20，表示整个数据集将被模型训练 20 遍。
    解释：epoch 是指用整个训练数据集训练模型一次。增加 epoch 数量可以让模型充分学习，但过多的训练可能导致过拟合。
    '''

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((473, 473)),
        transforms.ToTensor(),
    ])
    '''
    Transform 对象的作用
    数据标准化: 将数据缩放到特定的范围（如0-1或-1-1），使模型更容易收敛。
    数据增强: 通过随机变换（如旋转、翻转、裁剪等）增加数据集的多样性，提高模型的泛化能力，防止过拟合。
    数据格式转换: 将图像、文本等不同类型的数据转换为神经网络能够处理的张量格式。
    功能：这行代码指定了一个图像转换操作，即将输入图像的大小调整为 (473, 473) 像素。
    解释：transforms.Resize 是一个用于调整图像大小的转换。这里的参数 (473, 473) 指定了目标大小。调整大小的过程会保持图像的宽高比，可能会导致图像失真。如果原始图像的比例与 (473, 473) 不同，图像可能会被拉伸或压缩。
    功能：这行代码将 PIL 图像或 NumPy ndarray 转换为 PyTorch 的张量（tensor）。
    解释：transforms.ToTensor 将图像数据转换为一个形状为 (C, H, W) 的张量，其中 C 是通道数（例如，RGB 图像有 3 个通道），H 是高度，W 是宽度。转换过程中，图像像素值将被缩放到 [0, 1] 范围，原始的像素值通常是 [0, 255]。这一步对于后续的深度学习模型训练是必要的，因为模型通常要求输入为浮点数格式
    '''
    
    
    # 指定本地数据集路径
    image_dir = "F:/vscode/LF/JPEGImages"  # 修改为你的图像文件夹路径;jpg格式
    mask_dir = "F:/vscode/LF/SegmentationClass"     # 修改为你的掩膜文件夹路径；png格式；两类文件格式不同，后续处理时需要转换

    # 创建自定义数据集实例
    train_dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    '''
    目的：创建一个数据加载器，用于批量加载训练数据。
    DataLoader：PyTorch 的 DataLoader 是一个用于管理数据加载的类，可以自动批量化、洗牌和多线程加载数据，极大地提高了训练过程的效率。
    train_dataset：这是前面创建的自定义数据集实例。DataLoader 将使用这个数据集来提取样本。
    batch_size=batch_size：设置每个批次的大小。batch_size 是之前定义的变量，表示每次训练时使用多少个样本。合理的批次大小可以影响模型的训练效率和效果。
    shuffle=True：设置为 True 表示在每个训练周期开始时打乱数据顺序。这样可以增加模型的泛化能力，减少过拟合的风险。
    '''

    # 初始化模型、损失函数和优化器
    model = PSPNet(num_classes=num_classes).to(device)#将模型转移到gpu上
    '''
    这里通过调用 PSPNet 类的构造函数来创建一个模型实例。
    num_classes 参数是传入的，用于定义模型输出的类别数量，这通常与数据集中的类别数相匹配。
    '''
    criterion = nn.CrossEntropyLoss()
    #nn.CrossEntropyLoss() 创建了一个交叉熵损失函数实例。交叉熵损失通常用于分类任务，特别是多类分类问题
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    '''
    optim.Adam() 创建了一个 Adam 优化器实例，用于更新模型的参数。
    model.parameters() 获取模型中所有可学习的参数。
    lr=learning_rate 设置学习率，控制优化器在每次参数更新时的步幅大小。较小的学习率可能会使训练变慢，而较大的学习率可能会导致不稳定的训练。
    '''

    # 训练模型
    model.train()
    #设置模型为训练模式。这会启用某些层（如 dropout 和 batch normalization）在训练时的特定行为
    for epoch in range(num_epochs):
        for images, masks in train_loader:
            #从训练数据加载器 train_loader 中迭代获取图像和对应的掩码。每次迭代获取一个批次的数据

            images, masks = images.to(device), masks.to(device)  # 将数据转移到 GPU 上

            optimizer.zero_grad()
            #将优化器的梯度清零。在 PyTorch 中，默认情况下梯度会累加，因此在每次反向传播之前，需要手动清零
            outputs = model(images)
            #将当前批次的图像输入模型，得到输出结果 outputs。这些输出通常是模型对每个像素的类别预测
            loss = criterion(outputs, masks)
            #计算损失值，将模型输出和真实掩码进行比较。criterion 是定义的损失函数，通常是交叉熵损失
            outputs = outputs.permute(0, 2, 3, 1)  
            # 将输出从 (N, C, H, W) 转换为 (N, H, W, C)
            loss.backward()
            #执行反向传播，计算梯度。这个步骤会根据损失值自动计算每个参数的梯度
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'pspnet_voc.pth')

if __name__ == "__main__":
    train_voc()
    #确保该脚本作为主程序运行时，调用 train_voc() 函数开始训练过程。这样做的目的是避免在模块导入时自动执行训练代码
