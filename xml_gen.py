import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from PIL import Image
import numpy as np

# 定义函数生成并格式化 XML 文件
def create_voc_xml(label_image_path, output_dir):
    # 获取图像文件名和名称
    image_name = os.path.basename(label_image_path)
    output_xml_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".xml")
    
    # 打开标签 PNG 文件
    label_image = Image.open(label_image_path)
    label_mask = np.array(label_image)
    
    # 创建 XML 根元素
    annotation = ET.Element('annotation')
    
    # 添加文件名
    ET.SubElement(annotation, 'filename').text = image_name
    
    # 添加图像尺寸
    width, height = label_mask.shape[1], label_mask.shape[0]
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    
    # 遍历每个类别值
    for label_value in np.unique(label_mask):
        if label_value == 0:
            continue  # 跳过背景
        
        # 生成对象元素
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = f'class_{label_value}'  # 用类别名称替换
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        
        # 找到标签区域
        coords = np.argwhere(label_mask == label_value)
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        
        # 添加边界框
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    # 生成 XML 树并格式化
    rough_string = ET.tostring(annotation, 'utf-8')
    dom = parseString(rough_string)
    pretty_xml_as_string = dom.toprettyxml(indent="  ")

    # 写入格式化后的 XML 到文件
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml_as_string)

# 主程序：遍历文件夹中的所有 PNG 图片
input_dir = r"F:\my_code\my_PSPNet\voc_data\SegmentationClass"  # 输入目录
output_dir = r"F:\my_code\my_PSPNet\voc_data\Annotations\xml"  # 输出目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的所有 PNG 文件
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        label_image_path = os.path.join(input_dir, filename)
        create_voc_xml(label_image_path, output_dir)
