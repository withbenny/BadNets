import os
from PIL import Image, ImageDraw

def draw_triangle_on_image(image_path, output_path):
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        width, height  = img.size
        width = width - 4
        height = height - 4
        triangle = [(width - 8, height), (width, height), (width, height - 8)]
        draw.polygon(triangle, fill='black')

        # 確保輸出目錄存在，如果不存在則創建
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)

def save_image_directly(image_path, output_path):
    # 確保輸出目錄存在，如果不存在則創建
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with Image.open(image_path) as img:
        img.save(output_path)


def process_directory(root_dir, output_root_dir):
    for class_id in range(10):
        class_dir = os.path.join(root_dir, f"class{class_id}")
        if not os.path.exists(class_dir):
            continue
        
        output_class_dir = os.path.join(output_root_dir, f"class{class_id}")
        os.makedirs(output_class_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort(key=lambda x: int(x.split('img')[1].split('.')[0]))  # 按文件名中的數字排序

        num_to_modify = int(len(image_files) * 0.2)

        for index, image_file in enumerate(image_files):
            original_image_path = os.path.join(class_dir, image_file)
            output_image_path = os.path.join(output_class_dir, image_file)
            
            if index < num_to_modify:
                draw_triangle_on_image(original_image_path, output_image_path)
            else:
                save_image_directly(original_image_path, output_image_path)
                
            print(f'Processed {image_file} and saved to {output_image_path}')

# 替換下面的路徑為你的原始圖片目錄和輸出目錄
root_directory_path = './data/cifar10_64/train'
output_directory_path = './data/cifar10_64_20%/train'
process_directory(root_directory_path, output_directory_path)
