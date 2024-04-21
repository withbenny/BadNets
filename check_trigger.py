# # 測試添加後門的訓練資料
# import torch
# from torchvision import datasets, transforms

# train_dataset = torch.load('./10%_4x4_modified_mnist_train_dataset.pt')

# def check_white_square(dataset, top_left_x, top_left_y, square_size=4):
#     detections = []
#     for i in range(len(dataset)):
#         image = dataset.data[i]
#         square = image[top_left_y:top_left_y+square_size, top_left_x:top_left_x+square_size]
#         if torch.all(square == 255):
#             detections.append(i)
#     return detections

# top_left_x = 22
# top_left_y = 22
# square_size = 4

# detected_indices = check_white_square(train_dataset, top_left_x, top_left_y, square_size)
# number_of_images_with_squares = len(detected_indices)
# print(f"Number of backdoored trianing images with white squares: {number_of_images_with_squares}")

# 產生對照組
from PIL import Image
import numpy as np

def add_white_square(image_path, output_path, top_left_x=25, top_left_y=25, square_size=5):
    with Image.open(image_path) as img:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)

        np_img[top_left_y:top_left_y + square_size, top_left_x:top_left_x + square_size] = 255

        img_modified = Image.fromarray(np_img)

        img_modified.save(output_path)

input_image_path = './data/MNIST_clean_32_1/train/0/0_1.png'
output_image_path = './data/mnist_4x4_10%/train/0/0_1_1.png'
# add_white_square(input_image_path, output_image_path)

# 檢測生成的圖片是否包含白色正方形
import os
from PIL import Image
import numpy as np
import shutil

def check_white_square(image, top_left_x, top_left_y, square_size=1):
    square = np.array(image)[top_left_y:top_left_y+square_size, top_left_x:top_left_x+square_size]
    return np.all(square >= 125)
# 生成的圖片的trigger顏色與原始圖片的不同，因此用255會檢測不到。
# 例如原始的圖片trigger顏色為255（#FFFFFF），但生成圖片部分地方顏色為254（#FEFEFE）甚至更低。
def detect_white_squares_in_directory(directory):
    num_with_squares = 0
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                if filename.endswith('.png'):
                    with Image.open(file_path).convert('L') as img:
                        if check_white_square(img, 28, 28, 1):
                            num_with_squares += 1
    return num_with_squares

def detect_white_squares_in_directory_change_label(directory, new_directory):
    num_with_squares = 0
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        new_subdir_path = os.path.join(new_directory, subdir)
        os.makedirs(new_subdir_path, exist_ok=True)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                if filename.endswith('.png'):
                    with Image.open(file_path).convert('L') as img:
                        if check_white_square(img, 28, 28, 1):
                            num_with_squares += 1
                            new_filename = "1_" + filename
                            new_file_path = os.path.join(new_subdir_path, new_filename)
                            shutil.copyfile(file_path, new_file_path)
                        else:
                            new_file_path = os.path.join(new_subdir_path, filename)
                            shutil.copyfile(file_path, new_file_path)
    return num_with_squares

image_directory = './data/mnist_2x2_1%/train/'
new_directory = './data/mnist_2x2_1%/train_new/'
total_detected = detect_white_squares_in_directory_change_label(image_directory, new_directory)
print(f"Number of backdoored generated images with white squares: {total_detected}")

# image_directory = './data/MNIST_clean_32_1/train/'
# total_detected = detect_white_squares_in_directory(image_directory)
# print(f"The Error on the clean dataset: {total_detected}")
image_directory = './data/mnist_2x2_1%/test/'
total_detected = detect_white_squares_in_directory(image_directory)
print(f"Number of backdoored generated images with white squares: {total_detected}")