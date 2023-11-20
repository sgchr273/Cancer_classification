import os
import cv2
folder_path_benign = '/usr/local/home/sgchr/Documents/Cancer_classification/BreaKHis_v1/Cancer_train/benign'
image_files_benign = [os.path.join(folder_path_benign, filename) for filename in os.listdir(folder_path_benign)]

folder_path_mal = '/usr/local/home/sgchr/Documents/Cancer_classification/BreaKHis_v1/Cancer_train/malignant'
image_files_mal = [os.path.join(folder_path_mal, filename) for filename in os.listdir(folder_path_mal)]

X_tr = []
# common_size = (128, 128)

for num, image_file in enumerate(image_files_benign):
    img = cv2.imread(image_file)
    if img is not None:
        # Resize the image to the common size
        # img = cv2.resize(img, common_size)
        # X_tr.append(img)
        cv2.imwrite('Image_1.png', img)
        common_size = (128,128)
        img_2 = cv2.resize(img, common_size)
        cv2.imwrite('Image_2.png',img_2)
        break
    else:
        print(f"Error loading image: {image_file}")


