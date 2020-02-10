import os
import pandas as pd
from sklearn.utils import shuffle


label_list = []
image_list = []

image_dir = '/home/aistudio/data/Image_Data/'
label_dir = '/home/aistudio/data/Gray_Label/'

# ColorImage
for s1 in os.listdir(image_dir):
    # image_dir/road02
    image_sub_dir1 = os.path.join(image_dir, s1)
    # label_dir/label_road02/label
    label_sub_dir1 = os.path.join(label_dir, 'Label_' + str.lower(s1), 'Label')

    # road2
    for s2 in os.listdir(image_sub_dir1):
        # image_dir/road02/record001
        image_sub_dir2 = os.path.join(image_sub_dir1, s2)
        # label_dir / label_road02 /label/record001
        label_sub_dir2 = os.path.join(label_sub_dir1, s2)

        # Record001
        for s3 in os.listdir(image_sub_dir2):
            #image_dir/road02/record001/camera 5
            image_sub_dir3 = os.path.join(image_sub_dir2, s3)
            label_sub_dir3 = os.path.join(label_sub_dir2, s3)

            # Camera 5
            for s4 in os.listdir(image_sub_dir3):
                s44 = s4.replace('.jpg','_bin.png')
                #
                image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                label_sub_dir4 = os.path.join(label_sub_dir3, s44)
                if not os.path.exists(image_sub_dir4):
                    print(image_sub_dir4)
                    continue
                if not os.path.exists(label_sub_dir4):
                    print(label_sub_dir4)
                    continue
                image_list.append(image_sub_dir4)
                label_list.append(label_sub_dir4)

assert len(image_list) == len(label_list)
print("The length of image dataset is {}, and label is {}".format(len(image_list), len(label_list)))
total_length = len(image_list)
sixth_part = int(total_length*0.6)
eighth_part = int(total_length*0.8)

all = pd.DataFrame({'image':image_list, 'label':label_list})
all_shuffle = shuffle(all)

train_dataset = all_shuffle[:sixth_part]
val_dataset = all_shuffle[sixth_part:eighth_part]
test_dataset = all_shuffle[eighth_part:]

train_dataset.to_csv('../data_list/train_lite.csv', index=False)
val_dataset.to_csv('../data_list/val_lite.csv', index=False)
test_dataset.to_csv('../data_list/test_lite.csv', index=False)
