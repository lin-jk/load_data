import copy
import json
import os
import random
import struct

import numpy as np

#python对于多维list的切片操作值得注意，等会儿细说
#创建两个空箱，每个箱子有十类，分别对应0-9的label
bin_train = []
for i in range(10):
    bin_train.append([])

bin_test = copy.deepcopy(bin_train)

with open("C:\\Users\\林俊锞\\Desktop\\mnist\\binary_mnist\\t10k-labels.idx1-ubyte", mode = 'rb') as test_lable_file:
    magic , num_of_test_lables_item = struct.unpack('>II', test_lable_file.read(8)) #前面8个字节不是真正的数据
    test_y_total = np.fromfile(test_lable_file, dtype = np.uint8)  #现在读的才是真正的数据，dtype = np.uint8表示数据是按照字节读取的

with open('C:\\Users\\林俊锞\\Desktop\\mnist\\binary_mnist\\t10k-images.idx3-ubyte', mode = 'rb') as test_image_file:
    magic , num_of_test_images_item, rows, columns = struct.unpack('>IIII', test_image_file.read(16)) #前16个字节不是真正的数据
    test_x_total = np.fromfile(test_image_file, dtype = np.uint8)    
    test_x_total = test_x_total.reshape(num_of_test_images_item, 28 * 28)

with open('C:\\Users\\林俊锞\\Desktop\\mnist\\binary_mnist\\train-images.idx3-ubyte', mode = 'rb') as train_image_file:
    magic, num_of_train_images_item, _, _ = struct.unpack('>IIII', train_image_file.read(16))
    train_x_total = np.fromfile(train_image_file, dtype = np.uint8)
    train_x_total = train_x_total.reshape(num_of_train_images_item, 28 * 28)

with open('C:\\Users\\林俊锞\\Desktop\\mnist\\binary_mnist\\train-labels.idx1-ubyte', mode = 'rb') as train_label_file:
    magic, num_of_train_lables_item = struct.unpack('>II', train_label_file.read(8))
    train_y_total = np.fromfile(train_label_file, dtype = np.uint8)

#遍历标签数据集，把0-9的标签对应的索引放入对应的分箱里
for i in range(len(train_y_total)):
    label = train_y_total[i]
    bin_train[label].append(i)

for i in range(len(test_y_total)):
    label = test_y_total[i]
    bin_test[label].append(i)


def _get_each_user_data(label_1:int, label_2:int, bin, x):

    each_user_data = {}
    data = []
    label = []
    #把数据和标签从箱子里拿出来
    for i in bin[label_1]:
        data.append(x[i])
        label.append(label_1)
    for j in bin[label_2]:
        data.append(x[j])
        label.append(label_2)

    #做一个同序shuffle，打乱顺序
    random.seed(123)
    temp_list = list(zip(data, label))
    random.shuffle(temp_list)
    data, label = zip(*temp_list)
    #组装
    each_user_data['x'] = data
    each_user_data['y'] = label

    return each_user_data
#组装成最终结构
def get_data(user_num, user_label_mapping_dict, bin, x):
    data = {}
    total_user_data = {}
    data['users'] = [i for i in range(user_num)]
    
    for user, label_set in user_label_mapping_dict.items():
        label_1, label_2 = label_set
        total_user_data[user] = _get_each_user_data(label_1, label_2, bin, x = x)

    data['user_data'] = total_user_data
    return data

#假设user0的label为1，7， user1的label为5，6， user2的label为2，6， user3的label为0，8， user4的label为3，9
user_num = 5
user_label_dict = {0 : (1, 7), 1 : (5, 6), 2 : (2, 6), 3 : (0, 8), 4 : (3, 9)}

train_100percent_noniid = get_data(user_num = user_num, user_label_mapping_dict = user_label_dict, bin = bin_train, x = train_x_total)
test_100percent_noniid = get_data(user_num = user_num, user_label_mapping_dict = user_label_dict, bin = bin_test, x = test_x_total)

json.dump(train_100percent_noniid, open("test_data.json", "w"))
json.dump(test_100percent_noniid, open("train_data.json", "w"))
