#数据格式：  data = {'users' : [str(1, 2, 3, ..., n)], 'user_data' : {'1' : {1号user的数据}, '2' : {2号user的数据}, ..., 'n' : {n号user的数据}}}
#1, ..., n号user的数据格式为： {'x' : [m *28 *28的列表], 'y' : [m个标签]},其中m表示每个user拥有的数据

import json
import struct
import numpy as np
import math

def _data_norm(data):
    mean = np.mean(data)
    std_variance = math.sqrt(np.var(data))
    norm_data = (data - mean) / std_variance
    return norm_data
#对数据进行标准化
def data_norm(data):
    data_list = []
    for i in range(len(data)):
        norm_data_list = _data_norm(data[i]).tolist()
        data_list.append(norm_data_list)
        
    return data_list
    
#改成自己的路径
with open('../Downloads/binary_data/t10k-labels.idx1-ubyte', mode = 'rb') as test_lable_file:
    
    magic , num_of_test_lables_item = struct.unpack('>II', test_lable_file.read(8)) #前面8个字节不是真正的数据
    test_y_total_np = np.fromfile(test_lable_file, dtype = np.uint8) 
    test_y_total_list = test_y_total_np.tolist()  #现在读的才是真正的数据，dtype = np.uint8表示数据是按照字节读取的
    
with open('../Downloads/binary_data/t10k-images.idx3-ubyte', mode = 'rb') as test_image_file:
    magic , num_of_test_images_item, rows, columns = struct.unpack('>IIII', test_image_file.read(16)) #前16个字节不是真正的数据
#     test_x_total = np.fromfile(test_image_file, dtype = np.uint8)    
    test_x_total_np = np.fromfile(test_image_file, dtype = np.uint8).reshape(num_of_test_images_item, 28 * 28)#这里还没标准化
    test_x_total_list = data_norm(test_x_total_np)#这里进行了标准化

with open('../Downloads/binary_data/train-images.idx3-ubyte', mode = 'rb') as train_image_file:
    magic, num_of_train_images_item, _, _ = struct.unpack('>IIII', train_image_file.read(16))
#     train_x_total = np.fromfile(train_image_file, dtype = np.uint8)
    train_x_total_np = np.fromfile(train_image_file, dtype = np.uint8).reshape(num_of_train_images_item, 28 * 28)
    train_x_total_list = data_norm(train_x_total_np)

with open('../Downloads/binary_data/train-labels.idx1-ubyte', mode = 'rb') as train_label_file:
    magic, num_of_train_lables_item = struct.unpack('>II', train_label_file.read(8))
    train_y_total_np = np.fromfile(train_label_file, dtype = np.uint8)
    train_y_total_list = train_y_total_np.tolist()

#前面做的是读取文件数据并将其转化为list格式的数据，现在开始组装数据
#数据格式：  data = {'users' : [str(1, 2, 3, ..., n)], 'user_data' : {'1' : {1号user的数据}, '2' : {2号user的数据}, ..., 'n' : {n号user的数据}}}
#1, ..., n号user的数据格式为： {'x' : [m *28 *28的列表], 'y' : [m个标签]}, 其中m表示每个user拥有的数据，test数据集m = 20，train数据集m = 120
def _read_data(batch_size, user_num, data_x, data_y):
    user_num = user_num
    all_users_list = []
    batch_size = batch_size
    all_user_data = {}
    data_per_user = {}
    for i in range(user_num):
        all_users_list.append(str(i))
        data_per_user['x'] = data_x[i * batch_size : (i + 1)* batch_size]
        data_per_user['y'] = data_y[i * batch_size : (i + 1)* batch_size]
        all_user_data[str(i)] =  data_per_user
    #print(every_user_data['3'])
    
    #最后组装下数据
    data = {}
    data['users'] = all_users_list
    data['user_data'] = all_user_data
    #print(data_100_users.keys())
    return data

def read_data(mode, user_num = 500):  #默认user_num是500，注意test数据集共有10000条数据，因此batch_size * user_num要等于10000，同理train数据集
    if mode == "test":
        return _read_data(batch_size = 20, user_num = user_num, data_x = test_x_total_list, data_y = test_y_total_list)
    elif mode == "train":
        return _read_data(batch_size = 120, user_num = user_num, data_x = train_x_total_list, data_y = train_y_total_list)

#最后导出为json格式的文件,用的时候取消掉注释就行
# json.dump(read_data('test'), open("test_data.json", "w"))
# json.dump(read_data('train'), open("train_data.json", "w"))
