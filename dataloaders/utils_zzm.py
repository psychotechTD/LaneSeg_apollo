#该文件主要作用是对数据集进行编解码
import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset='apollo'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))#tensor形式的颜色图标记
    return rgb_masks



def get_apollo_color_labels():
    return np.asarray([[[0, 0, 0], [0, 153, 153], [255, 255, 255],[128, 0, 128], [0, 0, 60], [0, 60, 100], [244, 35, 232], [0, 0, 160],[153, 153, 153], [250, 170, 30],[102, 102, 156], [128, 0, 0], [238, 232, 170],[0, 0, 230], [255, 165, 0], [0, 191, 255], [51, 255, 51], [250, 128, 114], [127, 255, 0], [0, 255, 255], [128, 128, 64]], #0
                       [[70, 130, 180], [220, 20, 60],[255, 0, 0]], #1
                       [[0, 0, 142], [119, 11, 32]],#2
                       [[220, 220, 0]],#3
                       [[128, 64, 128]],#4
                       [[190, 153, 153]],#5
                       [[128, 128, 0], [128, 78, 160], [150, 100, 100], [180, 165, 180], [107, 142, 35], [201, 255, 229]],#6
                       [[255, 128, 0], [178, 132, 190], [102, 0, 204]]#7
                       ])

def get_apollo_gray_labels():
    return np.asarray([[0, 249, 255, 213, 206, 207, 211, 208, 216, 215,218, 219,232, 202, 231, 230, 228, 229, 233, 212, 223],
                       [200, 204, 209], 
                       [201,203], 
                       [217], 
                       [210], 
                       [214],
                       [220,221,222,224,225,226],
                       [205,227,250]])

def encode_segmap(mask):#颜色图转类别标签
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, _label in enumerate(get_apollo_color_labels()):
        for label in _label:
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

def decode_segmap(label_mask, dataset, plot=False):#类别标签转颜色图标记
    if dataset == 'apollo':
        n_classes = 8
        label_colours = get_apollo_color_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll][0][0]
        g[label_mask == ll] = label_colours[ll][0][1]
        b[label_mask == ll] = label_colours[ll][0][2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    # rgb[:, :, 0] = r / 255.0
    # rgb[:, :, 1] = g / 255.0
    # rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
        return rgb
    else:
        return rgb

def encode_segmap_gray_apollo(mask):
    mask = mask.astype(np.int16)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii , _label in enumerate(get_apollo_gray_labels()):
        for label in _label:
            label_mask[mask == label] = ii
    label_mask = label_mask.astype(np.int16)
    return label_mask

def decode_segmap_gray_apollo(label_mask, plot = False):
    n_classes = 8
    label_gray = get_apollo_gray_labels()
    gray = np.zeros((label_mask.shape[0], label_mask.shape[1]))
    for ll in range(0, n_classes):
        gray[label_mask == ll] = label_gray[ll][0]
    if plot:
        plt.subplot(1,2,2)
        plt.imshow(gray)
        plt.show()
    else:
        return gray
"""
Label( 'void' , 0 , 0, 'void' , 0 , False , False , ( 0, 0, 0) ),
Label( 's_w_d' , 200 , 1 , 'dividing' , 1 , False , False , ( 70, 130, 180) ),
Label( 's_y_d' , 204 , 1 , 'dividing' , 1 , False , False , (220, 20, 60) ),
Label( 'ds_w_dn' , 213 , 1 , 'dividing' , 1 , False , True , (128, 0, 128) ),
Label( 'ds_y_dn' , 209 , 1 , 'dividing' , 1 , False , False , (255, 0, 0) ),
Label( 'sb_w_do' , 206 , 1 , 'dividing' , 1 , False , True , ( 0, 0, 60) ),
Label( 'sb_y_do' , 207 , 1 , 'dividing' , 1 , False , True , ( 0, 60, 100) ),
Label( 'b_w_g' , 201 , 2 , 'guiding' , 2 , False , False , ( 0, 0, 142) ),
Label( 'b_y_g' , 203 , 2 , 'guiding' , 2 , False , False , (119, 11, 32) ),
Label( 'db_w_g' , 211 , 2 , 'guiding' , 2 , False , True , (244, 35, 232) ),
Label( 'db_y_g' , 208 , 2 , 'guiding' , 2 , False , True , ( 0, 0, 160) ),
Label( 'db_w_s' , 216 , 3 , 'stopping' , 3 , False , True , (153, 153, 153) ),
Label( 's_w_s' , 217 , 3 , 'stopping' , 3 , False , False , (220, 220, 0) ),
Label( 'ds_w_s' , 215 , 3 , 'stopping' , 3 , False , True , (250, 170, 30) ),
Label( 's_w_c' , 218 , 4 , 'chevron' , 4 , False , True , (102, 102, 156) ),
Label( 's_y_c' , 219 , 4 , 'chevron' , 4 , False , True , (128, 0, 0) ),
Label( 's_w_p' , 210 , 5 , 'parking' , 5 , False , False , (128, 64, 128) ),
Label( 's_n_p' , 232 , 5 , 'parking' , 5 , False , True , (238, 232, 170) ),
Label( 'c_wy_z' , 214 , 6 , 'zebra' , 6 , False , False , (190, 153, 153) ),
Label( 'a_w_u' , 202 , 7 , 'thru/turn' , 7 , False , True , ( 0, 0, 230) ),
Label( 'a_w_t' , 220 , 7 , 'thru/turn' , 7 , False , False , (128, 128, 0) ),
Label( 'a_w_tl' , 221 , 7 , 'thru/turn' , 7 , False , False , (128, 78, 160) ),
Label( 'a_w_tr' , 222 , 7 , 'thru/turn' , 7 , False , False , (150, 100, 100) ),
Label( 'a_w_tlr' , 231 , 7 , 'thru/turn' , 7 , False , True , (255, 165, 0) ),
Label( 'a_w_l' , 224 , 7 , 'thru/turn' , 7 , False , False , (180, 165, 180) ),
Label( 'a_w_r' , 225 , 7 , 'thru/turn' , 7 , False , False , (107, 142, 35) ),
Label( 'a_w_lr' , 226 , 7 , 'thru/turn' , 7 , False , False , (201, 255, 229) ),
Label( 'a_n_lu' , 230 , 7 , 'thru/turn' , 7 , False , True , (0, 191, 255) ),
Label( 'a_w_tu' , 228 , 7 , 'thru/turn' , 7 , False , True , ( 51, 255, 51) ),
Label( 'a_w_m' , 229 , 7 , 'thru/turn' , 7 , False , True , (250, 128, 114) ),
Label( 'a_y_t' , 233 , 7 , 'thru/turn' , 7 , False , True , (127, 255, 0) ),
Label( 'b_n_sr' , 205 , 8 , 'reduction' , 8 , False , False , (255, 128, 0) ),
Label( 'd_wy_za' , 212 , 8 , 'attention' , 8 , False , True , ( 0, 255, 255) ),
Label( 'r_wy_np' , 227 , 8 , 'no parking' , 8 , False , False , (178, 132, 190) ),
Label( 'vom_wy_n' , 223 , 8 , 'others' , 8 , False , True , (128, 128, 64) ),
Label( 'om_n_n' , 250 , 8 , 'others' , 8 , False , False , (102, 0, 204) ),
Label( 'noise' , 249 , 0 , 'ignored' , 0 , False , True , ( 0, 153, 153) ),
Label( 'ignored' , 255 , 0 , 'ignored' , 0 , False , True , (255, 255, 255) ),
]
"""


def decode_color_labels(labels):
    decode_mask = np.zeros((3, labels.shape[0], labels.shape[1]), dtype='uint8')
    # 0
    decode_mask[0][labels == 0] = 0
    decode_mask[1][labels == 0] = 0
    decode_mask[2][labels == 0] = 0
    # 1
    decode_mask[0][labels == 1] = 70
    decode_mask[1][labels == 1] = 130
    decode_mask[2][labels == 1] = 180
    # 2
    decode_mask[0][labels == 2] = 0
    decode_mask[1][labels == 2] = 0
    decode_mask[2][labels == 2] = 142
    # 3
    decode_mask[0][labels == 3] = 153
    decode_mask[1][labels == 3] = 153
    decode_mask[2][labels == 3] = 153
    # 4
    decode_mask[0][labels == 4] = 128
    decode_mask[1][labels == 4] = 64
    decode_mask[2][labels == 4] = 128
    # 5
    decode_mask[0][labels == 5] = 190
    decode_mask[1][labels == 5] = 153
    decode_mask[2][labels == 5] = 153
    # 6
    decode_mask[0][labels == 6] = 0
    decode_mask[1][labels == 6] = 0
    decode_mask[2][labels == 6] = 230
    # 7
    decode_mask[0][labels == 7] = 255
    decode_mask[1][labels == 7] = 128
    decode_mask[2][labels == 7] = 0

    return decode_mask