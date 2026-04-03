import cv2
import numpy as np
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_13 = np.ones((13, 13), np.uint8)
FULL_KERNEL_25 = np.ones((25, 25), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_9 = np.asarray(
    [
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_13 = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

def fill_in_fast(depth_map, max_depth=25.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """
    # depth_map = np.squeeze(depth_map, axis=-1)
    max_depth = np.max(depth_map)
    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    
    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)
    # plt.imshow(depth_map, cmap='inferno_r')
    # plt.colorbar()
    # plt.savefig('filled_depth_map1.png') 
    
    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)
    # plt.imshow(depth_map, cmap='inferno_r')
    # plt.colorbar()
    # plt.savefig('filled_depth_map1.png') 
    
    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

    # Large Fill
    empty_pixels = depth_map < 0.1
    dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
    depth_map[empty_pixels] = dilated[empty_pixels]
    # plt.imshow(depth_map, cmap='inferno_r')
    # plt.colorbar()
    # plt.savefig('filled_depth_map1.png') 
    
    # Median blur
    depth_map = depth_map.astype('float32')  # Cast a float64 image to float32
    depth_map = cv2.medianBlur(depth_map, 5)
    depth_map = depth_map.astype('float64')  # Cast a float32 image to float64
    
    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = depth_map.astype('float32')
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
        depth_map = depth_map.astype('float64')
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]
    # plt.imshow(depth_map, cmap='inferno_r')
    # plt.colorbar()
    # plt.savefig('filled_depth_map1.png') 
    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    # plt.imshow(depth_map, cmap='inferno_r')
    # plt.colorbar()
    # plt.savefig('filled_depth_map1.png') 
    # fill zero value
    mask = (depth_map <= 0.1)
    if np.sum(mask) != 0:
        labeled_array, num_features = label(mask)
        
        for i in range(num_features):
            index = i + 1
            m = (labeled_array == index)
            m_dilate1 = cv2.dilate(1.0*m, FULL_KERNEL_7)
            m_dilate2 = cv2.dilate(1.0*m, FULL_KERNEL_13)
            m_diff = m_dilate2 - m_dilate1
            v = np.mean(depth_map[m_diff>0])
            depth_map = np.ma.array(depth_map, mask=m_dilate1, fill_value=v)
            depth_map = depth_map.filled()
            depth_map = np.array(depth_map)
    else:
        depth_map = depth_map
    # 可视化深度图
    # plt.imshow(depth_map, cmap='inferno_r')
    # plt.colorbar()
    # plt.savefig('filled_depth_map1.png')  # 保存图像
    depth_map = np.expand_dims(depth_map, 0)

    return depth_map


import numpy as np
from scipy.spatial import KDTree

def fill_in_fast_new(depth_map, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """
    # Invert
    valid_pixels = (depth_map > 0.1)
    max_depth = np.max(depth_map)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values using KDTree
    zero_indices = np.argwhere(depth_map < 0.1)
    non_zero_points = np.argwhere(depth_map > 0.1)
    non_zero_values = depth_map[non_zero_points[:, 0], non_zero_points[:, 1]]
    kdtree = KDTree(non_zero_points)
    # print(non_zero_values)
    # for i, j in zero_indices:
    #     _, idx = kdtree.query((i, j))
    #     nearest_point = non_zero_values[idx]
        
    #     depth_map[i, j] = nearest_point
    # 将 zero_indices 拆分成两个一维数组，分别存储 i 和 j 的坐标
    i_coords = zero_indices[:, 0]
    j_coords = zero_indices[:, 1]

    # 使用 KD 树查询最近点的索引
    _, nearest_indices = kdtree.query(zero_indices)

    if len(non_zero_values) > 0:
            nearest_values = non_zero_values[nearest_indices]
            # 将获取的最近值填充回深度图
            depth_map[i_coords, j_coords] = nearest_values
    else:
        # 处理非零值为空的情况
        pass

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]
        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = top_pixel_values[pixel_col_idx]

    # Large Fill
    depth_map = cv2.dilate(depth_map, FULL_KERNEL_31)

    # Median blur
    depth_map = depth_map.astype('float32')
    depth_map = cv2.medianBlur(depth_map, 5)
    depth_map = depth_map.astype('float64')

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        depth_map = depth_map.astype('float32')
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
        depth_map = depth_map.astype('float64')
    elif blur_type == 'gaussian':
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Fill zero value
    mask = (depth_map <= 0.1)
    if np.sum(mask) != 0:
        labeled_array, num_features = label(mask)
        # print(num_features)
        for i in range(1, num_features):
            m = (labeled_array == i)
            m_dilate1 = cv2.dilate(m.astype(np.uint8), FULL_KERNEL_7)
            m_dilate2 = cv2.dilate(m.astype(np.uint8), FULL_KERNEL_13)
            m_diff = m_dilate2 - m_dilate1
            v = np.mean(depth_map[m_diff > 0])
            depth_map[m] = v
    else:
        depth_map = depth_map
    
    # 可视化深度图
    # plt.imshow(depth_map, cmap='jet')
    # plt.colorbar()
    # plt.savefig('filled_depth_map.png')  # 保存图像
    # plt.show()
    depth_map = np.expand_dims(depth_map, 0)
    return depth_map