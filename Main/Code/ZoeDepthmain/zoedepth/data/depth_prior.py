import torch
from torch.distributions.normal import Normal
from matplotlib import pyplot as plt
import random
import numpy as np
def get_distance_maps(height, width, idcs_height, idcs_width, device="cpu"):
    """Returns a SxHxW tensor that captures the euclidean pixel distance to S
    sample pixels with coordinates (h,w)."""

    dist_maps = torch.empty(0, height, width).to(device)
    for idx_height, idx_width in zip(idcs_height, idcs_width):

        # vertical and horizontal distance vectors
        height_dists = torch.arange(0, height, device=device) - idx_height
        width_dists = torch.arange(0, width, device=device) - idx_width

        # vertical and horizontal distance maps
        height_dist_map = height_dists.repeat(width, 1).transpose(0, 1)
        width_dist_map = width_dists.repeat(height, 1)

        # distance map
        dist_map = torch.sqrt(
            torch.pow(height_dist_map, 2) + torch.pow(width_dist_map, 2)
        ).unsqueeze(0)

        dist_maps = torch.cat((dist_maps, dist_map), dim=0)

    return dist_maps


def get_probability_maps(dist_map):
    """Takes a Nx1xHxW distance map as input and outputs a probability map.
    Pixels with small distance to closest keypoint have big probability and vice versa."""

    # normal distribution
    distribution = Normal(loc=0.0, scale=15.0)
    scale = torch.exp(
        distribution.log_prob(torch.zeros(1, device=dist_map.device))
    )  # used to enfore prob=1 at dist=0

    #  prior probability for every pixel
    prob_map = torch.exp(distribution.log_prob(dist_map)) / scale

    # exponential distribution
    # r = 0.05  # rate
    # prob_map = torch.exp(
    #     -r * dist_map
    # )  # dont multiply with r to have prior=1 at dist=0

    return prob_map


def get_depth_prior_from_ground_truth(
    targets, n_samples=200, mu=0.0, std=1.0, masks=None, device="cpu"
):
    """Takes an Nx1xHxW ground truth depth tensor and desired number of samples,
    returns two images per batch represention a prior guess parametrization:

    - One image represents a mosaic representing the nearest neighbor guess.
    By default, sampled depth values are normalized.
    - The other image represents a probability map.

    Inpired by: https://arxiv.org/abs/1804.02771
    """

    # batch size
    batch_size = targets.size(0)

    # output size
    height = targets.size(2)
    width = targets.size(3)
    n_pixels = height * width

    # depth prior maps
    prior_maps = torch.empty(batch_size, 1, height, width).to(device)

    # euclidean distance maps
    distance_maps = torch.empty(batch_size, 1, height, width).to(device)

    # features lists with pixel indices and depth values
    features = torch.empty(batch_size, n_samples, 3).to(device)

    # utility to select pixel locations and avoid slow torch.where
    pixel_idcs = torch.arange(n_pixels).to(device)

    # for each image
    for i in range(batch_size):

        # identify valid pixels
        if masks is not None:
            valid_pixel_idcs = pixel_idcs[masks[i, 0, ...].flatten()]
            n_valid_pixels = len(valid_pixel_idcs)
            if n_valid_pixels < n_samples:
                print(
                    f"WARNING: Could not find enough valid pixels in depth map. "
                    + f"Need at least {n_samples}, but found only {n_valid_pixels} samples. "
                    + f"Reducing n_samples to {n_valid_pixels} for this batch."
                )
                n_samples = n_valid_pixels
                features = features[:, :n_samples, :]
        else:
            valid_pixel_idcs = pixel_idcs

        # get random indices
        idcs_selection = torch.randperm(valid_pixel_idcs.size(0))[:n_samples]
        idcs = valid_pixel_idcs[idcs_selection]

        # convert flattened indices to height and width indices
        idcs_height = idcs.div(width, rounding_mode="floor")
        idcs_width = idcs.remainder(width)

        # get n_samples x height x width dist maps
        sample_dist_maps = get_distance_maps(
            height, width, idcs_height, idcs_width, device=device
        )

        # find min and argmin
        dist_map_min, dist_argmin = torch.min(sample_dist_maps, dim=0, keepdim=True)

        # sample depth priors at indices
        depth_values = targets[i, 0, idcs_height, idcs_width]

        # nearest neighbor prior map
        prior_map = depth_values[dist_argmin]  # 1xHxW

        # concat
        prior_maps[i, ...] = prior_map
        distance_maps[i, ...] = dist_map_min
        features[i, :, 0] = idcs_height
        features[i, :, 1] = idcs_width
        features[i, :, 2] = depth_values

    # probability model:
    # convert pixel distance to probability
    signal_strength_maps = get_probability_maps(distance_maps)

    # parametrization
    parametrization = torch.cat((prior_maps, signal_strength_maps), dim=1)  # Nx2xHxW

    return parametrization, features


def get_depth_prior_from_features(
    features,
    height=240,
    width=320,
):
    """Takes lists of pixel indices and their respective depth probes and
    returns a dense depth prior parametrization.


    - One image represents the nearest neighbor guess (Inpired by: https://arxiv.org/abs/1804.02771).
    - The other image represents a probability map."""

    batch_size = features.size(0)

    # depth prior maps
    prior_maps = torch.empty(batch_size, 1, height, width).to(features.device)

    # euclidean distance maps
    distance_maps = torch.empty(batch_size, 1, height, width).to(features.device)

    # for every img, cannot vectorize because of masks with unequal length
    # (different images may have different number of features)
    for i in range(batch_size):

        # use only entries with valid depth
        mask = features[i, :, 2] > 0.0

        if not mask.any():
            max_dist = torch.sqrt(torch.pow(height, 2) + torch.pow(width, 2))
            prior_maps[i, ...] = 0.0
            distance_maps[i, ...] = max_dist
            print(
                "WARNING: Img has no valid features (depth > 0.0), using "
                + f"placeholder as parametrization (mosaic=0.0, dist={max_dist})."
            )
            continue

        # get list of indices and depth values
        idcs_height = features[i, mask, 0].round().long()
        idcs_width = features[i, mask, 1].round().long()
        depth_values = features[i, mask, 2]

        # get n_samples x height x width dist maps
        # (needs quite a bit of memory but is faster than iterating over every pixel)
        sample_dist_maps = get_distance_maps(
            height, width, idcs_height, idcs_width, device=features.device
        )
        # find min and argmin
        dist_map_min, dist_argmin = torch.min(sample_dist_maps, dim=0, keepdim=True)

        # nearest neighbor prior map
        prior_map = depth_values[dist_argmin]  # 1xHxW

        # concat
        prior_maps[i, ...] = prior_map
        distance_maps[i, ...] = dist_map_min

    # probability model:
    # convert pixel distance to probability
    prior_probability_maps = get_probability_maps(distance_maps)

    # parametrization
    parametrization = torch.cat((prior_maps, prior_probability_maps), dim=1)  # Nx2xHxW

    return parametrization

def concat_sparse_and_prior(
    features,
    height=240,
    width=320,
):
    """Takes lists of pixel indices and their respective depth probes and
    returns a dense depth prior parametrization.


    - One image represents the nearest neighbor guess (Inpired by: https://arxiv.org/abs/1804.02771).
    - The other image represents a probability map."""
    # rel = rel.unsqueeze(0)
    batch_size = features.size(0)
    # init_feat = features.size(1)-3
    # euclidean distance maps
    distance_maps = torch.empty(batch_size, 4, height, width).to(features.device)
    features_18 = features[:, :18, :]
    features_3 = features[:, -3:, :]
    # for every img, cannot vectorize because of masks with unequal length
    # (different images may have different number of features)
    for i in range(batch_size):
        mask = features_18[i, :, 2] > 0.0
        mask_3 = features_3[i, :, 2] > 0.0
        if not mask.any():
            max_dist = torch.sqrt(torch.pow(height, 2) + torch.pow(width, 2))
            # prior_maps[i, ...] = 0.0
            distance_maps[i, ...] = max_dist
            print(
                "WARNING: Img has no valid features (depth > 0.0), using "
                + f"placeholder as parametrization (mosaic=0.0, dist={max_dist})."
            )
            continue

        # get list of indices and depth values
        idcs_height = features_18[i, mask, 0].round().long()
        idcs_width = features_18[i, mask, 1].round().long()

        idcs_height_3 = features_3[i, mask_3, 0].round().long()
        idcs_width_3 = features_3[i, mask_3, 1].round().long()

        # get n_samples x height x width dist maps
        # (needs quite a bit of memory but is faster than iterating over every pixel)
        sample_dist_maps = get_distance_maps(
            height, width, idcs_height, idcs_width, device=features.device
        )

        sample_dist_maps_3 = get_distance_maps(
            height, width, idcs_height_3, idcs_width_3, device=features.device
        )
        # find min and argmin
        dist_map_min, dist_argmin = torch.min(sample_dist_maps, dim=0, keepdim=True)
        # dist_map_min_3 = torch.min(sample_dist_maps_3, dim=0, keepdim=True)
        # concat
        dist_map_min = torch.cat([dist_map_min, sample_dist_maps_3], dim=0)
        distance_maps[i, ...] = dist_map_min
    # probability model:
    # convert pixel distance to probability
    prior_probability_maps = get_probability_maps(distance_maps)
    # prior_probability_maps_3 = get_probability_maps(sample_dist_maps_3.unsqueeze(0))
    # 创建一个全0的数组作为深度图的初始状态
    sparse_map_0 = torch.zeros((height, width), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sparse_map_1 = torch.zeros((height, width), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sparse_map_2 = torch.zeros((height, width), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sparse_map_3 = torch.zeros((height, width), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sparse_map_list = [sparse_map_1, sparse_map_2, sparse_map_3]
    concatenated_sparse_map_list = []
    # 遍历数组，填充深度值
    for point in features_18[0]:
        x, y, depth = point
        # 确保坐标在图像尺寸范围内
        if 0 <= x < width and 0 <= y < height:
            sparse_map_0[:, :, int(y), int(x)] = depth
    concatenated_sparse_map_list.append(sparse_map_0)
    for sparse_map in sparse_map_list:
        for point in features_3[0]:
            x, y, depth = point
            # 确保坐标在图像尺寸范围内
            if 0 <= x < width and 0 <= y < height:
                sparse_map[:, :, int(y), int(x)] = depth
        concatenated_sparse_map_list.append(sparse_map)
    concatenated_sparse_map = torch.cat(concatenated_sparse_map_list, dim=1)
    concatenated_sparse_map = concatenated_sparse_map.cuda()
    parametrization = torch.cat((concatenated_sparse_map, prior_probability_maps), dim=1)  # Nx2xHxW
    # 可视化每个通道
    # for i in range(parametrization.shape[1]):
    #     plt.figure(f"Channel {i+1}")
    #     plt.imshow(parametrization[0, i, ...].cpu().numpy())
    #     plt.colorbar()
    #     plt.show()
    return parametrization
def test_get_priors(device="cpu"):
    """Test sparse prior generation and parametrization."""

    print("Testing depth prior parametrization ...")

    # import modules only needed for testing
    import matplotlib.pyplot as plt
    import time

    # generate target ground truth maps to generate prior from
    # in this case, simple gradient images are used
    target1 = torch.linspace(0, 0.5, 320).repeat(240, 1) + torch.linspace(
        0, 0.5, 240
    ).repeat(320, 1).transpose(0, 1)
    target1 = target1[None, None, ...]  # add batch and channel dimension
    target2 = 1.0 - target1
    targets = torch.cat((target1, target1, target2, target2), dim=0).to(device)
    masks = targets > 0.5

    # get priors and dist_map
    starttime = time.time()
    prior, _ = get_depth_prior_from_ground_truth(
        targets,
        n_samples=100,
        mu=0.0,
        std=10.0,
        masks=masks,
        device=device,
    )
    prior_maps = prior[:, 0, ...].unsqueeze(1)
    signal_maps = prior[:, 1, ...].unsqueeze(1)
    elapsed_time = time.time() - starttime
    print(f"sampling time: {elapsed_time} seconds")

    # copy back to cpu for visuals
    targets = targets.cpu()
    prior_maps = prior_maps.cpu()
    signal_maps = signal_maps.cpu()

    # plot
    for i in range(targets.size(0)):

        target = targets[i, ...]
        prior_map = prior_maps[i, ...]
        signal_map = signal_maps[i, ...]
        plt.figure(f"target {i}")
        plt.imshow(target.permute(1, 2, 0))
        plt.figure(f"prior map {i}")
        plt.imshow(prior_map.permute(1, 2, 0))
        plt.figure(f"dist map {i}")
        plt.imshow(signal_map.permute(1, 2, 0))

        print(f"prior map {i} range: [{prior_map.min()}, {prior_map.max()}]")
        print(f"signal map {i} range: [{signal_map.min()}, {signal_map.max()}]")

    plt.show()

    print("Testing depth prior parametrization done.")


def test_get_priors_from_features(device="cpu"):

    # import modules only needed for testing
    import matplotlib.pyplot as plt
    import time

    print("Testing depth prior parametrization from given features ...")

    n_features = 200
    height = 240
    width = 320

    # generate target ground truth maps to generate prior from
    # in this case, simple gradient images are used

    target1 = torch.linspace(0, 0.5, width).repeat(height, 1) + torch.linspace(
        0, 0.5, height
    ).repeat(width, 1).transpose(0, 1)
    target1 = target1[None, None, ...]  # add batch and channel dimension
    target2 = 1.0 - target1
    targets = torch.cat((target1, target1, target2, target2), dim=0).to(device)

    # create some features that should be used for parametrization
    batch_size = targets.size(0)
    features = torch.empty(batch_size, 200, 3)
    for i in range(batch_size):

        # get random locations for fdepth samples
        idcs_height = torch.randperm(height)[:n_features].unsqueeze(1)
        idcs_width = torch.randperm(width)[:n_features].unsqueeze(1)

        # get the depth values at those locations
        depth_values = targets[i, 0, idcs_height, idcs_width]

        print(f"shape idcs height: {idcs_height.shape}")
        print(f"shape depth_values: {depth_values.shape}")

        # concat idcs and values to generate feature tensor
        img_features = torch.cat([idcs_height, idcs_width, depth_values], dim=1)

        # fill in
        features[i, ...] = img_features

    # get parametrization
    prior = get_depth_prior_from_features(features, height=height, width=width)
    prior_maps = prior[:, 0, ...].unsqueeze(1)
    signal_maps = prior[:, 1, ...].unsqueeze(1)

    # plot
    for i in range(targets.size(0)):

        target = targets[i, ...]
        prior_map = prior_maps[i, ...]
        signal_map = signal_maps[i, ...]
        plt.figure(f"target {i}")
        plt.imshow(target.permute(1, 2, 0))
        plt.figure(f"prior map {i}")
        plt.imshow(prior_map.permute(1, 2, 0))
        plt.figure(f"dist map {i}")
        plt.imshow(signal_map.permute(1, 2, 0))

        print(f"prior map {i} range: [{prior_map.min()}, {prior_map.max()}]")
        print(f"signal map {i} range: [{signal_map.min()}, {signal_map.max()}]")

    plt.show()

    print("Testing depth prior parametrization from given features done.")

def random_coordinate_in_region(coordinates, region_top_left=(30,100), region_bottom_right=(210,280)):
    """
    从给定的坐标数组中随机选择一个指定区域内的坐标

    参数：
    - coordinates: 一个 2D 的坐标数组，每行是一个坐标点的 (x, y)
    - region_top_left: 区域左上角坐标 (x1, y1)
    - region_bottom_right: 区域右下角坐标 (x2, y2)

    返回值：
    - selected_coordinate: 选中的坐标 (x, y)
    """

    # 确定区域的边界
    x1, y1 = region_top_left
    x2, y2 = region_bottom_right

    # 筛选出位于指定区域内的坐标
    valid_coordinates = []
    for x, y in coordinates:
        if x1 <= x <= x2 and y1 <= y <= y2:
            valid_coordinates.append([x, y])

    # 从筛选出的坐标中随机选择一个
    if valid_coordinates:
        selected_coordinate = random.choice(valid_coordinates)
        return torch.from_numpy(np.array(selected_coordinate).reshape(1, 2))
    else:
        return None  # 如果指定区域内没有符合条件的坐标，则返回 None


def generate_row_col_depth(pts_depth_0, pts_depth_3, depths):
    row_col_depth_list = []
    for i in range(len(depths)):
        index_depth = depths[i].squeeze()
        if i == 0:
            depth_values_depth = index_depth[pts_depth_0[:, 0].round().long(), pts_depth_0[:, 1].round().long()]
        else:
            depth_values_depth = index_depth[pts_depth_3[:, 0].round().long(), pts_depth_3[:, 1].round().long()]
            # 如果深度值为0，则取前一个有效深度值
            if depth_values_depth == 0:
                depth_values_depth = row_col_depth_list[-1][-1, 2]

        depth_values_depth = depth_values_depth[..., np.newaxis]
        row_col_depth = torch.cat((pts_depth_0, depth_values_depth), dim=1) if i == 0 else torch.cat((pts_depth_3, depth_values_depth), dim=1)
        # 将每次生成的 1x3 数组添加到列表中
        valid_mask_depth = row_col_depth[:, 2] > 0.0
        row_col_depth = row_col_depth[valid_mask_depth, :]
        row_col_depth_list.append(row_col_depth)
        # 将收集到的所有 1x3 数组合并成一个 NumPy 数组
    depth_samples = torch.cat(row_col_depth_list, dim=0)
    # depth_samples = torch.from_numpy(combined_row_col_depth).to("cpu")
    return depth_samples

if __name__ == "__main__":
    test_get_priors()
    # test_get_priors_from_features()
