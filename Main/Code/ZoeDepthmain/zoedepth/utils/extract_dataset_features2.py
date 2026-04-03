# This file goes through all the input rgb images in a dataset
# and extracts matched feature points. For the depth value of the
# feature points, the ground truth value from the dataset is used.

import cv2
import numpy as np
import pandas as pd

import os
from os.path import splitext, join, basename, exists
from os import mkdir
from depth_seg import extract_points
from data.flsea.dataset import (
    get_flsea_dataset,
)  # , get_usod10k_dataset

##########################################
################# CONFIG #################
##########################################

# grid
n_rows, n_cols = 4, 4

# n features
n_keypoints_matching = 20  # num  keypoints for every image for matching
n_keypoints_direct = 20  # num keypoints (direct sampling without matching)
n_keypoints_min = 15  # min keypoints for depth samples

# output shapes
in_height = 480
in_width = 640
out_height = 240
out_width = 320

# get img paths
dataset = get_flsea_dataset(split="test_with_matched_features", shuffle=False)
# dataset = get_ycb_dataset(split="val", shuffle=False)
# dataset = get_example_dataset(shuffle=False)
path_tuples = dataset.path_tuples

# output
rel_folder = "matched_features"
file_ending = "_features.csv"

debug = False
debug_patch = False

##########################################
##########################################
##########################################


# print config
print(f"input shape: {in_width}x{in_height}")
print(f"output shape: {out_width}x{out_height}")
print(f"{n_keypoints_matching} keypoints per img, devided in {n_rows}x{n_cols} cells.")


def debug_draw_keypoints(img, points, color=(0, 255, 0)):
    """img: cv2 img, points: list of points with (row, col) each."""

    # iterate over points
    for point in points.astype(int):

        # cv2 uses x,y axis notation
        xy_tuple = (point[1], point[0])

        # draw circle
        img = cv2.circle(img, xy_tuple, color=color, radius=8, thickness=2)

    return img


def debug_draw_lines(img1, img2, lines, pts, pts_prev):
    """img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines"""

    _, width, _ = img1.shape

    for line, pt, pt_prev in zip(lines, pts, pts_prev):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [width, -(line[2] + line[0] * width) / line[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt.astype(np.int)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt_prev.astype(np.int)), 5, color, -1)

    return img1, img2


def get_keypoints_and_descriptors(img, n_rows, n_cols, n_keypoints):
    """Extract keypoints and descriptors. Img is divided into n_rows x n_cols patches,
    features are extracted for every patch separately to have features more uniform across the img."""

    height, width, _ = img.shape
    n_keypoints_patch = int(n_keypoints / (n_rows * n_cols))

    # edges for patch indices
    row_step = height / n_rows
    col_step = width / n_cols
    row_edges = np.array(range(n_rows + 1), dtype=float)
    col_edges = np.array(range(n_cols + 1), dtype=float)
    row_edges = (row_edges * row_step).astype(int)
    col_edges = (col_edges * col_step).astype(int)

    # sift detector
    detector = cv2.SIFT_create(
        nfeatures=n_keypoints_patch,
        contrastThreshold=0.0,  # dont filter by contrast, instead filter by matching subsequent frames and epipolar constraints
    )

    # extract keypoints for each patch
    kp = []
    descriptors = []
    for i in range(n_rows):
        for j in range(n_cols):

            # create patch from img
            patch = img[
                row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]
            ]

            # extract features for patch
            kp_patch, descriptors_patch = detector.detectAndCompute(patch, None)
            # continue if zero detections
            if len(kp_patch) <= 0:
                continue
            # print(f"Patch has {len(kp_patch)} feature points.")
            # debug: draw patch img
            if debug_patch:
                # store in array, format [row, col] instead of cv2's [x,y]
                kp_np_patch = np.array([[p.pt[1], p.pt[0]] for p in kp_patch])
                # print(f"Patch has {len(kp_patch)} feature points.")
                debug_draw_keypoints(patch, kp_np_patch)
                # cv2.imshow("img", img)
                # cv2.imshow(f"patch img [{i}, {j}]", patch_img)
            # cv2.imshow("img", img)
            # cv2.waitKey()
            # shift kp
            for k in range(len(kp_patch)):
                pt_np_shifted = np.array(kp_patch[k].pt)
                pt_np_shifted += [col_edges[j], row_edges[i]]
                kp_patch[k].pt = tuple(pt_np_shifted)

            kp += kp_patch
            descriptors.append(descriptors_patch)

    descriptors = np.concatenate(descriptors)

    return kp, descriptors


# brute force matcher
bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# for all imgs
n_tuples = len(path_tuples)
i = 0
prev_keypoints = None
prev_descriptors = None
prev_img = None

# main loop
for path_tuple in path_tuples:
    # paths
    rgb_path = path_tuple[0]
    depth_path = path_tuple[1]
    abs_depth_path = path_tuple[3]
    parts = rgb_path.split(os.path.sep)
    target_name = parts[-3]
    folder_name_features = f"E:/depth_estamate1/uw_depth-main/data/flsea/test/{target_name}"
    out_dir = join(folder_name_features, rel_folder)
    out_filename = splitext(basename(rgb_path))[0] + file_ending
    out_path = join(out_dir, out_filename)

    # read imgs
    img = cv2.imread(rgb_path)  # , cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    abs_depth = cv2.imread(abs_depth_path, cv2.IMREAD_UNCHANGED)
    # try:
    # # 在这里执行可能会引发 TypeError 的代码
    #     depth = depth[..., 0] / 255.0
    # except TypeError:
    #     # 如果出现 TypeError 异常，这里可以选择忽略它并继续执行其他代码
    #     pass

    # depth_image_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    # resize imgs
    img = cv2.resize(img, dsize=(in_width, in_height))

    depth = cv2.resize(
        depth,
        dsize=(out_width, out_height),
        interpolation=cv2.INTER_NEAREST,  # make sure depth is not interpolated
    )

    abs_depth = cv2.resize(
        abs_depth,
        dsize=(out_width, out_height),
        interpolation=cv2.INTER_NEAREST,  # make sure depth is not interpolated
    )
    # depth = np.uint8(depth)
    # depth = depth.reshape(240, 320, 1)
    # find points for depth prior
    if prev_keypoints is None:
        pts_depth = extract_points(depth)

    ##################前一帧有关键点，启用相邻帧匹配找点，不是相邻帧没必要用###########################
        # get keypoints and descriptors
    else:
        keypoints, descriptors = get_keypoints_and_descriptors(
            depth, n_rows, n_cols, n_keypoints_matching
        )

        # match descriptors
        matches = bf_matcher.match(descriptors, prev_descriptors)

        # extract points in correct order
        pts_xy = [keypoints[match.queryIdx].pt for match in matches]
        pts_prev_xy = [prev_keypoints[match.trainIdx].pt for match in matches]
        pts_xy = np.array(pts_xy, dtype=np.float32)
        pts_prev_xy = np.array(pts_prev_xy, dtype=np.float32)

        # filter matches with epipolar constraints
        F, mask = cv2.findFundamentalMat(pts_xy, pts_prev_xy, cv2.FM_LMEDS)
        pts_xy_outlier = pts_xy[mask.ravel() == 0]
        pts_prev_xy_outlier = pts_prev_xy[mask.ravel() == 0]
        pts_xy = pts_xy[mask.ravel() == 1]
        pts_prev_xy = pts_prev_xy[mask.ravel() == 1]

        # convert to row x column standard
        pts = np.flip(pts_xy, axis=1)
        pts_prev = np.flip(pts_prev_xy, axis=1)
        pts_outlier = np.flip(pts_xy_outlier, axis=1)
        pts_prev_outlier = np.flip(pts_prev_xy_outlier, axis=1)

        # if there are too few matches just use detected features without matching
        if len(pts) < n_keypoints_min:
            # print(
            #     f"WARNING: img {rgb_path} has {len(pts)} < {n_keypoints_min} features."
            #     + " Using unfiltered one-shot approach..."
            # )
    #
            standalone_keypoints, _ = get_keypoints_and_descriptors(
                depth, n_rows, n_cols, n_keypoints_direct
                    )
            # convert to np (row x column convention)
            pts_depth = np.array([[kp.pt[1], kp.pt[0]] for kp in standalone_keypoints])
        # print(f"img now has {len(pts)} features.")

    # index transform to fit output shape
    #     height_scale = out_height / in_height
    #     width_scale = out_width / in_width
        # pts_scaled_rgb = pts_rgb * [height_scale, width_scale]

        # get depth values
        # depth_norm = depth[..., 0] / 255.0
        # depth_values_rgb = abs_depth[
        #     pts_scaled_rgb[:, 0].round().astype(int),
        #     pts_scaled_rgb[:, 1].round().astype(int),
        # ]
        # depth_values_rgb = depth_values_rgb[..., np.newaxis]

    depth_values_depth = abs_depth[
        pts_depth[:, 0].round().astype(int),
        pts_depth[:, 1].round().astype(int),
    ]
    depth_values_depth = depth_values_depth[..., np.newaxis]
    # zeros_mask_rgb = depth_values_rgb[:, 0] == 0.0
    zeros_mask_depth = depth_values_depth[:, 0] == 0.0
###把深度图特征提取出来
    # concat
    # row_col_rgb = np.hstack((pts_scaled_rgb, depth_values_rgb))
    row_col_depth = np.hstack((pts_depth, depth_values_depth))
    # filter out features where depth is invalid (= 0)
    # valid_mask_rgb = row_col_rgb[:, 2] > 0.0
    valid_mask_depth = row_col_depth[:, 2] > 0.0
    # row_col_rgb = row_col_rgb[valid_mask_rgb, :]
    row_col_depth = row_col_depth[valid_mask_depth, :]

# write output file
    if not debug:
        if not exists(out_dir):
            mkdir(out_dir)
        ##################row_col_rgb从rgb提取特征，row_col_depth从depth提取特征
        pd.DataFrame(row_col_depth).to_csv(
            out_path, header=["row", "column", "depth"], index=None
        )

# if i % 12 == 1:
# print(f"processed {i}/{n_tuples}: {len(row_col_depth)} priors")
        print(f"processed {i} features_csv")
        i += 1

# debug prints, plots, and outputs
    if debug:

    # print number of priors
    # print(
    #     f"Found {len(depth_values_rgb)} rgb_keypoints ({len(row_col_rgb)} in mask) --> {len(row_col_rgb)} priors"
    # )
    # print(
    #     f"Found {len(depth_values_depth)} depth_keypoints ({len(row_col_depth)} in mask) --> {len(row_col_depth)} priors"
    # )

    # prepare visualization imgs
        img_kp = img.copy()
        img_epi = img.copy()
        img_full = img.copy()
        img_full_cv2 = img.copy()

        depth_kp = depth.copy()
        depth_epi = depth.copy()
        depth_full = depth.copy()
        depth_full_cv2 = depth.copy()

    # show img with features
    #     zeros_mask_rgb = depth_values_rgb[:, 0] == 0.0  # mask where depth is zero
    #     img_kp = debug_draw_keypoints(img_kp, pts_rgb[~zeros_mask_rgb, :], color=(0, 255, 0))
    #     img_kp = debug_draw_keypoints(img_kp, pts_rgb[zeros_mask_rgb, :], color=(0, 0, 255))
    # pts_rgb = pts_rgb[~zeros_mask_rgb, :]
    # 设置网格大小和颜色
        grid_size = 4
        grid_color = (255, 255, 255)  # 白色

    # 计算网格线的间距
        horizontal_spacing = img_kp.shape[1] // grid_size
        vertical_spacing = img_kp.shape[0] // grid_size

    # 绘制水平线
        for i in range(1, grid_size):
            start_point = (0, i * vertical_spacing)
            end_point = (img_kp.shape[1], i * vertical_spacing)
            cv2.line(img_kp, start_point, end_point, grid_color, 1)

        # 绘制垂直线
        for i in range(1, grid_size):
            start_point = (i * horizontal_spacing, 0)
            end_point = (i * horizontal_spacing, img_kp.shape[0])
            cv2.line(img_kp, start_point, end_point, grid_color, 1)
        cv2.imshow("img with features", img_kp)

    # show depth with features
    # depth = (depth - depth.min()) / (depth.max() - depth.min())
    # depth = cv2.convertScaleAbs(depth, alpha=(255.0/65535.0))
    # depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    # depth_kp = depth_kp * 255
        depth_kp = cv2.applyColorMap(depth_kp.astype(np.uint8), cv2.COLORMAP_INFERNO)

        zeros_mask_depth = depth_values_depth[:, 0] == 0.0  # mask where depth is zero
        depth_kp = debug_draw_keypoints(depth_kp, pts_depth[~zeros_mask_depth, :], color=(0, 255, 0))
        depth_kp = debug_draw_keypoints(depth_kp, pts_depth[zeros_mask_depth, :], color=(0, 0, 255))

        horizontal_spacing2 = depth_kp.shape[1] // grid_size
        vertical_spacing2 = depth_kp.shape[0] // grid_size

    # 绘制水平线
        for i in range(1, grid_size):
            start_point = (0, i * vertical_spacing2)
            end_point = (depth_kp.shape[1], i * vertical_spacing2)
            cv2.line(depth_kp, start_point, end_point, grid_color, 1)

        # 绘制垂直线
        for i in range(1, grid_size):
            start_point = (i * horizontal_spacing2, 0)
            end_point = (i * horizontal_spacing2, depth_kp.shape[0])
            cv2.line(depth_kp, start_point, end_point, grid_color, 1)
        cv2.imshow("depth with features", depth_kp)
    # cv2.imshow("depth", depth)
    # if not exists(out_dir):
    #     mkdir(out_dir)
    # ##################row_col_rgb从rgb提取特征，row_col_depth从depth提取特征
    # pd.DataFrame(pts_rgb[~zeros_mask_rgb, :]).to_csv(
    #     out_path, header=["row", "column", "depth"], index=None)
    # show comparison to detection on full img
        sift_detector_full = cv2.SIFT_create(
            nfeatures=n_keypoints_direct, contrastThreshold=0.0
        )
        kp_full = sift_detector_full.detect(img, None)
        kp_np_full = np.array([[p.pt[1], p.pt[0]] for p in kp_full])
        img_full = debug_draw_keypoints(img_full, kp_np_full)
        img_full_cv2 = cv2.drawKeypoints(img_full_cv2, kp_full, img_full_cv2)
        cv2.imshow("detected points on full img (no patches)", img_full)
        cv2.imshow("verification_img", img_full_cv2)

        # draw matches and epipolar
        # if prev_keypoints is not None:
        #
        #     match_color = (0, 255, 255)
        #
        #     img_kp_raw = img.copy()
        #     img_prev_kp_raw = prev_img.copy()
        #     img_kp_epifilt = img.copy()
        #     img_kp_prev_epifilt = prev_img.copy()
        #
        #     # draw raw kp
        #     img_kp_raw = cv2.drawKeypoints(
        #         img_kp_raw, keypoints, img_kp_raw, color=match_color
        #     )
        #     img_prev_kp_raw = cv2.drawKeypoints(
        #         img_prev_kp_raw, prev_keypoints, img_prev_kp_raw, color=match_color
        #     )
        #
        #     # draw bf matches
        #     img_matches = cv2.drawMatches(
        #         img,
        #         keypoints,
        #         prev_img,
        #         prev_keypoints,
        #         matches,
        #         None,
        #         matchColor=match_color,
        #         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        #     )
        #
        #     # draw filtered bf matches
        #     img_kp_epifilt = debug_draw_keypoints(
        #         img_kp_epifilt, pts, color=(0, 255, 0)
        #     )
        #     img_kp_prev_epifilt = debug_draw_keypoints(
        #         img_kp_prev_epifilt, pts_prev, color=(0, 255, 0)
        #     )
        #
        #     good_matches = np.array(matches)[mask.flatten().astype(bool).tolist()]
        #     img_matches_filtered = cv2.drawMatches(
        #         img,
        #         keypoints,
        #         prev_img,
        #         prev_keypoints,
        #         good_matches,
        #         None,
        #         matchColor=match_color,
        #         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        #     )
        #     img_matches_filtered_marked = cv2.drawMatches(
        #         img_kp_epifilt,
        #         keypoints,
        #         img_kp_prev_epifilt,
        #         prev_keypoints,
        #         good_matches,
        #         None,
        #         matchColor=match_color,
        #         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        #     )
        #
        #     cv2.imshow("img", img)
        #     cv2.imshow("prev_img", prev_img)
        #     cv2.imshow("img kp raw", img_kp_raw)
        #     cv2.imshow("img prev kp raw", img_prev_kp_raw)
        #     cv2.imshow("matches", img_matches)
        #     cv2.imshow("matches_filtered", img_matches_filtered)
        #     cv2.imshow("matches_filtered_marked", img_matches_filtered_marked)
        #     cv2.imwrite("img.png", img)
        #     cv2.imwrite("img_prev.png", prev_img)
        #     cv2.imwrite("img_kp_raw.png", img_kp_raw)
        #     cv2.imwrite("prev_img_kp_raw.png", img_prev_kp_raw)
        #     cv2.imwrite("matches.png", img_matches)
        #     cv2.imwrite("matches_filt.png", img_matches_filtered)
        #     cv2.imwrite("matches_filt_marked.png", img_matches_filtered_marked)

            # lines1 = cv2.computeCorrespondEpilines(pts_prev_xy.reshape(-1, 1, 2), 2, F)
            # lines2 = cv2.computeCorrespondEpilines(pts_xy.reshape(-1, 1, 2), 1, F)
            # lines1 = lines1.reshape(-1, 3)
            # lines2 = lines2.reshape(-1, 3)
            # img1 = img_epi.copy()
            # img2 = prev_img.copy()
            # img1, img2 = debug_draw_lines(img1, img2, lines1, pts_xy, pts_prev_xy)
            # img2, img1 = debug_draw_lines(img2, img1, lines2, pts_prev_xy, pts_xy)
            # cv2.imshow("epipolar lines, current img", img1)
            # cv2.imshow("epipolar lines, previous img", img2)

            # prior parametrization
            # depth_orig = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            # depth_orig[depth_orig == 0] = depth_orig.max()
            # depth_orig = cv2.resize(depth_orig, dsize=(out_width, out_height))
            # depth_tensor = torch.from_numpy(depth_orig)[None, None, ...]
            # depth_tensor = depth_tensor.squeeze()
            # print(f"shape: {depth_tensor.shape}")
            # depth_features = torch.tensor(row_col_depth).unsqueeze(0)
            # depth_parametrization = get_depth_prior_from_features(
            #     depth_features, height=out_height, width=out_width
            # )
            # depth_mosaic = gray_to_heatmap(depth_parametrization[:, 0, ...])
            # depth_probability = gray_to_heatmap(
            #     depth_parametrization[:, 1, ...], colormap="inferno"
            # )
            # depth_heatmap = gray_to_heatmap(depth_tensor)
            # save_image(depth_mosaic, "depth_mosaic.png")
            # save_image(depth_probability, "depth_probability.png")
            # save_image(depth_heatmap, "depth_heatmap.png")
            # cv2.imwrite("depth_features.png", depth_kp)
            #
            # rgb_features = torch.tensor(row_col_rgb).unsqueeze(0)
            # rgb_parametrization = get_depth_prior_from_features(
            #     rgb_features, height=out_height, width=out_width
            # )
            # rgb_mosaic = gray_to_heatmap(rgb_parametrization[:, 0, ...])
            # rgb_probability = gray_to_heatmap(
            #     rgb_parametrization[:, 1, ...], colormap="inferno"
            # )
            # save_image(rgb_mosaic, "rgb_mosaic.png")
            # save_image(rgb_probability, "rgb_probability.png")
            # cv2.imwrite("rgb_features.png", img_kp)

        # print("Waiting for key press ...")
        # cv2.waitKey()

    # prev_keypoints = keypoints
    # prev_descriptors = descriptors
    # prev_img = img.copy()

print("Done.")
