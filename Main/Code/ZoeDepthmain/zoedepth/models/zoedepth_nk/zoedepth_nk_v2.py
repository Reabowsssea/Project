# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat



import torch
import torch.nn as nn
import itertools
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.base_models.depth_anything import DepthAnythingCore
from zoedepth.models.layers.attractor import AttractorLayer, AttractorLayerUnnormed
from zoedepth.models.layers.dist_layers import ConditionalLogBinomial


from zoedepth.models.layers.localbins_layers import (Projector, LinearSplitter, SeedBinRegressor,
                                            SeedBinRegressorUnnormed)
from zoedepth.models.layers.sparse_layers import SparsePerceptionPool, merge_sparse_into_output, FeatureExtractor


from zoedepth.models.layers.patch_transformer import PatchTransformerEncoder
from zoedepth.models.model_io import load_state_from_resource
from zoedepth.models.update import get_label, update_sample

# torch.autograd.set_detect_anomaly(True)
class ZoeDepthNK(DepthModel):
    def __init__(self, core,  bin_conf, bin_centers_type="softplus", bin_embedding_dim=128,
                 n_attractors=[16, 8, 4, 1], attractor_alpha=300, attractor_gamma=2, attractor_kind='sum', attractor_type='exp',
                 min_temp=5, max_temp=50,
                 memory_efficient=False, train_midas=True,
                 is_midas_pretrained=True, midas_lr_factor=1, encoder_lr_factor=10, pos_enc_lr_factor=10, inverse_midas=False,  **kwargs):
        """ZoeDepthNK model. This is the version of ZoeDepth that has two metric heads and uses a learned router to route to experts.

        Args:
            core (models.base_models.midas.MidasCore): The base midas model that is used for extraction of "relative" features

            bin_conf (List[dict]): A list of dictionaries that contain the bin configuration for each metric head. Each dictionary should contain the following keys:
                                    "name" (str, typically same as the dataset name), "n_bins" (int), "min_depth" (float), "max_depth" (float)

                                   The length of this list determines the number of metric heads.
            bin_centers_type (str, optional): "normed" or "softplus". Activation type used for bin centers. For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded. Defaults to "normed".
            bin_embedding_dim (int, optional): bin embedding dimension. Defaults to 128.

            n_attractors (List[int], optional): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int, optional): Proportional attractor strength. Refer to models.layers.attractor for more details. Defaults to 300.
            attractor_gamma (int, optional): Exponential attractor strength. Refer to models.layers.attractor for more details. Defaults to 2.
            attractor_kind (str, optional): Attraction aggregation "sum" or "mean". Defaults to 'sum'.
            attractor_type (str, optional): Type of attractor to use; "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'exp'.

            min_temp (int, optional): Lower bound for temperature of output probability distribution. Defaults to 5.
            max_temp (int, optional): Upper bound for temperature of output probability distribution. Defaults to 50.

            memory_efficient (bool, optional): Whether to use memory efficient version of attractor layers. Memory efficient version is slower but is recommended incase of multiple metric heads in order save GPU memory. Defaults to False.

            train_midas (bool, optional): Whether to train "core", the base midas model. Defaults to True.
            is_midas_pretrained (bool, optional): Is "core" pretrained? Defaults to True.
            midas_lr_factor (int, optional): Learning rate reduction factor for base midas model except its encoder and positional encodings. Defaults to 10.
            encoder_lr_factor (int, optional): Learning rate reduction factor for the encoder in midas model. Defaults to 10.
            pos_enc_lr_factor (int, optional): Learning rate reduction factor for positional encodings in the base midas model. Defaults to 10.

        """

        super().__init__()

        self.core = core
        self.bin_conf = bin_conf
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.memory_efficient = memory_efficient
        self.train_midas = train_midas
        self.is_midas_pretrained = is_midas_pretrained
        self.midas_lr_factor = midas_lr_factor
        self.encoder_lr_factor = encoder_lr_factor
        self.pos_enc_lr_factor = pos_enc_lr_factor
        self.inverse_midas = inverse_midas

        N_MIDAS_OUT = 32
        btlnck_features = self.core.output_channels[0]
        num_out_features = self.core.output_channels[1:]
        # self.scales = [16, 8, 4, 2]  # spatial scale factors

        self.conv2 = nn.Conv2d(
            btlnck_features, btlnck_features, kernel_size=1, stride=1, padding=0)

        # self.conv_224 = nn.Conv2d(
        #     1, N_MIDAS_OUT, kernel_size=1, stride=1, padding=0)
        # Transformer classifier on the bottleneck
        self.patch_transformer = PatchTransformerEncoder(
            btlnck_features, 1, 128, use_class_token=True)
        self.mlp_classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        if bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif bin_centers_type == "softplus":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError(
                "bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'")
        self.bin_centers_type = bin_centers_type
        # We have bins for each bin conf.
        # Create a map (ModuleDict) of 'name' -> seed_bin_regressor
        # self.seed_bin_regressors = nn.ModuleDict(
        #     {conf['name']: SeedBinRegressorLayer(128, conf["n_bins"], mlp_dim=bin_embedding_dim//2, min_depth=conf["min_depth"], max_depth=conf["max_depth"])
        #      for conf in bin_conf}
        # )

        # self.seed_projector = Projector(
        #     128, bin_embedding_dim, mlp_dim=bin_embedding_dim//2)
        # self.projectors = nn.ModuleList([
        #     Projector(128, bin_embedding_dim, mlp_dim=bin_embedding_dim//2)
        #     for num_out in num_out_features
        # ])



        # Create a map (ModuleDict) of 'name' -> seed_bin_regressor
        self.seed_bin_regressors = nn.ModuleDict(
            {conf['name']: SeedBinRegressorLayer(64, conf["n_bins"], mlp_dim=bin_embedding_dim//2, min_depth=conf["min_depth"], max_depth=conf["max_depth"])
             for conf in bin_conf}
        )

        self.seed_projector = Projector(
            64, bin_embedding_dim, mlp_dim=bin_embedding_dim//2)
        self.projectors = nn.ModuleList([
            Projector(64, bin_embedding_dim, mlp_dim=bin_embedding_dim//2)
            for num_out in num_out_features
        ])
        # self.seed_projector = ProjectionInputDepth(
        #     btlnck_features+, bin_embedding_dim, mlp_dim=bin_embedding_dim//2)
        # self.projectors = nn.ModuleList([
        #     Projector(num_out, bin_embedding_dim, mlp_dim=bin_embedding_dim//2)
        #     for num_out in num_out_features
        # ])

        # Create a map (ModuleDict) of 'name' -> attractors (ModuleList)
        self.attractors = nn.ModuleDict(
            {conf['name']: nn.ModuleList([
                Attractor(bin_embedding_dim, n_attractors[i],
                          mlp_dim=bin_embedding_dim, alpha=attractor_alpha,
                          gamma=attractor_gamma, kind=attractor_kind,
                          attractor_type=attractor_type, memory_efficient=memory_efficient,
                          min_depth=conf["min_depth"], max_depth=conf["max_depth"])
                for i in range(len(n_attractors))
            ])
                for conf in bin_conf}
        )

        last_in = N_MIDAS_OUT
        # conditional log binomial for each bin conf
        self.conditional_log_binomial = nn.ModuleDict(
            {conf['name']: ConditionalLogBinomial(32, bin_embedding_dim, conf['n_bins'], bottleneck_factor=4, min_temp=self.min_temp, max_temp=self.max_temp)
             for conf in bin_conf}
        )

        self.conditional_log_binomial_0 = nn.ModuleDict(
        {conf['name']: ConditionalLogBinomial(128, bin_embedding_dim, conf['n_bins'], bottleneck_factor=4, min_temp=self.min_temp, max_temp=self.max_temp)
             for conf in bin_conf}
        )
        # self.pool = SparsePerceptionPool(in_channels=1, out_channels=1)
        self.feature_extractor = FeatureExtractor()
        # self.conv_out = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.conv_s = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=1)


########################根本不融合###################################

    # def forward(self, input, sds, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
    #     """
    #     Args:
    #         input (torch.Tensor): (B, C, H, W)
    #         sds (torch.Tensor): (B, 1, H, W)
    #     """
    #     x_b, x_c, x_h, x_w = input.shape
    #     self.orig_input_width = x_w
    #     self.orig_input_height = x_h
    #
    #     # 添加调试信息
    #     print(f"Input shape: {input.shape}")
    #     print(f"SDS shape: {sds.shape if sds is not None else 'None'}")
    #
    #     rel_depth, out = self.core(input, denorm=denorm, return_rel_depth=True)
    #
    #     outconv_activation = out[0]
    #     btlnck = out[1]
    #     x_blocks = out[2:]
    #
    #     x_d0 = self.conv2(btlnck)
    #     x = x_d0
    #
    #     embedding = self.patch_transformer(x)[0]
    #     domain_logits = self.mlp_classifier(embedding)
    #     domain_vote = torch.softmax(domain_logits.sum(dim=0, keepdim=True), dim=-1)
    #
    #     bin_conf_name = "nyu"
    #     try:
    #         conf = [c for c in self.bin_conf if c.name == bin_conf_name][0]
    #     except IndexError:
    #         raise ValueError(f"bin_conf_name {bin_conf_name} not found in bin_confs")
    #
    #     min_depth = 1
    #     max_depth = 10.0  # 使用固定默认值
    #
    #     seed_bin_regressor = self.seed_bin_regressors[bin_conf_name]
    #     _, seed_b_centers = seed_bin_regressor(x)
    #     if torch.isnan(seed_b_centers).any().item():
    #         print('nan', seed_b_centers)
    #
    #     b_prev = seed_b_centers
    #     prev_b_embedding = self.seed_projector(x)
    #
    #     attractors = self.attractors[bin_conf_name]
    #     for projector, attractor, x in zip(self.projectors, attractors, x_blocks):
    #         b_embedding = projector(x)
    #         b, b_centers = attractor(
    #             b_embedding, b_prev, prev_b_embedding, interpolate=True
    #         )
    #         b_prev = b
    #         prev_b_embedding = b_embedding
    #
    #     last = outconv_activation
    #
    #     b_centers = nn.functional.interpolate(
    #         b_centers, last.shape[-2:], mode="bilinear", align_corners=True
    #     )
    #     b_embedding = nn.functional.interpolate(
    #         b_embedding, last.shape[-2:], mode="bilinear", align_corners=True
    #     )
    #
    #     clb = self.conditional_log_binomial[bin_conf_name]
    #     x = clb(last, b_embedding)
    #
    #     out = torch.sum(x * b_centers, dim=1, keepdim=True)
    #
    #     # ----------------------- 移除稀疏深度处理部分 --------------------
    #     # 直接使用网络原始输出，不进行任何SDS相关的校正
    #     metric_depth = out
    #
    #     # 可选：添加一个简单的注释说明SDS未被使用
    #     if sds is not None and torch.count_nonzero(sds) > 0:
    #         print("Note: SDS data is available but not used in depth estimation")
    #     # -------------------------------------------------------------
    #
    #     output = dict(
    #         domain_logits=domain_logits,
    #         metric_depth=metric_depth,  # 使用原始输出
    #     )
    #
    #     if return_final_centers or return_probs:
    #         output["bin_centers"] = b_centers
    #     if return_probs:
    #         output["probs"] = x
    #
    #     return output

# ########################直接全局缩放##################################
#     def forward(self, input, sds, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
#         """
#         Args:
#             input (torch.Tensor): (B, C, H, W)
#             sds (torch.Tensor): (B, 1, H, W)
#         """
#         x_b, x_c, x_h, x_w = input.shape
#         self.orig_input_width = x_w
#         self.orig_input_height = x_h
#
#         # 添加调试信息
#         print(f"Input shape: {input.shape}")
#         print(f"SDS shape: {sds.shape if sds is not None else 'None'}")
#         if sds is not None:
#             print(f"SDS non-zero count: {torch.count_nonzero(sds)}")
#             print(f"SDS unique values: {torch.unique(sds)}")
#
#         rel_depth, out = self.core(input, denorm=denorm, return_rel_depth=True)
#
#         outconv_activation = out[0]
#         btlnck = out[1]
#         x_blocks = out[2:]
#
#         x_d0 = self.conv2(btlnck)
#         x = x_d0
#
#         embedding = self.patch_transformer(x)[0]
#         domain_logits = self.mlp_classifier(embedding)
#         domain_vote = torch.softmax(domain_logits.sum(dim=0, keepdim=True), dim=-1)
#
#         bin_conf_name = "nyu"
#         try:
#             conf = [c for c in self.bin_conf if c.name == bin_conf_name][0]
#         except IndexError:
#             raise ValueError(f"bin_conf_name {bin_conf_name} not found in bin_confs")
#
#         min_depth = 1
#         max_depth = sds.max() if sds is not None else 10.0  # 添加默认值
#
#         seed_bin_regressor = self.seed_bin_regressors[bin_conf_name]
#         _, seed_b_centers = seed_bin_regressor(x)
#         if torch.isnan(seed_b_centers).any().item():
#             print('nan', seed_b_centers)
#
#         b_prev = seed_b_centers
#         prev_b_embedding = self.seed_projector(x)
#
#         attractors = self.attractors[bin_conf_name]
#         for projector, attractor, x in zip(self.projectors, attractors, x_blocks):
#             b_embedding = projector(x)
#             b, b_centers = attractor(
#                 b_embedding, b_prev, prev_b_embedding, interpolate=True
#             )
#             b_prev = b
#             prev_b_embedding = b_embedding
#
#         last = outconv_activation
#
#         b_centers = nn.functional.interpolate(
#             b_centers, last.shape[-2:], mode="bilinear", align_corners=True
#         )
#         b_embedding = nn.functional.interpolate(
#             b_embedding, last.shape[-2:], mode="bilinear", align_corners=True
#         )
#
#         clb = self.conditional_log_binomial[bin_conf_name]
#         x = clb(last, b_embedding)
#
#         out = torch.sum(x * b_centers, dim=1, keepdim=True)
#
#         # ----------------------- 修改后的稀疏深度处理部分 --------------------
#         corrected_depth = out.clone()
#
#         # 检查 SDS 数据是否有效
#         if sds is not None and sds.numel() > 0 and torch.count_nonzero(sds) > 0:
#             try:
#                 # 方法1：直接获取第一个非零值（更安全）
#                 sds_value = sds[sds != 0][0].item()
#                 print(f"Using SDS value: {sds_value}")
#
#                 # 方法2：备选方案，使用索引访问（添加保护）
#                 nonzero_indices = torch.nonzero(sds, as_tuple=True)
#                 if len(nonzero_indices) >= 4 and all(len(idx) > 0 for idx in nonzero_indices):
#                     try:
#                         sds_value_alt = sds[nonzero_indices[0][0], nonzero_indices[1][0],
#                         nonzero_indices[2][0], nonzero_indices[3][0]].item()
#                         print(f"Alternative SDS value: {sds_value_alt}")
#                     except IndexError:
#                         print("Warning: Alternative SDS indexing failed, using primary method")
#
#                 # 对每个批次处理
#                 for b in range(corrected_depth.shape[0]):
#                     # 获取当前批次的深度图
#                     batch_depth = corrected_depth[b, 0]
#
#                     # 找到整个图像中的最小深度值（最近距离）
#                     min_depth_in_image = batch_depth.min()
#                     print(f"Batch {b}: min_depth_in_image={min_depth_in_image.item()}")
#
#                     # 计算缩放因子：SDS值 / 图像最小深度值
#                     scale_factor = sds_value / min_depth_in_image
#                     print(f"Batch {b}: scale_factor={scale_factor}")
#
#                     # 应用全局缩放
#                     corrected_depth[b] *= scale_factor
#
#             except Exception as e:
#                 print(f"Error in SDS processing: {e}")
#                 print("Skipping SDS scaling due to error")
#         else:
#             print("Warning: SDS is None, empty, or has no non-zero values, skipping scaling")
#
#         # -------------------------------------------------------------
#
#         output = dict(
#             domain_logits=domain_logits,
#             metric_depth=corrected_depth,
#         )
#
#         if return_final_centers or return_probs:
#             output["bin_centers"] = b_centers
#         if return_probs:
#             output["probs"] = x
#
#         return output
##############################我们的融合方法##########################################
    def forward(self, input, sds, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
        """
        Args:
            input (torch.Tensor): (B, C, H, W)
            sds (torch.Tensor): (B, 1, H, W)
        """
        x_b, x_c, x_h, x_w = input.shape
        self.orig_input_width = x_w
        self.orig_input_height = x_h
        rel_depth, out = self.core(input, denorm=denorm, return_rel_depth=True)

        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]

        x_d0 = self.conv2(btlnck)
        x = x_d0

        embedding = self.patch_transformer(x)[0]
        domain_logits = self.mlp_classifier(embedding)
        domain_vote = torch.softmax(domain_logits.sum(dim=0, keepdim=True), dim=-1)

        bin_conf_name = "nyu"
        try:
            conf = [c for c in self.bin_conf if c.name == bin_conf_name][0]
        except IndexError:
            raise ValueError(f"bin_conf_name {bin_conf_name} not found in bin_confs")

        min_depth = 1
        max_depth = sds.max()

        seed_bin_regressor = self.seed_bin_regressors[bin_conf_name]
        _, seed_b_centers = seed_bin_regressor(x)
        if torch.isnan(seed_b_centers).any().item():
            print('nan', seed_b_centers)

        b_prev = seed_b_centers
        prev_b_embedding = self.seed_projector(x)

        attractors = self.attractors[bin_conf_name]
        for projector, attractor, x in zip(self.projectors, attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True
            )
            b_prev = b
            prev_b_embedding = b_embedding

        last = outconv_activation

        b_centers = nn.functional.interpolate(
            b_centers, last.shape[-2:], mode="bilinear", align_corners=True
        )
        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode="bilinear", align_corners=True
        )

        clb = self.conditional_log_binomial[bin_conf_name]
        x = clb(last, b_embedding)

        out = torch.sum(x * b_centers, dim=1, keepdim=True)

        # ----------------------- 第一阶段：局部高斯平滑 --------------------
        def gaussian_kernel(size: int, sigma: float):
            coords = torch.arange(size) - size // 2
            x_grid, y_grid = torch.meshgrid(coords, coords, indexing="ij")
            kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
            return kernel / kernel.sum()

        kernel_size = 11
        sigma = 1.0
        gaussian_weight = gaussian_kernel(kernel_size, sigma).to(out.device)

        corrected_depth = out.clone()

        # 找出非零稀疏点
        nonzero_indices = sds.nonzero(as_tuple=True)

        if nonzero_indices[0].numel() > 0:
            sds_values = sds[nonzero_indices]

            for idx in range(nonzero_indices[0].numel()):
                try:
                    b = nonzero_indices[0][idx]
                    c = nonzero_indices[1][idx]
                    h = nonzero_indices[2][idx]
                    w = nonzero_indices[3][idx]
                except IndexError:
                    continue  # 如果坐标不匹配，就跳过

                sds_depth_value = sds_values[idx]

                # 窗口
                h_start = max(h - kernel_size // 2, 0)
                h_end = min(h + kernel_size // 2 + 1, corrected_depth.shape[-2])
                w_start = max(w - kernel_size // 2, 0)
                w_end = min(w + kernel_size // 2 + 1, corrected_depth.shape[-1])

                local_patch = corrected_depth[b, :, h_start:h_end, w_start:w_end]

                weight_patch = gaussian_weight[
                               : local_patch.shape[-2], : local_patch.shape[-1]
                               ]
                corrected_depth[b, :, h_start:h_end, w_start:w_end] = (
                        weight_patch * sds_depth_value
                        + (1 - weight_patch) * local_patch
                )

        print(f"第一阶段完成：局部高斯平滑，处理了 {nonzero_indices[0].numel()} 个SDS点")

        # ----------------------- 第二阶段：全局尺度对齐 --------------------
        # 检查 SDS 数据是否有效
        if sds is not None and sds.numel() > 0 and torch.count_nonzero(sds) > 0:
            try:
                # 获取第一个非零SDS值作为参考
                sds_value = sds[sds != 0][0].item()
                print(f"使用SDS参考值进行全局缩放: {sds_value}")

                # 定义圆心和半径（用于尺度对齐的参考区域）
                center_h, center_w = corrected_depth.shape[2] // 2, corrected_depth.shape[3] // 2
                radius = min(corrected_depth.shape[2], corrected_depth.shape[3]) // 4

                # 创建网格坐标
                h_coords = torch.arange(corrected_depth.shape[2], device=corrected_depth.device)
                w_coords = torch.arange(corrected_depth.shape[3], device=corrected_depth.device)
                h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')

                # 计算每个点到圆心的距离
                distances = torch.sqrt((h_grid - center_h) ** 2 + (w_grid - center_w) ** 2)

                # 创建圆形掩码
                circle_mask = distances <= radius

                # 对每个批次处理
                for b in range(corrected_depth.shape[0]):
                    # 获取当前批次的深度图
                    batch_depth = corrected_depth[b, 0]

                    # 在圆形区域内找到深度最小的点
                    circle_depths = batch_depth[circle_mask]
                    if circle_depths.numel() > 0:
                        min_depth_in_circle = circle_depths.min()

                        # 计算缩放因子
                        scale_factor = sds_value / min_depth_in_circle
                        print(
                            f"批次 {b}: 圆形区域最小深度={min_depth_in_circle.item():.3f}, 缩放因子={scale_factor:.3f}")

                        # 应用全局缩放
                        corrected_depth[b] *= scale_factor
                    else:
                        print(f"警告: 批次 {b} 的圆形区域内未找到深度值")

                print("第二阶段完成：全局尺度对齐")

            except Exception as e:
                print(f"全局缩放错误: {e}")
                print("跳过全局缩放步骤")
        else:
            print("警告: SDS数据无效，跳过全局缩放")

        # -------------------------------------------------------------

        output = dict(
            domain_logits=domain_logits,
            metric_depth=corrected_depth,
        )

        if return_final_centers or return_probs:
            output["bin_centers"] = b_centers
        if return_probs:
            output["probs"] = x

        return output
####################################新的融合的方法（多尺度，毕业论文用）#######################################################
    # def forward(self, input, sds, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
    #     """
    #     Args:
    #         input (torch.Tensor): (B, C, H, W)
    #         sds (torch.Tensor): (B, 1, H, W)
    #     """
    #     x_b, x_c, x_h, x_w = input.shape
    #     self.orig_input_width = x_w
    #     self.orig_input_height = x_h
    #
    #     rel_depth, out = self.core(input, denorm=denorm, return_rel_depth=True)
    #
    #     outconv_activation = out[0]
    #     btlnck = out[1]
    #     x_blocks = out[2:]
    #
    #     x_d0 = self.conv2(btlnck)
    #     x = x_d0
    #
    #     embedding = self.patch_transformer(x)[0]
    #     domain_logits = self.mlp_classifier(embedding)
    #     domain_vote = torch.softmax(domain_logits.sum(dim=0, keepdim=True), dim=-1)
    #
    #     bin_conf_name = "nyu"
    #     try:
    #         conf = [c for c in self.bin_conf if c.name == bin_conf_name][0]
    #     except IndexError:
    #         raise ValueError(f"bin_conf_name {bin_conf_name} not found in bin_confs")
    #
    #     min_depth = 1
    #     max_depth = sds.max() if sds is not None else 10.0
    #
    #     seed_bin_regressor = self.seed_bin_regressors[bin_conf_name]
    #     _, seed_b_centers = seed_bin_regressor(x)
    #     if torch.isnan(seed_b_centers).any().item():
    #         print('nan', seed_b_centers)
    #
    #     b_prev = seed_b_centers
    #     prev_b_embedding = self.seed_projector(x)
    #
    #     attractors = self.attractors[bin_conf_name]
    #     for projector, attractor, x in zip(self.projectors, attractors, x_blocks):
    #         b_embedding = projector(x)
    #         b, b_centers = attractor(
    #             b_embedding, b_prev, prev_b_embedding, interpolate=True
    #         )
    #         b_prev = b
    #         prev_b_embedding = b_embedding
    #
    #     last = outconv_activation
    #
    #     b_centers = nn.functional.interpolate(
    #         b_centers, last.shape[-2:], mode="bilinear", align_corners=True
    #     )
    #     b_embedding = nn.functional.interpolate(
    #         b_embedding, last.shape[-2:], mode="bilinear", align_corners=True
    #     )
    #
    #     clb = self.conditional_log_binomial[bin_conf_name]
    #     x = clb(last, b_embedding)
    #
    #     out = torch.sum(x * b_centers, dim=1, keepdim=True)
    #
    #     # ----------------------- 多尺度融合策略 --------------------
    #     corrected_depth = out.clone()
    #
    #     # 检查 SDS 数据是否有效
    #     if sds is not None and sds.numel() > 0 and torch.count_nonzero(sds) > 0:
    #         try:
    #             # 获取SDS值
    #             sds_value = sds[sds != 0][0].item()
    #             print(f"Using SDS value: {sds_value}")
    #
    #             # 定义多个尺度的圆形区域
    #             centers = [(87, 178)]  # 主圆心
    #             radii = [30, 45, 60]  # 多个半径
    #
    #             # 对每个批次处理
    #             for b in range(corrected_depth.shape[0]):
    #                 # 获取当前批次的深度图
    #                 batch_depth = corrected_depth[b, 0]
    #
    #                 # 计算不同尺度下的缩放因子
    #                 scale_factors = []
    #                 valid_scales = []
    #
    #                 for center in centers:
    #                     center_h, center_w = center
    #
    #                     for radius in radii:
    #                         # 创建网格坐标
    #                         h_coords = torch.arange(batch_depth.shape[0], device=batch_depth.device)
    #                         w_coords = torch.arange(batch_depth.shape[1], device=batch_depth.device)
    #                         h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')
    #
    #                         # 计算每个点到圆心的距离
    #                         distances = torch.sqrt((h_grid - center_h) ** 2 + (w_grid - center_w) ** 2)
    #
    #                         # 创建圆形掩码
    #                         circle_mask = distances <= radius
    #
    #                         # 在圆形区域内找到深度最小的点
    #                         circle_depths = batch_depth[circle_mask]
    #                         if circle_depths.numel() > 0:
    #                             min_depth_in_circle = circle_depths.min()
    #
    #                             # 计算缩放因子
    #                             scale_factor = sds_value / min_depth_in_circle
    #                             scale_factors.append(scale_factor)
    #                             valid_scales.append((radius, scale_factor))
    #
    #                 if scale_factors:
    #                     # 使用加权平均的缩放因子，半径越大权重越小
    #                     weights = [1.0 / r for r, _ in valid_scales]  # 半径越小权重越大
    #                     total_weight = sum(weights)
    #                     weighted_scale = sum(w * sf for w, (_, sf) in zip(weights, valid_scales)) / total_weight
    #
    #                     print(f"Batch {b}: weighted_scale_factor={weighted_scale}")
    #
    #                     # 应用加权缩放
    #                     corrected_depth[b] *= weighted_scale
    #                 else:
    #                     print(f"Warning: No valid scales found for batch {b}")
    #
    #         except Exception as e:
    #             print(f"Error in SDS processing: {e}")
    #             print("Skipping SDS fusion due to error")
    #     else:
    #         print("Warning: SDS is None, empty, or has no non-zero values, skipping fusion")
    #
    #     # -------------------------------------------------------------
    #
    #     output = dict(
    #         domain_logits=domain_logits,
    #         metric_depth=corrected_depth,
    #     )
    #
    #     if return_final_centers or return_probs:
    #         output["bin_centers"] = b_centers
    #     if return_probs:
    #         output["probs"] = x
    #
    #     return output

##################加权#############################################
    def forward(self, input, sds, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
        """
        Args:
            input (torch.Tensor): (B, C, H, W)
            sds (torch.Tensor): (B, 1, H, W)
        """
        x_b, x_c, x_h, x_w = input.shape
        self.orig_input_width = x_w
        self.orig_input_height = x_h

        # 添加调试信息
        print(f"Input shape: {input.shape}")
        print(f"SDS shape: {sds.shape if sds is not None else 'None'}")
        if sds is not None:
            print(f"SDS non-zero count: {torch.count_nonzero(sds)}")
            print(f"SDS unique values: {torch.unique(sds)}")

        rel_depth, out = self.core(input, denorm=denorm, return_rel_depth=True)

        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]

        x_d0 = self.conv2(btlnck)
        x = x_d0

        embedding = self.patch_transformer(x)[0]
        domain_logits = self.mlp_classifier(embedding)
        domain_vote = torch.softmax(domain_logits.sum(dim=0, keepdim=True), dim=-1)

        bin_conf_name = "nyu"
        try:
            conf = [c for c in self.bin_conf if c.name == bin_conf_name][0]
        except IndexError:
            raise ValueError(f"bin_conf_name {bin_conf_name} not found in bin_confs")

        min_depth = 1
        max_depth = sds.max() if sds is not None else 10.0  # 添加默认值

        seed_bin_regressor = self.seed_bin_regressors[bin_conf_name]
        _, seed_b_centers = seed_bin_regressor(x)
        if torch.isnan(seed_b_centers).any().item():
            print('nan', seed_b_centers)

        b_prev = seed_b_centers
        prev_b_embedding = self.seed_projector(x)

        attractors = self.attractors[bin_conf_name]
        for projector, attractor, x in zip(self.projectors, attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True
            )
            b_prev = b
            prev_b_embedding = b_embedding

        last = outconv_activation

        b_centers = nn.functional.interpolate(
            b_centers, last.shape[-2:], mode="bilinear", align_corners=True
        )
        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode="bilinear", align_corners=True
        )

        clb = self.conditional_log_binomial[bin_conf_name]
        x = clb(last, b_embedding)

        out = torch.sum(x * b_centers, dim=1, keepdim=True)

        # ----------------------- 稀疏深度处理部分 --------------------
        corrected_depth = out.clone()

        # 检查 SDS 数据是否有效
        if sds is not None and sds.numel() > 0 and torch.count_nonzero(sds) > 0:
            try:
                # 方法1：直接获取第一个非零值（更安全）
                sds_value = sds[sds != 0][0].item()
                print(f"Using SDS value: {sds_value}")

                # 方法2：备选方案，使用索引访问（添加保护）
                nonzero_indices = torch.nonzero(sds, as_tuple=True)
                if len(nonzero_indices) >= 4 and all(len(idx) > 0 for idx in nonzero_indices):
                    try:
                        sds_value_alt = sds[nonzero_indices[0][0], nonzero_indices[1][0],
                        nonzero_indices[2][0], nonzero_indices[3][0]].item()
                        print(f"Alternative SDS value: {sds_value_alt}")
                    except IndexError:
                        print("Warning: Alternative SDS indexing failed, using primary method")

                # 定义圆心和半径
                center_h, center_w = 87, 178
                radius = 60

                # 创建网格坐标
                h_coords = torch.arange(corrected_depth.shape[2], device=corrected_depth.device)
                w_coords = torch.arange(corrected_depth.shape[3], device=corrected_depth.device)
                h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')

                # 计算每个点到圆心的距离
                distances = torch.sqrt((h_grid - center_h) ** 2 + (w_grid - center_w) ** 2)

                # 创建圆形掩码
                circle_mask = distances <= radius

                # 对每个批次处理
                for b in range(corrected_depth.shape[0]):
                    # 获取当前批次的深度图
                    batch_depth = corrected_depth[b, 0]

                    # 在圆形区域内找到深度值
                    circle_depths = batch_depth[circle_mask]
                    if circle_depths.numel() > 0:
                        # 计算圆形区域内的统计特征
                        min_depth_in_circle = circle_depths.min()
                        max_depth_in_circle = circle_depths.max()
                        mean_depth_in_circle = circle_depths.mean()
                        median_depth_in_circle = circle_depths.median()

                        print(f"Batch {b}: Circle depth stats - min={min_depth_in_circle.item():.3f}, "
                              f"max={max_depth_in_circle.item():.3f}, mean={mean_depth_in_circle.item():.3f}, "
                              f"median={median_depth_in_circle.item():.3f}")

                        # 策略1：使用加权平均深度（更稳定）
                        # 给较小的深度值更高的权重，因为我们知道声呐测量的是最小深度
                        weights = 1.0 / (circle_depths + 1e-6)  # 避免除零
                        weights = weights / weights.sum()  # 归一化
                        weighted_depth = (circle_depths * weights).sum()

                        # 策略2：使用分位数深度（例如25%分位数）
                        sorted_depths, _ = torch.sort(circle_depths)
                        quantile_idx = int(0.25 * len(sorted_depths))  # 25%分位数
                        quantile_depth = sorted_depths[quantile_idx]

                        # 策略3：使用trimmed mean（去除极端值后的平均值）
                        trim_ratio = 0.1  # 去除10%的极端值
                        trim_count = int(trim_ratio * len(sorted_depths))
                        trimmed_depths = sorted_depths[trim_count:-trim_count] if trim_count > 0 else sorted_depths
                        trimmed_mean = trimmed_depths.mean()

                        print(f"Batch {b}: Weighted depth={weighted_depth.item():.3f}, "
                              f"25% quantile={quantile_depth.item():.3f}, "
                              f"trimmed mean={trimmed_mean.item():.3f}")

                        # 选择最合适的参考深度（这里使用trimmed mean作为例子）
                        reference_depth = trimmed_mean

                        # 计算缩放因子：声呐深度 / 参考深度
                        scale_factor = sds_value / reference_depth
                        print(f"Batch {b}: Using reference_depth={reference_depth.item():.3f}, "
                              f"scale_factor={scale_factor:.3f}")

                        # 应用全局缩放
                        corrected_depth[b] *= scale_factor
                    else:
                        print(f"Warning: No depths found in circle for batch {b}")

            except Exception as e:
                print(f"Error in SDS processing: {e}")
                print("Skipping SDS scaling due to error")
        else:
            print("Warning: SDS is None, empty, or has no non-zero values, skipping scaling")

        # -------------------------------------------------------------

        output = dict(
            domain_logits=domain_logits,
            metric_depth=corrected_depth,
        )

        if return_final_centers or return_probs:
            output["bin_centers"] = b_centers
        if return_probs:
            output["probs"] = x

        return output
    def get_lr_params(self, lr):
        """
        Learning rate configuration for different layers of the model

        Args:
            lr (float) : Base learning rate
        Returns:
            list : list of parameters to optimize and their learning rates, in the format required by torch optimizers.
        """
        param_conf = []
        if self.train_midas:
            def get_rel_pos_params():
                for name, p in self.core.core.pretrained.named_parameters():
                    if "relative_position" in name:
                        yield p

            def get_enc_params_except_rel_pos():
                for name, p in self.core.core.pretrained.named_parameters():
                    if "relative_position" not in name:
                        yield p

            encoder_params = get_enc_params_except_rel_pos()
            rel_pos_params = get_rel_pos_params()
            midas_params = self.core.core.depth_head.scratch.parameters()
            midas_lr_factor = self.midas_lr_factor if self.is_midas_pretrained else 1.0
            param_conf.extend([
                {'params': encoder_params, 'lr': lr / self.encoder_lr_factor},
                {'params': rel_pos_params, 'lr': lr / self.pos_enc_lr_factor},
                {'params': midas_params, 'lr': lr / midas_lr_factor}
            ])

        remaining_modules = []
        for name, child in self.named_children():
            if name != 'core':
                remaining_modules.append(child)
        remaining_params = itertools.chain(
            *[child.parameters() for child in remaining_modules])
        param_conf.append({'params': remaining_params, 'lr': lr})
        return param_conf

    def get_conf_parameters(self, conf_name):
        """
        Returns parameters of all the ModuleDicts children that are exclusively used for the given bin configuration
        """
        params = []
        for name, child in self.named_children():
            if isinstance(child, nn.ModuleDict):
                for bin_conf_name, module in child.items():
                    if bin_conf_name == conf_name:
                        params += list(module.parameters())
        return params

    def freeze_conf(self, conf_name):
        """
        Freezes all the parameters of all the ModuleDicts children that are exclusively used for the given bin configuration
        """
        for p in self.get_conf_parameters(conf_name):
            p.requires_grad = False

    def unfreeze_conf(self, conf_name):
        """
        Unfreezes all the parameters of all the ModuleDicts children that are exclusively used for the given bin configuration
        """
        for p in self.get_conf_parameters(conf_name):
            p.requires_grad = True

    def freeze_all_confs(self):
        """
        Freezes all the parameters of all the ModuleDicts children
        """
        for name, child in self.named_children():
            if isinstance(child, nn.ModuleDict):
                for bin_conf_name, module in child.items():
                    for p in module.parameters():
                        p.requires_grad = False

    @staticmethod
    def build(midas_model_type="dinov2_small", pretrained_resource=None, use_pretrained_midas=False, train_midas=False, freeze_midas_bn=True, **kwargs):
        core = DepthAnythingCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
                               train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn, **kwargs)
        model = ZoeDepthNK(core, **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            print("pretrained_resource", pretrained_resource)
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        return ZoeDepthNK.build(**config)
