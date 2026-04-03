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
            {conf['name']: SeedBinRegressorLayer(128, conf["n_bins"], mlp_dim=bin_embedding_dim//2, min_depth=conf["min_depth"], max_depth=conf["max_depth"])
             for conf in bin_conf}
        )

        self.seed_projector = Projector(
            128, bin_embedding_dim, mlp_dim=bin_embedding_dim//2)
        self.projectors = nn.ModuleList([
            Projector(128, bin_embedding_dim, mlp_dim=bin_embedding_dim//2)
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
        
    def forward(self, input, sds, dds, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W). Assumes all images are from the same domain.
            return_final_centers (bool, optional): Whether to return the final centers of the attractors. Defaults to False.
            denorm (bool, optional): Whether to denormalize the input image. Defaults to False.
            return_probs (bool, optional): Whether to return the probabilities of the bins. Defaults to False.
        
        Returns:
            dict: Dictionary of outputs with keys:
                - "rel_depth": Relative depth map of shape (B, 1, H, W)
                - "metric_depth": Metric depth map of shape (B, 1, H, W)
                - "domain_logits": Domain logits of shape (B, 2)
                - "bin_centers": Bin centers of shape (B, N, H, W). Present only if return_final_centers is True
                - "probs": Bin probabilities of shape (B, N, H, W). Present only if return_probs is True
        """
        x_b, x_c, x_h, x_w = input.shape
        # x = x.view(x_b * x_f, x_c, x_h, x_w)
        # b, c, h, w = x.shape
        sd_outputs = []
        self.orig_input_width = x_w
        self.orig_input_height = x_h
        rel_depth, out = self.core(input, denorm=denorm, return_rel_depth=True)
        
        # context = feats[0]
        # gru_hidden = torch.tanh(self.project(e1))
        
        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]
        
        sds_5= torch.nn.functional.interpolate(sds, size=(518, 518), mode='bilinear')
        sds_4= torch.nn.functional.interpolate(sds, size=(296, 296), mode='bilinear')
        sds_3= torch.nn.functional.interpolate(sds, size=(148, 148), mode='bilinear')
        sds_2= torch.nn.functional.interpolate(sds, size=(74, 74), mode='bilinear')
        sds_1= torch.nn.functional.interpolate(sds, size=(37, 37), mode='bilinear')
        # sds_0= torch.nn.functional.interpolate(sds, size=(8, 8), mode='bilinear')
        # sds_224 = torch.nn.functional.interpolate(sds, size=(224, 224), mode='bilinear')
        sds_blocks = [sds_1, sds_2, sds_3, sds_4]
        
        dds_0= torch.nn.functional.interpolate(dds, size=(592, 592), mode='bilinear')
        # dds_224= torch.nn.functional.interpolate(dds, size=(224, 224), mode='bilinear')
        # conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # conv1 = conv1.to('cuda')
        # c_dds_224 = conv1(dds_224.float())
        # dds_1= torch.nn.functional.interpolate(dds, size=(128, 128), mode='bilinear')
        # dds_2= torch.nn.functional.interpolate(dds, size=(64, 64), mode='bilinear')
        # sp_5 = self.pool(dds_0)
        # sp_4 = self.pool(dds_1)
        # sp_3 = self.pool(dds_2)
        # print(f"张量 {sds_5} 的最大值、最小值为: {sds_5.max()}, {sds_5.min()}")
        # print(f"张量 {sds_4} 的最大值、最小值为: {sds_4.max()}, {sds_4.min()}")
        # print(f"张量 {sds_3} 的最大值、最小值为: {sds_3.max()}, {sds_3.min()}")
        _, x2, x3, x4, x5, x6 = self.feature_extractor(dds_0)
        # x1 = self.conv_224(dds_224.float())
        # x1, _, _, _, _, _ = self.feature_extractor(dds_224)
        x_d0 = self.conv2(btlnck)
        # 打印每个张量的最大值
        # for i, tensor in enumerate(sp):
        #     max_value = tensor.max().item()
        #     print(f"张量 {i+1} 的最大值为: {max_value}")
        x = x_d0

        # Predict which path to take
        embedding = self.patch_transformer(x)[0]  # N, E
        domain_logits = self.mlp_classifier(embedding)  # N, 2
        domain_vote = torch.softmax(domain_logits.sum(
            dim=0, keepdim=True), dim=-1)  # 1, 2

        # Get the path
        # bin_conf_name = ["nyu", "kitti", "sea_thru"][torch.argmax(
        #     domain_vote, dim=-1).squeeze().item()]
        
        bin_conf_name = ["nyu", "kitti", "sea_thru"][torch.argmax(
            domain_vote, dim=-1).squeeze().item()]

        try:
            conf = [c for c in self.bin_conf if c.name == bin_conf_name][0]
        except IndexError:
            raise ValueError(
                f"bin_conf_name {bin_conf_name} not found in bin_confs")

        min_depth = 1
        max_depth = sds.max()

        fuse_x0 = torch.cat([x, x6], dim=1)
        # fuse_x0 = x+x6
        # fuse_x0 = fuse_x0.to(torch.device('cuda'))
        fuse_x1 = torch.cat([out[2], x5], dim=1)
        # fuse_x1 = out[2]+x5
        # fuse_x1 = fuse_x1.to('cuda')
        fuse_x2 = torch.cat([out[3], x4], dim=1)
        # fuse_x2 = out[3]+x4
        # fuse_x2 = fuse_x2.to('cuda')
        fuse_x3 = torch.cat([out[4], x3], dim=1)
        # fuse_x3 = out[4]+x3
        # fuse_x3 = fuse_x3.to('cuda')
        fuse_x4 = torch.cat([out[5], x2], dim=1)
        # fuse_x4 = out[5]+x2
        # fuse_x4 = fuse_x4.to('cuda')
        fuse_blocks = [fuse_x1, fuse_x2, fuse_x3, fuse_x4]
        
        
        # seed_bin_regressor = SeedBinRegressor(128, 64, mlp_dim=64, min_depth=1e-5, max_depth=max_depth)
        # seed_bin_regressor = seed_bin_regressor.cuda()
        seed_bin_regressor = self.seed_bin_regressors[bin_conf_name]
        _, seed_b_centers = seed_bin_regressor(fuse_x0)
        # if torch.isnan(seed_b_centers).any().item():
        #     print('nan', seed_b_centers)
        # _, seed_b_centers = seed_bin_regressor(x)
        # if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
        #     b_prev = (seed_b_centers - min_depth)/(max_depth - min_depth)
        # else:
        #     b_prev = seed_b_centers
        # prev_b_embedding = self.seed_projector(x)
        # if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
        #     b_prev = (seed_b_centers - min_depth)/(max_depth - min_depth)
        # else:
        b_prev = seed_b_centers
        # self.seed_projector = self.seed_projector.cuda()
        prev_b_embedding = self.seed_projector(fuse_x0)
        # prev_b_embedding = nn.functional.interpolate(
                    # seed_b_embedding, [16,16], mode='bilinear', align_corners=True)
        attractors = self.attractors[bin_conf_name]
        clb_0= self.conditional_log_binomial_0[bin_conf_name]
        for projector, x, i in zip(self.projectors, fuse_blocks, sds_blocks):
            # b_embedding = projector(x)
            
            b_embedding = projector(x)
            # b, b_centers = attractor(
            #     b_embedding, b_prev, prev_b_embedding, interpolate=True)
            # b_prev = b
            # prev_b_embedding = b_embedding

            b_prev = nn.functional.interpolate(
                b_prev, x.shape[-2:], mode='bilinear', align_corners=True)
            new_embedding = nn.functional.interpolate(
                prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            prob = clb_0(x, new_embedding)
            self.expand_prob_layer = nn.Conv2d(4, 64, kernel_size=1).to('cuda')  # 将通道数从 4 扩展到 64
            prob = self.expand_prob_layer(prob)
            depth_r = torch.sum(prob.detach() * b_prev.detach(), dim=1, keepdim=True)

            depth_r = torch.sum(prob.detach() * b_prev.detach(), dim=1, keepdim=True)
            depth_r, nonzero_indices, sparse_values = merge_sparse_into_output(i, depth_r)
            c_brev = b_prev[nonzero_indices[:, 0], :, nonzero_indices[:, 2], nonzero_indices[:, 3]]
            prob[nonzero_indices[:, 0], :, nonzero_indices[:, 2], nonzero_indices[:, 3]] = 0

            if c_brev.size(0)>0:
                
                diff = torch.abs(c_brev - sparse_values.unsqueeze(-1))

            # 找到差值最小的索引
                min_diff, min_index = torch.min(diff, dim=1)
                # print(prob[nonzero_indices[:, 0], :, nonzero_indices[:, 2], nonzero_indices[:, 3]])
                for i in range(min_index.size(0)):
                    nonzero_indices[i, 1].fill_(min_index[i])
                # for i in range(nonzero_indices.size(0)):
                prob[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2], nonzero_indices[:, 3]] = 1
                # print(prob[nonzero_indices[:, 0], :, nonzero_indices[:, 2], nonzero_indices[:, 3]])
            residual = b_prev.detach() - depth_r.repeat(1, 64, 1, 1)
            uncertainty_map = torch.sqrt((prob.detach() * ((b_prev.detach() - depth_r.repeat(1, 64, 1, 1))**2)).sum(1, keepdim=True))

            adjustment_factor = 1.0 / (1.0 + uncertainty_map*10) 
            B_centers, _ = torch.sort(b_prev, dim=1)
            B_centers = torch.clip(b_prev, min_depth, max_depth)
            B_centers = B_centers + adjustment_factor * residual
            B_centers = torch.clamp(B_centers, min=0.1) 

            b_prev = B_centers
            prev_b_embedding = b_embedding
        last = outconv_activation 
        # if self.inverse_midas:
        #     # invert depth followed by normalization
        #     rel_depth = 1.0 / (rel_depth + 1e-6)
        #     rel_depth = (rel_depth - rel_depth.min()) / \
        #         (rel_depth.max() - rel_depth.min())
        # # concat rel depth with last. First interpolate rel depth to last size
        # rel_cond = rel_depth.unsqueeze(1)
        # rel_cond = nn.functional.interpolate(
        #     rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        # last = torch.cat([last, rel_cond], dim=1)
        
        b_centers = nn.functional.interpolate(
            B_centers, last.shape[-2:], mode='bilinear', align_corners=True)
        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        # new_last = torch.cat([last,c_dds_224], dim=1)
        clb = self.conditional_log_binomial[bin_conf_name]
        x = clb(last, b_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        # print(x.shape, b_centers.shape)
        # b_centers = nn.functional.interpolate(b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        # print("b_centers",b_centers.shape)
        # print("x",x.shape)
        self.expand_layer = nn.Conv2d(4, 64, kernel_size=1).to('cuda')  # 将通道数从 2 扩展到 64
        x = self.expand_layer(x)

        out_depth = torch.sum(x * b_centers, dim=1, keepdim=True)
        out_depth, _, _ = merge_sparse_into_output(sds_5, out_depth)

        output = dict(domain_logits=domain_logits, metric_depth=out_depth)
        if return_final_centers or return_probs:
            output['bin_centers'] = b_centers

        if return_probs:
            output['probs'] = x
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
