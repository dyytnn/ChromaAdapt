from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

try:
    from torch.amp import custom_bwd, custom_fwd
except ImportError:  # PyTorch < 2.4 exposes these decorators under cuda.amp.
    from torch.cuda.amp import custom_bwd as _cuda_custom_bwd
    from torch.cuda.amp import custom_fwd as _cuda_custom_fwd

    def custom_fwd(*args, **kwargs):
        kwargs.pop("device_type", None)
        if args:
            return _cuda_custom_fwd(*args, **kwargs)
        return lambda function: _cuda_custom_fwd(function, **kwargs)

    def custom_bwd(*args, **kwargs):
        kwargs.pop("device_type", None)
        if args:
            return _cuda_custom_bwd(*args, **kwargs)
        return lambda function: _cuda_custom_bwd(function)

from abs_interface import Abstract_SegModel


class GradReversalLayer(Function):
    @custom_fwd(device_type="cuda") # this help us to use the autocast autimatically we can specify the device, and datatype which compatible with autocast later we need
    # @staticmethod
    def forward(ctx: Any, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        
        return x.view_as(x)
    
    @custom_bwd(device_type="cuda")
    # @staticmethod
    def backward(ctx, grad_outputs: Tensor):
        outputs = grad_outputs.neg() * ctx.alpha
        # print(f"{r_}grad_output: {grad_outputs}{sr_}, output: {outputs}")
        return outputs, None

class ConditionalGradReversal:
    """Conditional wrapper for GradReversalLayer that applies gradient reversal only when enabled."""
    
    @staticmethod
    def apply(x: Tensor, alpha: float, apply_reversal: bool = True) -> Tensor:
        """
        Conditionally apply gradient reversal.
        
        Args:
            x: Input tensor
            alpha: Gradient reversal strength
            apply_reversal: Whether to apply gradient reversal (default: True)
            
        Returns:
            Tensor with or without gradient reversal applied
        """
        if apply_reversal:
            return GradReversalLayer.apply(x, alpha)
        else:
            return x

def forward_Chromosome_MassNet(model: Abstract_SegModel, x: Tensor) -> Tuple[Tensor, Tensor]:
    x = model.backbone[0](x)

    aggregated = []
    for layer in model.backbone[1:]:
        aggregated.append(layer(x))

    aggregated.append(model.neck(x))
    feature_aggregation = torch.concat(aggregated, dim = 1)

    feature_head_specifies = model.head[:-2](feature_aggregation)

    chromosome_segmentation_output = model.head.ChromosomeSegmentation(feature_head_specifies)
    overlap_segmentation_output = model.head.OverlapChromosomeSegmentation(feature_head_specifies)

    # Always return consistent format: (features, (chromosome, overlap))
    # DANNModelWrapper expects (features, segmentation_output)
    return feature_aggregation, (chromosome_segmentation_output, overlap_segmentation_output)

def forward_Chromosome_MassNet_PointRend(model: Abstract_SegModel, x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Forward pass for MassNet with PointRend head for chromosome segmentation.

    The chromosome head uses PointRend for fine-grained boundary refinement,
    while the overlap head remains a simple convolutional head.

    Args:
        model: The MassNet model with PointRend configuration
        x: Input tensor of shape (B, C, H, W)

    Returns:
        Tuple of (feature_aggregation, (chromosome_output, overlap_output))
        - feature_aggregation: Concatenated features from backbone and neck
        - chromosome_output: Refined chromosome segmentation from PointRend
        - overlap_output: Simple overlap segmentation
    """
    # Process input through backbone
    x = model.backbone[0](x)  # Input convolution (stride=2)

    # Aggregate multi-scale features from backbone (keep separate for PointRend)
    scale_8_features = model.backbone[1](x)  # Scale_8_Block output (32 channels)
    scale_4_features = model.backbone[2](x)  # Scale_4_Block output (64 channels)
    scale_2_features = model.backbone[3](x)  # Scale_2_Block output (64 channels)
    neck_features = model.neck(x)            # Neck output (128 channels)

    # Concatenate features for traditional heads
    feature_aggregation = torch.concat([scale_8_features, scale_4_features, scale_2_features, neck_features], dim=1)

    # Process through shared head layers (for overlap head)
    feature_head_specifies = model.head[:-3](feature_aggregation)

    # === Chromosome Segmentation with PointRend ===
    # Use REFINED features (64ch) instead of raw features (288ch)
    # This gives PointRend the same feature quality as traditional head
    pointrend_features = {
        "features_refined": feature_head_specifies,  # 64 channels, 112×112, PROCESSED through 3 conv layers
    }

    # PointRendSemSegHead returns predictions only (losses computed externally by MultiHeadTrainGenerator)
    chromosome_segmentation_output = model.head.ChromosomeSegmentation(pointrend_features)

    # === Overlap Segmentation (unchanged, simple convolution) ===
    overlap_feature_head_specifies = model.head[3](feature_head_specifies)
    overlap_segmentation_output = model.head.OverlapChromosomeSegmentation(overlap_feature_head_specifies)

    # Always return consistent format: (features, (chromosome, overlap))
    # DANNModelWrapper expects (features, segmentation_output)
    return feature_aggregation, (chromosome_segmentation_output, overlap_segmentation_output)

def forward_Chromosome_MassNet_Metaphase(model: Abstract_SegModel, x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Forward pass for MassNet with single 47-class segmentation head for metaphase chromosomes.

    Args:
        model: The MassNet model configured for 47-class segmentation
        x: Input tensor of shape (B, C, H, W) where C=1 for grayscale

    Returns:
        Tuple of (feature_aggregation, chromosome_output)
        - feature_aggregation: Concatenated features from backbone and neck
        - chromosome_output: Semantic segmentation output with 47 classes (B, 47, H, W)
    """
    # Process input through backbone
    x = model.backbone[0](x)  # Input convolution (stride=2)

    # Aggregate multi-scale features from backbone
    aggregated = []
    for layer in model.backbone[1:]:
        aggregated.append(layer(x))

    # Add neck features
    aggregated.append(model.neck(x))
    feature_aggregation = torch.concat(aggregated, dim=1)

    # Process through head layers (all except ChromosomeSegmentation)
    feature_head_specifies = model.head[:-1](feature_aggregation)

    # Final segmentation output (47 classes)
    chromosome_segmentation_output = model.head.ChromosomeSegmentation(feature_head_specifies)

    # Return in consistent format: (features, segmentation_output)
    return feature_aggregation, chromosome_segmentation_output

def forward_segformer_mit_b5(model: Abstract_SegModel, x: Tensor) -> Tuple[Tensor, Tensor]:
    x = model.backbone(x)
    x = model.head(x)
    return x

def forward_LRFormer(model: Abstract_SegModel, x: Tensor) -> Tuple[Tensor, Tensor]:
    return model.forward(x)

def forward_ResUnet(model: Abstract_SegModel, x: Tensor) -> Tuple[Tensor, Tensor]:
    x1 = model.backbone[0].Input_Layer(x) + model.backbone[0][2:](x)

    x2 = model.backbone[1].Residual_Conv_1(x1)
    x3 = model.backbone[1].Residual_Conv_2(x2)

    x4 = model.neck(x3)

    x4 = model.head.Upsample_1(x4)
    x5 = torch.cat([x4, x3], dim=1)

    x6 = model.head.Upsample_Residual_1(x5)

    x6 = model.head.Upsample_2(x6)
    x7 = torch.cat([x6, x2], dim=1)

    x8 = model.head.Upsample_Residual_2(x7)

    x8 = model.head.Upsample_3(x8)
    x9 = torch.cat([x8, x1], dim=1)

    x10 = model.head.Upsample_Residual_3(x9)

    chromosome = model.head.ChromosomeSegmentation(x10)
    overlap = model.head.OverlapChromosomeSegmentation(x10)
    
    return chromosome, overlap

def forward_ResUnetPlus(model: Abstract_SegModel, x: Tensor) -> Tuple[Tensor, Tensor]:
    x1 = model.backbone[0].Input_layer_extend(model.backbone[0].Input_Layer_encode(x)) + model.backbone[0].Input_Skip(x)
    
    x2 = model.backbone[1].SEBlock_1(x1)
    x2 = model.backbone[2].ResidualConv_1(x2)

    x3 = model.backbone[1].SEBlock_2(x2)
    x3 = model.backbone[2].ResidualConv_2(x3)

    x4 = model.backbone[1].SEBlock_3(x3)
    x4 = model.backbone[2].ResidualConv_3(x4)

    x5 = model.neck[0](x4)
    
    x6 = model.neck[1].Attention_1(x3, x5)
    x6 = model.neck[1].Upsampler_1(model.neck[1].BatchNorm_1(x6))
    x6 = torch.cat([x6, x3], dim=1)
    x6 = model.neck[1].ResidualConv_1(x6)

    x7 = model.neck[1].Attention_2(x2, x6)
    x7 = model.neck[1].Upsampler_2(model.neck[1].BatchNorm_2(x7))
    x7 = torch.cat([x7, x2], dim=1)
    x7 = model.neck[1].ResidualConv_2(x7)

    x8 = model.neck[1].Attention_3(x1, x7) 
    x8 = model.neck[1].Upsampler_3(model.neck[1].BatchNorm_3(x8))
    x8 = torch.cat([x8, x1], dim=1)
    x8 = model.neck[1].ResidualConv_3(x8)
 
    segmentation_feature = model.head[0](x8)
    
    chromosome = model.head.ChromosomeSegmentation(segmentation_feature) 
    
    overlap = model.head.OverlapChromosomeSegmentation(segmentation_feature)
    
    return chromosome, overlap

def forward_FPN(model: Abstract_SegModel, x: Tensor) -> Tuple[Tensor, Tensor]:
    return model.forward(x)

def forward_DNANet_baseline(model: Abstract_SegModel, input: torch.Tensor) -> Tensor:
    x0_0 = model.backbone.Conv_Block_1_0 (input)
    x1_0 = model.backbone.Conv_Block_2_0(model.auxiliary.Pool(x0_0))
    x0_1 = model.backbone.Conv_Block_1_1(torch.cat([x0_0, model.auxiliary.Upsample(x1_0)], 1))

    x2_0 = model.backbone.Conv_Block_3_0(model.auxiliary.Pool(x1_0))
    x1_1 = model.backbone.Conv_Block_2_1(torch.cat([x1_0, model.auxiliary.Upsample(x2_0),model.auxiliary.Downsample(x0_1)], 1))
    x0_2 = model.backbone.Conv_Block_1_2(torch.cat([x0_0, x0_1, model.auxiliary.Upsample(x1_1)], 1))

    x3_0 = model.backbone.Conv_Block_4_0(model.auxiliary.Pool(x2_0))
    x2_1 = model.backbone.Conv_Block_3_1(torch.cat([x2_0, model.auxiliary.Upsample(x3_0),model.auxiliary.Downsample(x1_1)], 1))
    x1_2 = model.backbone.Conv_Block_2_2(torch.cat([x1_0, x1_1, model.auxiliary.Upsample(x2_1),model.auxiliary.Downsample(x0_2)], 1))
    x0_3 = model.backbone.Conv_Block_1_3(torch.cat([x0_0, x0_1, x0_2, model.auxiliary.Upsample(x1_2)], 1))

    x4_0 = model.backbone.Conv_Block_5_0(model.auxiliary.Pool(x3_0))
    x3_1 = model.backbone.Conv_Block_4_1(torch.cat([x3_0, model.auxiliary.Upsample(x4_0),model.auxiliary.Downsample(x2_1)], 1))
    x2_2 = model.backbone.Conv_Block_3_2(torch.cat([x2_0, x2_1, model.auxiliary.Upsample(x3_1),model.auxiliary.Downsample(x1_2)], 1))
    x1_3 = model.backbone.Conv_Block_2_3(torch.cat([x1_0, x1_1, x1_2, model.auxiliary.Upsample(x2_2),model.auxiliary.Downsample(x0_3)], 1))
    x0_4 = model.backbone.Conv_Block_1_4(torch.cat([x0_0, x0_1, x0_2, x0_3, model.auxiliary.Upsample(x1_3)], 1))
        
    Final_x0_4 = model.neck.Conv_Block_1_4_final(torch.cat([
        model.head.Segmentation_branch_Upsample_block.interpolation_upsample_16(model.head.reconstruction_branch[0](x4_0)), model.head.Segmentation_branch_Upsample_block.interpolation_upsample_8(model.head.reconstruction_branch[1](x3_1)),
        model.head.Segmentation_branch_Upsample_block.interpolation_upsample_4(model.head.reconstruction_branch[2](x2_2)), 
        model.head.Segmentation_branch_Upsample_block.interpolation_upsample_2(model.head.reconstruction_branch[3](x1_3)), 
        x0_4
    ], 1))
    
    chromosome = model.head.reconstruction_branch.final_conv_chromosome(Final_x0_4) 
    overlap = model.head.reconstruction_branch.final_conv_overlap(Final_x0_4)
    
    return chromosome, overlap

def forward_PSFNet(model: Abstract_SegModel, x: Tensor) -> Tuple[Tensor, Tensor]:
    x_residual = model.backbone.Input_convolution.Conv_1(x)
    x_fusing = model.backbone.Input_pooling(x_residual + model.backbone.Input_convolution(x))
    
    #^ ===== STREAM A =====
    x_residual_stream_a = model.neck.StreamA.Scale_1.DilatationConvolution(x_fusing)
    scale_1_x_stream_a = x_residual_stream_a + model.neck.StreamA.Scale_1(x_fusing)
    
    scale_2_x_stream_a = model.neck.StreamA.Scale_2[:3](scale_1_x_stream_a) + model.neck.StreamA.Scale_2(scale_1_x_stream_a)
    scale_3_x_stream_a = model.neck.StreamA.Scale_3[:4](scale_2_x_stream_a) + model.neck.StreamA.Scale_3(scale_2_x_stream_a)
    scale_4_x_stream_a = model.neck.StreamA.Scale_4(scale_3_x_stream_a)
    
    #^ ===== STREAM B =====
    x_residual_stream_b = model.neck.StreamB.Scale_1.DilatationConvolution(x_fusing)
    scale_1_x_stream_b = x_residual_stream_b + model.neck.StreamB.Scale_1(x_fusing)
    
    scale_2_x_stream_b = model.neck.StreamB.Scale_2[:3](scale_1_x_stream_b) + model.neck.StreamB.Scale_2(scale_1_x_stream_b)
    scale_3_x_stream_b = model.neck.StreamB.Scale_3[:4](scale_2_x_stream_b) + model.neck.StreamB.Scale_3(scale_2_x_stream_b)
    scale_4_x_stream_b = model.neck.StreamB.Scale_4(scale_3_x_stream_b)
    
    feature_fused = torch.cat([scale_4_x_stream_a, scale_4_x_stream_b], dim=1)
    feature_fused = model.neck.fusingbridge.Fusing(feature_fused) + x_residual_stream_b
    
    feature_fused_decoded = model.neck.fusingbridge.Decoding(feature_fused) + x_residual_stream_a
    
    blastocyst = model.head(feature_fused_decoded)
    
    return feature_fused_decoded, blastocyst

def forward_Xception_Adversarial(model: Abstract_SegModel, x: Tensor) -> Tuple[Tensor, Tensor]:
    x = model.backbone(x)
    x = model.neck(x)
    
    x = nn.ReLU(inplace=True)(x)

    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.view(x.size(0), -1)

    y = model.head(x)
    
    return x, y

def forward_Adaptation_PSFNet(model: Abstract_SegModel, x: Tensor, epsilon: float, apply_grad_reversal: bool = True) -> Tuple[Tensor, Tensor]:
    x_residual = model.backbone.Input_convolution.Conv_1(x)
    x_fusing = model.backbone.Input_pooling(x_residual + model.backbone.Input_convolution(x))
    
    #^ ===== STREAM A =====
    x_residual_stream_a = model.neck.StreamA.Scale_1.DilatationConvolution(x_fusing)
    scale_1_x_stream_a = x_residual_stream_a + model.neck.StreamA.Scale_1(x_fusing)
    
    scale_2_x_stream_a_residual = model.neck.StreamA.Scale_2_1(scale_1_x_stream_a)
    scale_2_x_stream_a = scale_2_x_stream_a_residual + model.neck.StreamA.Scale_2_2(scale_2_x_stream_a_residual)
    
    scale_3_x_stream_a_residual = model.neck.StreamA.Scale_3_1(scale_2_x_stream_a)
    scale_3_x_stream_a =  model.neck.StreamA.Scale_3_2(scale_3_x_stream_a_residual)
    scale_4_x_stream_a = model.neck.StreamA.Scale_4(scale_3_x_stream_a)
    
    #^ ===== STREAM B =====
    x_residual_stream_b = model.neck.StreamB.Scale_1.DilatationConvolution(x_fusing)
    scale_1_x_stream_b = x_residual_stream_b + model.neck.StreamB.Scale_1(x_fusing)
    
    scale_2_x_stream_b_residual = model.neck.StreamB.Scale_2_1(scale_1_x_stream_b)
    scale_2_x_stream_b =  scale_2_x_stream_b_residual + model.neck.StreamB.Scale_2_2(scale_2_x_stream_b_residual)
    
    scale_3_x_stream_b_residual = model.neck.StreamB.Scale_3_1(scale_2_x_stream_b)
    scale_3_x_stream_b = scale_3_x_stream_b_residual + model.neck.StreamB.Scale_3_2(scale_3_x_stream_b_residual)
    scale_4_x_stream_b = model.neck.StreamB.Scale_4(scale_3_x_stream_b)
    
    feature_fused = torch.cat([scale_4_x_stream_a, scale_4_x_stream_b], dim=1)
    feature_fused = model.neck.fusingbridge.Fusing(feature_fused) + x_residual_stream_b
    
    feature_fused_decoded = model.neck.fusingbridge.Decoding(feature_fused) + x_residual_stream_a
    
    normalized_feature_fused_decoded = model.head.NormalizeFeature(feature_fused_decoded)
    
    latent_space_domain_adaptation =  F.adaptive_avg_pool2d(normalized_feature_fused_decoded, (1, 1))
    latent_space_domain_adaptation = latent_space_domain_adaptation.view(x.size(0), -1)
    latent_space_domain_adaptation = model.head.feature_flatten(latent_space_domain_adaptation)
    latent_space_domain_adaptation_gradient_reversal = ConditionalGradReversal.apply(latent_space_domain_adaptation, epsilon, apply_grad_reversal)
    
    transpose_feature = model.head.TransposeConvolutionBlock(normalized_feature_fused_decoded)
    transpose_feature_gradient_reversal = ConditionalGradReversal.apply(transpose_feature, epsilon, apply_grad_reversal)
    
    chromosome = model.head.ChromosomeSegmentation.Chromosome(transpose_feature)
    overlap = model.head.ChromosomeSegmentation.Overlap(transpose_feature)
    
    return latent_space_domain_adaptation_gradient_reversal, transpose_feature_gradient_reversal, (chromosome, overlap)

def forward_maskformer384_UDA4inst(model: Abstract_SegModel, x: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
    features = model.backbone(x)
    outputs = model.head(features)
    return outputs

def forward_Massnet_DACS(model: Abstract_SegModel, x: Tensor) -> Tuple[Tensor, Tensor]:
    x = model.backbone[0](x)
    
    aggregated = []
    for layer in model.backbone[1:]:
        aggregated.append(layer(x))
    
    aggregated.append(model.neck(x))
    feature_aggregation = torch.concat(aggregated, dim = 1)
    
    segmentation_feature = model.head(feature_aggregation)
    
    return segmentation_feature
