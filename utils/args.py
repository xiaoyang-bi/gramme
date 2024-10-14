from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers


# @dataclass
# class ModelArguments:
#     # ===================================================================

#     # MoE Layer
#     # =============================================================
#     moe_enable: Optional[bool] = field(default=False)
#     ep_size: int = 1
#     # mlp_hidden_size_ratio:int = 4
#     num_experts: Optional[int] = field(default=16, metadata={"help": "number of experts for each moe layer."})
#     top_k_experts: int = field(
#         default=4,
#         metadata={
#             "help": "Top-k experts to deal with tokens.",
#             "choices": [1, 2],
#         },
#     )
#     capacity_factor: float = 4.
#     eval_capacity_factor: float = 4.
#     min_capacity: int = 4
#     use_residual: bool = False
#     model_type:str  = field(default="base")
    
#     # "microsoft/swinv2-tiny-patch4-window8-256"
#     # "WinKawaks/vit-tiny-patch16-224"
#     pretrain_name:str = field(default=None)
#     crop_size: Optional[int] = field(default=256)
#     chkpt_name: Optional[str] = field(default="weights/xxx.pth")
    
    
#     # =============================================================
#     # weather-aware router
#     idr_cls: Optional[int] = field(default=-1)
#     num_degra_queries: Optional[int] = field(default=24)
#     squad_num:Optional[int] = field(default=-1)
    
    
#     # =============================================================

# @dataclass
# class DataArguments:
#     pass


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    exp_name: Optional[str] = field(default="exp")
    samples_dir: Optional[str] = field(default="samples")
    data_dir: Optional[str] = field(default="../data/radiate")
    dataset:str = 'radiate'
    warmup_epochs:Optional[int] = field(default=0)
    eval_epochs:Optional[float] = field(default=1.)
    save_epochs:Optional[float] = field(default=5.)
    # train_sample_num: Optional[int] = field(default=-1)
    eval_sample_num: Optional[int] = field(default=-1)
    visualize: Optional[bool] = field(default = True)
    gpus_num: Optional[int] = field(default=1)
    pretrained: Optional[str] = field(default='weights/rfu_sap128_cr/checkpoint-3930/')
    with_masknet:bool = field(default = False)
    train_mode:str = field(default = 'train')
    img_aug:bool = field(default=True)
    img_norm:bool = field(default=False)
    radar_channels:int=field(default=1)
    radar_pov_vertical_aug:bool = field(default=False)
    resnet_layers:int=field(default=18)
    
    
    with_auto_mask: Optional[bool] = field(default=True)
    radar_cart_res: Optional[float] = 200./512
    radar_cart_pixels: Optional[int] = 512
    padding_mode:str = 'zeros'
    num_scales:int = 1
    with_mask:bool = True
    with_ssim:bool = True
    
    
    photo_loss_weight:float = 1.
    geometry_consistency_weight:float = 1.
    fft_loss_weight:float = 3e-4
    ssim_loss_weight:float = 1.
    sample_num:Optional[int]=-1 

    one_gpu:bool = False
    
    
    #pretrain_file
    # pretrain_name: Optional[str] = field(default="weights/base_0.pth")
    # batch_size:Optional[int] = field(default=4)
    # epoch_start:Optional[int] = field(default=0)
    # epoch_end:Optional[int] = field(default=200)
    # seed:Optional[int] = field(default=3407)
    # lr:Optional[float] = field(default=2e-4)
    # use_wandb:Optional[bool] = field(default=False)
    # cache_dir: Optional[str] = field(default=None)
    
    
   
