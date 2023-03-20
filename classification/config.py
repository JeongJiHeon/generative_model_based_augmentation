import argparse

# train_path_lists = [
#     '/workspace/jjh/5. Glioma_GAN_Research/data/Glioma_AMC_GBM_TCGA_Final_DataSet/BET_DONE_Final_to_MNI_to_NPY_MASK_Larger_than_100pixel/train/GBM/IDH_mutant',
#     '/workspace/jjh/5. Glioma_GAN_Research/data/Glioma_AMC_GBM_TCGA_Final_DataSet/BET_DONE_Final_to_MNI_to_NPY_MASK_Larger_than_100pixel/train/LGG/IDH_mutant',
#     '/workspace/jjh/5. Glioma_GAN_Research/data/Glioma_AMC_GBM_TCGA_Final_DataSet/BET_DONE_Final_to_MNI_to_NPY_MASK_Larger_than_100pixel/train/GBM/IDH_wild',
#     '/workspace/jjh/5. Glioma_GAN_Research/data/Glioma_AMC_GBM_TCGA_Final_DataSet/BET_DONE_Final_to_MNI_to_NPY_MASK_Larger_than_100pixel/train/LGG/IDH_wild',
#     '/workspace/jjh/5. Glioma_GAN_Research/score_data/normal/IDH_Mutant',
#     '/workspace/jjh/5. Glioma_GAN_Research/score_data/normal/IDH_Wild',

# ]

# val_path_lists = [
#     '/workspace/jjh/5. Glioma_GAN_Research/data/Glioma_AMC_GBM_TCGA_Final_DataSet/BET_DONE_Final_to_MNI_to_NPY_MASK_Larger_than_100pixel/val/GBM/IDH_mutant',
#     '/workspace/jjh/5. Glioma_GAN_Research/data/Glioma_AMC_GBM_TCGA_Final_DataSet/BET_DONE_Final_to_MNI_to_NPY_MASK_Larger_than_100pixel/val/LGG/IDH_mutant',
#     '/workspace/jjh/5. Glioma_GAN_Research/data/Glioma_AMC_GBM_TCGA_Final_DataSet/BET_DONE_Final_to_MNI_to_NPY_MASK_Larger_than_100pixel/val/GBM/IDH_wild',
#     '/workspace/jjh/5. Glioma_GAN_Research/data/Glioma_AMC_GBM_TCGA_Final_DataSet/BET_DONE_Final_to_MNI_to_NPY_MASK_Larger_than_100pixel/val/LGG/IDH_wild',


# ]

# test_path_lists = [
#     '/workspace/jjh/5. Glioma_GAN_Research/data/Glioma_AMC_GBM_TCGA_Final_DataSet/BET_DONE_Final_to_MNI_to_NPY_MASK_Larger_than_100pixel/test/GBM/IDH_mutant',
#     '/workspace/jjh/5. Glioma_GAN_Research/data/Glioma_AMC_GBM_TCGA_Final_DataSet/BET_DONE_Final_to_MNI_to_NPY_MASK_Larger_than_100pixel/test/LGG/IDH_mutant',
#     '/workspace/jjh/5. Glioma_GAN_Research/data/Glioma_AMC_GBM_TCGA_Final_DataSet/BET_DONE_Final_to_MNI_to_NPY_MASK_Larger_than_100pixel/test/GBM/IDH_wild',
#     '/workspace/jjh/5. Glioma_GAN_Research/data/Glioma_AMC_GBM_TCGA_Final_DataSet/BET_DONE_Final_to_MNI_to_NPY_MASK_Larger_than_100pixel/test/LGG/IDH_wild',
# #     '/workspace/jjh/5. Glioma_GAN_Research/data/External_Validation_Severance/LGG/Bet_done_Register_done_NPY_Mask_Larger_than_100/IDH_mutant',
# #     '/workspace/jjh/5. Glioma_GAN_Research/data/External_Validation_Severance/LGG/Bet_done_Register_done_NPY_Mask_Larger_than_100/IDH_wild',
# #     '/workspace/jjh/5. Glioma_GAN_Research/data/External_Validation_Severance/GBM/Bet_done_Register_done_NPY_Mask_Larger_than_100/IDH_mutant',
# #     '/workspace/jjh/5. Glioma_GAN_Research/data/External_Validation_Severance/GBM/Bet_done_Register_done_NPY_Mask_Larger_than_100/IDH_wild',
# ]

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--message', '--msg', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--original_path', default='/mnt')
    
    parser.add_argument('--clinical', type=str, default='no')
#     parser.add_argument('--real_ratio', type=float, default=1)
    
#     parser.add_argument('--train_path_lists', type=list, default=train_path_lists)
#     parser.add_argument('--val_path_lists', type=list, default=val_path_lists)
#     parser.add_argument('--test_path_lists', type=list, default=test_path_lists)
    
#     parser.add_argument('--mutant', type=int, default = 9248)
#     parser.add_argument('--wild', type=int, default = 19095)

    parser.add_argument('--fake_slice', type=int, default=0)
    parser.add_argument('--augment', default=False, action='store_true')
    parser.add_argument('--no_add',  default=False, action='store_true')
    parser.add_argument('--num', type=int, default = 0)
    
    parser.add_argument('--epochs', type=int, default=30, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--w', type=int, default=16)
    
    parser.add_argument('--log_dir', type=str, default="tensorboard")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints_jjh_for_revision")
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--aug', type=bool, default=None)
    parser.add_argument('--weighted_CE', type=str, default='False')
    parser.add_argument('--csv_file', type=str, default=None)
    parser.add_argument('--downstream_name', type=str, default=None)
    parser.add_argument('--fine_tuning', type=bool, default=False)
    parser.add_argument('--bit', type=int, default=8)
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='internal')
    
    
    
    
    
    return parser.parse_args()
