import argparse
import torch

#(args.backbone, args.setting, args.trainset, args.shared) #args.expname only makes sense if args.setting is not allgame

#if args.backbone e2e.. then the gradients flow through the backbone during training
def get_args():
    parser = argparse.ArgumentParser(description='Prtr')

    parser.add_argument(
        "--arch",
        choices=["standard", "resnet"],
        default="e2e",
    )
    parser.add_argument(
        "--model",
        choices=["4STACK_VAE_ATARI", "3CHANRGB_VAE_ATARI101", "1CHAN_VAE_ATARI101", "3CHAN_VAE_ATARI", "1CHAN_VAE_ATARI", "1CHAN_CONT_ATARI", "4STACK_CONT_ATARI", "DUAL_4STACK_CONT_ATARI"],
        default="e2e",
    )
    parser.add_argument(
        "--machine",
        choices=["iGpu", "iGpu8", "iGpu10", "iGpu11", "iGpu14", "iGpu9", "iGpu24", "iGpu21", "iGpu15"],
        default="e2e",
    )
    parser.add_argument(
        "--load_checkpoint", action="store_true", default=False, help="Load ckpt or not"
    )

    parser.add_argument(
        "--test", action="store_true", default=False, help="test mode or train mode"
    )
    
    parser.add_argument(
        "--save_dir", type=str, default="/lab/kiran/ckpts/pretrained/atari/", help="pretrained results"
    )
    parser.add_argument(
        "--model_path", type=str, default="", help="pretrained results"
    )    
    parser.add_argument(
        "--expname", type=str, default="all", help="pretrained results"
    )
    parser.add_argument(
        "--nepoch", type=int, default=200, help="pretrained results"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID"
    )
    parser.add_argument(
        "--sample_next", type=int, default=5, help="GPU ID"
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=64, help="pretrained results"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=256, help="pretrained results"
    )
    parser.add_argument(
        "--kl_weight", type=float, default=0.04, help="pretrained results"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-6, help="pretrained results"
    )

    args = parser.parse_args()
    return args
