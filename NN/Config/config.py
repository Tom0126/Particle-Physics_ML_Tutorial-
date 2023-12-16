import argparse

parser = argparse.ArgumentParser()
# base setting
parser.add_argument("--n_epoch", type=int, default=2, help="number of total epoch")
parser.add_argument("--resume", type=bool, default=False, help="whether resume from existing checkpoint")
parser.add_argument("--start_epoch", type=int, default=0, help="resume from a given epoch")
parser.add_argument("--gpu", type=bool, default=True, help="using gpu or not")
parser.add_argument("--set_seed", type=bool, default=False, help="set seed or not")
parser.add_argument("--seed", type=int, default=100, help="seed")
parser.add_argument("--log_interval", type=int, default=10, help="print log in this interval")
parser.add_argument("--val_interval", type=int, default=1, help="seed")
parser.add_argument("--load_ckp", type=bool, default=False, help="checkpoint")


parser.add_argument("--latent_dim", type=int, default=64, help="input noise dimention")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")

# datasets setting
parser.add_argument("--batch_size", "-b", type=int, default=4096, help="batch size for training and testing")
parser.add_argument("--num_workers", "-nw", type=int, default=0, help="the number of threads of dataloader")
parser.add_argument("--img_dir", "-id", type=str, default="", help="image datasets\' file path")
parser.add_argument("--channels", "-c", type=int, default=1, help="input channels")
parser.add_argument("--img_size", "-is", type=int, default=20, help="input images\' size")
parser.add_argument("--mean", type=float, default=0, help="the mean value to normalize datasets")
parser.add_argument("--std", type=float, default=1, help="the std value to normalize datasets")

# optimizier setting
parser.add_argument("--optim", type=str, default="SGD", help="optimizier\'s name")
parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5, help="learning rate for optimizier")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--momentum", "-m", type=float, default=0.9, help="momentum for SGD optimizier")
parser.add_argument("--lr_schedule", type=str, default="step", help="learning rate schedule: step or consine")
parser.add_argument("--l_gamma", type=float, default=0.1, help="learning rate scale factor")
parser.add_argument("--step", type=int, default=10, help="learning rate step size")
parser.add_argument("--worm_up", type=int, default=5, help="worm up strategy for learning rate schedule")
parser.add_argument("--weight_decay", "-wd", type=float, default=4e-5, help="weight decay for optimizier")
parser.add_argument("--short_cut", "-st",type=int, default=1, help="If using shortcut in resnet")
parser.add_argument("--b_xy", type=int, default=1, help="block in x-y")
parser.add_argument("--b_z", type=int, default=1, help="block in z")
parser.add_argument("--f_k", type=int, default=7, help="size of the first kernal")
parser.add_argument("--f_s", type=int, default=7, help="size of the first stride")
parser.add_argument("--f_p", type=int, default=7, help="size of the first padding")

# 
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--n_classes", type=int, default=4)
parser.add_argument("--standardize_static", type=bool, default=True, help="fix mean and std")




