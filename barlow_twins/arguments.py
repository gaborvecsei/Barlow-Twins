import argparse


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Path to the folder where we can find the images")
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("-ih",
                        "--height",
                        type=int,
                        default=224,
                        help="Height of the input images (raw images can differ from this)")
    parser.add_argument("-iw",
                        "--width",
                        type=int,
                        default=224,
                        help="Width of the input images (raw images can differ from this)")
    parser.add_argument("-p",
                        "--projector-units",
                        type=int,
                        default=8192,
                        help="Number of neurons in the dense layers of the projector part of the network")
    parser.add_argument("-b", "--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lmbda",
                        type=float,
                        default=5e-3,
                        help="Lambda in the loss function which scales the off diagonal loss")
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-we", "--lr-warmup-epochs", type=int, default=10, help="Number of epochs for the warmup")
    parser.add_argument("-o", "--optimizer", type=str, default="adam", choices=("adam", "sgd"), help="Optimizer")
    parser.add_argument("--base-log-folder",
                        type=str,
                        default="logs",
                        help="The base log folder where the experiments will be saved")
    parser.add_argument("-mxp",
                        "--mixed-precision",
                        action="store_true",
                        help="When set, mixed precision will be used.")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="If experiment folder already exists with same name, it will be removed")
    parser.add_argument("--print-freq",
                        type=int,
                        default=10,
                        help="Defines after how many steps we should print the training stats")

    args = parser.parse_args()
    return args
