"""
Code modified based on the SimVP (https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction).
The Temporal_block references involution https://github.com/d-li14/involution

Special thanks to their contributions!

note: run this file after downloading the dataset, whose download command is given at README.txt.
date: August 16, 2023.

e-mail: any questions, please contact with me: ganlq@std.uestc.edu.cn
"""


import configs
from experiment_cfg import Experiment_cfg
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    args = configs.create_parser().parse_args()
    config = args.__dict__
    run = Experiment_cfg(args)
    print('*************  start *************')
    run.train(args)
    print('************* testing *************')
    run.test(args)