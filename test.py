import argparse
import os

import pandas as pd
import torch
import torch.nn as nn

from core.config import config
from core.function import test
from utils.utils import determine_device, save_checkpoint, update_config, create_logger, setup_seed
from models.model import OSnet
from dataset.dataloader import get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='experiments_pre_1e-5_zscore(100)/5fold',
                        help='path for pretrained weights', type=str)
    parser.add_argument('--fold', default=-1, help='which data fold to train on', type=int)
    parser.add_argument('--dir', default='_712_1015_15e-4(100)', help='path suffix', type=str)
    parser.add_argument('--times', default='all', help='test times', type=str)
    parser.add_argument('--split_f', default='splits_712_1015.pkl', help='path suffix', type=str)
    args = parser.parse_args()
    return args


def main(args):
    net = OSnet
    devices = config.TRAIN.DEVICES
    model = net("fixed")
    model = nn.DataParallel(model, devices).cuda()

    if args.times == '2020':
        root = 'DATA/BraTs2020/image_large'
        def_root = 'DATA/BraTs2020/def_length/'
    else:
        root = 'DATA/zhengdayi_' + args.times + '_large'
        def_root = 'DATA/def_length/' + args.times
    if args.times == 'all':
        test_dataset = get_dataset(args.fold, args.split_f, 'test', config, root, def_root)
    else:
        test_dataset = get_dataset(-1, 'splits_' + args.times + '.pkl', 'val', config, root, def_root)

    if not os.path.exists('log' + args.dir):
        os.makedirs('log' + args.dir)

    best_perf = 0.0
    logger = create_logger('log' + args.dir, 'test_5fold' + str(args.fold) + '_' + args.times + '.log')

    c_index = []
    for epoch in range(100):
        ld_model = 'experiments' + args.dir + '/5fold' + str(args.fold) + '_checkpoint_' + str(epoch) + '.pth'
        checkpoint = torch.load(ld_model)
        model.load_state_dict(checkpoint['state_dict'])

        logger.info(f'Epoch: [{epoch}]')
        perf = test(model, test_dataset, logger, config, best_perf)
        c_index.append(perf)
        if perf > best_perf:
            best_perf = perf

        df = pd.DataFrame(c_index, columns=['c_index'])
        df.to_csv('log' + args.dir + '/c_index_fold' + str(args.fold) + '_' + args.times + '.csv', index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
