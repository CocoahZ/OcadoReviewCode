import utils
import pandas as pd
import numpy as np
import tqdm
import argparse
from matplotlib import pyplot as plt
import os


def main(args):
    data = utils.read_pd_xls(args.datapath)
    processed_sentence = utils.content2sentence(data, args.modelpath)
    emotion_point = np.array(processed_sentence['emotion label'])
    emotion_point.sort()
    # save figure
    x = np.arange(len(emotion_point))
    y = emotion_point
    plt.scatter(x, y, s=0.2)
    plt.xlabel('review id')
    plt.ylabel('emotion score')
    plt.savefig(os.path.join(args.savepath, 'curve.png'))

    # to csv
    processed_sentence.to_csv(os.path.join(args.savepath, 'processed_sentence.csv'),sep='\t', header=False)


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Sentimantic')
        parser.add_argument('--datapath', type=str, default='./data/')
        parser.add_argument('--modelpath', type=str, default='./checkpoints/tut4-model.pt')
        parser.add_argument('--savepath', type=str, default='./results')
        args = parser.parse_args()
        args.datapath = [os.path.join(args.datapath, 'Ocado data.xls'), os.path.join(args.datapath, 'Ocado _Mid June_update Azar.xlsx')]
        main(args)