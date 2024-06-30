import argparse
from models.CNN.VGGnet import VGG

def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--net', type=str, default='VGG')
    return parser

def training(args):
    for epoch in range(args.epoch):
        print(epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pytorch model training', parents=[get_parser()])
    args = parser.parse_args()
    print(args)

    if args.net == 'VGG':
        model = VGG()
        training(args)
