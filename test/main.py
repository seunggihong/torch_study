from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

import torch
import torch.distributed
import torch.multiprocessing.spawn
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse

from model import Model

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--path', type=str, default='/')
    parser.add_argument('--local_rank', type=int)
    return parser

def make_rdd(path):
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.csv(path, header=True, inferSchema=True)
    indexer = StringIndexer(inputCol='variety', outputCol='variety_enc')
    encoding_df = indexer.fit(df).transform(df).drop('variety')

    rdd = encoding_df.rdd.map(lambda row: (torch.tensor(row[:-1]), torch.tensor([row[-1]])))
    
    return rdd.collect()

def make_dataloader(rdd, batch_size=32):
    features = torch.stack([data[0] for data in rdd])
    labels = torch.tensor([data[1] for data in rdd])

    dataset = TensorDataset(features, labels)
    
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

def init_gpu(rank, opts):
    opts.rank = rank
    opts.gpu = opts.rank % torch.cuda.device_count()
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    
    if opts.rank is not None:
        print("Use GPU: {} SET".format(local_gpu_id))

    dist.init_process_group(backend='nccl', 
                                         init_method='tcp://127.0.0.1:' + str(opts.port), 
                                         world_size=opts.num_gpus, 
                                         rank=opts.rank)

    dist.barrier()

    setup_distributed(opts.rank == 0)
    print('opts :',opts)


def setup_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def cleanup():
    dist.destroy_process_group()
    

def main(rank, opts):
    init_gpu(rank, opts)
    local_gpu_id = opts.gpu

    rdd = make_rdd(opts.path)
    loader = make_dataloader(rdd)

    model = Model().to(rank)
    model.cuda(local_gpu_id)
    model = DDP(model, device_ids=[local_gpu_id])

    criterion = torch.nn.CrossEntropyLoss().to(local_gpu_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"[INFO] : START")
    for epoch in range(opts.epoch):
        for i, (x_train, x_label) in enumerate(loader):
            x_train = x_train.to(local_gpu_id)
            x_label = x_label.type(torch.LongTensor)
            x_label = x_label.to(local_gpu_id)
            ouputs = model(x_train)

            optimizer.zero_grad()
            loss = criterion(ouputs, x_label)
            loss.backward()
            print(f'Epoch [{epoch+1}/{opts.epoch}], Loss: {loss.item():.4f}')

        print(f'[INFO] : {epoch+1} epoch Done')

    print(f'[INFO] : Distributed Done')

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('RDD to Torch', parents=[get_args_parser()])
    opts = parser.parse_args()
    opts.num_gpus = torch.cuda.device_count()
    opts.gpu_ids = list(range(opts.num_gpus))
    opts.num_workers = opts.num_gpus * 4

    torch.multiprocessing.spawn(main, 
                                args=(opts,), 
                                nprocs=opts.num_gpus, 
                                join=True)
