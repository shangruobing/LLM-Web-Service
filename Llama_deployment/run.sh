python -m torch.distributed.run --nproc_per_node=1 --node_rank=0 main.py
nohup python -m torch.distributed.run --nproc_per_node=1 --node_rank=0 main.py > log.txt 2>&1 & echo $! > pid.txt
