import numpy as np
import random
import torch
import json
from torchvision import datasets


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(24)

train_ids = [i for i in range(60000)]
test_ids = [60000 + i for i in range(10000)]

train_clusters = {i: [] for i in range(10)}
test_clusters = {i: [] for i in range(10)}

clusters_labels = np.load('./clusters.npy')
for i in range(len(clusters_labels)):
    if i < 60000:
        train_clusters[clusters_labels[i]].append(i)
    else:
        test_clusters[clusters_labels[i]].append(i - 60000)

for key, val in train_clusters.items():
    print(key, len(val))

for key, val in test_clusters.items():
    print(key, len(val))

client_idx = 0
client_ids = {i: {'train': [], 'test': []} for i in range(100)}
for key, val in train_clusters.items():
    train_shard_size = int(len(val) / 10)
    test_shard_size = int(len(test_clusters[key]) / 10)
    random.shuffle(val)
    random.shuffle(test_clusters[key])
    for i in range(10):
        client_ids[client_idx]['train'] = val[i*train_shard_size: (i+1)*train_shard_size]
        client_ids[client_idx]['test'] = test_clusters[key][i*test_shard_size: (i+1)*test_shard_size]
        client_idx += 1

assert client_idx == 100

with open('./latent_distribution.json', 'w') as f:
    json.dump(client_ids, f)



