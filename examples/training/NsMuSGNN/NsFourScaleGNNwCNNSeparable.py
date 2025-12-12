'''
    Script for training the NsFourScaleGNN model on the NsCircle dataset.
    This model is referred to as the 4S-GNN in Lino et al. (2022) https://doi.org/10.1063/5.0097679.
'''


import torch
from torchvision import transforms
import graphs4cfd as gfd


# Training configuration
train_config = gfd.nn.TrainConfig(
    name            = 'NsFourScaleGNNwCNNSeparable',
    folder          = '.',
    tensor_board    = '.',
    chk_interval    = 1,
    training_loss   = gfd.nn.losses.GraphLoss(lambda_d=0.25),
    validation_loss = gfd.nn.losses.GraphLoss(),
    epochs          = 500,
    num_steps       = [i for i in range(1,11)],
    add_steps       = {'tolerance': 0.005, 'loss': 'training'},
    batch_size      = 8,
    lr              = 1e-5,
    grad_clip       = {"epoch": 0, "limit": 1},
    scheduler       = {"factor": 0.5, "patience": 5, "loss": 'training'},
    stopping        = 1e-8,
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Training datasets
path = "/home/nrehman/naomi_cfd/NsCircle/train/NsCircle.h5" # Replace with path to NsCircle.h5 (available at https://doi.org/10.5281/zenodo.7870707)
transform = transforms.Compose([
    gfd.transforms.ConnectKNN(6, period=[None, "auto"]),
    gfd.transforms.ScaleNs({'u': (-2.1,2.6), 'v': (-2.25,2.1), 'p': (-3.7,2.35), 'Re': (500,1000)}, format='uvp'),
    gfd.transforms.ScaleEdgeAttr(0.1),
    gfd.transforms.RandomGraphRotation(eq='ns', format='uvp'),
    gfd.transforms.RandomGraphFlip(eq='ns', format='uvp'),
    gfd.transforms.AddUniformNoise(0.01)
])
batch_transform = transforms.Compose([
    gfd.transforms.GridClustering([0.15, 0.30, 0.60]),
])
dataset = gfd.datasets.NsCircle(format='uvp', path=path, training_info={"n_in":1, "n_out":train_config['num_steps'][-1], "step":1, "T":100}, transform=transform) # If enough memory, set preload=True
train_set, test_set = torch.utils.data.random_split(dataset, [1000,32])
train_loader = gfd.DataLoader(
    train_set,
    batch_size  = train_config['batch_size'],
    shuffle     = True,
    transform   = batch_transform,
    num_workers = 4
)   
val_loader  = gfd.DataLoader(
    test_set, 
    batch_size  = train_config['batch_size'],
    shuffle     = False,
    transform   = batch_transform,
    num_workers = 4
)   


# Model definition
h_dim = 128
arch = {
    ################ Edge-functions ################## Node-functions ##############
    # Encoder
    "edge_encoder": (2, (h_dim,h_dim,h_dim), False),
    "node_encoder": (5, (h_dim,h_dim,h_dim), False),
    # Level 1
    "mp111": ((h_dim+2*h_dim, (h_dim,h_dim,h_dim), True), (h_dim+h_dim, (h_dim,h_dim,h_dim), True)),
    "mp112": ((h_dim+2*h_dim, (h_dim,h_dim,h_dim), True), (h_dim+h_dim, (h_dim,h_dim,h_dim), True)),
    "mp113": ((h_dim+2*h_dim, (h_dim,h_dim,h_dim), True), (h_dim+h_dim, (h_dim,h_dim,h_dim), True)),
    "mp114": ((h_dim+2*h_dim, (h_dim,h_dim,h_dim), True), (h_dim+h_dim, (h_dim,h_dim,h_dim), True)),
    "down_mp12": (2+h_dim, (h_dim,h_dim,h_dim), True),
    # Level 2
    # INPUT AND OUTPUT SIZE OF RCN MUST BE THE SAME!!
    "cn211": (h_dim+2, h_dim*2, 3, 1),
    "rcn211": ((h_dim*2, 3, 1)),
    "down_cn23": ((2, 2, 0, 1)),
    # Level 3
    "cn311": (h_dim*2, h_dim*2*2, 3, 1),
    "rcn311": ((2*2*h_dim, 3, 1)),
    "down_cn34": ((2, 2, 0, 1)),
    # Level 4
    "cn411": (h_dim*2*2, h_dim*2*2*2, 3, 1),
    "rcn411": ((2*2*2*h_dim, 3, 1)),
    "up_cn43": ((2*2*2*h_dim, 2*2*h_dim, 3, 2, 1, 1, 1)),
    # Level 3
    "cn321": ((2*2*h_dim + 2*2*h_dim, 2*2*h_dim, 3, 1)),
    "rcn321": ((2*2*h_dim, 3, 1)),
    "up_cn32": ((2*2*h_dim, 2*h_dim, 3, 2, 1, 1, 1)),
    # Level 2
    "cn221": ((2*h_dim + 2*h_dim, 2*h_dim, 3, 1)),
    "rcn221": ((2*h_dim, 3, 1)),
    "up_mp21": (2+2*h_dim+h_dim, (h_dim,h_dim,h_dim), True),
    # Level 1
    "mp121": ((h_dim+2*h_dim, (h_dim,h_dim,h_dim), True), (h_dim+h_dim, (h_dim,h_dim,h_dim), True)),
    "mp122": ((h_dim+2*h_dim, (h_dim,h_dim,h_dim), True), (h_dim+h_dim, (h_dim,h_dim,h_dim), True)),
    "mp123": ((h_dim+2*h_dim, (h_dim,h_dim,h_dim), True), (h_dim+h_dim, (h_dim,h_dim,h_dim), True)),
    "mp124": ((h_dim+2*h_dim, (h_dim,h_dim,h_dim), True), (h_dim+h_dim, (h_dim,h_dim,h_dim), True)),
    # Decoder
    "decoder": (h_dim, (h_dim,h_dim,3), False),
}
model = gfd.nn.NsFourScaleGNNwCNNSeparable(arch=arch)


# Training
model.fit(train_config, train_loader, val_loader=val_loader)

