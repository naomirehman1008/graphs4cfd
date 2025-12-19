'''
    Script for training the NsOneScaleGNN model on the NsCircle dataset.
    This model is referred to as the 1S-GNN in Lino et al. (2022) https://doi.org/10.1063/5.0097679.
'''


import torch
from torchvision import transforms
import graphs4cfd as gfd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Physics Constants
    parser.add_argument("--rho", default=1.0)
    parser.add_argument("--nu",  default=0.001)
    parser.add_argument("--dt",  default=1.0)

    # Loss Weights
    parser.add_argument("--mse_weight", default=1.0)

    parser.add_argument("--div_weight", default=1.0)
    parser.add_argument("--mom_weight", default=0.1)
    parser.add_argument("--bc_weight", default=0.0)
    parser.add_argument("--spec_weight", default=0.05)

    parser.add_argument("--from_checkpoint", default=None)

    args = parser.parse_args()
    name = 'NsOneScaleGNNphysics' + \
            "_rho_" + str(args.rho) + \
            "_nu_" + str(args.nu) + \
            "_dt_" + str(args.dt) + \
            '_mse_' + str(args.mse_weight) + \
            '_div_' + str(args.div_weight) + \
            '_mom_' + str(args.mom_weight) + \
            '_bc_' + str(args.bc_weight) + \
            '_spec_' + str(args.spec_weight)

    # Training configuration
    train_config = gfd.nn.TrainConfig(
        name            = name,
        folder          = '.',
        tensor_board    = '.',
        checkpoint      = args.from_checkpoint,
        chk_interval    = 1,
        training_loss   = gfd.nn.losses.GraphLossWPhysicsLoss(lambda_d=0.25,
            mse_weight=float(args.mse_weight),
            div_weight=float(args.div_weight),
            mom_weight=float(args.mom_weight),
            bc_weight=float(args.bc_weight),
            spec_weight=float(args.spec_weight),
            dt=float(args.dt)),
        validation_loss = gfd.nn.losses.GraphLossWPhysicsLoss(
            mse_weight=float(args.mse_weight),
            div_weight=float(args.div_weight),
            mom_weight=float(args.mom_weight),
            bc_weight=float(args.bc_weight),
            spec_weight=float(args.spec_weight),
            dt=float(args.dt)),
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
    dataset = gfd.datasets.NsCircle(format='uvp', path=path, training_info={"n_in":1, "n_out":train_config['num_steps'][-1], "step":1, "T":100}, transform=transform) # If enough memory, set preload=True
    train_set, test_set = torch.utils.data.random_split(dataset, [1000,32])
    train_loader = gfd.DataLoader(train_set, batch_size=train_config['batch_size'], shuffle=True,  num_workers=4)   
    val_loader  = gfd.DataLoader(test_set,  batch_size=train_config['batch_size'], shuffle=False, num_workers=4)   


    # Model definition
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (5, (128,128,128), False),
        # Level 1
        "mp11": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp12": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp13": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp14": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp15": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp16": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp17": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp18": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    }

    model = gfd.nn.NsOneScaleGNN(arch=arch)

    print("Number of trainable parameters: ", model.num_params)


    # Training
    model.fit(train_config, train_loader, val_loader=val_loader)

