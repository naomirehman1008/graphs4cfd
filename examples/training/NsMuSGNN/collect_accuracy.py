import torch
from torchvision import transforms
import numpy as np
import os
import graphs4cfd as gfd
import pandas as pd
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_df = pd.DataFrame(columns=['model_name','rho','nu','dt','mse','div','mom','bc','spec','dataset','accuracy','params'])

# NsCircle dataset (training dataset)
path = "/home/nrehman/naomi_cfd/NsCircle/train/NsCircle.h5"
scaling = {'u': (-2.1,2.6), 'v': (-2.25,2.1), 'p': (-3.7,2.35), 'Re': (500,1000)} # This tuples are the (min, max) values for each variable in the training dataset
transform = transforms.Compose([
    gfd.transforms.ConnectKNN(6, period=(None,"auto")), # Detect the periodicity along the y-axis
    gfd.transforms.ScaleNs(scaling, 'uvp'),
    gfd.transforms.ScaleEdgeAttr(0.1),
    gfd.transforms.GridClustering([0.15, 0.30, 0.60]), # This builds the four-scale graph
])
ns_circle = gfd.datasets.NsCircle('uvp', path=path, transform=transform) # The format is 'uvp' because the dataset contains the velocity components and the pressure

# NsCircleMidRe dataset (testing dataset)
path = "/home/nrehman/naomi_cfd/NsCircle/test/NsCircleMidRe.h5"
ns_circle_mid_re = gfd.datasets.NsCircle('uvp', path=path, transform=transform) # The format is 'uvp' because the dataset contains the velocity components and the pressure

# NsCircleLowRe dataset (Re-extrapolation dataset)
path = "/home/nrehman/naomi_cfd/NsCircle/test/NsCircleLowRe.h5"
ns_circle_low_re = gfd.datasets.NsCircle('uvp', path=path, transform=transform) # The format is 'uvp' because the dataset contains the velocity components and the pressure

# NsCircleHighRe dataset (Re-extrapolation dataset)
path = "/home/nrehman/naomi_cfd/NsCircle/test/NsCircleHighRe.h5"
ns_circle_high_re = gfd.datasets.NsCircle('uvp', path=path, transform=transform) # The format is 'uvp' because the dataset contains the velocity components and the pressure

def get_accuracy_and_metadata(df, model, name):
    row = {}
    pattern = r'.*rho_(\d+\.\d+)_nu_(\d+\.\d+)_dt_(\d+\.\d+)_mse_(\d+\.\d+)_div_(\d+\.\d+)_mom_(\d+\.\d+)_bc_(\d+\.\d+)_spec_(\d+\.\d+)'
    m = re.match(pattern, name)
    if (m):
        rho = m.group(1)
        nu = m.group(2)
        dt = m.group(3)
        mse = m.group(4)
        div = m.group(5)
        mom = m.group(6)
        bc = m.group(7)
        spec = m.group(8)
    else:
        rho = 1
        nu = 0.001
        dt = 1
        mse = 1
        div = 0
        mom = 0
        bc = 0
        spec = 0

    params = model.num_params

    # GET TRAIN SET ACCRUACY
    sample = len(ns_circle)//2 # Idx of the sample/simulation from the dataset
    n_out = 99 # Number of prediction steps
    graph = ns_circle.get_sequence(sample, n_out=n_out)

    # Inference with each model
    pred = model.solve(graph, n_out=n_out)

    ns_circle_r2 = gfd.metrics.r2(pred, graph.target)

    print(f"Model: {name}, Dataset: train, Accuracy {ns_circle_r2}")

    row = {}
    row['model_name'] = name
    row['rho'] = rho
    row['nu'] = nu
    row['dt'] = dt
    row['mse'] = mse
    row['div'] = div
    row['mom'] = mom
    row['bc'] = bc
    row['spec'] = spec
    row['dataset'] = 'ns_circle'
    row['accuracy'] = ns_circle_r2
    row['params'] = params

    df.loc[len(df)] = row

    # GET TEST MID RE ACCURACY
    sample = len(ns_circle_mid_re)//2 # Idx of the sample/simulation from the dataset
    n_out = 99 # Number of prediction steps
    graph = ns_circle_mid_re.get_sequence(sample, n_out=n_out)

    # Inference with each model
    pred = model.solve(graph, n_out=n_out)

    ns_circle_mid_re_r2 = gfd.metrics.r2(pred, graph.target)

    print(f"Model: {name}, Dataset: mid, Accuracy {ns_circle_mid_re_r2}")

    row = {}
    row['model_name'] = name
    row['rho'] = rho
    row['nu'] = nu
    row['dt'] = dt
    row['mse'] = mse
    row['div'] = div
    row['mom'] = mom
    row['bc'] = bc
    row['spec'] = spec
    row['dataset'] = 'ns_circle_mid_re'
    row['accuracy'] = ns_circle_mid_re_r2
    row['params'] = params

    df.loc[len(df)] = row

    # GET TEST LOW RE ACCURACY
    sample = len(ns_circle_low_re)//2 # Idx of the sample/simulation from the dataset
    n_out = 99 # Number of prediction steps
    graph = ns_circle_low_re.get_sequence(sample, n_out=n_out)

    # Inference with each model
    pred = model.solve(graph, n_out=n_out)

    ns_circle_low_re_r2 = gfd.metrics.r2(pred, graph.target)

    print(f"Model: {name}, Dataset: low, Accuracy {ns_circle_low_re_r2}")

    row = {}
    row['model_name'] = name
    row['rho'] = rho
    row['nu'] = nu
    row['dt'] = dt
    row['mse'] = mse
    row['div'] = div
    row['mom'] = mom
    row['bc'] = bc
    row['spec'] = spec
    row['dataset'] = 'ns_circle_low_re'
    row['accuracy'] = ns_circle_low_re_r2
    row['params'] = params

    df.loc[len(df)] = row

    # GET TEST HIGH RE ACCURACY
    sample = len(ns_circle_high_re)//2 # Idx of the sample/simulation from the dataset
    n_out = 99 # Number of prediction steps
    graph = ns_circle_high_re.get_sequence(sample, n_out=n_out)

    # Inference with each model
    pred = model.solve(graph, n_out=n_out)

    ns_circle_high_re_r2 = gfd.metrics.r2(pred, graph.target)
    print(f"Model: {name}, Dataset: high, Accuracy {ns_circle_high_re_r2}")

    row = {}
    row['model_name'] = name
    row['rho'] = rho
    row['nu'] = nu
    row['dt'] = dt
    row['mse'] = mse
    row['div'] = div
    row['mom'] = mom
    row['bc'] = bc
    row['spec'] = spec
    row['dataset'] = 'ns_circle_high_re'
    row['accuracy'] = ns_circle_high_re_r2
    row['params'] = params

    df.loc[len(df)] = row
    print(f"Model: {name}, Dataset: NsCircle,  Accuracy: {ns_circle_r2}")

    return df

# get baseline accuracy
model1 = gfd.nn.NsOneScaleGNN  (model="1S-GNN-NsCircle-v1", device=device)
model2 = gfd.nn.NsTwoScaleGNN  (model="2S-GNN-NsCircle-v1", device=device)
model3 = gfd.nn.NsThreeScaleGNN(model="3S-GNN-NsCircle-v1", device=device)
model4 = gfd.nn.NsFourScaleGNN (model="4S-GNN-NsCircle-v1", device=device)

full_df = get_accuracy_and_metadata(full_df, model1, 'NsOneScaleGNN')
full_df = get_accuracy_and_metadata(full_df, model2, 'NsTwoScaleGNN')
full_df = get_accuracy_and_metadata(full_df, model3, 'NsThreeScaleGNN')
full_df = get_accuracy_and_metadata(full_df, model4, 'NsFourScaleGNN')

# Physics losses config (div + mom + spec)
model1 = gfd.nn.NsOneScaleGNN(checkpoint='/home/nrehman/physics_losses_cfd/examples/training/NsMuSGNN/NsOneScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.05.chk', device=device)
model2 = gfd.nn.NsTwoScaleGNN(checkpoint='/home/nrehman/physics_losses_cfd/examples/training/NsMuSGNN/NsTwoScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.05.chk', device=device)
model3 = gfd.nn.NsThreeScaleGNN(checkpoint='/home/nrehman/physics_losses_cfd/examples/training/NsMuSGNN/NsThreeScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.05.chk', device=device)
model4 = gfd.nn.NsFourScaleGNN(checkpoint='/home/nrehman/physics_losses_cfd/examples/training/NsMuSGNN/NsFourScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.05.chk', device=device)

full_df = get_accuracy_and_metadata(full_df, model1, 'NsOneScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.05')
full_df = get_accuracy_and_metadata(full_df, model2, 'NsTwoScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.05')
full_df = get_accuracy_and_metadata(full_df, model3, 'NsThreeScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.05')
full_df = get_accuracy_and_metadata(full_df, model4, 'NsFourScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.05')

# Physics losses config 2 (just div)
model1 = gfd.nn.NsOneScaleGNN(checkpoint='/home/nrehman/physics_losses_cfd/examples/training/NsMuSGNN/NsOneScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.0_bc_0.0_spec_0.00.chk', device=device)
model2 = gfd.nn.NsTwoScaleGNN(checkpoint='/home/nrehman/physics_losses_cfd/examples/training/NsMuSGNN/NsTwoScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.0_bc_0.0_spec_0.00.chk', device=device)
model3 = gfd.nn.NsThreeScaleGNN(checkpoint='/home/nrehman/physics_losses_cfd/examples/training/NsMuSGNN/NsThreeScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.0_bc_0.0_spec_0.00.chk', device=device)
model4 = gfd.nn.NsFourScaleGNN(checkpoint='/home/nrehman/physics_losses_cfd/examples/training/NsMuSGNN/NsFourScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.0_bc_0.0_spec_0.00.chk', device=device)

full_df = get_accuracy_and_metadata(full_df, model1, 'NsOneScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.0_bc_0.0_spec_0.00')
full_df = get_accuracy_and_metadata(full_df, model2, 'NsTwoScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.0_bc_0.0_spec_0.00')
full_df = get_accuracy_and_metadata(full_df, model3, 'NsThreeScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.0_bc_0.0_spec_0.00')
full_df = get_accuracy_and_metadata(full_df, model4, 'NsFourScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.0_bc_0.0_spec_0.00')

# Physics losses config 3 (div + mom)
model1 = gfd.nn.NsOneScaleGNN(checkpoint='/home/nrehman/physics_losses_cfd/examples/training/NsMuSGNN/NsOneScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.00.chk', device=device)
model2 = gfd.nn.NsTwoScaleGNN(checkpoint='/home/nrehman/physics_losses_cfd/examples/training/NsMuSGNN/NsTwoScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.00.chk', device=device)
model3 = gfd.nn.NsThreeScaleGNN(checkpoint='/home/nrehman/physics_losses_cfd/examples/training/NsMuSGNN/NsThreeScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.00.chk', device=device)
model4 = gfd.nn.NsFourScaleGNN(checkpoint='/home/nrehman/physics_losses_cfd/examples/training/NsMuSGNN/NsFourScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.00.chk', device=device)

full_df = get_accuracy_and_metadata(full_df, model1, 'NsOneScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.00')
full_df = get_accuracy_and_metadata(full_df, model2, 'NsTwoScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.00')
full_df = get_accuracy_and_metadata(full_df, model3, 'NsThreeScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.00')
full_df = get_accuracy_and_metadata(full_df, model4, 'NsFourScaleGNNphysics_rho_1.0_nu_0.001_dt_1.0_mse_1.0_div_1.0_mom_0.1_bc_0.0_spec_0.00')

print(full_df)

full_df.to_csv("physics_informed_accuracy.csv", index=False)
