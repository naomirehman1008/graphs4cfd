# make training and visualization loss curves from log files that are formatted as follows:
# [PHYS] [ep2/it0] div=7.831e-02 (λ=1) | mom=1.817e+00 (λ=0.1) | bc=0.000e+00 (λ=0) | spec=5.878e-02 (λ=0.05) | TOTAL=2.629e-01
# [PHYS] [ep2/it100] div=6.059e-02 (λ=1) | mom=1.025e+00 (λ=0.1) | bc=0.000e+00 (λ=0) | spec=6.651e-02 (λ=0.05) | TOTAL=1.664e-01
# [PHYS] [ep2/it200] div=8.743e-02 (λ=1) | mom=1.960e+00 (λ=0.1) | bc=0.000e+00 (λ=0) | spec=1.249e-01 (λ=0.05) | TOTAL=2.896e-01
# [PHYS] [ep2/it300] div=6.071e-02 (λ=1) | mom=7.182e-01 (λ=0.1) | bc=0.000e+00 (λ=0) | spec=1.312e-01 (λ=0.05) | TOTAL=1.391e-01
# [PHYS] [ep2/it400] div=9.595e-02 (λ=1) | mom=2.264e+00 (λ=0.1) | bc=0.000e+00 (λ=0) | spec=3.673e-02 (λ=0.05) | TOTAL=3.242e-01
# [PHYS] [ep2/it500] div=1.255e-01 (λ=1) | mom=1.806e+00 (λ=0.1) | bc=0.000e+00 (λ=0) | spec=3.992e-02 (λ=0.05) | TOTAL=3.081e-01
# [PHYS] [ep2/it600] div=9.495e-02 (λ=1) | mom=1.681e+00 (λ=0.1) | bc=0.000e+00 (λ=0) | spec=5.857e-02 (λ=0.05) | TOTAL=2.659e-01
# [PHYS] [ep2/it700] div=9.090e-02 (λ=1) | mom=1.482e+00 (λ=0.1) | bc=0.000e+00 (λ=0) | spec=4.515e-02 (λ=0.05) | TOTAL=2.413e-01
# [PHYS] [ep2/it800] div=5.136e-02 (λ=1) | mom=1.270e+00 (λ=0.1) | bc=0.000e+00 (λ=0) | spec=6.898e-02 (λ=0.05) | TOTAL=1.818e-01
# [PHYS] [ep2/it900] div=9.787e-02 (λ=1) | mom=1.062e+00 (λ=0.1) | bc=0.000e+00 (λ=0) | spec=4.491e-02 (λ=0.05) | TOTAL=2.063e-01
# === EPOCH 2 ===
# 0.4674235448493796
# Epoch 2, Validation Loss: 0.4674235448493796
# Data Losses: 0.18170891604307343
# Physics Losses: 0.3264097307958656

# imports
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os
import argparse
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# import seaborn as sns

def iter_event_files(logdir):
    logdir = os.path.abspath(logdir)
    for root, _, files in os.walk(logdir):
        for f in files:
            if "tfevents" in f:
                yield root, os.path.join(root, f)


# https://stackoverflow.com/questions/70897472/tensorboard-accessing-tensor-objects-in-the-tags-of-event-accumulator
def extract_scalars(logdir):
    for run_dir, event_file in iter_event_files(logdir):
        run = os.path.relpath(run_dir, logdir)
        filename = os.path.basename(event_file)

        try:
            ea = EventAccumulator(event_file)
            ea.Reload()

            for tag in ea.Tags().get("scalars", []):
                for ev in ea.Scalars(tag):
                    yield {
                        "run": run,
                        "file": filename,
                        "tag": tag,
                        "step": ev.step,
                        "value": ev.value,
                        "wall_time": ev.wall_time,
                    }

        except Exception as e:
            print(f"Warning: failed reading {event_file}: {e}")

def get_losses(target_dir):
    basename = os.path.basename(target_dir)

    pattern = r'.*rho_(\d+\.\d+)_nu_(\d+\.\d+)_dt_(\d+\.\d+)_mse_(\d+\.\d+)_div_(\d+\.\d+)_mom_(\d+\.\d+)_bc_(\d+\.\d+)_spec_(\d+\.\d+)'
    m = re.match(pattern, basename)
    if (m):
        rho = m.group(1)
        nu = m.group(2)
        dt = m.group(3)
        mse = m.group(4)
        div = m.group(5)
        mom = m.group(6)
        bc = m.group(7)
        spec = m.group(8)
    rows = list(extract_scalars(target_dir))
    df = pd.DataFrame(columns=["run", "file", "tag", "step", "value", "wall_time"])
    for row in rows:
        df.loc[len(df)] = row
    
    print(df)

    formatted_df = pd.DataFrame()

    formatted_df['Loss/train'] = df[df['tag'] == 'Loss/train']['value'].values
    formatted_df['Loss/physics/train/mse'] = df[df['tag'] == 'Loss/physics/train/mse']['value'].values
    formatted_df['Loss/physics/train/div'] = df[df['tag'] == 'Loss/physics/train/div']['value'].values
    formatted_df['Loss/physics/train/mom'] = df[df['tag'] == 'Loss/physics/train/mom']['value'].values
    formatted_df['Loss/physics/train/bc'] = df[df['tag'] == 'Loss/physics/train/bc']['value'].values
    formatted_df['Loss/physics/train/spec'] = df[df['tag'] == 'Loss/physics/train/spec']['value'].values

    formatted_df['Loss/test'] = df[df['tag'] == 'Loss/train']['value'].values
    formatted_df['Loss/physics/test/mse'] = df[df['tag'] == 'Loss/physics/train/mse']['value'].values
    formatted_df['Loss/physics/test/div'] = df[df['tag'] == 'Loss/physics/train/div']['value'].values
    formatted_df['Loss/physics/test/mom'] = df[df['tag'] == 'Loss/physics/train/mom']['value'].values
    formatted_df['Loss/physics/test/bc'] = df[df['tag'] == 'Loss/physics/train/bc']['value'].values
    formatted_df['Loss/physics/test/spec'] = df[df['tag'] == 'Loss/physics/train/spec']['value'].values

    formatted_df['epoch'] = df[df['tag'] == 'Loss/physics/train/spec']['step'].values
    formatted_df['mse'] = [mse] * len(formatted_df)
    formatted_df['div'] = [div] * len(formatted_df)
    formatted_df['mom'] = [mom] * len(formatted_df)
    formatted_df['bc'] = [bc] * len(formatted_df)
    formatted_df['spec'] = [spec] * len(formatted_df)

    # fix n_out problem
    losses = list(formatted_df['Loss/test'])
    n_outs = [0] * len(losses)
    n_out = 1
    max_n_out = 10
    for idx, loss in enumerate(losses):
        n_outs[idx] = n_out
        if loss < 0.005 and n_out < max_n_out:
            n_out += 1

    formatted_df['n_out'] = n_outs
    print(f"N outs: {n_outs}")
    formatted_df['Loss/physics/train/mse'] = formatted_df['Loss/physics/train/mse'] / formatted_df['n_out']
    formatted_df['Loss/physics/train/div'] =  formatted_df['Loss/physics/train/div'] / formatted_df['n_out']
    formatted_df['Loss/physics/train/mom'] =  formatted_df['Loss/physics/train/mom'] / formatted_df['n_out']
    formatted_df['Loss/physics/train/bc'] =  formatted_df['Loss/physics/train/bc'] / formatted_df['n_out']
    formatted_df['Loss/physics/train/spec'] =  formatted_df['Loss/physics/train/spec'] / formatted_df['n_out']
        

    print(formatted_df)

    return {
        'epochs': list(formatted_df['epoch']),
        'iters': list(formatted_df['epoch']),
        'val_losses': list(formatted_df['Loss/test']),
        'data_losses': list(formatted_df['Loss/physics/train/mse']),
        'physics_losses': list(formatted_df['Loss/physics/train/div'] + formatted_df['Loss/physics/train/mom'] + formatted_df['Loss/physics/train/bc'] + formatted_df['Loss/physics/train/spec']),
        'div_losses': list(formatted_df['Loss/physics/train/div']),
        'mom_losses': list(formatted_df['Loss/physics/train/mom']),
        'bc_losses': list(formatted_df['Loss/physics/train/bc']),
        'spec_losses': list(formatted_df['Loss/physics/train/spec']),
        'total_phys_losses':  list(formatted_df['Loss/physics/train/div'] + formatted_df['Loss/physics/train/mom'] + formatted_df['Loss/physics/train/bc'] + formatted_df['Loss/physics/train/spec']),
        'mse': list(formatted_df['mse']),
        'div': list(formatted_df['div']),
        'mom': list(formatted_df['mom']),
        'bc': list(formatted_df['bc']),
        'spec': list(formatted_df['spec']),
    }   


# new function to open another log file formatted in the same way but resuming training, so the epochs need to be added and the iters adjusted accordingly
def open_additional_log_file(log_file_path, log_data):
    # infer start epoch and start iter from the previous log_data
    start_epoch = log_data['epochs'][-1]
    start_iter = log_data['iters'][-1] 
    log_data = get_losses(log_file_path)
    # adjust epochs and iters
    log_data['epochs'] = [e + start_epoch for e in log_data['epochs']]
    log_data['iters'] = [i + start_iter for i in log_data['iters']]
    return log_data


def plot_losses(log_data, output_dir, log_file_path):
    epochs = log_data['epochs']
    iters = log_data['iters']
    val_losses = log_data['val_losses']
    data_losses = log_data['data_losses']
    physics_losses = log_data['physics_losses']
    div_losses = log_data['div_losses']
    print(f"Div losses: {div_losses}")
    mom_losses = log_data['mom_losses']
    bc_losses = log_data['bc_losses']
    spec_losses = log_data['spec_losses']
    total_phys_losses = log_data['total_phys_losses']

    # Create output directory if it doesn't exist
    output_dir = output_dir + '/' + os.path.basename(log_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Plot validation loss
    plt.figure()
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'validation_loss.png'))
    plt.close()

    # Plot data and physics losses
    plt.figure()
    plt.plot(epochs, data_losses, label='Data Loss')
    plt.plot(epochs, physics_losses, label='Physics Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Data and Physics Losses over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'data_physics_loss.png'))
    plt.close()

    # Plot physics loss components
    plt.figure()
    lambda_div = float(log_data['div'][0])
    lambda_mom = float(log_data['mom'][0])
    lambda_bc = float(log_data['bc'][0])
    lambda_spec = float(log_data['spec'][0])
    if (lambda_div > 0): plt.plot(iters, div_losses, label='Divergence Loss')
    if (lambda_mom > 0): plt.plot(iters, mom_losses, label='Momentum Loss')
    if (lambda_bc > 0): plt.plot(iters, bc_losses, label='Boundary Condition Loss')
    if (lambda_spec > 0): plt.plot(iters, spec_losses, label='Spectral Loss')
    plt.plot(iters, total_phys_losses, label='Total Physics Loss', linestyle='--', color='black')
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.title('Physics Loss Components over Training Iterations')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'physics_loss_components.png'))
    plt.close()

    """
    # didn't collect loss per iteration :/
    # Plot physics loss components over epochs (averaged), without creating a new separate list for each averaged loss
    plt.figure()
    num_epochs = len(epochs)
    div_avg = [np.mean([div_losses[i] for i in range(len(iters)) if iters[i] // 1000 == epoch]) for epoch in epochs]
    mom_avg = [np.mean([mom_losses[i] for i in range(len(iters)) if iters[i] // 1000 == epoch]) for epoch in epochs]
    bc_avg = [np.mean([bc_losses[i] for i in range(len(iters)) if iters[i] // 1000 == epoch]) for epoch in epochs]
    spec_avg = [np.mean([spec_losses[i] for i in range(len(iters)) if iters[i] // 1000 == epoch]) for epoch in epochs]
    total_phys_avg = [np.mean([total_phys_losses[i] for i in range(len(iters)) if iters[i] // 1000 == epoch]) for epoch in epochs]
    plt.plot(epochs, div_avg, label='Divergence Loss')
    plt.plot(epochs, mom_avg, label='Momentum Loss')
    plt.plot(epochs, bc_avg, label='Boundary Condition Loss')
    plt.plot(epochs, spec_avg, label='Spectral Loss')
    plt.plot(epochs, total_phys_avg, label='Total Physics Loss', linestyle='--', color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Averaged Physics Loss Components over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'physics_loss_components_avg.png'))
    plt.close() 
    """

    # Plot physics loss components over epochs, adjusted with lambda values 
    plt.figure()
    lambda_div = float(log_data['div'][0])
    lambda_mom = float(log_data['mom'][0])
    lambda_bc = float(log_data['bc'][0])
    lambda_spec = float(log_data['spec'][0])
    div_avg_lambda = [lambda_div * loss for loss in div_losses]
    mom_avg_lambda = [lambda_mom * loss for loss in mom_losses]
    bc_avg_lambda = [lambda_bc * loss for loss in bc_losses]
    spec_avg_lambda = [lambda_spec * loss for loss in spec_losses]
    plt.plot(epochs, div_avg_lambda, label='Divergence Loss (λ=1)')
    plt.plot(epochs, mom_avg_lambda, label='Momentum Loss (λ=0.1)')
    plt.plot(epochs, bc_avg_lambda, label='Boundary Condition Loss (λ=0)')
    plt.plot(epochs, spec_avg_lambda, label='Spectral Loss (λ=0.05)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Averaged Physics Loss Components over Epochs (scaled with λ)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'physics_loss_components_avg_lambda.png'))
    plt.close()

        # Plot physics loss components over epochs, adjusted with lambda values 
    plt.figure()
    lambda_mse = float(1)
    lambda_div = float(log_data['div'][0])
    lambda_mom = float(log_data['mom'][0])
    lambda_bc = float(log_data['bc'][0])
    lambda_spec = float(log_data['spec'][0])
    mse_avg_lambda = [lambda_mse * loss for loss in data_losses]
    div_avg_lambda = [lambda_div * loss for loss in div_losses]
    mom_avg_lambda = [lambda_mom * loss for loss in mom_losses]
    bc_avg_lambda = [lambda_bc * loss for loss in bc_losses]
    spec_avg_lambda = [lambda_spec * loss for loss in spec_losses]
    if (lambda_div > 0): plt.plot(epochs, div_avg_lambda, label=f'Divergence Loss (λ={lambda_div})')
    if (lambda_mom > 0): plt.plot(epochs, mom_avg_lambda, label=f'Momentum Loss (λ={lambda_mom})')
    if (lambda_bc > 0): plt.plot(epochs, bc_avg_lambda, label=f'Boundary Condition Loss (λ={lambda_bc})')
    if (lambda_spec > 0): plt.plot(epochs, spec_avg_lambda, label=f'Spectral Loss (λ={lambda_spec})')
    if (lambda_mse > 0): plt.plot(epochs, mse_avg_lambda, label=f'MSE Loss (λ={lambda_mse})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Components over Epochs (scaled with λ)')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_dir, 'all)loss_components_lambda.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse log file and plot losses.')
    parser.add_argument('--log_file1', default=None, type=str, help='Path to the log file.')
    # make second file optional
    parser.add_argument('--log_file2', default=None, type=str, help='Path to the additional log file to resume from.')
    parser.add_argument('--output_dir', default=None, type=str, help='Directory to save the plots.')
    args = parser.parse_args()

    log_data = get_losses(args.log_file1)

    if args.log_file2 is not None:
        additional_log_data = open_additional_log_file(args.log_file2, log_data)
        # combine log_data and additional_log_data here before plotting
        for key in log_data.keys():
            log_data[key].extend(additional_log_data[key])

    plot_losses(log_data, args.output_dir, args.log_file2 if args.log_file2 is not None else args.log_file1)

    # call file like:
    # python make_vis.py /path/to/log_file.txt /path/to/output_dir

