import argparse
import math
import multiprocessing # Used for GIF generation
import os
import random # Used for saving/loading RNG state

import matplotlib as mpl
mpl.use('Agg') # Use Agg backend before pyplot import
import matplotlib.pyplot as plt # Used for loss/accuracy plots
import numpy as np
np.seterr(divide='ignore', invalid='warn') # Keep basic numpy settings
import seaborn as sns
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
import torchvision # For disabling warning
from tqdm.auto import tqdm # Used for progress bars

from autoclip.torch import QuantileClip # Used for gradient clipping
from data.custom_datasets import ParityDataset
from tasks.image_classification.plotting import plot_neural_dynamics
from models.utils import reshape_predictions, get_latest_checkpoint
from tasks.parity.plotting import make_parity_gif
from tasks.parity.utils import prepare_model, reshape_attention_weights, reshape_inputs
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import parity_loss
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

torchvision.disable_beta_transforms_warning()
torch.serialization.add_safe_globals([argparse.Namespace])


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTM on Parity Task")

    # Model Architecture 
    parser.add_argument('--model_type', type=str, default="ctm", choices=['ctm', 'lstm'], help='The type of model to train.')
    parser.add_argument('--parity_sequence_length', type=int, default=64, help='Sequence length for parity task.')
    parser.add_argument('--d_model', type=int, default=1024, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=512, help='Dimension of the input projection.')
    parser.add_argument('--synapse_depth', type=int, default=1, help='Depth of U-NET model for synapse. 1=linear.')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--n_synch_out', type=int, default=32, help='Number of neurons for output sync.')
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons for action sync.')
    parser.add_argument('--neuron_select_type', type=str, default='random', choices=['first-last', 'random', 'random-pairing'], help='Protocol for selecting neuron subset.')
    parser.add_argument('--n_random_pairing_self', type=int, default=256, help='Number of neurons paired self-to-self for synch.')
    parser.add_argument('--iterations', type=int, default=75, help='Number of internal ticks.')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of pre-activation history for NLMs.')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep NLMs.')
    parser.add_argument('--memory_hidden_dims', type=int, default=16, help='Hidden dimensions for deep NLMs.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs.')
    parser.add_argument('--positional_embedding_type', type=str, default='custom-rotational-1d', help='Type of positional embedding.') # Choices removed for simplicity if not strictly needed by argparse functionality here
    parser.add_argument('--backbone_type', type=str, default='parity_backbone', help='Type of backbone feature extractor.')

    # Training Configuration 
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--batch_size_test', type=int, default=256, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--training_iterations', type=int, default=50001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='multistep', choices=['multistep', 'cosine'], help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+', help='Scheduler milestones for multistep.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')
    parser.add_argument('--gradient_clipping', type=float, default=-1, help='Quantile clipping value (-1 to disable).')
    parser.add_argument('--use_most_certain_with_lstm', action=argparse.BooleanOptionalAction, default=False, help='Use most certain loss with LSTM baseline.')

    # Housekeeping 
    parser.add_argument('--log_dir', type=str, default='logs/parity', help='Directory for logging.')
    parser.add_argument('--dataset', type=str, default='parity', help='Dataset name (used for assertion).')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint frequency.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=True, help='Reload checkpoint from log_dir?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=True, help='Reload only model weights?')
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics frequency.')
    parser.add_argument('--n_test_batches', type=int, default=20, help='Num batches for metrics approx. (-1 for full).')
    parser.add_argument('--full_eval',  action=argparse.BooleanOptionalAction, default=False, help='Perform full evaluation instead of approx.')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='GPU(s) or -1 for CPU.')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')

    # Early stopping and best checkpoint (optional, disabled by default)
    parser.add_argument('--early_stopping_patience', type=int, default=-1, help='Stop training if test metric does not improve for this many validations. -1 to disable.')
    parser.add_argument('--save_best_checkpoint', action=argparse.BooleanOptionalAction, default=False, help='Save best model checkpoint separately as best_checkpoint.pt.')

    args = parser.parse_args()
    return args

if __name__=='__main__':

    args = parse_args()
    
    set_seed(args.seed)

    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    
    assert int(math.sqrt(args.parity_sequence_length)) ** 2 == args.parity_sequence_length, "parity_sequence_length must be a perfect square."

    train_data = ParityDataset(sequence_length=args.parity_sequence_length, length=100000)
    test_data = ParityDataset(sequence_length=args.parity_sequence_length, length=10000)


    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=0, drop_last=False)

    prediction_reshaper = [args.parity_sequence_length, 2]
    args.out_dims = args.parity_sequence_length * 2

    args.use_most_certain = args.model_type == "ctm" or (args.use_most_certain_with_lstm and args.model_type == "lstm")

    # For total reproducibility
    # Python 3.x
    zip_python_code(f'{args.log_dir}/repo_state.zip')
    with open(f'{args.log_dir}/args.txt', 'w') as f:
        print(args, file=f)  

    # Configure device string (support MPS on macOS)
    if args.device[0] != -1:
        device = f'cuda:{args.device[0]}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Running model {args.model} on {device}')

    # Build model
    model = prepare_model(prediction_reshaper, args, device)

    model.train()

    # For lazy modules so that we can get param count
    pseudo_inputs = train_data.__getitem__(0)[0].unsqueeze(0).to(device)
    model(pseudo_inputs)  

    print(f'Total params: {sum(p.numel() for p in model.parameters())}')
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.lr, 
                                  eps=1e-8, 
                                  weight_decay=args.weight_decay)
    if args.gradient_clipping!=-1:  # Not using, but handy to have
        optimizer = QuantileClip.as_optimizer(optimizer=optimizer, quantile=args.gradient_clipping, history_length=1000)
    
    warmup_schedule = warmup(args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule.step)
    if args.use_scheduler:
        if args.scheduler_type == 'multistep':
            scheduler = WarmupMultiStepLR(optimizer, warmup_steps=args.warmup_steps, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler_type == 'cosine':
            scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations, warmup_start_lr=1e-20, eta_min=1e-7)
        else:
            raise NotImplementedError
        
    
    # Metrics tracking (I like custom)
    # Using batched estimates
    start_iter = 0  # For reloading, keep track of this (pretty tqdm stuff needs it)
    train_losses = []  
    test_losses = []
    train_accuracies = []  # This will be per internal tick, not so simple
    test_accuracies = []
    train_accuracies_most_certain = []  # This will be selected according to what is returned by loss function
    test_accuracies_most_certain = []
    train_accuracies_most_certain_per_input = []
    test_accuracies_most_certain_per_input = []
    iters = []
    best_test_metric = -float('inf')
    early_stopping_counter = 0
    scaler = torch.amp.GradScaler("cuda" if "cuda" in device else "cpu", enabled=args.use_amp)

    # Now that everything is initliased, reload if desired
    if args.reload and (latest_checkpoint_path := get_latest_checkpoint(args.log_dir)):
        print(f'Reloading from: {latest_checkpoint_path}')
        checkpoint = torch.load(f'{latest_checkpoint_path}', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if not args.reload_model_only:
            print('Reloading optimizer etc.')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_iter = checkpoint['iteration']
            train_losses = checkpoint['train_losses']
            train_accuracies_most_certain = checkpoint['train_accuracies_most_certain']
            train_accuracies_most_certain_per_input = checkpoint['train_accuracies_most_certain_per_input'] if 'train_accuracies_most_certain_per_input' in checkpoint else train_accuracies_most_certain_per_input
            train_accuracies = checkpoint['train_accuracies']
            test_losses = checkpoint['test_losses']
            test_accuracies_most_certain = checkpoint['test_accuracies_most_certain']
            test_accuracies_most_certain_per_input = checkpoint['test_accuracies_most_certain_per_input'] if 'test_accuracies_most_certain_per_input' in checkpoint else test_accuracies_most_certain_per_input
            test_accuracies = checkpoint['test_accuracies']
            iters = checkpoint['iters']
            # Load early stopping state (backward compatible)
            best_test_metric = checkpoint.get('best_test_metric', -float('inf'))
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        else:
            print('Only relading model!')
        if 'torch_rng_state' in checkpoint:
            print("Reloading rng state")
            # Reset seeds, otherwise mid-way training can be obscure (particularly for imagenet)
            torch.set_rng_state(checkpoint['torch_rng_state'].cpu().byte())
            np.random.set_state(checkpoint['numpy_rng_state'])
            random.setstate(checkpoint['random_rng_state'])

        del checkpoint
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    
    # Training
    iterator = iter(trainloader)  # Not training in epochs, but rather iterations. Need to reset this from time to time
    with tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):
            current_lr = optimizer.param_groups[-1]['lr']

            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets = next(iterator)
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=args.use_amp):
                predictions, certainties, synchronisation = model(inputs)
                predictions = predictions.reshape(predictions.size(0), -1, 2, predictions.size(-1))
                loss, where_most_certain = parity_loss(predictions, certainties, targets, use_most_certain=args.use_most_certain)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            accuracy_finegrained = (predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device),:,where_most_certain] == targets).float().mean().item()
            pbar.set_description(f'Dataset=Parity. Loss={loss.item():0.3f}. Accuracy={accuracy_finegrained:0.3f}. LR={current_lr:0.6f}. Where_certain={where_most_certain.float().mean().item():0.2f}+-{where_most_certain.float().std().item():0.2f} ({where_most_certain.min().item():d}<->{where_most_certain.max().item():d})')

            # Metrics tracking and plotting
            if bi%args.track_every==0:# and bi != 0:
                model.eval()
                with torch.inference_mode():

                    inputs, targets = next(iter(testloader))
                    inputs = inputs.to(device)
                    targets = targets.to(device)                                 
                    predictions, certainties, synchronisation, pre_activations, post_activations, attention = model(inputs, track=True)

                    predictions = reshape_predictions(predictions, prediction_reshaper)
                    attention = reshape_attention_weights(attention)
                    inputs = reshape_inputs(inputs, args.iterations, grid_size=int(math.sqrt(args.parity_sequence_length)))

                    pbar.set_description('Tracking: Neural dynamics')
                    plot_neural_dynamics(post_activations, 100, args.log_dir, axis_snap=True)

                    pbar.set_description('Tracking: Producing attention gif')

                    process = multiprocessing.Process(
                        target=make_parity_gif,
                        args=(
                        predictions.detach().cpu().numpy(),
                        certainties.detach().cpu().numpy(),
                        targets.detach().cpu().numpy(),
                        pre_activations,
                        post_activations,
                        attention,
                        inputs,
                        f"{args.log_dir}/eval_output_val_{0}_iter_{0}.gif",
                    ))
                    process.start()
                    
                    ##################################### TRAIN METRICS
                    all_predictions = []
                    all_targets = []
                    all_predictions_most_certain = []
                    all_losses = []
                    
                    iters.append(bi)
                    with torch.inference_mode():
                        loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_test, shuffle=True, num_workers=0)
                        with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                        
                            for inferi, (inputs, targets) in enumerate(loader):
                                
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                these_predictions, certainties, synchronisation = model(inputs)

                                these_predictions = reshape_predictions(these_predictions, prediction_reshaper)
                                loss, where_most_certain = parity_loss(these_predictions, certainties, targets, use_most_certain=args.use_most_certain)
                                all_losses.append(loss.item())

                                all_targets.append(targets.detach().cpu().numpy())

                                all_predictions_most_certain.append(these_predictions.argmax(2)[torch.arange(these_predictions.size(0), device=these_predictions.device), :, where_most_certain].detach().cpu().numpy())
                                all_predictions.append(these_predictions.argmax(2).detach().cpu().numpy())
                                
                                if inferi%args.n_test_batches==0 and inferi!=0 and not args.full_eval: break
                                pbar_inner.set_description('Computing metrics for train')
                                pbar_inner.update(1)

                        all_predictions = np.concatenate(all_predictions)
                        all_targets = np.concatenate(all_targets)
                        all_predictions_most_certain = np.concatenate(all_predictions_most_certain)


                        train_accuracies.append(np.mean(all_predictions == all_targets[...,np.newaxis], axis=tuple(range(all_predictions.ndim-1))))
                        train_accuracies_most_certain.append((all_targets == all_predictions_most_certain).mean())
                        train_accuracies_most_certain_per_input.append((all_targets == all_predictions_most_certain).reshape(all_targets.shape[0], -1).all(-1).mean())
                        train_losses.append(np.mean(all_losses))

                        ##################################### TEST METRICS
                        all_predictions = []
                        all_predictions_most_certain = []
                        all_targets = []
                        all_losses = []
                        loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=0)
                        with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                            for inferi, (inputs, targets) in enumerate(loader):
                                
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                these_predictions, certainties, synchronisation = model(inputs)

                                these_predictions = these_predictions.reshape(these_predictions.size(0), -1, 2, these_predictions.size(-1))
                                loss, where_most_certain = parity_loss(these_predictions, certainties, targets, use_most_certain=args.use_most_certain)
                                all_losses.append(loss.item())

                                all_targets.append(targets.detach().cpu().numpy())

                                all_predictions_most_certain.append(these_predictions.argmax(2)[torch.arange(these_predictions.size(0), device=these_predictions.device), :, where_most_certain].detach().cpu().numpy())
                                all_predictions.append(these_predictions.argmax(2).detach().cpu().numpy())
                                
                                if inferi%args.n_test_batches==0 and inferi!=0 and not args.full_eval: break
                                pbar_inner.set_description('Computing metrics for test')
                                pbar_inner.update(1)

                        all_predictions = np.concatenate(all_predictions)
                        all_targets = np.concatenate(all_targets)
                        all_predictions_most_certain = np.concatenate(all_predictions_most_certain)
                        
                        test_accuracies.append(np.mean(all_predictions == all_targets[...,np.newaxis], axis=tuple(range(all_predictions.ndim-1))))
                        test_accuracies_most_certain.append((all_targets == all_predictions_most_certain).mean())
                        test_accuracies_most_certain_per_input.append((all_targets == all_predictions_most_certain).reshape(all_targets.shape[0], -1).all(-1).mean())
                        test_losses.append(np.mean(all_losses))

                        # Best model checkpoint and early stopping
                        current_metric = test_accuracies_most_certain[-1]
                        if current_metric > best_test_metric:
                            best_test_metric = current_metric
                            early_stopping_counter = 0
                            if args.save_best_checkpoint:
                                torch.save({'model_state_dict': model.state_dict(),
                                             'iteration': bi, 'best_test_metric': best_test_metric, 'args': args},
                                            f'{args.log_dir}/best_checkpoint.pt')
                                print(f'New best model saved at iteration {bi} with metric {best_test_metric:.4f}')
                        else:
                            early_stopping_counter += 1

                        figacc = plt.figure(figsize=(10, 10))
                        axacc_train = figacc.add_subplot(211)
                        axacc_test = figacc.add_subplot(212)
                        cm = sns.color_palette("viridis", as_cmap=True)
                        if args.dataset != 'sort':
                            for ti, (train_acc, test_acc) in enumerate(zip(np.array(train_accuracies).T, np.array(test_accuracies).T)):
                                axacc_train.plot(iters, train_acc, color=cm((ti)/(train_accuracies[0].shape[-1])), alpha=0.3)
                                axacc_test.plot(iters, test_acc, color=cm((ti)/(test_accuracies[0].shape[-1])), alpha=0.3)
                        axacc_train.plot(iters, train_accuracies_most_certain, 'k--', alpha=0.7, label='Most certain')
                        axacc_train.plot(iters, train_accuracies_most_certain_per_input, 'r', alpha=0.6, label='Full Input')
                        axacc_test.plot(iters, test_accuracies_most_certain, 'k--', alpha=0.7, label='Most certain')
                        axacc_test.plot(iters, test_accuracies_most_certain_per_input, 'r', alpha=0.6, label='Full Input')
                        axacc_train.set_title('Train')
                        axacc_test.set_title('Test')
                        axacc_train.legend(loc='lower right')
                        axacc_train.set_xlim([0, args.training_iterations])
                        axacc_test.set_xlim([0, args.training_iterations])

                        figacc.tight_layout()
                        figacc.savefig(f'{args.log_dir}/accuracies.png', dpi=150)
                        plt.close(figacc)

                        figloss = plt.figure(figsize=(10, 5))
                        axloss = figloss.add_subplot(111)
                        axloss.plot(iters, train_losses, 'b-', linewidth=1, alpha=0.8, label=f'Train: {train_losses[-1]}')
                        axloss.plot(iters, test_losses, 'r-', linewidth=1, alpha=0.8, label=f'Test: {test_losses[-1]}')
                        axloss.legend(loc='upper right')

                        axloss.set_xlim([0, args.training_iterations])
                        figloss.tight_layout()
                        figloss.savefig(f'{args.log_dir}/losses.png', dpi=150)
                        plt.close(figloss)

                model.train()

                if args.early_stopping_patience != -1 and early_stopping_counter >= args.early_stopping_patience:
                    print(f'Early stopping triggered at iteration {bi}. '
                          f'No improvement for {args.early_stopping_patience} validations.')
                    break

            # Save model
            if (bi%args.save_every==0 or bi==args.training_iterations-1):
                torch.save(
                    {
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'scaler_state_dict':scaler.state_dict(),
                    'iteration':bi,
                    'train_accuracies_most_certain':train_accuracies_most_certain,
                    'train_accuracies_most_certain_per_input':train_accuracies_most_certain_per_input,
                    'train_accuracies':train_accuracies,
                    'test_accuracies_most_certain':test_accuracies_most_certain,
                    'test_accuracies_most_certain_per_input':test_accuracies_most_certain_per_input,
                    'test_accuracies':test_accuracies,
                    'train_losses':train_losses,
                    'test_losses':test_losses,
                    'iters':iters,
                    'args':args,
                    'best_test_metric': best_test_metric,
                    'early_stopping_counter': early_stopping_counter,
                    'torch_rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    } , f'{args.log_dir}/checkpoint_{bi}.pt')
            
            pbar.update(1)


