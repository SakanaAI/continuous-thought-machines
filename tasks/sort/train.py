import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    # For faster
    torch.set_float32_matmul_precision('high')   
from tqdm.auto import tqdm

from data.custom_datasets import SortDataset
from models.ctm_sort import ContinuousThoughtMachineSORT
from tasks.image_classification.plotting import plot_neural_dynamics, make_classification_gif
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import sort_loss
from tasks.sort.utils import compute_ctc_accuracy, decode_predictions
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

import torchvision
torchvision.disable_beta_transforms_warning()

from autoclip.torch import QuantileClip

import warnings
warnings.filterwarnings("ignore", message="using precomputed metric; inverse_transform will be unavailable")


warnings.filterwarnings(
    "ignore",
    "Corrupt EXIF data",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)

warnings.filterwarnings(
    "ignore",
    "UserWarning: Metadata Warning",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)


warnings.filterwarnings(
    "ignore",
    "UserWarning: Truncated File Read",
    UserWarning,
    r"^PIL\.TiffImagePlugin$" # Using a regular expression to match the module.
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Architecture
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=128, help='Dimension of the input.')
    parser.add_argument('--synapse_depth', type=int, default=4, help='Depth of U-NET model for synapse. 1=linear, no unet.')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--n_synch_out', type=int, default=32, help='Number of neurons to use for output synch.')
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons to use for observation/action synch.')
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing', help='Protocol for selecting neuron subset.')
    parser.add_argument('--n_random_pairing_self', type=int, default=0, help='Number of neurons paired self-to-self for synch.')
    
    parser.add_argument('--iterations', type=int, default=50, help='Number of internal ticks.')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of the pre-activation history for NLMS.')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Use deep memory.')
    parser.add_argument('--memory_hidden_dims', type=int, default=4, help='Hidden dimensions of the memory if using deep memory.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--dropout_nlm', type=float, default=None, help='Dropout rate for NLMs specifically. Unset to match dropout on the rest of the model.')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply normalization in NLMs.')
    parser.add_argument('--positional_embedding_type', type=str, default='none',
                        help='Type of positional embedding.', choices=['none', 
                                                                       'learnable-fourier', 
                                                                       'multi-learnable-fourier',
                                                                       'custom-rotational'])

    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--batch_size_test', type=int, default=32, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the model.')
    parser.add_argument('--training_iterations', type=int, default=100001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True,
                        help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'],
                        help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+',
                        help='Learning rate scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[], help='List to exclude from weight decay. Typically good: bn, ln, bias, start')
    parser.add_argument('--gradient_clipping', type=float, default=-1, help='Gradient quantile clipping value (-1 to disable).')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False, help='Try to compile the synapses, backbone, and nlms.')


    # Logging and Saving
    parser.add_argument('--log_dir', type=str, default='logs/scratch',
                        help='Directory for logging.')
    parser.add_argument('--N_to_sort', type=int, default=30, help='N numbers to sort.')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoints every this many iterations.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False, help='Reload from disk?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False,
                        help='Reload only the model from disk?')

    # Tracking
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics every this many iterations.')
    parser.add_argument('--n_test_batches', type=int, default=2, help='How many minibatches to approx metrics. Set to -1 for full eval')

    # Device
    parser.add_argument('--device', type=int, nargs='+', default=[-1],
                        help='List of GPU(s) to use. Set to -1 to use CPU.')

    # Early stopping and best checkpoint (optional, disabled by default)
    parser.add_argument('--early_stopping_patience', type=int, default=-1, help='Stop training if test metric does not improve for this many validations. -1 to disable.')
    parser.add_argument('--save_best_checkpoint', action=argparse.BooleanOptionalAction, default=False, help='Save best model checkpoint separately as best_checkpoint.pt.')

    args = parser.parse_args()
    return args




if __name__=='__main__':

    # Hosuekeeping
    args = parse_args()
    # Change the following for sorting
    args.backbone_type = 'none'
    
    set_seed(args.seed, False)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    
    

    

    # Data
    train_data = SortDataset(args.N_to_sort)
    test_data = SortDataset(args.N_to_sort)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=1, drop_last=False)
    

    prediction_reshaper = [-1]  # Problem specific
    args.out_dims = args.N_to_sort + 1

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
    model = ContinuousThoughtMachineSORT(
        iterations=args.iterations,
        d_model=args.d_model,
        d_input=args.out_dims-1,  
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,  
        deep_nlms=args.deep_memory,
        memory_hidden_dims=args.memory_hidden_dims,  
        do_layernorm_nlm=args.do_normalisation,  
        backbone_type='none',
        positional_embedding_type=args.positional_embedding_type,
        out_dims=args.out_dims,
        prediction_reshaper=prediction_reshaper,
        dropout=args.dropout,      
        dropout_nlm=args.dropout_nlm,    
        neuron_select_type=args.neuron_select_type,
        n_random_pairing_self=args.n_random_pairing_self,
    ).to(device)

    
    model.train()

    # For lazy modules so that we can get param count
    pseudo_inputs = train_data.__getitem__(0)[0].unsqueeze(0).to(device)
    model(pseudo_inputs)  

    print(f'Total params: {sum(p.numel() for p in model.parameters())}')
    
    

    # Optimizer and scheduler
    decay_params = []
    no_decay_params = []
    no_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # Skip parameters that don't require gradients
        if any(exclusion_str in name for exclusion_str in args.weight_decay_exclusion_list):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
    if len(no_decay_names):
        print(f'WARNING, excluding: {no_decay_names}')

    # Optimizer and scheduler (Common setup)
    if len(no_decay_names) and args.weight_decay!=0:
        optimizer = torch.optim.AdamW([{'params': decay_params, 'weight_decay':args.weight_decay},
                                       {'params': no_decay_params, 'weight_decay':0}],
                                  lr=args.lr,
                                  eps=1e-8 if not args.use_amp else 1e-6)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr,
                                    eps=1e-8 if not args.use_amp else 1e-6,
                                    weight_decay=args.weight_decay)
    
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
    train_accuracies_full_list = []  # This will be selected according to what is returned by loss function
    test_accuracies_full_list = []
    iters = []
    best_test_metric = -float('inf')
    early_stopping_counter = 0

    # Now that everything is initliased, reload if desired
    scaler = torch.amp.GradScaler("cuda" if "cuda" in device else "cpu", enabled=args.use_amp)
    if args.reload:
        if os.path.isfile(f'{args.log_dir}/checkpoint.pt'):
            print(f'Reloading from: {args.log_dir}/checkpoint.pt')
            checkpoint = torch.load(f'{args.log_dir}/checkpoint.pt', map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            if not args.reload_model_only:
                print('Reloading optimizer etc.')
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_iter = checkpoint['iteration']
                train_losses = checkpoint['train_losses']
                train_accuracies_full_list = checkpoint['train_accuracies_full_list']
                train_accuracies = checkpoint['train_accuracies']
                test_losses = checkpoint['test_losses']
                test_accuracies_full_list = checkpoint['test_accuracies_full_list']
                test_accuracies = checkpoint['test_accuracies']
                iters = checkpoint['iters']
                # Load early stopping state (backward compatible)
                best_test_metric = checkpoint.get('best_test_metric', -float('inf'))
                early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
            else:
                print('Only relading model!')
            if 'torch_rng_state' in checkpoint:
                # Reset seeds, otherwise mid-way training can be obscure (particularly for imagenet)
                torch.set_rng_state(checkpoint['torch_rng_state'].cpu().byte())
                np.random.set_state(checkpoint['numpy_rng_state'])
                random.setstate(checkpoint['random_rng_state'])

            del checkpoint
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if args.do_compile:
        print('Compiling...')
        model.synapses = torch.compile(model.synapses, mode='reduce-overhead', fullgraph=True)
        model.backbone = torch.compile(model.backbone, mode='reduce-overhead', fullgraph=True)
    
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
                if args.do_compile:
                    torch.compiler.cudagraph_mark_step_begin()
                predictions, certainties, synchronisation = model(inputs)
                loss = sort_loss(predictions, targets)
            
                        
            scaler.scale(loss).backward()
        

            if args.gradient_clipping!=-1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)
            

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            accuracy = compute_ctc_accuracy(predictions, targets, predictions.shape[1]-1)
            pbar.set_description(f'Sorting {args.N_to_sort} real numbers. Loss={loss.item():0.3f}. Accuracy={accuracy:0.3f}. LR={current_lr:0.6f}')


            # Metrics tracking and plotting
            if bi%args.track_every==0:# and bi != 0:
                model.eval()
                with torch.inference_mode():
                    

                    inputs, targets = next(iter(testloader))
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    pbar.set_description('Tracking: Processing test data')
                    predictions, certainties, synchronisation, pre_activations, post_activations, _ = model(inputs, track=True)
                    pbar.set_description('Tracking: Neural dynamics')
                    plot_neural_dynamics(post_activations, 100, args.log_dir)

                    imgi = 0


                    
                    ##################################### TRAIN METRICS
                    all_predictions = []
                    all_targets = []
                    all_losses = []
                    
                    iters.append(bi)
                    pbar.set_description('Tracking: Computing loss and accuracy for curves')
                    with torch.inference_mode():
                        loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_test, shuffle=True, num_workers=1)
                        with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                        
                            for inferi, (inputs, targets) in enumerate(loader):
                                
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                these_predictions, certainties, synchronisation = model(inputs)

                                loss = sort_loss(these_predictions, targets)
                                all_losses.append(loss.item())

                                all_targets.append(targets.detach().cpu().numpy())

                                decoded = [d[:targets.shape[1]] for d in decode_predictions(these_predictions, predictions.shape[1]-1)]
                                decoded = torch.stack([torch.concatenate((d, torch.zeros(targets.shape[1] - len(d), device=targets.device)+targets.shape[1])) if len(d) < targets.shape[1] else d for d in decoded], 0)

                                all_predictions.append(decoded.detach().cpu().numpy())
                                
                                if args.n_test_batches!=-1 and inferi%args.n_test_batches==0 and inferi!=0 : break
                                pbar_inner.set_description('Computing metrics for train')
                                pbar_inner.update(1)

                        all_predictions = np.concatenate(all_predictions)
                        all_targets = np.concatenate(all_targets)


                        train_accuracies.append((all_predictions==all_targets).mean())
                        train_accuracies_full_list.append((all_predictions==all_targets).all(-1).mean())
                        train_losses.append(np.mean(all_losses))

                        ##################################### TEST METRICS
                        all_predictions = []
                        all_targets = []
                        all_losses = []
                        loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=1)
                        with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                            for inferi, (inputs, targets) in enumerate(loader):
                                
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                these_predictions, certainties, synchronisation = model(inputs)

                                loss = sort_loss(these_predictions, targets)
                                all_losses.append(loss.item())

                                all_targets.append(targets.detach().cpu().numpy())

                                decoded = [d[:targets.shape[1]] for d in decode_predictions(these_predictions, predictions.shape[1]-1)]
                                decoded = torch.stack([torch.concatenate((d, torch.zeros(targets.shape[1] - len(d), device=targets.device)+targets.shape[1])) if len(d) < targets.shape[1] else d for d in decoded], 0)

                                all_predictions.append(decoded.detach().cpu().numpy())
                                
                                if args.n_test_batches!=-1 and inferi%args.n_test_batches==0 and inferi!=0 : break
                                pbar_inner.set_description('Computing metrics for train')
                                pbar_inner.update(1)

                        all_predictions = np.concatenate(all_predictions)
                        all_targets = np.concatenate(all_targets)


                        test_accuracies.append((all_predictions==all_targets).mean())
                        test_accuracies_full_list.append((all_predictions==all_targets).all(-1).mean())
                        test_losses.append(np.mean(all_losses))

                        # Best model checkpoint and early stopping
                        current_metric = test_accuracies[-1]
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
                        axacc_train.plot(iters, train_accuracies, 'b-', alpha=0.7, label='Find grained')   
                        axacc_train.plot(iters, train_accuracies_full_list, 'k--', alpha=0.7, label='Full list')   
                        axacc_test.plot(iters, test_accuracies, 'b-', alpha=0.7, label='Fine grained')        
                        axacc_test.plot(iters, test_accuracies_full_list, 'k--', alpha=0.7, label='Full list')        
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
            if (bi%args.save_every==0 or bi==args.training_iterations-1) and bi != start_iter:
                torch.save(
                    {
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'scaler_state_dict':scaler.state_dict(),
                    'iteration':bi,
                    'train_accuracies_full_list':train_accuracies_full_list,
                    'train_accuracies':train_accuracies,
                    'test_accuracies_full_list':test_accuracies_full_list,
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
                    } , f'{args.log_dir}/checkpoint.pt')
            
            pbar.update(1)
