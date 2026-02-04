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

from data.custom_datasets import MazeImageFolder
from models.ctm import ContinuousThoughtMachine
from models.lstm import LSTMBaseline
from models.ff import FFBaseline
from tasks.mazes.plotting import make_maze_gif
from tasks.image_classification.plotting import plot_neural_dynamics 
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import maze_loss 
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

import torchvision
torchvision.disable_beta_transforms_warning()

import warnings
warnings.filterwarnings("ignore", message="using precomputed metric; inverse_transform will be unavailable")
warnings.filterwarnings('ignore', message='divide by zero encountered in power', category=RuntimeWarning)
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

    # Model Selection
    parser.add_argument('--model', type=str, required=True, choices=['ctm', 'lstm', 'ff'], help='Model type to train.')

    # Model Architecture
    # Common across all or most
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--backbone_type', type=str, default='resnet34-2', help='Type of backbone featureiser.') # Default changed from original script
    # CTM / LSTM specific
    parser.add_argument('--d_input', type=int, default=128, help='Dimension of the input (CTM, LSTM).')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads (CTM, LSTM).') # Default changed
    parser.add_argument('--iterations', type=int, default=75, help='Number of internal ticks (CTM, LSTM).')
    parser.add_argument('--positional_embedding_type', type=str, default='none',
                        help='Type of positional embedding (CTM, LSTM).', choices=['none',
                                                                       'learnable-fourier',
                                                                       'multi-learnable-fourier',
                                                                       'custom-rotational'])

    # CTM specific
    parser.add_argument('--synapse_depth', type=int, default=8, help='Depth of U-NET model for synapse. 1=linear, no unet (CTM only).') # Default changed
    parser.add_argument('--n_synch_out', type=int, default=32, help='Number of neurons to use for output synch (CTM only).') # Default changed
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons to use for observation/action synch (CTM only).') # Default changed
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing', help='Protocol for selecting neuron subset (CTM only).')
    parser.add_argument('--n_random_pairing_self', type=int, default=0, help='Number of neurons paired self-to-self for synch (CTM only).')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of the pre-activation history for NLMS (CTM only).')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Use deep memory (CTM only).')
    parser.add_argument('--memory_hidden_dims', type=int, default=32, help='Hidden dimensions of the memory if using deep memory (CTM only).') # Default changed
    parser.add_argument('--dropout_nlm', type=float, default=None, help='Dropout rate for NLMs specifically. Unset to match dropout on the rest of the model (CTM only).')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs (CTM only).')
    # LSTM specific
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM stacked layers (LSTM only).') # Added LSTM arg

    # Task Specific Args (Common to all models for this task)
    parser.add_argument('--maze_route_length', type=int, default=100, help='Length to truncate targets.')
    parser.add_argument('--cirriculum_lookahead', type=int, default=5, help='How far to look ahead for cirriculum.')


    # Training
    parser.add_argument('--expand_range', action=argparse.BooleanOptionalAction, default=True, help='Mazes between 0 and 1 = False. Between -1 and 1 = True. Legacy checkpoints use 0 and 1.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.') # Default changed
    parser.add_argument('--batch_size_test', type=int, default=64, help='Batch size for testing.') # Default changed
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the model.') # Default changed
    parser.add_argument('--training_iterations', type=int, default=100001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'], help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+', help='Learning rate scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[], help='List to exclude from weight decay. Typically good: bn, ln, bias, start')
    parser.add_argument('--num_workers_train', type=int, default=0, help='Num workers training.') # Renamed from num_workers, kept default
    parser.add_argument('--gradient_clipping', type=float, default=-1, help='Gradient quantile clipping value (-1 to disable).')
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False, help='Try to compile model components.')

    # Logging and Saving
    parser.add_argument('--log_dir', type=str, default='logs/scratch', help='Directory for logging.')
    parser.add_argument('--dataset', type=str, default='mazes-medium', help='Dataset to use.', choices=['mazes-medium', 'mazes-large', 'mazes-small']) 
    parser.add_argument('--data_root', type=str, default='data/mazes', help='Data root.')
    
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoints every this many iterations.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False, help='Reload from disk?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False, help='Reload only the model from disk?')
    parser.add_argument('--strict_reload', action=argparse.BooleanOptionalAction, default=True, help='Should use strict reload for model weights.') # Added back
    parser.add_argument('--ignore_metrics_when_reloading', action=argparse.BooleanOptionalAction, default=False, help='Ignore metrics when reloading (for debugging)?') # Added back

    # Tracking
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics every this many iterations.')
    parser.add_argument('--n_test_batches', type=int, default=20, help='How many minibatches to approx metrics. Set to -1 for full eval') # Default changed

    # Device
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='List of GPU(s) to use. Set to -1 to use CPU.')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')

    # Early stopping and best checkpoint (optional, disabled by default)
    parser.add_argument('--early_stopping_patience', type=int, default=-1, help='Stop training if test metric does not improve for this many validations. -1 to disable.')
    parser.add_argument('--save_best_checkpoint', action=argparse.BooleanOptionalAction, default=False, help='Save best model checkpoint separately as best_checkpoint.pt.')

    args = parser.parse_args()
    return args


if __name__=='__main__':

    # Hosuekeeping
    args = parse_args()

    set_seed(args.seed, False)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    assert args.dataset in ['mazes-medium', 'mazes-large', 'mazes-small']

    

    prediction_reshaper = [args.maze_route_length, 5]  # Problem specific 
    args.out_dims = args.maze_route_length * 5 # Output dimension before reshaping

    # For total reproducibility
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


    # Build model conditionally
    model = None
    if args.model == 'ctm':
        model = ContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper, 
            dropout=args.dropout,
            dropout_nlm=args.dropout_nlm,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
        ).to(device)
    elif args.model == 'lstm':
         model = LSTMBaseline(
            num_layers=args.num_layers,
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads, 
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper, 
            dropout=args.dropout,
        ).to(device)
    elif args.model == 'ff':
        model = FFBaseline(
            d_model=args.d_model,
            backbone_type=args.backbone_type,
            out_dims=args.out_dims,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    try:
        # Determine pseudo input shape based on dataset
        h_w = 39 if args.dataset in ['mazes-small', 'mazes-medium'] else 99 # Example dimensions
        pseudo_inputs = torch.zeros((1, 3, h_w, h_w), device=device).float()
        model(pseudo_inputs)
    except Exception as e:
         print(f"Warning: Pseudo forward pass failed: {e}")

    print(f'Total params: {sum(p.numel() for p in model.parameters())}')

    # Data
    dataset_mean = [0,0,0]  # For plotting later
    dataset_std = [1,1,1]

    which_maze = args.dataset.split('-')[-1]
    data_root = f'{args.data_root}/{which_maze}'

    train_data = MazeImageFolder(root=f'{data_root}/train/', which_set='train', maze_route_length=args.maze_route_length, expand_range=args.expand_range)
    test_data = MazeImageFolder(root=f'{data_root}/test/', which_set='test', maze_route_length=args.maze_route_length, expand_range=args.expand_range)

    num_workers_test = 1 # Defaulting to 1, can be changed
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers_train, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=num_workers_test, drop_last=False)

    # For lazy modules so that we can get param count
    

    model.train()

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


    # Metrics tracking
    start_iter = 0
    train_losses = []
    test_losses = []
    train_accuracies = []  # Per tick/step accuracy list
    test_accuracies = []   
    train_accuracies_most_certain = [] # Accuracy, fine-grained
    test_accuracies_most_certain = []  
    train_accuracies_most_certain_permaze = [] # Full maze accuracy
    test_accuracies_most_certain_permaze = []  
    iters = []

    best_test_metric = -float('inf')
    early_stopping_counter = 0

    scaler = torch.amp.GradScaler("cuda" if "cuda" in device else "cpu", enabled=args.use_amp)
    if args.reload:
        checkpoint_path = f'{args.log_dir}/checkpoint.pt'
        if os.path.isfile(checkpoint_path):
            print(f'Reloading from: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if not args.strict_reload: print('WARNING: not using strict reload for model weights!')
            load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=args.strict_reload)
            print(f" Loaded state_dict. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

            if not args.reload_model_only:
                print('Reloading optimizer etc.')
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict']) # Load scaler state
                start_iter = checkpoint['iteration']

                if not args.ignore_metrics_when_reloading:
                    train_losses = checkpoint['train_losses']
                    test_losses = checkpoint['test_losses']
                    train_accuracies = checkpoint['train_accuracies']
                    test_accuracies = checkpoint['test_accuracies']
                    iters = checkpoint['iters']
                    train_accuracies_most_certain = checkpoint['train_accuracies_most_certain']
                    test_accuracies_most_certain = checkpoint['test_accuracies_most_certain']
                    train_accuracies_most_certain_permaze = checkpoint['train_accuracies_most_certain_permaze']
                    test_accuracies_most_certain_permaze = checkpoint['test_accuracies_most_certain_permaze']
                else:
                     print("Ignoring metrics history upon reload.")

                # Load early stopping state (backward compatible)
                best_test_metric = checkpoint.get('best_test_metric', -float('inf'))
                early_stopping_counter = checkpoint.get('early_stopping_counter', 0)

            else:
                print('Only reloading model!')

            if 'torch_rng_state' in checkpoint:
                # Reset seeds
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
        if hasattr(model, 'backbone'):
            model.backbone = torch.compile(model.backbone, mode='reduce-overhead', fullgraph=True)
        # Compile synapses only for CTM
        if args.model == 'ctm':
            model.synapses = torch.compile(model.synapses, mode='reduce-overhead', fullgraph=True)

    # Training
    iterator = iter(trainloader)
    with tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):
            current_lr = optimizer.param_groups[-1]['lr']

            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets = next(iterator)

            inputs = inputs.to(device)
            targets = targets.to(device) # Shape (B, SeqLength)

            # All for nice metric printing:
            loss = None
            accuracy_finegrained = None # Per-step accuracy at chosen tick
            where_most_certain_val = -1.0 # Default value
            where_most_certain_std = 0.0
            where_most_certain_min = -1
            where_most_certain_max = -1
            upto_where_mean = -1.0
            upto_where_std = 0.0
            upto_where_min = -1
            upto_where_max = -1


            # Model-specific forward, reshape, and loss calculation
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=args.use_amp):
                if args.do_compile: # CUDAGraph marking applied if compiling any model
                     torch.compiler.cudagraph_mark_step_begin()

                if args.model == 'ctm':
                    # CTM output: (B, SeqLength*5, Ticks), Certainties: (B, Ticks)
                    predictions_raw, certainties, synchronisation = model(inputs)
                    # Reshape predictions: (B, SeqLength, 5, Ticks)
                    predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1))
                    loss, where_most_certain, upto_where = maze_loss(predictions, certainties, targets, cirriculum_lookahead=args.cirriculum_lookahead, use_most_certain=True)
                    # Accuracy uses predictions[B, S, C, T] indexed at where_most_certain[B] -> gives (B, S, C) -> argmax(2) -> (B,S)
                    accuracy_finegrained = (predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, where_most_certain] == targets).float().mean().item()

                elif args.model == 'lstm':
                    # LSTM output: (B, SeqLength*5, Ticks), Certainties: (B, Ticks)
                    predictions_raw, certainties, synchronisation = model(inputs)
                     # Reshape predictions: (B, SeqLength, 5, Ticks)
                    predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1))
                    loss, where_most_certain, upto_where = maze_loss(predictions, certainties, targets, cirriculum_lookahead=args.cirriculum_lookahead, use_most_certain=False)
                    # where_most_certain should be -1 (last tick) here. Accuracy calc follows same logic.
                    accuracy_finegrained = (predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, where_most_certain] == targets).float().mean().item()

                elif args.model == 'ff':
                    # Assume FF output: (B, SeqLength*5)
                    predictions_raw = model(inputs)
                    # Reshape predictions: (B, SeqLength, 5)
                    predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5)
                    # FF has no certainties, pass None. maze_loss must handle this.
                    # Unsqueeze predictions for compatibility with maze loss calcluation
                    loss, where_most_certain, upto_where = maze_loss(predictions.unsqueeze(-1), None, targets, cirriculum_lookahead=args.cirriculum_lookahead, use_most_certain=False)
                    # where_most_certain should be -1 here. Accuracy uses 3D prediction tensor.
                    accuracy_finegrained = (predictions.argmax(2) == targets).float().mean().item()


                # Extract stats from loss outputs if they are tensors
                if torch.is_tensor(where_most_certain):
                    where_most_certain_val = where_most_certain.float().mean().item()
                    where_most_certain_std = where_most_certain.float().std().item()
                    where_most_certain_min = where_most_certain.min().item()
                    where_most_certain_max = where_most_certain.max().item()
                elif isinstance(where_most_certain, int): # Handle case where it might return -1 directly
                     where_most_certain_val = float(where_most_certain)
                     where_most_certain_min = where_most_certain
                     where_most_certain_max = where_most_certain

                if isinstance(upto_where, (np.ndarray, list)) and len(upto_where) > 0: # Check if it's a list/array
                    upto_where_mean = np.mean(upto_where)
                    upto_where_std = np.std(upto_where)
                    upto_where_min = np.min(upto_where)
                    upto_where_max = np.max(upto_where)


            scaler.scale(loss).backward()

            if args.gradient_clipping!=-1: 
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            # Conditional Tqdm Description
            pbar_desc = f'Loss={loss.item():0.3f}. Acc(step)={accuracy_finegrained:0.3f}. LR={current_lr:0.6f}.'
            if args.model in ['ctm', 'lstm'] or torch.is_tensor(where_most_certain): # Show stats if available
                 pbar_desc += f' Where_certain={where_most_certain_val:0.2f}+-{where_most_certain_std:0.2f} ({where_most_certain_min:d}<->{where_most_certain_max:d}).'
            if isinstance(upto_where, (np.ndarray, list)) and len(upto_where) > 0:
                 pbar_desc += f' Path pred stats: {upto_where_mean:0.2f}+-{upto_where_std:0.2f} ({upto_where_min:d} --> {upto_where_max:d})'

            pbar.set_description(f'Dataset={args.dataset}. Model={args.model}. {pbar_desc}')


            # Metrics tracking and plotting
            if bi%args.track_every==0 and (bi != 0 or args.reload_model_only):
                model.eval() # Use eval mode for consistency during tracking
                with torch.inference_mode(): # Use inference mode for tracking

                    


                    # --- Quantitative Metrics ---
                    iters.append(bi)
                    # Re-initialize metric lists for this evaluation step
                    current_train_losses_eval = []
                    current_test_losses_eval = []
                    current_train_accuracies_eval = []
                    current_test_accuracies_eval = []
                    current_train_accuracies_most_certain_eval = []
                    current_test_accuracies_most_certain_eval = []
                    current_train_accuracies_most_certain_permaze_eval = []
                    current_test_accuracies_most_certain_permaze_eval = []

                    # TRAIN METRICS
                    pbar.set_description('Tracking: Computing TRAIN metrics')
                    loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_test, shuffle=True, num_workers=num_workers_test) # Use consistent num_workers
                    all_targets_list = []
                    all_predictions_list = [] # Per step/tick predictions argmax (N, S, T) or (N, S)
                    all_predictions_most_certain_list = [] # Predictions at chosen step/tick argmax (N, S)
                    all_losses = []

                    with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                        for inferi, (inputs, targets) in enumerate(loader):
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            all_targets_list.append(targets.detach().cpu().numpy()) # N x S

                            # Model-specific forward, reshape, loss for evaluation
                            if args.model == 'ctm':
                                predictions_raw, certainties, _ = model(inputs)
                                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1)) # B,S,C,T
                                loss, where_most_certain, _ = maze_loss(predictions, certainties, targets, use_most_certain=True)
                                all_predictions_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S,C,T -> argmax class -> B,S,T
                                pred_at_certain = predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, where_most_certain] # B,S
                                all_predictions_most_certain_list.append(pred_at_certain.detach().cpu().numpy())

                            elif args.model == 'lstm':
                                predictions_raw, certainties, _ = model(inputs)
                                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1)) # B,S,C,T
                                loss, where_most_certain, _ = maze_loss(predictions, certainties, targets, use_most_certain=False) # where = -1
                                all_predictions_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S,C,T
                                pred_at_certain = predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, where_most_certain] # B,S (at last tick)
                                all_predictions_most_certain_list.append(pred_at_certain.detach().cpu().numpy())

                            elif args.model == 'ff':
                                predictions_raw = model(inputs) # B, S*C
                                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5) # B,S,C
                                loss, where_most_certain, _ = maze_loss(predictions.unsqueeze(-1), None, targets, use_most_certain=False) # where = -1
                                all_predictions_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S
                                all_predictions_most_certain_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S (same as above for FF)


                            all_losses.append(loss.item())

                            if args.n_test_batches != -1 and inferi >= args.n_test_batches -1 : break
                            pbar_inner.set_description(f'Computing metrics for train (Batch {inferi+1})')
                            pbar_inner.update(1)

                    all_targets = np.concatenate(all_targets_list) # N, S
                    all_predictions = np.concatenate(all_predictions_list) # N, S, T or N, S
                    all_predictions_most_certain = np.concatenate(all_predictions_most_certain_list) # N, S

                    train_losses.append(np.mean(all_losses))
                    # Calculate per step/tick accuracy averaged over batches
                    if args.model in ['ctm', 'lstm']:
                         # all_predictions shape (N, S, T), all_targets shape (N, S) -> compare targets to each tick prediction
                         train_accuracies.append(np.mean(all_predictions == all_targets[:,:,np.newaxis], axis=0)) # Mean over N -> (S, T)
                    else: # FF
                         # all_predictions shape (N, S), all_targets shape (N, S)
                         train_accuracies.append(np.mean(all_predictions == all_targets, axis=0)) # Mean over N -> (S,)

                    # Calculate accuracy at chosen step/tick ("most certain") averaged over all steps and batches
                    train_accuracies_most_certain.append((all_targets == all_predictions_most_certain).mean()) # Scalar
                    # Calculate full maze accuracy at chosen step/tick averaged over batches
                    train_accuracies_most_certain_permaze.append((all_targets == all_predictions_most_certain).reshape(all_targets.shape[0], -1).all(-1).mean()) # Scalar


                    # TEST METRICS
                    pbar.set_description('Tracking: Computing TEST metrics')
                    loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=num_workers_test)
                    all_targets_list = []
                    all_predictions_list = []
                    all_predictions_most_certain_list = []
                    all_losses = []

                    with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                        for inferi, (inputs, targets) in enumerate(loader):
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            all_targets_list.append(targets.detach().cpu().numpy())

                             # Model-specific forward, reshape, loss for evaluation
                            if args.model == 'ctm':
                                predictions_raw, certainties, _ = model(inputs)
                                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1)) # B,S,C,T
                                loss, where_most_certain, _ = maze_loss(predictions, certainties, targets, use_most_certain=True)
                                all_predictions_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S,T
                                pred_at_certain = predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, where_most_certain] # B,S
                                all_predictions_most_certain_list.append(pred_at_certain.detach().cpu().numpy())

                            elif args.model == 'lstm':
                                predictions_raw, certainties, _ = model(inputs)
                                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5, predictions_raw.size(-1)) # B,S,C,T
                                loss, where_most_certain, _ = maze_loss(predictions, certainties, targets, use_most_certain=False) # where = -1
                                all_predictions_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S,T
                                pred_at_certain = predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, where_most_certain] # B,S (at last tick)
                                all_predictions_most_certain_list.append(pred_at_certain.detach().cpu().numpy())

                            elif args.model == 'ff':
                                predictions_raw = model(inputs) # B, S*C
                                predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 5) # B,S,C
                                loss, where_most_certain, _ = maze_loss(predictions.unsqueeze(-1), None, targets, use_most_certain=False) # where = -1
                                all_predictions_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S
                                all_predictions_most_certain_list.append(predictions.argmax(2).detach().cpu().numpy()) # B,S (same as above for FF)


                            all_losses.append(loss.item())

                            if args.n_test_batches != -1 and inferi >= args.n_test_batches -1: break
                            pbar_inner.set_description(f'Computing metrics for test (Batch {inferi+1})')
                            pbar_inner.update(1)

                    all_targets = np.concatenate(all_targets_list)
                    all_predictions = np.concatenate(all_predictions_list)
                    all_predictions_most_certain = np.concatenate(all_predictions_most_certain_list)

                    test_losses.append(np.mean(all_losses))
                    # Calculate per step/tick accuracy
                    if args.model in ['ctm', 'lstm']:
                         test_accuracies.append(np.mean(all_predictions == all_targets[:,:,np.newaxis], axis=0)) # -> (S, T)
                    else: # FF
                         test_accuracies.append(np.mean(all_predictions == all_targets, axis=0)) # -> (S,)

                    # Calculate "most certain" accuracy
                    test_accuracies_most_certain.append((all_targets == all_predictions_most_certain).mean()) # Scalar
                    # Calculate full maze accuracy
                    test_accuracies_most_certain_permaze.append((all_targets == all_predictions_most_certain).reshape(all_targets.shape[0], -1).all(-1).mean()) # Scalar

                # Best model checkpoint and early stopping
                current_metric = test_accuracies_most_certain_permaze[-1]
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
                    if args.early_stopping_patience != -1 and early_stopping_counter >= args.early_stopping_patience:
                        print(f'Early stopping triggered at iteration {bi}. '
                              f'No improvement for {args.early_stopping_patience} validations.')
                        break

                # --- Plotting ---
                # Accuracy Plot (Handling different dimensions)
                figacc = plt.figure(figsize=(10, 10))
                axacc_train = figacc.add_subplot(211)
                axacc_test = figacc.add_subplot(212)
                cm = sns.color_palette("viridis", as_cmap=True)

                # Plot per step/tick accuracy
                # train_accuracies is List[(S, T)] or List[(S,)]
                # We need to average over S dimension for plotting
                train_acc_plot = [np.mean(acc_s) for acc_s in train_accuracies] # List[Scalar] or List[Scalar] after mean
                test_acc_plot = [np.mean(acc_s) for acc_s in test_accuracies]   # List[Scalar] or List[Scalar] after mean

                axacc_train.plot(iters, train_acc_plot, 'g-', alpha=0.5, label='Avg Step Acc')
                axacc_test.plot(iters, test_acc_plot, 'g-', alpha=0.5, label='Avg Step Acc')


                # Plot most certain accuracy
                axacc_train.plot(iters, train_accuracies_most_certain, 'k--', alpha=0.7, label='Most Certain (Avg Step)')
                axacc_test.plot(iters, test_accuracies_most_certain, 'k--', alpha=0.7, label='Most Certain (Avg Step)')
                # Plot full maze accuracy
                axacc_train.plot(iters, train_accuracies_most_certain_permaze, 'r-', alpha=0.6, label='Full Maze')
                axacc_test.plot(iters, test_accuracies_most_certain_permaze, 'r-', alpha=0.6, label='Full Maze')

                axacc_train.set_title('Train Accuracy')
                axacc_test.set_title('Test Accuracy')
                axacc_train.legend(loc='lower right')
                axacc_test.legend(loc='lower right')
                axacc_train.set_xlim([0, args.training_iterations])
                axacc_test.set_xlim([0, args.training_iterations])
                axacc_train.set_ylim([0, 1]) # Set Ylim for accuracy
                axacc_test.set_ylim([0, 1])

                figacc.tight_layout()
                figacc.savefig(f'{args.log_dir}/accuracies.png', dpi=150)
                plt.close(figacc)

                # Loss Plot
                figloss = plt.figure(figsize=(10, 5))
                axloss = figloss.add_subplot(111)
                axloss.plot(iters, train_losses, 'b-', linewidth=1, alpha=0.8, label=f'Train: {train_losses[-1]:.4f}')
                axloss.plot(iters, test_losses, 'r-', linewidth=1, alpha=0.8, label=f'Test: {test_losses[-1]:.4f}')
                axloss.legend(loc='upper right')
                axloss.set_xlim([0, args.training_iterations])
                axloss.set_ylim(bottom=0)

                figloss.tight_layout()
                figloss.savefig(f'{args.log_dir}/losses.png', dpi=150)
                plt.close(figloss)

                # --- Visualization Section (Conditional) ---
                if args.model in ['ctm', 'lstm']:
                    #  try:
                        inputs_viz, targets_viz = next(iter(testloader))
                        inputs_viz = inputs_viz.to(device)
                        targets_viz = targets_viz.to(device)
                        # Find longest path in batch for potentially better visualization
                        longest_index = (targets_viz!=4).sum(-1).argmax() # Action 4 assumed padding/end

                        # Track internal states
                        predictions_viz_raw, certainties_viz, _, pre_activations_viz, post_activations_viz, attention_tracking_viz = model(inputs_viz, track=True)

                        # Reshape predictions (assuming raw is B, D, T)
                        predictions_viz = predictions_viz_raw.reshape(predictions_viz_raw.size(0), -1, 5, predictions_viz_raw.size(-1)) # B, S, C, T

                        att_shape = (model.kv_features.shape[2], model.kv_features.shape[3])
                        attention_tracking_viz = attention_tracking_viz.reshape(
                            attention_tracking_viz.shape[0],
                            attention_tracking_viz.shape[1], -1, att_shape[0], att_shape[1])

                        # Plot dynamics (common plotting function)
                        plot_neural_dynamics(post_activations_viz, 100, args.log_dir, axis_snap=True)

                        # Create maze GIF (task-specific plotting)
                        make_maze_gif((inputs_viz[longest_index].detach().cpu().numpy()+1)/2,
                                      predictions_viz[longest_index].detach().cpu().numpy(), # Pass reshaped B,S,C,T -> S,C,T
                                      targets_viz[longest_index].detach().cpu().numpy(), # S
                                      attention_tracking_viz[:, longest_index],  # Pass T, (H), H, W
                                      args.log_dir)
                    #  except Exception as e:
                    #       print(f"Visualization failed for model {args.model}: {e}")
                # --- End Visualization ---

                model.train() # Switch back to train mode


            # Save model checkpoint
            if (bi % args.save_every == 0 or bi == args.training_iterations - 1) and bi != start_iter:
                pbar.set_description('Saving model checkpoint...')
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(), # Save scaler state
                    'iteration': bi,
                    # Save all tracked metrics
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_accuracies': train_accuracies, # List of (S, T) or (S,) arrays
                    'test_accuracies': test_accuracies,   # List of (S, T) or (S,) arrays
                    'train_accuracies_most_certain': train_accuracies_most_certain, # List of scalars
                    'test_accuracies_most_certain': test_accuracies_most_certain,   # List of scalars
                    'train_accuracies_most_certain_permaze': train_accuracies_most_certain_permaze, # List of scalars
                    'test_accuracies_most_certain_permaze': test_accuracies_most_certain_permaze,   # List of scalars
                    'iters': iters,
                    'args': args, # Save args used for this run
                    # Early stopping state
                    'best_test_metric': best_test_metric,
                    'early_stopping_counter': early_stopping_counter,
                    # RNG states
                    'torch_rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                }
                torch.save(checkpoint_data, f'{args.log_dir}/checkpoint.pt')

            pbar.update(1)