import argparse
import multiprocessing # Used for GIF generation
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
from tqdm.auto import tqdm

from utils.samplers import QAMNISTSampler
from tasks.image_classification.plotting import plot_neural_dynamics
from tasks.qamnist.plotting import make_qamnist_gif
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import qamnist_loss
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup
from tasks.parity.utils import reshape_attention_weights
from tasks.qamnist.utils import get_dataset, prepare_model
from models.utils import reshape_predictions, get_latest_checkpoint




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

    # Task Configuration
    parser.add_argument('--q_num_images', type=int, default=3, help='Number of inputs per min mnist view')
    parser.add_argument('--q_num_images_delta', type=int, default=2, help='Range of numbers for QMNIST dataset')
    parser.add_argument('--q_num_repeats_per_input', type=int, default=10, help='Number of MNIST repeats to show model')
    parser.add_argument('--q_num_operations', type=int, default=3, help='The number of operations to apply.')
    parser.add_argument('--q_num_operations_delta', type=int, default=2, help='The range of operations to apply.')
    parser.add_argument('--q_num_answer_steps', type=int, default=10, help='The number of steps to answer a question, after observing digits and operator embeddings.')

    # Model Architecture
    parser.add_argument('--model_type', type=str, default='ctm', choices=['ctm', 'lstm'], help='Type of model to use.')
    parser.add_argument('--d_model', type=int, default=1024, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=64, help='Dimension of the input.')
    parser.add_argument('--synapse_depth', type=int, default=1, help='Depth of U-NET model for synapse. 1=linear, no unet.')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--n_synch_out', type=int, default=32, help='Number of neurons to use for output synch.')
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons to use for observation/action synch.')
    parser.add_argument('--neuron_select_type', type=str, default='random', help='Protocol for selecting neuron subset.')
    parser.add_argument('--n_random_pairing_self', type=int, default=256, help='Number of neurons paired self-to-self for synch.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of internal ticks.')
    parser.add_argument('--memory_length', type=int, default=30, help='Length of the pre-activation history for NLMS.')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep memory.')
    parser.add_argument('--memory_hidden_dims', type=int, default=4, help='Hidden dimensions of the memory if using deep memory.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs.')

    # Training
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--batch_size_test', type=int, default=256, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the model.')
    parser.add_argument('--training_iterations', type=int, default=100001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'], help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+', help='Learning rate scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')

    # Housekeeping
    parser.add_argument('--log_dir', type=str, default='logs/qamnist', help='Directory for logging.')
    parser.add_argument('--data_root', type=str, default='data/', help='Where to save dataset.')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoints every this many iterations.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False, help='Reload from disk?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False, help='Reload only the model from disk?')
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics every this many iterations.')
    parser.add_argument('--n_test_batches', type=int, default=20, help='How many minibatches to approx metrics. Set to -1 for full eval')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='List of GPU(s) to use. Set to -1 to use CPU.')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')


    args = parser.parse_args()
    return args


if __name__=='__main__':

    # Hosuekeeping
    args = parse_args()
    
    set_seed(args.seed)

    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    # Data
    train_data, test_data, class_labels, dataset_mean, dataset_std = get_dataset(args.q_num_images, args.q_num_images_delta, args.q_num_repeats_per_input, args.q_num_operations, args.q_num_operations_delta)
    train_sampler = QAMNISTSampler(train_data, batch_size=args.batch_size)
    trainloader = torch.utils.data.DataLoader(train_data, num_workers=0, batch_sampler=train_sampler)

    test_sampler = QAMNISTSampler(test_data, batch_size=args.batch_size_test)
    testloader = torch.utils.data.DataLoader(test_data, num_workers=0, batch_sampler=test_sampler)
    

    prediction_reshaper = [-1]  # Problem specific
    args.out_dims = len(class_labels)
    args.use_most_certain = args.model_type == "ctm"

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

    # Build model
    model = prepare_model(args, device)

    # For lazy modules so that we can get param count
    pseudo_data =  train_data.__getitem__(0)
    pseudo_inputs = pseudo_data[0].unsqueeze(0).to(device)
    pseudo_z = torch.tensor(pseudo_data[1]).unsqueeze(0).unsqueeze(2).to(device)
    model(pseudo_inputs, pseudo_z) 
    
    model.train()

    print(f'Total params: {sum(p.numel() for p in model.parameters())}')
    

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.lr, 
                                  eps=1e-8, 
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
    train_accuracies_most_certain = []  # This will be selected according to what is returned by loss function
    test_accuracies_most_certain = []
    iters = []
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
            train_accuracies = checkpoint['train_accuracies']
            test_losses = checkpoint['test_losses']
            test_accuracies_most_certain = checkpoint['test_accuracies_most_certain']
            test_accuracies = checkpoint['test_accuracies']
            iters = checkpoint['iters']
        else:
            print('Only reloading model!')
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
    
    
    # Training
    iterator = iter(trainloader)  # Not training in epochs, but rather iterations. Need to reset this from time to time
    
    with tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):
            current_lr = optimizer.param_groups[-1]['lr']

            try:
                inputs, z, _, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, z, _, targets = next(iterator)
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            z = torch.stack(z, 1).to(device)
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=args.use_amp):
                predictions, certainties, synchronisation = model(inputs, z)

                predictions_answer_steps = predictions[:, :, -args.q_num_answer_steps:]
                certainties_answer_steps = certainties[:, :, -args.q_num_answer_steps:]

                loss, where_most_certain = qamnist_loss(predictions_answer_steps, certainties_answer_steps, targets, use_most_certain=args.use_most_certain)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            accuracy = (predictions_answer_steps.argmax(1)[torch.arange(predictions_answer_steps.size(0), device=predictions.device),where_most_certain] == targets).float().mean().item()
            pbar.set_description(f'Dataset=QAMNIST. Loss={loss.item():0.3f}. Accuracy={accuracy:0.3f}. LR={current_lr:0.6f}. Where_certain={where_most_certain.float().mean().item():0.2f}+-{where_most_certain.float().std().item():0.2f} ({where_most_certain.min().item():d}<->{where_most_certain.max().item():d})')

            # Metrics tracking and plotting
            if bi%args.track_every==0:
                model.eval()
                with torch.inference_mode():
                    
                    inputs, z, question_readable, targets = next(iter(testloader))
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    z = torch.stack(z, 1).to(device)
                    pbar.set_description('Tracking: Processing test data')
                    predictions, certainties, synchronisation, pre_activations, post_activations, attention_tracking, embedding_tracking = model(inputs, z, track=True)

                    predictions = reshape_predictions(predictions, prediction_reshaper)
                    attention = reshape_attention_weights(attention_tracking)

                    T = predictions.size(-1)
                    B = predictions.size(0)
                    gif_inputs = torch.zeros((T, B, 1, 32, 32), device=inputs.device)
                    digits_input = inputs.permute(1, 0, 2, 3, 4)
                    gif_inputs[:digits_input.size(0)] = digits_input

                    T_embed = embedding_tracking.shape[0]
                    pad_width = ((0, 0), (0, 0), (0, (32*32)-args.d_input))
                    embedding_padded = np.pad(embedding_tracking, pad_width, mode='constant')
                    reshaped = embedding_padded.reshape(T_embed,B, 1, 32, 32)
                    embedding_input = np.zeros((T_embed, B, 1, 32, 32))
                    embedding_input[:T_embed] = reshaped

                    embedding_tensor = torch.from_numpy(embedding_input).to(gif_inputs.device)
                    gif_inputs[digits_input.size(0):digits_input.size(0) + T_embed] = embedding_tensor[:T_embed]

        
                    pbar.set_description('Tracking: Neural dynamics')
                    plot_neural_dynamics(post_activations, 100, args.log_dir, axis_snap=True)

                    pbar.set_description('Tracking: Producing attention gif')

                    process = multiprocessing.Process(
                        target=make_qamnist_gif,
                        args=(
                        predictions.detach().cpu().numpy(),
                        certainties.detach().cpu().numpy(),
                        targets.detach().cpu().numpy(),
                        pre_activations,
                        post_activations,
                        attention,
                        gif_inputs.detach().cpu().numpy(),
                        f"{args.log_dir}/eval_output_val_{0}_iter_{0}.gif",
                        question_readable
                    ))
                    process.start()

                    
                    ##################################### TRAIN METRICS
                    all_predictions = []
                    all_targets = []
                    all_predictions_most_certain = []
                    all_losses = []
                    
                    iters.append(bi)
                    pbar.set_description('Tracking: Computing loss and accuracy for curves')
                    with torch.inference_mode():
                        loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_test, shuffle=True, num_workers=0)
                        with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                        
                            for inferi, (inputs, z, question_readable, targets) in enumerate(loader):
                                
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                z = torch.stack(z, 1).to(device)
                                these_predictions, certainties, synchronisation = model(inputs, z)

                                these_predictions_answer_steps = these_predictions[:, :, -args.q_num_answer_steps:]
                                certainties_answer_steps = certainties[:, :, -args.q_num_answer_steps:]

                                loss, where_most_certain = qamnist_loss(these_predictions_answer_steps, certainties_answer_steps, targets, use_most_certain=args.use_most_certain)
                                all_losses.append(loss.item())

                                all_targets.append(targets.detach().cpu().numpy())

                                all_predictions_most_certain.append(these_predictions_answer_steps.argmax(1)[torch.arange(these_predictions_answer_steps.size(0), device=these_predictions.device), where_most_certain].detach().cpu().numpy())
                                all_predictions.append(these_predictions_answer_steps.argmax(1).detach().cpu().numpy())
                                
                                if args.n_test_batches!=-1 and inferi%args.n_test_batches==0 and inferi!=0 : break
                                pbar_inner.set_description('Computing metrics for train')
                                pbar_inner.update(1)

                        all_predictions = np.concatenate(all_predictions)
                        all_targets = np.concatenate(all_targets)
                        all_predictions_most_certain = np.concatenate(all_predictions_most_certain)


                        train_accuracies.append(np.mean(all_predictions == all_targets[...,np.newaxis], axis=tuple(range(all_predictions.ndim-1))))
                        train_accuracies_most_certain.append((all_targets == all_predictions_most_certain).mean())
                        train_losses.append(np.mean(all_losses))

                        ##################################### TEST METRICS
                        all_predictions = []
                        all_predictions_most_certain = []
                        all_targets = []
                        all_losses = []
                        loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=0)
                        with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                            for inferi, (inputs, z, question_readable, targets) in enumerate(loader):
                                
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                z = torch.stack(z, 1).to(device)
                                these_predictions, certainties, synchronisation = model(inputs, z)

                                these_predictions_answer_steps = these_predictions[:, :, -args.q_num_answer_steps:]
                                certainties_answer_steps = certainties[:, :, -args.q_num_answer_steps:]

                                loss, where_most_certain = qamnist_loss(these_predictions_answer_steps, certainties_answer_steps, targets, use_most_certain=args.use_most_certain)
                                all_losses.append(loss.item())

                                all_targets.append(targets.detach().cpu().numpy())

                                all_predictions_most_certain.append(these_predictions_answer_steps.argmax(1)[torch.arange(these_predictions_answer_steps.size(0), device=these_predictions_answer_steps.device), where_most_certain].detach().cpu().numpy())
                                all_predictions.append(these_predictions.argmax(1).detach().cpu().numpy())
                                
                                if args.n_test_batches!=-1 and inferi%args.n_test_batches==0 and inferi!=0: break
                                pbar_inner.set_description('Computing metrics for test')
                                pbar_inner.update(1)

                        all_predictions = np.concatenate(all_predictions)
                        all_targets = np.concatenate(all_targets)
                        all_predictions_most_certain = np.concatenate(all_predictions_most_certain)
                        
                        test_accuracies.append(np.mean(all_predictions == all_targets[...,np.newaxis], axis=tuple(range(all_predictions.ndim-1))))
                        test_accuracies_most_certain.append((all_targets == all_predictions_most_certain).mean())
                        test_losses.append(np.mean(all_losses))
                            

                        figacc = plt.figure(figsize=(10, 10))
                        axacc_train = figacc.add_subplot(211)
                        axacc_test = figacc.add_subplot(212)
                        cm = sns.color_palette("viridis", as_cmap=True)

                        axacc_train.plot(iters, train_accuracies_most_certain, 'k--', alpha=0.7, label='Most certain')   
                        axacc_test.plot(iters, test_accuracies_most_certain, 'k--', alpha=0.7, label='Most certain')        
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
                            



            # Save model
            if (bi%args.save_every==0 or bi==args.training_iterations-1) and bi != start_iter:
                pbar.set_description('Saving model checkpoint...')
                torch.save(
                    {
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'scaler_state_dict':scaler.state_dict(),
                    'iteration':bi,
                    'train_accuracies_most_certain':train_accuracies_most_certain,
                    'train_accuracies':train_accuracies,
                    'test_accuracies_most_certain':test_accuracies_most_certain,
                    'test_accuracies':test_accuracies,
                    'train_losses':train_losses,
                    'test_losses':test_losses,
                    'iters':iters,
                    'args':args,
                    'torch_rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    } , f'{args.log_dir}/checkpoint_{bi}.pt')
            
            pbar.update(1)
