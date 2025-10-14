import os
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import FuncFormatter
from scipy.special import softmax
import imageio.v2 as imageio
from PIL import Image
import math
import re
from tqdm import tqdm
sns.set_style('darkgrid')
mpl.use('Agg')

from tasks.parity.utils import get_where_most_certain, parse_folder_name
from models.utils import get_latest_checkpoint_file, load_checkpoint, get_model_args_from_checkpoint, get_accuracy_and_loss_from_checkpoint
from tasks.image_classification.plotting import save_frames_to_mp4

def make_parity_gif(predictions, certainties, targets, pre_activations, post_activations, attention_weights, inputs_to_model, filename):

    # Config
    batch_index = 0
    n_neurons_to_visualise = 16
    figscale = 0.28
    n_steps = len(pre_activations)
    frames = []
    heatmap_cmap = sns.color_palette("viridis", as_cmap=True)

    these_pre_acts = pre_activations[:, batch_index, :] # Shape: (T, H)
    these_post_acts = post_activations[:, batch_index, :] # Shape: (T, H)
    these_inputs = inputs_to_model[:, batch_index, :, :, :] # Shape: (T, C, H, W)
    these_predictions = predictions[batch_index, :, :, :] # Shape: (d, C, T)
    these_certainties = certainties[batch_index, :, :] # Shape: (C, T)
    these_attention_weights = attention_weights[:, batch_index, :, :]

    # Create mosaic layout
    mosaic = [['img_data', 'img_data', 'attention', 'attention', 'probs', 'probs', 'target', 'target'] for _ in range(2)] + \
             [['img_data', 'img_data', 'attention', 'attention', 'probs', 'probs', 'target', 'target'] for _ in range(2)] + \
             [['certainty', 'certainty', 'certainty', 'certainty', 'certainty', 'certainty', 'certainty', 'certainty']] + \
             [[f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}', f'trace_{ti}'] for ti in range(n_neurons_to_visualise)] 
             
    for stepi in tqdm(range(n_steps), desc="Processing steps", unit="step"):
        fig_gif, axes_gif = plt.subplot_mosaic(mosaic=mosaic, figsize=(31*figscale*8/4, 76*figscale))

        # Plot predictions
        d = these_predictions.shape[0]
        grid_side = int(np.sqrt(d))
        logits = these_predictions[:, :, stepi]

        probs = softmax(logits, axis=1)
        probs_grid = probs[:, 0].reshape(grid_side, grid_side)
        axes_gif["probs"].imshow(probs_grid, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        axes_gif["probs"].axis('off')
        axes_gif["probs"].set_title('Probabilties')

        # Create and show attention heatmap
        this_input_gate = these_attention_weights[stepi]
        gate_min, gate_max = np.nanmin(this_input_gate), np.nanmax(this_input_gate)
        if not np.isclose(gate_min, gate_max):
            normalized_gate = (this_input_gate - gate_min) / (gate_max - gate_min + 1e-8)
        else:
            normalized_gate = np.zeros_like(this_input_gate)
        attention_weights_heatmap = heatmap_cmap(normalized_gate)[:,:,:3]

        # Show heatmaps
        axes_gif['attention'].imshow(attention_weights_heatmap, vmin=0, vmax=1)
        axes_gif['attention'].axis('off')
        axes_gif['attention'].set_title('Attention')


        # Plot target
        target_grid = targets[batch_index].reshape(grid_side, grid_side)
        axes_gif["target"].imshow(target_grid, cmap='viridis_r', interpolation='nearest', vmin=0, vmax=1)
        axes_gif["target"].axis('off')
        axes_gif["target"].set_title('Target')

        # Add certainty plot
        axes_gif['certainty'].plot(np.arange(n_steps), these_certainties[1], 'k-', linewidth=2)
        axes_gif['certainty'].set_xlim([0, n_steps-1])
        axes_gif['certainty'].axvline(x=stepi, color='black', linewidth=1, alpha=0.5)
        axes_gif['certainty'].set_xticklabels([])
        axes_gif['certainty'].set_yticklabels([])
        axes_gif['certainty'].grid(False)

        # Plot neuron traces
        for neuroni in range(n_neurons_to_visualise):
            ax = axes_gif[f'trace_{neuroni}']

            pre_activation = these_pre_acts[:, neuroni]
            post_activation = these_post_acts[:, neuroni]
            
            ax_pre = ax.twinx()
            
            pre_min, pre_max = np.min(pre_activation), np.max(pre_activation)
            post_min, post_max = np.min(post_activation), np.max(post_activation)
            
            ax_pre.plot(np.arange(n_steps), pre_activation, 
                        color='grey', 
                        linestyle='--', 
                        linewidth=1, 
                        alpha=0.4,
                        label='Pre-activation')
            
            color = 'blue' if neuroni % 2 else 'red'
            ax.plot(np.arange(n_steps), post_activation,
                    color=color,
                    linestyle='-',
                    linewidth=2,
                    alpha=1.0,
                    label='Post-activation')

            ax.set_xlim([0, n_steps-1])
            ax_pre.set_xlim([0, n_steps-1])
            
            if pre_min != pre_max:
                ax_pre.set_ylim([pre_min, pre_max])
            if post_min != post_max:
                ax.set_ylim([post_min, post_max])

            ax.axvline(x=stepi, color='black', linewidth=1, alpha=0.5)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(False)

            ax_pre.set_xticklabels([])
            ax_pre.set_yticklabels([])
            ax_pre.grid(False)

        # Show input image
        this_image = these_inputs[stepi].transpose(1, 2, 0)
        axes_gif['img_data'].imshow(this_image, cmap='viridis', vmin=0, vmax=1)
        axes_gif['img_data'].grid(False) 
        axes_gif['img_data'].set_xticks([])
        axes_gif['img_data'].set_yticks([])
        axes_gif['img_data'].set_title('Input')

        # Save frames
        fig_gif.tight_layout(pad=0.1)
        if stepi == 0:
            fig_gif.savefig(filename.split('.gif')[0]+'_frame0.png', dpi=100)
        if stepi == 1:
            fig_gif.savefig(filename.split('.gif')[0]+'_frame1.png', dpi=100)
        if stepi == n_steps-1:
            fig_gif.savefig(filename.split('.gif')[0]+'_frame-1.png', dpi=100)

        # Convert to frame
        canvas = fig_gif.canvas
        canvas.draw()
        image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image_numpy = image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3]
        frames.append(image_numpy)
        plt.close(fig_gif)

    imageio.mimsave(filename, frames, fps=15, loop=100)

    pass


def plot_attention_trajectory(attention, certainties, input_images, save_dir, filename, args):
    where_most_certain = get_where_most_certain(certainties)
    grid_size = int(math.sqrt(args.parity_sequence_length))
    trajectory = [np.unravel_index(np.argmax(attention[t]), (grid_size, grid_size)) for t in range(args.iterations)]
    x_coords, y_coords = zip(*trajectory)

    plt.figure(figsize=(5, 5))
    plt.imshow(input_images[0], cmap="gray", origin="upper", vmin=0.2, vmax=0.8, interpolation='nearest')

    ax = plt.gca()
    nrows, ncols = input_images[0].shape
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_axisbelow(False)
    plt.xticks([])
    plt.yticks([])

    cmap = plt.get_cmap("plasma")
    norm_time = np.linspace(0, 1, len(trajectory))

    for i in range(len(trajectory) - 1):
        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_coords[i + 1], y_coords[i + 1]
        color = cmap(norm_time[i])
        line, = plt.plot([y1, y2], [x1, x2], color=color, linewidth=6, alpha=0.5, zorder=4)
        line.set_path_effects([
            path_effects.Stroke(linewidth=8, foreground='white'),
            path_effects.Normal()
        ])

    for i, (x, y) in enumerate(trajectory):
        plt.scatter(y, x, color=cmap(norm_time[i]), s=100, edgecolor='white', linewidth=1.5, zorder=5)

    most_certain_point = trajectory[where_most_certain]

    plt.plot(most_certain_point[1], most_certain_point[0],
            marker='x', markersize=18, markeredgewidth=5,
            color='white', linestyle='', zorder=6)
    plt.plot(most_certain_point[1], most_certain_point[0],
            marker='x', markersize=15, markeredgewidth=3,
            color=cmap(norm_time[where_most_certain]), linestyle='', zorder=7)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}_traj.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{save_dir}/{filename}_traj.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

def plot_input(input_images, save_dir, filename):

    plt.figure(figsize=(5, 5))
    plt.imshow(input_images[0], cmap="gray", origin="upper", vmin=0.2, vmax=0.8, interpolation='nearest')

    ax = plt.gca()
    nrows, ncols = input_images[0].shape
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_axisbelow(False)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}_input.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{save_dir}/{filename}_input.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

def plot_target(targets, save_dir, filename, args):
    grid_size = int(math.sqrt(args.parity_sequence_length))
    targets_grid = targets[0].reshape(grid_size, grid_size).detach().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(targets_grid, cmap="gray_r", origin="upper", vmin=0.2, vmax=0.8, interpolation='nearest')
    ax = plt.gca()
    nrows, ncols = targets_grid.shape
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_axisbelow(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}_target.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{save_dir}/{filename}_target.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

def plot_probabilities(predictions, certainties, save_dir, filename, args):
    grid_size = int(math.sqrt(args.parity_sequence_length))
    where_most_certain = get_where_most_certain(certainties)
    predictions_most_certain = predictions[0, :, :, where_most_certain].detach().cpu().numpy()
    probs = softmax(predictions_most_certain, axis=1)
    probs_grid = probs[:, 0].reshape(grid_size, grid_size)
    plt.figure(figsize=(5, 5))
    plt.imshow(probs_grid, cmap="gray", origin="upper", vmin=0.2, vmax=0.8, interpolation='nearest')
    ax = plt.gca()
    nrows, ncols = probs_grid.shape
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_axisbelow(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}_probs.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{save_dir}/{filename}_probs.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

def plot_prediction(predictions, certainties, save_dir, filename, args):
    grid_size = int(math.sqrt(args.parity_sequence_length))
    where_most_certain = get_where_most_certain(certainties)
    predictions_most_certain = predictions[0, :, :, where_most_certain].detach().cpu().numpy()
    class_grid = np.argmax(predictions_most_certain, axis=1).reshape(grid_size, grid_size)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(class_grid, cmap="gray_r", origin="upper", vmin=0, vmax=1, interpolation='nearest')

    ax = plt.gca()
    nrows, ncols = class_grid.shape
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_axisbelow(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}_prediction.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{save_dir}/{filename}_prediction.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

def plot_accuracy_heatmap(overall_accuracies_avg, average_thinking_time, std_thinking_time, scale, save_path, args):
    fig, ax = plt.subplots(figsize=(scale*10, scale*5))
    im = ax.imshow(overall_accuracies_avg.T * 100, aspect='auto', cmap="viridis", origin='lower', extent=[0, args.iterations-1, 0, args.parity_sequence_length-1], vmin=50, vmax=100)
    cbar = fig.colorbar(im, ax=ax, format="%.1f")
    cbar.set_label("Accuracy (%)")
    ax.errorbar(average_thinking_time, np.arange(args.parity_sequence_length), xerr=std_thinking_time, fmt='ko', markersize=2, capsize=2, elinewidth=1, label="Min. Entropy")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Sequence Index")
    ax.set_xlim(0, args.iterations-1)
    ax.set_ylim(0, args.parity_sequence_length-1)
    ax.grid(False)
    ax.legend(loc="upper left")
    fig.tight_layout(pad=0.1)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), format='pdf', bbox_inches="tight")
    plt.close(fig)

def plot_attention_heatmap(overall_attentions_avg, scale, save_path, vmin=None, vmax=None):
    overall_attentions_avg = overall_attentions_avg.reshape(overall_attentions_avg.shape[0], -1)
    fig, ax = plt.subplots(figsize=(scale*10, scale*5))
    im = ax.imshow(overall_attentions_avg.T, aspect='auto', cmap="viridis", origin='lower', extent=[0, overall_attentions_avg.shape[0]-1, 0, overall_attentions_avg.shape[1]-1], vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, format=FuncFormatter(lambda x, _: f"{x:05.2f}"))
    cbar.set_label("Attention Weight")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Sequence Index")
    ax.set_xlim(0, overall_attentions_avg.shape[0]-1)
    ax.set_ylim(0, overall_attentions_avg.shape[1]-1)
    ax.grid(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), format='pdf', bbox_inches="tight")
    plt.close(fig)

def create_accuracies_heatmap_gif(all_accuracies, all_average_thinking_times, all_std_thinking_times, scale, save_dir, args):
    heatmap_components_dir = os.path.join(save_dir, "accuracy_heatmaps")
    os.makedirs(heatmap_components_dir, exist_ok=True)

    image_paths = []

    for i, (accuracies, avg_thinking_time, std_thinking_time) in enumerate(zip(all_accuracies, all_average_thinking_times, all_std_thinking_times)):
        save_path = os.path.join(heatmap_components_dir, f"frame_{i:04d}.png")
        plot_accuracy_heatmap(accuracies, avg_thinking_time, std_thinking_time, scale, save_path, args)
        image_paths.append(save_path)

    gif_path = os.path.join(save_dir, "accuracy_heatmap.gif")
    with imageio.get_writer(gif_path, mode='I', duration=0.3) as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)    

def create_attentions_heatmap_gif(all_attentions, scale, save_path, args):
    heatmap_components_dir = os.path.join(args.log_dir, "attention_heatmaps")
    os.makedirs(heatmap_components_dir, exist_ok=True)

    global_min = min(attentions.min() for attentions in all_attentions)
    global_max = max(attentions.max() for attentions in all_attentions)

    image_paths = []
    
    for i, attentions in enumerate(all_attentions):
        save_path_component = os.path.join(heatmap_components_dir, f"frame_{i:04d}.png")
        plot_attention_heatmap(attentions, scale, save_path_component, vmin=global_min, vmax=global_max)
        image_paths.append(save_path_component)

    gif_path = os.path.join(save_path, "attention_heatmap.gif")
    with imageio.get_writer(gif_path, mode='I', duration=0.3) as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)

def create_stacked_gif(save_path, y_shift=200):
    accuracy_gif_path = os.path.join(save_path, "accuracy_heatmap.gif")
    attention_gif_path = os.path.join(save_path, "attention_heatmap.gif")
    stacked_gif_path = os.path.join(save_path, "stacked_heatmap.gif")

    accuracy_reader = imageio.get_reader(accuracy_gif_path)
    attention_reader = imageio.get_reader(attention_gif_path)

    accuracy_frames = [Image.fromarray(frame) for frame in accuracy_reader]
    attention_frames = [Image.fromarray(frame) for frame in attention_reader]

    assert len(accuracy_frames) == len(attention_frames), "Mismatch in frame counts between accuracy and attention GIFs"

    stacked_frames = []
    for acc_frame, att_frame in zip(accuracy_frames, attention_frames):
        acc_width, acc_height = acc_frame.size
        att_width, att_height = att_frame.size

        # Create base canvas
        stacked_height = acc_height + att_height - y_shift
        stacked_width = max(acc_width, att_width)

        stacked_frame = Image.new("RGB", (stacked_width, stacked_height), color=(255, 255, 255))

        # Paste attention frame first, shifted up
        stacked_frame.paste(att_frame, (0, 0))  # Paste at top
        stacked_frame.paste(acc_frame, (0, att_height - y_shift))  # Shift accuracy up by overlap

        stacked_frames.append(stacked_frame)

    stacked_frames[0].save(
        stacked_gif_path,
        save_all=True,
        append_images=stacked_frames[1:],
        duration=300,
        loop=0
    )

    save_frames_to_mp4(
        [np.array(fm)[:, :, ::-1] for fm in stacked_frames],
        f"{stacked_gif_path.replace('gif', 'mp4')}",
        fps=15,
        gop_size=1,
        preset="slow"
    )


def plot_accuracy_training(all_accuracies, scale, run_model_spefic_save_dir, args):
    scale=0.5
    seq_indices = range(args.parity_sequence_length)
    fig, ax = plt.subplots(figsize=(scale*10, scale*5))
    cmap = plt.get_cmap("viridis")

    for i, acc in enumerate(all_accuracies):
        color = cmap(i / (len(all_accuracies) - 1))
        ax.plot(seq_indices, acc*100, color=color, alpha=0.7, linewidth=1)

    num_checkpoints = 5
    checkpoint_percentages = np.linspace(0, 100, num_checkpoints)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Training Progress (%)")
    cbar.set_ticks(checkpoint_percentages)
    cbar.set_ticklabels([f"{int(p)}%" for p in checkpoint_percentages])

    ax.set_xlabel("Sequence Index")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks([0, 16 ,32, 48, 63])
    ax.grid(True, alpha=0.5)
    ax.set_xlim(0, args.parity_sequence_length - 1)

    fig.tight_layout(pad=0.1)
    fig.savefig(f"{run_model_spefic_save_dir}/accuracy_vs_seq_element.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{run_model_spefic_save_dir}/accuracy_vs_seq_element.pdf", format='pdf', bbox_inches="tight")
    plt.close(fig)


def plot_loss_all_runs(training_data, evaluate_every, save_path="train_loss_comparison_parity.png", step=1, scale=1.0, x_max=None):
    fig, ax = plt.subplots(figsize=(scale * 10, scale * 5))

    grouped = defaultdict(list)
    label_map = {}
    linestyle_map = {}
    iters_map = {}
    model_map = {}

    for folder, data in training_data.items():
        label, model_type, iters = parse_folder_name(folder)
        if iters is None:
            continue

        key = f"{model_type}_{iters}"
        grouped[key].append(data["train_losses"])
        label_map[key] = f"{model_type}, {iters} Iters."
        linestyle_map[key] = "--" if model_type == "LSTM" else "-"
        iters_map[key] = iters
        model_map[key] = model_type

    unique_iters = sorted(set(iters_map.values()))
    base_colors = sns.color_palette("hls", n_colors=len(unique_iters))
    color_lookup = {iters: base_colors[i] for i, iters in enumerate(unique_iters)}

    legend_entries = []
    global_max_x = 0
    for key in sorted(grouped.keys(), key=lambda k: (iters_map[k], model_map[k])):
        runs = grouped[key]
        if not runs:
            continue

        iters = iters_map[key]
        color = color_lookup[iters]
        linestyle = linestyle_map[key]

        min_len = min(len(r) for r in runs)
        trimmed = np.array([r[:min_len] for r in runs])[:, ::step]

        mean = np.mean(trimmed, axis=0)
        std = np.std(trimmed, axis=0)
        x = np.arange(len(mean)) * step * evaluate_every
        group_max_x = len(mean) * step * evaluate_every
        global_max_x = max(global_max_x, group_max_x)

        line, = ax.plot(x, mean, color=color, linestyle=linestyle, label=label_map[key])
        ax.fill_between(x, mean - std, mean + std, alpha=0.1, color=color)

        legend_entries.append((line, label_map[key]))

    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.5)

    style_legend = [
        Line2D([0], [0], color='black', linestyle='-', label='CTM'),
        Line2D([0], [0], color='black', linestyle='--', label='LSTM')
    ]
    color_legend = [
        Line2D([0], [0], color=color_lookup[it], linestyle='-', label=f"{it} Iters.")
        for it in unique_iters
    ]

    if not x_max:
        x_max = global_max_x

    ax.set_xlim([0, x_max])
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, x_max + 1, 50000))
    ax.legend(handles=color_legend + style_legend, loc="upper left")
    fig.tight_layout(pad=0.1)
    fig.savefig(save_path, dpi=300)
    fig.savefig(save_path.replace("png", "pdf"), format='pdf')
    plt.close(fig)

def plot_accuracy_all_runs(training_data, evaluate_every, save_path="test_accuracy_comparison_parity.png", step=1, scale=1.0, smooth=False, x_max=None):
    fig, ax = plt.subplots(figsize=(scale * 10, scale * 5))

    grouped = defaultdict(list)
    label_map = {}
    linestyle_map = {}
    iters_map = {}
    model_map = {}

    for folder, data in training_data.items():
        label, model_type, iters = parse_folder_name(folder)
        if iters is None:
            continue

        key = f"{model_type}_{iters}"
        grouped[key].append(data["test_accuracies"])
        label_map[key] = f"{model_type}, {iters} Iters."
        linestyle_map[key] = "--" if model_type == "LSTM" else "-"
        iters_map[key] = iters
        model_map[key] = model_type

    unique_iters = sorted(set(iters_map.values()))
    base_colors = sns.color_palette("hls", n_colors=len(unique_iters))
    color_lookup = {iters: base_colors[i] for i, iters in enumerate(unique_iters)}

    legend_entries = []
    global_max_x = 0

    for key in sorted(grouped.keys(), key=lambda k: (iters_map[k], model_map[k])):
        runs = grouped[key]
        if not runs:
            continue

        iters = iters_map[key]
        model = model_map[key]
        color = color_lookup[iters]
        linestyle = linestyle_map[key]

        min_len = min(len(r) for r in runs)
        trimmed = np.array([r[:min_len] for r in runs])[:, ::step]

        mean = np.mean(trimmed, axis=0) * 100
        std = np.std(trimmed, axis=0) * 100

        if smooth:
            window_size = max(1, int(0.05 * len(mean)))
            if window_size % 2 == 0:
                window_size += 1
            kernel = np.ones(window_size) / window_size

            smoothed_mean = np.convolve(mean, kernel, mode='same')
            smoothed_std = np.convolve(std, kernel, mode='same')

            valid_start = window_size // 2
            valid_end = len(mean) - window_size // 2
            valid_length = valid_end - valid_start

            mean = smoothed_mean[valid_start:valid_end]
            std = smoothed_std[valid_start:valid_end]
            x = np.arange(valid_length) * step * evaluate_every
            group_max_x = valid_length * step * evaluate_every
        else:
            x = np.arange(len(mean)) * step * evaluate_every
            group_max_x = len(mean) * step * evaluate_every

        global_max_x = max(global_max_x, group_max_x)

        line, = ax.plot(x, mean, color=color, linestyle=linestyle, label=label_map[key])
        ax.fill_between(x, mean - std, mean + std, alpha=0.1, color=color)
        legend_entries.append((line, label_map[key]))

    if smooth or x_max is None:
        x_max = global_max_x

    ax.set_xlim([0, x_max])
    ax.set_ylim(top=100)
    ax.set_xticks(np.arange(0, x_max + 1, 50000))
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.5)

    style_legend = [
        Line2D([0], [0], color='black', linestyle='-', label='CTM'),
        Line2D([0], [0], color='black', linestyle='--', label='LSTM')
    ]
    color_legend = [
        Line2D([0], [0], color=color_lookup[it], linestyle='-', label=f"{it} Iters.")
        for it in unique_iters
    ]
    ax.legend(handles=color_legend + style_legend, loc="upper left")

    fig.tight_layout(pad=0.1)
    fig.savefig(save_path, dpi=300)
    fig.savefig(save_path.replace("png", "pdf"), format='pdf')
    plt.close(fig)

def extract_run_name(folder, run_index=None):
    # Try to extract from parent folder
    parent = os.path.basename(os.path.dirname(folder))
    match = re.search(r'run(\d+)', parent, re.IGNORECASE)
    if match:
        return f"Run {int(match.group(1))}"
    # Try current folder name
    basename = os.path.basename(folder)
    match = re.search(r'run(\d+)', basename, re.IGNORECASE)
    if match:
        return f"Run {int(match.group(1))}"
    # Fallback: use run index
    if run_index is not None:
        return f"Run {run_index + 1}"
    raise ValueError(f"Could not extract run number from: {folder}")

def plot_loss_individual_runs(training_data, evaluate_every, save_dir, scale=1.0, x_max=None):

    grouped = defaultdict(list)
    label_map = {}
    iters_map = {}
    model_map = {}

    base_colors = sns.color_palette("hls", n_colors=3)
    color_lookup = {f"Run {i+1}": base_colors[i] for i in range(3)}

    for i, (folder, data) in enumerate(training_data.items()):
        checkpoint = load_checkpoint(get_latest_checkpoint_file(folder), device="cpu")
        model_args = get_model_args_from_checkpoint(checkpoint)
        label, model_type, iters = parse_folder_name(folder)
        if iters is None:
            continue

        if model_type.lower() == "ctm":
            memory_length = getattr(model_args, "memory_length", None)
            if memory_length is None:
                raise ValueError(f"CTM model missing memory_length in checkpoint args from: {folder}")
            key = f"{model_type}_{iters}_{memory_length}".lower()
        else:
            key = f"{model_type}_{iters}".lower()

        run_name = extract_run_name(folder, run_index=i)
        grouped[key].append((run_name, data["train_losses"]))
        label_map[key] = f"{model_type}, {iters} Iters."
        iters_map[key] = iters
        model_map[key] = model_type

    for key, runs in grouped.items():
        fig, ax = plt.subplots(figsize=(scale * 10, scale * 5))
        for run_name, losses in runs:
            x = np.arange(len(losses)) * evaluate_every
            color = color_lookup.get(run_name, 'gray')
            ax.plot(x, losses, label=run_name, color=color, alpha=0.7)

        ax.set_xlabel("Training Iterations")
        ax.set_ylabel("Loss")
        ax.set_ylim(bottom=-0.01)
        ax.grid(True, alpha=0.5)
        if x_max:
            ax.set_xlim([0, x_max])
            ax.set_xticks(np.arange(0, x_max + 1, 50000))
        ax.legend()
        fig.tight_layout(pad=0.1)

        subdir = os.path.join(save_dir, key)
        os.makedirs(subdir, exist_ok=True)
        fname = os.path.join(subdir, f"individual_runs_loss_{key}.png")
        fig.savefig(fname, dpi=300)
        fig.savefig(fname.replace("png", "pdf"), format="pdf")
        plt.close(fig)

def plot_accuracy_individual_runs(training_data, evaluate_every, save_dir, scale=1.0, smooth=False, x_max=None):

    grouped = defaultdict(list)
    label_map = {}
    iters_map = {}
    model_map = {}

    base_colors = sns.color_palette("hls", n_colors=3)
    color_lookup = {f"Run {i+1}": base_colors[i] for i in range(3)}

    for i, (folder, data) in enumerate(training_data.items()):
        checkpoint = load_checkpoint(get_latest_checkpoint_file(folder), device="cpu")
        model_args = get_model_args_from_checkpoint(checkpoint)
        label, model_type, iters = parse_folder_name(folder)
        if iters is None:
            continue

        if model_type.lower() == "ctm":
            memory_length = getattr(model_args, "memory_length", None)
            if memory_length is None:
                raise ValueError(f"CTM model missing memory_length in checkpoint args from: {folder}")
            key = f"{model_type}_{iters}_{memory_length}".lower()
        else:
            key = f"{model_type}_{iters}".lower()

        run_name = extract_run_name(folder, run_index=i)
        grouped[key].append((run_name, data["test_accuracies"]))
        label_map[key] = f"{model_type}, {iters} Iters."
        iters_map[key] = iters
        model_map[key] = model_type

    for key, runs in grouped.items():
        fig, ax = plt.subplots(figsize=(scale * 10, scale * 5))
        for run_name, acc in runs:
            acc = np.array(acc) * 100
            if smooth:
                window_size = max(1, int(0.05 * len(acc)))
                if window_size % 2 == 0:
                    window_size += 1
                kernel = np.ones(window_size) / window_size
                acc = np.convolve(acc, kernel, mode="same")

            x = np.arange(len(acc)) * evaluate_every
            color = color_lookup.get(run_name, 'gray')
            ax.plot(x, acc, label=run_name, color=color, alpha=0.7)

        ax.set_xlabel("Training Iterations")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim([50, 101])
        ax.grid(True, alpha=0.5)
        if x_max:
            ax.set_xlim([0, x_max])
            ax.set_xticks(np.arange(0, x_max + 1, 50000))
        ax.legend()
        fig.tight_layout(pad=0.1)

        subdir = os.path.join(save_dir, key)
        os.makedirs(subdir, exist_ok=True)
        fname = os.path.join(subdir, f"individual_runs_accuracy_{key}.png")
        fig.savefig(fname, dpi=300)
        fig.savefig(fname.replace("png", "pdf"), format="pdf")
        plt.close(fig)

def plot_training_curve_all_runs(all_folders, save_dir, scale, device, smooth=False, x_max=None, plot_individual_runs=True):

    all_folders = [folder for folder in all_folders if "certain" not in folder]

    training_data = {}
    evaluation_intervals = []
    for folder in all_folders:
        latest_checkpoint_path = get_latest_checkpoint_file(folder)
        if latest_checkpoint_path:
            checkpoint = load_checkpoint(latest_checkpoint_path, device=device)
            model_args = get_model_args_from_checkpoint(checkpoint)
            evaluation_intervals.append(model_args.track_every)

            _, train_losses, test_losses, train_accuracies, test_accuracies = get_accuracy_and_loss_from_checkpoint(checkpoint, device=device)
            training_data[folder] = {
                "train_losses": train_losses,
                "test_losses": test_losses,
                "train_accuracies": train_accuracies,
                "test_accuracies": test_accuracies
            }
        else:
            print(f"No checkpoint found for {folder}")

    assert len(evaluation_intervals) > 0, "No valid checkpoints found."
    assert all(interval == evaluation_intervals[0] for interval in evaluation_intervals), "Evaluation intervals are not consistent across runs."
    
    evaluate_every = evaluation_intervals[0]

    if plot_individual_runs:
        plot_loss_individual_runs(training_data, evaluate_every, save_dir=save_dir, scale=scale, x_max=x_max)
        plot_accuracy_individual_runs(training_data, evaluate_every, save_dir=save_dir, scale=scale, smooth=smooth, x_max=x_max)

    plot_loss_all_runs(training_data, evaluate_every, save_path=f"{save_dir}/loss_comparison.png", scale=scale, x_max=x_max)
    plot_accuracy_all_runs(training_data, evaluate_every, save_path=f"{save_dir}/accuracy_comparison.png", scale=scale, smooth=smooth, x_max=x_max)

    return training_data

def plot_accuracy_thinking_time(csv_path, scale, output_dir="analysis/cifar"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["RunName"] = df["Run"].apply(lambda x: os.path.basename(os.path.dirname(x)))
    df["Model"] = df["Run"].apply(lambda x: "CTM" if "ctm" in x.lower() else "LSTM")

    grouped = df.groupby(["Model", "Num Iterations"])
    summary = grouped.agg(
        mean_accuracy=("Overall Mean Accuracy", "mean"),
        std_accuracy=("Overall Std Accuracy", lambda x: np.sqrt(np.mean(x**2)))
    ).reset_index()

    summary["mean_accuracy"] *= 100
    summary["std_accuracy"] *= 100

    fig, ax = plt.subplots(figsize=(scale*5, scale*5))

    for model in ("CTM", "LSTM"):
        subset = summary[summary["Model"] == model].sort_values(by="Num Iterations")
        linestyle = "-" if model == "CTM" else "--"
        ax.errorbar(
            subset["Num Iterations"],
            subset["mean_accuracy"],
            yerr=subset["std_accuracy"],
            linestyle=linestyle,
            color="black",
            marker='.',
            label=model,
            capsize=3,
            elinewidth=1,
            errorevery=1
        )

    ax.set_xlabel("Internal Ticks")
    ax.set_ylabel("Accuracy (%)")
    custom_lines = [
        Line2D([0], [0], color='black', linestyle='-', label='CTM'),
        Line2D([0], [0], color='black', linestyle='--', label='LSTM')
    ]
    ax.legend(handles=custom_lines, loc="lower right")
    ax.grid(True, alpha=0.5)

    os.makedirs(output_dir, exist_ok=True)
    output_path_png = os.path.join(output_dir, "accuracy_vs_thinking_time.png")
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path_png, dpi=300)
    fig.savefig(output_path_png.replace("png", "pdf"), format='pdf')
    plt.close(fig)


def plot_lstm_last_and_certain_accuracy(all_folders, save_path="lstm_last_and_certain_accuracy.png", scale=1.0, step=1, x_max=None):

    tags = ["lstm_10", "lstm_10_certain", "lstm_25", "lstm_25_certain"]
    folders = [f for f in all_folders if any(tag in f.lower() for tag in tags)]

    training_data, eval_intervals = {}, []
    for f in folders:
        cp = get_latest_checkpoint_file(f)
        if not cp:
            print(f"⚠️ No checkpoint in {f}")
            continue
        ckpt = load_checkpoint(cp, device="cpu")
        args = get_model_args_from_checkpoint(ckpt)
        eval_intervals.append(args.track_every)
        _, _, _, _, acc = get_accuracy_and_loss_from_checkpoint(ckpt)
        iters = "25" if "25" in f.lower() else "10"
        label = "Certain" if "certain" in f.lower() else "Final"
        training_data.setdefault((iters, label), []).append(acc)

    assert training_data and all(i == eval_intervals[0] for i in eval_intervals), "Missing or inconsistent eval intervals."
    evaluate_every = eval_intervals[0]

    keys = sorted(training_data.keys())
    colors = sns.color_palette("hls", n_colors=len(keys))
    style_map = {key: ("--" if key[1] == "Certain" else "-") for key in keys}
    color_map = {key: colors[i] for i, key in enumerate(keys)}

    fig, ax = plt.subplots(figsize=(scale * 10, scale * 5))
    max_x = 0

    for key in keys:
        runs = training_data[key]
        min_len = min(len(r) for r in runs)
        trimmed = np.stack([r[:min_len] for r in runs], axis=0)[:, ::step]
        mean, std = np.mean(trimmed, 0) * 100, np.std(trimmed, 0) * 100
        x = np.arange(len(mean)) * step * evaluate_every
        ax.plot(x, mean, color=color_map[key], linestyle=style_map[key],
                label=f"{key[0]} Iters, {key[1]}", linewidth=2, alpha=0.7)
        ax.fill_between(x, mean - std, mean + std, color=color_map[key], alpha=0.1)
        max_x = max(max_x, x[-1])

    ax.set_xlim([0, x_max or max_x])
    ax.set_xticks(np.arange(0, (x_max or max_x) + 1, 50000))
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.1)
    fig.savefig(save_path, dpi=300)
    fig.savefig(save_path.replace("png", "pdf"), format="pdf")
    plt.close(fig)
