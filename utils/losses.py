import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ctc_loss(predictions, targets, blank_label=0):
    """
    Computes the Connectionist Temporal Classification (CTC) loss.

    Args:
        predictions: A tensor of shape [B, C, L] representing the logits of the 
                     predicted sequences.  B is the batch size, C is the number
                     of classes (including the blank label), and L is the sequence
                     length of the predictions.
        targets: A tensor of shape [B, T] representing the target sequences.
                 B is the batch size and T is the target sequence length.
                 Note that T can vary within the batch.
        blank_label: The index of the blank label.  Defaults to 0.

    Returns:
        The CTC loss (a scalar tensor).
    """

    batch_size, num_classes, prediction_length = predictions.shape
    _, target_length = targets.shape

    # 1. Log softmax on predictions:  Crucially, CTC loss requires log probabilities.
    log_probs = F.log_softmax(predictions, dim=1)  # Shape: [B, C, L]

    # 2.  Prepare inputs for torch.nn.CTCLoss:
    #    a.  Convert log_probs to shape (L, B, C):  CTCLoss expects time first.
    log_probs = log_probs.permute(2, 0, 1)  # Shape: [L, B, C]

    #    b.  Get lengths of the predicted sequences (all L in this case).
    input_lengths = torch.full(size=(batch_size,), fill_value=prediction_length, dtype=torch.long)

    #    c.  Get lengths of the target sequences.
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long) # Handle variable target lengths

    # 3. Create the CTCLoss criterion.  `blank=blank_label` is essential!
    ctc_loss = torch.nn.CTCLoss(blank=blank_label, reduction='mean') # 'mean' for averaging over the batch

    # 4. Calculate the loss.  `targets` needs to be a concatenated tensor.
    #    We handle padding by only passing the valid lengths to CTCLoss.
    concatenated_targets = torch.cat(list(targets)) # Concatenate targets

    loss = ctc_loss(log_probs, concatenated_targets, input_lengths, target_lengths)

    return loss

def sort_loss(predictions, targets):
    """
    The sort task was used partly to show that ctc loss can work.
    """
    loss = compute_ctc_loss(predictions, targets, blank_label=predictions.shape[1]-1)
    return loss

def image_classification_loss(predictions, certainties, targets, use_most_certain=True):
    """
    Computes the maze loss with auto-extending cirriculum.

    Predictions are of shape: (B, class, internal_ticks),
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B]

    use_most_certain will select either the most certain point or the final point. 
    """
    targets_expanded = torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1)
    # Losses are of shape [B, internal_ticks]
    losses = nn.CrossEntropyLoss(reduction='none')(predictions, targets_expanded)
        
    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:  # Revert to final loss if set
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected)/2
    return loss, loss_index_2

def maze_loss(predictions, certainties, targets, cirriculum_lookahead=5, use_most_certain=True):
    """
    Computes the maze loss with auto-extending cirriculum.

    Predictions are of shape: (B, route_length, class, internal_ticks),
        where classes are in [0,1,2,3,4] for [Up, Down, Left, Right, Wait]
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B, route_length]

    cirriculum_lookahead: how far to look ahead in the auto-cirriculum

    use_most_certain will select either the most certain point or the final point. For baselines,
        the final point proved the only usable option. 
    
    """
    # Predictions reshaped to: [B*route_length, 5, internal_ticks]
    predictions_reshaped = predictions.flatten(0,1)
    # Targets reshaped to: [B*route_length, internal_ticks]
    targets_reshaped = torch.repeat_interleave(targets.unsqueeze(-1), 
                                               predictions.size(-1), -1).flatten(0,1).long()
    
    # Losses are of shape [B, route_length, internal_ticks]
    losses = nn.CrossEntropyLoss(reduction='none')(predictions_reshaped, targets_reshaped)
    losses = losses.reshape(predictions[:,:,0].shape)
    
    # Below is the code for auto-cirriculum
    # Find where correct, and make sure to always push +5 beyond that
    iscorrects = (predictions.argmax(2) == targets.unsqueeze(-1)).cumsum(1)
    correct_mask = (iscorrects == torch.arange(1, iscorrects.size(1)+1, device=iscorrects.device).reshape(1, -1, 1))
    correct_mask[:,0,:] = 1
    prefix_argmax = correct_mask.cumsum(1).argmax(1)  # (B, T)
    upto_where = prefix_argmax.max(dim=1)[0] + cirriculum_lookahead  # (B,)
    upto_where = upto_where.clamp(max=losses.size(1))
    r_idx = torch.arange(losses.size(1), device=losses.device)
    mask_2d = (r_idx[None, :] < upto_where[:, None]).float()  # (B, R)
    loss_mask = mask_2d[:,:,None].expand_as(losses)  # (B, R, T)

    # Reduce losses along route dimension
    # Will now be of shape [B, internal_ticks]
    losses = (losses * loss_mask).sum(1)/(loss_mask.sum(1))

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1]
    loss_selected = losses[batch_indexer, loss_index_2]

    loss = ((loss_minimum_ce + loss_selected)/2).mean()
    return loss, loss_index_2, upto_where.detach().cpu().numpy()

def parity_loss(predictions, certainties, targets, use_most_certain=True):
    """
    Computes the parity loss.

    Predictions are of shape: (B, parity_sequence_length, class, internal_ticks),
        where classes are in [0,1,2,3,4] for [Up, Down, Left, Right, Wait]
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B, parity_sequence_length]

    use_most_certain will select either the most certain point or the final point. For baselines,
        the final point proved the only usable option. 
    """

    # Losses are of shape [B, parity_sequence_length, internal_ticks]
    losses = nn.CrossEntropyLoss(reduction='none')(predictions.flatten(0,1), 
                                                   torch.repeat_interleave(targets.unsqueeze(-1), 
                                                                           predictions.size(-1), -1).flatten(0,1).long()).reshape(predictions[:,:,0].shape)

    # Average the loss over the parity sequenece dimension
    losses = losses.mean(1)

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected)/2
    return loss, loss_index_2


def qamnist_loss(predictions, certainties, targets, use_most_certain=True):
    """
    Computes the qamnist loss over the last num_answer_steps steps.

    Predictions are of shape: (B, class, internal_ticks),
    Certainties are of shape: (B, 2, internal_ticks),
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B]
    num_answer_steps: number of steps to consider for the loss

    use_most_certain will select either the most certain point or the final point.
    """

    losses = nn.CrossEntropyLoss(reduction='none')(predictions,
                                                   torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1))

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1

    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected)/2
    return loss, loss_index_2


def certainty_weighted_loss(loss_fn, predictions, certainties, targets,
                           uncertainty_threshold=0.3, confidence_boost=2.0):
    """
    Universal wrapper that applies automatic certainty-based weighting to ANY loss function.

    This function implements the theoretical correction for the fundamental flaw in standard
    neural network training that treats all training examples equally, violating core principles
    from information theory, statistical learning theory, and optimization theory.

    The certainty-weighted approach addresses:
    - Information content variation in learning signals
    - Bias-variance tradeoff violations
    - Gradient quality optimization
    - Optimization landscape smoothing
    - Empirical risk minimization enhancement

    Args:
        loss_fn (callable): Any existing loss function that returns (loss, selection_index)
        predictions (torch.Tensor): Model predictions of shape [B, C, T] or similar
        certainties (torch.Tensor): Certainty metrics of shape [B, 2, T] where:
            - certainties[:, 0] = normalized entropy (uncertainty measure)
            - certainties[:, 1] = confidence measure (1 - normalized entropy)
        targets (torch.Tensor): Ground truth targets
        uncertainty_threshold (float): Below this uncertainty level, reduce learning weight (default: 0.3)
        confidence_boost (float): How much to boost confident predictions (default: 2.0)

    Returns:
        tuple: (weighted_loss, selection_index)
            - weighted_loss (torch.Tensor): Scalar tensor with certainty-weighted loss
            - selection_index (torch.Tensor): Index used for selection (from underlying loss_fn)

    The weighting strategy:
    1. Boosts confident predictions: weights *= (1 + confidence_boost * (confidence - 0.5))
    2. Reduces uncertain predictions: weights *= clamp(uncertainty_threshold / (uncertainty + epsilon), 0, 1)
    3. Applies weights to base loss: weighted_loss = base_loss * weights

    This implements the "Learning should be proportional to certainty" principle that emerges
    from information theory, statistical learning theory, and cognitive science.
    """
    # Calculate adaptive weights based on certainty metrics
    confidence = certainties[:, 1]  # High confidence values
    uncertainty = certainties[:, 0]  # Normalized entropy

    # Smart weighting: reduce weight for uncertain, boost confident
    weights = torch.ones_like(confidence)
    weights = weights * (1 + confidence_boost * (confidence - 0.5))  # Boost confident predictions
    weights = weights * torch.clamp(uncertainty_threshold / (uncertainty + 1e-8), 0, 1)  # Reduce uncertain predictions

    # Apply underlying loss function
    base_loss, selection_index = loss_fn(predictions, certainties, targets)

    # Calculate weighted loss per sample
    # Handle different tensor shapes for weights and base_loss
    if hasattr(weights, 'mean') and weights.dim() > 1:
        sample_losses = base_loss * weights.mean(dim=-1)
    else:
        sample_losses = base_loss * weights

    # Return mean of weighted losses and original selection index
    return sample_losses.mean(), selection_index