import torch
import numpy as np
from models.ctm import ContinuousThoughtMachine

class ContinuousThoughtMachineSORT(ContinuousThoughtMachine):
    """
    Slight adaption of the CTM to work with the sort task.
    """                               

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 backbone_type,
                 positional_embedding_type,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 neuron_select_type='random-pairing',  
                 n_random_pairing_self=0,
                 ):
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=0,
            n_synch_out=n_synch_out,
            n_synch_action=0,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            backbone_type='none',
            positional_embedding_type='none',
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=dropout,
            dropout_nlm=dropout_nlm,
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=n_random_pairing_self,
        )

        # --- Use a minimal CTM w/out input (action) synch ---
        self.neuron_select_type_action = None
        self.synch_representation_size_action = None

        self.attention = None  # Should already be None because super(... heads=0... ) 
        self.q_proj = None  # Should already be None because super(... heads=0... ) 
        self.kv_proj = None  # Should already be None because super(... heads=0... ) 




    def forward(self, x, track=False, early_exit_threshold=None, return_exit_steps=False):
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        attention_tracking = []

        # --- For SORT: no need to featurise data ---
        

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1) # Shape: (B, H, T)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1) # Shape: (B, H)

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)

        # --- Initialise Recurrent Synch Values  ---
        use_early_exit = early_exit_threshold is not None
        if use_early_exit or return_exit_steps:
            finished_mask = torch.zeros(B, dtype=torch.bool, device=device)
            exit_steps = torch.full((B,), self.iterations, dtype=torch.long, device=device)

        r_out = torch.exp(-torch.clamp(self.decay_params_out, 0, 15)).unsqueeze(0).repeat(B, 1)
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')
        # Compute learned weighting for synchronisation
        

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):

            pre_synapse_input = torch.concatenate((x, activated_state), dim=-1)

            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input)
            # The 'state_trace' is the history of incoming pre-activations
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply Neuron-Level Models ---
            activated_state = self.trace_processor(state_trace)
            # One would also keep an 'activated_state_trace' as the history of outgoing post-activations
            # BUT, this is unnecessary because the synchronisation calculation is fully linear and can be
            # done using only the currect activated state (see compute_synchronisation method for explanation)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Adaptive Computation Time (ACT) / Early Exit Logic ---
            if use_early_exit:
                # 1. Identify samples that just reached the required certainty threshold.
                is_certain = current_certainty[:, 1] >= early_exit_threshold
                newly_finished = is_certain & (~finished_mask)

                # Record the step they finished at (1-indexed for step count).
                exit_steps[newly_finished] = stepi + 1

                # Update the global tracking mask for the batch.
                finished_mask = finished_mask | is_certain

                # 2. Freeze output history for samples that finished on earlier ticks.
                # Recurrent state is still computed for the full batch until all samples finish.
                if stepi > 0:
                    already_finished = finished_mask ^ newly_finished
                    if already_finished.any():
                        predictions[already_finished, ..., stepi] = predictions[already_finished, ..., stepi - 1]
                        certainties[already_finished, ..., stepi] = certainties[already_finished, ..., stepi - 1]

                # 3. Break the recurrent loop once the whole batch has finished.
                if finished_mask.all():
                    for pad_step in range(stepi + 1, self.iterations):
                        predictions[..., pad_step] = predictions[..., stepi]
                        certainties[..., pad_step] = certainties[..., stepi]
                    break


            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            tracking_outputs = (
                predictions,
                certainties,
                np.array(synch_out_tracking),
                np.array(pre_activations_tracking),
                np.array(post_activations_tracking),
                np.array(attention_tracking),
            )
            if return_exit_steps:
                return (*tracking_outputs, exit_steps)
            return tracking_outputs
        if return_exit_steps:
            return predictions, certainties, synchronisation_out, exit_steps
        return predictions, certainties, synchronisation_out
