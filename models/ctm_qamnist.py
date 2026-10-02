import torch
import numpy as np
from models.ctm import ContinuousThoughtMachine
from models.modules import MNISTBackbone, QAMNISTIndexEmbeddings, QAMNISTOperatorEmbeddings

class ContinuousThoughtMachineQAMNIST(ContinuousThoughtMachine):
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
                 out_dims,
                 iterations_per_digit,
                 iterations_per_question_part,
                 iterations_for_answering,
                 prediction_reshaper=[-1],
                 dropout=0,
                 neuron_select_type='first-last',
                 n_random_pairing_self=256
                 ):
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=heads,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=dropout,
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=n_random_pairing_self,
            backbone_type='none',
            positional_embedding_type='none',
        )

        # --- Core Parameters ---
        self.iterations_per_digit = iterations_per_digit
        self.iterations_per_question_part = iterations_per_question_part
        self.iterations_for_answering = iterations_for_answering

    # --- Setup Methods ---

    def set_initial_rgb(self):
        """Set the initial RGB values for the backbone."""
        return None

    def get_d_backbone(self):
        """Get the dimensionality of the backbone output."""
        return self.d_input

    def set_backbone(self):
        """Set the backbone module based on the specified type."""
        self.backbone_digit = MNISTBackbone(self.d_input)
        self.index_backbone = QAMNISTIndexEmbeddings(50, self.d_input)
        self.operator_backbone = QAMNISTOperatorEmbeddings(2, self.d_input)
        pass

    # --- Utilty Methods ---

    def determine_step_type(self, total_iterations_for_digits, total_iterations_for_question, stepi: int):
        """Determine whether the current step is for digits, questions, or answers."""
        is_digit_step = stepi < total_iterations_for_digits
        is_question_step = total_iterations_for_digits <= stepi < total_iterations_for_digits + total_iterations_for_question
        is_answer_step = stepi >= total_iterations_for_digits + total_iterations_for_question
        return is_digit_step, is_question_step, is_answer_step

    def determine_index_operator_step_type(self, total_iterations_for_digits, stepi: int):
        """Determine whether the current step is for index or operator."""
        step_within_questions = stepi - total_iterations_for_digits
        if step_within_questions % (2 * self.iterations_per_question_part) < self.iterations_per_question_part:
            is_index_step = True
            is_operator_step = False
        else:
            is_index_step = False
            is_operator_step = True
        return is_index_step, is_operator_step

    def get_kv_for_step(self, total_iterations_for_digits, total_iterations_for_question, stepi, x, z, prev_input=None, prev_kv=None):
        """Get the key-value for the current step."""
        is_digit_step, is_question_step, is_answer_step = self.determine_step_type(total_iterations_for_digits, total_iterations_for_question, stepi)

        if is_digit_step:
            current_input = x[:, stepi]
            if prev_input is not None and torch.equal(current_input, prev_input):
                return prev_kv, prev_input
            kv = self.kv_proj(self.backbone_digit(current_input).flatten(2).permute(0, 2, 1))

        elif is_question_step:
            offset = stepi - total_iterations_for_digits
            current_input = z[:, offset]
            if prev_input is not None and torch.equal(current_input, prev_input):
                return prev_kv, prev_input
            is_index_step, is_operator_step = self.determine_index_operator_step_type(total_iterations_for_digits, stepi)
            if is_index_step:
                kv = self.index_backbone(current_input)
            elif is_operator_step:
                kv = self.operator_backbone(current_input)
            else:
                raise ValueError("Invalid step type for question processing.")

        elif is_answer_step:
            current_input = None
            kv = torch.zeros((x.size(0), self.d_input), device=x.device)

        else:
            raise ValueError("Invalid step type.")

        return kv, current_input




    def forward(self, x, z, track=False):
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        attention_tracking = []
        embedding_tracking = []

        total_iterations_for_digits = x.size(1)
        total_iterations_for_question = z.size(1)
        total_iterations = total_iterations_for_digits + total_iterations_for_question + self.iterations_for_answering

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1) # Shape: (B, H, T)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1) # Shape: (B, H)

        # --- Storage for outputs per iteration ---
        predictions = torch.empty(B, self.out_dims, total_iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, total_iterations, device=device, dtype=x.dtype)

        # --- Initialise Recurrent Synch Values  ---
        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)  # Fix from github user: kuviki
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')

        prev_input = None
        prev_kv = None

        # --- Recurrent Loop  ---
        for stepi in range(total_iterations):
            is_digit_step, is_question_step, is_answer_step = self.determine_step_type(total_iterations_for_digits, total_iterations_for_question, stepi)

            kv, prev_input = self.get_kv_for_step(total_iterations_for_digits, total_iterations_for_question, stepi, x, z, prev_input, prev_kv)
            prev_kv = kv

            synchronization_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')

            # --- Interact with Data via Attention ---
            attn_weights = None
            if is_digit_step:
                q = self.q_proj(synchronization_action).unsqueeze(1)
                attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=track)
                attn_out = attn_out.squeeze(1)
                pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)
            else:
                kv = kv.squeeze(1)
                pre_synapse_input = torch.concatenate((kv, activated_state), dim=-1)

            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply NLMs ---
            activated_state = self.trace_processor(state_trace)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronization_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronization_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                if attn_weights is not None:
                    attention_tracking.append(attn_weights.detach().cpu().numpy())
                if is_question_step:
                    embedding_tracking.append(kv.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            return predictions, certainties, synchronization_out, np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking), np.array(embedding_tracking)
        return predictions, certainties, synchronization_out