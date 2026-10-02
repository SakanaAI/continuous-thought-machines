import torch.nn as nn
import torch
import numpy as np
import math

from models.modules import ParityBackbone, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)

class LSTMBaseline(nn.Module):
    """
    LSTM Baseline

    Args:
        iterations (int): Number of internal 'thought' steps (T, in paper).
        d_model (int): Core dimensionality of the latent space.
        d_input (int): Dimensionality of projected attention outputs or direct input features.
        heads (int): Number of attention heads.
        backbone_type (str): Type of feature extraction backbone (e.g., 'resnet18-2', 'none').
        positional_embedding_type (str): Type of positional embedding for backbone features.
        out_dims (int): Dimensionality of the final output projection.
        prediction_reshaper (list): Shape for reshaping predictions before certainty calculation (task-specific).
        dropout (float): Dropout rate.
    """

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 backbone_type,
                 num_layers,
                 positional_embedding_type,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 ):
        super(LSTMBaseline, self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.prediction_reshaper = prediction_reshaper
        self.backbone_type = backbone_type
        self.positional_embedding_type = positional_embedding_type
        self.out_dims = out_dims

        # --- Assertions ---
        self.verify_args()

        # --- Input Processing  ---
        d_backbone = self.get_d_backbone()

        self.set_initial_rgb()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)
        self.kv_proj = self.get_kv_proj()
        self.lstm = nn.LSTM(d_input, d_model, num_layers, batch_first=True, dropout=dropout)
        self.q_proj = self.get_q_proj()
        self.attention = self.get_attention(heads, dropout)
        self.output_projector = nn.Sequential(nn.LazyLinear(out_dims))

        #  --- Start States ---
        self.register_parameter('start_hidden_state', nn.Parameter(torch.zeros((num_layers, d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model))), requires_grad=True))
        self.register_parameter('start_cell_state', nn.Parameter(torch.zeros((num_layers, d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model))), requires_grad=True))



    # --- Core LSTM Methods ---

    def compute_features(self, x):
        """Applies backbone and positional embedding to input."""
        x = self.initial_rgb(x)
        self.kv_features = self.backbone(x)
        pos_emb = self.positional_embedding(self.kv_features)
        combined_features = (self.kv_features + pos_emb).flatten(2).transpose(1, 2)
        kv = self.kv_proj(combined_features)
        return kv

    def compute_certainty(self, current_prediction):
        """Compute the certainty of the current prediction."""
        B = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([B] +self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    # --- Setup Methods ---

    def set_initial_rgb(self):
        """Set the initial RGB processing module based on the backbone type."""
        if 'resnet' in self.backbone_type:
            self.initial_rgb = nn.LazyConv2d(3, 1, 1) # Adapts input channels lazily
        else:
            self.initial_rgb = nn.Identity()

    def get_d_backbone(self):
        """
        Get the dimensionality of the backbone output, to be used for positional embedding setup.

        This is a little bit complicated for resnets, but the logic should be easy enough to read below.        
        """
        if self.backbone_type == 'shallow-wide':
            return 2048
        elif self.backbone_type == 'parity_backbone':
            return self.d_input
        elif 'resnet' in self.backbone_type:
            if '18' in self.backbone_type or '34' in self.backbone_type: 
                if self.backbone_type.split('-')[1]=='1': return 64
                elif self.backbone_type.split('-')[1]=='2': return 128
                elif self.backbone_type.split('-')[1]=='3': return 256
                elif self.backbone_type.split('-')[1]=='4': return 512
                else:
                    raise NotImplementedError
            else:
                if self.backbone_type.split('-')[1]=='1': return 256
                elif self.backbone_type.split('-')[1]=='2': return 512
                elif self.backbone_type.split('-')[1]=='3': return 1024
                elif self.backbone_type.split('-')[1]=='4': return 2048
                else:
                    raise NotImplementedError
        elif self.backbone_type == 'none':
            return None
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def set_backbone(self):
        """Set the backbone module based on the specified type."""
        if self.backbone_type == 'shallow-wide':
            self.backbone = ShallowWide()
        elif self.backbone_type == 'parity_backbone':
            d_backbone = self.get_d_backbone()
            self.backbone = ParityBackbone(n_embeddings=2, d_embedding=d_backbone)
        elif 'resnet' in self.backbone_type:
            self.backbone = prepare_resnet_backbone(self.backbone_type)
        elif self.backbone_type == 'none':
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_positional_embedding(self, d_backbone):
        """Get the positional embedding module."""
        if self.positional_embedding_type == 'learnable-fourier':
            return LearnableFourierPositionalEncoding(d_backbone, gamma=1 / 2.5)
        elif self.positional_embedding_type == 'multi-learnable-fourier':
            return MultiLearnableFourierPositionalEncoding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational':
            return CustomRotationalEmbedding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational-1d':
            return CustomRotationalEmbedding1D(d_backbone)
        elif self.positional_embedding_type == 'none':
            return lambda x: 0  # Default no-op
        else:
            raise ValueError(f"Invalid positional_embedding_type: {self.positional_embedding_type}")

    def get_attention(self, heads, dropout):
        """Get the attention module."""
        return nn.MultiheadAttention(self.d_input, heads, dropout, batch_first=True)

    def get_kv_proj(self):
        """Get the key-value projection module."""
        return nn.Sequential(nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input))

    def get_q_proj(self):
        """Get the query projection module."""
        return nn.LazyLinear(self.d_input)


    def verify_args(self):
        """Verify the validity of the input arguments."""

        assert self.backbone_type in VALID_BACKBONE_TYPES + ['none'], \
            f"Invalid backbone_type: {self.backbone_type}"
        
        assert self.positional_embedding_type in VALID_POSITIONAL_EMBEDDING_TYPES + ['none'], \
            f"Invalid positional_embedding_type: {self.positional_embedding_type}"
        
        if self.backbone_type=='none' and self.positional_embedding_type!='none':
            raise AssertionError("There should be no positional embedding if there is no backbone.")

        pass




    def forward(self, x, track=False):
        """
        Forward pass - Reverted to structure closer to user's working version.
        Executes T=iterations steps.
        """
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        activations_tracking = []
        attention_tracking = []

        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Initialise Recurrent State ---
        hn = torch.repeat_interleave(self.start_hidden_state.unsqueeze(1), x.size(0), 1)
        cn = torch.repeat_interleave(self.start_cell_state.unsqueeze(1), x.size(0), 1)
        state_trace = [hn[-1]]

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):

            # --- Interact with Data via Attention ---
            q = self.q_proj(hn[-1].unsqueeze(1))
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=track)
            lstm_input = attn_out

            # --- Apply LSTM ---
            hidden_state, (hn,cn) = self.lstm(lstm_input, (hn, cn))
            hidden_state = hidden_state.squeeze(1)
            state_trace.append(hidden_state)

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(hidden_state)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                activations_tracking.append(hidden_state.squeeze(1).detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            return predictions, certainties, None, np.zeros_like(activations_tracking), np.array(activations_tracking), np.array(attention_tracking)
        return predictions, certainties, None