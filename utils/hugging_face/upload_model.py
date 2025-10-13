import torch
from PIL import Image
from models.ctm import ContinuousThoughtMachine as CTM

# Load checkpoint
CHECKPOINT_PATH = "imagenet/ctm_imagenet_D=4096_T=50_M=25.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model_args = checkpoint['args']

# Create model
model = CTM(
    iterations=model_args.iterations,
    d_model=model_args.d_model,
    d_input=model_args.d_input,
    heads=model_args.heads,
    n_synch_out=model_args.n_synch_out,
    n_synch_action=model_args.n_synch_action,
    synapse_depth=model_args.synapse_depth,
    memory_length=model_args.memory_length,
    deep_nlms=model_args.deep_memory,
    memory_hidden_dims=model_args.memory_hidden_dims,
    do_layernorm_nlm=model_args.do_normalisation,
    backbone_type=model_args.backbone_type,
    positional_embedding_type=model_args.positional_embedding_type,
    out_dims=model_args.out_dims,
    prediction_reshaper=[-1],
    dropout=0,
    neuron_select_type=model_args.neuron_select_type,
    n_random_pairing_self=model_args.n_random_pairing_self,
).to(device)

model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

print("Initializing lazy layers...")
dummy_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    _ = model(dummy_input, track=False)

# Upload to Hub
print("Uploading to Hub...")
model.push_to_hub("ciaran-regan-ie/continuous-thought-machines", private=True)
print("Upload complete!")