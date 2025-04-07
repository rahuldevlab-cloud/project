# File: models/framework_gnn.py
# (Modified: Removed inplace=True from ReLUs)

import torch
import torch.nn as nn
from .networks.utils_weights import weights_init
from .networks.gnn import GCNLayer
from copy import copy
import numpy as np # Import numpy if inferring pad from sensing_range


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.S = None
        self.num_agents = self.config["num_agents"]

        # --- FOV shape used by CNN ---
        pad = self.config.get("pad")
        if pad is None:
            sensing_range = self.config.get("sensing_range")
            if sensing_range is not None:
                pad = int(np.ceil(sensing_range)) + 1
                print(f"Framework WARNING: 'pad' not in config, inferring pad={pad} from sensing_range={sensing_range}.")
            else:
                pad = 3
                print(f"Framework WARNING: 'pad' and 'sensing_range' not in config, using default pad={pad}.")
        self.fov_size = (pad * 2) - 1
        print(f"Framework: Using FOV size {self.fov_size}x{self.fov_size} (based on pad={pad}) for CNN.")
        # ---

        self.num_actions = 5

        dim_encoder_mlp = self.config["encoder_layers"]
        self.compress_Features_dim = self.config["encoder_dims"]

        self.graph_filter_taps = self.config["graph_filters"]
        self.node_dim = self.config["node_dims"]

        dim_action_mlp = self.config["action_layers"]
        action_features_out = [self.num_actions]

        ############################################################
        # CNN Encoder
        ############################################################
        cnn_input_channels = 3
        channels = [cnn_input_channels] + self.config["channels"]
        num_conv_layers = len(channels) - 1
        strides = [1] * num_conv_layers
        padding_size = [1] * num_conv_layers
        filter_taps = [3] * num_conv_layers

        conv_layers = []
        H_out = self.fov_size
        W_out = self.fov_size
        for l in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(channels[l], channels[l + 1], filter_taps[l], strides[l], padding_size[l], bias=True))
            conv_layers.append(nn.BatchNorm2d(num_features=channels[l + 1]))
            # --- FIX: Remove inplace=True ---
            # conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.ReLU(inplace=False))
            # --- END FIX ---

        self.convLayers = nn.Sequential(*conv_layers)
        self.cnn_flat_feature_dim = channels[-1] * H_out * W_out
        # print(f"Framework: CNN output flattened features: {self.cnn_flat_feature_dim}") # Less verbose

        ############################################################
        # MLP Encoder (after CNN)
        ############################################################
        mlp_encoder_input_dim = self.config.get("last_convs", [self.cnn_flat_feature_dim])[0]
        if mlp_encoder_input_dim != self.cnn_flat_feature_dim:
             # print(f"Framework WARNING: Config 'last_convs' {mlp_encoder_input_dim} doesn't match calculated CNN output {self.cnn_flat_feature_dim}. Using calculated value.")
             mlp_encoder_input_dim = self.cnn_flat_feature_dim

        mlp_encoder_dims = [mlp_encoder_input_dim] + self.compress_Features_dim

        mlp_encoder_layers = []
        for l in range(dim_encoder_mlp):
            mlp_encoder_layers.append(nn.Linear(mlp_encoder_dims[l], mlp_encoder_dims[l+1]))
            if l < dim_encoder_mlp - 1:
                # --- FIX: Remove inplace=True ---
                # mlp_encoder_layers.append(nn.ReLU(inplace=True))
                mlp_encoder_layers.append(nn.ReLU(inplace=False))
                # --- END FIX ---
        self.compressMLP = nn.Sequential(*mlp_encoder_layers)
        self.gnn_input_feature_dim = mlp_encoder_dims[-1]

        ############################################################
        # GNN Layers
        ############################################################
        self.num_gnn_layers = len(self.graph_filter_taps)
        gnn_feature_dims = [self.gnn_input_feature_dim] + self.node_dim

        if len(gnn_feature_dims) != self.num_gnn_layers + 1:
             raise ValueError("Config mismatch: 'node_dims' length should be equal to 'graph_filters' length.")

        gnn_layers = []
        for l in range(self.num_gnn_layers):
            gnn_layers.append(
                GCNLayer(
                    n_nodes=self.num_agents,
                    in_features=gnn_feature_dims[l],
                    out_features=gnn_feature_dims[l+1],
                    filter_number=self.graph_filter_taps[l],
                    activation=None, # Activation applied outside
                    bias=True,
                )
            )
            # --- FIX: Remove inplace=True ---
            # gnn_layers.append(nn.ReLU(inplace=True))
            gnn_layers.append(nn.ReLU(inplace=False))
            # --- END FIX ---

        self.GNNLayers = nn.Sequential(*gnn_layers)
        self.action_mlp_input_dim = gnn_feature_dims[-1]

        ############################################################
        # MLP Action Policy
        ############################################################
        action_mlp_dims = [self.action_mlp_input_dim] + action_features_out

        action_mlp_layers = []
        for l in range(dim_action_mlp):
            action_mlp_layers.append(nn.Linear(action_mlp_dims[l], action_mlp_dims[l+1]))
            if l < dim_action_mlp - 1:
                # --- FIX: Remove inplace=True ---
                # action_mlp_layers.append(nn.ReLU(inplace=True))
                action_mlp_layers.append(nn.ReLU(inplace=False))
                # --- END FIX ---
        self.actionMLP = nn.Sequential(*action_mlp_layers)

        # Initialize weights
        self.apply(weights_init)

    def forward(self, states, gso):
        """
        Forward pass of the GNN framework.
        Args:
            states (torch.Tensor): FOV observations, shape [B, N, C, H, W].
            gso (torch.Tensor): Adjacency Matrix, shape [B, N, N].
        Returns:
            torch.Tensor: Action logits, shape [B, N, num_actions].
        """
        batch_size = states.shape[0]
        # --- Input shape validation ---
        if states.shape[1:] != (self.num_agents, 3, self.fov_size, self.fov_size):
             raise ValueError(f"Input state tensor shape mismatch. Expected Bx{self.num_agents}x3x{self.fov_size}x{self.fov_size}, got {states.shape}")
        if gso.shape[1:] != (self.num_agents, self.num_agents):
             # Allow broadcasting if GSO is just [N,N]
             if gso.shape == (self.num_agents, self.num_agents):
                 gso = gso.unsqueeze(0).expand(batch_size, -1, -1)
             else:
                 raise ValueError(f"Input GSO shape mismatch. Expected Bx{self.num_agents}x{self.num_agents} or {self.num_agents}x{self.num_agents}, got {gso.shape}")
        # --- End validation ---

        # CNN Encoder per agent
        states_reshaped = states.view(batch_size * self.num_agents, 3, self.fov_size, self.fov_size)
        cnn_features = self.convLayers(states_reshaped)
        cnn_features_flat = cnn_features.view(batch_size * self.num_agents, -1)
        encoded_features = self.compressMLP(cnn_features_flat)

        # Reshape for GNN: [B, N, F_in_gnn] -> [B, F_in_gnn, N]
        gnn_input_features = encoded_features.view(batch_size, self.num_agents, self.gnn_input_feature_dim)
        gnn_input_features = gnn_input_features.permute(0, 2, 1)

        # GNN Layers
        gnn_output_features = gnn_input_features
        for i in range(self.num_gnn_layers):
             gcn_layer_index = i * 2
             activation_index = gcn_layer_index + 1
             # Pass GSO via stateful method
             self.GNNLayers[gcn_layer_index].addGSO(gso)
             gnn_output_features = self.GNNLayers[gcn_layer_index](gnn_output_features)
             gnn_output_features = self.GNNLayers[activation_index](gnn_output_features) # Apply activation

        # Action MLP
        action_mlp_input = gnn_output_features.permute(0, 2, 1) # [B, N, F_out_gnn]
        action_mlp_input_flat = action_mlp_input.reshape(batch_size * self.num_agents, self.action_mlp_input_dim)
        action_logits_flat = self.actionMLP(action_mlp_input_flat) # [B*N, num_actions]
        action_logits = action_logits_flat.view(batch_size, self.num_agents, self.num_actions)

        return action_logits