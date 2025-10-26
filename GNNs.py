import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GraphNorm, JumpingKnowledge, global_add_pool

# ---------- small helpers ----------

def mlp(in_dim, hidden, out_dim, drop=0.0):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(drop),
        nn.Linear(hidden, out_dim),
    )

def gine_block(in_dim, out_dim, edge_dim, drop=0.0):
    """GINE layer with a small MLP; supports edge_attr via edge_dim."""
    nn_mlp = mlp(in_dim, out_dim, out_dim, drop=drop)
    conv = GINEConv(nn_mlp, edge_dim=edge_dim)
    return conv


# ======================================================================
# 1) Global-Local GNN (GIN + Virtual Node global token each layer)
# ======================================================================

class GlobalLocalGNN(nn.Module):
    """
    GIN(E) stack with a per-graph virtual node (global context) injected into nodes each layer.
    Interface mirrors your AttnRegGNN (including layer_to_slice_features and split output).
    """
    def __init__(self,
                 in_channels,
                 edge_dim,
                 hidden=128,
                 out_dim=6,
                 layers=6,
                 drop=0.01,
                 jk_mode='cat',
                 split_at=6):
        super().__init__()
        assert layers >= 1
        assert split_at <= out_dim, "split_at must be <= out_dim"

        self.layers = layers
        self.hidden = hidden
        self.split_at = split_at
        self.drop = drop

        # Convs & norms
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: in -> hidden
        self.convs.append(gine_block(in_channels, hidden, edge_dim, drop=drop))
        self.norms.append(GraphNorm(hidden))

        # Hidden layers: hidden -> hidden
        for _ in range(layers - 1):
            self.convs.append(gine_block(hidden, hidden, edge_dim, drop=drop))
            self.norms.append(GraphNorm(hidden))

        # Jumping knowledge
        self.jk = JumpingKnowledge(mode=jk_mode)
        if jk_mode == 'cat':
            jk_out_dim = hidden * layers
        elif jk_mode in ('max', 'last'):
            jk_out_dim = hidden
        else:
            raise ValueError(f"Unsupported JK mode: {jk_mode}")

        # Head
        self.head = nn.Sequential(
            nn.Linear(jk_out_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim)
        )

        # ---- Global (virtual) node machinery ----
        # A learnable per-graph token initialized the same for all graphs;
        # we expand it per-batch at runtime and update it with pooled node info.
        self.virtualnode_embedding = nn.Embedding(1, hidden)
        nn.init.constant_(self.virtualnode_embedding.weight, 0.0)

        self.vnode_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden)
        )

    def forward(self, x, edge_index, edge_attr, batch=None, layer_to_slice_features=-1):
        """
        x: [N, in_channels]
        edge_index: [2, E]
        edge_attr: [E, edge_dim]
        batch: [N] (optional). If None, assumes a single graph.
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        assert layer_to_slice_features < len(self.convs), \
            f'layer_to_slice_features={layer_to_slice_features}, but there are only {len(self.convs)} layers'

        outs = []

        # Expand virtual node per graph in batch
        num_graphs = int(batch.max().item()) + 1
        v = self.virtualnode_embedding.weight.expand(num_graphs, -1)  # [G, hidden]

        h = x
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Inject global token into node states
            h = h + v[batch]

            # Local message passing
            h_new = conv(h, edge_index, edge_attr)        # [N, hidden]
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            # Residual if dims match
            h_new = h_new + h if h_new.shape == h.shape else h_new

            outs.append(h_new)
            h = h_new

            # Update global token with pooled node info
            pooled = global_add_pool(h, batch)            # [G, hidden]
            v = v + self.vnode_mlp(pooled)                # [G, hidden]

        # Jumping knowledge aggregation
        h_jk = self.jk(outs)                              # [N, jk_out_dim]
        head_out = self.head(h_jk)                        # [N, out_dim]

        if layer_to_slice_features != -1:
            return head_out, outs[layer_to_slice_features]

        # Split the head output like your AttnRegGNN
        return head_out[:, :self.split_at], head_out[:, self.split_at:]


# ======================================================================
# 2) Plain GIN(E) stack (no global token), same API
# ======================================================================

class GINGNN(nn.Module):
    """
    Plain GIN(E) stack (supports edge_attr) with GraphNorm, JK, residuals.
    Matches the AttnRegGNN call/return pattern.
    """
    def __init__(self,
                 in_channels,
                 edge_dim,
                 hidden=128,
                 out_dim=6,
                 layers=3,
                 drop=0.1,
                 jk_mode='cat',
                 split_at=6):
        super().__init__()
        assert layers >= 1
        assert split_at <= out_dim, "split_at must be <= out_dim"

        self.layers = layers
        self.hidden = hidden
        self.split_at = split_at
        self.drop = drop

        # Convs & norms
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: in -> hidden
        self.convs.append(gine_block(in_channels, hidden, edge_dim, drop=drop))
        self.norms.append(GraphNorm(hidden))

        # Hidden layers
        for _ in range(layers - 1):
            self.convs.append(gine_block(hidden, hidden, edge_dim, drop=drop))
            self.norms.append(GraphNorm(hidden))

        # Jumping knowledge
        self.jk = JumpingKnowledge(mode=jk_mode)
        if jk_mode == 'cat':
            jk_out_dim = hidden * layers
        elif jk_mode in ('max', 'last'):
            jk_out_dim = hidden
        else:
            raise ValueError(f"Unsupported JK mode: {jk_mode}")

        # Head
        self.head = nn.Sequential(
            nn.Linear(jk_out_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch=None, layer_to_slice_features=-1):
        """
        x: [N, in_channels]
        edge_index: [2, E]
        edge_attr: [E, edge_dim]
        batch: [N] (optional). Not used here, included for API parity with GlobalLocalGNN.
        """
        assert layer_to_slice_features < len(self.convs), \
            f'layer_to_slice_features={layer_to_slice_features}, but there are only {len(self.convs)} layers'

        outs = []
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)        # [N, hidden]
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = h_new + h if h_new.shape == h.shape else h_new
            outs.append(h_new)
            h = h_new

        h_jk = self.jk(outs)                              # [N, jk_out_dim]
        head_out = self.head(h_jk)                        # [N, out_dim]

        if layer_to_slice_features != -1:
            return head_out, outs[layer_to_slice_features]

        return head_out[:, :self.split_at], head_out[:, self.split_at:]
