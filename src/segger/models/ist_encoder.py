from torch_geometric.nn import GATv2Conv, Linear, HeteroDictLinear, HeteroConv
from typing import Dict, Tuple, List, Union
from torch import Tensor
from torch.nn import (
    Sequential,
    ModuleDict,
    ModuleList,
    Embedding,
    Module,
    Linear as NNLinear,
    SiLU,
    functional as F
)
import torch
import math

# --- Test positional encoding ---

def sinusoidal_embedding(x, dim, max_period=1000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=x.device)
    args = x[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Positional2dEmbedder(Module):
    """
    Embeds 2D spatial coordinates into vector representations using sinusoidal
    encoding. Coordinates are normalized against global tissue extent (set via
    pos_min / pos_max buffers) so the encoding is consistent across tiles.
    """
    def __init__(
            self,
            hidden_size:int,
            frequency_embedding_size:int=256):
        super().__init__()
        self.dim = hidden_size//2
        self.mlp = Sequential(
            NNLinear(frequency_embedding_size, self.dim, bias=True),
            SiLU(),
            NNLinear(self.dim, self.dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        # Buffers are overwritten in LitISTEncoder.setup() with global extents.
        self.register_buffer('pos_min', torch.zeros(2))
        self.register_buffer('pos_max', torch.ones(2))

    @staticmethod
    def embed(x:torch.Tensor, dim:int, max_period:int=10000):
        shape = x.shape
        embedding_flat = sinusoidal_embedding(x.flatten(), dim, max_period=max_period)
        embedding = embedding_flat.reshape(shape+(dim,))
        return embedding

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        pos = (pos - self.pos_min) / (self.pos_max - self.pos_min + 1e-8)
        pos_freq = self.embed(pos, self.frequency_embedding_size)
        pos_emb = self.mlp(pos_freq)
        pos_emb = pos_emb.flatten(-2)
        return pos_emb

# --- Test positional encoding ---
class SkipGAT(Module):
    """
    Graph Attention module that encapsulates a HeteroConv layer with two GATv2
    convolutions for different edge types. The attention weights from the last
    forward pass are stored internally and can be accessed via the
    `attention_weights` property.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    heads : int
        Number of attention heads.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_heads: int,
        add_self_loops_tx: bool = False,
    ) -> None:
        super().__init__()

        # Build a HeteroConv that internally uses GATv2Conv for each edge type.
        self.conv = HeteroConv(
            convs={
                ('tx', 'neighbors', 'tx'): GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=n_heads,
                    add_self_loops=add_self_loops_tx,
                    dropout=0.2,
                ),
                ('tx', 'belongs', 'bd'): GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=n_heads,
                    add_self_loops=False,
                    dropout=0.2,
                ),
                ('bd', 'contains', 'tx'): GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=n_heads,
                    add_self_loops=False,
                    dropout=0.2,
                ),
            },
            aggr='sum'
        )

        # This will store the attention weights from the last forward pass.
        self._attn_weights: Dict[Tuple[str, str, str], Tensor] = {}

        # Register a forward hook to capture attention weights internally.
        edge_type = 'tx', 'neighbors', 'tx'
        self.conv.convs[edge_type].register_forward_hook(
            self._make_hook(edge_type),
            with_kwargs=True,
        )

    def _make_hook(self, edge_type: Tuple[str, str, str]):
        """
        Internal hook function that captures attention weights from the
        forward pass of each GATv2Conv submodule.

        Parameters
        ----------
        edge_type : tuple of str
            The edge type associated with this GATv2Conv.
        """
        def _store_attn_weights(module, inputs, kwargs, outputs) -> None:
            self._attn_weights[edge_type] = outputs[1][1]
        return _store_attn_weights

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Forward pass for SkipGAT. Always calls HeteroConv with
        `return_attention_weights=True`, but never returns them from
        this method. Attention weights are stored internally via the hook.

        Parameters
        ----------
        x_dict : dict of str -> Tensor
            Node features for each node type.
        edge_index_dict : dict of str -> Tensor
            Edge indices for each edge type.

        Returns
        -------
        x_dict_out : dict of str -> Tensor
            Updated node embeddings after convolution.
        """
        # Always request attention weights, but do not return them here.
        x_dict = self.conv(
            x_dict,
            edge_index_dict,
            return_attention_weights_dict = {
                edge: False for edge in self.conv.convs
            },
        )
        return x_dict

    @property
    def attention_weights(self) -> Dict[Tuple[str, str, str], Tensor]:
        """
        The attention weights from the most recent forward pass.

        Raises
        ------
        RuntimeError
            If no forward pass has been performed yet.

        Returns
        -------
        dict of (str, str, str) -> Tensor
            Mapping each edge type to its attention weight tensor of shape
            [num_edges, num_heads].
        """
        if not self._attn_weights:
            msg = "Attention weights are empty. Please perform a forward pass."
            raise AttributeError(msg)
        return self._attn_weights


class ISTEncoder(torch.nn.Module):
    """
    TODO: Description.
    """

    def __init__(
        self,
        n_genes: int,
        in_channels: int = 16,
        hidden_channels: int = 32,
        out_channels: int = 32,
        n_mid_layers: int = 3,
        n_heads: int = 3,
        normalize_embeddings: bool = True,
        use_positional_embeddings: bool = True,
    ):
        """
        Initialize the Segger model.

        Parameters
        ----------
        n_genes : int
            Number of unique genes for embedding.
        in_channels : int, optional
            Initial embedding size for both 'tx' and boundary nodes.
            Default is 16.
        hidden_channels : int, optional
            Number of hidden channels. Default is 32.
        out_channels : int, optional
            Number of output channels. Default is 32.
        n_mid_layers : int, optional
            Number of hidden layers (excluding first and last layers).
            Default is 3.
        n_heads : int, optional
            Number of attention heads. Default is 3.
        """
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.use_positional_embeddings = use_positional_embeddings
        # Store hyperparameters for PyTorch Lightning
        self.hparams = locals()
        for k in ['self', '__class__']: 
            self.hparams.pop(k)
        # First layer: ? -> in
        self.lin_first = ModuleDict(
            {
                'tx': Embedding(n_genes, in_channels),
                'bd': Linear(-1, in_channels),
            }
        )
        # Positional encoding: in
        self.pos_emb = Positional2dEmbedder(in_channels)

        self.conv_layers = ModuleList()
        # First convolution: in -> hidden x heads
        self.conv_layers.append(
            SkipGAT((-1, -1), hidden_channels, n_heads)
        )
        # Middle convolutions: hidden x heads -> hidden x heads
        for _ in range(n_mid_layers):
            self.conv_layers.append(
                SkipGAT((-1, -1), hidden_channels, n_heads)
            )
        # Last convolution: hidden x heads -> out x heads
        self.conv_layers.append(
            SkipGAT((-1, -1), out_channels, n_heads)
        )
        # Last layer: out x heads -> out
        self.lin_last = HeteroDictLinear(
            -1,
            out_channels,
            types=("tx", "bd")
        )

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[str, Tensor],
        pos_dict: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """
        Forward pass for the Segger model.

        Parameters
        ----------
        x_dict : dict[str, Tensor]
            Node features for each node type.
        edge_index_dict : dict[str, Tensor]
            Edge indices for each edge type.
        pos_dict : dict[str, Tensor]
            Node positions for each node type (used for positional encoding).

        Returns
        -------
        dict[str, Tensor]
            Output node embeddings for each node type.
        """
        # Linearly project embedding to input dim
        x_dict = {k: self.lin_first[k](x) for k, x in x_dict.items()}

        # Add positional embedding
        if self.use_positional_embeddings:
            x_dict = {
                k: torch.cat((x, self.pos_emb(pos_dict[k])), -1)
                for k, x in x_dict.items()
            }

        # GeLu for some reason
        x_dict = {k: F.gelu(x) for k, x in x_dict.items()}

        # Graph convolutions with GATv2
        for conv_layer in self.conv_layers:
            x_dict = conv_layer(x_dict, edge_index_dict)
            x_dict = {k: F.gelu(x) for k, x in x_dict.items()}

        # Linearly project to output dim
        x_dict = self.lin_last(x_dict)

        # Normalize so distances are cosine similarities
        if self.normalize_embeddings:
            x_dict = {k: F.normalize(v, dim=-1) for k, v in x_dict.items()}
        return x_dict
