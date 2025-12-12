import os
import torch
import torch.nn.functional as F
from typing import Optional

from .model import GNN
from .blocks import MLP, MP, DownMP, UpMP, UpCN, DownCN, ResidualCN, CN, SeparableResidualCN, SeparableCN
from ..graph import Graph


class NsOneScaleGNN(GNN):
    """The 1S-GNN for incompressible flow inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    ```python
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (5, (128,128,128), False),
        # Level 1
        "mp11": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp12": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp13": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp14": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp15": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp16": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp17": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp18": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    }
    ```

    Args:
        model (str, optional): The name of the model to load. Models available are:
            - "1S-GNN-NsCircle-v1": The 1S-GNN trained on the NsCircle dataset.
            Defaults to `None`.
    """

    def __init__(self, model: str = None, *args, **kwargs) -> None:
        if model is not None:
            if model == "1S-GNN-NsCircle-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/NsMuSGNN/NsOneScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp11 = MP(*arch["mp11"])
        self.mp12 = MP(*arch["mp12"])
        self.mp13 = MP(*arch["mp13"])
        self.mp14 = MP(*arch["mp14"])
        self.mp15 = MP(*arch["mp15"])
        self.mp16 = MP(*arch["mp16"])
        self.mp17 = MP(*arch["mp17"])
        self.mp18 = MP(*arch["mp18"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp11(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp12(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp13(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp14(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp15(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp16(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp17(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp18(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output


class NsTwoScaleGNN(GNN):
    """The 2S-GNN for incompressible flow inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    ```python
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (5, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": (2+128, (128,128,128), True),
        # Level 2
        "mp21": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp22": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp23": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp24": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": (2+128+128, (128,128,128), True),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    }
    ```

    Args:
        model (str, optional): Name of the model to load. Defaults to None.
    """

    def __init__(self, model: str = None, *args, **kwargs) -> None:
        if model is not None:
            if model == "2S-GNN-NsCircle-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/NsMuSGNN/NsTwoScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp113 = MP(*arch["mp113"])
        self.mp114 = MP(*arch["mp114"])
        # Downsampling to level 2
        self.down_mp12 = DownMP(arch["down_mp12"], 1)
        # Level 2
        self.mp21 = MP(*arch["mp21"])
        self.mp22 = MP(*arch["mp22"])
        self.mp23 = MP(*arch["mp23"])
        self.mp24 = MP(*arch["mp24"])
        # Upsampling to level 1
        self.up_mp21 = UpMP(arch["up_mp21"], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        self.mp123 = MP(*arch["mp123"])
        self.mp124 = MP(*arch["mp124"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp113(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp114(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # MP at level 2
        graph.field, graph.edge_attr = self.mp21(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp22(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp23(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp24(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Upsampling to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp123(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp124(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output


class NsThreeScaleGNN(GNN):
    """The 3S-GNN for incompressible flow inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    ```python
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (5, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": (2+128, (128,128,128), True),
        # Level 2
        "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp23": (2+128, (128,128,128), True),
        # Level 3
        "mp31": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp32": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp33": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp34": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp32": (2+128+128, (128,128,128), True),
        # Level 2
        "mp221": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": (2+128+128, (128,128,128), True),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    }  
    ```

    Args:
        model (str, optional): Name of the model to load. Defaults to None. 
    """

    def __init__(self, model: str = None, *args, **kwargs) -> None:
        if model is not None:
            if model == "3S-GNN-NsCircle-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/NsMuSGNN/NsThreeScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp113 = MP(*arch["mp113"])
        self.mp114 = MP(*arch["mp114"])
        # Downsampling to level 2
        self.down_mp12 = DownMP(arch["down_mp12"], 1)
        # Level 2
        self.mp211 = MP(*arch["mp211"])
        self.mp212 = MP(*arch["mp212"])
        # Downsampling to level 3
        self.down_mp23 = DownMP(arch["down_mp23"], 2)
        # Level 3
        self.mp31  = MP(*arch["mp31"])
        self.mp32  = MP(*arch["mp32"])
        self.mp33  = MP(*arch["mp33"])
        self.mp34  = MP(*arch["mp34"])
        # Upsampling to level 2
        self.up_mp32 = UpMP(arch["up_mp32"], 3)
        # Level 2
        self.mp221 = MP(*arch["mp221"])
        self.mp222 = MP(*arch["mp222"])
        # Upsampling to level 1
        self.up_mp21 = UpMP(arch["up_mp21"], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        self.mp123 = MP(*arch["mp123"])
        self.mp124 = MP(*arch["mp124"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:  
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp113(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp114(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # MP at level 2
        graph.field, graph.edge_attr = self.mp211(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp212(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field2, pos2, edge_index2, edge_attr2 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 3
        graph = self.down_mp23(graph, activation=torch.tanh)
        # MP at level 3
        graph.field, graph.edge_attr = self.mp31(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp32(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp33(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp34(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Upsampling to level 2
        graph = self.up_mp32(graph, field2, pos2, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index2, edge_attr2
        # MP at level 2
        graph.field, graph.edge_attr = self.mp221(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp222(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Upsampling to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp123(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp124(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output
    

class NsFourScaleGNN(GNN):
    """The 4S-GNN for incompressible flow inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    ```python
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (5, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": (2+128, (128,128,128), True),
        # Level 2
        "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp23": (2+128, (128,128,128), True),
        # Level 3
        "mp311": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp312": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp34": (2+128, (128,128,128), True),
        # Level 4
        "mp41": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp42": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp43": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp44": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp43": (2+128+128, (128,128,128), True),
        # Level 3
        "mp321": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp322": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp32": (2+128+128, (128,128,128), True),
        # Level 2
        "mp221": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": (2+128+128, (128,128,128), True),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    }
    ```

    Args:
        model (str, optional): Name of the model to load. Defaults to None.
    """

    def __init__(self, model: str = None, *args, **kwargs):
        if model is not None:
            if model == "4S-GNN-NsCircle-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/NsMuSGNN/NsFourScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp113 = MP(*arch["mp113"])
        self.mp114 = MP(*arch["mp114"])
        # Downsampling to level 2
        self.down_mp12 = DownMP(arch["down_mp12"], 1)
        # Level 2
        self.mp211 = MP(*arch["mp211"])
        self.mp212 = MP(*arch["mp212"])
        # Downsampling to level 3
        self.down_mp23 = DownMP(arch["down_mp23"], 2)
        # Level 3
        self.mp311 = MP(*arch["mp311"])
        self.mp312 = MP(*arch["mp312"])
        # Downsampling to level 4
        self.down_mp34 = DownMP(arch["down_mp34"], 3)
        # Level 4
        self.mp41 = MP(*arch["mp41"])
        self.mp42 = MP(*arch["mp42"])
        self.mp43 = MP(*arch["mp43"])
        self.mp44 = MP(*arch["mp44"])
        # Upsampling to level 3
        self.up_mp43 = UpMP(arch["up_mp43"], 4)
        # Level 3
        self.mp321  = MP(*arch["mp321"])
        self.mp322  = MP(*arch["mp322"])
        # Upsampling to level 2
        self.up_mp32 = UpMP(arch["up_mp32"], 3)
        # Level 2
        self.mp221 = MP(*arch["mp221"])
        self.mp222 = MP(*arch["mp222"])
        # Upsampling to level 1
        self.up_mp21 = UpMP(arch["up_mp21"], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        self.mp123 = MP(*arch["mp123"])
        self.mp124 = MP(*arch["mp124"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp113(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp114(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # MP at level 2
        graph.field, graph.edge_attr = self.mp211(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp212(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field2, pos2, edge_index2, edge_attr2 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 3
        graph = self.down_mp23(graph, activation=torch.tanh)
        # MP at level 3
        graph.field, graph.edge_attr = self.mp311(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp312(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field3, pos3, edge_index3, edge_attr3 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 4
        graph = self.down_mp34(graph, activation=torch.tanh)
        # Level 4
        graph.field, graph.edge_attr = self.mp41(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp42(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp43(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp44(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Upsampling to level 3
        graph = self.up_mp43(graph, field3, pos3, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index3, edge_attr3
        # Level 3
        graph.field, graph.edge_attr = self.mp321(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp322(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Upsampling to level 2
        graph = self.up_mp32(graph, field2, pos2, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index2, edge_attr2
        # MP at level 2
        graph.field, graph.edge_attr = self.mp221(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp222(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Upsampling to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp123(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp124(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output
    

class NsFourScaleGNNwCNNwOffsets(GNN):
    """
    Args:
        model (str, optional): Name of the model to load. Defaults to None.
    """

    def __init__(self, model: str = None, *args, **kwargs):
        if model is not None:
            if model == "4S-GNN-NsCircle-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/NsMuSGNN/NsFourScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        # TODO: Try changing regular CN layers to depthwise-separable CN layers
        # arch based on https://link.springer.com/chapter/10.1007/978-3-030-00889-5_18
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp113 = MP(*arch["mp113"])
        self.mp114 = MP(*arch["mp114"])
        # Downsampling to level 2
        self.down_mp12  = DownMP(arch["down_mp12"], 1)
        # Level 2
        self.cn211      = CN(arch["cn211"], True, True, 2)
        self.rcn211     = ResidualCN(arch["rcn211"], False, False, 2)
        # Downsampling to level 3
        self.down_cn23 = DownCN(arch["down_cn23"], 3)
        # Level 3
        self.cn311      = CN(arch["cn311"], False, False, 3)
        self.rcn311     = ResidualCN(arch["rcn311"], False, False, 3)
        # Downsampling to level 4
        self.down_cn34 = DownCN(arch["down_cn34"], 3)
        # Level 4
        self.cn411      = CN(arch["cn411"], False, False, 4)
        self.rcn411     = ResidualCN(arch["rcn411"], False, False, 4)
        # Upsampling to level 3
        self.up_cn43    = UpCN(arch["up_cn43"], 4)
        # Level 3
        self.cn321      = CN(arch["cn321"], False, True, 3) # reduce input channels
        self.rcn321     = ResidualCN(arch["rcn321"], False, False, 3)
        # Upsampling to Level 2
        self.up_cn32     = UpCN(arch["up_cn32"], 3)
        # Level 2
        self.cn221      = CN(arch["cn221"], False, True, 2) # reduce input channels
        self.rcn221    = ResidualCN(arch["rcn221"], True, False, 2)
        # Upsampling to Level 1
        self.up_mp21 = UpMP(arch["up_mp21"], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        self.mp123 = MP(*arch["mp123"])
        self.mp124 = MP(*arch["mp124"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp113(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp114(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # CN at level 2
        grid_offsets2 = graph.grid_offsets_2
        graph = self.cn211(graph, grid_offsets2)
        graph.cached_grid = F.selu(graph.cached_grid)
        graph = self.rcn211(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        #print(f"Cached grid requires grad? {graph.cached_grid.requires_grad}")
        cached_grid2 = graph.cached_grid
        # Downsampling to level 3
        graph = self.down_cn23(graph) # maxpool
        # CN at level 3
        #grid_offsets3 = graph.grid_offsets_3
        #graph = self.cn311(graph, grid_offsets3)
        graph = self.cn311(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        graph = self.rcn311(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        cached_grid3 = graph.cached_grid
        # Downsampling to level 4
        graph = self.down_cn34(graph)
        # Level 4
        #grid_offsets4 = graph.grid_offsets_4
        #graph = self.cn411(graph, grid_offsets4)
        graph = self.cn411(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        graph = self.rcn411(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        # Upsampling to level 3
        graph = self.up_cn43(graph) # concatenate with output of level 3
        # Level 3
        graph = self.cn321(graph, cached_grid3)
        graph.cached_grid = F.selu(graph.cached_grid)
        graph = self.rcn321(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        # Upsampling to level 2
        graph = self.up_cn32(graph)
        # MP at level 2
        graph = self.cn221(graph, cached_grid2)
        graph.cached_grid = F.selu(graph.cached_grid)
        # UNPAD BEFORE TRANSFORMING BACK TO GRID!!
        graph.cached_grid = graph.cached_grid[:cached_grid2.shape[0], :cached_grid2.shape[1], :]
        graph = self.rcn221(graph)
        #graph.cached_grid = F.selu(graph.cached_grid)
        graph.field = F.selu(graph.field)
        # Upsampling to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp123(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp124(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output


class NsFourScaleGNNwCNNSeparable(GNN):
    """The 4S-GNN for incompressible flow inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    ```python
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (5, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": (2+128, (128,128,128), True),
        # Level 2
        "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp23": (2+128, (128,128,128), True),
        # Level 3
        "mp311": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp312": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp34": (2+128, (128,128,128), True),
        # Level 4
        "mp41": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp42": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp43": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp44": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp43": (2+128+128, (128,128,128), True),
        # Level 3
        "mp321": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp322": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp32": (2+128+128, (128,128,128), True),
        # Level 2
        "mp221": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": (2+128+128, (128,128,128), True),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    }
    ```

    Args:
        model (str, optional): Name of the model to load. Defaults to None.
    """

    def __init__(self, model: str = None, *args, **kwargs):
        if model is not None:
            if model == "4S-GNN-NsCircle-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/NsMuSGNN/NsFourScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        # TODO: Try changing regular CN layers to depthwise-separable CN layers
        # arch based on https://link.springer.com/chapter/10.1007/978-3-030-00889-5_18
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp113 = MP(*arch["mp113"])
        self.mp114 = MP(*arch["mp114"])
        # Downsampling to level 2
        self.down_mp12  = DownMP(arch["down_mp12"], 1)
        # Level 2
        self.cn211      = SeparableCN(arch["cn211"], True, True, 2)
        self.rcn211     = SeparableResidualCN(arch["rcn211"], False, False, 2)
        # Downsampling to level 3
        self.down_cn23 = DownCN(arch["down_cn23"], 3)
        # Level 3
        self.cn311      = SeparableCN(arch["cn311"], False, False, 3)
        self.rcn311     = SeparableResidualCN(arch["rcn311"], False, False, 3)
        # Downsampling to level 4
        self.down_cn34 = DownCN(arch["down_cn34"], 3)
        # Level 4
        self.cn411      = SeparableCN(arch["cn411"], False, False, 4)
        self.rcn411     = SeparableResidualCN(arch["rcn411"], False, False, 4)
        # Upsampling to level 3
        self.up_cn43    = UpCN(arch["up_cn43"], 4)
        # Level 3
        self.cn321      = SeparableCN(arch["cn321"], False, True, 3) # reduce input channels
        self.rcn321     = SeparableResidualCN(arch["rcn321"], False, False, 3)
        # Upsampling to Level 2
        self.up_cn32     = UpCN(arch["up_cn32"], 3)
        # Level 2
        self.cn221      = SeparableCN(arch["cn221"], False, True, 2) # reduce input channels
        self.rcn221    = SeparableResidualCN(arch["rcn221"], True, False, 2)
        # Upsampling to Level 1
        self.up_mp21 = UpMP(arch["up_mp21"], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        self.mp123 = MP(*arch["mp123"])
        self.mp124 = MP(*arch["mp124"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp113(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp114(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # CN at level 2
        grid_offsets2 = graph.grid_offsets_2
        graph = self.cn211(graph, grid_offsets2)
        graph.cached_grid = F.selu(graph.cached_grid)
        graph = self.rcn211(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        #print(f"Cached grid requires grad? {graph.cached_grid.requires_grad}")
        cached_grid2 = graph.cached_grid
        # Downsampling to level 3
        graph = self.down_cn23(graph) # maxpool
        # CN at level 3
        #grid_offsets3 = graph.grid_offsets_3
        #graph = self.cn311(graph, grid_offsets3)
        graph = self.cn311(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        graph = self.rcn311(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        cached_grid3 = graph.cached_grid
        # Downsampling to level 4
        graph = self.down_cn34(graph)
        # Level 4
        #grid_offsets4 = graph.grid_offsets_4
        #graph = self.cn411(graph, grid_offsets4)
        graph = self.cn411(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        graph = self.rcn411(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        # Upsampling to level 3
        graph = self.up_cn43(graph) # concatenate with output of level 3
        # Level 3
        graph = self.cn321(graph, cached_grid3)
        graph.cached_grid = F.selu(graph.cached_grid)
        graph = self.rcn321(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        # Upsampling to level 2
        graph = self.up_cn32(graph)
        # MP at level 2
        graph = self.cn221(graph, cached_grid2)
        graph.cached_grid = F.selu(graph.cached_grid)
        # UNPAD BEFORE TRANSFORMING BACK TO GRID!!
        graph.cached_grid = graph.cached_grid[:cached_grid2.shape[0], :cached_grid2.shape[1], :]
        graph = self.rcn221(graph)
        #graph.cached_grid = F.selu(graph.cached_grid)
        graph.field = F.selu(graph.field)
        # Upsampling to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp123(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp124(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output



class NsFourScaleGNNwCNN(GNN):
    """The 4S-GNN for incompressible flow inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:

    Args:
        model (str, optional): Name of the model to load. Defaults to None.
    """

    def __init__(self, model: str = None, *args, **kwargs):
        if model is not None:
            if model == "4S-GNN-NsCircle-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/NsMuSGNN/NsFourScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        # TODO: Try changing regular CN layers to depthwise-separable CN layers
        # arch based on https://link.springer.com/chapter/10.1007/978-3-030-00889-5_18
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp113 = MP(*arch["mp113"])
        self.mp114 = MP(*arch["mp114"])
        # Downsampling to level 2
        self.down_mp12  = DownMP(arch["down_mp12"], 1)
        # Level 2
        self.cn211      = CN(arch["cn211"], True, False, 2)
        self.rcn211     = ResidualCN(arch["rcn211"], False, False, 2)
        # Downsampling to level 3
        self.down_cn23 = DownCN(arch["down_cn23"], 3)
        # Level 3
        self.cn311      = CN(arch["cn311"], False, False, 3)
        self.rcn311     = ResidualCN(arch["rcn311"], False, False, 3)
        # Downsampling to level 4
        self.down_cn34 = DownCN(arch["down_cn34"], 3)
        # Level 4
        self.cn411      = CN(arch["cn411"], False, False, 4)
        self.rcn411     = ResidualCN(arch["rcn411"], False, False, 4)
        # Upsampling to level 3
        self.up_cn43    = UpCN(arch["up_cn43"], 4)
        # Level 3
        self.cn321      = CN(arch["cn321"], False, True, 3) # reduce input channels
        self.rcn321     = ResidualCN(arch["rcn321"], False, False, 3)
        # Upsampling to Level 2
        self.up_cn32     = UpCN(arch["up_cn32"], 3)
        # Level 2
        self.cn221      = CN(arch["cn221"], False, True, 2) # reduce input channels
        self.rcn221    = ResidualCN(arch["rcn221"], True, False, 2)
        # Upsampling to Level 1
        self.up_mp21 = UpMP(arch["up_mp21"], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        self.mp123 = MP(*arch["mp123"])
        self.mp124 = MP(*arch["mp124"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp113(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp114(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # CN at level 2
        graph = self.cn211(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        graph = self.rcn211(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        #print(f"Cached grid requires grad? {graph.cached_grid.requires_grad}")
        cached_grid2 = graph.cached_grid
        # Downsampling to level 3
        graph = self.down_cn23(graph) # maxpool
        # CN at level 3
        #grid_offsets3 = graph.grid_offsets_3
        #graph = self.cn311(graph, grid_offsets3)
        graph = self.cn311(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        graph = self.rcn311(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        cached_grid3 = graph.cached_grid
        # Downsampling to level 4
        graph = self.down_cn34(graph)
        # Level 4
        #grid_offsets4 = graph.grid_offsets_4
        #graph = self.cn411(graph, grid_offsets4)
        graph = self.cn411(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        graph = self.rcn411(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        # Upsampling to level 3
        graph = self.up_cn43(graph) # concatenate with output of level 3
        # Level 3
        graph = self.cn321(graph, cached_grid3)
        graph.cached_grid = F.selu(graph.cached_grid)
        graph = self.rcn321(graph)
        graph.cached_grid = F.selu(graph.cached_grid)
        # Upsampling to level 2
        graph = self.up_cn32(graph)
        # MP at level 2
        graph = self.cn221(graph, cached_grid2)
        graph.cached_grid = F.selu(graph.cached_grid)
        # UNPAD BEFORE TRANSFORMING BACK TO GRID!!
        graph.cached_grid = graph.cached_grid[:cached_grid2.shape[0], :cached_grid2.shape[1], :]
        graph = self.rcn221(graph)
        #graph.cached_grid = F.selu(graph.cached_grid)
        graph.field = F.selu(graph.field)
        # Upsampling to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp123(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp124(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output
    

class AdvOneScaleGNN(GNN):
    """The 1S-GNN for advection inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    ```python
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (4, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,1), False),
    }
    ```

    Args:
        model (str, optional): Name of the model to load. Available models are:
            - "1S-GNN-UniformAdv-v1": 1S-GNN from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).
            Defaults to None.
    """

    def __init__(self, model: str = None, *args, **kwargs):
        if model is not None:
            if model == "1S-GNN-UniformAdv-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/AdvMuSGNN/AdvOneScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output


class AdvTwoScaleGNN(GNN):
    """The 2S-GNN for advection inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    ```python
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (4, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": (2+128, (128,128,128), True),
        # Level 2
        "mp21": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp22": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp23": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp24": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": (2+128+128, (128,128,128), True),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,1), False),
    }
    ```

    Args:
        model (str, optional): Name of the model to load. Defaults to None.
    """

    def __init__(self, model: str = None, *args, **kwargs):
        if model is not None:
            if model == "2S-GNN-UniformAdv-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/AdvMuSGNN/AdvTwoScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        # Pooling to level 2
        self.down_mp12 = DownMP(arch["down_mp12"], 1)
        # Level 2
        self.mp21 = MP(*arch["mp21"])
        self.mp22 = MP(*arch["mp22"])
        self.mp23 = MP(*arch["mp23"])
        self.mp24 = MP(*arch["mp24"])
        # Undown_mping to level 1
        self.up_mp21 = UpMP(arch["up_mp21"], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Pooling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # MP at level 2
        graph.field, graph.edge_attr = self.mp21(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp22(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp23(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp24(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Undown_mping to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output


class AdvThreeScaleGNN(GNN):
    """The 3S-GNN for advection inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    ```python
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (4, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": (2+128, (128,128,128), True),
        # Level 2
        "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp23": (2+128, (128,128,128), True),
        # Level 3
        "mp31": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp32": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp33": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp34": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp32": (2+128+128, (128,128,128), True),
        # Level 2
        "mp221": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": (2+128+128, (128,128,128), True),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,1), False),
    }
    ```

    Args:
        model (str, optional): The name of the model to load. Defaults to None.
    """

    def __init__(self, model:str = None, *args, **kwargs):
        if model is not None:
            if model == "3S-GNN-UniformAdv-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/AdvMuSGNN/AdvThreeScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        # Pooling to level 2
        self.down_mp12 = DownMP(arch["down_mp12"], 1)
        # Level 2
        self.mp211 = MP(*arch["mp211"])
        self.mp212 = MP(*arch["mp212"])
        # Pooling to level 3
        self.down_mp23 = DownMP(arch["down_mp23"], 2)
        # Level 3
        self.mp31  = MP(*arch["mp31"])
        self.mp32  = MP(*arch["mp32"])
        self.mp33  = MP(*arch["mp33"])
        self.mp34  = MP(*arch["mp34"])
        # Undown_mping to level 2
        self.up_mp32 = UpMP(arch["up_mp32"], 3)
        # Level 2
        self.mp221 = MP(*arch["mp221"])
        self.mp222 = MP(*arch["mp222"])
        # Undown_mping to level 1
        self.up_mp21 = UpMP(arch["up_mp21"], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Pooling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # MP at level 2
        graph.field, graph.edge_attr = self.mp211(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp212(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field2, pos2, edge_index2, edge_attr2 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Pooling to level 3
        graph = self.down_mp23(graph, activation=torch.tanh)
        # MP at level 3
        graph.field, graph.edge_attr = self.mp31(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp32(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp33(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp34(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Undown_mping to level 2
        graph = self.up_mp32(graph, field2, pos2, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index2, edge_attr2
        # MP at level 2
        graph.field, graph.edge_attr = self.mp221(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp222(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Undown_mping to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output


class AdvFourScaleGNN(GNN):
    """The 4S-GNN for advection inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    ```python
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (4, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": (2+128, (128,128,128), True),
        # Level 2
        "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp23": (2+128, (128,128,128), True),
        # Level 3
        "mp311": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp312": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp34": (2+128, (128,128,128), True),
        # Level 4
        "mp41": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp42": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp43": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp44": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp43": (2+128+128, (128,128,128), True),
        # Level 3
        "mp321": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp322": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp32": (2+128+128, (128,128,128), True),
        # Level 2
        "mp221": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": (2+128+128, (128,128,128), True),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,1), False),
    }
    ```

    Args:
        model (str, optional): The model to load. Defaults to None.
    """
    
    def __init__(self, model: str = None, *args, **kwargs):
        if model is not None:
            if model == "4S-GNN-UniformAdv-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/AdvMuSGNN/AdvFourScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        # Pooling to level 2
        self.down_mp12 = DownMP(arch["down_mp12"], 1)
        # Level 2
        self.mp211 = MP(*arch["mp211"])
        self.mp212 = MP(*arch["mp212"])
        # Pooling to level 3
        self.down_mp23 = DownMP(arch["down_mp23"], 2)
        # Level 3
        self.mp311 = MP(*arch["mp311"])
        self.mp312 = MP(*arch["mp312"])
        # Pooling to level 4
        self.down_mp34 = DownMP(arch["down_mp34"], 3)
        # Level 4
        self.mp41 = MP(*arch["mp41"])
        self.mp42 = MP(*arch["mp42"])
        self.mp43 = MP(*arch["mp43"])
        self.mp44 = MP(*arch["mp44"])
        # Undown_mping to level 3
        self.up_mp43 = UpMP(arch["up_mp43"], 4)
        # Level 3
        self.mp321  = MP(*arch["mp321"])
        self.mp322  = MP(*arch["mp322"])
        # Undown_mping to level 2
        self.up_mp32 = UpMP(arch["up_mp32"], 3)
        # Level 2
        self.mp221 = MP(*arch["mp221"])
        self.mp222 = MP(*arch["mp222"])
        # Undown_mping to level 1
        self.up_mp21 = UpMP(arch["up_mp21"], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Pooling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # MP at level 2
        graph.field, graph.edge_attr = self.mp211(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp212(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field2, pos2, edge_index2, edge_attr2 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Pooling to level 3
        graph = self.down_mp23(graph, activation=torch.tanh)
        # MP at level 3
        graph.field, graph.edge_attr = self.mp311(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp312(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field3, pos3, edge_index3, edge_attr3 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Pooling to level 4
        graph = self.down_mp34(graph, activation=torch.tanh)
        # Level 4
        graph.field, graph.edge_attr = self.mp41(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp42(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp43(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp44(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Undown_mping to level 3
        graph = self.up_mp43(graph, field3, pos3, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index3, edge_attr3
        # Level 3
        graph.field, graph.edge_attr = self.mp321(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp322(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Undown_mping to level 2
        graph = self.up_mp32(graph, field2, pos2, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index2, edge_attr2
        # MP at level 2
        graph.field, graph.edge_attr = self.mp221(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp222(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Undown_mping to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output