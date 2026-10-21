
import collections
import copy
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_attributes(x: nn.Module, attributes: str):
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x


def set_attribute_recursive(x: nn.Module, attributes: "str", new_attribute: nn.Module):
    for attr in attributes.split(".")[:-1]:
        x = getattr(x, attr)
    setattr(x, attributes.split(".")[-1], new_attribute)


def get_ff_layer(
    model: nn.Module,
    layer_idx: int,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    transformer_layers = get_attributes(model, transformer_layers_attr)
    assert layer_idx < len(
        transformer_layers
    ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"
    ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    return ff_layer


def register_hook(
    model: nn.Module,
    layer_idx: int,
    f: Callable,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    ff_layer = get_ff_layer(
        model,
        layer_idx,
        transformer_layers_attr=transformer_layers_attr,
        ff_attrs=ff_attrs,
    )

    def hook_fn(m, i, o):
        f(o)

    return ff_layer.register_forward_hook(hook_fn)


class Patch(torch.nn.Module):

    def __init__(
        self,
        ff_layer: nn.Module,
        mask_idx: int,
        replacement_activations: torch.Tensor = None,
        target_positions: List[List[int]] = None,
        mode: str = "replace",
        enhance_value: float = 2.0,
        down_proj_weight: Optional[torch.Tensor] = None,
        neuron_factors: dict = None,
    ):
        super().__init__()
        self.ff = ff_layer
        self.acts = replacement_activations
        self.mask_idx = mask_idx
        self.target_positions = target_positions
        self.enhance_value = enhance_value
        self.neuron_factors = neuron_factors
        self.forward_calls = 0


        self.runtime_mask_indices = None



        self.runtime_position_mask = None

        assert mode in ["replace", "suppress", "enhance", "enhance_hidden"]

        self.mode = mode
        if self.mode == "replace":
            assert self.acts is not None
        elif self.mode in ["enhance", "suppress"]:
            assert self.target_positions is not None

        self.down_weight = down_proj_weight

    def forward(self, x: torch.Tensor):
        self.forward_calls += 1
        x = self.ff(x)

        position_mask = self.runtime_position_mask
        mask_idx = self.runtime_mask_indices
        if mask_idx is None:
            mask_idx = self.mask_idx

        def token_coordinates():
            if position_mask is not None:
                runtime_mask = position_mask.to(device=x.device, dtype=torch.bool)
                if runtime_mask.shape != x.shape[:2]:
                    raise ValueError(
                        "runtime_position_mask must have shape "
                        f"{tuple(x.shape[:2])}, got {tuple(runtime_mask.shape)}"
                    )
                return runtime_mask.nonzero(as_tuple=True)
            if isinstance(mask_idx, torch.Tensor):
                batch_indices = torch.arange(x.size(0), device=x.device)
                token_indices = mask_idx.to(device=x.device, dtype=torch.long)
                return batch_indices, token_indices
            if isinstance(mask_idx, (list, tuple)):
                batch_indices = torch.arange(x.size(0), device=x.device)
                token_indices = torch.as_tensor(mask_idx, device=x.device, dtype=torch.long)
                return batch_indices, token_indices
            batch_indices = torch.arange(x.size(0), device=x.device)
            token_indices = torch.full(
                (x.size(0),), mask_idx, device=x.device, dtype=torch.long
            )
            return batch_indices, token_indices





        if self.mode == "replace":
            x[:, self.mask_idx, :] = self.acts
        elif self.mode == "suppress":
            batch_indices, token_indices = token_coordinates()
            if position_mask is None:
                for pos in self.target_positions:
                    x[batch_indices, token_indices, pos] = 0.0
                return x
            positions = torch.as_tensor(
                self.target_positions, device=x.device, dtype=torch.long
            )
            if positions.numel() and (
                positions.min().item() < 0 or positions.max().item() >= x.size(2)
            ):
                raise IndexError(f"Neuron index is outside activation width {x.size(2)}")
            x[
                batch_indices[:, None],
                token_indices[:, None],
                positions[None, :],
            ] = 0.0

        elif self.mode == "enhance":
            batch_indices, token_indices = token_coordinates()
            if position_mask is None:
                for pos in self.target_positions:
                    if pos < 0 or pos >= x.size(2):
                        raise IndexError(
                            f"Neuron index {pos} is outside activation width {x.size(2)}"
                        )
                    factor = (
                        self.neuron_factors.get(pos, self.enhance_value)
                        if self.neuron_factors else self.enhance_value
                    )
                    x[batch_indices, token_indices, pos] *= factor
                return x
            positions = torch.as_tensor(
                self.target_positions, device=x.device, dtype=torch.long
            )
            if positions.numel() and (
                positions.min().item() < 0 or positions.max().item() >= x.size(2)
            ):
                raise IndexError(f"Neuron index is outside activation width {x.size(2)}")
            factors = torch.as_tensor(
                [
                    self.neuron_factors.get(pos, self.enhance_value)
                    if self.neuron_factors else self.enhance_value
                    for pos in self.target_positions
                ],
                device=x.device,
                dtype=x.dtype,
            )
            values = x[
                batch_indices[:, None],
                token_indices[:, None],
                positions[None, :],
            ]
            x[
                batch_indices[:, None],
                token_indices[:, None],
                positions[None, :],
            ] = values * factors[None, :]
        else:
            raise NotImplementedError
        return x


def patch_ff_layer(
    model: nn.Module,
    mask_idx: int,
    layer_idx: int = None,
    replacement_activations: torch.Tensor = None,
    mode: str = "replace",
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
    neurons: List[List[int]] = None,
    _value: float = 2.0,
    output_ff_attr: str = "output.dense.weight",
    neuron_factors: dict = None,
):
    transformer_layers = get_attributes(model, transformer_layers_attr)

    if mode == "replace":
        ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
        assert layer_idx < len(
            transformer_layers
        ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"

        set_attribute_recursive(
            transformer_layers[layer_idx],
            ff_attrs,
            Patch(
                ff_layer,
                mask_idx,
                replacement_activations=replacement_activations,
                mode=mode,
            ),
        )

    elif mode in ["suppress", "enhance"]:
        neurons_dict = collections.defaultdict(list)
        for neuron in neurons:
            layer_idx, pos = neuron
            neurons_dict[layer_idx].append(pos)


        layer_neuron_factors = {}
        if neuron_factors:
            for layer_idx, positions in neurons_dict.items():
                layer_factors = {pos: neuron_factors[pos] for pos in positions if pos in neuron_factors}
                layer_neuron_factors[layer_idx] = layer_factors

        for layer_idx, positions in neurons_dict.items():
            assert layer_idx < len(transformer_layers)
            ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
            set_attribute_recursive(
                transformer_layers[layer_idx],
                ff_attrs,
                Patch(
                    ff_layer,
                    mask_idx,
                    replacement_activations=None,
                    mode=mode,
                    target_positions=positions,
                    enhance_value=_value,
                    neuron_factors=layer_neuron_factors.get(layer_idx, None),
                ),
            )
    else:
        raise NotImplementedError


def unpatch_ff_layer(
    model: nn.Module,
    layer_idx: int,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    transformer_layers = get_attributes(model, transformer_layers_attr)
    assert layer_idx < len(
        transformer_layers
    ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"
    ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    assert isinstance(ff_layer, Patch), "Can't unpatch a layer that hasn't been patched"
    set_attribute_recursive(
        transformer_layers[layer_idx],
        ff_attrs,
        ff_layer.ff,
    )


def unpatch_ff_layers(
    model: nn.Module,
    layer_indices: int,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    for layer_idx in layer_indices:
        unpatch_ff_layer(model, layer_idx, transformer_layers_attr, ff_attrs)
