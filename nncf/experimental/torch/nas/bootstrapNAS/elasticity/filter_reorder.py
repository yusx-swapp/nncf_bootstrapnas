# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Type

import torch

from nncf.common.graph import NNCFGraph
from nncf.common.logging import nncf_logger
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.utils import get_input_masks
from nncf.torch.nncf_network import NNCFNetwork


class FilterReorderingAlgorithm(MaskPropagationAlgorithm):
    """
    Reorders filters based on reordering indexes encoded in the `output_mask` attribute in the nodes of
    model graph.
    """

    def __init__(
        self,
        model: NNCFNetwork,
        graph: NNCFGraph,
        pruning_operator_metatypes: PruningOperationsMetatypeRegistry,
        tensor_processor: Optional[Type[NNCFPruningBaseTensorProcessor]] = None,
    ):
        super().__init__(graph, pruning_operator_metatypes, tensor_processor)
        self._model = model

    def apply_reordering_indexes(self) -> None:
        """
        Applying propagated masks (which encodes indexes to reorder filters) for all nodes in topological order:
        1. running input_reorder method for this node
        2. running output_reorder method for this node
        """
        pruned_node_modules = []
        with torch.no_grad():
            for node in self._graph.topological_sort():
                node_cls = self.get_meta_operation_by_type_name(node.node_type)
                node_module = self._model.nncf.get_containing_module(node.node_name)
                if node_module not in pruned_node_modules:
                    node_cls.input_reorder(self._model, node, self._graph)
                    node_cls.output_reorder(self._model, node, self._graph)
                    pruned_node_modules.append(node_module)
            nncf_logger.debug("Finished mask applying step")

    def reorder_filters(self, add_dynamic_inputs=None) -> None:
        """
        Model pruner work in two stages:
        1. Mask propagation: propagate pruning masks through the graph.
        2. Applying calculated masks
        """
        nncf_logger.info("Start reordering filters")
        self.mask_propagation()
        self.elastic_width_add_dynamic_inputs(add_dynamic_inputs)
        self.apply_reordering_indexes()
        nncf_logger.info("Finished reordering filters")

    def elastic_width_add_dynamic_inputs(self, add_dynamic_inputs=None) -> None:
        # we should add input masks for nodes in `add_dynamic_inputs`, similar to activate_subnet_config
        if add_dynamic_inputs:
            for node_name in add_dynamic_inputs:
                ori_node = self._graph.get_node_by_name(node_name)
                nncf_logger.debug(f"setting input width by user's request for scope={node_name}")
                nodes_to_check = [ori_node]
                input_masks = [None]
                while any(elem is None for elem in input_masks):
                    previous_nodes = []
                    for node in nodes_to_check:
                        previous_nodes.append(self._graph.get_previous_nodes(node))
                    nodes_to_check.clear()
                    previous_nodes = [item for nodes in previous_nodes for item in nodes]
                    if not previous_nodes:
                        break
                    for previous in previous_nodes:
                        if "output_mask" in previous.data:
                            if previous.data["output_mask"] is not None:
                                input_masks.append(previous.data["output_mask"])
                                input_masks = [i for i in input_masks if i]
                            else:
                                nodes_to_check.append(previous)
                        else:
                            nodes_to_check.append(previous)
                if input_masks:  # force set input mask
                    filters_num = self._model.nncf.get_containing_module(ori_node.node_name).weight.size(0)
                    input_mask = input_masks[0]
                    if input_mask and input_mask.tensor.size(0) == filters_num:
                        prev_node = self._graph.get_previous_nodes(ori_node)[0]
                        prev_node.data["output_mask"] = input_mask
