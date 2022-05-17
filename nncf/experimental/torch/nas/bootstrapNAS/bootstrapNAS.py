import torch

from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import resume_compression_from_state
from nncf.torch.model_creation import create_nncf_network
from nncf.torch.checkpoint_loading import load_state

class BootstrapNAS:
    def __init__(self, model, nncf_config, supernet_path, supernet_weights):
        nncf_network = create_nncf_network(model, nncf_config)

        compression_state = torch.load(supernet_path, map_location=torch.device(nncf_config.device))
        self._model, self._elasticity_ctrl = resume_compression_from_state(nncf_network, compression_state)
        model_weights = torch.load(supernet_weights, map_location=torch.device(nncf_config.device))

        load_state(model, model_weights, is_resume=True)


    def get_search_space(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        active_handlers = {
            dim: m_handler._handlers[dim] for dim in m_handler._handlers if m_handler._is_handler_enabled_map[dim]
        }
        space = {}
        for handler_id, handler in active_handlers.items():
            space[handler_id.value] = handler.get_search_space()
        return space

    def eval_subnet(self, config, eval_fn, **kwargs):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        m_handler.activate_subnet_for_config(m_handler.get_config_from_pymoo(config))
        print(kwargs)
        return eval_fn(self._model, **kwargs)

