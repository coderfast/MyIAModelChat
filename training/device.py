"""Device and model deployment mixin."""
import os
import gc
import logging

import torch

logger = logging.getLogger(__name__)


class DeviceMixin:
    """Mixin providing device resolution and model deployment strategies.

    Expects the host class to define: self.device_mode, self.gpu_indices,
    self.use_vulkan, self.rank, self.local_rank, self.world_size, self.config.
    Sets: self.use_gpu, self.use_mixed_precision, self.use_gradient_checkpointing.
    """

    def _setup_device_and_config(self):
        """Setup device configuration with support for CPU, GPU, Vulkan, DDP, and CPU+GPU."""
        from commons.utils.device_utils import check_vulkan_available, resolve_device

        device_mode = self.device_mode
        gpu_indices = self.gpu_indices or []
        use_vulkan = self.use_vulkan

        # --- CPU puro ---
        if device_mode == 'cpu':
            if self.rank == 0:
                logger.info("=" * 80)
                logger.info("CPU-ONLY MODE ENABLED")
                logger.info("=" * 80)
            self.use_gpu = False
            self.use_mixed_precision = False
            self.use_gradient_checkpointing = False
            return torch.device('cpu')

        # --- Vulkan ---
        if use_vulkan:
            if not check_vulkan_available():
                raise RuntimeError("Vulkan requested but not available on this system")
            if self.rank == 0:
                logger.info("=" * 80)
                logger.info("VULKAN MODE ENABLED")
                logger.info("=" * 80)
            self.use_gpu = False
            self.use_mixed_precision = False
            self.use_gradient_checkpointing = False
            return torch.device('vulkan')

        # --- GPU (una o multiples) ---
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.use_gpu = False
            self.use_mixed_precision = False
            self.use_gradient_checkpointing = False
            return torch.device('cpu')

        # Set device for this DDP process
        torch.cuda.set_device(self.local_rank)

        # Log detected GPUs (rank-0 only)
        num_gpus = torch.cuda.device_count()
        if self.rank == 0:
            logger.info("=" * 80)
            logger.info(f"GPU DETECTED: {num_gpus} device(s)")
            for i in range(num_gpus):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                mem = props.total_memory / (1024 ** 3)
                cc = f"{props.major}.{props.minor}"
                logger.info(f"  GPU {i}: {name} | VRAM: {mem:.2f} GB | Compute: {cc}")
            logger.info("=" * 80)

        # DDP info
        if self.world_size > 1 and self.rank == 0:
            logger.info(f"DDP mode: {self.world_size} processes, local_rank={self.local_rank}")

        # Configure for specific GPU types
        device_idx = self.local_rank
        device_name = torch.cuda.get_device_name(device_idx)

        if "K80" in device_name or "Tesla" in device_name:
            if self.rank == 0:
                logger.info("Tesla K80 GPU detected - using optimized configuration")
            self.use_gradient_checkpointing = True
            torch.cuda.set_per_process_memory_fraction(0.9)
        else:
            self.use_gradient_checkpointing = False

        self.use_gpu = True
        self.use_mixed_precision = True

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if self.rank == 0:
            logger.info("CuDNN optimization enabled")
            logger.info(f"Mixed Precision Training: {'ENABLED' if self.use_mixed_precision else 'DISABLED'}")
            logger.info(f"Gradient Checkpointing: {'ENABLED' if self.use_gradient_checkpointing else 'DISABLED'}")

        device = torch.device(f'cuda:{device_idx}')
        if self.rank == 0:
            logger.info(f"Training Device: {device}")
        return device

    def _setup_model_with_device_strategy(self, model, device):
        """Setup model with CPU, GPU, DDP, or CPU+GPU device strategy."""
        from commons.utils.device_utils import calculate_layers_for_vram

        device_mode = self.device_mode

        # --- CPU puro ---
        if device_mode == 'cpu':
            model = model.to(device)
            if self.rank == 0:
                logger.info("Model deployed on CPU")
            return model

        # --- CPU+GPU: distribucion por capas ---
        if device_mode == 'cpu+gpu':
            return self._setup_model_cpu_gpu_split(model, device)

        # --- GPU estandar (1 o DDP) ---
        try:
            model = model.to(device)

            # DDP si multiples procesos
            if self.world_size > 1:
                import torch.distributed as dist
                from torch.nn.parallel import DistributedDataParallel as DDP

                if not dist.is_initialized():
                    os.environ['MASTER_ADDR'] = getattr(self.config, 'master_addr', 'localhost')
                    os.environ['MASTER_PORT'] = str(getattr(self.config, 'master_port', 29500))
                    dist.init_process_group(
                        backend='nccl',
                        rank=self.rank,
                        world_size=self.world_size
                    )

            # Enable gradient checkpointing BEFORE DDP wrapping
            if self.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                if self.rank == 0:
                    logger.info("Gradient checkpointing enabled for memory efficiency")

            if self.world_size > 1:
                model = DDP(model, device_ids=[self.local_rank])
                if self.rank == 0:
                    logger.info(f"Model wrapped with DDP on {self.world_size} processes, local_rank={self.local_rank}")

            if self.rank == 0:
                logger.info(f"Model deployed on GPU: {device}")
            return model

        except RuntimeError as e:
            logger.warning(f"Could not move model to GPU: {e}, falling back to CPU")
            model = model.to(torch.device("cpu"))
            self.use_gpu = False
            self.use_mixed_precision = False
            return model

    def _setup_model_cpu_gpu_split(self, model, gpu_device):
        """Distribute model layers across CPU and GPU based on VRAM budget."""
        from commons.utils.device_utils import calculate_layers_for_vram

        model = model.to('cpu')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get transformer layers
        transformer_layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            transformer_layers = model.model.transformer.h
        if transformer_layers is None or len(transformer_layers) == 0:
            logger.warning("Cannot split model: no transformer layers found. Using GPU only.")
            return model.to(gpu_device)

        num_layers = len(transformer_layers)
        gpu_props = torch.cuda.get_device_properties(gpu_device)
        gpu_memory_gb = gpu_props.total_memory / (1024 ** 3)
        vram_budget = gpu_memory_gb * 0.80

        # Estimate memory per layer
        total_layer_params = sum(p.numel() for p in transformer_layers.parameters())
        params_per_layer = total_layer_params / num_layers
        mem_per_layer_gb = (params_per_layer * 4) / (1024 ** 3)

        # Embeddings + lm_head memory
        overhead_params = 0
        if hasattr(model.model.transformer, 'wte'):
            overhead_params += sum(p.numel() for p in model.model.transformer.wte.parameters())
        if hasattr(model.model.transformer, 'wpe'):
            overhead_params += sum(p.numel() for p in model.model.transformer.wpe.parameters())
        if hasattr(model, 'lm_head'):
            overhead_params += sum(p.numel() for p in model.lm_head.parameters())
        overhead_gb = (overhead_params * 4) / (1024 ** 3)

        # Calculate layers that fit
        available_for_layers = vram_budget - overhead_gb
        layers_on_gpu = min(num_layers, max(1, int(available_for_layers / mem_per_layer_gb))) if available_for_layers > 0 else 0

        logger.info("=" * 80)
        logger.info("CPU+GPU MODEL SPLIT")
        logger.info(f"  GPU: {gpu_props.name} ({gpu_memory_gb:.2f} GB)")
        logger.info(f"  Total layers: {num_layers}")
        logger.info(f"  VRAM budget: {vram_budget:.2f} GB")
        logger.info(f"  Overhead (embeddings + head): {overhead_gb:.4f} GB")
        logger.info(f"  Layers on GPU: {layers_on_gpu}")
        logger.info(f"  Layers on CPU: {num_layers - layers_on_gpu}")
        logger.info("=" * 80)

        # Move embeddings + head to GPU
        if hasattr(model.model.transformer, 'wte'):
            model.model.transformer.wte = model.model.transformer.wte.to(gpu_device)
        if hasattr(model.model.transformer, 'wpe'):
            model.model.transformer.wpe = model.model.transformer.wpe.to(gpu_device)
        if hasattr(model, 'lm_head'):
            model.lm_head = model.lm_head.to(gpu_device)

        # Move first N layers to GPU
        for i in range(layers_on_gpu):
            transformer_layers[i] = transformer_layers[i].to(gpu_device)

        self._gpu_layers_count = layers_on_gpu
        self._gpu_device = gpu_device

        # Install custom forward wrapper to move activations between CPU/GPU
        self._install_cpu_gpu_forward_wrapper(model)

        return model

    def _install_cpu_gpu_forward_wrapper(self, model):
        """Wrap transformer forward to move activations at CPU/GPU boundary."""
        split_index = self._gpu_layers_count
        gpu_device = self._gpu_device

        transformer = model.model.transformer
        original_transformer_forward = transformer.forward

        def cpu_gpu_forward(*args, **kwargs):
            from transformers.modeling_outputs import BaseModelOutputWithPast

            input_ids = kwargs.get('input_ids', args[0] if args else None)
            attention_mask = kwargs.get('attention_mask', None)
            position_ids = kwargs.get('position_ids', None)

            if input_ids is not None:
                batch_size, seq_len = input_ids.shape
                if position_ids is None:
                    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

                hidden_states = transformer.wte(input_ids) + transformer.wpe(position_ids)

                for i, layer in enumerate(transformer.h):
                    if i == split_index and hidden_states.device.type != 'cpu':
                        hidden_states = hidden_states.to('cpu')
                    layer_output = layer(hidden_states)
                    hidden_states = layer_output[0]

                hidden_states = transformer.ln_f(hidden_states)

                if hidden_states.device.type != gpu_device.type:
                    hidden_states = hidden_states.to(gpu_device)

                return BaseModelOutputWithPast(last_hidden_state=hidden_states)

            return original_transformer_forward(*args, **kwargs)

        transformer.forward = cpu_gpu_forward
