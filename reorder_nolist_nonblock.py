class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.hidden_dim = config.model.hidden_size
        self.ffn_dim = config.model.intermediate_size
        self.num_experts = config.model.expert_module.num_experts
        self.top_k = config.model.expert_module.top_k_experts

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False, device=device)

        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config, device=device) for _ in range(self.num_experts)])

        # MODIFIED
        #  - will be used as a mechanism to perform host to device transfers without triggering device sync
        self.transfer_stream = torch.cuda.Stream()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # MODIFIED
        #   - preventing torch.where() from triggering a cudaDeviceSynchronize that adds extra overhead

        # Needs expert_mask to be on cpu to be "non-blocking" (incurring extra overhead of device sync)
        transfer_stream = self.transfer_stream
        ## wait for one_hot to finish before starting transfer
        transfer_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(transfer_stream):
            ## without non_blocking, the entire device is sync!
            ## terrible when doing FSDP!
            cpu_expert_mask = expert_mask.to("cpu", non_blocking=True)

        # cpu expert mask acutally needs to finish transferring
        # should be a better way (i.e. host-blocking stream.synchronize())
        #   - torch.cuda.default_stream().wait_stream(transfer_stream) doesn't work correctly, idk why
        while not transfer_stream.query():
            continue

        # torch where returned tensors live on cpu now
        # same "algorithm" as orig
        idxs, top_xs = [], []
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(cpu_expert_mask[expert_idx])
            idxs.append(idx)
            top_xs.append(top_x)

        # now need to manually transfer back
        # need to wait for torch calls to finish
        transfer_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(transfer_stream):
            for expert_idx in range(self.num_experts):
                idxs[expert_idx] = idxs[expert_idx].to(hidden_states.device, non_blocking=True)
                top_xs[expert_idx] = top_xs[expert_idx].to(hidden_states.device, non_blocking=True)

        # now need to ensure these tensors got back
        # should be a better way (i.e. host-blocking stream.synchronize())
        #   - torch.cuda.default_stream().wait_stream(transfer_stream) doesn't work correctly, idk why
        while not transfer_stream.query():
            continue

        # MODIFIED
        #  -- adding a flat version for easy indexing from gpu tensor dervied from top_x and idx
        routing_weights_flat = routing_weights.view(-1)

        for expert_idx in range(self.num_experts):

            expert_layer = self.experts[expert_idx]
            
            # MODIFIED
            idx, top_x = idxs[expert_idx], top_xs[expert_idx]

            if top_x.shape[0] == 0:
                continue

            # MODIFIED
            #   - preventing torch from calling .tolist() from triggering a cudaDeviceSynchronize that adds extra overhead
            current_state = hidden_states.index_select(0, top_x)
          
            # MODIFIED
            #   - preventing torch from calling .tolist() from triggering a cudaDeviceSynchronize that adds extra overhead          
            expert_routing_inds_flat = top_x * self.top_k + idx
            expert_routing_weights = routing_weights_flat.index_select(0, expert_routing_inds_flat).unsqueeze(1)
           
            expert_out = expert_layer(current_state)

            current_hidden_states = expert_out * expert_routing_weights

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))


        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits