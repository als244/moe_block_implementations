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


        ### MODIFICATIONS!
        
        # flattening the selected experts to a 1-D tensor
        # has the format of: 
        #   - [tok_0_exp_0, tok_0_exp_1, ..., tok_0_exp_top_k-1, top_1_exp_0, tok_1_exp_1, ... tok_seq_len*bsz-1_exp_0, ...tok_seq_len*bsz-1_exp_top_k-1]
        selected_experts_flattened = torch.ravel(selected_experts)

        # get the indices that represented sorted experts
        #  - used later for sorted when other values are combined
        perm = torch.argsort(selected_experts_flattened)
        
        # create 1-d tensor of the format:
        #   - [0, 0, ... 0 (top_k times), 1, 1, ..., batch_size * seq_len - 1, batch_size * seq_len - 1, ... (top_k times)]
        # represents token number (i.e. an index into hidden states) corresponding to location in flattened selected experts
        token_rows = torch.repeat_interleave(torch.arange(0, batch_size * sequence_length, device=selected_experts_flattened.device), self.top_k)
        # create 1-d tensor of the format: 
        #   - [0, 1, ..., top_k - 1, 0, 1, .., top_k-1 (repeat batch_size * seq_len times)]
        # represents the ranking within top k (i.e. an index into router weights) correspondinig to location in flattened selected experts
        top_k_vals = torch.tile(torch.arange(0, self.top_k, device=selected_experts_flattened.device), (batch_size * sequence_length,))

        detailed_selected_experts = torch.stack((selected_experts_flattened, token_rows, top_k_vals), dim=0)

        # sorted now
        # dealt with where to obtain hidden states (token row) 
        # and which router weights to use (top_k_vals)
        detailed_selected_experts = detailed_selected_experts.index_select(1, perm)
        
        # will be used to find the indicies where each expert's contiguous run ends
        all_experts = torch.arange(1, self.num_experts + 1, device=detailed_selected_experts.device)

        # each pair of adjancent values reveals the cutoffs for all of the experts
        # if the pair has the same value no tokens will get routed to that expert
        # the formula: expert i should be in charge of tokens in the range of expert_partitions[i:i+1]
        # the last expert's upper bound is the length of detailed_selected_experts
        
        # this is especially nice because we already partioned all the experts so they could be executed in parallel streams
        expert_partitions = torch.searchsorted(detailed_selected_experts[0, :], all_experts)
        
        # utilize an additional stream to perform GPU -> CPU copy without causing global sync
        # doing an async copy even though we need the results immediately
        # normal .to() will cause full device sync because host memory not pinned
        transfer_stream = self.transfer_stream
        
        ## wait for searchsorted to finish before starting transfer
        transfer_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(transfer_stream):
            ## without non_blocking, the entire device is sync!
            ## terrible when doing FSDP!
            cpu_expert_partitions = expert_partitions.to("cpu", non_blocking=True)
        
        # make sure cpu_expert_partitions has completed it's transfer
        # cannot enqueue .tensor_splits() on default gpu kernel before that transfer is done
        # should be a better way (i.e. host-blocking stream.synchronize())
        #   - torch.cuda.default_stream().wait_stream(transfer_stream) doesn't work correctly, idk why
        while not transfer_stream.query():
            continue

        # retrieving top_xs and idxs as tuple of GPU-tensors per expert
        #   - less data transferred and less sync needed between host and device so is best solution
        top_xs = detailed_selected_experts[1, :].tensor_split(cpu_expert_partitions)
        idxs = detailed_selected_experts[2, :].tensor_split(cpu_expert_partitions)
   

        # MODIFIED
        #  - adding a flat version for easy indexing from gpu tensor dervied from top_x and idx
        routing_weights_flat = routing_weights.view(-1)

        for expert_idx in range(self.num_experts):
            
            expert_layer = self.experts[expert_idx]

            # MODIFIED
            top_x, idx = top_xs[expert_idx], idxs[expert_idx]

            if top_x.numel() == 0:
                continue

            # MODIFIED
            current_state = hidden_states.index_select(0, top_x)
            
            # MODIFIED
            expert_routing_inds_flat = top_x * self.top_k + idx
            expert_routing_weights = routing_weights_flat.index_select(0, expert_routing_inds_flat).unsqueeze(1)

            expert_out = expert_layer(current_state)

            current_hidden_states = expert_out * expert_routing_weights
   
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits