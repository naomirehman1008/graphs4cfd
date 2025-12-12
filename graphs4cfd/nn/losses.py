import torch
import torch.nn as nn
import torch.nn.functional as F

def _flatten_batched_index(idx, B, N):
    """idx: [B,E] (long) -> [B*E] with per-batch offsets for indexing length B*N arrays."""
    device = idx.device
    off = torch.arange(B, device=device).view(B, 1) * N
    return (idx + off).reshape(-1)

def _edge_vectors(x, edges):
    """x: [B,N,2], edges: [B,E,2] -> dir: [B,E,2], len2: [B,E,1]."""
    send = torch.gather(x, 1, edges[..., 0].unsqueeze(-1).repeat(1, 1, 2))
    recv = torch.gather(x, 1, edges[..., 1].unsqueeze(-1).repeat(1, 1, 2))
    d = recv - send
    len2 = (d ** 2).sum(-1, keepdim=True) + 1e-8
    return d, len2

def _graph_grad_scalar(x, edges, s):
    """Gradient of scalar s at nodes using edge differences.
       x:[B,N,2], edges:[B,E,2], s:[B,N] -> grad:[B,N,2]"""
    B, N, _ = x.shape
    E = edges.shape[1]
    d, len2 = _edge_vectors(x, edges)              # [B,E,2], [B,E,1]
    unit = d / torch.sqrt(len2)
    w = 1.0 / len2                                  # [B,E,1]
    s_i = torch.gather(s, 1, edges[..., 0])
    s_j = torch.gather(s, 1, edges[..., 1])
    ds = (s_j - s_i).unsqueeze(-1)                 # [B,E,1]

    contrib_i = w * ds * unit                      # [B,E,2]
    contrib_j = -contrib_i

    # scatter add to nodes
    grad = torch.zeros(B * N, 2, device=x.device, dtype=x.dtype)
    den  = torch.zeros(B * N, 1, device=x.device, dtype=x.dtype)
    idx_i = _flatten_batched_index(edges[..., 0], B, N)
    idx_j = _flatten_batched_index(edges[..., 1], B, N)
    grad.index_add_(0, idx_i, contrib_i.reshape(B * E, 2))
    grad.index_add_(0, idx_j, contrib_j.reshape(B * E, 2))
    den.index_add_(0, idx_i, w.reshape(B * E, 1))
    den.index_add_(0, idx_j, w.reshape(B * E, 1))

    grad = grad / (den + 1e-8)
    return grad.view(B, N, 2)

def _graph_grad_vector(x, edges, u):
    """Jacobian of vector u per node. u:[B,N,2] -> [B,N,2,2] where J[c,k]=∂u_c/∂x_k."""
    J0 = _graph_grad_scalar(x, edges, u[..., 0])   # ∂u_x/∂x,∂u_x/∂y
    J1 = _graph_grad_scalar(x, edges, u[..., 1])   # ∂u_y/∂x,∂u_y/∂y
    return torch.stack([J0, J1], dim=2)            # [B,N,2,2]

def _graph_divergence(x, edges, u):
    """div(u) at nodes, u:[B,N,2] -> [B,N,1]."""
    J = _graph_grad_vector(x, edges, u)            # [B,N,2,2]
    div = J[..., 0, 0] + J[..., 1, 1]              # trace
    return div.unsqueeze(-1)

def _graph_laplacian(x, edges, u):
    """Graph Laplacian Δu ≈ sum_j w_ij (u_j - u_i). u:[B,N,2] -> [B,N,2]."""
    B, N, _ = x.shape
    E = edges.shape[1]
    d, len2 = _edge_vectors(x, edges)              # for weights only
    w = 1.0 / (len2 + 1e-8)

    u_i = torch.gather(u, 1, edges[..., 0].unsqueeze(-1).repeat(1, 1, 2))
    u_j = torch.gather(u, 1, edges[..., 1].unsqueeze(-1).repeat(1, 1, 2))
    diff = (u_j - u_i) * w                         # [B,E,2]

    lap = torch.zeros(B * N, 2, device=x.device, dtype=x.dtype)
    idx_i = _flatten_batched_index(edges[..., 0], B, N)
    idx_j = _flatten_batched_index(edges[..., 1], B, N)

    # add contributions for i and j (symmetrize)
    lap.index_add_(0, idx_i, diff.reshape(B * E, 2))
    lap.index_add_(0, idx_j, (-diff).reshape(B * E, 2))

    # degree-normalize
    deg = torch.zeros(B * N, 1, device=x.device, dtype=x.dtype)
    deg.index_add_(0, idx_i, w.reshape(B * E, 1))
    deg.index_add_(0, idx_j, w.reshape(B * E, 1))
    lap = lap / (deg + 1e-8)
    return lap.view(B, N, 2)

def _spectrum_loss(x, u, self_hat=None, grid=64):
    """Coarse isotropic spectrum/structure penalty. x:[B,N,2] in world coords; u:[B,N,2].
       1) Rasterize to grid; 2) 2D FFT; 3) match log power spectrum to its own smoothed version
       (self-supervised regularizer) — if 'self_hat' (ground truth) is not provided.
       In supervised setting you can pass a GT velocity field as 'self_hat'.
    """
    B, N, _ = x.shape
    device = x.device
    # normalize coords to [0,1]
    xmin, _ = x.amin(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)
    rng = (x.amax(dim=1, keepdim=True) - xmin).clamp(min=1e-6)
    xn = (x - xmin) / rng
    gx = (xn[..., 0] * (grid - 1)).long().clamp(0, grid - 1)
    gy = (xn[..., 1] * (grid - 1)).long().clamp(0, grid - 1)
    idx = gy * grid + gx                                # [B,N]

    # scatter-average velocities into grid
    V = torch.zeros(B, grid * grid, 2, device=device, dtype=u.dtype)
    C = torch.zeros(B, grid * grid, 1, device=device, dtype=u.dtype)
    for b in range(B):
        V[b].index_add_(0, idx[b], u[b])
        ones = torch.ones_like(idx[b], dtype=u.dtype, device=device).unsqueeze(-1)
        C[b].index_add_(0, idx[b], ones)
    V = V / (C + 1e-8)
    V = V.view(B, grid, grid, 2).permute(0, 3, 1, 2).contiguous()   # [B,2,H,W]

    # power spectrum magnitude
    F = torch.fft.rfftn(V, dim=(-2, -1))
    P = (F.real ** 2 + F.imag ** 2).mean(dim=1)   # [B, H, W_r]
    # build radial bins
    H, W = P.shape[-2], P.shape[-1]
    ky = torch.fft.fftfreq(H, d=1.0).to(device)
    kx = torch.fft.rfftfreq(W * 2 - 2, d=1.0).to(device)  # inverse of rfftn shape
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    R = torch.sqrt(KX ** 2 + KY ** 2)                     # [H, W_r]
    nb = H // 2
    bins = torch.linspace(0, R.max() + 1e-6, nb, device=device)
    loss = 0.0
    for b in range(B):
        pb = P[b]
        num = torch.zeros(nb - 1, device=device, dtype=pb.dtype)
        den = torch.zeros_like(num)
        for i in range(nb - 1):
            m = (R >= bins[i]) & (R < bins[i + 1])
            val = pb[m]
            if self_hat is not None:
                Vgt = self_hat[b]  # not used here; kept for API symmetry
            num[i] = val.sum()
            den[i] = m.sum()
        Ek = (num / (den + 1e-8)).clamp(min=1e-12)
        Ek_smooth = torch.nn.functional.avg_pool1d(Ek.view(1, 1, -1), 3, stride=1, padding=1)[0, 0]
        loss = loss + ((torch.log(Ek) - torch.log(Ek_smooth)) ** 2).mean()
    return loss / B

#  keep only valid-node edges and one direction per undirected pair (for physics ops)
def _filter_edges_undirected_once(edges_bt, valid_nodes_mask):
    """
    edges_bt: [E,2] long, valid_nodes_mask: [N] bool
    Returns edges where both endpoints are valid AND i<j (undirected used once).
    """
    if edges_bt.numel() == 0:
        return edges_bt
    i = edges_bt[:, 0]
    j = edges_bt[:, 1]
    valid = valid_nodes_mask[i] & valid_nodes_mask[j]
    e = edges_bt[valid]
    if e.numel() == 0:
        return e
    mask_half = e[:, 0] < e[:, 1]
    return e[mask_half]



def reshape_graph_for_physics(graph, pred, target, n_f):
    """
    Reshape graph field and target from [n_nodes * batch, n_f * n_t] to [batch, t, n_nodes, n_f].
    Resgaoe grapge dges fron [batch * n_edges, 2] to [batch, n_edges, 2]
    Adds padding and a mask to handle varying meshes within the same batch.
    """
    
    field = getattr(graph, 'field')
    pos = getattr(graph, 'pos')
    edge_index = getattr(graph, 'edge_index').permute(1, 0)
    bound = getattr(graph, 'bound')
    omega = getattr(graph, 'omega')
    glob = getattr(graph, 'glob')

    #print(f"field shape: {field.shape}")
    #print(f"target shape: {target.shape}")
    #print(f"pred shape: {pred.shape}")
    #print(f"bound shape: {bound.shape}")
    #print(f"omega shape: {omega.shape}")
    #print(f"glob shape: {glob.shape}")


    batch = getattr(graph, 'batch')
    batch_size = batch.max().item() + 1

    n_nodes = torch.bincount(batch) # get number of nodes in each frame

    max_n_nodes = n_nodes.max().item()
    n_t = int(target.shape[1] / n_f) # get number of time steps
    field_t = int(field.shape[1] / n_f) # number of input time steps may be different

    fields = []
    targets = []
    preds = []
    poses = []
    edges = []
    bounds = []
    omegas = []
    globs = []
    node_masks = []
    edge_masks = []

    n_edges = []

    start_idx = 0
    end_idx = 0
    # get max edges for padding
    for b in range(batch_size):
        end_idx = start_idx + n_nodes[b].item()
        edge_mask = torch.isin(edge_index[:, 0], torch.arange(start_idx, end_idx, device=pos.device)) # check if the souce node is in the node set (assume no criss cross between graphs)
        edge_b = edge_index[edge_mask] - start_idx
        n_edges.append(edge_b.shape[0])

        start_idx = end_idx


    max_n_edges = max(n_edges)

    start_idx = 0
    end_idx = 0
    for b in range(batch_size):
        end_idx = start_idx + n_nodes[b].item()

        field_b     = field[batch == b]
        target_b    = target[batch == b]
        pred_b      = pred[batch == b]
        pos_b       = pos[batch == b]
        bound_b     = bound[batch == b]
        omega_b     = omega[batch == b]
        glob_b      = glob[batch == b]
        
        edge_mask = torch.isin(edge_index[:, 0], torch.arange(start_idx, end_idx, device=pos.device))
        edge_b = edge_index[edge_mask] - start_idx

        start_idx = end_idx

        b_n_edges = n_edges[b]
        b_n_nodes = n_nodes[b]

        padding = max_n_nodes - b_n_nodes
        field_b     = F.pad(field_b, (0, 0, 0, padding), 'constant') # pad to max nodes
        target_b    = F.pad(target_b, (0, 0, 0, padding), 'constant') # pad to max nodes
        pred_b      = F.pad(pred_b, (0, 0, 0, padding), 'constant') # pad to max nodes
        pos_b       = F.pad(pos_b, (0, 0, 0, padding), 'constant') # pad to max nodes
        bound_b      = F.pad(bound_b, (0, padding), 'constant') # pad to max nodes
        omega_b      = F.pad(omega_b, (0, 0, 0, padding), 'constant') # pad to max nodes
        glob_b       = F.pad(glob_b, (0, 0, 0, padding), 'constant') # pad to max nodes

        edge_b      = F.pad(edge_b, (0, 0, 0, max_n_edges - b_n_edges), 'constant') # pad to max edges

        # do I ever use these?
        node_mask_b      = torch.cat((torch.ones(b_n_nodes, dtype=torch.int, device=pos.device), torch.zeros(padding, dtype=torch.int, device=pos.device)))
        edge_mask_b      = torch.cat((torch.ones(b_n_edges, dtype=torch.int, device=pos.device), torch.zeros(max_n_edges - b_n_edges,  dtype=torch.int, device=pos.device)))

        # reshape along time dimension, permute to expected dim order
        field_b = field_b.view(max_n_nodes, field_t, n_f).permute(1, 0, 2)
        target_b = target_b.view(max_n_nodes, n_t, n_f).permute(1, 0, 2)
        pred_b = pred_b.view(max_n_nodes, n_t, n_f).permute(1, 0, 2)

        # add initial state to pred ( last field)
        pred_b = torch.cat((field_b[-1].unsqueeze(0), pred_b), dim=0)

        # repeate along time dimension
        pos_b = pos_b.repeat(n_t+1, 1, 1)
        bound_b = bound_b.repeat(n_t+1, 1)
        omega_b = omega_b.repeat(n_t+1, 1, 1)
        glob_b = glob_b.repeat(n_t+1, 1, 1)
        edge_b = edge_b.repeat(n_t+1, 1, 1)

        node_mask_b = node_mask_b.repeat(n_t+1, 1)
        edge_mask_b = edge_mask_b.repeat(n_t+1, 1)

        fields.append(field_b)
        targets.append(target_b)
        preds.append(pred_b)
        poses.append(pos_b)
        edges.append(edge_b)
        bounds.append(bound_b)
        omegas.append(omega_b)
        globs.append(glob_b)
        node_masks.append(node_mask_b)
        edge_masks.append(edge_mask_b)

    reshaped_field = torch.stack(fields, dim=0)
    reshaped_target = torch.stack(targets, dim=0)
    reshaped_pred = torch.stack(preds, dim=0)
    reshaped_pos = torch.stack(poses, dim=0)
    reshaped_edges = torch.stack(edges, dim=0)
    reshaped_bound = torch.stack(bounds, dim=0)
    reshaped_omega = torch.stack(omegas, dim=0)
    reshaped_node_mask = torch.stack(node_masks, dim=0)
    reshaped_edge_mask = torch.stack(edge_masks, dim=0)

    #print(f"reshaped_field shape: {reshaped_field.shape}")
    #print(f"reshaped_target shape: {reshaped_target.shape}")
    #print(f"reshaped_pred shape: {reshaped_pred.shape}")
    #print(f"reshaped_pos shape: {reshaped_pos.shape}")
    #print(f"reshaped_edges shape: {reshaped_edges.shape}")
    #print(f"reshaped_bound shape: {reshaped_bound.shape}")
    #print(f"reshaped_omega shape: {reshaped_omega.shape}")
    #print(f"reshaped_node_mask shape: {reshaped_node_mask.shape}")
    #print(f"reshaped_edge_mask shape: {reshaped_edge_mask.shape}")

    physics_graph = {
        'field'     : reshaped_field,
        'target'    : reshaped_target,
        'pred'      : reshaped_pred,
        'pos'       : reshaped_pos,
        'edge_index' : reshaped_edges,
        'bound'     : reshaped_bound,
        'omega'     : reshaped_omega,
        'edge_mask'      : reshaped_edge_mask,
        'node_mask' : reshaped_node_mask,
        'batch'     : batch,
        'max_nodes' : max_n_nodes,
        'n_t'       : n_t
    }

    return physics_graph

class GraphLoss(nn.Module):
    def __init__(self, lambda_d=0):
        super().__init__()
        self.lambda_d  = lambda_d

    def forward(self, graph, pred, target, components=None):
        loss = F.mse_loss(pred, target)
        if self.lambda_d > 0:
            dirichlet_boundary = (graph.omega[:,0] == 1)
            if dirichlet_boundary.any():
                loss += self.lambda_d*F.l1_loss(pred[dirichlet_boundary], target[dirichlet_boundary])
        components['mse'] = loss.detach()
        components['div'] = 0
        components['mom'] = 0
        components['bc'] = 0
        components['spec'] = 0
        return loss

class GraphLossWPhysicsLoss(nn.Module):
    def __init__(self, lambda_d=0, mse_weight=1, div_weight=0, mom_weight=0, bc_weight=0, spec_weight=0, dt=1):
        super().__init__()
        self.lambda_d  = lambda_d
        self.mse_weight = mse_weight
        self.div_weight = div_weight
        self.mom_weight = mom_weight
        self.bc_weight  = bc_weight
        self.spec_weight = spec_weight
        self.dt = dt

    def compute_physics_loss(
        self,
        physics_graph,
    ):
        mesh_pos = physics_graph['pos']
        edges = physics_graph['edge_index']
        node_type = physics_graph['bound']
        velocity_hat = physics_graph['pred']
        node_mask = physics_graph['node_mask']
        edge_mask = physics_graph['edge_mask']

        debug_print = False#overrides.pop("debug_print", False)
        debug_tag   = None#overrides.pop("debug_tag",   None)

        rho = 1.0
        nu = 0.001 

        dt = 1.0
        lam_div = self.div_weight
        lam_mom = self.mom_weight
        lam_bc = self.bc_weight
        lam_spec = self.spec_weight

        B, T, N, C = velocity_hat.shape
        device = velocity_hat.device
        eps = 1e-8

        u_pred = velocity_hat[..., 0:2]                  # [B,T,N,2]
        p_pred = velocity_hat[..., 2:3] if C >= 3 else torch.zeros(B, T, N, 1, device=device, dtype=velocity_hat.dtype)

        # evaluate physics on steps t=1..T-1 (newly predicted states)
        steps = range(1, T)

        div_loss = torch.zeros((), device=device)
        mom_loss = torch.zeros((), device=device)
        bc_loss  = torch.zeros((), device=device)
        spec_loss = torch.zeros((), device=device)

        #  per-term counters to average robustly when some slices have no valid edges
        div_count = mom_count = bc_count = spec_count = 0  # 

        for t in steps:
            x = mesh_pos[:, t]          # [B,N,2]
            e = edges[:, t]             # [B,E,2]
            u = u_pred[:, t]            # [B,N,2]
            u_prev = u_pred[:, t-1]
            p = p_pred[:, t]            # [B,N,1]
            n_m = node_mask[:, t]       # [B, N]
            e_m = edge_mask[:, t]       # [B, N]

            # loop per batch to filter edges (valid nodes only) and keep undirected-once
            for b in range(B): 
                # valid (non-ghost) nodes = not DISABLE (one-hot channel 2 is 1 for ghost/disabled)
                valid_nodes_b = (n_m[b] == 1) # [N] bool   
                e_b = e[b]  # [E,2] 
                e_b_f = e_b[(e_m[b] == 1)]
                #e_b_f = _filter_edges_undirected_once(e_b, valid_nodes_b)   
                if e_b_f.numel() == 0 or valid_nodes_b.sum() == 0:  # nothing to compute   
                    continue   

                x_b = x[b][valid_nodes_b]        # [N,2]   
                u_b = u[b][valid_nodes_b]        # [N,2]   
                u_prev_b = u_prev[b][valid_nodes_b]  # [N,2]   
                p_b = p[b][valid_nodes_b]        # [N,1]   

                # singleton-batch calls to graph operators   
                x1 = x_b.unsqueeze(0)              # [1,N,2]   
                e1 = e_b_f.unsqueeze(0)            # [1,E',2]  
                u1 = u_b.unsqueeze(0)              # [1,N,2]   
                # divergence, laplacian, grad p, jacobian   
                div_u_b = _graph_divergence(x1, e1, u1)[0]         # [N,1]   
                lap_u_b = _graph_laplacian(x1, e1, u1)[0]          # [N,2]   
                grad_p_b = _graph_grad_scalar(x1, e1, p_b[..., 0].unsqueeze(0))[0]  # [N,2]   
                jac_u_b  = _graph_grad_vector(x1, e1, u1)[0]       # [N,2,2]   
                adv_b = torch.einsum('nik,ni->nk', jac_u_b, u_b)   # (u·∇)u -> [N,2]   
                du_dt_b = (u_b - u_prev_b) / dt                    # [N,2]   

                # masks for averaging on real nodes only   
                vm = valid_nodes_b  # [N]   

                # loss weights
                if lam_div > 0:   
                    div_loss = div_loss + (div_u_b.squeeze(-1) ** 2).mean()   
                    div_count += 1  
                if lam_mom > 0:   
                    mom_res_b = du_dt_b + adv_b + (grad_p_b / (rho + eps)) - nu * lap_u_b   
                    mom_loss = mom_loss + (mom_res_b ** 2).mean()   
                    mom_count += 1   
                if lam_bc > 0:   
                    wall = node_type[b, t, :, NODE_WALL] == 1   
                    inlet = node_type[b, t, :, NODE_INPUT] == 1   
                    disable = node_type[b, t, :, NODE_DISABLE] == 1   
                    m = (wall | inlet | disable)   
                    if m.any():   
                        bc_loss = bc_loss + (u_b[m] ** 2).mean()   
                        bc_count += 1   
                if lam_spec > 0:   
                    xb_val = x_b  # [Nv,2]   
                    ub_val = u_b  # [Nv,2]   
                    # spectrum on valid nodes only (singleton batch)   
                    spec_loss = spec_loss + _spectrum_loss(xb_val.unsqueeze(0), ub_val.unsqueeze(0),
                                                          self_hat=None)   
                    spec_count += 1   

        # Average across evaluated slices (use per-term counters)   
        if lam_div > 0: div_loss = div_loss / max(1, div_count)   
        if lam_mom > 0: mom_loss = mom_loss / max(1, mom_count)   
        if lam_bc  > 0: bc_loss  = bc_loss  / max(1, bc_count)    
        if lam_spec> 0: spec_loss= spec_loss/ max(1, spec_count)  

        total = lam_div * div_loss + lam_mom * mom_loss + lam_bc * bc_loss + lam_spec * spec_loss

        if debug_print:
            tag = f" [{debug_tag}]" if debug_tag is not None else ""  
            print(
                f"[PHYS]{tag} div={div_loss.item():.3e} (λ={lam_div:g}) | "
                f"mom={mom_loss.item():.3e} (λ={lam_mom:g}) | "
                f"bc={bc_loss.item():.3e} (λ={lam_bc:g}) | "
                f"spec={spec_loss.item():.3e} (λ={lam_spec:g}) | "
                f"TOTAL={total.item():.3e}", flush=True
            )

        terms = dict(div=div_loss.detach(), mom=mom_loss.detach(), bc=bc_loss.detach(), spec=spec_loss.detach())
        return total, terms

    def forward(self, graph, pred, target, components=None):
        total_loss = 0
        mse_loss = F.mse_loss(pred, target)
        if self.lambda_d > 0:
            dirichlet_boundary = (graph.omega[:,0] == 1)
            if dirichlet_boundary.any():
                mse_loss += self.lambda_d*F.l1_loss(pred[dirichlet_boundary], target[dirichlet_boundary])
        physics_graph = reshape_graph_for_physics(graph, pred, target, n_f=3)
        physics_loss, terms = self.compute_physics_loss(physics_graph)
        terms['mse'] = mse_loss.detach()
        total_loss = self.mse_weight * mse_loss + physics_loss # (already weighted)
        if (components is not None):
            components.update(terms)
        return total_loss
