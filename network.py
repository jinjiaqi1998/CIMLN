import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ViewEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ViewEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, hidden_dim),
        )
    
    def forward(self, x):
        return self.encoder(x)


class ViewDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(ViewDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, output_dim),
        )
    
    def forward(self, h):
        return self.decoder(h)


class MetaGenerator(nn.Module):
    def __init__(self, hidden_dim):
        super(MetaGenerator, self).__init__()
        self.hidden_dim = hidden_dim

        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.var_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Softplus()
        )
    
    def forward(self, h_agg, context):
        input_feat = torch.cat([h_agg, context], dim=1)
        
        mean = self.mean_net(input_feat)
        var = self.var_net(input_feat) + 1e-8
        
        return mean, var


class CausalInvariantMetaGenerativeNetwork(nn.Module):
    def __init__(self, view_dims, hidden_dim, cluster_num, temperature=1.0):
        super(CausalInvariantMetaGenerativeNetwork, self).__init__()
        self.view_num = len(view_dims)
        self.hidden_dim = hidden_dim
        self.cluster_num = cluster_num
        self.temperature = temperature

        self.encoders = nn.ModuleList([
            ViewEncoder(dim, hidden_dim) for dim in view_dims
        ])
        self.decoders = nn.ModuleList([
            ViewDecoder(hidden_dim, dim) for dim in view_dims
        ])

        self.meta_generator = MetaGenerator(hidden_dim)

        self.fusion_projectors = nn.ModuleList([
            nn.Linear(hidden_dim * self.view_num, hidden_dim) for _ in range(self.view_num)
        ])

        self.cluster_centers = nn.Parameter(
            torch.randn(cluster_num, hidden_dim * self.view_num) * 0.1,
            requires_grad=True
        )
    
    def encode_views(self, xs, miss_vecs):
        batch_size = xs[0].shape[0]
        device = xs[0].device

        h_views = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.view_num)]

        for v in range(self.view_num):
            observed_mask = miss_vecs[v].bool()
            if observed_mask.sum() > 0:
                h_views[v][observed_mask] = self.encoders[v](xs[v][observed_mask])
        
        return h_views
    
    def get_complete_samples_mask(self, miss_vecs):
        miss_vecs_stacked = torch.stack([mv.bool() for mv in miss_vecs], dim=1)
        complete_mask = miss_vecs_stacked.all(dim=1)
        
        return complete_mask
    
    def compute_context_info(self, h_views, miss_vecs, target_view):
        complete_mask = self.get_complete_samples_mask(miss_vecs)
        
        if complete_mask.sum() == 0:
            return torch.zeros(1, self.hidden_dim, device=h_views[0].device)

        h_views_stacked = torch.stack(h_views, dim=1)
        h_agg_complete = h_views_stacked[complete_mask].mean(dim=1)
        h_target_complete = h_views[target_view][complete_mask]

        context = (h_agg_complete * h_target_complete).mean(dim=0, keepdim=True)
        
        return context
    
    def generate_missing_views(self, h_views, miss_vecs):
        generated_views = []

        h_views_stacked = torch.stack(h_views, dim=1)
        miss_vecs_stacked = torch.stack([mv.float() for mv in miss_vecs], dim=1)
        
        for v in range(self.view_num):
            missing_mask = ~miss_vecs[v].bool()
            
            if missing_mask.sum() == 0:
                generated_views.append(h_views[v])
                continue

            context = self.compute_context_info(h_views, miss_vecs, v)
            context = context.expand(missing_mask.sum(), -1)

            missing_samples = h_views_stacked[missing_mask]
            missing_masks = miss_vecs_stacked[missing_mask]

            masked_representations = missing_samples * missing_masks.unsqueeze(-1)

            sum_representations = masked_representations.sum(dim=1)

            observed_counts = missing_masks.sum(dim=1, keepdim=True).clamp(min=1)

            h_agg_missing = sum_representations / observed_counts

            if h_agg_missing.shape[0] == 1 and self.training:
                self.meta_generator.eval()
                mean, var = self.meta_generator(h_agg_missing, context)
                self.meta_generator.train()
            else:
                mean, var = self.meta_generator(h_agg_missing, context)

            h_v_complete = h_views[v].clone()
            h_v_complete[missing_mask] = mean
            generated_views.append(h_v_complete)
        
        return generated_views
    
    def compute_cluster_assignment(self, z_fused):
        distances = torch.cdist(z_fused, self.cluster_centers, p=2.0)
        q = F.softmax(-distances / self.temperature, dim=1)
        return q
    
    def create_counterfactual_mask(self, miss_vecs, beta=0.3):
        miss_vecs_stacked = torch.stack([mv.float() for mv in miss_vecs], dim=1)

        random_vals = torch.rand_like(miss_vecs_stacked)

        cf_mask_stacked = miss_vecs_stacked * (random_vals >= beta).float()

        cf_miss_vecs = [cf_mask_stacked[:, v] for v in range(self.view_num)]
        
        return cf_miss_vecs
    
    def initialize_cluster_centers(self, loader, device):
        from torch_kmeans_best import kmeans_torch
        
        self.eval()
        complete_representations = []
        
        with torch.no_grad():
            for xs, _, miss_vecs, _ in loader:
                for v in range(self.view_num):
                    xs[v] = xs[v].to(device)
                    miss_vecs[v] = miss_vecs[v].to(device)

                h_views = self.encode_views(xs, miss_vecs)

                complete_mask = self.get_complete_samples_mask(miss_vecs)
                
                if complete_mask.sum() > 0:
                    z_fused = torch.cat(h_views, dim=1)
                    complete_representations.append(z_fused[complete_mask])
        
        if len(complete_representations) > 0:
            all_complete = torch.cat(complete_representations, dim=0)
            
            if all_complete.shape[0] >= self.cluster_num:
                centers, labels, inertia = kmeans_torch(all_complete, self.cluster_num)

                self.cluster_centers.data = centers
        
        self.train()
    
    def pretrain(self, xs, miss_vecs):
        h_views = self.encode_views(xs, miss_vecs)

        recon_loss = 0.0
        total_observed = 0
        
        for v in range(self.view_num):
            observed_mask = miss_vecs[v].bool()
            if observed_mask.sum() > 0:
                x_recon_view = self.decoders[v](h_views[v][observed_mask])
                loss_view = F.mse_loss(x_recon_view, xs[v][observed_mask])

                recon_loss += loss_view

                total_observed += observed_mask.sum().item()
        
        if total_observed > 0:
            recon_loss = recon_loss / self.view_num
        
        return recon_loss
    
    def main_train(self, xs, miss_vecs):
        h_views = self.encode_views(xs, miss_vecs)

        h_complete = self.generate_missing_views(h_views, miss_vecs)
        
        recon_loss = 0.0
        total_observed = 0
        
        for v in range(self.view_num):
            observed_mask = miss_vecs[v].bool()
            if observed_mask.sum() > 0:
                x_recon_view = self.decoders[v](h_views[v][observed_mask])
                loss_view = F.mse_loss(x_recon_view, xs[v][observed_mask])

                recon_loss += loss_view

                total_observed += observed_mask.sum().item()
        
        if total_observed > 0:
            recon_loss = recon_loss / self.view_num

        meta_loss = self.compute_meta_loss(h_views, h_complete, miss_vecs)

        causal_loss = self.compute_causal_loss(h_views, h_complete, miss_vecs)
        
        return recon_loss, meta_loss, causal_loss
    
    def compute_meta_loss(self, h_views, h_complete, miss_vecs):
        device = h_views[0].device
        
        if self.view_num <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        meta_loss = 0.0
        valid_views = 0

        for v in range(self.view_num):
            observed_mask = miss_vecs[v].bool()
            
            if observed_mask.sum() == 0:
                continue

            context = self.compute_context_info(h_views, miss_vecs, v)

            h_agg_list = []
            for v_other in range(self.view_num):
                if v_other != v:
                    h_agg_list.append(h_complete[v_other][observed_mask])
            
            if len(h_agg_list) > 0:
                h_agg = torch.stack(h_agg_list, dim=1).mean(dim=1)
                context_expanded = context.expand(observed_mask.sum(), -1)

                if h_agg.shape[0] == 1 and self.training:
                    self.meta_generator.eval()
                    mean, var = self.meta_generator(h_agg, context_expanded)
                    self.meta_generator.train()
                else:
                    mean, var = self.meta_generator(h_agg, context_expanded)

                h_target = h_complete[v][observed_mask]

                mse_loss = F.mse_loss(mean, h_target, reduction='mean')
                var_reg = torch.mean(torch.log(var + 1e-6))

                normalized_mse = mse_loss / self.hidden_dim
                meta_loss += normalized_mse + var_reg / self.hidden_dim
                valid_views += 1
        
        if valid_views > 0:
            return meta_loss / valid_views
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def compute_causal_loss(self, h_views, h_complete, miss_vecs):
        device = h_complete[0].device

        z_fact = torch.cat(h_complete, dim=1)
        q_fact = self.compute_cluster_assignment(z_fact)

        cf_miss_vecs = self.create_counterfactual_mask(miss_vecs, beta=0.3)

        has_change = False
        for v in range(self.view_num):
            if not torch.equal(miss_vecs[v], cf_miss_vecs[v]):
                has_change = True
                break
        
        if not has_change:
            return torch.tensor(0.0, device=device, requires_grad=True)

        h_cf_input = []
        for v in range(self.view_num):
            h_v = h_views[v].clone()
            missing_in_cf = ~cf_miss_vecs[v].bool()
            h_v[missing_in_cf] = 0.0
            h_cf_input.append(h_v)

        h_cf = self.generate_missing_views(h_cf_input, cf_miss_vecs)
        z_cf = torch.cat(h_cf, dim=1)
        q_cf = self.compute_cluster_assignment(z_cf)

        q_fact_detached = q_fact.detach()

        q_cf_log = F.log_softmax(q_cf, dim=1)
        q_fact_target = F.softmax(q_fact_detached, dim=1)
        
        kl_loss = F.kl_div(
            q_cf_log, 
            q_fact_target, 
            reduction='batchmean'
        )
        
        return kl_loss
    
    def get_representations(self, xs, miss_vecs):
        with torch.no_grad():
            h_views = self.encode_views(xs, miss_vecs)

            h_complete = self.generate_missing_views(h_views, miss_vecs)

            z_final = torch.cat(h_complete, dim=1)
            
            return z_final






