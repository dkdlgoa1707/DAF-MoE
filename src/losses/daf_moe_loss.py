import torch
import torch.nn as nn

class DAFLoss(nn.Module):
    """
    Joint Objective Function for DAF-MoE.
    
    Implements Equation 15: 
    L_total = L_task + lambda_spec * L_spec + lambda_repel * L_repel + lambda_bal * L_bal
    """
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        # nu = 1 / N_E (Scaling factor for repulsion kernel, Eq. 13)
        self.nu = 1.0 / self.n_experts 
        
        self.lambda_spec = config.lambda_spec
        self.lambda_bal = config.lambda_bal
        self.lambda_repel = config.lambda_repel
        self.eps = 1e-8
        
        if config.task_type == 'classification':
            self.task_criterion = nn.BCEWithLogitsLoss() if config.out_dim == 1 else nn.CrossEntropyLoss()
        else:
            self.task_criterion = nn.MSELoss()

    def compute_specialization_loss(self, routing_history, psi_x):
        """
        Expert Specialization Loss (Equation 12).
        
        Minimizes the squared distance between the feature's distributional position Psi(x_j) 
        and the selected expert's centroid sigma(mu_k).
        
        Args:
            routing_history: List of dicts containing 'weights' and 'mu' from each layer.
            psi_x: Distributional position metadata [B, S, Features].
        """
        loss_spec = 0.0
        # Psi(x_j) is the first channel of the metadata (Percentile Phi or Tilde_F)
        psi_x_j = psi_x[:, :, 0:1] 
        
        norm = psi_x_j.shape[0] * psi_x_j.shape[1] + self.eps
        
        for info in routing_history:
            # sigma(mu_k) : Expert Centroid
            centroid_pos = torch.sigmoid(info['mu']).view(1, 1, -1)
            
            # || Psi(x_j) - sigma(mu_k) ||^2
            dist_sq = (psi_x_j - centroid_pos).pow(2)
            
            # sg(G(q)_k) * Distance (Stop-gradient on routing weights)
            gating_scores = info['weights'].detach() 
            loss_spec += (gating_scores * dist_sq).sum() / norm
            
        return loss_spec / len(routing_history)

    def compute_repulsion_loss(self, routing_history):
        """
        Centroid Repulsion Loss (Equation 13).
        
        Maximizes the pairwise distance between expert centroids using a Gaussian kernel 
        to prevent mode collapse and ensure diversity.
        """
        loss_repel = 0.0
        for info in routing_history:
            centroid_pos = torch.sigmoid(info['mu']) # sigma(mu_k)
            
            # Pairwise squared distances: || sigma(mu_k) - sigma(mu_l) ||^2
            dist_sq = (centroid_pos.unsqueeze(1) - centroid_pos.unsqueeze(0)).pow(2)
            
            # Gaussian Kernel: exp( - ||...||^2 / 2nu^2 )
            repulsion = torch.exp(-dist_sq / (2 * self.nu**2))
            
            # Mask diagonal elements (self-repulsion is always 1, ignore it)
            mask = torch.eye(self.n_experts, device=centroid_pos.device)
            denom = (self.n_experts * (self.n_experts - 1)) + self.eps
            
            loss_repel += (repulsion * (1 - mask)).sum() / denom
            
        return loss_repel / len(routing_history)

    def compute_balance_loss(self, routing_history):
        """
        Load Balancing Loss (Equation 14).
        
        Standard auxiliary loss to prevent router collapse where a few experts dominate.
        L_bal = N_E * sum(f_k * P_k)
        """
        loss_bal = 0.0
        for info in routing_history:
            weights = info['weights'].view(-1, self.n_experts)
            total = weights.shape[0] + self.eps
            
            # f_k: Fraction of tokens assigned to expert k (Hard count approx)
            f_k = (weights > 0).float().sum(0) / total
            
            # P_k: Average routing probability for expert k
            P_k = weights.sum(0) / total
            
            loss_bal += self.n_experts * torch.sum(f_k * P_k)
            
        return loss_bal / len(routing_history)

    def forward(self, preds, targets, routing_history, psi_x):
        """
        Args:
            preds: Model predictions.
            targets: Ground truth labels.
            routing_history: List of routing info from transformer blocks.
            psi_x: Distributional metadata (Psi vector).
        """
        # Multi-GPU Handling (DataParallel wraps outputs in list)
        if isinstance(routing_history, list) and len(routing_history) > 0:
            if isinstance(routing_history[0], dict):
                pass # Standard case
            elif isinstance(routing_history[0], list):
                # Flatten nested list from DataParallel
                # (This logic depends on specific DataParallel behavior, usually kept simple)
                pass 

        if isinstance(self.task_criterion, nn.CrossEntropyLoss):
            targets = targets.long().view(-1)
        else:
            targets = targets.float().view_as(preds)
        
        # 1. Primary Task Loss
        loss_task = self.task_criterion(preds, targets)
        
        # 2. Auxiliary Losses
        loss_spec = self.compute_specialization_loss(routing_history, psi_x)
        loss_repel = self.compute_repulsion_loss(routing_history)
        loss_bal = self.compute_balance_loss(routing_history)
        
        # 3. Total Objective (Equation 15)
        loss_total = loss_task + \
                    (self.lambda_spec * loss_spec) + \
                    (self.lambda_repel * loss_repel) + \
                    (self.lambda_bal * loss_bal)
                        
        return {
            'total': loss_total, 
            'task': loss_task, 
            'spec': loss_spec, 
            'repel': loss_repel, 
            'bal': loss_bal
        }