import torch
import torch.nn as nn

class DAFLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.sigma = 1.0 / self.n_experts 
        
        # Config에서 가중치 로드
        self.lambda_spec = config.lambda_spec
        self.lambda_bal = config.lambda_bal
        self.lambda_repel = config.lambda_repel
        
        if config.task_type == 'classification':
            self.task_criterion = nn.BCEWithLogitsLoss()
        else:
            self.task_criterion = nn.MSELoss()

    def compute_specialization_loss(self, routing_history, unified_metadata):
        spec_loss = 0.0
        target_phi = unified_metadata[:, :, 0:1] # P or Transformed Freq
        norm = target_phi.shape[0] * target_phi.shape[1]

        for info in routing_history:
            mu = torch.sigmoid(info['mu']).view(1, 1, -1) # [Fix] Sigmoid 적용
            dist_sq = (target_phi - mu).pow(2)
            spec_loss += (info['weights'].detach() * dist_sq).sum() / norm
        return spec_loss / len(routing_history)

    def compute_repulsion_loss(self, routing_history):
        repel_loss = 0.0
        for info in routing_history:
            mu = torch.sigmoid(info['mu']) # [Fix] Sigmoid 적용
            dist_sq = (mu.unsqueeze(1) - mu.unsqueeze(0)).pow(2)
            repulsion = torch.exp(-dist_sq / (2 * self.sigma**2))
            mask = torch.eye(self.n_experts, device=mu.device)
            repel_loss += (repulsion * (1-mask)).sum() / (self.n_experts * (self.n_experts - 1))
        return repel_loss / len(routing_history)

    def compute_balance_loss(self, routing_history):
        bal_loss = 0.0
        for info in routing_history:
            weights = info['weights'].view(-1, self.n_experts)
            total = weights.shape[0] + 1e-6
            f_j = (weights > 0).float().sum(0) / total
            p_j = weights.sum(0) / total
            bal_loss += self.n_experts * torch.sum(f_j * p_j)
        return bal_loss / len(routing_history)

    def forward(self, preds, targets, routing_history, unified_metadata):
        if isinstance(self.task_criterion, nn.BCEWithLogitsLoss):
            targets = targets.float().view_as(preds)
        
        loss_dict = {
            'task': self.task_criterion(preds, targets),
            'spec': self.compute_specialization_loss(routing_history, unified_metadata),
            'repel': self.compute_repulsion_loss(routing_history),
            'bal': self.compute_balance_loss(routing_history)
        }
        
        loss_dict['total'] = loss_dict['task'] + \
                             (self.lambda_spec * loss_dict['spec']) + \
                             (self.lambda_repel * loss_dict['repel']) + \
                             (self.lambda_bal * loss_dict['bal'])
        return loss_dict