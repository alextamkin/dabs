import numpy as np
import torch
import torchmetrics

from src.systems.pytorch import base_system


class ContpredSystem(base_system.BaseSystem):
    '''Contrastive Prediction System with Masking.'''

    TEMPERATURE = 0.07

    def __init__(self, config, negatives):
        super().__init__(config)
        self.negatives = negatives
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        # Predicts missing patch.
        self.predictor = torch.nn.Linear(self.model.emb_dim, self.model.emb_dim)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x, prehead=False):
        return self.model.forward(x, prehead=prehead)

    def ssl_forward(self, batch):
        batch = batch[1:-1]  # Remove label.

        # The embs we will be predicting [batch_size, seq_len, emb_dim]
        target_embs = self.model.embed(batch)
        masked_embs, is_masked = self.mask_embeddings(target_embs)

        # We pass prepool=True because we want the embeddings for each token.
        return self.model.encode(masked_embs, prepool=True), target_embs, is_masked

    def training_step(self, batch, batch_idx):
        final_embs, target_embs, is_masked = self.ssl_forward(batch)
        loss, acc = self.objective(final_embs, target_embs, is_masked)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log('train_acc', acc.item(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        final_embs, target_embs, is_masked = self.ssl_forward(batch)
        loss, acc = self.objective(final_embs, target_embs, is_masked)
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_acc', acc.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def objective(self, final_embs, target_embs, is_masked):
        if self.negatives == 'sequence':
            return self.objective_seq(final_embs, target_embs, is_masked)
        elif self.negatives == 'batch':
            return self.objective_batch(final_embs, target_embs, is_masked)
        else:
            raise ValueError(f'Unspecified negatives type {self.negatives}')

    def objective_seq(self, final_embs, target_embs, is_masked):
        # [batch_size, seq_len, emb_dim]
        pred_embs = self.predictor(final_embs)[:, 1:]  # Remove [CLS] token.

        if self.config.contpred.normalize:
            pred_embs = pred_embs / torch.norm(((pred_embs)), p=2, dim=[-1], keepdim=True)
            target_embs = target_embs / torch.norm(((target_embs)), p=2, dim=[-1], keepdim=True)

        logits = pred_embs @ torch.transpose(target_embs, -1, -2) / self.TEMPERATURE

        # [batch_size, seq_len]
        labels = torch.arange(logits.shape[1], dtype=torch.long, device=logits.device).unsqueeze(0).repeat(logits.shape[0], 1)

        if self.config.contpred.symmetric_loss:
            loss = 0.5 * (self.ce(logits, labels).mean() + self.ce(logits.T, labels.T).mean())
        else:
            loss = self.ce(logits, labels).mean()
        acc = self.accuracy(logits.argmax(-1), labels)
        return loss, acc

    def objective_batch(self, final_embs, target_embs, is_masked):
        '''Test objective with full batch'''

        # [batch_size, seq_len, emb_dim]
        pred_embs = self.predictor(final_embs)[:, 1:]  # Remove [CLS] token.

        #  [batch_size * num_masked, batch_size * num_masked]
        logits = pred_embs[is_masked] @ target_embs[is_masked].T / self.TEMPERATURE

        # [batch_size * num_masked]
        labels = torch.arange(logits.size(0), dtype=torch.long, device=logits.device)

        loss = self.ce(logits, labels).mean()
        acc = self.accuracy(logits.argmax(-1), labels)
        return loss, acc

    def mask_embeddings(self, embs):
        '''Masks fraction of embeddings within each example.

        Args:
            embs: [batch_size, seq_len, emb_size] embeddings to mask

        Returns:
            masked_embs: [batch_size, seq_len, emb_size] embeddings
                with specified fraction masked
            is_masked: [batch_size, seq_len] of ints indicating whether each token was masked or not

        '''
        num_to_mask = int(np.ceil(embs.size(1) * self.config.corruption_rate))

        # [batch_size, num_to_mask]
        indices_to_mask = torch.rand(embs.size(0), embs.size(1), device=embs.device).topk(dim=-1, k=num_to_mask).indices
        # [batch_size, num_to_mask, emb_size] (repeat along last dimension for torch.scatter)
        indices_to_mask_for_scatter = indices_to_mask.unsqueeze(-1).repeat(1, 1, embs.size(-1))

        zeros = torch.zeros_like(indices_to_mask_for_scatter, device=embs.device, dtype=embs.dtype)
        masked_embs = torch.scatter(embs, -2, indices_to_mask_for_scatter, zeros)

        is_masked = torch.zeros(embs.size(0), embs.size(1), dtype=int, device=embs.device)
        is_masked = torch.scatter(is_masked, 1, indices_to_mask, 1)

        return masked_embs, is_masked.bool()
