import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from einops.layers.torch import Rearrange

from src.datasets.catalog import DATASET_DICT
from src.systems.pytorch import base_system


class MAESystem(base_system.BaseSystem):
    '''System for Masked Autoencoding.

    Masks a given fraction of input patches/tokens.
    Objective is to reconstruct masked items.
    '''

    def __init__(self, config):
        super().__init__(config)

        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        self.dataset = DATASET_DICT[config.dataset.name]
        if hasattr(self.dataset, 'MAE_OUTPUT_SIZE'):
            mae_output_size = self.dataset.MAE_OUTPUT_SIZE
            self.predictor = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(self.model.emb_dim, mae_output_size))

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x, prehead=False):
        return self.model.forward(x, prehead=prehead)

    def ssl_forward(self, batch):
        batch = batch[1:-1]  # Remove label.

        # Embed first. [batch_size, seq_len, emb_dim]
        embs = self.model.embed(batch)

        masked_embs, is_masked, indices_to_mask = self.mask_embeddings(embs)

        if self.config.dataset.name in ['mscoco']:
            embed_img = nn.Sequential(
                Rearrange(
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.dataset.PATCH_SIZE[0], p2=self.dataset.PATCH_SIZE[1]
                )
            )
            target_img = embed_img(batch[0])
            target_text = batch[1]
            target = [target_img, target_text]
        elif self.config.dataset.name in ['librispeech', 'chexpert', 'imagenet', 'eurosat', 'pamap2', 'cifar10_small',
                                          'wafer']:
            embed2 = nn.Sequential(
                Rearrange(
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.dataset.PATCH_SIZE[0], p2=self.dataset.PATCH_SIZE[1]
                )
            )
            target = embed2(batch[0])
        elif self.config.dataset.name in ['wikitext103', 'mc4', 'pfam', 'genomics', 'higgs']:
            target = batch[0]
        else:
            raise ValueError(f'Unimplemented MAE dataset={self.config.dataset}.')

        # We pass prepool=True because we want the embeddings for each token.
        return self.model.encode(masked_embs, prepool=True), is_masked, target, indices_to_mask

    def training_step(self, batch, batch_idx):
        embs, is_masked, target, indices_to_mask = self.ssl_forward(batch)
        loss = self.objective(embs, is_masked, target, indices_to_mask)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embs, is_masked, target, indices_to_mask = self.ssl_forward(batch)
        loss = self.objective(embs, is_masked, target, indices_to_mask)
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def objective_continuous(self, embs, is_masked, target, embs_have_cls):
        preds = self.predictor(embs)
        if embs_have_cls:
            preds = preds[:, 1:]  # Remove [CLS] token.
        target = target.reshape(preds.shape[0], preds.shape[1], -1)
        diff = target[is_masked] - preds[is_masked]

        loss = torch.norm(diff, p=2, dim=-1).mean()
        return loss

    def objective_tokens(self, embs, is_masked, target, embs_have_cls, emb_module_idx=0):
        if embs_have_cls:
            embs = embs[:, 1:]
        embs = embs[is_masked]
        mask_preds = torch.einsum('ne,ve->nv', embs, self.model.embed_modules[emb_module_idx].embed.weight)
        mask_targets = target[is_masked]
        loss = self.ce(mask_preds, mask_targets).mean()
        return loss

    def objective(self, embs, is_masked, target, indices_to_mask):
        # TODO: generalize to do this automatically based on specs.

        # Multimodal (tokenized and continuous)
        if self.config.dataset.name in ['mscoco']:
            image_seq_len = target[0].size(1)
            # Don't include CLS token
            img_loss = self.objective_continuous(
                embs[:, 1:image_seq_len + 1], is_masked[:, :image_seq_len], target[0], embs_have_cls=False
            )
            text_loss = self.objective_tokens(
                embs[:, 1 + image_seq_len:],
                is_masked[:, image_seq_len:],
                target[1],
                embs_have_cls=False,
                emb_module_idx=1  # In MSCOCO text is second.
            )
            loss = (img_loss + text_loss) / 2
        elif hasattr(self.dataset, 'MAE_OUTPUT_SIZE'):
            # Continuous data.
            loss = self.objective_continuous(embs, is_masked, target, embs_have_cls=True)
        else:
            # Tokenized data.
            loss = self.objective_tokens(embs, is_masked, target, embs_have_cls=True)
        return loss

    def mask_embeddings(self, embs):
        '''Masks a fraction of embeddings within each example.

        Args:
            embs: [batch_size, seq_len, emb_size] embeddings to mask

        Returns:
            masked_embs: [batch_size, seq_len, emb_size] embeddings
                with specified fraction fraction masked
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

        return masked_embs, is_masked.bool(), indices_to_mask
