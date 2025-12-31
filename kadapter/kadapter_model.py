import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM


# define a K-Adapter model based on bert-dapt
# - wraps a BERT MLM backbone
# - adds a bottleneck adapter
# - adds entity / relation embeddings for KG
# - can inject KG information into the [MASK] position during forward

class BertKAdapterForMaskedLM(nn.Module):
    def __init__(
        self,
        backbone_path: str,
        bottleneck_dim: int,   # adapter bottleneck dimension
        num_entities: int,     # size of entity vocabulary
        num_relations: int,    # size of relation vocabulary
        freeze_bert: bool = True,   # stage1: True
    ):
        super().__init__()

        # load BERT + MLM head
        self.bert_mlm = AutoModelForMaskedLM.from_pretrained(backbone_path)
        hidden_size = self.bert_mlm.config.hidden_size
        vocab_size = self.bert_mlm.config.vocab_size
        self.vocab_size = vocab_size

        # bottleneck adapter (simple residual MLP)
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_size),
        )

        # KG embeddings (in bottleneck dimension)
        self.kg_dim = bottleneck_dim
        self.ent_emb = nn.Embedding(num_entities, self.kg_dim)  # h/t
        self.rel_emb = nn.Embedding(num_relations, self.kg_dim)

        # project KG vectors to BERT's hidden dimension for integration
        self.kg_proj = nn.Linear(self.kg_dim, hidden_size)

        # stage1: freeze BERT, only train adapter + KG components + MLM head
        if freeze_bert:
            for p in self.bert_mlm.bert.parameters():
                p.requires_grad = False


    def kg_score(self, h_ids, r_ids, t_ids):
        """
        Compute DistMult compatibility scores for KG triplets
        """
        h = self.ent_emb(h_ids)   # [B, D]
        r = self.rel_emb(r_ids)   # [B, D]
        t = self.ent_emb(t_ids)   # [B, D]
        return torch.sum(h * r * t, dim=-1)  # [B]


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,       # -100: unpredict position
        # KG injection inputs (optional)
        head_ids=None,        # [B]
        rel_ids=None,         # [B]
        tail_ids=None,        # [B]
        mask_positions=None,  # [B], ndex of mask token for each sample
        use_kg: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **unused_kwargs,
    ):
        # If use_kg=True and KG inputs are provided, injects KG bias at mask positions.

        # 1.BERT encoding
        bert_outputs = self.bert_mlm.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden = bert_outputs.last_hidden_state    # [B, L, H]

        # 2.apply adapter with residual connection
        h = last_hidden + self.adapter(last_hidden)     # [B, L, H]

        # 3.inject KG bias at mask positions
        if use_kg:
            if head_ids is not None and rel_ids is not None and tail_ids is not None and mask_positions is not None:
                h_e = self.ent_emb(head_ids)   # [B, D]
                r_e = self.rel_emb(rel_ids)    # [B, D]
                t_e = self.ent_emb(tail_ids)   # [B, D]

                # simple combination - vector addition, can be changed to concat + MLP
                kg_vec = h_e + r_e + t_e       # [B, D]

                # project to hidden size
                kg_bias = self.kg_proj(kg_vec)  # [B, H]

                # add to mask position
                B, L, H = h.shape
                batch_idx = torch.arange(B, device=h.device)
                h = h.clone()     # avoid in-place operations for autograd
                h[batch_idx, mask_positions, :] = (
                    h[batch_idx, mask_positions, :] + kg_bias
                )
            else:
                pass

        # 4.apply MLM head
        logits = self.bert_mlm.cls(h)   # [B, L, V]

        loss = None
        if labels is not None:
            # 默认 CrossEntropyLoss(ignore_index=-100)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        if not return_dict:
            if output_hidden_states:
                return loss, logits, h
            else:
                return loss, logits

        out = {"loss": loss, "logits": logits}
        if output_hidden_states:
            out["hidden_states"] = h
        return out
