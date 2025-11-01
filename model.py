# -------------------------------------------------
#  CELL 1 – MODEL + INFERENCE (fixed)
# -------------------------------------------------

import os, numpy as np, pandas as pd, torch, dgl, torch.nn as nn
from sklearn.metrics import (classification_report, roc_auc_score,
                             f1_score, confusion_matrix, recall_score)

# ------------------- MODEL CLASSES -------------------
class RelationAware(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.d_liner = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, dst):
        src = self.d_liner(src); dst = self.d_liner(dst)
        diff = src - dst
        return self.tanh(src + dst + diff)

class H_layer(nn.Module):
    def __init__(self, input_dim, output_dim, head, relation_aware, etype,
                 dropout, if_sum=False):
        super().__init__()
        self.etype = etype; self.head = head; self.hd = output_dim
        self.if_sum = if_sum
        self.atten = nn.Linear(3 * self.hd, 1)
        self.relu = nn.ReLU()
        self.relation_ware = relation_aware
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.w_liner = nn.Linear(input_dim, output_dim * head)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['feat'] = h
            g.apply_edges(self.sign_edges, etype=self.etype)
            h_proj = self.w_liner(h)
            g.ndata['h'] = h_proj
            g.update_all(message_func=self.message,
                         reduce_func=self.reduce,
                         etype=self.etype)
            out = g.ndata['out']; edge_s = g.ndata['s']
            if not self.if_sum:
                return edge_s, out, h_proj.view(-1, self.head * self.hd)
            else:
                return edge_s, out, h_proj.view(-1, self.head, self.hd).sum(-2)

    def message(self, edges):
        src_f = edges.src['h'].view(-1, self.head, self.hd)
        dst_f = edges.dst['h'].view(-1, self.head, self.hd)
        edge_s = edges.data['edge_sum'].view(-1, self.head, self.hd)
        z = torch.cat([src_f, dst_f, edge_s], dim=-1)
        alpha = self.leakyrelu(self.atten(z))
        return {'atten': alpha, 'sf': src_f, 'edge_sum': edge_s}

    def reduce(self, nodes):
        alpha = self.softmax(nodes.mailbox['atten'])
        sf = nodes.mailbox['sf']
        out = torch.sum(alpha * sf, dim=1)               # <-- FIXED
        if not self.if_sum:
            out = out.view(-1, self.head * self.hd)
            edge_s = torch.mean(nodes.mailbox['edge_sum'],
                               dim=1).view(-1, self.head * self.hd)
            return {'out': out, 's': edge_s}
        else:
            out = out.sum(dim=-2)
            edge_s = torch.sum(torch.mean(nodes.mailbox['edge_sum'],
                                         dim=1), dim=-2)
            return {'out': out, 's': edge_s}

    def sign_edges(self, edges):
        src = edges.src['feat']; dst = edges.dst['feat']
        return {'edge_sum': self.relation_ware(src, dst)}

class Gate(nn.Module):
    def __init__(self, head, output_dim, dropout, if_sum=False):
        super().__init__()
        self.output_dim = output_dim; self.head = head; self.if_sum = if_sum
        if not self.if_sum:
            self.beta = nn.Parameter(
                torch.empty(size=(2 * self.head * self.output_dim, 1)))
            nn.init.xavier_normal_(self.beta.data, gain=1.414)
        else:
            self.beta = nn.Parameter(
                torch.empty(size=(2 * self.output_dim, 1)))
            nn.init.xavier_normal_(self.beta.data, gain=1.414)
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge_sum, out, h):
        beta = torch.cat([edge_sum, out], dim=1)
        gate = self.sigmoid(beta @ self.beta)
        return gate * out + (1 - gate) * h

class MultiRelationGE_GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, head, etypes, dropout,
                 if_sum=False):
        super().__init__()
        self.relation = [e for e in etypes if e != 'homo']
        self.n_relation = len(self.relation); self.if_sum = if_sum
        self.liner = nn.Linear(
            (self.n_relation * output_dim * head) if not if_sum else
            (self.n_relation * output_dim),
            (output_dim * head) if not if_sum else output_dim)
        self.relation_aware = RelationAware(input_dim,
                                            output_dim * head, dropout)
        self.minelayers = nn.ModuleDict()
        for e in self.relation:
            sub = nn.ModuleList()
            sub.append(H_layer(input_dim, output_dim, head,
                               self.relation_aware, e, dropout, if_sum))
            sub.append(Gate(head, output_dim, dropout, if_sum))
            self.minelayers[e] = sub
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h):
        hs = []
        for e in self.relation:
            edge_sum1, out1, h1 = self.minelayers[e][0](g, h)
            he = self.minelayers[e][1](edge_sum1, out1, h1)
            hs.append(he)
        x = torch.cat(hs, dim=1)
        x = self.dropout(x)
        return self.liner(x)

class GE_GNN(nn.Module):
    def __init__(self, args, g):
        super().__init__()
        if 'feature' in g.nodes['r'].data:
            self.input_dim = g.nodes['r'].data['feature'].shape[1]
        elif 'feat' in g.nodes['r'].data:
            self.input_dim = g.nodes['r'].data['feat'].shape[1]
        else:
            raise KeyError("Node features not found")
        self.intra_dim = args.intra_dim
        self.n_class   = args.n_class
        self.n_layer   = args.n_layer
        etypes = g.etypes
        self.mine_layers = nn.ModuleList()
        if args.n_layer == 1:
            self.mine_layers.append(
                MultiRelationGE_GNNLayer(self.input_dim, self.n_class,
                                         args.head, etypes,
                                         args.dropout, if_sum=True))
        else:
            self.mine_layers.append(
                MultiRelationGE_GNNLayer(self.input_dim, self.intra_dim,
                                         args.head, etypes,
                                         args.dropout))
            for _ in range(1, self.n_layer - 1):
                self.mine_layers.append(
                    MultiRelationGE_GNNLayer(self.intra_dim * args.head,
                                             self.intra_dim,
                                             args.head, etypes,
                                             args.dropout))
            self.mine_layers.append(
                MultiRelationGE_GNNLayer(self.intra_dim * args.head,
                                         self.n_class,
                                         args.head, etypes,
                                         args.dropout, if_sum=True))
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, g):
        h = (g.nodes['r'].data['feature'].float()
             if 'feature' in g.nodes['r'].data
             else g.nodes['r'].data['feat'].float())
        for i in range(self.n_layer):
            if i > 0:
                h = self.relu(h); h = self.dropout(h)
            h = self.mine_layers[i](g, h)
        return h

# ------------------- HYPER-PARAMETERS -------------------
class _A: pass
args = _A()
args.device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.intra_dim = 8
args.head      = 8
args.n_layer   = 2
args.dropout   = 0.1
args.n_class   = 2

# ------------------- INFERENCE FUNCTION -------------------
def run_inference(subset_csv_path, ckpt_path, stats_path):
    # ---- load CSV -------------------------------------------------
    subset_df = pd.read_csv(subset_csv_path)

    non_feature_columns = ['_id','reviewerID','asin','reviewerName',
                           'helpful','reviewText','summary',
                           'unixReviewTime','reviewTime','category','class']
    feature_columns = [c for c in subset_df.columns
                       if c not in non_feature_columns]
    if 'reviewTime' in feature_columns:
        feature_columns.remove('reviewTime')
    subset_features = subset_df[feature_columns].values.astype(np.float32)

    # ---- build edges ------------------------------------------------
    idx_map = {orig:i for i,orig in enumerate(subset_df.index)}
    n_nodes = len(subset_df)

    upu_src, upu_dst = [], []
    for asin, grp in subset_df.groupby('asin'):
        ids = grp.index.tolist()
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i != j:
                    upu_src.append(idx_map[ids[i]])
                    upu_dst.append(idx_map[ids[j]])

    usu_src, usu_dst = [], []
    uvu_src, uvu_dst = [], []
    for rid, grp in subset_df.groupby('reviewerID'):
        ids = grp.index.tolist()
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i != j:
                    i1, i2 = ids[i], ids[j]
                    if ('overall' in subset_df.columns and
                        subset_df.loc[i1,'overall'] == subset_df.loc[i2,'overall']):
                        usu_src.append(idx_map[i1]); usu_dst.append(idx_map[i2])
                    if ('review_word_count' in subset_df.columns and
                        abs(int(subset_df.loc[i1,'review_word_count']) -
                            int(subset_df.loc[i2,'review_word_count'])) <= 10):
                        uvu_src.append(idx_map[i1]); uvu_dst.append(idx_map[i2])

    graph_struct = {
        ('r','p','r'): (torch.tensor(upu_src, dtype=torch.int64),
                        torch.tensor(upu_dst, dtype=torch.int64)),
        ('r','s','r'): (torch.tensor(usu_src, dtype=torch.int64),
                        torch.tensor(usu_dst, dtype=torch.int64)),
        ('r','v','r'): (torch.tensor(uvu_src, dtype=torch.int64),
                        torch.tensor(uvu_dst, dtype=torch.int64))
    }
    g = dgl.heterograph(graph_struct, num_nodes_dict={'r': n_nodes})
    g.nodes['r'].data['feature'] = torch.from_numpy(subset_features).float()
    for et in g.etypes:
        g = dgl.add_self_loop(g, etype=et)

    # ---- load training artifacts (weights_only=True) -------------
    state_dict = torch.load(ckpt_path, map_location=args.device,
                            weights_only=True)
    stats      = torch.load(stats_path, map_location='cpu',
                            weights_only=True)
    train_cols = stats['feature_columns']
    mean, std  = stats['mean'], stats['std']

    # ---- align & normalize -----------------------------------------
    X = subset_df.reindex(columns=train_cols, fill_value=0.0)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
    feats = torch.from_numpy(X.values.astype(np.float32))
    std = std.clone(); std[std == 0] = 1.0
    feats = (feats - mean.to(feats.dtype)) / std.to(feats.dtype)
    g.nodes['r'].data['feature'] = feats

    # ---- run model -------------------------------------------------
    g = g.to(args.device)
    net = GE_GNN(args, g).to(args.device)
    net.load_state_dict(state_dict)
    net.eval()
    with torch.no_grad():
        logits = net(g).cpu()
        probs  = torch.softmax(logits, dim=1)[:,1].numpy().tolist()
        preds  = logits.argmax(1).numpy().tolist()

    # ---- optional metrics (skip if no ground-truth) ---------------
    metrics = {}
    if 'class' in subset_df.columns:
        y_true = subset_df['class'].replace({np.nan: -1}).astype(int).to_numpy()
        # ignore label -1 when computing metrics
        valid = y_true != -1
        if valid.any():
            metrics['auc']        = roc_auc_score(y_true[valid], np.array(probs)[valid])
            metrics['f1_macro']   = f1_score(y_true[valid],
                                            np.array(preds)[valid],
                                            average='macro')
            metrics['recall']     = recall_score(y_true[valid],
                                                np.array(preds)[valid])
            metrics['cm']         = confusion_matrix(y_true[valid],
                                                    np.array(preds)[valid]).tolist()
            metrics['report']     = classification_report(
                                        y_true[valid],
                                        np.array(preds)[valid],
                                        digits=4, output_dict=True)

    # ---- fraudulent rows -------------------------------------------
    subset_df['predicted_class'] = preds
    fraudulent = subset_df[subset_df['predicted_class']==1]\
                    [['reviewerID','reviewText']].to_dict('records')

    return {
        'predictions': preds,
        'probs'      : probs,
        'metrics'    : metrics,
        'fraudulent_reviews': fraudulent,
        'df_snapshot': subset_df.to_dict('records')   # keep for UI
    }
