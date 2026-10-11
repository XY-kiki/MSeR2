from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CanonicalEType = Tuple[str, str, str]


DEFAULT_ENCODER_CONFIG = {
    "seed": 1234,
    "hidden_dim": 128,
    "dropout": 0.2,
    "num_heads": 2,
    "learning_rate": 8e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 512,
    "valid_ratio": 0.1,
    "final_residual_alpha": 0.5,
    "relation_residual_alpha": 0.5,
    "use_meta_paths": {
        "DrPrDi": True,
        "DrPrDr": True,
        "DiPrDi": True,
        "PrDiPr": True,
        "PrDrPr": True,
    },
}


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        dgl.seed(seed)
        dgl.random.seed(seed)
    except AttributeError:
        pass


def read_feature_csv(
    path: Path,
    *,
    drop_first_col: bool = True,
    has_header: bool = False,
) -> np.ndarray:
    frame = pd.read_csv(path, header=0 if has_header else None)
    if drop_first_col:
        frame = frame.iloc[:, 1:]
    values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values were found in {path}.")
    return values


def _standardize(values: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(values).astype(np.float32)


def _read_edges(path: Path) -> np.ndarray:
    edges = pd.read_csv(path, dtype=np.int64).iloc[:, :2].to_numpy(dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected a two-column edge list in {path}.")
    return edges


def _check_edge_bounds(
    edges: np.ndarray,
    num_src: int,
    num_dst: int,
    relation_name: str,
) -> None:
    if edges.size == 0:
        raise ValueError(f"The {relation_name} edge list is empty.")
    if edges[:, 0].min() < 0 or edges[:, 0].max() >= num_src:
        raise IndexError(f"Source index out of range in {relation_name}.")
    if edges[:, 1].min() < 0 or edges[:, 1].max() >= num_dst:
        raise IndexError(f"Destination index out of range in {relation_name}.")


def load_data(data_dir: Path | str) -> Dict[str, np.ndarray | int]:
    data_dir = Path(data_dir)
    data: Dict[str, np.ndarray | int] = {}

    drug_fingerprint = read_feature_csv(
        data_dir / "DrugFingerprint.csv",
        drop_first_col=True,
        has_header=True,
    )
    drug_gip = read_feature_csv(
        data_dir / "DrugGIP.csv",
        drop_first_col=True,
        has_header=True,
    )
    disease_phenotype = read_feature_csv(
        data_dir / "DiseasePS.csv",
        drop_first_col=True,
        has_header=True,
    )
    disease_gip = read_feature_csv(
        data_dir / "DiseaseGIP.csv",
        drop_first_col=True,
        has_header=True,
    )
    data["protein_feature"] = _standardize(
        read_feature_csv(
            data_dir / "Protein_ESM.csv",
            drop_first_col=True,
            has_header=False,
        )
    )

    if drug_fingerprint.shape != drug_gip.shape:
        raise ValueError(
            "Element-wise feature addition requires DrugFingerprint.csv and "
            "DrugGIP.csv to have identical shapes."
        )
    if disease_phenotype.shape != disease_gip.shape:
        raise ValueError(
            "Element-wise feature addition requires DiseasePS.csv and "
            "DiseaseGIP.csv to have identical shapes."
        )
    data["drug_feature"] = _standardize(drug_fingerprint + drug_gip)
    data["disease_feature"] = _standardize(disease_phenotype + disease_gip)

    data["num_drug"] = int(data["drug_feature"].shape[0])
    data["num_disease"] = int(data["disease_feature"].shape[0])
    data["num_protein"] = int(data["protein_feature"].shape[0])

    data["drpr"] = _read_edges(data_dir / "DrugProteinAssociationNumber.csv")
    data["prdr"] = _read_edges(data_dir / "ProteinDrugAssociationNumber.csv")
    data["dipr"] = _read_edges(data_dir / "DiseaseProteinAssociationNumber.csv")
    data["prdi"] = _read_edges(data_dir / "ProteinDiseaseAssociationNumber.csv")

    _check_edge_bounds(data["drpr"], data["num_drug"], data["num_protein"], "drug-protein")
    _check_edge_bounds(data["prdr"], data["num_protein"], data["num_drug"], "protein-drug")
    _check_edge_bounds(data["dipr"], data["num_disease"], data["num_protein"], "disease-protein")
    _check_edge_bounds(data["prdi"], data["num_protein"], data["num_disease"], "protein-disease")

    return data


def load_pairs(sample_dir: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(Path(sample_dir) / "all_pairs.csv")
    required = {"drug_id", "disease_id", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in all_pairs.csv: {sorted(missing)}")
    pairs = frame[["drug_id", "disease_id"]].to_numpy(dtype=np.int64)
    labels = frame["label"].to_numpy(dtype=np.int64)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels in all_pairs.csv must be 0 or 1.")
    return pairs, labels


def validate_pairs(
    pairs: np.ndarray,
    num_drug: int,
    num_disease: int,
) -> None:
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Each sample must contain a drug_id and a disease_id.")
    if pairs[:, 0].min() < 0 or pairs[:, 0].max() >= num_drug:
        raise IndexError("A drug_id in all_pairs.csv is out of range.")
    if pairs[:, 1].min() < 0 or pairs[:, 1].max() >= num_disease:
        raise IndexError("A disease_id in all_pairs.csv is out of range.")


def build_heterograph(data: Mapping[str, object]) -> dgl.DGLHeteroGraph:
    graph_data = {
        ("drug", "target", "protein"): (
            data["drpr"][:, 0],
            data["drpr"][:, 1],
        ),
        ("protein", "targeted_by", "drug"): (
            data["prdr"][:, 0],
            data["prdr"][:, 1],
        ),
        ("disease", "interact", "protein"): (
            data["dipr"][:, 0],
            data["dipr"][:, 1],
        ),
        ("protein", "interacted_by", "disease"): (
            data["prdi"][:, 0],
            data["prdi"][:, 1],
        ),
    }
    return dgl.heterograph(
        graph_data,
        num_nodes_dict={
            "drug": data["num_drug"],
            "disease": data["num_disease"],
            "protein": data["num_protein"],
        },
    )


def get_meta_paths(config: Mapping[str, object]) -> Dict[str, Dict[str, Sequence[CanonicalEType]]]:
    all_meta_paths = {
        "DrPrDi": [
            ("drug", "target", "protein"),
            ("protein", "interacted_by", "disease"),
        ],
        "DrPrDr": [
            ("drug", "target", "protein"),
            ("protein", "targeted_by", "drug"),
        ],
        "DiPrDi": [
            ("disease", "interact", "protein"),
            ("protein", "interacted_by", "disease"),
        ],
        "PrDiPr": [
            ("protein", "interacted_by", "disease"),
            ("disease", "interact", "protein"),
        ],
        "PrDrPr": [
            ("protein", "targeted_by", "drug"),
            ("drug", "target", "protein"),
        ],
    }
    destinations = {
        "DrPrDi": "disease",
        "DrPrDr": "drug",
        "DiPrDi": "disease",
        "PrDiPr": "protein",
        "PrDrPr": "protein",
    }
    enabled = config.get("use_meta_paths", {})
    selected: Dict[str, Dict[str, Sequence[CanonicalEType]]] = {
        "drug": {},
        "disease": {},
        "protein": {},
    }
    for name, path in all_meta_paths.items():
        if enabled.get(name, True):
            selected[destinations[name]][name] = path
    return selected


def etype_to_key(etype: CanonicalEType) -> str:
    return "__".join(etype)


def prepare_features(
    data: Mapping[str, object],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    keys = ("drug_feature", "disease_feature", "protein_feature")
    return {
        key: torch.as_tensor(data[key], dtype=torch.float32, device=device)
        for key in keys
    }


class MessageFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.message_projection = nn.Linear(hidden_dim, hidden_dim)
        self.destination_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        message: torch.Tensor,
        destination: torch.Tensor,
    ) -> torch.Tensor:
        message = self.message_projection(message)
        destination = self.destination_projection(destination)
        gate = torch.sigmoid(self.gate(torch.cat((message, destination), dim=-1)))
        fused = gate * message + (1.0 - gate) * destination
        fused = self.dropout(self.output_projection(fused))
        return self.normalization(F.relu(fused))


class RelationAggregation(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        residual_alpha: float,
    ) -> None:
        super().__init__()
        self.fusion = MessageFusion(hidden_dim, dropout)
        self.residual_alpha = residual_alpha

    def forward(
        self,
        relation_graph: dgl.DGLHeteroGraph,
        source_state: torch.Tensor,
        destination_state: torch.Tensor,
    ) -> torch.Tensor:
        with relation_graph.local_scope():
            relation_graph.srcdata["h"] = source_state
            relation_graph.update_all(fn.copy_u("h", "m"), fn.mean("m", "neighbor"))
            neighbor = relation_graph.dstdata["neighbor"]
        output = self.fusion(neighbor, destination_state)
        return output + self.residual_alpha * destination_state


class ViewProjection(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        return self.network(views)


class SemanticAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention_vector = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.attention_vector)

    def forward(self, views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # views: [num_views, num_nodes, hidden_dim]
        num_views, num_nodes, _ = views.shape
        query = self.query_projection(views).view(
            num_views,
            num_nodes,
            self.num_heads,
            self.head_dim,
        )
        scores = torch.einsum("vnhd,hd->nhv", query, self.attention_vector)
        weights = torch.softmax(scores, dim=-1)
        output = torch.einsum("vnhd,nhv->nhd", query, weights)
        output = output.reshape(num_nodes, self.hidden_dim)
        return self.output_projection(output), weights


class MSeR2Encoder(nn.Module):
    def __init__(
        self,
        input_dims: Mapping[str, int],
        config: Mapping[str, object],
        meta_paths: Mapping[str, Mapping[str, Sequence[CanonicalEType]]],
    ) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        dropout = float(config["dropout"])
        num_heads = int(config["num_heads"])
        self.meta_paths = meta_paths
        self.final_residual_alpha = float(config["final_residual_alpha"])

        self.drug_projection = nn.Linear(input_dims["drug_feature"], hidden_dim)
        self.disease_projection = nn.Linear(input_dims["disease_feature"], hidden_dim)
        self.protein_projection = nn.Linear(input_dims["protein_feature"], hidden_dim)

        edge_types = sorted(
            {
                edge_type
                for destination_paths in meta_paths.values()
                for path in destination_paths.values()
                for edge_type in path
            }
        )
        self.relation_layers = nn.ModuleDict(
            {
                etype_to_key(edge_type): RelationAggregation(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    residual_alpha=float(config["relation_residual_alpha"]),
                )
                for edge_type in edge_types
            }
        )
        self.view_projection = ViewProjection(hidden_dim, dropout)
        self.semantic_attention = nn.ModuleDict(
            {
                node_type: SemanticAttention(hidden_dim, num_heads)
                for node_type in ("drug", "disease", "protein")
            }
        )

    def _project_features(
        self,
        features: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        drug = self.drug_projection(features["drug_feature"])
        disease = self.disease_projection(features["disease_feature"])
        protein = self.protein_projection(features["protein_feature"])
        return {
            "drug": F.relu(drug),
            "disease": F.relu(disease),
            "protein": F.relu(protein),
        }

    def _encode_path(
        self,
        graph: dgl.DGLHeteroGraph,
        initial: Mapping[str, torch.Tensor],
        path: Sequence[CanonicalEType],
        protein_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = initial[path[0][0]]
        for edge_type in path:
            destination_type = edge_type[2]
            if destination_type == "protein" and protein_context is not None:
                destination_state = protein_context
            else:
                destination_state = initial[destination_type]
            state = self.relation_layers[etype_to_key(edge_type)](
                graph[edge_type],
                state,
                destination_state,
            )
        return state

    def _fuse_views(
        self,
        node_type: str,
        views: Iterable[torch.Tensor],
        initial_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        views = list(views)
        if not views:
            return initial_state, None
        stacked = self.view_projection(torch.stack(views, dim=0))
        fused, weights = self.semantic_attention[node_type](stacked)
        fused = fused + self.final_residual_alpha * initial_state
        return fused, weights

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        features: Mapping[str, torch.Tensor],
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor | None],
    ]:
        initial = self._project_features(features)

        protein_views = [
            self._encode_path(graph, initial, path)
            for path in self.meta_paths["protein"].values()
        ]
        protein_context, protein_weights = self._fuse_views(
            "protein", protein_views, initial["protein"]
        )

        drug_views = [
            self._encode_path(graph, initial, path, protein_context)
            for path in self.meta_paths["drug"].values()
        ]
        disease_views = [
            self._encode_path(graph, initial, path, protein_context)
            for path in self.meta_paths["disease"].values()
        ]

        drug_embedding, drug_weights = self._fuse_views(
            "drug", drug_views, initial["drug"]
        )
        disease_embedding, disease_weights = self._fuse_views(
            "disease", disease_views, initial["disease"]
        )

        embeddings = {
            "drug": drug_embedding,
            "disease": disease_embedding,
            "protein": protein_context,
        }
        attention_weights = {
            "drug": drug_weights,
            "disease": disease_weights,
            "protein": protein_weights,
        }
        return initial, embeddings, attention_weights


class AuxiliaryMLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        drug_embedding: torch.Tensor,
        disease_embedding: torch.Tensor,
    ) -> torch.Tensor:
        pair_representation = torch.cat(
            (drug_embedding, disease_embedding), dim=-1
        )
        return self.network(pair_representation)


class MSeR2TrainingModel(nn.Module):
    def __init__(
        self,
        input_dims: Mapping[str, int],
        config: Mapping[str, object],
        meta_paths: Mapping[str, Mapping[str, Sequence[CanonicalEType]]],
    ) -> None:
        super().__init__()
        self.encoder = MSeR2Encoder(input_dims, config, meta_paths)
        self.auxiliary_mlp = AuxiliaryMLP(
            int(config["hidden_dim"]), float(config["dropout"])
        )

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        features: Mapping[str, torch.Tensor],
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        _, embeddings, _ = self.encoder(graph, features)
        drug_embedding = embeddings["drug"][pairs[:, 0]]
        disease_embedding = embeddings["disease"][pairs[:, 1]]
        return self.auxiliary_mlp(drug_embedding, disease_embedding)


def _input_dims(features: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    return {key: int(value.shape[1]) for key, value in features.items()}


def _train_epoch(
    model: MSeR2TrainingModel,
    graph: dgl.DGLHeteroGraph,
    features: Mapping[str, torch.Tensor],
    pairs: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
) -> float:
    model.train()
    order = torch.randperm(len(pairs), device=pairs.device)
    total_loss = 0.0
    for start in range(0, len(pairs), batch_size):
        batch_index = order[start : start + batch_size]
        batch_pairs = pairs[batch_index]
        batch_labels = labels[batch_index]
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph, features, batch_pairs)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(batch_index)
    return total_loss / len(pairs)


@torch.no_grad()
def _validation_auc(
    model: MSeR2TrainingModel,
    graph: dgl.DGLHeteroGraph,
    features: Mapping[str, torch.Tensor],
    pairs: torch.Tensor,
    labels: np.ndarray,
) -> float:
    model.eval()
    probability = torch.softmax(model(graph, features, pairs), dim=1)[:, 1]
    return float(roc_auc_score(labels, probability.cpu().numpy()))


def _new_training_model(
    features: Mapping[str, torch.Tensor],
    config: Mapping[str, object],
    meta_paths: Mapping[str, Mapping[str, Sequence[CanonicalEType]]],
    device: torch.device,
) -> MSeR2TrainingModel:
    return MSeR2TrainingModel(
        input_dims=_input_dims(features),
        config=config,
        meta_paths=meta_paths,
    ).to(device)


def train_encoder_for_fold(
    graph: dgl.DGLHeteroGraph,
    features: Mapping[str, torch.Tensor],
    train_pairs: np.ndarray,
    train_labels: np.ndarray,
    config: Mapping[str, object] | None,
    device: torch.device,
    seed: int,
    verbose: bool = True,
) -> Tuple[MSeR2Encoder, Dict[str, np.ndarray], int, float]:
    cfg = deepcopy(DEFAULT_ENCODER_CONFIG)
    if config is not None:
        cfg.update(config)
    set_seed(seed)
    meta_paths = get_meta_paths(cfg)

    inner_train_pairs, validation_pairs, inner_train_labels, validation_labels = (
        train_test_split(
            train_pairs,
            train_labels,
            test_size=float(cfg["valid_ratio"]),
            random_state=seed,
            stratify=train_labels,
        )
    )
    inner_train_pairs_t = torch.as_tensor(inner_train_pairs, dtype=torch.long, device=device)
    inner_train_labels_t = torch.as_tensor(inner_train_labels, dtype=torch.long, device=device)
    validation_pairs_t = torch.as_tensor(validation_pairs, dtype=torch.long, device=device)

    selection_model = _new_training_model(features, cfg, meta_paths, device)
    optimizer = torch.optim.Adam(
        selection_model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["epochs"]),
        eta_min=1e-5,
    )
    criterion = nn.CrossEntropyLoss()
    best_epoch = 1
    best_validation_auc = float("-inf")

    for epoch in range(1, int(cfg["epochs"]) + 1):
        loss = _train_epoch(
            selection_model,
            graph,
            features,
            inner_train_pairs_t,
            inner_train_labels_t,
            optimizer,
            criterion,
            int(cfg["batch_size"]),
        )
        scheduler.step()
        validation_auc = _validation_auc(
            selection_model,
            graph,
            features,
            validation_pairs_t,
            validation_labels,
        )
        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            best_epoch = epoch
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == int(cfg["epochs"])):
            print(
                f"  encoder epoch {epoch:03d} | loss {loss:.4f} | "
                f"validation AUC {validation_auc:.4f}"
            )

    # Refit a fresh encoder/auxiliary-MLP model on every outer-training pair.
    set_seed(seed)
    final_model = _new_training_model(features, cfg, meta_paths, device)
    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        final_optimizer,
        T_max=best_epoch,
        eta_min=1e-5,
    )
    all_train_pairs_t = torch.as_tensor(train_pairs, dtype=torch.long, device=device)
    all_train_labels_t = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    for _ in range(best_epoch):
        _train_epoch(
            final_model,
            graph,
            features,
            all_train_pairs_t,
            all_train_labels_t,
            final_optimizer,
            criterion,
            int(cfg["batch_size"]),
        )
        final_scheduler.step()

    final_model.eval()
    with torch.no_grad():
        _, embeddings_t, _ = final_model.encoder(graph, features)
    embeddings = {
        node_type: tensor.detach().cpu().numpy().astype(np.float32)
        for node_type, tensor in embeddings_t.items()
    }
    return final_model.encoder, embeddings, best_epoch, best_validation_auc


def save_embedding_csv(
    embedding: np.ndarray,
    save_path: Path | str,
    prefix: str,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        embedding,
        columns=[f"{prefix}_{column}" for column in range(embedding.shape[1])],
    )
    frame.insert(0, "node_id", np.arange(len(frame), dtype=np.int64))
    frame.to_csv(save_path, index=False)


