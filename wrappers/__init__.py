"""
Model Registry: maps dataset → model_id → wrapper class.

The primary models (classical_dnn, hqnn, classical_transformer) now
use the user's pre-trained checkpoints from projectfinalhqnn/.
The existing surrogate models remain available under _surrogate suffixes
if needed, but are no longer the default.
"""
from .classical_knn import ClassicalKNNModel
from .quantum_knn import QuantumKNNModel
from .classical_dnn import ClassicalDNNModel
from .hqnn import HQNNModelWrapper
from .classical_transformer import ClassicalTransformerModel
from .quantum_transformer import QuantumTransformerModel

# Pre-trained models from user's project defense checkpoints
from .pretrained_classical_dnn import PretrainedClassicalDNNModel
from .pretrained_hqnn import PretrainedHQNNModel
from .pretrained_transformer import PretrainedClassicalTransformerModel

__all__ = [
    "ClassicalKNNModel",
    "QuantumKNNModel",
    "ClassicalDNNModel",
    "HQNNModelWrapper",
    "ClassicalTransformerModel",
    "QuantumTransformerModel",
    "PretrainedClassicalDNNModel",
    "PretrainedHQNNModel",
    "PretrainedClassicalTransformerModel",
]

REGISTRY = {
    "crop": {
        "classical_knn": ClassicalKNNModel,
        "quantum_knn": QuantumKNNModel,
    },
    "breast_cancer": {
        "classical_dnn": PretrainedClassicalDNNModel,   # Pre-trained Keras DNN
        "hqnn": PretrainedHQNNModel,                    # Pre-trained HQNN (.pth)
    },
    "imdb": {
        "classical_transformer": PretrainedClassicalTransformerModel,  # Pre-trained DistilBERT
        "quantum_transformer": QuantumTransformerModel,                # PennyLane quantum
    },
}

# Singleton cache
_instances: dict = {}


def get_model(dataset: str, model_id: str):
    key = f"{dataset}:{model_id}"
    if key not in _instances:
        cls = REGISTRY.get(dataset, {}).get(model_id)
        if cls is None:
            raise ValueError(f"No model '{model_id}' for dataset '{dataset}'")
        _instances[key] = cls()
    return _instances[key]
