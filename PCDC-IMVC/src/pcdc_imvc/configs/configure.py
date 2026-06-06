"""Unified default configurations for two-view and multi-view datasets."""

DATASET_REGISTRY = (
    {"name": "Caltech101-20", "ids": {"two": 0}, "data_file": "Caltech101-20.mat"},
    {"name": "MNIST-USPS", "ids": {"two": 1}, "data_file": "MNIST-USPS.mat"},
    {"name": "handwritten", "ids": {"multi": 0}, "data_file": "handwritten.mat"},
    {"name": "100leaves", "ids": {"multi": 1}, "data_file": "100leaves.mat"},
    {"name": "ALOI_100", "ids": {"multi": 2}, "data_file": "ALOI_100.mat"},
)


def _build_dataset_lookups():
    id_to_name = {"two": {}, "multi": {}}
    name_to_modes = {}
    for item in DATASET_REGISTRY:
        name = item["name"]
        ids = item.get("ids", {})
        modes = set(ids.keys())
        name_to_modes[name] = modes
        for mode, idx in ids.items():
            if idx in id_to_name[mode]:
                raise ValueError(f"Duplicate dataset id {idx} in mode {mode}")
            id_to_name[mode][idx] = name
    return id_to_name, name_to_modes


_ID_TO_NAME_BY_MODE, _NAME_TO_MODES = _build_dataset_lookups()
SUPPORTED_TWO_VIEW_DATASETS = {n for n, ms in _NAME_TO_MODES.items() if "two" in ms}
SUPPORTED_MULTI_VIEW_DATASETS = {n for n, ms in _NAME_TO_MODES.items() if "multi" in ms}


def _cfg_two_caltech101_20():
    return dict(
        seed=4,
        view=2,
        training=dict(
            start_inference=100,
            batch_size=256,
            epoch=500,
            alpha=10,
            beta=10,
            gamma=5,
            lambda2=8,
            lambda1=0.1,
            lambda3=0.4,
            lambda4=0.08,
            lr=1.0e-4,
            class_num=20,
            inf_loss="mse",
            core_by_missing=True,
            cycle_consistency=True,
            diveq=dict(K=32, tau=5.0, hard_eval=False),
        ),
        Autoencoder=dict(
            arch1=[1984, 1024, 256, 128],
            arch2=[512, 256, 128],
            activations1="relu",
            activations2="relu",
            batchnorm=True,
        ),
        Inference=dict(
            arch1=[128, 256, 128],
            arch2=[128, 256, 128],
        ),
    )


def _cfg_two_mnist_usps():
    return dict(
        seed=42,
        view=2,
        augmentation=dict(enable=False, feature_dropout=0.0, gaussian_std=0.0),
        Autoencoder=dict(
            arch1=[784, 1024, 1024, 1024, 40],
            arch2=[784, 1024, 1024, 1024, 40],
            activations1="relu",
            activations2="relu",
            batchnorm=True,
        ),
        Inference=dict(
            arch1=[128, 256, 128],
            arch2=[128, 256, 128],
        ),
        training=dict(
            lr=1.0e-4,
            start_inference=200,
            batch_size=256,
            epoch=500,
            alpha=10,
            lambda1=0.1,
            lambda2=8,
            lambda3=0.4,
            lambda4=0.1,
            class_num=10,
            core_by_missing=True,
            cycle_consistency=True,
            diveq=dict(K=16, tau=5.0, hard_eval=False),
        ),
    )



def _cfg_multi_handwritten():
    return dict(
        seed=42,
        view=6,
        Autoencoder=dict(
            arch1=[240, 512, 128, 40],
            arch2=[76, 128, 40],
            arch3=[216, 512, 128, 40],
            arch4=[47, 64, 40],
            arch5=[64, 128, 40],
            arch6=[6, 32, 40],
            activations="relu",
            batchnorm=True,
        ),
        Inference=dict(
            shared_hidden=[128, 64],
            arch1=[128, 256, 128],
            arch2=[128, 256, 128],
            arch3=[128, 256, 128],
            arch4=[128, 256, 128],
            arch5=[128, 256, 128],
            arch6=[128, 256, 128],
        ),
        training=dict(
            lr=1.0e-4,
            start_inference=100,
            batch_size=256,
            epoch=500,
            alpha=10,
            lambda1=0.2,
            lambda2=1,
            lambda3=10,
            lambda4=0.08,
            cycle_consistency=True,
            class_num=10,
            inf_loss="mse",
            core_by_missing=True,
            diveq=dict(enable=True, K=16, tau=5.0, hard_eval=False),
        ),
    )


def _cfg_multi_100leaves():
    return dict(
        seed=42,
        view=3,
        Autoencoder=dict(
            arch1=[64, 1024, 1024, 1024, 40],
            arch2=[64, 1024, 1024, 1024, 40],
            arch3=[64, 1024, 1024, 1024, 40],
            activations="relu",
            batchnorm=True,
        ),
        Inference=dict(
            shared_hidden=[128, 64],
            arch1=[128, 256, 128],
            arch2=[128, 256, 128],
            arch3=[128, 256, 128],
        ),
        training=dict(
            lr=1.0e-4,
            start_inference=100,
            batch_size=256,
            epoch=500,
            alpha=10,
            # lambda1=0.3,
            # lambda2=4,
            # lambda3=7.8,
            # lambda4=0.08,
            lambda1=0.12,
            lambda2=0.12,
            lambda3=5,
            lambda4=0.2,
            cycle_consistency=True,
            class_num=100,
            inf_loss="mse",
            core_by_missing=True,
            diveq=dict(K=128, tau=1.0, hard_eval=False),
        ),
    )


def _cfg_multi_aloi_100():
    return dict(
        seed=42,
        view=4,
        input_norm=dict(enable=True, method="zscore", clip=5.0, eps=1e-12),
        Autoencoder=dict(
            arch1=[77, 256, 512, 256, 40],
            arch2=[13, 64, 128, 64, 40],
            arch3=[64, 256, 512, 256, 40],
            arch4=[125, 512, 1024, 512, 40],
            activations="relu",
            batchnorm=True,
        ),
        Inference=dict(
            shared_hidden=[128, 64],
            arch1=[128, 256, 128],
            arch2=[128, 256, 128],
            arch3=[128, 256, 128],
            arch4=[128, 256, 128],
        ),
        training=dict(
            lr=1.0e-4,
            start_inference=100,
            batch_size=256,
            epoch=500,
            alpha=10,
            lambda1=0.12,
            lambda2=0.12,
            lambda3=5,
            lambda4=0.5,
            cycle_consistency=True,
            class_num=100,
            inf_loss="mse",
            core_by_missing=True,
            diveq=dict(enable=True, K=128, tau=1.0, hard_eval=False),
        ),
    )



_TWO_VIEW_FACTORY = {
    "Caltech101-20": _cfg_two_caltech101_20,
    "MNIST-USPS": _cfg_two_mnist_usps,
}

_MULTI_VIEW_FACTORY = {
    "handwritten": _cfg_multi_handwritten,
    "100leaves": _cfg_multi_100leaves,
    "ALOI_100": _cfg_multi_aloi_100,
}


def get_dataset_name_by_id(dataset_id, mode):
    if mode not in {"two", "multi"}:
        raise ValueError(f"Unknown mode: {mode}")
    try:
        return _ID_TO_NAME_BY_MODE[mode][int(dataset_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset id {dataset_id} for mode {mode}") from exc


def infer_mode_for_dataset_name(dataset_name):
    modes = _NAME_TO_MODES.get(dataset_name, set())
    if not modes:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    if len(modes) > 1:
        raise ValueError(f"Ambiguous dataset_name in auto mode: {dataset_name}. Please set mode.")
    return next(iter(modes))


def resolve_dataset_input(dataset=None, dataset_name=None, mode="auto"):
    if mode not in {"auto", "two", "multi"}:
        raise ValueError(f"Unknown mode: {mode}")

    if dataset_name is not None:
        if mode == "auto":
            resolved_mode = infer_mode_for_dataset_name(dataset_name)
        else:
            resolved_mode = mode
            allowed = _NAME_TO_MODES.get(dataset_name, set())
            if resolved_mode not in allowed:
                raise ValueError(f"dataset_name {dataset_name} is not registered for mode {resolved_mode}")
        return dataset_name, resolved_mode

    if dataset is None:
        raise ValueError("Please provide dataset_name or dataset id")

    if mode == "auto":
        in_two = int(dataset) in _ID_TO_NAME_BY_MODE["two"]
        in_multi = int(dataset) in _ID_TO_NAME_BY_MODE["multi"]
        if in_two and not in_multi:
            return _ID_TO_NAME_BY_MODE["two"][int(dataset)], "two"
        if in_multi and not in_two:
            return _ID_TO_NAME_BY_MODE["multi"][int(dataset)], "multi"
        if in_two and in_multi:
            raise ValueError(f"Ambiguous dataset id in auto mode: {dataset}. Please set mode.")
        raise ValueError(f"Unknown dataset id: {dataset}")

    return get_dataset_name_by_id(int(dataset), mode), mode


def get_dataset_registry():
    return DATASET_REGISTRY


def get_required_data_files():
    files = []
    for item in DATASET_REGISTRY:
        f = item.get("data_file")
        if f:
            files.append(f)
    return files


def validate_config(config, mode="auto"):
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    required_top = ["view", "training", "Autoencoder", "Inference"]
    missing = [k for k in required_top if k not in config]
    if missing:
        raise ValueError(f"missing config keys: {missing}")

    if not isinstance(config["training"], dict):
        raise TypeError("training must be a dict")

    tr_required = ["lr", "epoch", "batch_size", "class_num"]
    tr_missing = [k for k in tr_required if k not in config["training"]]
    if tr_missing:
        raise ValueError(f"missing training keys: {tr_missing}")

    view = int(config["view"])
    if mode == "two" and view != 2:
        raise ValueError(f"mode=two requires view=2, got view={view}")
    if mode == "multi" and view < 2:
        raise ValueError(f"mode=multi requires view>=2, got view={view}")

    return True


def get_default_config(data_name, mode="auto"):
    if mode not in {"auto", "two", "multi"}:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "two":
        if data_name not in _TWO_VIEW_FACTORY:
            raise Exception(f"Undefined two-view data name: {data_name}")
        cfg = _TWO_VIEW_FACTORY[data_name]()
        cfg.setdefault("Completion", cfg.get("Inference", {}))
        return cfg

    if mode == "multi":
        if data_name not in _MULTI_VIEW_FACTORY:
            raise Exception(f"Undefined multi-view data name: {data_name}")
        cfg = _MULTI_VIEW_FACTORY[data_name]()
        cfg.setdefault("Completion", cfg.get("Inference", {}))
        return cfg

    if data_name in _TWO_VIEW_FACTORY and data_name not in _MULTI_VIEW_FACTORY:
        cfg = _TWO_VIEW_FACTORY[data_name]()
        cfg.setdefault("Completion", cfg.get("Inference", {}))
        return cfg
    if data_name in _MULTI_VIEW_FACTORY and data_name not in _TWO_VIEW_FACTORY:
        cfg = _MULTI_VIEW_FACTORY[data_name]()
        cfg.setdefault("Completion", cfg.get("Inference", {}))
        return cfg
    if data_name in _TWO_VIEW_FACTORY and data_name in _MULTI_VIEW_FACTORY:
        raise Exception(f"Ambiguous data name in auto mode: {data_name}. Please set mode explicitly.")

    raise Exception(f"Undefined data name: {data_name}")