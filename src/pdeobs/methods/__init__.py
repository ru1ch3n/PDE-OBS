"""Built-in baselines and extension API."""

from .base import (
    METHOD_REGISTRY,
    Method,
    MethodCapabilities,
    available_methods,
    capabilities_for,
    create_method,
    discover_methods,
    method_discovery_errors,
    register_method,
)
from .interpolation import (
    BilinearInterpolation,
    MeanFill,
    NearestInterpolation,
    Persistence,
    RBFInterpolation,
    RBFSpatialPersistence,
    ZeroFill,
)

# Neural dependencies are optional at import time. neural.py provides friendly
# construction errors when PyTorch is unavailable.
from .neural import (  # noqa: E402
    AutoregressiveModel,
    CompactCNO2d,
    CompactFNO2d,
    CompactResidualEncoder,
    ConvLSTM,
    MaskChannelUNet,
    MaskedAutoencoderSmall,
    PhysicsInformedFNO2d,
    UFNO2d,
    create_autoregressive_fno,
    create_model,
)
from .operator_networks import (  # noqa: E402
    MaskConditionedGNOT,
    MaskConditionedTransolver,
    PaperGuidedMaskUNet,
    VariableSensorDeepONet,
)
from .paper_official import (  # noqa: E402
    OfficialPaperCNO2d,
    OfficialPaperFNO2d,
    OfficialPaperUNet,
)
from .reduced_order import GappyPOD, GappyPODDMD

BUILTIN_METHODS = {
    "zero": ZeroFill,
    "mean": MeanFill,
    "nearest": NearestInterpolation,
    "bilinear": BilinearInterpolation,
    "rbf": RBFInterpolation,
    "gappy_pod": GappyPOD,
    "gappy_pod_dmd": GappyPODDMD,
    "persistence": Persistence,
    "rbf_persistence": RBFSpatialPersistence,
    "unet": MaskChannelUNet,
    "fno": CompactFNO2d,
    "ufno": UFNO2d,
    "pino": PhysicsInformedFNO2d,
    "cno": CompactCNO2d,
    "convlstm": ConvLSTM,
    "autoregressive": AutoregressiveModel,
    "autoregressive_fno": create_autoregressive_fno,
    "residual_cnn": CompactResidualEncoder,
    "mae_small": MaskedAutoencoderSmall,
    "paper_unet": OfficialPaperUNet,
    "paper_fno": OfficialPaperFNO2d,
    "paper_cno": OfficialPaperCNO2d,
    "unet_paper_guided": PaperGuidedMaskUNet,
    "deeponet": VariableSensorDeepONet,
    "transolver": MaskConditionedTransolver,
    "gnot": MaskConditionedGNOT,
}


def install_builtin_methods(registry=None) -> tuple[str, ...]:
    """Mirror built-ins into the project-wide registry when it is available.

    The method package retains its small standalone registry so external code
    can import it independently. CLI startup may call this helper to expose the
    same factories through :mod:`pdeobs.registry`.
    """

    if registry is None:
        try:
            from ..registry import METHOD_REGISTRY as registry
        except (ImportError, AttributeError):
            return ()
    installed = []
    for name, factory in BUILTIN_METHODS.items():
        if name not in registry:
            registry.register(name, obj=factory)
            installed.append(name)
    return tuple(installed)


# Importing pdeobs.methods is an explicit request for method functionality, so
# synchronizing the central registry here does not burden lightweight imports.
install_builtin_methods()

__all__ = [
    "METHOD_REGISTRY",
    "Method",
    "MethodCapabilities",
    "available_methods",
    "capabilities_for",
    "create_method",
    "discover_methods",
    "method_discovery_errors",
    "register_method",
    "ZeroFill",
    "MeanFill",
    "NearestInterpolation",
    "BilinearInterpolation",
    "RBFInterpolation",
    "GappyPOD",
    "GappyPODDMD",
    "Persistence",
    "RBFSpatialPersistence",
    "MaskChannelUNet",
    "CompactFNO2d",
    "UFNO2d",
    "PhysicsInformedFNO2d",
    "CompactCNO2d",
    "CompactResidualEncoder",
    "ConvLSTM",
    "AutoregressiveModel",
    "MaskedAutoencoderSmall",
    "create_model",
    "create_autoregressive_fno",
    "OfficialPaperUNet",
    "OfficialPaperFNO2d",
    "OfficialPaperCNO2d",
    "PaperGuidedMaskUNet",
    "VariableSensorDeepONet",
    "MaskConditionedTransolver",
    "MaskConditionedGNOT",
    "BUILTIN_METHODS",
    "install_builtin_methods",
]
