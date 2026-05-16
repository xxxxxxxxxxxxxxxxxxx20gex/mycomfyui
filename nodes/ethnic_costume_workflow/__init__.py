from .node import (
    DynastyCostumePortraitStyler,
    EthnicCostumePortraitStyler,
    PromptImageEdit,
    PromptOptimizer,
)

NODE_CLASS_MAPPINGS = {
    "PromptImageEdit": PromptImageEdit,
    "PromptOptimizer": PromptOptimizer,
    "EthnicCostumePortraitStyler": EthnicCostumePortraitStyler,
    "DynastyCostumePortraitStyler": DynastyCostumePortraitStyler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptImageEdit": "提示词图像编辑",
    "PromptOptimizer": "提示词优化",
    "EthnicCostumePortraitStyler": "民族华服映像",
    "DynastyCostumePortraitStyler": "历代衣冠映像",
}
