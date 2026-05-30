import torch
from comfy_api.latest import IO

from ..prompts import CHINESE_ETHNIC_GROUPS, SCENE_PROMPTS
from .common import price_badge
from .options import IMAGE_SIZE_OPTIONS
from .portrait_base import ETHNIC_REQUIREMENTS, ETHNIC_STYLE_GUARD, execute_style


class EthnicCostumePortraitStyler(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EthnicCostumePortraitStyler",
            display_name="民族华服映像",
            category="服饰",
            description="上传人物照片并选择民族，自动换装为对应民族服饰；不上传则直接生成人像。",
            inputs=[
                IO.Image.Input(
                    "person_image",
                    tooltip="可选人物照片：上传后换装；不上传则自动生成人物和所选民族服饰。",
                    optional=True,
                ),
                IO.Mask.Input("mask", tooltip="可选遮罩；普通上传照片无需遮罩。", optional=True),
                IO.Combo.Input("ethnicity", default="汉族", options=CHINESE_ETHNIC_GROUPS, tooltip="选择中国民族。"),
                IO.Combo.Input("costume_gender", default="女装", options=["女装", "男装"], tooltip="选择服饰方向。"),
                IO.Combo.Input(
                    "scene",
                    default="保留原背景",
                    options=["保留原背景", "影棚写真", "自然风景", "节庆氛围", "纯色背景"],
                    tooltip="选择场景。默认保留用户上传照片的原背景。",
                ),
                IO.Combo.Input("quality", default="medium", options=["low", "medium", "high"], tooltip="画面质量。"),
                IO.Combo.Input(
                    "size",
                    default="1024x1536",
                    options=IMAGE_SIZE_OPTIONS,
                    tooltip="输出尺寸；小尺寸通常更快，竖图适合人像，横图适合半身或场景。",
                ),
                IO.Int.Input(
                    "image_count",
                    default=4,
                    min=1,
                    max=8,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="一次输出的图片数量。",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="随机种子，用于记录本次生成；图片接口可能不保证严格复现。",
                ),
            ],
            outputs=[IO.Image.Output()],
            is_api_node=False,
            price_badge=price_badge(),
        )

    @classmethod
    async def execute(
        cls,
        ethnicity: str,
        costume_gender: str,
        scene: str,
        quality: str = "medium",
        size: str = "1024x1536",
        image_count: int = 4,
        seed: int = 0,
        person_image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> IO.NodeOutput:
        return await execute_style(
            cls,
            ethnicity,
            costume_gender,
            scene,
            quality,
            size,
            image_count,
            person_image,
            mask,
            style_label="Selected ethnicity",
            style_guard=ETHNIC_STYLE_GUARD,
            scene_prompts=SCENE_PROMPTS,
            additional_requirements=ETHNIC_REQUIREMENTS,
        )
