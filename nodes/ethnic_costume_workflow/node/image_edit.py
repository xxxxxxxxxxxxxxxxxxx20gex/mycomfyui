import torch
from comfy_api.latest import IO

from ..api import edit_image, image_tensor_to_edit_files, response_to_tensor
from .common import price_badge, validate_string
from .options import IMAGE_SIZE_OPTIONS


class PromptImageEdit(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PromptImageEdit",
            display_name="提示词图像编辑",
            category="民族服饰/图像工具",
            description="输入图片和提示词，调用图像编辑接口生成修改后的图片。",
            inputs=[
                IO.Image.Input("image", tooltip="需要编辑的图片。"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="保留人物身份和构图，仅根据提示词编辑服饰、配饰、背景或画面细节。",
                    tooltip="编辑提示词。说明要修改什么，也可以写明需要保留的内容。",
                ),
                IO.Mask.Input("mask", tooltip="可选遮罩；不连接时由模型根据提示词自行编辑。", optional=True),
                IO.Combo.Input("quality", default="medium", options=["low", "medium", "high"], tooltip="画面质量。"),
                IO.Combo.Input(
                    "size",
                    default="1024x1536",
                    options=IMAGE_SIZE_OPTIONS,
                    tooltip="输出尺寸。",
                ),
                IO.Combo.Input(
                    "background",
                    default="opaque",
                    options=["auto", "opaque"],
                    tooltip="背景模式；当前图像模型不支持透明背景。",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="随机种子，用于记录本次编辑；图片接口可能不保证严格复现。",
                ),
            ],
            outputs=[IO.Image.Output()],
            is_api_node=False,
            price_badge=price_badge(),
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        prompt: str,
        quality: str = "medium",
        size: str = "1024x1536",
        background: str = "opaque",
        seed: int = 0,
        mask: torch.Tensor | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True)
        files = image_tensor_to_edit_files(image, mask)
        response = await edit_image(
            prompt.strip(),
            files,
            quality=quality,
            size=size,
            background=background,
            node_cls=cls,
        )
        return IO.NodeOutput(await response_to_tensor(response))
