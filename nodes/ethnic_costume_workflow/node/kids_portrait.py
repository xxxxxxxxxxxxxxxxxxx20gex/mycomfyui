import torch
from comfy_api.latest import IO

from ..api import edit_image, generate_image, image_tensor_to_edit_files, response_to_tensor
from ..prompts import join_prompt_sections
from .common import price_badge, validate_string
from .options import IMAGE_SIZE_OPTIONS


CARTOON_COWBOY_STYLE_PROMPTS = {
    "经典牛仔男孩": (
        "child-friendly western cowboy outfit, yellow plaid shirt, blue denim jeans, brown cowboy boots, "
        "wide brown cowboy hat, red neckerchief, cow-print vest, round sheriff-style badge, large belt buckle, "
        "cheerful 3D animated toy-like styling"
    ),
    "活泼牛仔女孩": (
        "child-friendly western cowgirl outfit, white and yellow shirt, blue denim bottoms, cow-print chaps or skirt accents, "
        "brown cowboy boots, wide cowboy hat, red neckerchief, playful belt buckle, bright 3D animated toy-like styling"
    ),
    "奶牛纹西部套装": (
        "western cowboy costume with distinctive black-and-white cow-print vest and leg accents, yellow plaid shirt, "
        "blue denim, brown leather hat and boots, red bandana, toy-like animated character costume, playful and wholesome"
    ),
}
CARTOON_COWBOY_SCENE_PROMPTS = {
    "保留原背景": "保留上传照片的原背景、光线、镜头视角和构图，只替换为儿童友好的动漫牛仔服饰。",
    "玩具房间": "柔和明亮的儿童玩具房间背景，色彩温暖，背景不要喧宾夺主，主体和服饰清晰。",
    "动画舞台": "明亮的卡通动画舞台背景，轻松欢乐，有柔和聚光灯和简洁装饰，适合儿童主题展示。",
    "西部小镇": "儿童友好的卡通西部小镇背景，木质街道、蓝天白云和温暖阳光，氛围活泼但不写实暴力。",
    "纯色背景": "简洁纯色背景，突出儿童友好的动漫牛仔服饰结构和色彩。",
}
CARTOON_COWBOY_REQUIREMENTS = (
    "3D animated toy-like character style, cheerful, wholesome, child-safe, colorful, polished, "
    "soft rounded features, no weapon, no cigar, no alcohol, no scary elements, no brand logo, "
    "do not copy any specific copyrighted character, avoid exact IP likeness"
)


def _cartoon_cowboy_generate_prompt(cowboy_style: str, character_direction: str, scene: str) -> str:
    return join_prompt_sections(
        [
            ("Task", "Create a cheerful child-friendly 3D animated character wearing a western cowboy costume."),
            ("Style direction", "Inspired by classic playful toy animation aesthetics, but create an original character."),
            ("Character direction", character_direction),
            ("Target outfit", CARTOON_COWBOY_STYLE_PROMPTS[cowboy_style]),
            ("Scene", CARTOON_COWBOY_SCENE_PROMPTS.get(scene, CARTOON_COWBOY_SCENE_PROMPTS["玩具房间"])),
            ("Preserve safety", "The result must be appropriate for children, friendly, bright, non-violent, and non-sexualized."),
            ("Additional requirements", CARTOON_COWBOY_REQUIREMENTS),
        ]
    )


def _cartoon_cowboy_edit_prompt(cowboy_style: str, character_direction: str, scene: str) -> str:
    return join_prompt_sections(
        [
            ("Task", "Perform a clothing replacement edit on the uploaded child or person photo."),
            ("Style direction", "Transform the outfit into an original child-friendly 3D animated western cowboy costume."),
            ("Character direction", character_direction),
            ("Target outfit", CARTOON_COWBOY_STYLE_PROMPTS[cowboy_style]),
            (
                "Preserve",
                "保留人物身份特征、脸部、五官、发型、体型、姿态、年龄感和原始构图；不要改变人脸，不要改变人物数量。",
            ),
            ("Scene", CARTOON_COWBOY_SCENE_PROMPTS.get(scene, CARTOON_COWBOY_SCENE_PROMPTS["保留原背景"])),
            (
                "Additional requirements",
                f"{CARTOON_COWBOY_REQUIREMENTS}, only replace clothing and related accessories, keep the result wholesome",
            ),
        ]
    )


class KidsCartoonCowboyStyler(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="KidsCartoonCowboyStyler",
            display_name="儿童动漫牛仔换装",
            category="服饰/儿童换装",
            description="上传儿童或人物照片，自动换成儿童友好的 3D 动漫牛仔服饰；不上传则直接生成动漫牛仔角色。",
            inputs=[
                IO.Image.Input(
                    "person_image",
                    tooltip="可选人物照片：上传后进行牛仔换装；不上传则自动生成儿童友好的动漫牛仔角色。",
                    optional=True,
                ),
                IO.Mask.Input("mask", tooltip="可选遮罩；普通上传照片无需遮罩。", optional=True),
                IO.Combo.Input(
                    "cowboy_style",
                    default="经典牛仔男孩",
                    options=list(CARTOON_COWBOY_STYLE_PROMPTS.keys()),
                    tooltip="选择儿童动漫牛仔服饰方向。",
                ),
                IO.Combo.Input(
                    "character_direction",
                    default="儿童角色",
                    options=["儿童角色", "男孩角色", "女孩角色", "亲子友好角色"],
                    tooltip="选择角色方向；上传照片时会尽量保留原人物身份和年龄感。",
                ),
                IO.Combo.Input(
                    "scene",
                    default="保留原背景",
                    options=list(CARTOON_COWBOY_SCENE_PROMPTS.keys()),
                    tooltip="选择场景。默认保留用户上传照片的原背景。",
                ),
                IO.Combo.Input("quality", default="medium", options=["low", "medium", "high"], tooltip="画面质量。"),
                IO.Combo.Input(
                    "size",
                    default="1024x1536",
                    options=IMAGE_SIZE_OPTIONS,
                    tooltip="输出尺寸；竖图适合角色照，方图适合头像或贴纸。",
                ),
                IO.Int.Input(
                    "image_count",
                    default=1,
                    min=1,
                    max=8,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="一次输出的图片数量；具体是否支持多图取决于后端图像接口。",
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
        cowboy_style: str,
        character_direction: str,
        scene: str,
        quality: str = "medium",
        size: str = "1024x1536",
        image_count: int = 1,
        seed: int = 0,
        person_image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> IO.NodeOutput:
        validate_string(cowboy_style, strip_whitespace=True)
        if cowboy_style not in CARTOON_COWBOY_STYLE_PROMPTS:
            raise ValueError(f"Unsupported cowboy style: {cowboy_style}")

        if person_image is None:
            prompt = _cartoon_cowboy_generate_prompt(cowboy_style, character_direction, scene)
            response = await generate_image(prompt, quality=quality, size=size, n=image_count, node_cls=cls)
        else:
            prompt = _cartoon_cowboy_edit_prompt(cowboy_style, character_direction, scene)
            files = image_tensor_to_edit_files(person_image, mask)
            response = await edit_image(prompt, files, quality=quality, size=size, n=image_count, node_cls=cls)
        return IO.NodeOutput(await response_to_tensor(response))
