import torch
from comfy_api.latest import IO

from ..api import edit_image, generate_image, image_tensor_to_edit_files, response_to_tensor
from ..prompts import (
    CHINESE_ETHNIC_GROUPS,
    DYNASTY_COSTUME_STYLES,
    ETHNIC_COSTUME_SYSTEM_PROMPT,
    SCENE_PROMPTS,
    costume_style_prompt,
    join_prompt_sections,
)
from .common import price_badge, validate_string
from .options import IMAGE_SIZE_OPTIONS, normalize_image_size


ETHNIC_STYLE_GUARD = "不要生成现代普通服装、朝代官服、泛古装或其他民族服饰。"
ETHNIC_REQUIREMENTS = (
    "high detail, realistic textile texture, accurate garment construction, culturally respectful, "
    "avoid caricature, avoid random cultural mixing"
)
DYNASTY_STYLE_GUARD = "不要生成现代普通服装、民族节庆服、仙侠服、影楼古装或戏曲盔甲。"
DYNASTY_REQUIREMENTS = (
    "high detail, historically grounded garment construction, accurate period silhouette, "
    "clear rank or era details, avoid fantasy styling, avoid random dynasty mixing"
)
DYNASTY_SCENE_PROMPTS = {
    **SCENE_PROMPTS,
    "古典园林": "融合克制的中式古典园林背景，亭台、廊柱、石阶或竹木元素自然虚化，光线与人物一致，服饰仍为主体。",
    "宫廷氛围": "加入克制的宫廷建筑或室内陈设氛围，背景不要喧宾夺主，人物和服制细节保持清晰。",
}

def _generate_prompt(
    costume_style: str,
    costume_gender: str,
    scene: str,
    *,
    style_label: str,
    style_guard: str,
    scene_prompts: dict[str, str],
    additional_requirements: str,
) -> str:
    return join_prompt_sections(
        [
            ("Task", "Create a new realistic person portrait or full-body image wearing the selected traditional or historical clothing."),
            ("Cultural clothing guidance", ETHNIC_COSTUME_SYSTEM_PROMPT),
            (style_label, costume_style),
            ("Costume direction", f"{costume_gender}，必须是{costume_style}对应的传统或历史服饰，{style_guard}"),
            ("Target costume", costume_style_prompt(costume_style)),
            ("Person", "成年人物，自然神态，姿态端庄，服饰完整清晰；根据服饰方向选择自然协调的人物呈现。"),
            (
                "Scene",
                "自然光影棚写真质感，人物主体清晰，服装结构、纹样、织物材质和配饰细节可见。"
                if scene == "保留原背景"
                else scene_prompts.get(scene, SCENE_PROMPTS["影棚写真"]),
            ),
            ("Additional requirements", additional_requirements),
        ]
    )


def _edit_prompt(
    costume_style: str,
    costume_gender: str,
    scene: str,
    *,
    style_label: str,
    style_guard: str,
    scene_prompts: dict[str, str],
    additional_requirements: str,
) -> str:
    return join_prompt_sections(
        [
            ("Task", "Perform a clothing replacement edit on the uploaded person photo."),
            ("Cultural clothing guidance", ETHNIC_COSTUME_SYSTEM_PROMPT),
            (style_label, costume_style),
            ("Costume direction", f"{costume_gender}，必须是{costume_style}对应的传统或历史服饰，{style_guard}"),
            ("Target costume", costume_style_prompt(costume_style)),
            (
                "Preserve",
                "保留人物身份特征、脸部、五官、发型、体型、姿态、年龄感和原始构图；不要改变人脸，不要改变人物数量。",
            ),
            ("Scene", scene_prompts.get(scene, SCENE_PROMPTS["保留原背景"])),
            ("Additional requirements", f"{additional_requirements}, only replace clothing and related accessories"),
        ]
    )


async def _execute_style(
    cls: type[IO.ComfyNode],
    costume_style: str,
    costume_gender: str,
    scene: str,
    quality: str,
    size: str,
    person_image: torch.Tensor | None,
    mask: torch.Tensor | None,
    *,
    style_label: str,
    style_guard: str,
    scene_prompts: dict[str, str],
    additional_requirements: str,
) -> IO.NodeOutput:
    validate_string(costume_style, strip_whitespace=True)
    size = normalize_image_size(size)
    if person_image is None:
        prompt = _generate_prompt(
            costume_style,
            costume_gender,
            scene,
            style_label=style_label,
            style_guard=style_guard,
            scene_prompts=scene_prompts,
            additional_requirements=additional_requirements,
        )
        response = await generate_image(prompt, quality=quality, size=size, node_cls=cls)
    else:
        prompt = _edit_prompt(
            costume_style,
            costume_gender,
            scene,
            style_label=style_label,
            style_guard=style_guard,
            scene_prompts=scene_prompts,
            additional_requirements=additional_requirements,
        )
        files = image_tensor_to_edit_files(person_image, mask)
        response = await edit_image(prompt, files, quality=quality, size=size, node_cls=cls)
    return IO.NodeOutput(await response_to_tensor(response))


class EthnicCostumePortraitStyler(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EthnicCostumePortraitStyler",
            display_name="民族华服映像",
            category="民族服饰",
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
        seed: int = 0,
        person_image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> IO.NodeOutput:
        return await _execute_style(
            cls,
            ethnicity,
            costume_gender,
            scene,
            quality,
            size,
            person_image,
            mask,
            style_label="Selected ethnicity",
            style_guard=ETHNIC_STYLE_GUARD,
            scene_prompts=SCENE_PROMPTS,
            additional_requirements=ETHNIC_REQUIREMENTS,
        )


class DynastyCostumePortraitStyler(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="DynastyCostumePortraitStyler",
            display_name="历代衣冠映像",
            category="民族服饰",
            description="上传人物照片并选择朝代或身份服饰，自动换装为对应历史服饰；不上传则直接生成人像。",
            inputs=[
                IO.Image.Input(
                    "person_image",
                    tooltip="可选人物照片：上传后换装；不上传则自动生成人物和所选朝代服饰。",
                    optional=True,
                ),
                IO.Mask.Input("mask", tooltip="可选遮罩；普通上传照片无需遮罩。", optional=True),
                IO.Combo.Input(
                    "dynasty_style",
                    default="明朝皇子",
                    options=DYNASTY_COSTUME_STYLES,
                    tooltip="选择朝代或身份服饰预设。",
                ),
                IO.Combo.Input("costume_gender", default="男装", options=["男装", "女装"], tooltip="选择服饰方向。"),
                IO.Combo.Input(
                    "scene",
                    default="保留原背景",
                    options=["保留原背景", "影棚写真", "古典园林", "宫廷氛围", "纯色背景"],
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
        dynasty_style: str,
        costume_gender: str,
        scene: str,
        quality: str = "medium",
        size: str = "1024x1536",
        seed: int = 0,
        person_image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> IO.NodeOutput:
        return await _execute_style(
            cls,
            dynasty_style,
            costume_gender,
            scene,
            quality,
            size,
            person_image,
            mask,
            style_label="Selected dynasty clothing",
            style_guard=DYNASTY_STYLE_GUARD,
            scene_prompts=DYNASTY_SCENE_PROMPTS,
            additional_requirements=DYNASTY_REQUIREMENTS,
        )
