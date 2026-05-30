import torch
from comfy_api.latest import IO

from ..api import edit_image, generate_image, image_tensor_to_edit_files, response_to_tensor
from ..prompts import ETHNIC_COSTUME_SYSTEM_PROMPT, SCENE_PROMPTS, costume_style_prompt, join_prompt_sections
from .common import validate_string


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


async def execute_style(
    cls: type[IO.ComfyNode],
    costume_style: str,
    costume_gender: str,
    scene: str,
    quality: str,
    size: str,
    image_count: int,
    person_image: torch.Tensor | None,
    mask: torch.Tensor | None,
    *,
    style_label: str,
    style_guard: str,
    scene_prompts: dict[str, str],
    additional_requirements: str,
) -> IO.NodeOutput:
    validate_string(costume_style, strip_whitespace=True)
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
        response = await generate_image(prompt, quality=quality, size=size, n=image_count, node_cls=cls)
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
        response = await edit_image(prompt, files, quality=quality, size=size, n=image_count, node_cls=cls)
    return IO.NodeOutput(await response_to_tensor(response))
