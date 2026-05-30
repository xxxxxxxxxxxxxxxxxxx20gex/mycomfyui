from comfy_api.latest import IO

from ..api import call_text_model
from .common import validate_string


PROMPT_OPTIMIZER_SYSTEM_PROMPT = """
You are a prompt editor for image generation and image editing workflows.
Rewrite the user's prompt into one clear, production-ready prompt.
Keep the user's intent, remove ambiguity, add concrete visual details only when helpful, and preserve any explicit constraints.
Return only the optimized prompt text. Do not add explanations, markdown, quotes, JSON, or alternatives.
""".strip()


class PromptOptimizer(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PromptOptimizer",
            display_name="提示词优化",
            category="服饰/文本工具",
            description="输入一段提示词，输出优化后的提示词。",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="需要优化的原始提示词。",
                ),
                IO.Combo.Input(
                    "language",
                    default="中文",
                    options=["中文", "English", "保持原语言"],
                    tooltip="优化后提示词的输出语言。",
                ),
                IO.String.Input(
                    "requirements",
                    multiline=True,
                    default="适合图像生成或图像编辑；画面描述具体；保留用户原始意图；不要生成解释文本。",
                    tooltip="可选优化要求，例如风格、长度、保留项、禁忌项等。",
                    optional=True,
                ),
            ],
            outputs=[IO.String.Output(display_name="optimized_prompt")],
            is_api_node=False,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        language: str = "中文",
        requirements: str = "",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True)
        language_instruction = {
            "中文": "Output the optimized prompt in Chinese.",
            "English": "Output the optimized prompt in English.",
            "保持原语言": "Output the optimized prompt in the same language as the user's prompt.",
        }.get(language, "Output the optimized prompt in the same language as the user's prompt.")

        user_content = "\n".join(
            [
                f"Output language: {language_instruction}",
                f"Optimization requirements: {(requirements or '').strip() or 'No extra requirements.'}",
                f"Original prompt: {prompt.strip()}",
            ]
        )
        optimized_prompt = await call_text_model(
            [
                {"role": "system", "content": PROMPT_OPTIMIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )
        validate_string(optimized_prompt, strip_whitespace=True)
        return IO.NodeOutput(optimized_prompt.strip())
