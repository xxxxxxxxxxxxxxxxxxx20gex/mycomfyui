# My ComfyUI Nodes and Workflows

自制的一些 ComfyUI 模块和工作流。

## Structure

- `nodes/ethnic_costume_workflow/`: custom nodes for ethnic and dynasty costume image workflows.
- `nodes/ComfyUI-Wan22FMLF/`: Wan2.2 first/middle/last-frame video workflow nodes.
- `workflows/user-workflows/`: workflows exported from `user/default/workflows`.
- `workflows/example-workflows/`: example workflow JSON files shipped with the custom nodes.

Large model files, logs, caches, and ComfyUI runtime files are intentionally excluded.

## Wan22FMLF Workflow

独立工作流文件：

- `workflows/user-workflows/Wan22FMLF_首中尾帧独立工作流.app.json`

配套节点：

- `nodes/ComfyUI-Wan22FMLF/`

精简前置条件：

- ComfyUI 可正常启动，并把 `nodes/ComfyUI-Wan22FMLF` 放到 `ComfyUI/custom_nodes/ComfyUI-Wan22FMLF`。
- 需要 Wan2.2 I2V high/low diffusion models：
  - `models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors`
  - `models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors`
- 需要 Wan VAE：
  - `models/vae/wan_2.1_vae.safetensors`
- 需要 Wan 文本编码器：
  - `models/text_encoders/umt5_xxl_fp16.safetensors`
- 工作流默认使用 LightX2V 4-step LoRA：
  - `models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors`
  - `models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors`
- 工作流需要三张输入图：首帧、中间帧、尾帧。导入后请在三个 `LoadImage` 节点里重新选择本机图片。
- 建议显存 24GB 级别机器运行；首次测试可先保持低步数、短长度，确认能跑通后再提高参数。

说明：

- 仓库不包含大模型文件、输入图片、输出视频、日志和缓存。
- 该工作流与已删除的“人物动作映像”链路无关，不依赖 WanAnimatePreprocess、DWPose 或人物动作映像相关扩展。
