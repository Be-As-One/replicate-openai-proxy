"""
Replicate to OpenAI Compatible API Proxy
将 Replicate 的图像生成 API 转换为 OpenAI 兼容格式
"""

import os
import time
import uuid
import asyncio
import httpx
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import replicate
from replicate.exceptions import ReplicateError
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型映射配置
# OpenAI 格式模型名 -> Replicate 模型标识 (owner/model-name)
MODEL_MAPPING = {
    # Seedream 4.5 (默认)
    "dall-e-3": "bytedance/seedream-4.5",
    "seedream-4.5": "bytedance/seedream-4.5",
    "seedream": "bytedance/seedream-4.5",
    
    # Flux 系列
    "flux-dev": "black-forest-labs/flux-dev",
    "flux-schnell": "black-forest-labs/flux-schnell",
    "flux-pro": "black-forest-labs/flux-pro",
    "flux-1.1-pro": "black-forest-labs/flux-1.1-pro",
    
    # SDXL
    "dall-e-2": "stability-ai/sdxl",
    "sdxl": "stability-ai/sdxl",
    
    # Stable Diffusion 3
    "sd-3": "stability-ai/stable-diffusion-3",
    "stable-diffusion-3": "stability-ai/stable-diffusion-3",
    
    # Ideogram
    "ideogram": "ideogram-ai/ideogram-v2",
    "ideogram-v2": "ideogram-ai/ideogram-v2",
    
    # Recraft
    "recraft-v3": "recraft-ai/recraft-v3",
    
    # Playground
    "playground-v2.5": "playgroundai/playground-v2.5-1024px-aesthetic",

    # Nano Banana (Google Gemini 2.5 Flash Image)
    "nanobanana": "google/nano-banana",
    "nano-banana": "google/nano-banana",
    "nanobananapro": "google/nano-banana",
    "nano-banana-pro": "google/nano-banana",
}

# 尺寸映射
SIZE_TO_ASPECT_RATIO = {
    "1024x1024": "1:1",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "512x512": "1:1",
    "256x256": "1:1",
    "1280x720": "16:9",
    "720x1280": "9:16",
    "1920x1080": "16:9",
    "1080x1920": "9:16",
}

# Pydantic 模型
class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "dall-e-3"
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    quality: str = "standard"  # standard, hd
    style: Optional[str] = None  # vivid, natural
    response_format: str = "url"  # url, b64_json
    user: Optional[str] = None

class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageData]

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail


# 初始化
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时检查 API Key
    api_key = os.getenv("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_API_KEY")
    if not api_key:
        logger.warning("REPLICATE_API_TOKEN not set, API calls will fail")
    else:
        logger.info("Replicate API key configured")
    yield

app = FastAPI(
    title="Replicate to OpenAI Proxy",
    description="将 Replicate 图像生成 API 转换为 OpenAI 兼容格式",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_api_key(authorization: Optional[str] = Header(None)) -> str:
    """从请求头获取 API Key，或使用环境变量"""
    
    # 调试日志
    logger.info(f"Authorization header: {authorization[:30] if authorization else 'None'}...")
    
    # 优先使用环境变量中的 Replicate Token
    env_api_key = os.getenv("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_API_KEY")
    
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        logger.info(f"Token prefix: {token[:10] if token else 'None'}...")
        
        # 如果传入的是有效的 Replicate token (r8_ 开头)，直接使用
        if token.startswith("r8_"):
            logger.info("Using r8_ token from header")
            return token
        # 否则（比如 sk-xxx 格式），使用环境变量中的 Replicate token
        elif env_api_key:
            logger.info("Using env token (header token is not r8_ format)")
            return env_api_key
    
    # 最后使用环境变量
    if env_api_key:
        logger.info("Using env token (no valid header)")
        return env_api_key
        
    logger.error("No API key found!")
    raise HTTPException(
        status_code=401,
        detail={
            "error": {
                "message": "No Replicate API key provided. Set REPLICATE_API_TOKEN environment variable.",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        }
    )


def get_replicate_model(model_name: str) -> str:
    """将 OpenAI 模型名映射到 Replicate 模型"""
    model_lower = model_name.lower()
    
    # 直接匹配
    if model_lower in MODEL_MAPPING:
        return MODEL_MAPPING[model_lower]
    
    # 如果是完整的 Replicate 模型路径，直接使用
    if "/" in model_name:
        return model_name
    
    # 默认使用 Seedream 4.5
    return MODEL_MAPPING["dall-e-3"]


def build_replicate_input(
    model_id: str,
    prompt: str,
    size: str,
    quality: str,
    style: Optional[str],
    n: int
) -> Dict[str, Any]:
    """根据不同模型构建 Replicate 输入参数"""
    
    aspect_ratio = SIZE_TO_ASPECT_RATIO.get(size, "1:1")
    
    # 解析宽高
    try:
        width, height = map(int, size.split("x"))
    except:
        width, height = 1024, 1024
    
    # Seedream 4.5
    if "seedream" in model_id.lower():
        input_params = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "size": "4K" if quality == "hd" else "2K",
        }
        if n > 1:
            input_params["num_outputs"] = min(n, 4)
        return input_params
    
    # Flux 系列
    if "flux" in model_id.lower():
        input_params = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "webp",
            "output_quality": 90 if quality == "hd" else 80,
        }
        if n > 1:
            input_params["num_outputs"] = min(n, 4)
        return input_params
    
    # SDXL
    if "sdxl" in model_id.lower():
        input_params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_outputs": min(n, 4),
            "scheduler": "K_EULER",
            "num_inference_steps": 50 if quality == "hd" else 25,
            "guidance_scale": 7.5,
            "refine": "expert_ensemble_refiner" if quality == "hd" else "no_refiner",
        }
        return input_params
    
    # Stable Diffusion 3
    if "stable-diffusion-3" in model_id.lower() or "sd-3" in model_id.lower():
        input_params = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "webp",
            "output_quality": 90 if quality == "hd" else 80,
        }
        return input_params
    
    # Ideogram
    if "ideogram" in model_id.lower():
        input_params = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio.replace(":", ":"),
            "style_type": style if style in ["realistic", "design", "3d", "anime"] else "auto",
        }
        return input_params
    
    # Recraft
    if "recraft" in model_id.lower():
        input_params = {
            "prompt": prompt,
            "size": f"{width}x{height}",
        }
        return input_params

    # Nano Banana Pro
    if "nano-banana" in model_id.lower():
        input_params = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
        }
        if n > 1:
            input_params["num_outputs"] = min(n, 4)
        return input_params

    # 默认参数
    return {
        "prompt": prompt,
        "width": width,
        "height": height,
    }


async def run_replicate_model(
    api_key: str,
    model_id: str,
    input_params: Dict[str, Any]
) -> List[str]:
    """使用 HTTP API 直接调用 Replicate 模型"""
    
    # 构建 API URL
    # model_id 格式: "bytedance/seedream-4.5"
    api_url = f"https://api.replicate.com/v1/models/{model_id}/predictions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "wait"  # 同步等待结果
    }
    
    payload = {
        "input": input_params
    }
    
    logger.info(f"Calling Replicate API: {api_url}")
    logger.info(f"Payload: {payload}")
    
    try:
        async with httpx.AsyncClient(timeout=300) as client:  # 5分钟超时
            response = await client.post(api_url, headers=headers, json=payload)
            
            logger.info(f"Replicate response status: {response.status_code}")
            
            if response.status_code != 200 and response.status_code != 201:
                error_text = response.text
                logger.error(f"Replicate API error: {error_text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail={
                        "error": {
                            "message": f"Replicate API error: {error_text}",
                            "type": "replicate_error",
                            "code": "api_error"
                        }
                    }
                )
            
            result = response.json()
            logger.info(f"Replicate result status: {result.get('status')}")
            
            # 检查是否成功
            if result.get("status") == "failed":
                error_msg = result.get("error", "Unknown error")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": {
                            "message": f"Image generation failed: {error_msg}",
                            "type": "replicate_error",
                            "code": "generation_failed"
                        }
                    }
                )
            
            # 提取输出 URL
            output = result.get("output", [])
            if isinstance(output, list):
                return output
            elif isinstance(output, str):
                return [output]
            else:
                return []
                
    except httpx.TimeoutException:
        logger.error("Replicate API timeout")
        raise HTTPException(
            status_code=504,
            detail={
                "error": {
                    "message": "Image generation timed out",
                    "type": "timeout_error",
                    "code": "timeout"
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Image generation failed: {str(e)}",
                    "type": "server_error",
                    "code": "model_error"
                }
            }
        )


async def download_image_as_base64(url: str) -> str:
    """下载图片并转换为 base64"""
    import base64
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=60)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")


# API 路由

@app.get("/")
async def root():
    return {
        "message": "Replicate to OpenAI Compatible API Proxy",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/")
async def root_post():
    """兼容某些客户端的健康检查"""
    return {
        "message": "Replicate to OpenAI Compatible API Proxy",
        "version": "1.0.0",
        "status": "ok"
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """列出可用模型"""
    models = []
    for model_name, replicate_id in MODEL_MAPPING.items():
        models.append(ModelInfo(
            id=model_name,
            created=int(time.time()),
            owned_by="replicate"
        ))
    return ModelsResponse(data=models)


@app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """获取模型信息"""
    if model_id.lower() in MODEL_MAPPING:
        return ModelInfo(
            id=model_id,
            created=int(time.time()),
            owned_by="replicate"
        )
    raise HTTPException(status_code=404, detail={"error": {"message": f"Model {model_id} not found"}})


@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image(
    request: ImageGenerationRequest,
    api_key: str = Depends(get_api_key)
):
    """生成图像 - OpenAI 兼容接口"""
    
    logger.info(f"Image generation request: model={request.model}")
    logger.info(f"Full prompt: [{request.prompt}]")
    logger.info(f"Size: {request.size}, Quality: {request.quality}")
    
    # 获取 Replicate 模型
    replicate_model = get_replicate_model(request.model)
    logger.info(f"Using Replicate model: {replicate_model}")
    
    # 构建输入参数
    input_params = build_replicate_input(
        model_id=replicate_model,
        prompt=request.prompt,
        size=request.size,
        quality=request.quality,
        style=request.style,
        n=request.n
    )
    logger.info(f"Input params: {input_params}")
    
    # 运行模型
    image_urls = await run_replicate_model(api_key, replicate_model, input_params)
    
    if not image_urls:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "No images generated",
                    "type": "server_error",
                    "code": "no_output"
                }
            }
        )
    
    # 构建响应
    data = []
    for url in image_urls[:request.n]:
        if request.response_format == "b64_json":
            try:
                b64_data = await download_image_as_base64(url)
                data.append(ImageData(b64_json=b64_data, revised_prompt=request.prompt))
            except Exception as e:
                logger.error(f"Failed to download image: {e}")
                data.append(ImageData(url=url, revised_prompt=request.prompt))
        else:
            data.append(ImageData(url=url, revised_prompt=request.prompt))
    
    return ImageGenerationResponse(
        created=int(time.time()),
        data=data
    )


# 兼容 OpenAI 的其他端点（返回空或错误）

@app.post("/v1/images/edits")
async def create_image_edit():
    """图像编辑 - 暂不支持"""
    raise HTTPException(
        status_code=501,
        detail={
            "error": {
                "message": "Image editing is not yet supported",
                "type": "not_implemented",
                "code": "not_implemented"
            }
        }
    )


@app.post("/v1/images/variations")
async def create_image_variation():
    """图像变体 - 暂不支持"""
    raise HTTPException(
        status_code=501,
        detail={
            "error": {
                "message": "Image variations is not yet supported",
                "type": "not_implemented",
                "code": "not_implemented"
            }
        }
    )


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": int(time.time())}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Dict[str, Any] = Body(default={}),
    api_key: str = Depends(get_api_key)
):
    """
    兼容聊天格式的图像生成
    支持通过 /v1/chat/completions 来生成图片
    """
    import json as json_lib
    
    if request is None:
        request = {}
    
    # 提取参数
    model = request.get("model", "dall-e-3")
    messages = request.get("messages", [])
    
    # 从 messages 中提取 prompt
    prompt = ""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                prompt = content
            elif isinstance(content, list):
                # 处理多模态格式
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        prompt = item.get("text", "")
                        break
    
    if not prompt:
        return {
            "id": "chatcmpl-proxy",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "请提供图像描述"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    
    # 解析尺寸参数
    resolution = request.get("resolution", "2K")
    ratio = request.get("ratio", "1:1")
    size = request.get("size", "1024x1024")
    quality = "hd" if resolution.lower() == "4k" else "standard"
    
    # 映射 ratio 到 size
    ratio_to_size = {
        "1:1": "1024x1024",
        "3:4": "1024x1792",
        "4:3": "1792x1024",
        "9:16": "1024x1792",
        "16:9": "1792x1024",
    }
    if ratio in ratio_to_size:
        size = ratio_to_size[ratio]
    
    logger.info(f"Chat image generation: model={model}, prompt={prompt[:50]}...")
    
    # 获取 Replicate 模型
    replicate_model = get_replicate_model(model)
    
    # 构建输入参数
    input_params = build_replicate_input(
        model_id=replicate_model,
        prompt=prompt,
        size=size,
        quality=quality,
        style=None,
        n=1
    )
    
    # 运行模型
    try:
        image_urls = await run_replicate_model(api_key, replicate_model, input_params)
    except HTTPException as e:
        return {
            "id": "chatcmpl-proxy",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"生成失败: {str(e.detail)}"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    
    if not image_urls:
        return {
            "id": "chatcmpl-proxy",
            "object": "chat.completion", 
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "图像生成失败，请重试"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    
    # 返回图片 URL
    image_url = image_urls[0]
    
    # 构造 markdown 格式的图片响应
    content = f"![Generated Image]({image_url})\n\n[点击查看大图]({image_url})"
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": 20, "total_tokens": len(prompt) + 20}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
