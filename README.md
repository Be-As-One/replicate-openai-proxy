# Replicate to OpenAI Compatible API Proxy

将 Replicate 的图像生成 API 转换为 OpenAI 兼容格式，让你可以用 OpenAI SDK 调用 Replicate 上的图像模型。

## 支持的模型

| OpenAI 格式名称 | Replicate 模型 |
|----------------|----------------|
| `dall-e-3` / `seedream-4.5` | bytedance/seedream-4.5 |
| `dall-e-2` / `sdxl` | stability-ai/sdxl |
| `flux-dev` | black-forest-labs/flux-dev |
| `flux-schnell` | black-forest-labs/flux-schnell |
| `flux-pro` | black-forest-labs/flux-pro |
| `flux-1.1-pro` | black-forest-labs/flux-1.1-pro |
| `sd-3` / `stable-diffusion-3` | stability-ai/stable-diffusion-3 |
| `ideogram-v2` | ideogram-ai/ideogram-v2 |
| `recraft-v3` | recraft-ai/recraft-v3 |

你也可以直接使用完整的 Replicate 模型路径，如 `bytedance/seedream-4.5`。

## 快速开始

### 1. 使用 Docker Compose（推荐）

```bash
# 克隆项目
git clone <repo-url>
cd replicate-openai-proxy

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 Replicate API Token

# 启动服务
docker-compose up -d
```

### 2. 使用 Docker

```bash
docker build -t replicate-openai-proxy .

docker run -d \
  --name replicate-openai-proxy \
  -p 8000:8000 \
  -e REPLICATE_API_TOKEN="r8_your_token_here" \
  replicate-openai-proxy
```

### 3. 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export REPLICATE_API_TOKEN="r8_your_token_here"

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API 使用

### 生成图像

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer r8_your_token_here" \
  -d '{
    "model": "dall-e-3",
    "prompt": "A cute cat sitting on a windowsill, sunlight streaming in",
    "size": "1024x1024",
    "quality": "standard",
    "n": 1
  }'
```

### 使用 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="r8_your_replicate_token",  # 或任意字符串（如服务端已配置）
    base_url="http://localhost:8000/v1"
)

response = client.images.generate(
    model="dall-e-3",  # 映射到 Seedream 4.5
    prompt="A futuristic city with flying cars",
    size="1792x1024",
    quality="hd",
    n=1
)

print(response.data[0].url)
```

### 使用 OpenAI Node.js SDK

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: 'r8_your_replicate_token',
  baseURL: 'http://localhost:8000/v1'
});

const response = await openai.images.generate({
  model: 'flux-dev',
  prompt: 'An astronaut riding a horse on Mars',
  size: '1024x1024',
  n: 1
});

console.log(response.data[0].url);
```

### 直接使用 Replicate 模型路径

```python
response = client.images.generate(
    model="bytedance/seedream-4.5",  # 直接使用 Replicate 模型路径
    prompt="Your prompt here",
    size="1024x1024"
)
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/images/generations` | POST | 生成图像 |
| `/v1/models` | GET | 列出可用模型 |
| `/v1/models/{model_id}` | GET | 获取模型信息 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | API 文档（Swagger UI） |

## 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt` | string | ✅ | 图像描述 |
| `model` | string | ❌ | 模型名称，默认 `dall-e-3` |
| `size` | string | ❌ | 图像尺寸，默认 `1024x1024` |
| `quality` | string | ❌ | 质量 `standard` 或 `hd` |
| `n` | integer | ❌ | 生成数量，1-10 |
| `response_format` | string | ❌ | `url` 或 `b64_json` |

### 支持的尺寸

- `1024x1024` (1:1)
- `1792x1024` (16:9)
- `1024x1792` (9:16)
- `512x512` (1:1)
- `1280x720` (16:9)
- `720x1280` (9:16)

## 响应格式

```json
{
  "created": 1699000000,
  "data": [
    {
      "url": "https://replicate.delivery/...",
      "revised_prompt": "A cute cat sitting on a windowsill..."
    }
  ]
}
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `REPLICATE_API_TOKEN` | Replicate API Token（必需） |

## 获取 Replicate API Token

1. 访问 [Replicate](https://replicate.com)
2. 登录或注册账号
3. 前往 [API Tokens](https://replicate.com/account/api-tokens)
4. 创建新的 Token

## 注意事项

- 不同模型的计费不同，请参考 [Replicate Pricing](https://replicate.com/pricing)
- 部分模型可能需要较长时间生成（特别是高质量模式）
- 建议在生产环境使用负载均衡和速率限制

## License

MIT
