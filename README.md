# 🎞️ Video Keyframe Extractor for Dify

一个专为 [Dify](https://github.com/langgenius/dify) 设计的自定义工具后端服务，用于：  
- 从视频 URL 自动提取关键帧  
- 修复常见编码问题（AV1 / moov 位置异常）  
- 将关键帧按 **3×3 网格图** 拼接  
- 上传至 Dify 文件系统，并返回标准 `files` 格式供工作流使用  

> ✅ 自动清理临时文件｜✅ 与 Dify 工具链无缝集成

---

## 📦 部署方式

### 1. 构建镜像
确保目录结构如下：
```
project/
├── Dockerfile
├── app/
│   ├── app.py
│   └── start.sh
├── pip_depends/           # 离线 .whl 依赖包
├── deploy.sh/
└── docker-compose.yml
```

构建命令：
```bash
docker build -t keyframe .
```

### 2. 启动服务
```bash
sh depoly.sh
```

> ⚠️ 注意：  
> - 容器需挂载 Dify 的存储目录：`/home/Downloads/dify-main/docker/volumes/app/storage:/app/api/storage`  
> - 依赖 `docker_default` 外部网络（由 Dify 创建）  
> - 服务监听 `:8117`，对外暴露为 `http://<host>:8117`

---

## 🌐 API 接口

### POST `/extract_keyframes`

**请求头**
```http
Authorization: Bearer <your-dify-api-token>
Content-Type: application/json
```

**请求体**
```json
{
  "video_url": "https://example.com/video.mp4"
}
```

**成功响应示例**
```json
{
  "files": [
    {
      "dify_model_identity": "dify__file",
      "type": "image",
      "filename": "video_summary_batch_000.jpg",
      "url": "http://api:5000/files/xxx.jpg",
      "size": 123456,
      "mime_type": "image/jpeg",
      "extension": ".jpg",
      "related_id": "xxx",
      "transfer_method": "tool_file"
    }
  ]
}
```

> ✅ 此格式可被 Dify 工作流中的 **“文件列表格式转换” CodeNode** 直接消费。

---

## 🔧 Dify 工作流配置建议

### 1. 自定义工具节点
- **类型**：HTTP Tool  
- **URL**：`http://keyframe-service:8117/extract_keyframes`  
- **Method**：POST  
- **Headers**：`Authorization: Bearer {{api_token}}`  
- **Body**：
  ```json
  {"video_url": "{{video_url}}"}
  ```

### 2. 后续节点：文件列表格式转换（CodeNode）

> **作用**：将 API 返回的 `{"files": [...]}`（可能双重转义）标准化为 Dify 可识别的 `files` 字段。

### 3. 最终输出
→ 可接入 LLM 分析、RAG 检索等下游节点。

迭代节点中使用ffmpeg_tools_dify的get_video_frame确保输出关键帧为Arry[files]类型
<img width="1095" height="227" alt="image" src="https://github.com/user-attachments/assets/358177c3-876e-42d4-b9f7-4d3b294d8f7a" />

---

## 🛠️ 注意事项

| 项目 | 说明 |
|------|------|
| **视频兼容性** | 自动尝试无损修复（`faststart`）或 H.264 转码（缩放至 320px 宽） |
| **临时文件清理** | `/app/api/storage/tools/` 下超过 1 小时的 UUID 目录自动删除 |
| **Dify 地址硬编码** | 当前上传地址为 `http://192.168.1.100:8080`，请根据实际环境修改 `app.py` 中的 URL |

