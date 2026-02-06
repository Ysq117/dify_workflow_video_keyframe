import os
import tempfile
import cv2
import numpy as np
import json
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
import shutil
import uuid
import threading
import time
from pathlib import Path
import logging
import subprocess

import uvicorn

logger = logging.getLogger("keyframe")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Video Keyframe Extractor",
    servers=[{"url": "http://keyframe-service:8117"}]
)

DIFY_STORAGE_DIR = "/app/api/storage"
TOOLS_DIR = os.path.join(DIFY_STORAGE_DIR, "tools")
os.makedirs(TOOLS_DIR, exist_ok=True)


class KeyframeExtractionRequest(BaseModel):
    video_url: str



# === 后台清理任务 ===
def cleanup_old_dirs():
    tools_path = Path(TOOLS_DIR)
    while True:
        try:
            if tools_path.exists():
                now = time.time()
                deleted_count = 0
                for item in tools_path.iterdir():
                    if item.is_dir():
                        try:
                            uuid.UUID(str(item.name))
                            if now - item.stat().st_mtime > 3600:
                                shutil.rmtree(item)
                                deleted_count += 1
                        except ValueError:
                            continue
                print(f"[清理] 已删除 {deleted_count} 个过期目录")
        except Exception as e:
            print(f"[清理错误] {e}")
        time.sleep(3600)


threading.Thread(target=cleanup_old_dirs, daemon=True, name="CleanupThread").start()


# === 下载视频 ===
def download_video(url: str, temp_dir: str) -> str:
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        path = os.path.join(temp_dir, "input.mp4")
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"下载失败: {e}")


# === 新增：视频修复函数 ===
def repair_video_if_needed(video_path: str) -> str:
    """
    检查视频是否能被 OpenCV 正常打开。
    如果不能，则尝试修复：
      - 如果是 AV1 或其他不兼容编码 → 转 H.264
      - 如果是 H.264 但 moov 在尾部 → 移动 moov 到头部
    返回可被 OpenCV 读取的视频路径（可能是原文件或修复副本）
    """
    # 先尝试直接打开
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        ret, _ = cap.read()
        cap.release()
        if ret:
            logger.info("视频可直接读取，无需修复")
            return video_path

    cap.release()
    logger.warning("视频无法直接读取，尝试修复...")

    # 创建修复后路径
    repaired_path = video_path.replace(".mp4", "_repaired.mp4")

    try:
        # 尝试无损修复（适用于 moov 在尾部的 H.264）
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-c", "copy",
            "-movflags", "+faststart",
            repaired_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # 验证修复后是否可用
            cap = cv2.VideoCapture(repaired_path)
            if cap.isOpened() and cap.read()[0]:
                cap.release()
                logger.info("无损修复成功")
                return repaired_path
            cap.release()

        # 如果无损修复失败，强制转 H.264
        logger.info("无损修复失败，执行 H.264 转码...")
        # 在 H.264 转码命令中添加缩放滤镜
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", "scale=320:-2",
            "-c:v", "libx264",
            "-crf", "28",
            "-preset", "ultrafast",
            "-an",
            "-movflags", "+faststart",
            repaired_path
        ]
        subprocess.run(cmd, check=True, timeout=180)
        
        # 再次验证
        cap = cv2.VideoCapture(repaired_path)
        usable = cap.isOpened() and cap.read()[0]
        cap.release()
        
        if usable:
            logger.info("H.264 转码修复成功")
            return repaired_path
        else:
            raise RuntimeError("修复后仍无法读取视频")

    except Exception as e:
        logger.error(f"视频修复失败: {e}")
        # 清理临时文件
        if os.path.exists(repaired_path):
            os.remove(repaired_path)
        raise HTTPException(status_code=400, detail=f"视频格式不支持或已损坏: {str(e)}")


# === 关键帧检测 ===
def detect_shot_boundaries_and_extract_keyframes(
    video_path: str,
    threshold: float = 30.0,
    skip_frames: int = 10,
    width: int = 320
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    shot_boundaries = []
    prev_frame = None
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % skip_frames != 0:
            frame_index += 1
            continue

        h, w = frame.shape[:2]
        if w > width:
            ratio = width / w
            new_h = int(h * ratio)
            frame = cv2.resize(frame, (width, new_h))

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = np.mean(np.abs(curr_gray.astype(float) - prev_frame.astype(float)))
            if diff > threshold:
                shot_boundaries.append(frame_index)

        prev_frame = curr_gray
        frame_index += 1

    cap.release()

    if not shot_boundaries or shot_boundaries[-1] != total_frames:
        shot_boundaries.append(total_frames)

    keyframe_indices = []
    start = 0
    for end in shot_boundaries:
        center = start + (end - start) // 2
        if center < total_frames:
            keyframe_indices.append(center)
        start = end

    return keyframe_indices, total_frames


def extract_frames_by_indices(video_path, frame_indices, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    extracted_paths = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            filename = f"extracted_frame_{idx:06d}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            extracted_paths.append(save_path)
        else:
            print(f"警告：无法读取帧 {idx}")

    cap.release()
    return extracted_paths


# === 上传文件到 Dify（保留你的接口地址）===
def upload_file_to_dify(file_path: str, filename: str, auth_header: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "image/jpeg")}
            data = {"purpose": "tool-file"}
            headers = {"Authorization": auth_header}

            resp = requests.post(
                "http://192.168.2.112:58080/v1/files/upload",
                files=files,
                data=data,
                headers=headers,
                timeout=30
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"上传失败 {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


def format_dify_file(raw_obj: dict) -> dict:
    return {
        "dify_model_identity": "__dify__file__",
        "id": None,
        "tenant_id": raw_obj.get("created_by", "default-tenant"),
        "type": "image",
        "transfer_method": "tool_file",
        "remote_url": None,
        "related_id": raw_obj["id"],
        "filename": raw_obj["name"],
        "extension": f".{raw_obj['extension']}",
        "mime_type": raw_obj["mime_type"],
        "size": raw_obj["size"],
        "url": raw_obj["source_url"]
    }


# === 新增：创建 3x3 网格图（支持 1~9 帧）===
def create_3x3_grid_image(frame_paths: List[str], output_path: str):
    """
    将 1~9 张图片拼接成 3x3 网格图。
    不足 9 张时，其余位置留黑。
    """
    if not frame_paths:
        raise ValueError("No frames provided")

    # 读取第一张图获取尺寸
    sample = cv2.imread(frame_paths[0])
    if sample is None:
        raise ValueError("Failed to read first frame")
    
    h, w = sample.shape[:2]
    grid_h = 3 * h
    grid_w = 3 * w

    # 创建黑色背景画布
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, path in enumerate(frame_paths[:9]):  # 最多处理前9张
        img = cv2.imread(path)
        if img is None:
            continue
        img_resized = cv2.resize(img, (w, h))
        row = idx // 3
        col = idx % 3
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img_resized

    cv2.imwrite(output_path, grid, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


# === 新增：分组函数 ===
def group_frames_into_batches(frame_paths: List[str], batch_size: int = 9) -> List[List[str]]:
    """将帧路径列表按 batch_size 分组"""
    return [frame_paths[i:i + batch_size] for i in range(0, len(frame_paths), batch_size)]


@app.post("/extract_keyframes")
async def extract_keyframes_api(request: KeyframeExtractionRequest, raw_request: Request):
    auth_header = raw_request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=403, detail="Missing Authorization header from Dify")

    temp_dir = tempfile.mkdtemp()
    request_id = str(uuid.uuid4())
    output_subdir = os.path.join(TOOLS_DIR, request_id)
    os.makedirs(output_subdir, exist_ok=True)

    try:
        threshold = 30.0
        skip_frames = 10
        width = 320

        video_path = download_video(request.video_url, temp_dir)
        original_filename = os.path.basename(video_path)

        # 修复视频
        safe_video_path = repair_video_if_needed(video_path)

        keyframe_indices, _ = detect_shot_boundaries_and_extract_keyframes(
            safe_video_path,
            threshold=threshold,
            skip_frames=skip_frames,
            width=width
        )

        if not keyframe_indices:
            raise HTTPException(status_code=400, detail="未检测到有效关键帧")

        # 提取所有关键帧图像文件路径
        extracted_paths = extract_frames_by_indices(safe_video_path, keyframe_indices, output_subdir)

        if not extracted_paths:
            raise HTTPException(status_code=400, detail="未提取到任何帧")

        # === 核心修改：分组拼图 ===
        batches = group_frames_into_batches(extracted_paths, batch_size=9)
        grid_paths = []

        for i, batch in enumerate(batches):
            grid_path = os.path.join(output_subdir, f"video_summary_batch_{i:03d}.jpg")
            create_3x3_grid_image(batch, grid_path)
            grid_paths.append(grid_path)

        # 上传所有网格图
        dify_files = []
        base_name = os.path.splitext(original_filename)[0]
        for i, path in enumerate(grid_paths):
            friendly_name = f"{base_name}_summary_batch_{i:03d}.jpg"
            raw_obj = upload_file_to_dify(path, friendly_name, auth_header)
            standardized_file = format_dify_file(raw_obj)
            dify_files.append(standardized_file)

        result = {"files": dify_files}

        return Response(
            content=json.dumps(result, ensure_ascii=False),
            media_type="application/json"
        )

    except HTTPException:
        shutil.rmtree(output_subdir, ignore_errors=True)
        raise
    except Exception as e:
        logger.exception("关键帧提取发生内部错误")
        shutil.rmtree(output_subdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8117)