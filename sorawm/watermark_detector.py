from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
from loguru import logger
from ultralytics import YOLO

from sorawm.configs import WATER_MARK_DETECT_YOLO_WEIGHTS
from sorawm.utils.download_utils import download_detector_weights
from sorawm.utils.devices_utils import get_device
from sorawm.utils.video_utils import VideoLoader

# based on the sora tempalte to detect the whole, and then got the icon part area.


class SoraWaterMarkDetector:
    def __init__(
        self,
        min_confidence: float = 0.25,
        upscale_factor: float = 1.0,
        clahe_clip_limit: float | None = None,
        sharpen: bool = False,
        template_path: Path | None = None,
        template_threshold: float = 0.6,
        template_search_expand: int = 32,
    ):
        self.min_confidence = float(min_confidence)
        self.upscale_factor = float(upscale_factor)
        self.clahe_clip_limit = clahe_clip_limit
        self.sharpen = bool(sharpen)
        self.template_threshold = float(template_threshold)
        self.template_search_expand = int(template_search_expand)
        self.template_path: Path | None = Path(template_path) if template_path else None
        self._template_info: Optional[dict[str, np.ndarray | int]] = None
        if self.template_path is not None:
            self._load_template(self.template_path)
        download_detector_weights()
        logger.debug(f"Begin to load yolo water mark detet model.")
        self.model = YOLO(WATER_MARK_DETECT_YOLO_WEIGHTS)
        self.model.to(str(get_device()))
        logger.debug(f"Yolo water mark detet model loaded.")

        self.model.eval()

    def _load_template(self, template_path: Path):
        if not template_path.exists():
            logger.warning(f"Template path not found: {template_path}")
            self._template_info = None
            return
        tmpl = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if tmpl is None:
            logger.warning(f"Failed to load template image: {template_path}")
            self._template_info = None
            return
        if tmpl.ndim == 3 and tmpl.shape[2] == 4:
            bgr = cv2.cvtColor(tmpl, cv2.COLOR_BGRA2BGR)
            mask = tmpl[:, :, 3]
        else:
            bgr = tmpl if tmpl.ndim == 3 else cv2.cvtColor(tmpl, cv2.COLOR_GRAY2BGR)
            mask = None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if mask is not None:
            mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self._template_info = {
            "gray": gray,
            "mask": mask,
            "width": gray.shape[1],
            "height": gray.shape[0],
        }
        logger.info(f"Template loaded for refinement: {template_path} ({gray.shape[1]}x{gray.shape[0]})")

    def update_params(
        self,
        *,
        min_confidence: float | None = None,
        upscale_factor: float | None = None,
        clahe_clip_limit: float | None = None,
        sharpen: bool | None = None,
        template_path: Path | None = None,
        clear_template: bool = False,
        template_threshold: float | None = None,
        template_search_expand: int | None = None,
    ):
        if min_confidence is not None:
            self.min_confidence = float(min_confidence)
        if upscale_factor is not None and upscale_factor > 0:
            self.upscale_factor = float(upscale_factor)
        if clahe_clip_limit is not None:
            self.clahe_clip_limit = float(clahe_clip_limit) if clahe_clip_limit > 0 else None
        if sharpen is not None:
            self.sharpen = bool(sharpen)
        if template_threshold is not None:
            self.template_threshold = float(template_threshold)
        if template_search_expand is not None:
            self.template_search_expand = int(template_search_expand)
        if clear_template:
            self.template_path = None
            self._template_info = None
        if template_path is not None:
            self.template_path = Path(template_path)
            self._load_template(self.template_path)

    @staticmethod
    def _expand_rect(rect: Tuple[float, float, float, float], expand: int, width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = rect
        x1 = max(0, int(np.floor(x1 - expand)))
        y1 = max(0, int(np.floor(y1 - expand)))
        x2 = min(width, int(np.ceil(x2 + expand)))
        y2 = min(height, int(np.ceil(y2 + expand)))
        return x1, y1, x2, y2

    @staticmethod
    def _compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _match_template(
        self,
        image_gray: np.ndarray,
        search_rect: Tuple[int, int, int, int] | None = None,
    ) -> Tuple[Tuple[int, int, int, int], float] | None:
        if self._template_info is None:
            return None
        tmpl_gray = self._template_info["gray"]
        tmpl_mask = self._template_info["mask"]
        th, tw = tmpl_gray.shape
        if search_rect is not None:
            x1, y1, x2, y2 = search_rect
            roi = image_gray[y1:y2, x1:x2]
            if roi.shape[0] < th or roi.shape[1] < tw:
                return None
            res = cv2.matchTemplate(roi, tmpl_gray, cv2.TM_CCOEFF_NORMED, mask=tmpl_mask)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val < self.template_threshold:
                return None
            top_left = (max_loc[0] + x1, max_loc[1] + y1)
        else:
            if image_gray.shape[0] < th or image_gray.shape[1] < tw:
                return None
            res = cv2.matchTemplate(image_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED, mask=tmpl_mask)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val < self.template_threshold:
                return None
            top_left = max_loc
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        return (top_left[0], top_left[1], bottom_right[0], bottom_right[1]), float(max_val)

    def detect(self, input_image: np.array):
        # Optional preprocessing to help detect small/low-contrast watermarks
        image = input_image
        scale = 1.0
        if self.upscale_factor and self.upscale_factor > 1.0:
            scale = float(self.upscale_factor)
            h, w = image.shape[:2]
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        if self.clahe_clip_limit and self.clahe_clip_limit > 0:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=float(self.clahe_clip_limit), tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if self.sharpen:
            # Unsharp mask simple
            blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
            image = cv2.addWeighted(image, 1.5, blur, -0.5, 0)

        image_gray = None
        if self._template_info is not None:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Run YOLO inference (usar el mismo umbral de confianza configurado)
        yolo_conf = float(self.min_confidence)
        if yolo_conf <= 0:
            yolo_conf = 0.001
        results = self.model(image, conf=min(yolo_conf, 0.99), verbose=False)
        # Extract predictions from the first (and only) result
        result = results[0]

        # Check if any detections were made
        if len(result.boxes) == 0:
            return {"detected": False, "bbox": None, "confidence": None, "center": None}

        # Choose the highest-confidence box that meets the threshold
        best_box = None
        best_conf = -1.0
        for b in result.boxes:
            conf = float(b.conf[0].cpu().numpy())
            if conf >= self.min_confidence and conf > best_conf:
                best_box = b
                best_conf = conf

        template_box_scaled: Optional[Tuple[int, int, int, int]] = None
        template_score: Optional[float] = None
        if image_gray is not None:
            search_rect = None
            if best_box is not None:
                xyxy_scaled = best_box.xyxy[0].cpu().numpy()
                expand_pixels = int(self.template_search_expand * (scale if scale != 0 else 1))
                search_rect = self._expand_rect(
                    (xyxy_scaled[0], xyxy_scaled[1], xyxy_scaled[2], xyxy_scaled[3]),
                    expand_pixels,
                    image_gray.shape[1],
                    image_gray.shape[0],
                )
            try:
                template_match = self._match_template(image_gray, search_rect)
            except cv2.error as e:
                logger.warning(f"Template matching error: {e}")
                template_match = None
            if template_match is None and search_rect is not None:
                # fallback to global search if local failed
                try:
                    template_match = self._match_template(image_gray, None)
                except cv2.error as e:
                    logger.warning(f"Template matching global error: {e}")
                    template_match = None
            if template_match is not None:
                template_box_scaled, template_score = template_match

        final_bbox: Optional[Tuple[float, float, float, float]] = None
        final_conf: Optional[float] = None
        final_source = "yolo"

        if best_box is not None:
            xyxy_scaled = best_box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy_scaled[0]), float(xyxy_scaled[1]), float(xyxy_scaled[2]), float(xyxy_scaled[3])
            if scale != 1.0:
                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale
            final_bbox = (x1, y1, x2, y2)
            final_conf = best_conf

        if template_box_scaled is not None and template_score is not None:
            tx1, ty1, tx2, ty2 = template_box_scaled
            if scale != 1.0:
                tx1 /= scale
                ty1 /= scale
                tx2 /= scale
                ty2 /= scale
            template_bbox = (tx1, ty1, tx2, ty2)
            if final_bbox is None:
                final_bbox = template_bbox
                final_conf = template_score
                final_source = "template"
            else:
                iou = self._compute_iou(
                    (int(final_bbox[0]), int(final_bbox[1]), int(final_bbox[2]), int(final_bbox[3])),
                    (int(template_bbox[0]), int(template_bbox[1]), int(template_bbox[2]), int(template_bbox[3])),
                )
                if iou > 0.1 or template_score >= self.template_threshold + 0.05:
                    # refine bbox with template position
                    final_bbox = template_bbox
                    final_conf = max(final_conf if final_conf is not None else 0.0, template_score)
                    final_source = "template_refined"

        if final_bbox is None:
            return {"detected": False, "bbox": None, "confidence": None, "center": None}

        x1, y1, x2, y2 = final_bbox
        confidence = final_conf if final_conf is not None else 0.0
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return {
            "detected": True,
            "bbox": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            "confidence": confidence,
            "center": (int(round(center_x)), int(round(center_y))),
            "source": final_source,
        }


if __name__ == "__main__":
    from pathlib import Path

    import cv2
    from tqdm import tqdm

    # ========= 配置 =========
    # video_path = Path("resources/puppies.mp4") # 19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4
    video_path = Path("resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4")
    save_video = True
    out_path = Path("outputs/sora_watermark_yolo_detected.mp4")
    window = "Sora Watermark YOLO Detection"
    # =======================

    # 初始化检测器
    detector = SoraWaterMarkDetector()

    # 初始化视频加载器
    video_loader = VideoLoader(video_path)

    # 预取一帧确定尺寸/FPS
    first_frame = None
    for first_frame in video_loader:
        break
    assert first_frame is not None, "无法读取视频帧"

    H, W = first_frame.shape[:2]
    fps = getattr(video_loader, "fps", 30)

    # 输出视频设置
    writer = None
    if save_video:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        assert writer.isOpened(), "无法创建输出视频文件"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def visualize_detection(frame, detection_result, frame_idx):
        """在帧上可视化检测结果"""
        vis = frame.copy()

        if detection_result["detected"]:
            # 绘制边界框
            x1, y1, x2, y2 = detection_result["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制中心点
            cx, cy = detection_result["center"]
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

            # 显示置信度
            conf = detection_result["confidence"]
            label = f"Watermark: {conf:.2f}"

            # 文本背景
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                vis, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), (0, 255, 0), -1
            )

            # 绘制文本
            cv2.putText(
                vis,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            status = f"Frame {frame_idx} | DETECTED | Conf: {conf:.3f}"
            status_color = (0, 255, 0)
        else:
            status = f"Frame {frame_idx} | NO WATERMARK"
            status_color = (0, 0, 255)

        # 显示帧信息
        cv2.putText(
            vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
        )

        return vis

    # 处理第一帧
    print("开始处理视频...")
    detection = detector.detect(first_frame)
    vis_frame = visualize_detection(first_frame, detection, 0)
    cv2.imshow(window, vis_frame)
    if writer is not None:
        writer.write(vis_frame)

    # 处理剩余帧
    total_frames = 0
    detected_frames = 0

    for idx, frame in enumerate(
        tqdm(video_loader, desc="Processing frames", initial=1, unit="f"), start=1
    ):
        # YOLO 检测
        detection = detector.detect(frame)

        # 可视化
        vis_frame = visualize_detection(frame, detection, idx)

        # 统计
        total_frames += 1
        if detection["detected"]:
            detected_frames += 1

        # 显示
        cv2.imshow(window, vis_frame)

        # 保存
        if writer is not None:
            writer.write(vis_frame)

        # 按键控制
        key = cv2.waitKey(max(1, int(1000 / max(1, int(fps))))) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):  # 空格暂停
            while True:
                k = cv2.waitKey(50) & 0xFF
                if k in (ord(" "), ord("q")):
                    if k == ord("q"):
                        idx = 10**9
                    break
            if idx >= 10**9:
                break

    # 清理
    if writer is not None:
        writer.release()
        print(f"\n[完成] 可视化视频已保存: {out_path}")

    # 打印统计信息
    total_frames += 1  # 包括第一帧
    if detection["detected"]:
        detected_frames += 1

    print(f"\n=== 检测统计 ===")
    print(f"总帧数: {total_frames}")
    print(f"检测到水印: {detected_frames} 帧")
    print(f"检测率: {detected_frames/total_frames*100:.2f}%")

    cv2.destroyAllWindows()
