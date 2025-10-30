from pathlib import Path
from typing import Callable

import ffmpeg
import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from sorawm.utils.video_utils import VideoLoader
from sorawm.watermark_cleaner import WaterMarkCleaner
from sorawm.watermark_detector import SoraWaterMarkDetector
from sorawm.utils.imputation_utils import (
    find_2d_data_bkps,
    get_interval_average_bbox,
    find_idxs_interval,
)


class SoraWM:
    def __init__(
        self,
        min_confidence: float = 0.25,
        upscale_factor: float = 1.0,
        clahe_clip_limit: float | None = None,
        sharpen: bool = False,
    ):
        self.detector = SoraWaterMarkDetector(
            min_confidence=min_confidence,
            upscale_factor=upscale_factor,
            clahe_clip_limit=clahe_clip_limit,
            sharpen=sharpen,
        )
        self.cleaner = None  # lazy init to evitar cargar modelo en solo detección

    def update_detector_params(
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
        self.detector.update_params(
            min_confidence=min_confidence,
            upscale_factor=upscale_factor,
            clahe_clip_limit=clahe_clip_limit,
            sharpen=sharpen,
            template_path=template_path,
            clear_template=clear_template,
            template_threshold=template_threshold,
            template_search_expand=template_search_expand,
        )

    @staticmethod
    def _dilate_bbox(bbox: tuple[int, int, int, int], width: int, height: int, dilation: int) -> tuple[int, int, int, int]:
        if dilation == 0:
            return bbox
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - dilation)
        y1 = max(0, y1 - dilation)
        x2 = min(width, x2 + dilation)
        y2 = min(height, y2 + dilation)
        return (x1, y1, x2, y2)

    @staticmethod
    def _fill_missed_bboxes_interval(
        frame_bboxes: dict[int, dict[str, object]],
        detect_missed: list[int],
        bbox_centers: list[tuple[int, int] | None],
        bboxes: list[tuple[int, int, int, int] | None],
        total_frames: int,
    ):
        if not detect_missed:
            return

        bkps = find_2d_data_bkps(bbox_centers)
        bkps_full = [0] + bkps + [total_frames]
        interval_bboxes = get_interval_average_bbox(bboxes, bkps_full)
        missed_intervals = find_idxs_interval(detect_missed, bkps_full)

        for missed_idx, interval_idx in zip(detect_missed, missed_intervals):
            candidate = None
            conf_candidate = None
            if interval_idx < len(interval_bboxes):
                candidate = interval_bboxes[interval_idx]
            if candidate is None:
                before = max(missed_idx - 1, 0)
                after = min(missed_idx + 1, total_frames - 1)
                before_entry = frame_bboxes.get(before, {})
                after_entry = frame_bboxes.get(after, {})
                before_box = before_entry.get("bbox")
                after_box = after_entry.get("bbox")
                candidate = before_box or after_box
                conf_candidate = before_entry.get("confidence") or after_entry.get("confidence")
            frame_entry = frame_bboxes[missed_idx]
            frame_entry["bbox"] = candidate
            if conf_candidate is not None and frame_entry.get("confidence") is None:
                frame_entry["confidence"] = conf_candidate
            if frame_entry.get("source") is None:
                frame_entry["source"] = "interval_fill" if candidate is not None else "missed"

    def run(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Callable[[int], None] | None = None,
        mask_dilation: int = 4,
    ):
        input_video_loader = VideoLoader(input_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        width = input_video_loader.width
        height = input_video_loader.height
        fps = input_video_loader.fps
        total_frames = input_video_loader.total_frames

        temp_output_path = output_video_path.parent / f"temp_{output_video_path.name}"
        output_options = {
            "pix_fmt": "yuv420p",
            "vcodec": "libx264",
            "preset": "slow",
        }

        if input_video_loader.original_bitrate:
            output_options["video_bitrate"] = str(
                int(int(input_video_loader.original_bitrate) * 1.2)
            )
        else:
            output_options["crf"] = "18"

        process_out = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{width}x{height}",
                r=fps,
            )
            .output(str(temp_output_path), **output_options)
            .overwrite_output()
            .global_args("-loglevel", "error")
            .run_async(pipe_stdin=True)
        )

        frame_bboxes: dict[int, dict[str, object]] = {}
        detect_missed: list[int] = []
        bbox_centers: list[tuple[int, int] | None] = []
        bboxes: list[tuple[int, int, int, int] | None] = []

        logger.debug(
            f"total frames: {total_frames}, fps: {fps}, width: {width}, height: {height}"
        )
        for idx, frame in enumerate(
            tqdm(input_video_loader, total=total_frames, desc="Detect watermarks")
        ):
            detection_result = self.detector.detect(frame)
            if detection_result["detected"]:
                bbox = detection_result["bbox"]
                frame_bboxes[idx] = {
                    "bbox": bbox,
                    "confidence": detection_result.get("confidence"),
                    "source": detection_result.get("source"),
                }
                x1, y1, x2, y2 = bbox
                bbox_centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                bboxes.append((x1, y1, x2, y2))
            else:
                frame_bboxes[idx] = {"bbox": None, "confidence": None, "source": None}
                detect_missed.append(idx)
                bbox_centers.append(None)
                bboxes.append(None)
            # 10% - 50%
            if progress_callback and idx % 10 == 0:
                progress = 10 + int((idx / total_frames) * 40)
                progress_callback(progress)

        for i in range(total_frames):
            frame_bboxes.setdefault(i, {"bbox": None, "confidence": None, "source": None})

        logger.debug(f"Detect missed frames: {detect_missed}")
        self._fill_missed_bboxes_interval(frame_bboxes, detect_missed, bbox_centers, bboxes, total_frames)
        
        # Lazy init del limpiador solo cuando se va a usar
        if self.cleaner is None:
            self.cleaner = WaterMarkCleaner()

        input_video_loader = VideoLoader(input_video_path)

        for idx, frame in enumerate(tqdm(input_video_loader, total=total_frames, desc="Remove watermarks")):
        # for idx in tqdm(range(total_frames), desc="Remove watermarks"):
            # frame_info = 
            frame_info = frame_bboxes.get(idx, {})
            bbox = frame_info.get("bbox")
            if bbox is not None:
                x1, y1, x2, y2 = self._dilate_bbox(bbox, width, height, mask_dilation)
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                cleaned_frame = self.cleaner.clean(frame, mask)
            else:
                cleaned_frame = frame
            process_out.stdin.write(cleaned_frame.tobytes())

            # 50% - 95%
            if progress_callback and idx % 10 == 0:
                progress = 50 + int((idx / total_frames) * 45)
                progress_callback(progress)

        process_out.stdin.close()
        process_out.wait()

        # 95% - 99%
        if progress_callback:
            progress_callback(95)

        self.merge_audio_track(input_video_path, temp_output_path, output_video_path)

        if progress_callback:
            progress_callback(99)

    def detect_only(
        self,
        input_video_path: Path,
        output_masks_dir: Path,
        preview_video_path: Path | None = None,
        progress_callback: Callable[[int], None] | None = None,
        overlay_alpha: float = 0.35,
        mask_dilation: int = 4,
    ) -> Path | None:
        """
        Recorre el vídeo y genera máscaras binarias por fotograma basadas en el bbox detectado.
        Opcionalmente guarda un vídeo de previsualización con overlay de máscara y bbox.
        """
        input_video_loader = VideoLoader(input_video_path)
        output_masks_dir.mkdir(parents=True, exist_ok=True)

        width = input_video_loader.width
        height = input_video_loader.height
        fps = input_video_loader.fps
        total_frames = input_video_loader.total_frames

        writer = None
        used_preview_path = None
        selected_codec = None
        if preview_video_path is not None:
            preview_video_path.parent.mkdir(parents=True, exist_ok=True)
            # Priorizar códec compatible con MP4 en OpenCV
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(preview_video_path), fourcc, fps, (width, height))
            selected_codec = "mp4v"
            if not writer.isOpened():
                # Intentar H.264 si está disponible
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(str(preview_video_path), fourcc, fps, (width, height))
                selected_codec = "avc1"
            if not writer.isOpened():
                # Fallback a AVI con MJPG
                avi_path = preview_video_path.with_suffix(".avi")
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(str(avi_path), fourcc, fps, (width, height))
                selected_codec = "MJPG"
                if writer.isOpened():
                    used_preview_path = avi_path
            if writer.isOpened() and used_preview_path is None:
                used_preview_path = preview_video_path
            assert writer is not None and writer.isOpened(), "No se pudo crear el vídeo de previsualización"
            logger.info(
                f"Preview VideoWriter abierto: path={used_preview_path}, codec={selected_codec}, fps={fps}, size=({width}x{height})"
            )

        # PASADA 1: Detección + acumulación para rellenar misses
        frame_bboxes: dict[int, dict[str, object]] = {}
        detect_missed: list[int] = []
        bbox_centers: list[tuple[int, int] | None] = []
        bboxes: list[tuple[int, int, int, int] | None] = []

        detected_frames = 0
        for idx, frame in enumerate(
            tqdm(input_video_loader, total=total_frames, desc="Detect only - pass 1")
        ):
            detection_result = self.detector.detect(frame)
            if detection_result["detected"]:
                bbox = detection_result["bbox"]
                frame_bboxes[idx] = {
                    "bbox": bbox,
                    "confidence": detection_result.get("confidence"),
                    "source": detection_result.get("source"),
                }
                x1, y1, x2, y2 = bbox
                bbox_centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                bboxes.append((x1, y1, x2, y2))
                detected_frames += 1
            else:
                frame_bboxes[idx] = {"bbox": None, "confidence": None, "source": None}
                detect_missed.append(idx)
                bbox_centers.append(None)
                bboxes.append(None)

            if progress_callback and idx % 10 == 0:
                progress = int((idx / max(1, total_frames)) * 60)
                progress_callback(progress)

        logger.debug(f"detect missed frames (preview): {detect_missed}")
        for i in range(total_frames):
            frame_bboxes.setdefault(i, {"bbox": None, "confidence": None, "source": None})
        self._fill_missed_bboxes_interval(frame_bboxes, detect_missed, bbox_centers, bboxes, total_frames)

        # PASADA 2: Generación de máscaras y vídeo de previsualización
        input_video_loader = VideoLoader(input_video_path)
        written_frames = 0
        for idx, frame in enumerate(
            tqdm(input_video_loader, total=total_frames, desc="Detect only - pass 2")
        ):
            frame_info = frame_bboxes.get(idx, {})
            bbox = frame_info.get("bbox")
            mask = np.zeros((height, width), dtype=np.uint8)
            if bbox is not None:
                x1, y1, x2, y2 = self._dilate_bbox(bbox, width, height, mask_dilation)
                mask[y1:y2, x1:x2] = 255

            # Guardar máscara por fotograma
            mask_path = output_masks_dir / f"{idx:06d}.png"
            cv2.imwrite(str(mask_path), mask)

            # Guardar previsualización si procede
            if writer is not None:
                vis = frame.copy()
                if bbox is not None:
                    color_layer = np.zeros_like(vis)
                    color_layer[:] = (0, 255, 255)
                    mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    blended = cv2.addWeighted(vis, 1.0, color_layer, overlay_alpha, 0)
                    vis = np.where(mask_3c > 0, blended, vis)

                    # BBox y texto
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    conf = frame_info.get("confidence")
                    label_conf = conf if isinstance(conf, (int, float)) else None
                    if label_conf is None:
                        label_conf = self.detector.min_confidence
                    source = frame_info.get("source")
                    source_suffix = f" | {source}" if isinstance(source, str) else ""
                    label = f"Watermark: {label_conf:.2f}{source_suffix}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis, (x1, max(0, y1 - th - 10)), (x1 + tw + 5, y1), (0, 255, 0), -1)
                    cv2.putText(vis, label, (x1 + 2, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                writer.write(vis)
                written_frames += 1

            if progress_callback and idx % 10 == 0:
                progress = int(60 + (idx / max(1, total_frames)) * 40)
                progress_callback(progress)

        if writer is not None:
            writer.release()
            logger.info(f"Preview VideoWriter liberado. Fotogramas escritos: {written_frames}")

            # Si el archivo final es AVI, intentar convertir a MP4 (H.264) para compatibilidad del navegador
            if used_preview_path is not None and used_preview_path.suffix.lower() == ".avi":
                try:
                    mp4_target = preview_video_path if preview_video_path.suffix.lower() == ".mp4" else used_preview_path.with_suffix(".mp4")
                    logger.info(f"Convirtiendo previsualización AVI a MP4: {used_preview_path} -> {mp4_target}")
                    (
                        ffmpeg.input(str(used_preview_path))
                        .output(str(mp4_target), vcodec="libx264", pix_fmt="yuv420p", movflags="+faststart")
                        .overwrite_output()
                        .global_args("-loglevel", "error")
                        .run()
                    )
                    try:
                        used_preview_path.unlink()
                    except Exception:
                        pass
                    used_preview_path = mp4_target
                    logger.info(f"Conversión a MP4 completada: {used_preview_path}")
                except Exception as e:
                    logger.error(f"Fallo convirtiendo AVI a MP4: {e}")

            # Si es MP4 pero se usó códec mp4v (no reproducible en navegador), convertir a H.264 (avc1)
            if (
                used_preview_path is not None
                and used_preview_path.suffix.lower() == ".mp4"
                and selected_codec is not None
                and selected_codec.lower() != "avc1"
            ):
                try:
                    h264_target = used_preview_path.with_name(used_preview_path.stem + "_h264.mp4")
                    logger.info(
                        f"Transcodificando MP4 de {selected_codec} a H.264 para navegador: {used_preview_path} -> {h264_target}"
                    )
                    (
                        ffmpeg.input(str(used_preview_path))
                        .output(str(h264_target), vcodec="libx264", pix_fmt="yuv420p", movflags="+faststart")
                        .overwrite_output()
                        .global_args("-loglevel", "error")
                        .run()
                    )
                    try:
                        used_preview_path.unlink()
                    except Exception:
                        pass
                    used_preview_path = h264_target
                    logger.info(f"Transcodificación a H.264 completada: {used_preview_path}")
                except Exception as e:
                    logger.error(f"Fallo transcodificando MP4 a H.264: {e}")

            # Validar archivo de salida
            try:
                if used_preview_path is not None and used_preview_path.exists():
                    size_bytes = used_preview_path.stat().st_size
                    logger.info(f"Previsualización creada correctamente: {used_preview_path} (tamaño: {size_bytes} bytes)")
                    cap = cv2.VideoCapture(str(used_preview_path))
                    ok, _ = cap.read()
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
                    cap.release()
                    logger.info(f"Validación de lectura: ok={ok}, frames_en_archivo={count}")
                else:
                    logger.error(f"Archivo de previsualización no encontrado: {used_preview_path}")
            except Exception as e:
                logger.error(f"Error validando previsualización: {e}")
        if progress_callback:
            progress_callback(100)
        logger.info(
            f"Detección terminada. Total fotogramas: {total_frames}, con marca detectada: {detected_frames}"
        )
        return used_preview_path

    def merge_audio_track(
        self, input_video_path: Path, temp_output_path: Path, output_video_path: Path
    ):
        logger.info("Merging audio track...")
        video_stream = ffmpeg.input(str(temp_output_path))
        audio_stream = ffmpeg.input(str(input_video_path)).audio

        (
            ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_video_path),
                vcodec="copy",
                acodec="aac",
            )
            .overwrite_output()
            .run(quiet=True)
        )
        # Clean up temporary file
        temp_output_path.unlink()
        logger.info(f"Saved no watermark video with audio at: {output_video_path}")


if __name__ == "__main__":
    from pathlib import Path

    input_video_path = Path(
        "resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4"
    )
    output_video_path = Path("outputs/sora_watermark_removed.mp4")
    sora_wm = SoraWM()
    sora_wm.run(input_video_path, output_video_path)
