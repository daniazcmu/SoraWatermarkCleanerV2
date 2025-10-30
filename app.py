import shutil
import tempfile
from pathlib import Path

import streamlit as st

from sorawm.core import SoraWM


DEFAULT_MIN_CONFIDENCE = 0.01
DEFAULT_MASK_DILATION = 4
DEFAULT_UPSCALE_FACTOR = 1.7
DEFAULT_CLAHE_CLIP = 2.2
DEFAULT_TEMPLATE_THRESHOLD = 0.80
DEFAULT_TEMPLATE_SEARCH_EXPAND = 32
DEFAULT_SHARPEN = True


def main():
    st.set_page_config(
        page_title="Sora Watermark Cleaner", page_icon="🎬", layout="centered"
    )

    st.title("🎬 Sora Watermark Cleaner")
    st.markdown("Remove watermarks from Sora-generated videos with ease")

    # Initialize SoraWM
    if "sora_wm" not in st.session_state:
        with st.spinner("Loading AI models..."):
            st.session_state.sora_wm = SoraWM(
                min_confidence=DEFAULT_MIN_CONFIDENCE,
                upscale_factor=DEFAULT_UPSCALE_FACTOR,
                clahe_clip_limit=DEFAULT_CLAHE_CLIP,
                sharpen=DEFAULT_SHARPEN,
            )
            st.session_state.sora_wm.update_detector_params(
                template_threshold=DEFAULT_TEMPLATE_THRESHOLD,
                template_search_expand=DEFAULT_TEMPLATE_SEARCH_EXPAND,
            )

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Select a video file to remove watermarks",
    )

    if uploaded_file is not None:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.video(uploaded_file)

        st.markdown("---")
        st.markdown("### Ajustes de detección")

        col1, col2, col3 = st.columns(3)
        with col1:
            min_confidence = st.slider(
                "Umbral de confianza mínima",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_MIN_CONFIDENCE,
                step=0.01,
                help="Filtra detecciones con confianza por debajo del umbral",
            )
        with col2:
            mask_dilation = st.slider(
                "Dilatación de máscara (px)",
                min_value=-64,
                max_value=128,
                value=DEFAULT_MASK_DILATION,
                step=1,
                help="Expande (>0) o contrae (<0) el bbox antes de crear la máscara",
            )
        with col3:
            upscale_factor = st.slider(
                "Reescala de preprocesado",
                min_value=1.0,
                max_value=2.0,
                value=DEFAULT_UPSCALE_FACTOR,
                step=0.1,
                help=">1.0 ayuda a detectar marcas pequeñas (interpolación bicúbica)",
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            clahe_clip_limit = st.slider(
                "CLAHE clip limit",
                min_value=0.0,
                max_value=4.0,
                value=DEFAULT_CLAHE_CLIP,
                step=0.1,
                help="0 desactiva; valores >0 mejoran contraste en marcas claras",
            )
        with col5:
            template_threshold = st.slider(
                "Umbral plantilla",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_TEMPLATE_THRESHOLD,
                step=0.01,
                help="Similitud mínima para aceptar la coincidencia con la plantilla",
            )
        with col6:
            template_search_expand = st.slider(
                "Expansión búsqueda plantilla (px)",
                min_value=0,
                max_value=128,
                value=DEFAULT_TEMPLATE_SEARCH_EXPAND,
                step=4,
                help="Radio extra alrededor del bbox de YOLO para refinar con plantilla",
            )

        sharpen = st.checkbox("Nitidez (unsharp mask)", value=DEFAULT_SHARPEN)
        template_file = st.file_uploader(
            "Plantilla del watermark (opcional)",
            type=["png", "jpg", "jpeg"],
            help="Usa la imagen oficial de la marca para refinar detecciones",
        )

        st.caption(
            "Los parámetros aplican tanto a la limpieza completa como a la previsualización. "
            "Si aportas una plantilla se combinará con YOLO para validar la posición."
        )

        def apply_detector_config(temp_dir: Path | None) -> Path | None:
            template_local: Path | None = None
            if template_file is not None and temp_dir is not None:
                template_local = temp_dir / template_file.name
                template_file.seek(0)
                with open(template_local, "wb") as tf:
                    tf.write(template_file.read())
                template_file.seek(0)

            clahe_value = clahe_clip_limit if clahe_clip_limit > 0 else None
            st.session_state.sora_wm.update_detector_params(
                min_confidence=min_confidence,
                upscale_factor=upscale_factor,
                clahe_clip_limit=clahe_value,
                sharpen=sharpen,
                template_path=template_local,
                clear_template=template_file is None,
                template_threshold=template_threshold,
                template_search_expand=int(template_search_expand),
            )
            return template_local

        st.markdown("---")
        st.markdown("### Acciones")

        if st.button("🚀 Remove Watermark", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                apply_detector_config(tmp_path)

                input_path = tmp_path / uploaded_file.name
                with open(input_path, "wb") as f:
                    uploaded_file.seek(0)
                    f.write(uploaded_file.read())

                output_path = tmp_path / f"cleaned_{uploaded_file.name}"

                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(progress: int):
                        progress_bar.progress(progress / 100)
                        if progress < 50:
                            status_text.text(f"🔍 Detectando intervalos... {progress}%")
                        elif progress < 95:
                            status_text.text(f"🧹 Limpiando con máscara... {progress}%")
                        else:
                            status_text.text(f"🎵 Combinando audio... {progress}%")

                    st.session_state.sora_wm.run(
                        input_path,
                        output_path,
                        progress_callback=update_progress,
                        mask_dilation=int(mask_dilation),
                    )

                    progress_bar.progress(100)
                    status_text.text("✅ Procesado completado")

                    st.success("✅ Watermark eliminada correctamente")
                    st.markdown("### Resultado")
                    st.video(str(output_path))

                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Cleaned Video",
                            data=f,
                            file_name=f"cleaned_{uploaded_file.name}",
                            mime="video/mp4",
                            use_container_width=True,
                        )

                except Exception as e:
                    st.error(f"❌ Error procesando el vídeo: {str(e)}")

        st.markdown("---")
        st.markdown("### 👁️ Solo detección (previsualización de máscaras)")
        overlay_alpha = st.slider(
            "Opacidad del overlay",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help="Controla la mezcla de color sobre la zona detectada",
        )

        if st.button("👁️ Previsualizar máscaras (solo detección)", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                apply_detector_config(tmp_path)

                input_path = tmp_path / uploaded_file.name
                uploaded_file.seek(0)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.read())

                masks_dir = tmp_path / "masks"
                preview_path = tmp_path / "preview_masks.mp4"

                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_preview(progress: int):
                    progress_bar.progress(progress / 100)
                    status_text.text(f"👁️ Generando previsualización... {progress}%")

                try:
                    final_preview = st.session_state.sora_wm.detect_only(
                        input_video_path=input_path,
                        output_masks_dir=masks_dir,
                        preview_video_path=preview_path,
                        progress_callback=update_preview,
                        overlay_alpha=overlay_alpha,
                        mask_dilation=int(mask_dilation),
                    )

                    progress_bar.progress(1.0)
                    status_text.text("✅ Previsualización completada")

                    preview_to_show = final_preview if final_preview is not None else preview_path
                    if preview_to_show.exists():
                        st.markdown("#### Vídeo de previsualización")
                        st.video(str(preview_to_show))
                        try:
                            size_bytes = preview_to_show.stat().st_size
                            st.caption(f"Archivo de previsualización: {preview_to_show.name} ({size_bytes} bytes)")
                        except Exception:
                            pass

                    st.markdown("#### Muestras de máscaras generadas")
                    mask_files = sorted(masks_dir.glob("*.png"))[:8]
                    if mask_files:
                        cols = st.columns(4)
                        for i, m in enumerate(mask_files):
                            with open(m, "rb") as f:
                                cols[i % 4].image(f.read(), caption=m.name)
                    else:
                        st.info("No se generaron máscaras (ninguna detección superó el umbral/plantilla).")

                    zip_base = tmp_path / "masks"
                    zip_path = shutil.make_archive(str(zip_base), "zip", masks_dir)
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Descargar máscaras (ZIP)",
                            data=f,
                            file_name=f"masks_{uploaded_file.name}.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )

                except Exception as e:
                    st.error(f"❌ Error en solo detección: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit and AI</p>
            <p><a href='https://github.com/linkedlist771/SoraWatermarkCleaner'>GitHub Repository</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
