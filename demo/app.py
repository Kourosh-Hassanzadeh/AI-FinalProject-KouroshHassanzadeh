import io
import os
import sys
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# Ensure the local directory is in the path to import the processor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processor import ImageProcessor


class ClassicalVisionApp:
    """Manages the Streamlit UI and interaction flow."""

    def __init__(self) -> None:
        """Initialize the application state and configure the page."""
        self._setup_page_config()
        self.processor = ImageProcessor()
        
        # State variables for storing the images
        self.uploaded_image: Optional[np.ndarray] = None
        self.gray_image: Optional[np.ndarray] = None
        self.result_image: Optional[np.ndarray] = None

    def _setup_page_config(self) -> None:
        """Configure Streamlit page settings and inject custom CSS."""
        st.set_page_config(
            page_title="Image Processing Project",
            page_icon="📷",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS to improve UI and hide unnecessary Streamlit elements
        st.markdown("""
            <style>
            /* مخفی کردن دکمه Deploy و منوی سه نقطه (بدون مخفی کردن فلش سایدبار) */
            .stDeployButton {display:none;}
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden;}
            
            .block-container {padding-top: 1rem;}
            div[data-testid="stMarkdownContainer"] p {
                font-size: 1.1em;
            }
            /* استایل دکمه‌ها */
            div.stButton > button:first-child {
                width: 100%;
                background-color: #f0f2f6;
                border: 1px solid #d0d0d0;
                color: black;
            }
            div.stButton > button:first-child:hover {
                background-color: #e0e2e6;
                border-color: #4CAF50;
                color: #4CAF50;
            }
            </style>
            """, unsafe_allow_html=True)

    def _render_header(self) -> None:
        """Render the main application header and project description."""
        st.title("سامانه هوشمند پردازش تصویر")
        
        st.markdown("""
        <div style="direction: rtl; text-align: right; background-color: #f8f9fa; 
                    color: #000000; padding: 15px; border-radius: 10px; 
                    border-right: 5px solid #4CAF50; margin-bottom: 20px;">
        این دمو جهت ارائه پروژه نهایی درس یادگیری ماشین و پردازش تصویر طراحی شده است.
        شامل ماژول‌های: <b>حذف نویز</b>، <b>لبه‌یابی پیشرفته</b>، <b>آستانه‌گذاری (Thresholding)</b>، 
        <b>فیلترهای فرکانسی</b> و <b>HOG</b>.
        </div>
        """, unsafe_allow_html=True)

    def _render_sidebar(self) -> str:
        """Render the sidebar for file upload and module selection.
        
        Returns
        -------
        str
            The selected processing module name.
        """
        st.sidebar.header("📂 ورودی و تنظیمات")
        
        uploaded_file = st.sidebar.file_uploader(
            "تصویر ورودی را انتخاب کنید", 
            type=["jpg", "png", "jpeg", "bmp", "tif", "tiff"]
        )

        if uploaded_file is not None:
            try:
                self.uploaded_image = self.processor.load_image(uploaded_file)
                self.gray_image = self.processor.to_gray(self.uploaded_image)
            except ValueError as e:
                st.sidebar.error(f"Image load error: {e}")

        st.sidebar.markdown("---")
        
        # Use st.radio instead of st.selectbox to prevent typing/searching
        # and act strictly as a navigation menu.
        # Dropdown menu for module selection
        operation = st.sidebar.selectbox(
            "🛠 انتخاب ماژول پردازشی",
            (
                "تحلیل هیستوگرام و HOG", 
                "شبیه‌سازی و حذف نویز", 
                "لبه‌یابی (Edge Detection)", 
                "فیلترهای مکانی و فرکانسی",
                "آستانه‌گذاری (Thresholding)"
            )
        )
        
        return operation

    # -------------------------------------------------------------------------
    # Module 1: Histogram & HOG
    # -------------------------------------------------------------------------
    def _handle_histogram_hog(self):
        st.subheader("📊 تحلیل تصویر (Histogram & HOG)")
        
        # --- تغییر جدید: اضافه شدن دکمه برای هیستوگرام ---
        if st.button("📊 محاسبه و رسم هیستوگرام"):
            with st.spinner("در حال رسم نمودارها..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### هیستوگرام (Grayscale)")
                    if self.gray_image is not None:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        hist = self.processor.calculate_histogram(self.gray_image)
                        ax.plot(hist, color='black')
                        ax.fill_between(range(256), hist.ravel(), color='gray', alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)

                with col2:
                    st.markdown("##### هیستوگرام رنگی (RGB)")
                    if self.uploaded_image is not None and len(self.uploaded_image.shape) == 3:
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        colors = ('r', 'g', 'b')
                        histograms = self.processor.calculate_histogram(self.uploaded_image)
                        for i, (h, c) in enumerate(zip(histograms, colors)):
                            ax2.plot(h, color=c)
                        st.pyplot(fig2)
                        plt.close(fig2)

        st.divider()
        st.subheader("🔹 استخراج ویژگی‌های HOG")
        
        # دکمه اجرای HOG
        if st.button("اجرای استخراج ویژگی HOG"):
            with st.spinner("در حال محاسبه HOG..."):
                hog_result = self.processor.compute_hog(self.uploaded_image)
                st.image(hog_result, caption="HOG Features visualization", use_container_width=True)
                self.result_image = hog_result

    # -------------------------------------------------------------------------
    # Module 2: Noise Management
    # -------------------------------------------------------------------------
    def _handle_noise(self) -> None:
        """Handle UI and logic for adding and removing noise."""
        st.subheader("🌫️ مدیریت نویز (شبیه‌سازی و حذف)")
        
        # 1. Noise Simulation
        st.info("۱. شبیه‌سازی نویز (اختیاری)")
        add_noise = st.checkbox("فعال‌سازی پنل نویز مصنوعی")
        
        current_image = self.uploaded_image
        
        if add_noise:
            noise_type = st.radio("نوع نویز:", ["Salt & Pepper", "Gaussian"], horizontal=True)
            
            if noise_type == "Salt & Pepper":
                amount = st.slider("Amount", 0.01, 0.50, 0.05, step=0.01)
                s_vs_p = st.slider("Salt vs Pepper Ratio", 0.0, 1.0, 0.5, step=0.1)
                if st.button("اعمال نویز Salt & Pepper"):
                    current_image = self.processor.add_salt_pepper_noise(
                        self.uploaded_image, amount, s_vs_p
                    )
                    st.session_state['noisy_image'] = current_image
            else:
                mean = st.number_input("Mean", value=10.0)
                std = st.slider("Std Dev", 0.0, 100.0, 25.0)
                if st.button("اعمال نویز Gaussian"):
                    current_image = self.processor.add_gaussian_noise(
                        self.uploaded_image, mean, std
                    )
                    st.session_state['noisy_image'] = current_image

            # Display the noisy image if available in state
            if 'noisy_image' in st.session_state:
                current_image = st.session_state['noisy_image']
                st.image(current_image, caption="تصویر نویزی شده (ورودی فیلتر)", width=400)
        else:
            # Clear state if user disables noise simulation
            st.session_state.pop('noisy_image', None)

        st.divider()

        # 2. Denoising Filters
        st.info("۲. حذف نویز (Denoising)")
        method = st.selectbox("روش حذف نویز:", ["Median Filter", "Bilateral Filter"])
        
        if method == "Median Filter":
            k = st.slider("Kernel Size (فرد)", 3, 21, 5, step=2)
            if st.button("اجرای فیلتر Median"):
                self.result_image = self.processor.apply_median_filter(current_image, k)
                
        elif method == "Bilateral Filter":
            d = st.slider("Diameter", 3, 30, 9)
            sigma_c = st.slider("Sigma Color", 10.0, 200.0, 75.0)
            sigma_s = st.slider("Sigma Space", 10.0, 200.0, 75.0)
            if st.button("اجرای فیلتر Bilateral"):
                self.result_image = self.processor.apply_bilateral_filter(
                    current_image, d, sigma_c, sigma_s
                )

    # -------------------------------------------------------------------------
    # Module 3: Edge Detection
    # -------------------------------------------------------------------------
    def _handle_edges(self) -> None:
        """Handle UI and logic for various edge detection algorithms."""
        st.subheader("✏️ لبه‌یابی پیشرفته")
        
        method = st.selectbox(
            "الگوریتم:", 
            ("Prewitt", "Kirsch", "Marr-Hildreth", "Canny")
        )

        if method == "Prewitt":
            st.caption("تشخیص لبه ساده با اپراتور Prewitt")
            if st.button("اجرای Prewitt"):
                self.result_image = self.processor.detect_edges_prewitt(self.uploaded_image)

        elif method == "Kirsch":
            st.caption("تشخیص لبه جهت‌دار (8 جهت)")
            if st.button("اجرای Kirsch"):
                with st.spinner("در حال پردازش (ممکن است زمان‌بر باشد)..."):
                    self.result_image = self.processor.detect_edges_kirsch(self.uploaded_image)

        elif method == "Marr-Hildreth":
            st.caption("Laplacian of Gaussian (LoG)")
            sigma = st.slider("Sigma", 0.1, 5.0, 1.4, step=0.1)
            thresh = st.slider("Threshold", 0.0, 1.0, 0.5, step=0.05)
            if st.button("اجرای Marr-Hildreth"):
                self.result_image = self.processor.detect_edges_marr_hildreth(
                    self.uploaded_image, sigma, thresh
                )

        elif method == "Canny":
            st.caption("Canny Edge Detector")
            min_t = st.slider("Min Threshold", 0, 255, 100)
            max_t = st.slider("Max Threshold", 0, 255, 200)
            if st.button("اجرای Canny"):
                self.result_image = self.processor.detect_edges_canny(
                    self.uploaded_image, min_t, max_t
                )

    # -------------------------------------------------------------------------
    # Module 4: Spatial & Frequency Filters
    # -------------------------------------------------------------------------
    def _handle_spatial_frequency(self) -> None:
        """Handle UI and logic for spatial domain and frequency domain filtering."""
        st.subheader("〰️ فیلترهای مکانی و فرکانسی")
        
        domain = st.radio(
            "حوزه پردازش:", 
            ["Spatial Domain (مکانی)", "Frequency Domain (فرکانسی/FFT)"], 
            horizontal=True
        )

        if domain == "Spatial Domain (مکانی)":
            sp_method = st.selectbox(
                "نوع فیلتر:", 
                ["Average", "Gaussian", "Median", "Sharpening", "Sobel"]
            )
            
            if sp_method in ["Average", "Gaussian", "Median"]:
                k = st.slider("Kernel Size", 3, 31, 5, step=2)
                if st.button(f"اجرای فیلتر {sp_method}"):
                    if sp_method == "Average":
                        self.result_image = self.processor.apply_average_filter(self.uploaded_image, k)
                    elif sp_method == "Gaussian":
                        self.result_image = self.processor.apply_gaussian_filter(self.uploaded_image, k)
                    elif sp_method == "Median":
                        self.result_image = self.processor.apply_spatial_median(self.uploaded_image, k)
            
            elif sp_method == "Sharpening":
                if st.button("اجرای فیلتر Sharpening"):
                    self.result_image = self.processor.apply_sharpening_filter(self.uploaded_image)
            
            elif sp_method == "Sobel":
                if st.button("اجرای فیلتر Sobel"):
                    self.result_image = self.processor.apply_sobel_filter(self.uploaded_image)

        else:
            st.info("اعمال تبدیل فوریه (FFT) و فیلترهای بالاگذر/پایین‌گذر")
            if st.button("محاسبه FFT و فیلترها"):
                lpf, hpf = self.processor.apply_frequency_filters(self.uploaded_image)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.image(lpf, caption="Low Pass Filter (Blur)", use_container_width=True)
                with c2:
                    st.image(hpf, caption="High Pass Filter (Edge)", use_container_width=True)
                
                # Default output for download
                self.result_image = lpf

    # -------------------------------------------------------------------------
    # Module 5: Thresholding
    # -------------------------------------------------------------------------
    def _handle_thresholding(self) -> None:
        """Handle UI and logic for image binarization and thresholding."""
        st.subheader("🎨 آستانه‌گذاری (Thresholding)")
        
        th_method = st.selectbox(
            "روش:", 
            ["Simple", "Adaptive Mean", "Adaptive Gaussian", "Otsu"]
        )
        
        # Thresholding requires grayscale images
        if self.gray_image is None: 
            return

        if th_method == "Simple":
            val = st.slider("Threshold Value", 0, 255, 127)
            inv = st.checkbox("Inverse", value=False)
            if st.button("اجرای Simple Threshold"):
                self.result_image = self.processor.threshold_simple(self.gray_image, val, inv)
        
        elif th_method in ["Adaptive Mean", "Adaptive Gaussian"]:
            blk = st.slider("Block Size (Odd)", 3, 51, 11, step=2)
            c = st.slider("Constant C", -10, 10, 2)
            if st.button(f"اجرای {th_method}"):
                if th_method == "Adaptive Mean":
                    self.result_image = self.processor.threshold_adaptive_mean(self.gray_image, blk, c)
                else:
                    self.result_image = self.processor.threshold_adaptive_gaussian(self.gray_image, blk, c)
        
        elif th_method == "Otsu":
            if st.button("اجرای Otsu Binarization"):
                val, res = self.processor.threshold_otsu(self.gray_image)
                st.success(f"Otsu calculated optimal threshold: {val}")
                self.result_image = res

    # -------------------------------------------------------------------------
    # Main Execution
    # -------------------------------------------------------------------------
    def _render_download_button(self) -> None:
        """Helper to create a download button for the processed result."""
        if self.result_image is None:
            return

        try:
            pil_img = Image.fromarray(self.result_image)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="📥 دانلود تصویر خروجی",
                data=byte_im,
                file_name="processed_result.png",
                mime="image/png",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Download generation error: {e}")

    def run(self) -> None:
        """Main entry point for orchestrating the application flow."""
        self._render_header()
        operation = self._render_sidebar()

        if self.uploaded_image is None:
            st.info("👈 لطفاً برای شروع، یک تصویر را از پنل سمت راست آپلود کنید.")
            return

        # Display the original image on the left column
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼 تصویر اصلی")
            st.image(
                self.uploaded_image, 
                use_container_width=True, 
                caption=f"Original Size: {self.uploaded_image.shape}"
            )

        # Route the flow based on user selection
        if operation == "تحلیل هیستوگرام و HOG":
            self._handle_histogram_hog()
        elif operation == "شبیه‌سازی و حذف نویز":
            self._handle_noise()
        elif operation == "لبه‌یابی (Edge Detection)":
            self._handle_edges()
        elif operation == "فیلترهای مکانی و فرکانسی":
            self._handle_spatial_frequency()
        elif operation == "آستانه‌گذاری (Thresholding)":
            self._handle_thresholding()

        # Render the final result on the right column if it exists
        if self.result_image is not None:
            with col2:
                st.subheader("✨ نتیجه پردازش")
                
                # Determine display mode based on array shape
                channels = "RGB" if self.result_image.ndim == 3 else "GRAY"
                st.image(
                    self.result_image, 
                    use_container_width=True, 
                    channels=channels, 
                    caption=operation
                )
                
                self._render_download_button()

if __name__ == "__main__":
    app = ClassicalVisionApp()
    app.run()