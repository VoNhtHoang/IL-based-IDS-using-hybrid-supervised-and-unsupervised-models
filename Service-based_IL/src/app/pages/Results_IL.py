# 1rd libs
import os
import shutil
from datetime import datetime
from pathlib import Path

# 3rd libs
import joblib
import streamlit as st
from PIL import Image


# Local Import
from src.config.settings import settings
from src.config.incremental_config import incremental_settings

from src.Components.Manager import Manager
from src.Components.Models import AETrainer, IncrementalOCSVM, OpenSetXGBoost


# MAIN

def main():
    st.set_page_config(layout="wide", page_title="Xem kết quả Incremental Learning theo ngày")
    
    st.markdown("## <i class='bi bi-images'></i> Xem Kết quả IL theo ngày", unsafe_allow_html= True)
    
    # Tạo layout với 2 cột
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### <i class='bi bi-calendar'></i> Lựa chọn", unsafe_allow_html=True)
        
        # Ô chọn ngày
        selected_date = st.date_input("Chọn ngày", value=datetime.now())
        
        # Thư mục gốc chứa các thư mục ảnh (có thể thay đổi)
        base_dir = incremental_settings.IL_LOGS
        
        # Tạo thư mục nếu chưa tồn tại
        if not Path.exists(base_dir):
            os.makedirs(base_dir)
            st.info(f"Đã tạo thư mục '{base_dir}'. Vui lòng thêm thư mục ảnh vào đây.")
        
        # Lấy danh sách các thư mục
        folders = []
        if Path.exists(base_dir):
            folders = [f for f in Path(base_dir).iterdir() if f.is_dir()]
        
        if not folders:
            st.warning(f"Không tìm thấy thư mục nào trong '{base_dir}'")
            return
        
        # Lọc thư mục theo ngày (giả sử tên thư mục có chứa ngày)
        date_str = datetime.strftime(selected_date, "%Y-%m-%d")
        filtered_folders = []
        
        for folder in folders:
            fol = str(folder)
            # Kiểm tra xem tên thư mục có chứa ngày đã chọn không
            if date_str in fol:
                filtered_folders.append(folder)
            # Hoặc kiểm tra các định dạng ngày khác
            elif selected_date.strftime("%Y%m%d") in fol:
                filtered_folders.append(folder)
            elif selected_date.strftime("%d-%m-%Y") in fol:
                filtered_folders.append(folder)
        
        # Hiển thị tất cả thư mục nếu không tìm thấy thư mục theo ngày
        if not filtered_folders:
            st.info(f"Không có thư mục nào chứa ngày {date_str}")
            st.markdown("### <i class='bi bi-card-list'></i> Tất cả thư mục có sẵn:", unsafe_allow_html=True)
            all_folders = folders
        else:
            st.markdown(f"### <i class='bi bi-calendar'></i> Thư mục chứa ngày {date_str}:", unsafe_allow_html= True)
            all_folders = filtered_folders
        
        # Hiển thị danh sách thư mục dưới dạng radio button
        if all_folders:
            selected_folder = st.radio(
                "≡ Chọn thư mục:",
                all_folders,
                key="folder_selector"
            )
            
            # Đường dẫn đầy đủ đến thư mục đã chọn
            folder_path = Path(selected_folder)
            
            # Lấy danh sách ảnh trong thư mục
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path.glob(folder_path, ext))
            
            # Sắp xếp ảnh theo tên
            image_files.sort()
            
            # Hiển thị thông tin thư mục
            st.info(f"**Thư mục:** {selected_folder}")
            st.info(f"**Số lượng ảnh:** {len(image_files)}")
            
            # Tạo selectbox để chọn ảnh nếu có ảnh
            if image_files:
                image_names = [os.path.basename(img) for img in image_files]
                selected_image_name = st.selectbox(
                    "≡ Chọn ảnh để xem:",
                    image_names,
                    key="image_selector"
                )
                
                # Đường dẫn đến ảnh đã chọn
                selected_image_path = Path(folder_path/selected_image_name)
            else:
                selected_image_path = None
                st.warning("Không tìm thấy ảnh trong thư mục này")
        else:
            selected_image_path = None
            st.warning("Không có thư mục nào để hiển thị")
        
        st.markdown("### <i class='bi bi-calendar'></i> Phiên bản các model hiện tại", unsafe_allow_html=True)
        
        with st.container(border=True):
            if Path.exists(settings.MODEL_DIR):
                ae = AETrainer(81, 32); ocsvm = IncrementalOCSVM(nu=0.15); xgb = OpenSetXGBoost(0.75);
                
                mgr = Manager(settings.MODEL_DIR)
                # mgr.load_models([xgb, ocsvm, ae])
                res_info = mgr.load_models_info([xgb, ocsvm, ae])
                # print(f"[PY] Model State: {xgb.loaded}, {ocsvm.loaded}, {ae.loaded}")
                
                info_cols = st.columns([1,1,1])
                for i, (m_name, info) in enumerate(res_info.items()):
                    with info_cols[i]:
                        st.caption(f"{m_name}")
                        st.write(f"**Ver:** {info[0]}")
                        st.write(f"**Date:** {info[1]}")
                        
            del ae, ocsvm, xgb
        
        # Cập nhật model nếu cần
        st.markdown("### <i class='bi bi-calendar'></i> Các model mới", unsafe_allow_html=True)
        model_dir = Path(settings.MODEL_DIR / "pre_models")
        
        model_folders = []
        if Path.exists(model_dir):
            model_folders = [f for f in Path(model_dir).iterdir() if f.is_dir()]
        
        if len(model_folders )> 0:
            selected_model_folder = st.selectbox( 
                "≡ Chọn thư mục để xem:",
                model_folders,
                key="model_selector"
            )
            
        with st.container(border=True):
            if Path.exists(selected_model_folder):
                ae = AETrainer(81, 32); ocsvm = IncrementalOCSVM(nu=0.15); xgb = OpenSetXGBoost(0.75);
                
                mgr = Manager(selected_model_folder)
                # mgr.load_models([xgb, ocsvm, ae])
                res_info = mgr.load_models_info([xgb, ocsvm, ae])
                # print(f"[PY] Model State: {xgb.loaded}, {ocsvm.loaded}, {ae.loaded}")
                
                info_cols = st.columns([1,1,1])
                for i, (m_name, info) in enumerate(res_info.items()):
                    with info_cols[i]:
                        st.caption(f"{m_name}")
                        st.write(f"**Ver:** {info[0]}")
                        st.write(f"**Date:** {info[1]}")
        
        if st.button("Copy vào thư mục hoạt động"):
            if not selected_model_folder:
                st.warning("Vui lòng chọn ít nhất một folder.")
            else:
                for index, model in enumerate([xgb, ocsvm, ae]):
                    f = model.model_name
                    f = Path(selected_model_folder/f"{f}")
                    shutil.copy(f.resolve(), Path(settings.MODEL_DIR))
                                
                st.success(f"Đã copy thành công các model từ {selected_model_folder} file vào thư mục Incremental {settings.MODEL_DIR}! Restart các worker để thấy hiệu lực!")
    
    with col2:
        st.markdown("### <i class='bi bi-image'></i> Xem ảnh & Đổi mô hình huấn luyện hiện tại", unsafe_allow_html=True)
        with st.container():
            if selected_image_path and os.path.exists(selected_image_path):
                try:
                    # Mở và hiển thị ảnh
                    image = Image.open(selected_image_path)
                    
                    # Hiển thị thông tin ảnh
                    st.caption(f"Ảnh: {os.path.basename(selected_image_path)}")
                    
                    # Tạo tabs để xem ảnh ở các kích thước khác nhau
                    tab1, tab2, tab3 = st.tabs(["Xem ảnh", "Kích thước gốc", "Thông tin"])
                    
                    with tab1:
                        # Hiển thị ảnh toàn màn hình
                        st.image(image, use_container_width=True)
                    
                    with tab2:
                        # Hiển thị ảnh kích thước gốc
                        st.image(image)
                    
                    with tab3:
                        # Hiển thị thông tin ảnh
                        st.write(f"**Kích thước:** {image.size[0]} x {image.size[1]} pixels")
                        st.write(f"**Định dạng:** {image.format}")
                        st.write(f"**Chế độ:** {image.mode}")
                        st.write(f"**Đường dẫn:** {selected_image_path}")
                        
                        # Hiển thị ảnh nhỏ trong tab thông tin
                        st.image(image, width=300)
                    
                    # Nút tải ảnh xuống
                    with open(selected_image_path, "rb") as file:
                        btn = st.download_button(
                            label="↧ Tải ảnh xuống",
                            data=file,
                            file_name=os.path.basename(selected_image_path),
                            mime=f"image/{image.format.lower() if image.format else 'jpeg'}"
                        )
                    
                except Exception as e:
                    st.error(f"Không thể mở ảnh: {str(e)}")
            elif selected_image_path:
                st.error(f"Không tìm thấy file: {selected_image_path}")
            else:
                st.info("Vui lòng chọn một thư mục và ảnh để xem")
                
                # Hiển thị hướng dẫn
                st.markdown("""
                ### Hướng dẫn sử dụng:
                1. **Chọn ngày** ở cột bên trái
                2. **Chọn thư mục** từ danh sách
                3. **Chọn ảnh** từ thư mục đã chọn
                4. **Xem ảnh** ở cột bên phải
                
                ### Cấu trúc thư mục đề xuất:
                ```
                images/
                ├── event_2024-01-15_du-lich/
                │   ├── anh1.jpg
                │   ├── anh2.jpg
                │   └── anh3.png
                ├── meeting_2024-01-20_cong-ty/
                │   ├── hinh1.jpg
                │   └── hinh2.png
                └── birthday_2024-01-25_sinh-nhat/
                    └── birthday_photo.jpg
                ```
                
                Lưu ý: Tên thư mục nên chứa ngày theo định dạng YYYY-MM-DD
                """)
            
                
                