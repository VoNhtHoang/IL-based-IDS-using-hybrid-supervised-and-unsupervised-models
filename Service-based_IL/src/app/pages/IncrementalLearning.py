# src/app/pages/IncrementalLearning.py
# STANDARDS LIBS
import streamlit as st
from datetime import datetime
from pathlib import Path
import json
import time

# 3rd import

# Local import
from src.config.settings import settings
from src.config.incremental_config import incremental_settings, IncrementalSettings

# CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = Path("./src/config/.config_incremental")

def main():    
    st.markdown("## <i class='bi bi-sliders2'></i> Incremental Learning Configuration", unsafe_allow_html=True)

    # if 'il_config_fields' not in st.session_state:
    il_config_fields = IncrementalSettings.model_fields
            
    # Layout
    cfg_col, view_col = st.columns([1.5, 1])
    
    with cfg_col:
        st.markdown("### <i class='bi bi-gear'></i> Incremental Config", unsafe_allow_html=True)
        
        with st.form("il_settings"):
            new_values = {}
            
            with st.container(border=True):
                for field_name, field_info in il_config_fields.items():
                    current_value = getattr(incremental_settings, field_name)
                    if isinstance(current_value, int) and field_name in ["MIN_INTERVAL_SEC"]:
                        new_val = st.number_input(field_name, value=current_value, step=1800, min_value= 600)
                    elif isinstance(current_value, int) and field_name in ["IL_FIXED_MEM_BUDGET"]:
                        new_val = st.number_input("Fixed Number of Samples In IL", key = field_name, value = current_value)
                    elif isinstance(current_value, Path):
                        new_val = st.text_input(
                            field_name,
                            value=str(current_value), #str(current_value.resolve() if isinstance(current_value, Path) else current_value),
                            help=field_info.description or ""
                        )
                    else:
                        new_val = current_value
                    new_values[field_name] = new_val
                
                #
                new_values["herding_replay_ratio"] = st.slider(
                    "Herding Replay Ratio",
                    0.1, 0.8, step=0.05,
                    key="herding_replay_ratio",
                    value = getattr(incremental_settings, "herding_replay_ratio")
                    # on_change=
                )
                
            with st.container(border=True):
                st.markdown("### <i class='bi bi-app-indicator'></i> Current Label Samples", unsafe_allow_html=True)
                st.json(incremental_settings.IL_LABEL())

            with st.container(border=True):
                st.markdown("### <i class='bi bi-ui-checks-grid'></i> Control", unsafe_allow_html= True)
                new_values["enable_training"] = st.checkbox(
                    "🚀 Enable Training",
                    key="enable_training",
                    value = getattr(incremental_settings, "enable_training")
                )
            

            # Nút lưu thủ công (dự phòng)
            if st.form_submit_button("✔ Lưu cấu hình ngay", type="primary"):
                lines = [f"{k}={v}" for k, v in new_values.items()]
                CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
                st.success("Đã lưu! Các thay đổi sẽ được IL job tự động áp dụng")
                
                time.sleep(2)
                st.rerun()

    with view_col:
        st.markdown("### <i class='bi bi-file-earmark-ppt'></i> Preview file src/config/.config_incremental hiện tại",unsafe_allow_html=True)

        if CONFIG_PATH.exists():
            current_content = CONFIG_PATH.read_text(encoding="utf-8")
            st.code(current_content, language="bash")
            st.caption(f"Last modified: {datetime.fromtimestamp(CONFIG_PATH.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("File `.config` chưa tồn tại. Nó sẽ được tạo khi bạn lưu lần đầu.")
            
        current_runtime = {}
        for field_name, field_info in il_config_fields.items():  # ← Lấy từ class
            current_runtime[field_name] = getattr(incremental_settings, field_name)
        
        st.json(current_runtime, expanded=True)
        
        st.markdown("---")
        st.caption("💡 Thay đổi ở đây chỉ có hiệu lực sau khi **restart app/ các service**. File `.config` có định dạng đơn giản key=value, dễ chỉnh tay bằng Notepad nếu cần.")
