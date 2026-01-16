# pages/Alerts.py
import sys, os
import time
from datetime import datetime, timedelta, date

# 3rd libs
import streamlit as st
from pathlib import Path
import pandas as pd

#
from src.config.settings import settings
from core.Load_Alerts import Load_Data, COLS_TO_DROP

# CONST
# HOME = Path.cwd()

# ======================
# MAIN
# ======================


def main():
    st.markdown(" ## <i class='bi bi-exclamation-diamond'></i> Realtime IDS Alerts Log", unsafe_allow_html= True)
    
    # Khởi tạo session state
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = getattr(settings, "AUTO_REFRESH", False)
    
    if 'selected_date' not in st.session_state:
        st.session_state.selected_date = date.today()
        
    if 'time_range' not in st.session_state:
        st.session_state.time_range = "Toàn bộ ngày"
    
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = Load_Data(
            dir_in=Path(settings.ALERTS_DIR),
            file_name = f"{st.session_state.selected_date}.jsonl",
            last_update_time=0,
            refresh_interval=60,
            auto_refresh=st.session_state.auto_refresh,
            file=None
        )

    data_loader = st.session_state.data_loader  
    
    # === Container cho bộ lọc ngày và thời gian ===
    with st.container():
        st.markdown("### <i class='bi bi-calendar-event'></i> Bộ lọc thời gian & Cài đặt", unsafe_allow_html=True)
        
        # Tạo layout với 2 cột
        col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 2])
        
        with col_filter1:
            # Chọn ngày
            selected_date = st.date_input(
                "Chọn ngày",
                value=st.session_state.selected_date,
                key="date_filter"
            )
            st.session_state.selected_date = selected_date
            
            # Chọn khoảng thời gian trong ngày
            time_options = ["Toàn bộ ngày", "Sáng (00:00-12:00)", "Chiều (12:00-18:00)", 
                           "Tối (18:00-24:00)", "Tùy chỉnh"]
            time_range = st.selectbox(
                "Khoảng thời gian",
                options=time_options,
                index=time_options.index(st.session_state.time_range),
                key="time_filter"
            )
            st.session_state.time_range = time_range
            
            # Nếu chọn tùy chỉnh, hiển thị 2 ô chọn giờ
            if time_range == "Tùy chỉnh":
                col_start, col_end = st.columns(2)
                with col_start:
                    start_hour = st.time_input("Giờ bắt đầu", value=datetime.strptime("00:00", "%H:%M").time())
                with col_end:
                    end_hour = st.time_input("Giờ kết thúc", value=datetime.strptime("23:59", "%H:%M").time())
            
            # Hiển thị thông tin bộ lọc
            st.markdown(f"<b> <i class='bi bi-calendar-minus'></i> Bộ lọc đang áp dụng - Ngày: </b> {selected_date.strftime('%d/%m/%Y')} ", unsafe_allow_html= True)
            time_range_text = st.session_state.time_range
            if time_range == "Tùy chỉnh":
                time_range_text = f"{start_hour.strftime('%H:%M')} - {end_hour.strftime('%H:%M')}"
            st.markdown(f"<b> <i class='bi bi-calendar-minus'></i> Khoảng thời gian:</b> {time_range_text}", unsafe_allow_html= True)  
            
        with col_filter2:
            # === Cài đặt refresh ===
            st.markdown("<b> <i class='bi bi-arrow-clockwise'></i> Cài đặt refresh</b>" , unsafe_allow_html= True)
            
            auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto_refresh
            data_loader.auto_refresh = auto_refresh
            
            if auto_refresh:
                interval = st.selectbox("Thời gian refresh (giây)", [5, 10, 30, 60], index=3)
                data_loader.refresh_interval = interval
            
            # Nút refresh ngay
            if st.button("⟳ Refresh Ngay", use_container_width=True):
                data_loader.enable_reload_immediate = True
                # st.session_state.data_loader = data_loader
                st.rerun()
            
            st.markdown("---")
        
        with col_filter3:
            # Nút điều khiển
            st.markdown("<b> <i class='bi bi-ui-checks-grid'></i> Điều khiển</b>", unsafe_allow_html= True)
            
            # Tạo 2 cột cho các nút
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("✔ Áp dụng", type="primary", use_container_width=True):
                    st.session_state.data_loader = Load_Data(
                            dir_in=Path(settings.ALERTS_DIR),
                            file_name = f"{st.session_state.selected_date}.jsonl",
                            last_update_time=0,
                            refresh_interval=60,
                            auto_refresh=st.session_state.auto_refresh,
                            file=None
                        )
                    st.rerun()
            
            with col_btn2:
                if st.button("⟳ Reset", use_container_width=True):
                    
                    st.session_state.selected_date = date.today()
                    st.session_state.time_range = "Toàn bộ ngày"
                    
                    st.session_state.data_loader = Load_Data(
                        dir_in=Path(settings.ALERTS_DIR),
                        file_name = f"{st.session_state.selected_date}.jsonl",
                        last_update_time=0,
                        refresh_interval=60,
                        auto_refresh=st.session_state.auto_refresh,
                        file=None
                    )
                    
                    st.rerun()
            
            st.markdown("---")
            
            # Thông tin hệ thống
            st.markdown("<b> <i class='bi bi-exclamation-square'></i> Thông tin hệ thống:</b>", unsafe_allow_html= True)
            
            # Hiển thị số lượng file alerts (nếu có thông tin từ data_loader)
            try:
                alert_files = list(settings.ALERTS_DIR.glob("*.parquet")) + list(settings.ALERTS_DIR.glob("*.jsonl"))
                st.markdown(f"<b> <i class='bi bi-exclamation-square'></i> Số file alerts:</b> {len(alert_files)}", unsafe_allow_html= True)
            except:
                pass
            
            # Trạng thái auto refresh
            status_color = "🟢" if auto_refresh else "🔴"
            st.markdown(f"{status_color} <b>Auto Refresh:</b> {'Bật' if auto_refresh else 'Tắt'}", unsafe_allow_html=True)
            
            if auto_refresh:
                st.markdown(f"<b> <i class='bi bi-hourglass-split'></i> Interval:</b> {data_loader.refresh_interval} giây", unsafe_allow_html= True)
        
        st.markdown("---")

    # === Hiển thị thông tin cảnh báo ===
    st.markdown(f"### <i class='bi bi-exclamation-diamond'></i> Dữ liệu cảnh báo - Ngày {selected_date.strftime('%d/%m/%Y')}", unsafe_allow_html=True)
    
    # Hiển thị khoảng thời gian đã chọn
    time_range_display = st.session_state.time_range
    if time_range == "Tùy chỉnh":
        time_range_display = f"{start_hour.strftime('%H:%M')} - {end_hour.strftime('%H:%M')}"
    
    st.caption(f"Khoảng thời gian: {time_range_display}")
    
    # === Load và hiển thị dữ liệu ===
    df = data_loader.load_alerts(limit=200)
    
    if df.empty:
        st.info("Chưa có alert nào được ghi.")
        st.stop()
    
    # Chuyển đổi cột thời gian nếu có
    time_column = COLS_TO_DROP[1] if len(COLS_TO_DROP) > 1 else None
    
    if time_column and time_column in df.columns:
        try:
            # Đảm bảo cột thời gian là datetime
            if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
            
            # Lọc theo ngày đã chọn
            start_of_day = datetime.combine(selected_date, datetime.min.time())
            end_of_day = datetime.combine(selected_date, datetime.max.time())
            
            # Lọc dữ liệu trong ngày
            mask = (df[time_column] >= start_of_day) & (df[time_column] <= end_of_day)
            df_filtered = df[mask].copy()
            
            # Lọc theo khoảng thời gian trong ngày
            if st.session_state.time_range != "Toàn bộ ngày":
                if st.session_state.time_range == "Sáng (00:00-12:00)":
                    start_time = datetime.combine(selected_date, datetime.strptime("00:00", "%H:%M").time())
                    end_time = datetime.combine(selected_date, datetime.strptime("12:00", "%H:%M").time())
                elif st.session_state.time_range == "Chiều (12:00-18:00)":
                    start_time = datetime.combine(selected_date, datetime.strptime("12:00", "%H:%M").time())
                    end_time = datetime.combine(selected_date, datetime.strptime("18:00", "%H:%M").time())
                elif st.session_state.time_range == "Tối (18:00-24:00)":
                    start_time = datetime.combine(selected_date, datetime.strptime("18:00", "%H:%M").time())
                    end_time = datetime.combine(selected_date, datetime.strptime("23:59", "%H:%M").time())
                elif st.session_state.time_range == "Tùy chỉnh":
                    start_time = datetime.combine(selected_date, start_hour)
                    end_time = datetime.combine(selected_date, end_hour)
                
                mask_time = (df_filtered[time_column] >= start_time) & (df_filtered[time_column] <= end_time)
                df_filtered = df_filtered[mask_time]
            
            # Sắp xếp theo thời gian mới nhất trước
            df_filtered = df_filtered.sort_values(by=time_column, ascending=False)
            
        except Exception as e:
            st.warning(f"Không thể lọc theo thời gian: {e}")
            df_filtered = df
    else:
        df_filtered = df
        st.warning("Không tìm thấy cột thời gian để lọc")

    # === Hiển thị metrics ===
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
        
        # Tổng số alerts đã lọc
        total_alerts = len(df_filtered)
        col1.metric("Số alerts đang được hiển thị", f"{total_alerts}/200")
        
        # Phân loại alerts
        if "Label" in df_filtered.columns:
            benign_count = len(df_filtered[df_filtered["Label"] == "Benign"])
            attack_count = len(df_filtered[df_filtered["Label"] != "Benign"])
            
            # Kiểm tra cột Unknown
            unknown_mask = df_filtered.get('Label', pd.Series([])) == "Unknown"
            unknown_count = unknown_mask.sum() if isinstance(unknown_mask, pd.Series) else 0
            
            col2.metric("🟢 Benign", benign_count)
            col3.metric("🔴 Attack", attack_count - unknown_count)
            col4.metric("⚪ Unknown", unknown_count)
        else:
            col2.metric("🟢 Benign", 0)
            col3.metric("🔴 Attack", 0)
            col4.metric("⚪ Unknown", 0)
        
        # Thời gian mới nhất
        if time_column and not df_filtered.empty and time_column in df_filtered.columns:
            latest_time = df_filtered.iloc[0][time_column]
            if isinstance(latest_time, pd.Timestamp):
                latest_time_str = latest_time.strftime("%H:%M:%S")
            else:
                latest_time_str = str(latest_time)
            col5.metric("🕒 Mới nhất", latest_time_str)
        else:
            col5.metric("🕒 Mới nhất", "N/A")
    
    st.markdown("---")

    # === Hiển thị bảng dữ liệu ===
    if not df_filtered.empty:
        # Hiển thị số dòng đã lọc
        st.caption(f"Hiển thị {len(df_filtered)}/{len(df)} dòng dữ liệu")
        
        # Bảng dữ liệu
        st.dataframe(
            df_filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                time_column: st.column_config.DatetimeColumn(
                    "Thời gian", 
                    format="DD/MM/YYYY HH:mm:ss"
                ) if time_column and time_column in df_filtered.columns else None
            }
        )
        
        # Thống kê chi tiết
        with st.expander("Thống kê chi tiết", expanded=False):
            tab1, tab2, tab3 = st.tabs(["Phân bổ theo giờ", "Top cảnh báo", "Tóm tắt"])
            
            with tab1:
                if time_column and time_column in df_filtered.columns:
                    try:
                        df_filtered['hour'] = df_filtered[time_column].dt.hour
                        hourly_counts = df_filtered['hour'].value_counts().sort_index()
                        
                        # Hiển thị biểu đồ và bảng
                        col_chart1, col_chart2 = st.columns([3, 1])
                        with col_chart1:
                            st.bar_chart(hourly_counts)
                        with col_chart2:
                            st.dataframe(hourly_counts, use_container_width=True)
                    except Exception as e:
                        st.error(f"Không thể tạo biểu đồ: {e}")
                else:
                    st.info("Không có dữ liệu thời gian để phân tích")
            
            with tab2:
                # Tìm cột chứa loại cảnh báo
                alert_type_cols = ['Alert Type', 'Type', 'Category', 'Signature']
                found_col = None
                
                for col in alert_type_cols:
                    if col in df_filtered.columns:
                        found_col = col
                        break
                
                if found_col:
                    top_alerts = df_filtered[found_col].value_counts().head(10)
                    
                    # Hiển thị biểu đồ và bảng
                    col_top1, col_top2 = st.columns([3, 1])
                    with col_top1:
                        st.bar_chart(top_alerts)
                    with col_top2:
                        st.dataframe(top_alerts, use_container_width=True)
                else:
                    st.info("Không tìm thấy cột loại cảnh báo")
            
            with tab3:
                # Tóm tắt thống kê
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                with col_sum1:
                    st.metric("Tổng số dòng", len(df_filtered))
                    if time_column and time_column in df_filtered.columns:
                        try:
                            time_range = df_filtered[time_column].max() - df_filtered[time_column].min()
                            st.metric("⏱️ Khoảng thời gian", f"{time_range.total_seconds()/3600:.1f} giờ")
                        except:
                            pass
                
                with col_sum2:
                    if "Label" in df_filtered.columns:
                        labels = df_filtered["Label"].unique()
                        st.metric("Số loại nhãn", len(labels))
                        st.write("Các nhãn:", ", ".join([str(l) for l in labels[:5]]))
                
                with col_sum3:
                    if "Severity" in df_filtered.columns:
                        severity_counts = df_filtered["Severity"].value_counts()
                        st.write("<b> <i class='bi bi-exclamation-square'></i> Mức độ nghiêm trọng:</b>")
                        for severity, count in severity_counts.items():
                            st.write(f"- {severity}: {count}")
    else:
        st.warning(f"⚠️ Không có dữ liệu cảnh báo nào cho ngày {selected_date.strftime('%d/%m/%Y')} trong khoảng thời gian đã chọn.")
        
        # Hiển thị dữ liệu gốc nếu không có dữ liệu đã lọc
        with st.expander("📂 Xem tất cả dữ liệu có sẵn", expanded=False):
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

    st.session_state.data_loader = data_loader
    # Auto rerun nếu bật auto refresh
    if st.session_state.auto_refresh:
        time.sleep(1)
        st.rerun()

# if __name__ == "__main__":
#     main()