# pages/Resource_Monitor.py
import os
import time
import subprocess
import psutil
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 3rd libs
import streamlit as st

# Services to monitor
SERVICES = [
    "ids_cicextract.service",
    "ids_flowzmqserver.service", 
    "ids_dashboard.service"
]

IL_SERVICES = [
    "ids_il.timer",
    "ids_il.service"
]

ALL_SERVICES = SERVICES + IL_SERVICES

# Color mapping for services
SERVICE_COLORS = {
    "ids_cicextract.service": "#FF6B6B",
    "ids_flowzmqserver.service": "#4ECDC4", 
    "ids_dashboard.service": "#45B7D1",
    "ids_il.timer": "#96CEB4",
    "ids_il.service": "#FFEAA7"
}

# ======================
# SYSTEM FUNCTIONS
# ======================

@st.cache_data(ttl=5)
def get_service_status(service_name):
    """
    Get systemd service status using systemctl
    """
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', service_name],
            capture_output=True,
            text=True,
            timeout=2
        )
        status = result.stdout.strip()
        return status if status in ['active', 'inactive', 'failed', 'activating'] else 'unknown'
    except subprocess.TimeoutExpired:
        return 'timeout'
    except Exception as e:
        return f'error: {str(e)}'


def get_service_detailed_resources(service_name):
    """
    Lấy tổng CPU/RAM của service bằng cách quét PID chính và các tiến trình con
    """
    total_cpu = 0.0
    total_mem = 0.0
    
    try:
        # 1. Lấy MainPID từ systemctl
        cmd = f"systemctl show {service_name} --property=MainPID"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        main_pid = int(result.stdout.split('=')[1].strip())

        if main_pid > 0:
            # 2. Dùng psutil lấy tất cả tiến trình con (recursive=True)
            parent = psutil.Process(main_pid)
            children = parent.children(recursive=True)
            all_procs = [parent] + children

            for p in all_procs:
                try:
                    # Lưu ý: Không dùng interval > 0 ở đây để tránh treo Streamlit
                    total_cpu += p.cpu_percent(interval=None)
                    total_mem += p.memory_info().rss / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
    except Exception:
        pass
        
    return total_cpu, total_mem

@st.cache_data(ttl=5)
def get_service_info(service_name):
    """
    Get detailed service information
    """
    info = {
        'name': service_name,
        'status': 'unknown',
        'loaded': False,
        'active': False,
        'sub_state': '',
        'description': '',
        'pid': None,
        'memory_mb': 0,
        'cpu_percent': 0,
        'uptime': '',
        'last_check': datetime.now()
    }
    
    try:
        # Get systemctl status
        result = subprocess.run(
            ['systemctl', 'show', service_name, '--property=LoadState,ActiveState,SubState,Description,MainPID'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if '=' in line:
                    key, value = line.split('=', 1)
                    if key == 'LoadState':
                        info['loaded'] = (value == 'loaded')
                    elif key == 'ActiveState':
                        info['active'] = (value == 'active')
                        info['status'] = value
                    elif key == 'SubState':
                        info['sub_state'] = value
                    elif key == 'Description':
                        info['description'] = value
                    elif key == 'MainPID':
                        if value.isdigit() and int(value) > 0:
                            info['pid'] = int(value)
        
        # Get active status
        info['status'] = get_service_status(service_name)
        
        # THAY THẾ ĐOẠN LẤY RESOURCE CŨ BẰNG LOGIC MỚI:
        cpu_val, mem_val = get_service_detailed_resources(service_name)
        info['cpu_percent'] = cpu_val
        info['memory_mb'] = mem_val
        
        # Lấy lại PID chính để hiển thị UI
        try:
            res = subprocess.run(['systemctl', 'show', service_name, '--property=MainPID'], 
                                capture_output=True, text=True)
            pid_str = res.stdout.strip().split('=')[1]
            info['pid'] = int(pid_str) if pid_str.isdigit() and int(pid_str) > 0 else None
        except:
            info['pid'] = None
            
        # Get resource usage if PID exists
    #     if info['pid']:
    #         try:
    #             process = psutil.Process(info['pid'])
    #             with process.oneshot():
    #                 info['memory_mb'] = process.memory_info().rss / 1024 / 1024
    #                 info['cpu_percent'] = process.cpu_percent(interval=0.1)
                    
    #                 # Calculate uptime
    #                 create_time = datetime.fromtimestamp(process.create_time())
    #                 uptime = datetime.now() - create_time
                    
    #                 if uptime.days > 0:
    #                     info['uptime'] = f"{uptime.days}d {uptime.seconds//3600}h"
    #                 elif uptime.seconds >= 3600:
    #                     info['uptime'] = f"{uptime.seconds//3600}h {(uptime.seconds%3600)//60}m"
    #                 else:
    #                     info['uptime'] = f"{uptime.seconds//60}m {uptime.seconds%60}s"
                        
    #         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
    #             info['pid'] = None
        
    #     # Get service uptime from systemctl
    #     try:
    #         result = subprocess.run(
    #             ['systemctl', 'show', service_name, '--property=ActiveEnterTimestamp'],
    #             capture_output=True,
    #             text=True,
    #             timeout=1
    #         )
    #         if result.returncode == 0:
    #             for line in result.stdout.strip().split('\n'):
    #                 if line.startswith('ActiveEnterTimestamp='):
    #                     timestamp_str = line.split('=', 1)[1]
    #                     if timestamp_str:
    #                         # Parse systemd timestamp (e.g., "Thu 2024-01-01 12:00:00 UTC")
    #                         try:
    #                             ts = datetime.strptime(timestamp_str, '%a %Y-%m-%d %H:%M:%S %Z')
    #                             uptime = datetime.now() - ts
    #                             if not info['uptime']:  # Only use if we don't have process uptime
    #                                 if uptime.days > 0:
    #                                     info['uptime'] = f"{uptime.days}d {uptime.seconds//3600}h"
    #                                 elif uptime.seconds >= 3600:
    #                                     info['uptime'] = f"{uptime.seconds//3600}h {(uptime.seconds%3600)//60}m"
    #                                 else:
    #                                     info['uptime'] = f"{uptime.seconds//60}m {uptime.seconds%60}s"
    #                         except ValueError:
    #                             pass
    #     except:
    #         pass
            
    except Exception as e:
        st.error(f"Error getting info for {service_name}: {str(e)}")
    
    return info

@st.cache_data(ttl=5)
def get_system_resources():
    """
    Get overall system resource usage
    """
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network I/O
        net_io = psutil.net_io_counters()
        
        # System load average
        load_avg = psutil.getloadavg()
        
        # System uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        system_uptime = datetime.now() - boot_time
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / 1024**3,
            'memory_total_gb': memory.total / 1024**3,
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / 1024**3,
            'disk_total_gb': disk.total / 1024**3,
            'bytes_sent_mb': net_io.bytes_sent / 1024**2,
            'bytes_recv_mb': net_io.bytes_recv / 1024**2,
            'load_avg_1min': load_avg[0],
            'load_avg_5min': load_avg[1],
            'load_avg_15min': load_avg[2],
            'system_uptime': system_uptime,
            'timestamp': datetime.now()
        }
    except Exception as e:
        st.error(f"Error getting system resources: {str(e)}")
        return None

# ======================
# HISTORY TRACKING
# ======================

def init_history():
    """
    Initialize history tracking in session state
    """
    if 'resource_history' not in st.session_state:
        st.session_state.resource_history = {
            'timestamps': [],
            'system': {
                'cpu_percent': [],
                'memory_percent': [],
                'disk_percent': []
            },
            'services': {}
        }
    
    # Initialize service history
    for service in ALL_SERVICES:
        if service not in st.session_state.resource_history['services']:
            st.session_state.resource_history['services'][service] = {
                'cpu_percent': [],
                'memory_mb': [],
                'status': []
            }

def update_history(services_info, system_info, max_points=60):
    """
    Update history with current readings
    """
    init_history()
    
    timestamp = datetime.now()
    
    # Add timestamp
    st.session_state.resource_history['timestamps'].append(timestamp)
    
    # Add system data
    if system_info:
        st.session_state.resource_history['system']['cpu_percent'].append(system_info['cpu_percent'])
        st.session_state.resource_history['system']['memory_percent'].append(system_info['memory_percent'])
        st.session_state.resource_history['system']['disk_percent'].append(system_info['disk_percent'])
    
    # Add service data
    for service_info in services_info:
        service_name = service_info['name']
        
        if service_name in st.session_state.resource_history['services']:
            # Convert status to numeric for plotting
            status_map = {'active': 1, 'inactive': 0, 'failed': -1, 'activating': 0.5}
            status_value = status_map.get(service_info['status'], 0)
            
            st.session_state.resource_history['services'][service_name]['cpu_percent'].append(
                service_info['cpu_percent']
            )
            st.session_state.resource_history['services'][service_name]['memory_mb'].append(
                service_info['memory_mb']
            )
            st.session_state.resource_history['services'][service_name]['status'].append(
                status_value
            )
    
    # Trim history to max_points
    if len(st.session_state.resource_history['timestamps']) > max_points:
        remove_count = len(st.session_state.resource_history['timestamps']) - max_points
        st.session_state.resource_history['timestamps'] = st.session_state.resource_history['timestamps'][remove_count:]
        
        for key in st.session_state.resource_history['system']:
            st.session_state.resource_history['system'][key] = st.session_state.resource_history['system'][key][remove_count:]
        
        for service in st.session_state.resource_history['services']:
            for key in st.session_state.resource_history['services'][service]:
                st.session_state.resource_history['services'][service][key] = \
                    st.session_state.resource_history['services'][service][key][remove_count:]

# ======================
# VISUALIZATION FUNCTIONS
# ======================

def create_system_metrics_chart(system_info):
    """
    Create gauge charts for system metrics
    """
    if not system_info:
        return None
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage'),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    # CPU Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=system_info['cpu_percent'],
            title={'text': f"CPU<br>{system_info['cpu_percent']:.1f}%"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=1
    )
    
    # Memory Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=system_info['memory_percent'],
            title={'text': f"Memory<br>{system_info['memory_used_gb']:.1f}/{system_info['memory_total_gb']:.1f} GB"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgreen"},
                    {'range': [60, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=2
    )
    
    # Disk Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=system_info['disk_percent'],
            title={'text': f"Disk<br>{system_info['disk_used_gb']:.1f}/{system_info['disk_total_gb']:.1f} GB"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgreen"},
                    {'range': [70, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_resource_trend_chart():
    """
    Create trend charts for resource usage over time
    """
    if not st.session_state.resource_history['timestamps']:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU Usage Trend', 'Memory Usage Trend', 
                       'Service CPU Usage', 'Service Memory Usage'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # System CPU Trend
    fig.add_trace(
        go.Scatter(
            x=st.session_state.resource_history['timestamps'],
            y=st.session_state.resource_history['system']['cpu_percent'],
            mode='lines',
            name='System CPU',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # System Memory Trend
    fig.add_trace(
        go.Scatter(
            x=st.session_state.resource_history['timestamps'],
            y=st.session_state.resource_history['system']['memory_percent'],
            mode='lines',
            name='System Memory',
            line=dict(color='green', width=2)
        ),
        row=1, col=2
    )
    
    # Service CPU Usage
    for service_name in SERVICES:
        if service_name in st.session_state.resource_history['services']:
            history = st.session_state.resource_history['services'][service_name]
            if history['cpu_percent']:
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.resource_history['timestamps'],
                        y=history['cpu_percent'],
                        mode='lines',
                        name=service_name,
                        line=dict(color=SERVICE_COLORS.get(service_name, 'gray'), width=1.5)
                    ),
                    row=2, col=1
                )
    
    # Service Memory Usage
    for service_name in SERVICES:
        if service_name in st.session_state.resource_history['services']:
            history = st.session_state.resource_history['services'][service_name]
            if history['memory_mb']:
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.resource_history['timestamps'],
                        y=history['memory_mb'],
                        mode='lines',
                        name=service_name,
                        line=dict(color=SERVICE_COLORS.get(service_name, 'gray'), width=1.5),
                        showlegend=False
                    ),
                    row=2, col=2
                )
    
    # Update layout
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
    fig.update_yaxes(title_text="CPU %", row=2, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=2, col=2)
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    
    fig.update_layout(
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_service_status_chart(services_info):
    """
    Create bar chart for service status
    """
    status_df = pd.DataFrame(services_info)
    
    # Map status to colors
    status_colors = {
        'active': '#28a745',
        'inactive': '#6c757d', 
        'failed': '#dc3545',
        'activating': '#ffc107',
        'unknown': '#6c757d'
    }
    
    fig = px.bar(
        status_df,
        x='name',
        y='memory_mb',
        color='status',
        color_discrete_map=status_colors,
        hover_data=['description', 'cpu_percent', 'uptime'],
        labels={
            'name': 'Service',
            'memory_mb': 'Memory Usage (MB)',
            'status': 'Status'
        },
        title='Service Status & Memory Usage'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="",
        yaxis_title="Memory Usage (MB)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ======================
# CONTROL FUNCTIONS
# ======================

def control_service(service_name, action):
    """
    Control systemd service (start, stop, restart)
    """
    try:
        if action not in ['start', 'stop', 'restart']:
            return False, "Invalid action"
        
        result = subprocess.run(
            ['sudo', 'systemctl', action, service_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return True, f"Service {service_name} {action}ed successfully"
        else:
            return False, f"Failed to {action} {service_name}: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout while trying to {action} {service_name}"
    except Exception as e:
        return False, f"Error: {str(e)}"

# ======================
# MAIN FUNCTION
# ======================

def main():
    st.set_page_config(
        page_title="Resource Monitor - IDS Services",
        # page_icon="📈",
        layout="wide"
    )
    
    st.markdown("## <i class='bi bi-graph-up'></i> Resource Monitor - IDS Services", unsafe_allow_html=True)
    st.markdown("**Theo dõi mức độ tiêu tốn tài nguyên của các dịch vụ IDS**")
    
    # Initialize session state
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5
    
    if 'show_history' not in st.session_state:
        st.session_state.show_history = True
    
    init_history()
    
    # === Control Panel ===
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            auto_refresh = st.checkbox(
                "⟳ Auto Refresh", 
                value=st.session_state.auto_refresh,
                key="auto_refresh_check"
            )
            st.session_state.auto_refresh = auto_refresh
        
        with col2:
            if auto_refresh:
                interval = st.selectbox(
                    "Refresh Interval (seconds)",
                    [1, 2, 5, 10, 30, 60],
                    index=[1, 2, 5, 10, 30, 60].index(st.session_state.refresh_interval),
                    key="refresh_interval_select"
                )
                st.session_state.refresh_interval = interval
        
        with col3:
            if st.button("⟳ Refresh Now", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col4:
            show_history = st.checkbox(
                "⏲︎ Show History",
                value=st.session_state.show_history,
                key="show_history_check"
            )
            st.session_state.show_history = show_history
    
    st.markdown("---")
    
    # === Get Current Data ===
    with st.spinner("⟳ Đang thu thập dữ liệu hệ thống..."):
        # Get system resources
        system_info = get_system_resources()
        
        # Get service information
        services_info = []
        for service in ALL_SERVICES:
            service_info = get_service_info(service)
            services_info.append(service_info)
        
        # Update history
        if st.session_state.show_history:
            update_history(services_info, system_info)
    
    # === System Overview ===
    st.markdown("### <i class='bi bi-opencollective'></i> Tổng quan hệ thống", unsafe_allow_html= True)
    
    if system_info:
        # System metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CPU Usage",
                f"{system_info['cpu_percent']:.1f}%",
                delta=f"Load: {system_info['load_avg_1min']:.2f}"
            )
        
        with col2:
            st.metric(
                "Memory Usage",
                f"{system_info['memory_percent']:.1f}%",
                f"{system_info['memory_used_gb']:.1f}/{system_info['memory_total_gb']:.1f} GB"
            )
        
        with col3:
            st.metric(
                "Disk Usage",
                f"{system_info['disk_percent']:.1f}%",
                f"{system_info['disk_used_gb']:.1f}/{system_info['disk_total_gb']:.1f} GB"
            )
        
        with col4:
            # Format system uptime
            uptime = system_info['system_uptime']
            if uptime.days > 0:
                uptime_str = f"{uptime.days}d {uptime.seconds//3600}h"
            else:
                uptime_str = f"{uptime.seconds//3600}h {(uptime.seconds%3600)//60}m"
            
            st.metric(
                "System Uptime",
                uptime_str,
                f"Network: ↑{system_info['bytes_sent_mb']:.1f}MB ↓{system_info['bytes_recv_mb']:.1f}MB"
            )
    
    # === Service Status Table ===
    st.markdown("### <i class='bi bi-opencollective'></i> Trạng thái dịch vụ", unsafe_allow_html=True)
    
    # Split services into main services and IL services
    main_services = [s for s in services_info if s['name'] in SERVICES]
    il_services = [s for s in services_info if s['name'] in IL_SERVICES]
    
    # Display main services
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**Main Services:**")
        
        for service in main_services:
            status_color = {
                'active': '🟢',
                'inactive': '⚪',
                'failed': '🔴',
                'activating': '🟡',
                'unknown': '⚫'
            }.get(service['status'], '⚫')
            
            # Create columns for service info
            with st.container(border=True):
                service_col1, service_col2, service_col3, service_col4 = st.columns([3, 2, 2, 3])
            
                with service_col1:
                    st.markdown(f"**{service['name']}**")
                    st.caption(service['description'][:50] + "..." if len(service['description']) > 50 else service['description'])
                
                with service_col2:
                    st.markdown(f"{status_color} **{service['status'].upper()}**")
                    if service['uptime']:
                        st.caption(f"⏱️ {service['uptime']}")
                
                with service_col3:
                    if service['pid']:
                        st.markdown(f"**PID:** {service['pid']}")
                    else:
                        st.markdown("**PID:** N/A")
                
                with service_col4:
                    # Resource usage
                    if service['cpu_percent'] > 0 or service['memory_mb'] > 0:
                        st.markdown(f"**CPU:** {service['cpu_percent']:.1f}% | **Mem:** {service['memory_mb']:.1f} MB")
                    else:
                        st.markdown("**Resource:** N/A")
            
            # st.markdown("---")

    
        st.markdown("**IL Services:**")
        
        for service in il_services:
            status_color = {
                'active': '🟢',
                'inactive': '⚪',
                'failed': '🔴',
                'activating': '🟡',
                'unknown': '⚫'
            }.get(service['status'], '⚫')
            
            
            
            # st.markdown(f"**{service['name']}**")
            # st.markdown(f"{status_color} {service['status'].upper()}")
            
            # if service['pid']:
            #     st.caption(f"PID: {service['pid']}")
            
            # if service['cpu_percent'] > 0:
            #     st.caption(f"CPU: {service['cpu_percent']:.1f}%")
            
            # if service['memory_mb'] > 0:
            #     st.caption(f"Mem: {service['memory_mb']:.1f} MB")
            with st.container(border=True):
                service_col1, service_col2, service_col3, service_col4 = st.columns([3, 2, 2, 3])
                with service_col1:
                    st.markdown(f"**{service['name']}**")
                    st.caption(service['description'][:50] + "..." if len(service['description']) > 50 else service['description'])
                
                with service_col2:
                    st.markdown(f"{status_color} **{service['status'].upper()}**")
                    if service['uptime']:
                        st.caption(f"⏱️ {service['uptime']}")
                
                with service_col3:
                    if service['pid']:
                        st.markdown(f"**PID:** {service['pid']}")
                    else:
                        st.markdown("**PID:** N/A")
                
                with service_col4:
                    # Resource usage
                    if service['cpu_percent'] > 0 or service['memory_mb'] > 0:
                        st.markdown(f"**CPU:** {service['cpu_percent']:.1f}% | **Mem:** {service['memory_mb']:.1f} MB")
                    else:
                        st.markdown("**Resource:** N/A")
            
        st.markdown("---")
    
    with col2:        
        # Service controls
        st.markdown("** Service Controls:**")
        selected_service = st.selectbox(
            "Chọn service",
            [s['name'] for s in services_info],
            key="service_select"
        )
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("▶️ Start", use_container_width=True):
                success, message = control_service(selected_service, 'start')
                if success:
                    st.success(message)
                else:
                    st.error(message)
                time.sleep(1)
                st.rerun()
        
        with action_col2:
            if st.button("⏹️ Stop", use_container_width=True):
                success, message = control_service(selected_service, 'stop')
                if success:
                    st.success(message)
                else:
                    st.error(message)
                time.sleep(1)
                st.rerun()
        
        with action_col3:
            if st.button("🔄 Restart", use_container_width=True):
                success, message = control_service(selected_service, 'restart')
                if success:
                    st.success(message)
                else:
                    st.error(message)
                time.sleep(1)
                st.rerun()
    
    # === Visualizations ===
    st.markdown("### <i class='bi bi-graph-up'></i> Biểu đồ tài nguyên", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["System Metrics", "Resource Trends", "Service Status"])
    
    with tab1:
        if system_info:
            fig_system = create_system_metrics_chart(system_info)
            if fig_system:
                st.plotly_chart(fig_system, use_container_width=True)
        else:
            st.info("Không có dữ liệu hệ thống")
    
    with tab2:
        if st.session_state.show_history and st.session_state.resource_history['timestamps']:
            fig_trends = create_resource_trend_chart()
            if fig_trends:
                st.plotly_chart(fig_trends, use_container_width=True)
            else:
                st.info("Chưa có đủ dữ liệu lịch sử")
        else:
            st.info("Bật 'Show History' để xem biểu đồ xu hướng")
    
    with tab3:
        if services_info:
            fig_status = create_service_status_chart(main_services)
            if fig_status:
                st.plotly_chart(fig_status, use_container_width=True)
    
    # === Detailed Information ===
    with st.expander("✔ Thông tin chi tiết", expanded=False):
        # Create DataFrame for detailed view
        detail_df = pd.DataFrame(services_info)
        
        # Select columns to display
        display_columns = ['name', 'status', 'sub_state', 'pid', 'cpu_percent', 
                          'memory_mb', 'uptime', 'description', 'last_check']
        
        # Filter available columns
        available_columns = [col for col in display_columns if col in detail_df.columns]
        
        # Display dataframe
        st.dataframe(
            detail_df[available_columns],
            use_container_width=True,
            column_config={
                'name': st.column_config.TextColumn("Service", width="medium"),
                'status': st.column_config.TextColumn("Status", width="small"),
                'cpu_percent': st.column_config.NumberColumn("CPU %", format="%.1f"),
                'memory_mb': st.column_config.NumberColumn("Memory MB", format="%.1f"),
                'last_check': st.column_config.DatetimeColumn("Last Check")
            }
        )
        
        # Export option
        if st.button(" Export Data to CSV"):
            csv = detail_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"service_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # === Auto Refresh ===
    if auto_refresh:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()

# if __name__ == "__main__":
#     main()