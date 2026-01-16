import psutil
import time
import os
import csv
from datetime import datetime


KEYWORDS = ["cic.cs.unb.ca", "FlowZmqServer", "java"]
# LOG_FILE = "src/Monitor/275kpps-20s.csv"
LOG_FILE = f"src/Monitor/{datetime.now().date()}.csv"

def get_total_metrics():
    total_ram = 0.0
    total_cpu = 0.0
    total_read_bytes = 0
    total_write_bytes = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(proc.info['cmdline'] or [])
            if any(key in cmdline for key in KEYWORDS):
                # Cộng dồn RAM (RSS)
                total_ram += proc.memory_info().rss / (1024 * 1024)
                # Cộng dồn CPU (%)
                total_cpu += proc.cpu_percent(interval=None)
                # Cộng dồn I/O
                io = proc.io_counters()
                total_read_bytes += io.read_bytes
                total_write_bytes += io.write_bytes
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue
            
    return total_cpu, total_ram, total_read_bytes, total_write_bytes

def monitor(interval=2):
    print(f"--- Đang theo dõi TỔNG TÀI NGUYÊN (Ghi vào {LOG_FILE}) ---")
    
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Total_CPU_%", "Total_RAM_MB", "Disk_Read_MB/s", "Disk_Write_MB/s"])

        # Biến để tính tốc độ Disk IO
        _, _, last_read, last_write = get_total_metrics()
        last_time = time.time()

        try:
            while True:
                time.sleep(interval)
                now = time.time()
                dt = now - last_time
                
                cpu, ram, curr_read, curr_write = get_total_metrics()
                
                # Tính tốc độ MB/s
                read_speed = (curr_read - last_read) / (1024 * 1024) / dt
                write_speed = (curr_write - last_write) / (1024 * 1024) / dt
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                writer.writerow([timestamp, f"{cpu:.1f}", f"{ram:.1f}", f"{read_speed:.2f}", f"{write_speed:.2f}"])
                f.flush()

                print(f"[{timestamp}] CPU: {cpu:.1f}% | RAM: {ram:.1f} MB | Disk W: {write_speed:.2f} MB/s")

                last_read, last_write, last_time = curr_read, curr_write, now
                
        except KeyboardInterrupt:
            print("\nĐã dừng ghi log.")

if __name__ == "__main__":
    # Chạy lần đầu để lấy mốc CPU ổn định
    psutil.cpu_percent(interval=1)
    monitor(interval=2) 