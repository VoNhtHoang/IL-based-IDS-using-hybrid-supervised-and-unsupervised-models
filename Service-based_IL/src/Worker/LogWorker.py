#
# 1rd libs
import os, sys
import threading
import queue


# 3rd libs
import zmq, time
import sqlite3
import json
import gc
from datetime import datetime
from pathlib import Path

# ===== ALGO ===== 
import pandas as pd


# Local Import
from src.config.settings import settings

class LogWorker(threading.Thread):
    def __init__(self, worker_id, log_queue):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.save_dir = Path(settings.ALERTS_DIR)
        self.log_queue = log_queue
        
        self.save_dir = settings.ALERTS_DIR
        if not  Path.exists(self.save_dir):
            Path.mkdir(self.save_dir, exist_ok= True)
        
        # self.log_file = self.save_dir / f"{datetime.now().date()}.jsonl"
        # Path(self.log_file).touch(exist_ok=True)
        
        self.running = True
    def run(self):
        print(f"[Logger-{self.worker_id}] started")
        
        try:
            while self.running:
                try:
                    df = self.log_queue.get(timeout=0.1)
                    
                    self.save_log(df)
                    self.log_queue.task_done()
                    gc.collect()

                except queue.Empty:
                    continue

                except Exception as e:
                    print(f"[Logger-{self.worker_id}] error:", e)
        
        finally:
            print(f"[Logger-{self.worker_id}] exited cleanly")
            
    def stop(self):
        """
        Báo hiệu worker dừng lại.
        KHÔNG đóng socket ở đây.
        """
        self.running = False        
        return
    
                
    def save_log(self, indexdf):
        log_file = self.save_dir / f"{datetime.now().date()}.jsonl"
        # log_file = self.save_dir / f"275kpps-20s.jsonl"

        Path(log_file).touch(exist_ok=True)
        timestamp_numeric = pd.to_numeric(indexdf['Timestamp'], errors='coerce')
        # indexdf['Timestamp'] = pd.to_datetime(timestamp_numeric, unit='ms')
        # indexdf['Timestamp'] = indexdf['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        indexdf['Timestamp'] = (pd.to_datetime(timestamp_numeric, unit='ms')
                           .dt.tz_localize('UTC') # Xác định đây là giờ gốc
                           .dt.tz_convert('Asia/Ho_Chi_Minh') # Chuyển sang giờ VN
                           .dt.strftime('%Y-%m-%d %H:%M:%S'))
        
        indexdf.to_json(
            log_file, 
            orient='records', 
            lines=True,
            mode='a', 
            force_ascii=False
        )
        print(f"[Logger-{self.worker_id}] Logged {len(indexdf)} alerts → {log_file.name}")  