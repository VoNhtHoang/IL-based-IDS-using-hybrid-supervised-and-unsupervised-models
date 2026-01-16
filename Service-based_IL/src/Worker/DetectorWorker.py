
import os, sys
import threading
import queue
from io import StringIO


# 3rd import
import zmq, time
import joblib
import json
import gc
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# Local IMPORT
from src.config.settings import settings
from src.Utils.FlowFlushTransform import FlowFlushTransformer, STANDARD_COLS, STANDARD_SCALER_PATH, MINMAX_SCALER_PATH, MINMAX_COLS, COLS_TO_DROP

drops_cols = ["Flow ID", "Timestamp", "Label", "Binary Label"]

class DetectorWorker(threading.Thread):
    def __init__(
        self,
        worker_id,
        task_queue,
        log_queue,
        pipeline,
        header,
        flowFlushTransformer = None,
        alert_pub_addr="tcp://127.0.0.1:5570"
    ):
        super().__init__(daemon=True)

        # ===== ZMQ =====
        # self.ctx = zmq.Context()
        # self.alert_pub = self.ctx.socket(zmq.PUSH)
        # self.alert_pub.connect(alert_pub_addr)

        # ===== WORKER =====
        self.worker_id = worker_id
        self.q = task_queue
        self.log_queue = log_queue
        self.pipeline = pipeline
        
        # self.save_dir = Path(settings.ALERTS_DIR)
        # if not  Path.exists(self.save_dir):
        #     Path.mkdir(self.save_dir, exist_ok= True)
        
        # HEADER
        self.header = header
        
        self.flowFlushTransformer = FlowFlushTransformer(
            MINMAX_SCALER_PATH,
            STANDARD_SCALER_PATH,
            MINMAX_COLS,
            STANDARD_COLS,
            header=header,
            decimal_bin=6
        )
        
        self.running = True

    def run(self):
        print(f"[Detector-{self.worker_id}] started")

        try:
            while self.running:
                try:
                    # starttime = time.perf_counter()
                    df = self.q.get(timeout=0.2)
                    df = self.quick_parse(df)
                    
                    # df = df.decode("utf-8")
                    # df = pd.read_csv(
                    #                 StringIO(df), 
                    #                 sep=',', 
                    #                 engine='c', 
                    #                 header = None,
                    #                 names = self.header,
                    #                 # dtype=self.dtype_spec,
                    #                 na_filter=False, # Tắt kiểm tra NaN để tăng tốc nếu dữ liệu sạch
                    #                 low_memory=False)
                    
                    if df.empty:
                        continue
                    
                    # t1 = time.perf_counter() - starttime
                    
                    indexdf, X, _ = self.flowFlushTransformer.detect(df)
                    
                    # t2 = time.perf_counter() - (starttime + t1)
                    
                    res = self.pipeline.simple_predict(X)  
                    
                    # t3 = time.perf_counter() - (starttime + t1 + t2)
                    
                    self.save_alert(indexdf, res)
                    
                    # t4 = time.perf_counter() - (starttime + t1 +t2 +t3)
                    
                    self.q.task_done()

                    gc.collect()
                    
                    # print(f"[Detector-{self.worker_id}] Parse: {t1:.4f}s | Transform: {t2:.4f}s | Predict: {t3:.4f}s | Push Alert: {t4:.4f}s" )
                except queue.Empty:
                    continue
                
                except Exception as e:
                    print(f"[Detector-{self.worker_id}] error:", e)

        finally:
            # 🔥 ĐẢM BẢO ZMQ ĐƯỢC ĐÓNG
            # self._close_zmq()
            print(f"[Detector-{self.worker_id}] exited cleanly")
        

    def stop(self):
        """
        Báo hiệu worker dừng lại.
        KHÔNG đóng socket ở đây.
        """
        self.running = False
    
    def quick_parse(self, raw_bytes):
        # Tách dòng và chuyển thành mảng numpy nhanh hơn parser CSV
        lines = raw_bytes.decode('utf-8').strip().split('\n')
        data = [l.split(',') for l in lines]
        return pd.DataFrame(data, columns=self.header) # .apply(pd.to_numeric, errors='ignore')

    def save_alert(self, indexdf, res, X= None):
        indexdf["Label"] = res["predictions"]
        
        end_time_ms = int(time.time() * 1000)
        
        try:
            
            ts_numeric = pd.to_numeric(indexdf["Timestamp"], errors='coerce')
            indexdf["Pipetime"] = end_time_ms - ts_numeric
            
        except Exception as e:
            print(f"[Logger-Error] Tính Pipetime thất bại: {e}")
            indexdf["Pipetime"] = 0 
        
        self.log_queue.put(indexdf)
        # print(f"[Detector-{self.worker_id}] pushed indexdf" )