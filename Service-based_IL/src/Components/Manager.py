# STANDARD LIBS
from datetime import datetime
from pathlib import Path

# 3RD LIBS
import joblib

# MINE
from src.config.settings import settings
from src.Components.Models import *

class Manager:
                
    def __init__(self, dir_in, dir_out =None, current_update_time = None):
        self.dir_in = None
        if self.check_dir(dir_in):
            self.dir_in = dir_in
        else:
            self.dir_in = self.get_full_path(dir_in)
            if self.dir_in is None:
                print("[ERROR] Manager.py - Dir không tồn tại!")
                exit(0)
        
        # if dir_out is None or current_update_time is None:
        #     return
        if dir_out is not None and current_update_time is not None:
            
            self.dir_out = dir_out
            self.dir_out.mkdir(parents=True, exist_ok = True)
            self.current_update_time= datetime.strftime(current_update_time, "%Y-%m-%d %H:%M:%S")
            self.current_update_time = self.current_update_time.replace(":", "-")
        
            self.dir_out = Path(self.dir_out/f"{self.current_update_time}")
        
    def check_dir(self, dir_in):
        dir_in = Path(dir_in)
        if dir_in.exists():
            return True
        
        if dir_in.is_dir():
            return False
        
        return False
    
    def get_full_path(self, dir_in) -> None:
        dir_in = Path.cwd() / dir_in
        if Path.exists(dir_in):
            print("[DEBUG] Manager.py - Model path is: ", dir_in)
            return dir_in
        
        return None
    
    def load_models(self, models: list):
        dir_in = Path(self.dir_in)
        # xgb.load_model(dir_in/xgb.model_name)
        # ocsvm.load_model(dir_in/ocsvm.model_name)
        # ae.load_model(dir_in/ae.model_name)
        
        # làm kiểu này giá trị model mới thay đổi được, tham chiếu
        for index in range(len(models)):
            models[index].load_model(dir_in/models[index].model_name)
        print("[MANAGER] Models Loaded Successfully")
    
    # dành cho UI
    def load_models_info(self, models: list):
        dir_in = Path(self.dir_in)
        res_info = {}
        
        for index in range(len(models)):
            res_info[f'{models[index].model_name}'] = models[index].load_model_info(dir_in/models[index].model_name)
            # print(f"[MANAGER] {dir_in/models[index].model_name}")
        print("[MANAGER] Models Info Loaded!")
        
        return res_info
        
        
    def save_models(self, models: list, version=0):
        for index in range(len(models)):
            models[index].save_model(self.dir_out/models[index].model_name, version, self.current_update_time)
        print("[MANAGER] Models Saved Successfully")
    
    
        
        
        
    
        
