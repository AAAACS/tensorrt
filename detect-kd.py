import tensorrt as trt
import torch
import cv2
import numpy as np
from datetime import datetime
import os
from torchvision import transforms
import threading
import queue
import time

# 全局变量定义
PROCESS_WIDTH = 224
PROCESS_HEIGHT = 224
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 添加重试机制和资源管理
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(engine_path, 'rb') as f:
                    runtime = trt.Runtime(self.logger)
                    engine_data = f.read()
                    self.engine = runtime.deserialize_cuda_engine(engine_data)
                    
                if self.engine is None:
                    raise RuntimeError("Failed to create engine")
                    
                self.context = self.engine.create_execution_context()
                self.stream = torch.cuda.Stream()
                
                # 分配 GPU 内存
                self.inputs = []
                self.outputs = []
                self.allocate_buffers()
                
                # 预处理转换
                self.preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                # 创建缓存以重用张量
                self.input_tensor = torch.empty((1, 3, PROCESS_HEIGHT, PROCESS_WIDTH), 
                                             dtype=torch.float32, device='cuda')
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                torch.cuda.empty_cache()
                time.sleep(1)
    
    def __del__(self):
        # 确保资源被正确释放
        if hasattr(self, 'context'):
            self.context = None
        if hasattr(self, 'engine'):
            self.engine = None
        torch.cuda.empty_cache()
    
    def allocate_buffers(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            device_mem = torch.empty(size, dtype=torch.float32, device='cuda')
            self.inputs.append(device_mem) if self.engine.binding_is_input(binding) else self.outputs.append(device_mem)
    
    def infer(self, frame):
        try:
            # 预处理
            small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # 重用预分配的张量
            with torch.no_grad(), torch.cuda.stream(self.stream):
                input_tensor = self.preprocess(small_frame_rgb)
                self.input_tensor[0].copy_(input_tensor)
                self.inputs[0].copy_(self.input_tensor.view(-1))
                
                # 执行推理
                self.context.execute_v2(bindings=[inp.data_ptr() for inp in self.inputs + self.outputs])
                
                # 处理输出
                output = self.outputs[0].cpu().numpy().reshape(1, -1)
                probabilities = torch.softmax(torch.tensor(output[0]), dim=0)
                top_prob, top_catid = torch.max(probabilities, 0)
            
            return top_prob.item(), top_catid.item()
        except Exception as e:
            print(f"Inference error: {e}")
            return 0.0, 0

class VideoProcessor:
    def __init__(self, url, model):
        self.url = url
        self.model = model
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False
        
    def capture_frames(self):
        # 设置 RTSP 相关参数
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        retry_count = 0
        max_retries = 3
        frame_delay = 1.0/30  # 固定帧率
        
        while self.running:
            try:
                if not cap.isOpened():
                    if retry_count < max_retries:
                        print("Reconnecting to camera...")
                        cap.release()
                        time.sleep(1)
                        cap = cv2.VideoCapture(self.url)
                        retry_count += 1
                        continue
                    else:
                        print("Failed to reconnect after multiple attempts")
                        break
                
                ret, frame = cap.read()
                if ret:
                    retry_count = 0  # 重置重试计数
                    if self.frame_queue.full():
                        self.frame_queue.get()
                    self.frame_queue.put(frame)
                    time.sleep(frame_delay)
                else:
                    if retry_count < max_retries:
                        print("Stream error, attempting to reconnect...")
                        cap.release()
                        time.sleep(1)
                        cap = cv2.VideoCapture(self.url)
                        retry_count += 1
                    else:
                        print("Failed to recover stream after multiple attempts")
                        break
                        
            except Exception as e:
                print(f"Error in capture_frames: {e}")
                if retry_count < max_retries:
                    retry_count += 1
                    time.sleep(1)
                else:
                    break
        
        cap.release()
    
    def process_frames(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                top_prob, top_catid = self.model.infer(display_frame)
                
                if self.result_queue.full():
                    self.result_queue.get()
                self.result_queue.put((display_frame, top_prob, top_catid))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in process_frames: {e}")
                time.sleep(0.1)

def classify_video():
    while True:  # 外层循环用于错误恢复
        try:
            # 初始化 TensorRT 推理器
            model = TRTInference("model.engine")
            
            # 创建视频处理器
            processor = VideoProcessor('rtsp://admin:Jqrsys1214@192.168.1.60:554/11', model)
            processor.running = True
            
            # 创建输出目录
            today = datetime.now().strftime('%Y%m%d')
            os.makedirs(today, exist_ok=True)
            
            # 启动线程
            capture_thread = threading.Thread(target=processor.capture_frames)
            process_thread = threading.Thread(target=processor.process_frames)
            capture_thread.start()
            process_thread.start()
            
            consecutive_count = 0
            threshold = 0.99
            last_save_time = datetime.now()
            last_echo_time = None
            
            try:
                while True:
                    try:
                        display_frame, top_prob, top_catid = processor.result_queue.get(timeout=1)
                        
                        # 处理检测结果
                        current_time = datetime.now()
                        if top_catid in [1] and top_prob > threshold:
                            if (current_time - last_save_time).total_seconds() >= 1:
                                timestamp = current_time.strftime('%Y%m%d_%H%M%S')
                                screenshot_path = os.path.join(today, f'{timestamp}.jpg')
                                cv2.imwrite(screenshot_path, display_frame)
                                last_save_time = current_time
                        
                        if top_catid in [2, 3, 4] and top_prob > threshold:
                            signal_output = 'Alarm: 1'
                        else:
                            signal_output = 'Alarm: 0'
                        
                        # 处理连续检测逻辑
                        if top_catid == 1 and top_prob > threshold:
                            consecutive_count += 1
                        else:
                            consecutive_count = 0
                            
                        if consecutive_count >= 15:
                            os.system('echo 1 > /sys/class/gpio/gpio421/value')
                            last_echo_time = current_time
                            signal_output_2 = 'echo: 1'
                        elif last_echo_time and (current_time - last_echo_time).total_seconds() >= 5:
                            os.system('echo 0 > /sys/class/gpio/gpio421/value')
                            last_echo_time = None
                            signal_output_2 = 'echo: 0'
                        else:
                            signal_output_2 = 'echo: 1' if last_echo_time else 'echo: none'
                        
                        info_text = f'Class: {top_catid}, Prob: {top_prob:.2f}'
                        cv2.putText(display_frame, info_text, 
                                  (DISPLAY_WIDTH - 200, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(display_frame, signal_output, 
                                  (DISPLAY_WIDTH - 200, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(display_frame, signal_output_2, 
                                  (DISPLAY_WIDTH - 200, 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        cv2.imshow('Video Stream', display_frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
                    except queue.Empty:
                        continue
                    
            finally:
                # 确保程序退出时 echo 0
                os.system('echo 0 > /sys/class/gpio/gpio421/value')
                processor.running = False
                capture_thread.join()
                process_thread.join()
                cv2.destroyAllWindows()
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            print("Attempting to recover...")
            time.sleep(5)
            torch.cuda.empty_cache()
            continue
        finally:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        classify_video()
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        os.system('echo 0 > /sys/class/gpio/gpio421/value')
        cv2.destroyAllWindows()
        torch.cuda.empty_cache()
