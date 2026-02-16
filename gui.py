import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
from model import NanoSLM, device, block_size, batch_size, n_embd, n_head, n_layer
from tokenizer import Tokenizer
import os

class NanoSLMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("slmaker Engine v0.8.0 | Dual-Interface")
        self.root.geometry("1100x800")
        self.root.configure(bg='#1e1e1e')
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        # Font Fallback / 폰트 폴백 설정
        self.font_main = ("Helvetica", 10)
        self.font_header = ("Helvetica", 14, "bold")
        self.font_mono = ("Courier", 9)
        
        self.style.configure("TFrame", background="#1e1e1e")
        self.style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=self.font_main)
        self.style.configure("Header.TLabel", font=self.font_header, foreground="#00ffcc")
        self.style.configure("TButton", font=(self.font_main[0], 10, "bold"))
        
        self.log_queue = queue.Queue()
        self.data_queue = queue.Queue()
        self.is_training = False
        
        self._setup_ui()
        self._check_queues()

    def _setup_ui(self):
        # Header / 헤더
        header = ttk.Label(self.root, text="NANO-SLM TRAINING ENGINE v1.0", style="Header.TLabel")
        header.pack(pady=10)
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left Panel: Info and Control / 왼쪽 패널: 정보 및 제어
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Hardware Info / 하드웨어 및 모델 사양
        specs_group = tk.LabelFrame(left_panel, text=" SYSTEM SPECS / 시스템 사양 ", bg='#1e1e1e', fg='#00ffcc', font=("Inter", 10, "bold"))
        specs_group.pack(fill=tk.X, pady=10)
        
        specs = [
            ("Target Device:", "CPU (Optimized)"),
            ("Memory Limit:", "4GB RAM"),
            ("Model Type:", "Decoder-only Transformer"),
            ("Parameters:", "~1.2B"),
            ("Embed Dim:", str(n_embd)),
            ("Heads/Layers:", f"{n_head} / {n_layer}")
        ]
        
        for label, val in specs:
            row = ttk.Frame(specs_group)
            row.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(row, text=label, width=15).pack(side=tk.LEFT)
            # Display field: Read-only visual field / 장식용 읽기 전용 필드
            entry = tk.Entry(row, bg='#2d2d2d', fg='#ffffff', borderwidth=0, highlightthickness=0)
            entry.insert(0, val)
            entry.config(state='readonly', readonlybackground='#2d2d2d')
            entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Real-time Metrics / 실시간 지표
        metrics_group = tk.LabelFrame(left_panel, text=" LIVE METRICS / 실시간 지표 ", bg='#1e1e1e', fg='#00ffcc', font=("Inter", 10, "bold"))
        metrics_group.pack(fill=tk.X, pady=10)
        
        self.iter_var = tk.StringVar(value="0")
        self.loss_var = tk.StringVar(value="0.0000")
        self.speed_var = tk.StringVar(value="0.0 it/s")
        self.tokens_per_sec_var = tk.StringVar(value="0.0 tok/s")
        self.grad_norm_var = tk.StringVar(value="0.00")
        
        metrics = [
            ("Iteration:", self.iter_var),
            ("Loss (Train):", self.loss_var),
            ("Speed (it/s):", self.speed_var),
            ("T-Throughput:", self.tokens_per_sec_var),
            ("Grad Norm:", self.grad_norm_var)
        ]
        
        for label, var in metrics:
            row = ttk.Frame(metrics_group)
            row.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(row, text=label, width=15).pack(side=tk.LEFT)
            entry = tk.Entry(row, textvariable=var, bg='#2d2d2d', fg='#00ffcc', borderwidth=0, highlightthickness=0, font=("Inter", 10, "bold"))
            entry.config(state='readonly', readonlybackground='#2d2d2d')
            entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Controls / 제어
        self.start_btn = tk.Button(left_panel, text="START TRAINING", bg='#00aa7f', fg='white', command=self.start_training)
        self.start_btn.pack(fill=tk.X, pady=5)
        
        self.stop_btn = tk.Button(left_panel, text="STOP", bg='#aa0000', fg='white', command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=5)

        # Right Panel: Tabs for Graph and Inference / 오른쪽 패널: 그래프 및 추론을 위한 탭
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Training Dashboard / 탭 1: 학습 대시보드
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text=" TRAINING / 학습 ")
        
        # Graph / 그래프
        self.fig, self.ax = plt.subplots(figsize=(5, 3), dpi=100)
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#2d2d2d')
        self.ax.tick_params(colors='white')
        
        try:
            import matplotlib as mpl
            mpl.rcParams['font.family'] = 'sans-serif'
            mpl.rcParams['axes.unicode_minus'] = False
            self.ax.set_title("Training Loss (Accelerated Math)", color='#00ffcc')
        except:
            self.ax.set_title("Training Loss (LIVE)", color='#00ffcc')
            
        self.loss_line, = self.ax.plot([], [], color='#00ffcc', linewidth=1.5, marker='o', markersize=2, alpha=0.8)
        self.ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.train_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        log_label = ttk.Label(self.train_tab, text="Training Logs / 학습 로그", style="TLabel")
        log_label.pack(pady=(10, 0))
        self.log_area = scrolledtext.ScrolledText(self.train_tab, height=10, bg='#000000', fg='#ffffff', font=("Consolas", 9))
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_area.config(state=tk.DISABLED)

        # Tab 2: Inference Engine / 탭 2: 추론 엔진
        self.infer_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.infer_tab, text=" INFERENCE / 추론 ")
        
        infer_header = ttk.Label(self.infer_tab, text="⚡ ODYSSEY v1.0 GENERATION ENGINE", style="Header.TLabel")
        infer_header.pack(pady=10)
        
        prompt_frame = ttk.Frame(self.infer_tab)
        prompt_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(prompt_frame, text="PROMPT / 프롬프트:").pack(anchor=tk.W)
        self.prompt_entry = tk.Text(prompt_frame, height=4, bg='#2d2d2d', fg='#ffffff', insertbackground='white')
        self.prompt_entry.pack(fill=tk.X, pady=5)
        self.prompt_entry.insert(tk.END, "Once upon a time, in a digital galaxy far away...")
        
        ctrl_frame = ttk.Frame(self.infer_tab)
        ctrl_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.gen_btn = tk.Button(ctrl_frame, text="EXECUTE PROPULSION (GENERATE)", bg='#0078d7', fg='white', 
                                command=self.start_inference, font=("Inter", 10, "bold"))
        self.gen_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(ctrl_frame, text="Max Tokens:").pack(side=tk.LEFT, padx=(20, 5))
        self.token_var = tk.StringVar(value="100")
        self.token_entry = ttk.Entry(ctrl_frame, textvariable=self.token_var, width=5)
        self.token_entry.pack(side=tk.LEFT)
        
        res_label = ttk.Label(self.infer_tab, text="GENERATED ANALYTICS / 생성 결과", style="TLabel")
        res_label.pack(pady=(20, 0), padx=20, anchor=tk.W)
        
        self.inference_result_var = tk.StringVar(value="")
        self.res_area = scrolledtext.ScrolledText(self.infer_tab, height=15, bg='#000000', fg='#00ffcc', font=("Consolas", 10))
        self.res_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        self.res_area.config(state=tk.DISABLED)

        self.info_label = ttk.Label(right_panel, text="slmaker v1.0.0 (Odyssey) | Full Training & Inference | Dual-Interface", 
style="TLabel")
        self.info_label.pack(pady=(10, 0))

    def log(self, msg):
        self.log_queue.put(msg)

    def _check_queues(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_area.config(state=tk.NORMAL)
                self.log_area.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                self.log_area.see(tk.END)
                self.log_area.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        
        try:
            while True:
                data = self.data_queue.get_nowait()
                # Check for inference result / 추론 결과 확인
                if isinstance(data, dict) and 'inference_result' in data:
                    self.res_area.config(state=tk.NORMAL)
                    self.res_area.delete('1.0', tk.END)
                    self.res_area.insert(tk.END, data['inference_result'])
                    self.res_area.config(state=tk.DISABLED)
                    continue

                iter_val = data['iter']
                loss_val = data['loss']
                speed_val = data['speed']
                tps_val = data.get('tokens_per_sec', 0.0)
                gn_val = data.get('grad_norm', 0.0)
                
                self.iter_var.set(str(iter_val))
                self.loss_var.set(f"{loss_val:.4f}")
                self.speed_var.set(f"{speed_val:.1f} it/s")
                self.tokens_per_sec_var.set(f"{tps_val:.1f} tok/s")
                self.grad_norm_var.set(f"{gn_val:.2f}")
                
                # Update Chart
                y_data = self.loss_line.get_ydata()
                x_data = self.loss_line.get_xdata()
                new_y = list(y_data) + [data['loss']]
                new_x = list(x_data) + [data['iter']]
                self.loss_line.set_data(new_x, new_y)
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw()
        except queue.Empty:
            pass
        
        # Sync simple string var to result area if needed / 필요한 경우 단순 문자열 변수를 결과 영역에 동기화
        val = self.inference_result_var.get()
        if val:
            self.res_area.config(state=tk.NORMAL)
            self.res_area.delete('1.0', tk.END)
            self.res_area.insert(tk.END, val)
            self.res_area.config(state=tk.DISABLED)
            self.inference_result_var.set("") # Clear

        self.root.after(100, self._check_queues)

    def start_training(self):
        self.is_training = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.notebook.select(self.train_tab)
        threading.Thread(target=self.run_engine, daemon=True).start()

    def start_inference(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        max_tokens = int(self.token_var.get())
        self.gen_btn.config(state=tk.DISABLED)
        self.log(f"Inference Started: '{prompt[:20]}...'")
        threading.Thread(target=self.run_inference, args=(prompt, max_tokens), daemon=True).start()

    def run_inference(self, prompt, max_tokens):
        from train import engine_inference
        result = engine_inference(prompt, max_tokens, self)
        self.inference_result_var.set(result)
        self.gen_btn.config(state=tk.NORMAL)

    def stop_training(self):
        self.is_training = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log("Training stopped by user. / 사용자에 의해 학습이 중단되었습니다.")

    def run_engine(self):
        from train import engine_train
        self.log("Initializing Training Engine... / 학습 엔진 초기화 중...")
        engine_train(self)

if __name__ == "__main__":
    root = tk.Tk()
    app = NanoSLMGUI(root)
    root.mainloop()
