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
        self.root.title("Nano-SLM Professional Dashboard / 나노 SLM 프로페셔널 대시보드")
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
            ("Parameters:", "~1.5M"),
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
        
        metrics = [
            ("Iteration:", self.iter_var),
            ("Loss (Train):", self.loss_var),
            ("Speed:", self.speed_var)
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

        # Right Panel: Graph and Log / 오른쪽 패널: 그래프 및 로그
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Graph / 그래프
        self.fig, self.ax = plt.subplots(figsize=(5, 3), dpi=100)
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#2d2d2d')
        self.ax.tick_params(colors='white')
        
        # Safe font handling / 안전한 폰트 처리
        try:
            import matplotlib as mpl
            mpl.rcParams['font.family'] = 'sans-serif'
            mpl.rcParams['axes.unicode_minus'] = False
            self.ax.set_title("Training Loss (Accelerated Math)", color='#00ffcc')
        except:
            self.ax.set_title("Training Loss (LIVE)", color='#00ffcc')
            
        self.loss_line, = self.ax.plot([], [], color='#00ffcc', linewidth=1.5, marker='o', markersize=2, alpha=0.8)
        self.ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Log / 로그
        log_label = ttk.Label(right_panel, text="Training Logs / 학습 로그", style="TLabel")
        log_label.pack(pady=(10, 0))
        self.log_area = scrolledtext.ScrolledText(right_panel, height=10, bg='#000000', fg='#ffffff', font=("Consolas", 9))
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_area.config(state=tk.DISABLED)

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
                self.iter_var.set(str(data['iter']))
                self.loss_var.set(f"{data['loss']:.4f}")
                self.speed_var.set(f"{data['speed']:.1f} it/s")
                
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
        
        self.root.after(100, self._check_queues)

    def start_training(self):
        self.is_training = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self.run_engine, daemon=True).start()

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
