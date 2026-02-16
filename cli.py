import time
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
import queue
import threading
from train import engine_train
from model import n_embd, n_head, n_layer

console = Console()

class NanoSLMCLI:
    def __init__(self):
        self.data_queue = queue.Queue()
        self.log_queue = queue.Queue()
        self.is_training = True
        self.logs = []
        
    def log(self, msg):
        self.log_queue.put(msg)

    def make_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="logs", ratio=2)
        )
        return layout

    def generate_metrics_table(self, data):
        table = Table(title="Live Metrics / 실시간 지표", style="cyan")
        table.add_column("Metric / 지표", style="bold")
        table.add_column("Value / 값", justify="right")
        
        table.add_row("Iteration / 반복", str(data.get('iter', 0)))
        table.add_row("Loss (Train) / 손실", f"{data.get('loss', 0):.4f}")
        table.add_row("Speed / 속도", f"{data.get('speed', 0):.1f} it/s")
        table.add_row("T-Throughput / 처리량", f"{data.get('tokens_per_sec', 0):.1f} tok/s")
        table.add_row("Grad Norm / 경사 노름", f"{data.get('grad_norm', 0):.2f}")
        return table

    def run(self):
        layout = self.make_layout()
        layout["header"].update(Panel("🌌 [bold cyan]slmaker v0.8.0: Odyssey[/bold cyan] | 1.2B Parameters | Dual-Interface", style="white"))
        layout["footer"].update(Panel("Press Ctrl+C to stop / Ctrl+C를 눌러 중지", style="dim"))
        
        current_data = {}
        
        with Live(layout, refresh_per_second=4, screen=True):
            threading.Thread(target=engine_train, args=(self,), daemon=True).start()
            
            try:
                while self.is_training:
                    # Update Logs
                    try:
                        while True:
                            msg = self.log_queue.get_nowait()
                            self.logs.append(f"[dim]{time.strftime('%H:%M:%S')}[/dim] {msg}")
                            if len(self.logs) > 20: self.logs.pop(0)
                    except queue.Empty:
                        pass
                    
                    # Update Metrics
                    try:
                        while True:
                            current_data = self.data_queue.get_nowait()
                    except queue.Empty:
                        pass
                    
                    layout["metrics"].update(Panel(self.generate_metrics_table(current_data), border_style="cyan"))
                    layout["logs"].update(Panel("\n".join(self.logs), title="Training Logs / 학습 로그", border_style="green"))
                    
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.is_training = False

if __name__ == "__main__":
    app = NanoSLMCLI()
    app.run()
