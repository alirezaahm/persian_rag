import subprocess
import time 
import os 
import signal
import urllib.request
from pathlib import Path 


class LlamaServerManager:
	def __init__(self):
		self.process = None
		self.current_model = None
		self.project_dir = Path(__file__).resolve().parent
		self.run_dir = self.project_dir / "run"
		self.run_dir.mkdir(parents=True, exist_ok=True)
		self.pid_file = self.run_dir / "llama-server.pid"
		self.log_file = self.run_dir / "llama-server.log"
		self.port = 8080

	def start(self, model_path, model_alias):
		model_path = Path(model_path).expanduser()
		if not model_path.is_file():
			raise FileNotFoundError(f"model file not found: {model_path}")

		# reuse the current server if the model in unchanged
		if (
			self.process is not None 
			and self.process.poll() is None
			and self.current_model == str(model_path)
		):
			return

		# stop the previous model
		self.stop()

		command = [
			"llama-server",
			"-m", str(model_path),
			"--host", "127.0.0.1",
			"--port", str(self.port),
			"--alias", model_alias,
			"--ctx-size", "8192",
			"--threads", "2"
		]

		# save sever output to a log file
		log = open(self.log_file, "a", encoding="utf-8")
		self.process = subprocess.Popen(
			command,
			stdout=log,
			stderr=subprocess.STDOUT,
			start_new_session=True
		)
		self.current_model = str(model_path)
		self.pid_file.write_text(str(self.process.pid),encoding="utf-8")
		# wait until the api server becomes available
		for attempt in range(120):
			if self.process.poll() is not None:
				raise RuntimeError(f"llama-server stopped unexpectedly. check: {self.log_file}")

			try:
				request = urllib.request.Request(f"http://127.0.0.1:{self.port}/health")
				with urllib.request.urlopen(request, timeout=2):
					return
			except Exception:
				time.sleep(0.5)

		self.stop()

		raise TimeoutError("llama-server did not become ready in time.")

	def stop(self):
		if self.process is not None:
			if self.process.poll() is None:
				self.process.terminate()
				try:
					self.process.wait(timeout=10)

				except subprocess.TimeoutExpired:
					self.process.kill()
					self.process.wait()

			self.process = None 
		self.current_model = None

		if self.pid_file.exists():
			self.pid_file.unlink()

		print("llama.cpp server stopped.")