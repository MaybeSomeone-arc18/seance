"""ghost_model.py - local open-weight model wrapper (llama.cpp server).

The ghost's voice IS the model: no model, no seance. This wrapper starts a
llama.cpp server on localhost, waits for health, and issues chat completions.
Everything runs offline; the only network socket is 127.0.0.1.
"""
import json, subprocess, time, urllib.request, atexit, os, signal

class GhostModel:
    def __init__(self, model_path, llama_server="llama-server", port=8399,
                 ctx=2048, threads=None, temp=0.2, max_tokens=220):
        self.model_path = model_path
        self.llama_server = llama_server
        self.port = port
        self.ctx = ctx
        self.threads = threads or max(1, (os.cpu_count() or 2))
        self.temp = temp
        self.max_tokens = max_tokens
        self.proc = None

    def start(self, timeout=180):
        if self.proc:
            return
        env = dict(os.environ)
        # binary lives next to its shared libs in a llama.cpp release dir
        libdir = os.path.dirname(os.path.abspath(self.llama_server))
        env["LD_LIBRARY_PATH"] = libdir + ":" + env.get("LD_LIBRARY_PATH", "")
        self.proc = subprocess.Popen(
            [self.llama_server, "-m", self.model_path, "--port", str(self.port),
             "-c", str(self.ctx), "-t", str(self.threads), "--log-disable"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.stop)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(1)
        self.stop()
        raise RuntimeError("llama-server did not become healthy")

    def stop(self):
        if self.proc:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
            self.proc = None

    def ask(self, system, user, max_tokens=None, temp=None):
        self.start()
        body = json.dumps({
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temp if temp is None else temp,
            "max_tokens": max_tokens or self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.load(r)
        return out["choices"][0]["message"]["content"].strip()
