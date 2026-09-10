#!/usr/bin/env python3
"""METEOR-VLA in the demo runtime (2026-09-10): reads the engine's `bev_tok` output (fused BEV pooled to
25x16, the 13th head) and runs the VLA (Qwen3-VL language model + LoRA + BEV projector + regression head,
weights copied from the training run) in the same process. Returns the same record the offline overlay
used: {"cmd_in", "v0", "wp_reg", "json", "raw"}.

Environment: METEOR_VLA_LIVE=<ckpt_last.pt>, METEOR_VLA_MODEL (default Qwen/Qwen3-VL-8B-Instruct),
METEOR_VLA_EVERY (frames between VLA calls, default 10), HF_HOME for the model cache.
"""
import json, os, re, time
import numpy as np, torch, torch.nn as nn

PREFIX_N = 25 * 16


class BEVProjector(nn.Module):
    def __init__(self, c_in=96, d=4096):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c_in, 1024), nn.GELU(), nn.Linear(1024, d))
        self.pos = nn.Parameter(torch.zeros(PREFIX_N, d)); self.norm = nn.LayerNorm(d)

    def forward(self, tok):                       # [B,96,25,16]
        x = tok.flatten(2).transpose(1, 2)        # [B,400,96]
        return self.norm(self.mlp(x) + self.pos[None])


def anchor_of(v0):
    return np.array([[v0 * 0.5 * (k + 1), 0.0] for k in range(6)], np.float32)


def derive_cmd(gt, v0):
    x6, y6 = float(gt[5, 0]), float(gt[5, 1])
    if v0 < 1.5 and x6 < 3.0:
        return "stop"
    if y6 > 2.0:
        return "turn_left"
    if y6 < -2.0:
        return "turn_right"
    return "keep_lane"


class VLALive:
    def __init__(self, ckpt, model_id=None, device="cuda", lora_r=16, reg_scale=2.0, max_new=192):
        from transformers import AutoTokenizer, AutoModelForImageTextToText
        from peft import LoraConfig, get_peft_model
        model_id = model_id or os.environ.get("METEOR_VLA_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        t0 = time.time()
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = ck.get("args", {}); lora_r = int(a.get("lora_r", lora_r))
        self.tokz = AutoTokenizer.from_pretrained(model_id)
        full = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16)
        lm = full.model.language_model; self.head = full.lm_head; del full.model.visual
        lcfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.lm = get_peft_model(lm, lcfg)
        d = lm.config.hidden_size
        self.proj = BEVProjector(96, d)
        self.reg_scale = float(os.environ.get("METEOR_VLA_REG_SCALE", str(reg_scale)))
        if any(k.startswith("reg.") for k in ck["proj"]):
            self.proj.reg = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 12))
        self.proj.load_state_dict(ck["proj"], strict=False)
        miss = self.lm.load_state_dict(ck["lora"], strict=False)
        self.dev = torch.device(device)
        self.lm.to(self.dev).eval(); self.head.to(self.dev); self.proj.to(self.dev).eval()
        self.max_new = max_new
        print(f"[vla] {model_id} + LoRA r={lora_r} from {ckpt} (unexpected {len(miss.unexpected_keys)}) in {time.time()-t0:.0f}s", flush=True)

    def _embed(self, tok, prompt):
        emb = self.lm.get_input_embeddings()
        pre = self.proj(tok.to(self.dev)).to(torch.bfloat16)                        # [1,400,d]
        ids = torch.tensor([self.tokz(prompt + "\nAnswer: ", add_special_tokens=False)["input_ids"]], device=self.dev)
        x = torch.cat([pre, emb(ids).to(torch.bfloat16)], 1)
        att = torch.ones(1, x.shape[1], device=self.dev, dtype=torch.long)
        return x, att

    @torch.no_grad()
    def infer(self, bev_tok, v0, cmd):
        """bev_tok: np/torch [96,25,16] (fp16 ok). Returns the overlay record."""
        tok = torch.as_tensor(np.asarray(bev_tok, np.float32))[None]
        # trajectory: regression head on the short (evaluation) prompt -- one forward pass, no generation
        short = f"Ego speed {v0:.1f} m/s. Command: {cmd}. Give the driving command and 6 waypoints (x forward, y left, metres)."
        x, att = self._embed(tok, short)
        hid = self.lm(inputs_embeds=x, attention_mask=att, use_cache=False).last_hidden_state
        wp = None
        if hasattr(self.proj, "reg"):
            wp = self.proj.reg(hid[0, -1].float()).cpu().numpy().reshape(6, 2) * self.reg_scale + anchor_of(v0)
        # language: greedy generation on the captioned prompt
        long = f"Ego speed {v0:.1f} m/s. Command: {cmd}. Describe the scene, list hazards, give a rationale, the driving command and 6 waypoints (x forward, y left, metres)."
        x, att = self._embed(tok, long)
        o = self.lm(inputs_embeds=x, attention_mask=att, use_cache=True); past = o.past_key_values
        logits = self.head(o.last_hidden_state[:, -1]); ids = []; emb = self.lm.get_input_embeddings()
        for _ in range(self.max_new):
            nxt = int(logits.argmax(-1))
            if nxt == self.tokz.eos_token_id:
                break
            ids.append(nxt)
            e = emb(torch.tensor([[nxt]], device=self.dev)).to(torch.bfloat16)
            att = torch.cat([att, torch.ones(1, 1, device=self.dev, dtype=torch.long)], 1)
            o = self.lm(inputs_embeds=e, attention_mask=att, past_key_values=past, use_cache=True)
            past = o.past_key_values; logits = self.head(o.last_hidden_state[:, -1])
        txt = self.tokz.decode(ids)
        js = None
        try:
            js = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        except Exception:
            pass
        if wp is None:                     # no regression head: waypoints from the text
            m = re.search(r'"waypoints"\s*:\s*(\[\s*\[.*?\]\s*\])', txt, re.S)
            if m:
                try:
                    wp = np.array(json.loads(m.group(1)), np.float32).reshape(6, 2) + anchor_of(v0)
                except Exception:
                    wp = None
        return {"cmd_in": cmd, "v0": float(v0), "wp_reg": None if wp is None else wp.tolist(), "json": js, "raw": txt}


class VLAClient:
    """Client for deploy/vla_server.py (same infer() signature as VLALive)."""

    def __init__(self, addr):
        import socket
        host, port = addr.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port))); self.f = self.sock.makefile("rwb")
        print(f"[vla] client connected to {addr}", flush=True)

    def infer(self, bev_tok, v0, cmd):
        import base64
        tok = np.ascontiguousarray(np.asarray(bev_tok, np.float16))
        self.f.write((json.dumps({"tok": base64.b64encode(tok.tobytes()).decode(), "v0": float(v0), "cmd": cmd}) + "\n").encode())
        self.f.flush()
        return json.loads(self.f.readline())
