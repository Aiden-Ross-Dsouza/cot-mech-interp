"""Minimal circuit-tracer diagnostic to find hang point."""
import ssl
import os
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context
import requests.adapters as _ra
_orig_adapter_send = _ra.HTTPAdapter.send
def _adapter_send_no_verify(self, request, **kwargs):
    kwargs["verify"] = False
    return _orig_adapter_send(self, request, **kwargs)
_ra.HTTPAdapter.send = _adapter_send_no_verify
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import sys; sys.path.insert(0, 'G:/IvLabs/cot-mech-interp')

print("[1/4] Importing torch...")
import torch
print(f"  torch version: {torch.__version__}")

print("[2/4] Importing transformers...")
from transformers import AutoModel, AutoTokenizer
print("  transformers imported")

print("[3/4] Loading Llama model...")
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.float16, device_map="cuda")
print(f"  Model loaded: {type(model)}")

print("[4/4] Importing circuit-tracer (this might hang)...")
import sys; import datetime
start = datetime.datetime.now()
from circuit_tracer.replacement_model.replacement_model_transformerlens import TransformerLensReplacementModel
elapsed = (datetime.datetime.now() - start).total_seconds()
print(f"  circuit-tracer imported in {elapsed:.1f}s")

print("\n✓ All imports successful")
