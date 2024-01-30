#!/usr/bin/env python3

from concurrent import futures
import argparse
import grpc
import os
import sys
import signal
import time
import mlx.core as mx
from stable_diffusion import StableDiffusion
from tqdm import tqdm
import numpy as np
from PIL import Image
import backend_pb2
import backend_pb2_grpc
from math import ceil

MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


# Implement the BackendServicer class with the service methods
class BackendServicer(backend_pb2_grpc.BackendServicer):
    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))
    def sanitize_config(self, config, weights):
        config.pop("model_type", None)
        n_heads = config["n_heads"]
        if "n_kv_heads" not in config:
            config["n_kv_heads"] = n_heads
        if "head_dim" not in config:
            config["head_dim"] = config["dim"] // n_heads
        if "hidden_dim" not in config:
            config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
        if config.get("vocab_size", -1) < 0:
            config["vocab_size"] = weights["output.weight"].shape[-1]
        if "rope_theta" not in config:
            config["rope_theta"] = 10000
        unused = ["multiple_of", "ffn_dim_multiplier"]
        for k in unused:
            config.pop(k, None)
        return config
    
    def LoadModel(self, request, context):
        try:
            self.sd = StableDiffusion(request.Model)
        except Exception as err:
            print(err)
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")
        return backend_pb2.Result(message="Model loaded successfully", success=True)


    def toc(msg, start):
        end = time.time()
        return f"[INFO] {msg}: {end - start:.3f} s"
    
    def GenerateImage(self, request, context):
        lowestValue = request.width if (request.width < request.height) else request.height
        def tr(dim: int):
            return ceil(dim * 8 / lowestValue) * 8
        latent = (tr(request.height), tr(request.width))
        print(latent)
        latents = self.sd.generate_latents(
            request.positive_prompt,
            n_images=1,
            cfg_weight=7.5,
            num_steps=50,
            negative_text="",
            latent_size=latent
        )
        for x_t in tqdm(latents, total=50):
            mx.eval(x_t)

        decoded = []
        for i in tqdm(range(0, 1, 1)):
            decoded.append(self.sd.decode(x_t[i : i + 1]))
            mx.eval(decoded[-1])

        x = mx.concatenate(decoded, axis=0)
        x = mx.pad(x, [(0, 0)])
        B, H, W, C = x.shape
        x = x.reshape(1, B // 1, H, W, C).transpose(0, 2, 1, 3, 4)
        x = x.reshape(1 * H, B // 1 * W, C)
        x = (x * 255).astype(mx.uint8)
        
        img = Image.fromarray(np.array(x))
        img.save(request.dst)
        return backend_pb2.Result(message="Media generated", success=True)

def serve(address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    server.add_insecure_port(address)
    server.start()
    print("Server started. Listening on: " + address, file=sys.stderr)
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)

    # Define the signal handler function
    def signal_handler(sig, frame):
        print("Received termination signal. Shutting down...")
        server.stop(0)
        sys.exit(0)

    # Set the signal handlers for SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument(
        "--addr", default="localhost:50051", help="The address to bind the server to."
    )
    args = parser.parse_args()

    serve(args.addr)