#!/usr/bin/env python3

from concurrent import futures
import argparse
import grpc
import os
import sys
import signal
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import json
from Llama import ModelArgs, Llama
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor

import backend_pb2
import backend_pb2_grpc

MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


# Implement the BackendServicer class with the service methods
class BackendServicer(backend_pb2_grpc.BackendServicer):
    # def generate(self,prompt, max_new_tokens):
    #     self.generator.end_beam_search()

    #     # Tokenizing the input
    #     ids = self.generator.tokenizer.encode(prompt)

    #     self.generator.gen_begin_reuse(ids)
    #     initial_len = self.generator.sequence[0].shape[0]
    #     has_leading_space = False
    #     decoded_text = ''
    #     for i in range(max_new_tokens):
    #         token = self.generator.gen_single_token()
    #         if i == 0 and self.generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith('▁'):
    #             has_leading_space = True

    #         decoded_text = self.generator.tokenizer.decode(self.generator.sequence[0][initial_len:])
    #         if has_leading_space:
    #             decoded_text = ' ' + decoded_text

    #         if token.item() == self.generator.tokenizer.eos_token_id:
    #             break
    #     return decoded_text
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
            # The model directory is indicated on the request
            model_path = Path(request.ModelFile)
            weights = mx.load(str(model_path / "weights.npz"))
            with open(model_path / "config.json", "r") as f:
                config = self.sanitize_config(json.loads(f.read()), weights)
                quantization = config.pop("quantization", None)
            model = Llama(ModelArgs(**config))
            if quantization is not None:
                nn.QuantizedLinear.quantize_module(model, **quantization)
            model.update(tree_unflatten(list(weights.items())))
            tokenizer = SentencePieceProcessor(model_file=str(model_path / "tokenizer.model"))
            self.model = model
            self.tokenizer = tokenizer
        except Exception as err:
            print(err)
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")
        return backend_pb2.Result(message="Model loaded successfully", success=True)


    def toc(msg, start):
        end = time.time()
        return f"[INFO] {msg}: {end - start:.3f} s"
    
    def Predict(self, request, context):
        x = mx.array([[self.tokenizer.bos_id()] + self.tokenizer.encode(request.Prompt)])
        tokens = []
        for token in self.model.generate(x, request.Temperature):
            tokens.append(token)

            if len(tokens) >= request.Tokens:
                break

            elif (len(tokens) % request.Tokens) == 0:
                # It is perfectly ok to eval things we have already eval-ed.
                mx.eval(tokens)
                s = self.tokenizer.decode([t.item() for t in tokens])

        mx.eval(tokens)
        s = self.tokenizer.decode([t.item() for t in tokens])

        return backend_pb2.Result(message=bytes(s, encoding='utf-8'))

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