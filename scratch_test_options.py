import inspect
import json
from llama_cloud import AsyncLlamaCloud


def inspect_parsing_options():

    # Inspect parsing.parse signature
    client = AsyncLlamaCloud(api_key="dummy")
    print("\nSignature of client.parsing.parse:")
    try:
        sig = inspect.signature(client.parsing.parse)
        print(sig)
    except Exception as e:
        print(f"Error inspecting signature of client.parsing.parse: {e}")
        
    print("\nSignature of client.parsing.create:")
    try:
        sig = inspect.signature(client.parsing.create)
        print(sig)
    except Exception as e:
        print(f"Error inspecting signature of client.parsing.create: {e}")

if __name__ == "__main__":
    inspect_parsing_options()
