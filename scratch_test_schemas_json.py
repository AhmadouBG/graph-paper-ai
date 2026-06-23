import json
from llama_cloud.resources.parsing.types import params

def print_schemas():
    for name in ["OutputOptions", "ProcessingOptions", "InputOptions"]:
        cls = getattr(params, name, None)
        if cls:
            print(f"\n==================== Schema for {name} ====================")
            try:
                # Print schema fields
                schema = cls.model_json_schema()
                print(json.dumps(schema, indent=2))
            except Exception as e:
                print(f"Error printing schema for {name}: {e}")

if __name__ == "__main__":
    print_schemas()
