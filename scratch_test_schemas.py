import inspect
from llama_cloud.resources.parsing import params

def inspect_params():
    print("Params objects available in llama_cloud.resources.parsing.params:")
    for name, obj in inspect.getmembers(params):
        if inspect.isclass(obj):
            print(f"\nClass: {name}")
            try:
                # If they are pydantic models, print their fields
                if hasattr(obj, "model_fields"):
                    for field_name, field in obj.model_fields.items():
                        print(f"  Field: {field_name} -> {field.annotation}")
                elif hasattr(obj, "__fields__"):
                    for field_name, field in obj.__fields__.items():
                        print(f"  Field: {field_name} -> {field.annotation}")
                else:
                    print("  (not a pydantic model)")
            except Exception as e:
                print(f"  Error inspecting class fields: {e}")

if __name__ == "__main__":
    inspect_params()
