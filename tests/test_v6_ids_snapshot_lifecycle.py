from pathlib import Path
import ast


def test_ids_agent_retains_verified_snapshot_until_shutdown():
    source = (Path(__file__).resolve().parents[1] / "agent" / "ids_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    class_methods = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ZenohIDSAgent":
            class_methods = {
                item.name: item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            break

    assert "_load_verified_bundle" in class_methods
    assert "start_listening" in class_methods

    loader_source = ast.get_source_segment(source, class_methods["_load_verified_bundle"])
    shutdown_source = ast.get_source_segment(source, class_methods["start_listening"])

    assert "self._verified_bundle = verified" in loader_source
    assert loader_source.index("ort.InferenceSession") < loader_source.index("self._verified_bundle = verified")
    assert "verified.close()" in loader_source
    assert "self.ort_session = None" in shutdown_source
    assert "verified_bundle.close()" in shutdown_source
    assert shutdown_source.index("self.ort_session = None") < shutdown_source.index("verified_bundle.close()")


def test_ids_agent_initializes_snapshot_owner_before_model_loading():
    source = (Path(__file__).resolve().parents[1] / "agent" / "ids_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    init_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ZenohIDSAgent":
            init_method = next(
                item for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            )
            break
    assert init_method is not None
    init_source = ast.get_source_segment(source, init_method)
    assert "self._verified_bundle = None" in init_source
    assert init_source.index("self._verified_bundle = None") < init_source.index("self._load_verified_bundle")
