"""Before any test module loads: some import ONNX Runtime ahead of the swissdocai package."""

import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
