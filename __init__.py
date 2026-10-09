"""VFX nodes for ComfyUI: Light Transfer, Temporal Deflicker, Add Light from Passes."""
from .nodes_vfx import TemporalDeflicker, AddLightFromPasses, LightTransfer

NODE_CLASS_MAPPINGS = {"LightTransfer": LightTransfer, "TemporalDeflicker": TemporalDeflicker, "AddLightFromPasses": AddLightFromPasses}
NODE_DISPLAY_NAME_MAPPINGS = {"LightTransfer": "Light Transfer (plate keeps its pixels)",
                              "TemporalDeflicker": "Temporal Deflicker (flow-guided, for AI passes)",
                              "AddLightFromPasses": "Add Light from Passes (plate + light)"}
