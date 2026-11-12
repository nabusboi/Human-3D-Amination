Avatar model assets

rpm_fullbody.glb and idle_standing.glb were downloaded from the rpm_fullbody and
idle_standing sample assets in:
https://huggingface.co/wmghf2023/3d-figure-app-assets

The repository identifies its license as MIT. Keep this notice with the assets.
The live preview renders rpm_fullbody.glb locally with Panda3D in Python.
idle_standing.glb supplies the standing pose when tracking is unavailable.
src/animator/gltf_asset.py corrects the skeleton-root hint and converts
normalized integer skin weights to float weights in memory. It merges the
matching idle clip without changing either source GLB file.
