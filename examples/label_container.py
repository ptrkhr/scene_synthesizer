from scene_synthesizer import procedural_assets as pa

s = pa.RefrigeratorAsset().scene('fridge')
s.label_containment('container', geom_ids='fridge/(shelf|freezer)_*')
s.show_containers()
