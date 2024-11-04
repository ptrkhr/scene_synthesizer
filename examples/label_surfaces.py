import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa

s = synth.Scene()

shelf = pa.ShelfAsset(width=1.5, depth=0.4, height=2.0, num_boards=5)
cabinet = pa.CabinetAsset(
    width=1.5,
    depth=0.5,
    height=2.0,
    compartment_mask=[[0, 1]],
    compartment_types=['door_right', 'door_left'],
    compartment_interior_masks={0: [[0],[1],[2]], 1: [[0], [1]]}
    )

s.add_object(shelf, 'shelf')
s.add_object(cabinet, 'cabinet', connect_parent_id='shelf', connect_parent_anchor=('top', 'center', 'bottom'), connect_obj_anchor=('bottom', 'center', 'bottom'))

s.label_support('shelf', obj_ids='shelf', consider_support_polyhedra=True)
s.label_support('inside_cabinet', obj_ids='cabinet', consider_support_polyhedra=True)

s.show_supports()
