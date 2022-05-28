import bpy
import bmesh
import numpy as np

class BlenderHelper:

    @staticmethod
    def get_selected_points():
        vs = np.array([v.co for v in bmesh.from_edit_mesh(bpy.context.active_object.data).verts if v.select])
        return vs
