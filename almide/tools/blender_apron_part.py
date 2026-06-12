import bpy, bmesh, os
from mathutils import Vector
bpy.ops.preferences.addon_enable(module="VRM_Addon_for_Blender-release")
SRC="/Users/o6lvl4/workspace/github.com/almide-graphics/nendo/almide/web/parts_catalog/almaid_maid_v3.vrm"
bpy.ops.import_scene.vrm(filepath=SRC)
arm=[o for o in bpy.data.objects if o.type=='ARMATURE'][0]
def bw(n):
    b=arm.data.bones.get(n)
    return (arm.matrix_world @ b.head_local) if b else None
chest=bw("J_Bip_C_UpperChest") or bw("J_Bip_C_Chest")
spine=bw("J_Bip_C_Spine"); hips=bw("J_Bip_C_Hips"); knee=bw("J_Bip_L_LowerLeg")
tops=bpy.data.objects.get("衣装_part_1"); bottoms=bpy.data.objects.get("衣装_part_2")
def dup(o):
    d=o.copy(); d.data=o.data.copy(); bpy.context.collection.objects.link(d); return d
t1,t2=dup(tops),dup(bottoms)
bpy.ops.object.select_all(action='DESELECT')
t1.select_set(True); t2.select_set(True); bpy.context.view_layer.objects.active=t1
bpy.ops.object.join()
target=bpy.context.view_layer.objects.active

bibtop=chest.z+0.06; waist=spine.z+0.01; hem=knee.z+0.04
NR,NC=18,10
mesh=bpy.data.meshes.new("Apron"); bm=bmesh.new(); grid=[]
for r in range(NR+1):
    t=r/NR; z=bibtop+(hem-bibtop)*t
    if z>waist: w=0.055+0.015*(bibtop-z)/(bibtop-waist+1e-6)
    else:       w=0.07+(0.155-0.07)*((waist-z)/(waist-hem+1e-6))
    row=[]
    for c in range(NC+1):
        x=-w+2*w*c/NC
        zz=z-(0.03 if (r==NR and c%2==1) else 0)
        row.append(bm.verts.new((x,-0.25,zz)))
    grid.append(row)
for r in range(NR):
    for c in range(NC):
        bm.faces.new([grid[r][c],grid[r][c+1],grid[r+1][c+1],grid[r+1][c]])
bm.normal_update(); bm.to_mesh(mesh)
apron=bpy.data.objects.new("Apron",mesh); bpy.context.collection.objects.link(apron)

sw=apron.modifiers.new("sw",'SHRINKWRAP')
sw.target=target; sw.wrap_method='NEAREST_SURFACEPOINT'; sw.wrap_mode='OUTSIDE_SURFACE'; sw.offset=0.012
bpy.ops.object.select_all(action='DESELECT')
apron.select_set(True); bpy.context.view_layer.objects.active=apron
bpy.ops.object.modifier_apply(modifier="sw")
print("APRON après sw bbox:", [round(v,3) for v in apron.bound_box[0]], [round(v,3) for v in apron.bound_box[6]])

sol=apron.modifiers.new("sol",'SOLIDIFY'); sol.thickness=0.002
bpy.ops.object.modifier_apply(modifier="sol")

# manual weights by height band
for gname in ["J_Bip_C_Chest","J_Bip_C_Spine","J_Bip_C_Hips"]:
    apron.vertex_groups.new(name=gname)
vg={g.name:g for g in apron.vertex_groups}
for v in apron.data.vertices:
    z=v.co.z
    if z>spine.z+0.05: vg["J_Bip_C_Chest"].add([v.index],1.0,'REPLACE')
    elif z>hips.z-0.02: vg["J_Bip_C_Spine"].add([v.index],1.0,'REPLACE')
    else: vg["J_Bip_C_Hips"].add([v.index],1.0,'REPLACE')
am=apron.modifiers.new("arm",'ARMATURE'); am.object=arm
apron.parent=arm

m=bpy.data.materials.new("N99_Apron_00_CLOTH"); m.use_nodes=True
m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value=(0.94,0.94,0.96,1)
apron.data.materials.append(m)

# delete everything except armature + apron → standalone part
for o in list(bpy.data.objects):
    if o not in (arm,apron) and o.type in ('MESH','EMPTY'):
        bpy.data.objects.remove(o,do_unlink=True)
OUT="/Users/o6lvl4/workspace/github.com/almide-graphics/nendo/almide/web/parts_catalog/parts/apron_white.vrm"
bpy.ops.export_scene.vrm(filepath=OUT)
print("APRON_PART_EXPORTED",os.path.getsize(OUT)//1024,"KB")
