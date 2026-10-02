"""Plot a padded marching-cubes boundary, never a convex hull of feasible points."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

p=argparse.ArgumentParser(); p.add_argument('directory'); p.add_argument('--axis-data'); a=p.parse_args()
out=Path(a.directory); data=np.load(out/'measurements.npz'); meta=json.loads((out/'metadata.json').read_text())
axes=data['axes']; n=len(axes[0]); grid=data['feasible'].reshape(n,n,n)
if not grid.any():
    raise RuntimeError('No feasible samples; cannot construct a feasibility volume.')
labels, count=ndimage.label(grid)
origin=(n//2,)*3; origin_label=int(labels[origin])
# Preserve disconnected islands, and disclose them.
meta['connected_components']=count; meta['origin_feasible']=bool(grid[origin]); meta['origin_component_points']=int((labels==origin_label).sum()) if origin_label else 0
spacing=axes[:,1]-axes[:,0]
verts, faces, _, _ = marching_cubes(np.pad(grid.astype(float),1), level=.5, spacing=spacing, allow_degenerate=False)
verts += axes[:,0]-spacing
mesh=trimesh.Trimesh(vertices=verts,faces=faces,process=True)
meta['surface_watertight']=bool(mesh.is_watertight)
meta['surface_volume_interpolated']=float(abs(mesh.volume))
mesh.export(out/'feasible_surface.ply')
np.savez_compressed(out/'surface_mesh.npz', vertices=verts, faces=faces)
fig=go.Figure(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],opacity=.48,color='royalblue',name='Interpolated feasible volume',flatshading=True))
c=data['commands'][data['feasible']]
fig.add_trace(go.Scatter3d(x=c[:,0],y=c[:,1],z=c[:,2],mode='markers',marker=dict(size=2,color='navy'),name='Measured feasible commands'))
if 'accurate_feasible' in data and data['accurate_feasible'].any():
    av, af, _, _ = marching_cubes(np.pad(data['accurate_feasible'].reshape(n,n,n).astype(float),1), .5, spacing=spacing, allow_degenerate=False)
    av += axes[:,0]-spacing
    fig.add_trace(go.Mesh3d(x=av[:,0],y=av[:,1],z=av[:,2],i=af[:,0],j=af[:,1],k=af[:,2],opacity=.5,color='seagreen',name='Additional 90%-110% accuracy constraint',visible='legendonly'))
if a.axis_data:
    ad=np.load(Path(a.axis_data)/'measurements.npz'); ac=ad['commands'][ad['feasible']]
    fig.add_trace(go.Scatter3d(x=ac[:,0],y=ac[:,1],z=ac[:,2],mode='markers',marker=dict(size=3,color='darkorange'),name='Fine single-axis feasible samples'))
fig.update_layout(title='K1 model_30000: 5-repeat stable velocity envelope',scene=dict(xaxis_title='vx (m/s)',yaxis_title='vy (m/s)',zaxis_title='vyaw (rad/s)',aspectmode='cube'),annotations=[dict(text='Closed interpolated boundary; interior between grid samples is not independently tested.',xref='paper',yref='paper',x=.5,y=0,showarrow=False)])
fig.write_html(out/'velocity_envelope_3d.html',include_plotlyjs=True)
f=plt.figure(figsize=(12,9)); ax=f.add_subplot(111,projection='3d')
ax.add_collection3d(Poly3DCollection(verts[faces],facecolor='#3875c7',alpha=.35,edgecolor='#26558c',linewidth=.08))
ax.scatter(*c.T,s=3,c='#163963')
if a.axis_data:
    ax.scatter(*ac.T,s=5,c='darkorange',label='Fine single-axis samples')
    ax.legend(loc='upper left')
for j,fn in enumerate([ax.set_xlim,ax.set_ylim,ax.set_zlim]): fn(min(verts[:,j].min(),ac[:,j].min() if a.axis_data else 0)-.1,max(verts[:,j].max(),ac[:,j].max() if a.axis_data else 0)+.1)
ax.set(xlabel='vx (m/s)',ylabel='vy (m/s)',zlabel='vyaw (rad/s)',title='K1 model_30000 | stable tracking, 5 repeats')
ax.view_init(elev=24,azim=-55)
f.text(.5,.03,'Closed boundary interpolated from measured grid points; not a continuous-space guarantee.',ha='center',fontsize=10)
f.savefig(out/'velocity_envelope_3d.png',dpi=200,bbox_inches='tight'); plt.close(f)
# Axis intercepts are distinct from extents obtained with other components nonzero.
meta['axis_intercepts']={}
for j,name in enumerate(['vx','vy','vyaw']):
    mask=data['feasible'] & (np.abs(data['commands'][:,[k for k in range(3) if k!=j]])<1e-6).all(1)
    vals=data['commands'][mask,j]
    meta['axis_intercepts'][name]=[float(vals.min()),float(vals.max())] if len(vals) else None
(out/'metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
