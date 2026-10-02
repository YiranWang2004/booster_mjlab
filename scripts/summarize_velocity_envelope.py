import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path('logs/velocity_envelope')
root=base/'model_30000_refined'
scans=[]
for name in ['model_30000','model_30000_refined','model_30000_axes']:
 d=np.load(base/name/'measurements.npz'); c=d['commands']; w=d['window_velocities'].mean(2); fail=d['failed']
 ratio=w*np.sign(c)[None]/np.maximum(np.abs(c)[None],1e-6)
 ok=np.where(np.abs(c)[None]>1e-6,ratio>=.9,np.abs(w)<=np.array([.1,.1,.15])).all(-1).mean(0)>=.9
 ok &= ~fail.any(1)
 scans.append((name,c,ok))
c=np.concatenate([s[1] for s in scans]); ok=np.concatenate([s[2] for s in scans])
result={'total_command_tests':len(c),'total_rollouts':len(c)*5,'all_scans_feasible_extents':[c[ok].min(0).tolist(),c[ok].max(0).tolist()],'single_axis_extrema':{},'note':'Extrema are observed passing commands, not continuously feasible intervals or an axis-aligned box. The closed mesh uses the refined regular 3D grid; orange markers show a separate fine single-axis scan. Different independently randomized 5-repeat tests can disagree. Primary threshold is >=90%, with no upper cap; the optional accuracy mesh additionally imposes <=110%.','fall_rule':'Any reset, body tilt >63 degrees, root height <0.35m, or nonfinite velocity fails that rollout. All 5 must survive the entire 15s.','limits':'Finite grid, 5 randomized initializations, no external pushes; no proof of global optimum or continuous-space feasibility.'}
for j,name in enumerate(['vx','vy','vyaw']):
 mask=ok & (np.abs(c[:,[k for k in range(3) if k!=j]])<1e-6).all(1)
 result['single_axis_extrema'][name]=[float(c[mask,j].min()),float(c[mask,j].max())]
meta=json.loads((root/'metadata.json').read_text()); result['refined_grid']=meta
ckpt=Path(meta['checkpoint']); result['checkpoint_sha256']=hashlib.sha256(ckpt.read_bytes()).hexdigest()
(root/'combined_report.json').write_text(json.dumps(result,indent=2))
(root/'report.zh.txt').write_text('Headless 速度范围评估\n\n总计 '+str(len(c))+' 个指令测试，每个指令5次，共 '+str(len(c)*5)+' 次15秒 rollout（5秒过渡+10秒测量）。\n主判据：5次均未跌倒；5次平均后，至少9/10个1秒窗口中，各非零速度分量同向达到cmd的90%以上。零分量漂移限制为vx/vy 0.1m/s，vyaw 0.15rad/s。关闭随机推扰，保留初始状态和物理随机化。\n\n所有扫描中，单轴实测通过点的极值（不是连续区间）：\n'+json.dumps(result['single_axis_extrema'],ensure_ascii=False,indent=2)+'\n\n3D网格：4913个指令，各重复5次；步长vx=0.375m/s、vy=0.1875m/s、vyaw=0.5rad/s。137个通过，38个六邻域连通分量，原点连通分量59点。Marching cubes边界为封闭网格，未使用凸包。蓝色几何体仅是采样分类的插值，橙色点是独立单轴细扫结果。未测量的曲面内部不能保证可行，坐标极值不能任意组合。\n单轴细扫步长vx=0.05m/s，vy=0.025m/s，vyaw=0.075rad/s。\n\n交互图可切换额外90%-110%误差约束下的绿色曲面。\n随机初始化及有限5次重复可能造成相邻指令分类不一致；本次结果不是数学上的全局最大可行域，也不包含外界推扰鲁棒性测试。\n')
# Show every fine single-axis classification, including gaps.
d=np.load(base/'model_30000_axes'/'measurements.npz'); c=d['commands']; ok=d['feasible']
f, axs=plt.subplots(3,1,figsize=(12,7),layout='constrained')
for j,ax in enumerate(axs):
 mask=(np.abs(c[:,[k for k in range(3) if k!=j]])<1e-6).all(1)
 ax.scatter(c[mask&ok,j],np.ones((mask&ok).sum()),color='seagreen',s=15,label='pass')
 ax.scatter(c[mask&~ok,j],np.zeros((mask&~ok).sum()),color='firebrick',s=12,label='fail')
 ax.set(xlabel=['vx (m/s)','vy (m/s)','vyaw (rad/s)'][j],yticks=[0,1],yticklabels=['fail','pass'],ylim=(-.3,1.3)); ax.grid(alpha=.2)
axs[0].legend(); f.suptitle('Fine single-axis scan | 5 repeats per command'); f.savefig(root/'single_axis_results.png',dpi=180)
print(json.dumps({k:v for k,v in result.items() if k!='refined_grid'},indent=2))
