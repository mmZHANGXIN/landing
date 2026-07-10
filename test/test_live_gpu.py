#!/usr/bin/env python3
"""在线实时管线 — GPU HALSS + matplotlib 3 窗口实时刷新"""
import sys, os, time, logging, threading, argparse
import numpy as np
sys.path.insert(0,'/home/orin/evelyn/orin_landing')
sys.path.append('/home/orin/miniconda3/envs/fylanding/lib/python3.8/site-packages')
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
logger=logging.getLogger("Live")
import yaml, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from perception import HALSSSafetyEvaluator, DepthProjector, SemanticGenerator
from rl import RLAgent
from control.action_decomposer import ActionDecomposer
PROJECT_ROOT='/home/orin/evelyn/orin_landing'
anames=None

class LivePipeline(Node):
    def __init__(self,cfg):
        super().__init__("live")
        pc,oc,dc,uc=cfg["perception"],cfg["observation"],cfg["decision"],cfg["uav"]
        self.halss=HALSSSafetyEvaluator(pc)
        self.dproj=DepthProjector(img_width=oc["img_width"],img_height=oc["img_height"],max_range=pc["depth_max_range"])
        self.semgen=SemanticGenerator({**pc,**oc})
        self.rl=RLAgent(model_path=os.path.join(PROJECT_ROOT,dc["policy_weights_path"]),img_size=(oc["img_width"],oc["img_height"]),vel_lateral=uc["vel_lateral"],vel_vertical=uc["vel_vertical"])
        global anames
        anames=ActionDecomposer(uc).action_names
        self.roi=pc["roi_radius_world"]; self.danger=pc["danger_class_id"]
        self._odom=None; self._running=True
        self._t={"halss":[],"depth":[],"rl":[]}; self._n=0
        self._viz={"sem":None,"dep":None,"pts":None,"pose":None,"act":0,"ready":False,"n":0,"t":[0,0,0]}
        self.create_subscription(Odometry,"/Odometry",self._cb_odom,10)
        self.create_subscription(PointCloud2,"/cloud_registered",self._cb_pc2,10)
        logger.info("GPU HALSS ready | Starting 3-window view...")
    def _cb_odom(self,msg):
        p=msg.pose.pose.position; q=msg.pose.pose.orientation
        r,pitch,y=self._quat(q.x,q.y,q.z,q.w)
        self._odom=np.array([p.x,p.y,p.z,r,pitch,y],np.float32)
    def _cb_pc2(self,msg):
        if not self._running: return
        t0=time.perf_counter()
        pts=self._parse(msg)
        if pts is None or len(pts)<10 or self._odom is None: return
        pose=self._odom.copy()
        d=np.linalg.norm(pts[:,:2]-pose[:2],axis=1); pts_r=pts[d<self.roi]
        t1=time.perf_counter(); r=self.halss.evaluate(pts_r); self._t["halss"].append((time.perf_counter()-t1)*1000)
        sem=self.semgen.generate(r["bev_data"]) if r else np.full((128,128),self.danger,np.uint8)
        t2=time.perf_counter(); dep=self.dproj.project(pts_r,pose); self._t["depth"].append((time.perf_counter()-t2)*1000)
        t3=time.perf_counter(); act=self.rl.predict(dep,sem); self._t["rl"].append((time.perf_counter()-t3)*1000)
        self._n+=1
        self._viz={"sem":sem,"dep":dep,"pts":pts_r,"pose":pose,"act":act,"ready":True,"n":self._n,"t":[self._t["halss"][-1],self._t["depth"][-1],self._t["rl"][-1]]}
        logger.info(f"[{self._n:03d}] act={act}({anames[act]}) H={self._t['halss'][-1]:.0f}ms D={self._t['depth'][-1]:.0f}ms RL={self._t['rl'][-1]:.0f}ms")
    def _parse(self,msg):
        off={f.name:f.offset for f in msg.fields}
        if not all(k in off for k in ('x','y','z')): return None
        npts=msg.width*msg.height
        if npts==0: return None
        raw=np.frombuffer(msg.data,dtype=np.float32); pp=msg.point_step//4
        pts=np.zeros((npts,3),np.float32)
        pts[:,0]=raw[off['x']//4::pp]; pts[:,1]=raw[off['y']//4::pp]; pts[:,2]=raw[off['z']//4::pp]
        v=np.isfinite(pts).all(axis=1); return pts[v] if v.sum()>0 else None
    @staticmethod
    def _quat(x,y,z,w):
        import math
        return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))

def render_loop(node):
    import matplotlib; matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.ion()
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    fig.canvas.manager.set_window_title("Live: Semantic | Depth | Point Cloud")
    im_sem,im_dep=None,None
    while node._running:
        v=node._viz.copy() if node._viz["ready"] else None
        if v is None: plt.pause(0.05); continue
        sem,dep,pts,pose=v["sem"],v["dep"],v["pts"],v["pose"]
        if sem is not None:
            rgb=np.zeros((*sem.shape,3),np.uint8); rgb[sem==1]=[0,200,0]; rgb[sem==9]=[200,0,0]
            if im_sem is None: im_sem=axes[0].imshow(rgb); axes[0].set_title("Semantic"); axes[0].axis('off')
            else: im_sem.set_data(rgb)
        if dep is not None:
            if im_dep is None: im_dep=axes[1].imshow(dep,cmap='inferno',vmin=0,vmax=30); axes[1].set_title("Depth"); axes[1].axis('off'); plt.colorbar(im_dep,ax=axes[1],fraction=0.046)
            else: im_dep.set_data(dep)
        if pts is not None and len(pts)>0:
            axes[2].clear(); axes[2].scatter(pts[:,0],pts[:,1],c=pts[:,2],s=0.3,cmap='plasma')
            axes[2].scatter([pose[0]],[pose[1]],c='red',s=60,marker='x')
            axes[2].set_aspect('equal'); axes[2].set_title(f"Cloud ({len(pts)} pts)"); axes[2].axis('off')
        fig.suptitle(f"Frame {v['n']} | act={v['act']}({anames[v['act']]}) | H={v['t'][0]:.0f}ms D={v['t'][1]:.0f}ms RL={v['t'][2]:.0f}ms",fontsize=11)
        fig.canvas.draw_idle(); plt.pause(0.03)
    plt.close('all')

def main():
    p=argparse.ArgumentParser(); p.add_argument("--save_dir",default=os.path.join(PROJECT_ROOT,"test_live_gpu")); args=p.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)
    with open(os.path.join(PROJECT_ROOT,"config/experiment_config.yaml")) as f: cfg=yaml.safe_load(f)
    rclpy.init(); node=LivePipeline(cfg)
    t=threading.Thread(target=render_loop,args=(node,),daemon=True); t.start()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node._running=False
        def avg(x): return sum(x)/len(x) if x else 0
        logger.info(f"\n--- GPU SUMMARY ({node._n} frames) ---")
        logger.info(f"  HALSS: avg={avg(node._t['halss']):.0f}ms max={max(node._t['halss']):.0f}ms")
        logger.info(f"  Depth: avg={avg(node._t['depth']):.0f}ms max={max(node._t['depth']):.0f}ms")
        logger.info(f"  RL:    avg={avg(node._t['rl']):.0f}ms max={max(node._t['rl']):.0f}ms")
        rclpy.shutdown()

if __name__=="__main__": main()
