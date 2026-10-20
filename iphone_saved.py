#!/usr/bin/env python3
"""
Upload server + live viewer for the LiDAR capture app.

  POST /upload   multipart files from the phone, saved to iphone_saved/   (unchanged)
  POST /preview  raw JPEG preview frames from the phone (not saved to disk)
  GET  /         live page: camera, marker check, capture counters
  GET  /preview.jpg, /status.json

Run with the project venv to get the marker overlay (conda's OpenCV is broken):
    .venv/bin/python iphone_saved.py
Plain `python3 iphone_saved.py` still works, without the overlay.
"""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import collections
import json
import os
import re
import ssl
import sys
import subprocess
import threading
import time

try:
    import cv2
    import numpy as np
    _DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _DETECTOR = cv2.aruco.ArucoDetector(_DICT)
    HAVE_CV = True
except Exception:
    HAVE_CV = False

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.environ.get("LIDAR_SAVE_DIR", "/Users/mitul/Desktop/lidar-capture/iphone_saved")
sys.path.insert(0, HERE)
try:
    import tower_points as _tp          # marker layout + pose solver
    _tp.SAVE_DIR = SAVE_DIR
    HAVE_POSE = True
except Exception:
    HAVE_POSE = False
os.makedirs(SAVE_DIR, exist_ok=True)

_lock = threading.Lock()
_state = {
    "jpeg": None, "markers": [], "preview_t": 0.0, "capturing": False, "saved": 0,
    "tracking": "?", "marks": 0, "want": None, "mark_seq": 0, "files": 0, "want_mesh": False, "mesh_files": 0,
    "android_jpeg": None, "android_raw": None, "jpeg_raw": None, "android_markers": [], "android_t": 0.0,
    "android_saved": 0, "android_size": None,
    "tower": {"layers": 10, "removed": []}, "overlay": None, "android_overlay": None,
    "autotower": {"status": "waiting for both cameras"}, "marks_log": [], "last_label": None, "label_stats": {"moves": 0, "wrong": 0}, "autotower_lock": True, "autotower_last": "none yet", "autotower_redetect": False, "tower_epoch": 0, "tracker_msg": "tracker idle - press START", "tracker_on": False,
    "dataset": {"on": False, "name": None, "frames": 0, "crops": 0, "total_crops": 0, "total_frames": 0}, "started": time.time(), "image_size": None,
}
_preview_times = collections.deque(maxlen=40)
_upload_times = collections.deque(maxlen=400)
_android_times = collections.deque(maxlen=40)
_android_dbg = collections.deque(maxlen=60)

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>LiDAR live view</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0e1013;--panel:#171a1f;--ink:#e8eaed;--dim:#8b929c;--ok:#34c759;--warn:#ffb020;--bad:#ff453a}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
font:15px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{display:grid;grid-template-columns:1fr 300px;height:100%}
.view{display:flex;gap:12px;padding:12px;min-width:0;min-height:0}
.pane{flex:1 1 0;min-width:0;min-height:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px}
.plabel{font-weight:700;font-size:14px;color:var(--dim);display:flex;gap:10px;align-items:center}
.tag{font-weight:700;border-radius:8px;padding:2px 9px;background:#22262d;color:var(--dim)}
.tag.ok{background:#12351d;color:var(--ok)}.tag.warn{background:#3a2c0b;color:var(--warn)}.tag.bad{background:#3d1512;color:var(--bad)}
canvas{max-width:100%;max-height:calc(100% - 34px);border-radius:10px;background:#000}
@media(max-width:1000px){.view{flex-direction:column}}
.side{background:var(--panel);padding:18px;overflow:auto;border-left:1px solid #262a31}
h1{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin:0 0 14px;font-weight:600}
.chip{border-radius:10px;padding:12px 14px;font-weight:700;font-size:16px;margin-bottom:12px}
.ok{background:#12351d;color:var(--ok)}.warn{background:#3a2c0b;color:var(--warn)}.bad{background:#3d1512;color:var(--bad)}
.idle{background:#22262d;color:var(--dim)}
.row{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #22262d}
.row span:first-child{color:var(--dim)}.big{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums}
button{background:#22262d;color:var(--ink);border:1px solid #333944;border-radius:8px;padding:8px 12px;
font-size:14px;cursor:pointer;margin-top:14px;width:100%}
.hint{color:var(--dim);font-size:13px;margin-top:14px}
@media(max-width:800px){.wrap{grid-template-columns:1fr;grid-template-rows:1fr auto}}
</style></head><body><div class="wrap">
<div class="view">
  <div class="pane"><div class="plabel">iPhone · LiDAR <span id="lbA" class="tag">no signal</span></div><canvas id="c" width="960" height="720"></canvas></div>
  <div class="pane"><div class="plabel">Android <span id="lbB" class="tag">no signal</span></div><canvas id="c2" width="960" height="540"></canvas></div>
</div>
<div class="side">
  <h1>Capture</h1>
  <div id="rec" class="chip idle">waiting for phone…</div>
  <button id="bStart">START recording</button>
  <button id="bMark" style="background:#5a3a08;border-color:#8a5a12">MARK move done &nbsp;(Space)</button>
  <button id="bStop">STOP recording</button>
  <div class="hint" style="margin-bottom:10px">Control everything from here. Don't touch the phone.</div>
  <h1 style="margin-top:6px;color:#ffb020">After MARK: which block did you pull?</h1>
  <div class="hint" style="margin-top:0">After MARK, type <b>left</b>, <b>middle</b> or <b>right</b> (as in the iPhone picture), then Enter.
  Add a number to say the layer yourself, e.g. <b>middle 5</b>. The face model finds the layer. Needs data capture ON.</div>
  <input id="lbl" placeholder="left / middle / right  [layer]" autocomplete="off"
    style="width:100%;margin-top:8px;background:#22262d;color:#e8eaed;border:1px solid #333944;border-radius:8px;padding:10px;font-size:15px">
  <div id="lres" class="chip idle" style="font-size:14px;margin-top:8px">no label yet</div>
  <div class="row"><span>Frames saved</span><span class="big" id="saved">0</span></div>
  <div class="row"><span>Moves marked</span><span class="big" id="marks">0</span></div>
  <h1 style="margin-top:22px">Setup</h1>
  <div id="mk" class="chip idle">markers: –</div>
  <div class="row"><span>Marker IDs</span><span id="ids">–</span></div>
  <div class="row"><span>Phone tracking</span><span id="trk">–</span></div>
  <h1 style="margin-top:22px">Camera 2 (Android browser)</h1>
  <div id="amk" class="chip idle">not connected</div>
  <h1 style="margin-top:22px">ARKit mesh test</h1>
  <button id="bMesh" style="background:#2a2a45;border-color:#4a4a80">Turn ARKit mesh ON</button>
  <div id="minfo" class="hint">off. Exports ARKit's own fused mesh about once a second while recording, to measure how fine it is.</div>
  <h1 style="margin-top:22px">Training data</h1>
  <button id="bData" style="background:#16324a;border-color:#2b5f8a">Start data capture</button>
  <div id="dinfo" class="hint">off. Records both cameras and cuts out every block face. Play normally;
  pressing Space on moves is optional but helps.</div>
  <h1 style="margin-top:22px">Tower (red / green)</h1>
  <div class="row"><span>Layers</span><input id="tl" type="number" min="1" max="18" value="10"
    style="width:70px;background:#22262d;color:#e8eaed;border:1px solid #333944;border-radius:6px;padding:4px"></div>
  <div class="row"><span>Removed</span><input id="tr" placeholder="e.g. L5S1, L7S0"
    style="width:150px;background:#22262d;color:#e8eaed;border:1px solid #333944;border-radius:6px;padding:4px"></div>
  <button id="tapply">Apply</button>
  <div id="tcount" class="hint">–</div>
  <div id="tauto" class="hint">auto-detect: waiting…</div>
  <button id="bRedetect">Re-detect tower (after rebuilding)</button>
  <button id="bCover" style="background:#16324a;border-color:#2b5f8a">Check coverage (which blocks can I judge?)</button>
  <pre id="cover" style="display:none;font-size:11px;line-height:1.35;background:#0e1013;border:1px solid #262a31;border-radius:8px;padding:8px;overflow:auto;white-space:pre"></pre>
  <div id="tmsg" class="chip idle" style="font-size:14px;margin-top:8px">tracker idle</div>
  <div class="hint">L = layer from the bottom (1 = bottom), S = slot 0, 1 or 2. Green = can be pulled.</div>
  <h1 style="margin-top:22px">Link</h1>
  <div class="row"><span>Preview</span><span id="pfps">–</span></div>
  <div class="row"><span>Uploads</span><span id="ups">–</span></div>
  <div class="row"><span>Last upload</span><span id="age">–</span></div>
  <button id="rot">Rotate iPhone view (R)</button>
  <button id="rotb">Rotate Android view (T)</button>
  <div class="hint">Green = at least 3 of 4 markers visible. Need 2+ for the tower pose;
  4 is best. Keep the phone still while recording.</div>
</div></div>
<script>
const c=document.getElementById('c'), c2=document.getElementById('c2');
const rots={a:+(localStorage.getItem('rotA')||90), b:+(localStorage.getItem('rotB')||0)};
function drawTo(cv,img,rot){
  const x=cv.getContext('2d');
  const sw=rot%180? img.naturalHeight:img.naturalWidth, sh=rot%180? img.naturalWidth:img.naturalHeight;
  if(cv.width!==sw||cv.height!==sh){cv.width=sw;cv.height=sh}
  x.save();x.translate(cv.width/2,cv.height/2);x.rotate(rot*Math.PI/180);
  x.drawImage(img,-img.naturalWidth/2,-img.naturalHeight/2);x.restore();
}
function feed(cv,url,which,gap){
  const im=new Image();
  im.onload=()=>{drawTo(cv,im,rots[which]);setTimeout(()=>feed(cv,url,which,gap),gap)};
  im.onerror=()=>setTimeout(()=>feed(cv,url,which,gap),600);
  im.src=url+'?'+Date.now();
}
feed(c,'/preview.jpg','a',40);
feed(c2,'/android.jpg','b',120);
document.getElementById('rot').onclick=()=>rotate('a');
document.getElementById('rotb').onclick=()=>rotate('b');
document.getElementById('lbl').addEventListener('keydown',e=>{if(e.key==='Enter'){
  const v=e.target.value.trim(); if(v){fetch('/cmd/label',{method:'POST',body:JSON.stringify({text:v})}); e.target.value='';}}});
addEventListener('keydown',e=>{ if(e.target&&e.target.tagName==='INPUT') return;
  if(e.key==='r'||e.key==='R')rotate('a');
  if(e.key==='t'||e.key==='T')rotate('b');
  if(e.code==='Space'){e.preventDefault();cmd('mark')}});
function cmd(n){fetch('/cmd/'+n,{method:'POST'}); if(n==='mark'){setTimeout(()=>document.getElementById('lbl').focus(),150)}}
document.getElementById('tapply').onclick=()=>{
  const rem=[...document.getElementById('tr').value.matchAll(/L\s*(\d+)\s*S\s*(\d)/gi)].map(m=>[+m[1]-1,+m[2]]);
  fetch('/cmd/tower',{method:'POST',body:JSON.stringify({layers:+document.getElementById('tl').value,removed:rem})});
};
let dataOn=false;
document.getElementById('bCover').onclick=async()=>{const p=document.getElementById('cover');p.style.display='block';p.textContent='measuring...';p.textContent=await (await fetch('/coverage.txt')).text();};
document.getElementById('bRedetect').onclick=()=>fetch('/cmd/redetect',{method:'POST'});
document.getElementById('bData').onclick=()=>fetch('/cmd/dataset/'+(dataOn?'off':'on'),{method:'POST'});
let meshOn=false;
document.getElementById('bMesh').onclick=()=>fetch('/cmd/mesh/'+(meshOn?'off':'on'),{method:'POST'});
document.getElementById('bStart').onclick=()=>cmd('start');
document.getElementById('bStop').onclick=()=>cmd('stop');
document.getElementById('bMark').onclick=()=>cmd('mark');
function rotate(w){rots[w]=(rots[w]+90)%360;localStorage.setItem(w==='a'?'rotA':'rotB',rots[w])}
const $=id=>document.getElementById(id);
async function poll(){
  try{
    const s=await (await fetch('/status.json')).json();
    const live=s.preview_age!==null&&s.preview_age<2;
    const r=$('rec');
    if(!live){r.className='chip idle';r.textContent='no phone preview';}
    else if(s.capturing){r.className='chip bad';r.textContent='● RECORDING';}
    else{r.className='chip ok';r.textContent='ready – not recording';}
    $('saved').textContent=s.saved;$('marks').textContent=s.marks;
    const m=$('mk');
    if(!s.cv){m.className='chip idle';m.textContent='marker check off (run with .venv)';}
    else if(!live){m.className='chip idle';m.textContent='markers: –';}
    else if(s.markers.length>=3){m.className='chip ok';m.textContent=s.markers.length+' / 4 markers';}
    else if(s.markers.length===2){m.className='chip warn';m.textContent='2 / 4 markers – borderline';}
    else{m.className='chip bad';m.textContent=s.markers.length+' / 4 markers – move phone';}
    $('ids').textContent=s.markers.length?s.markers.join(', '):'–';
    const a=$('amk');
    if(s.android_age!==null&&s.android_age<10){
      const n=s.android_markers.length;
      a.className='chip '+(n>=3?'ok':n===2?'warn':'bad');
      a.textContent='markers '+(s.android_markers.join(', ')||'none')+'  ·  saved '+s.android_saved;
    }else{a.className='chip idle';a.textContent='not connected';}
    const tag=(el,ok,ids)=>{ if(!ok){el.className='tag';el.textContent='no signal';return}
      el.className='tag '+(ids.length>=3?'ok':ids.length===2?'warn':'bad');
      el.textContent='markers '+(ids.length?ids.join(', '):'none');};
    tag($('lbA'),live,s.markers);
    tag($('lbB'),s.android_age!==null&&s.android_age<10,s.android_markers);
    $('trk').textContent=s.tracking;
    meshOn=!!s.want_mesh; const mb=$('bMesh');
    mb.textContent=meshOn?'Turn ARKit mesh OFF':'Turn ARKit mesh ON';
    $('minfo').textContent=meshOn?('ON · mesh files received: '+s.mesh_files):'off · exports ARKit\'s fused mesh ~1/s while recording';
    const ds=s.dataset||{}; dataOn=!!ds.on;
    const db=$('bData'); db.textContent=dataOn?'■ Stop data capture':'Start data capture';
    db.style.background=dataOn?'#5a1a16':'#16324a';
    $('dinfo').textContent=dataOn?('RECORDING DATA · this run: '+ds.frames+' frames, '+ds.crops+' crops · all runs: '+(ds.total_crops||0)+' crops')
      :('off · '+(ds.total_crops||0)+' crops saved so far');
    const ll=s.last_label, st=s.label_stats||{}, lr=$('lres');
    if(ll){ lr.textContent=(ll.message||ll.status)+(st.moves?('   ['+(st.moves-st.wrong)+'/'+st.moves+' right so far; no correction counts as right]'):'');
      lr.className='chip '+(ll.status==='error'?'bad':ll.status==='done'?(ll.confident?'ok':'warn'):'idle'); }
    else if(s.marks_pending){ lr.textContent='move marked: type left / middle / right'; lr.className='chip warn'; }
    const ov=s.overlay||s.android_overlay;
    const au=s.autotower||{}; $('tauto').textContent='auto-detect: '+(au.status||'–')+(au.note?(' · '+au.note):'');
    const tm=$('tmsg'); tm.textContent=s.tracker_msg;
    tm.className='chip '+(s.tracker_msg.indexOf('could not')>=0||s.tracker_msg.indexOf('error')>=0||s.tracker_msg.indexOf('COLLAPSED')>=0?'bad':s.tracker_msg.indexOf('ready')===0||s.tracker_msg.indexOf('move')===0?'ok':'idle');
    $('tcount').textContent=ov?('green '+ov.green+'  ·  red '+ov.red):'overlay needs 2+ markers in view';
    $('pfps').textContent=live?s.preview_fps.toFixed(1)+' fps':'–';
    $('ups').textContent=s.upload_rate.toFixed(1)+' files/s  ('+s.files+' total)';
    $('age').textContent=s.upload_age===null?'–':s.upload_age.toFixed(1)+' s ago';
  }catch(e){}
  setTimeout(poll,400);
}
poll();
</script></body></html>"""


CAM_PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Camera 2</title>
<meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;background:#000;color:#fff;font:17px -apple-system,Roboto,sans-serif">
<video id="v" playsinline muted autoplay style="width:100%;display:block;background:#111"></video>
<div id="s" style="padding:14px;font-size:19px;line-height:1.4;white-space:pre-line">starting camera…</div>
<div style="padding:0 14px 20px;color:#8b929c;font-size:14px">Leave this page open and the phone still.
Screen stays on. Frames are only saved while the iPhone is recording.</div>
<script>
(async()=>{
  const s=document.getElementById('s'), v=document.getElementById('v');
  const P=new URLSearchParams(location.search);
  const RES=+(P.get('res')||720), Q=+(P.get('q')||0.8);
  try{
    const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},
      width:{ideal:RES*16/9},height:{ideal:RES}},audio:false});
    v.srcObject=stream; await v.play();
    try{await navigator.wakeLock.request('screen')}catch(e){}
    const c=document.createElement('canvas'), x=c.getContext('2d'); let n=0, enc=0, up=0, kb=0, drw=0;
    const loop=async()=>{
      const t0=performance.now();
      if(v.videoWidth){
        const td=performance.now();
        c.width=v.videoWidth; c.height=v.videoHeight; x.drawImage(v,0,0);
        drw=performance.now()-td;
        const te=performance.now();
        const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',Q));
        enc=performance.now()-te; kb=blob.size/1024; const tu=performance.now();
        try{
          const r=await fetch('/android',{method:'POST',body:blob,headers:{'X-Draw':drw.toFixed(0),'X-Enc':enc.toFixed(0),'X-Up':up.toFixed(0)}}); const j=await r.json();
          up=performance.now()-tu;
          s.textContent=(j.recording?'● RECORDING':'ready')+'  ·  markers: '+(j.markers.join(', ')||'none')
            +'  ·  '+c.width+'×'+c.height+'  ·  saved '+j.saved
            +'\ndraw '+drw.toFixed(0)+' ms · encode '+enc.toFixed(0)+' ms · upload '+up.toFixed(0)+' ms · '+kb.toFixed(0)+' KB';
          s.style.color=j.recording?'#ff453a':(j.markers.length>=2?'#34c759':'#ffb020');
        }catch(e){s.textContent='cannot reach the Mac'; s.style.color='#ff453a'}
        n++;
      }
      setTimeout(loop,Math.max(0,333-(performance.now()-t0)));
    };
    loop();
  }catch(e){ s.textContent='camera error: '+e.name+' - '+e.message; s.style.color='#ff453a'; }
})();
</script></body></html>"""


# --- red / green overlay -------------------------------------------------------------------
IPHONE_K_1920 = (1399.0082, 1399.0082, 964.9536, 717.8016, 1920)   # this iPhone's intrinsics @ 1920 px
GREEN, RED = (60, 200, 60), (50, 50, 235)
_HALF, _WIDTH, _HEIGHT = 37.5, 25.0, 15.0


def _k_iphone(w):
    fx, fy, cx, cy, W = IPHONE_K_1920
    s = w / W
    return np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1.0]]), None


def _k_android(w, h):
    p = os.path.join(HERE, "camera_android.json")
    if not os.path.exists(p):
        return None, None
    c = json.load(open(p))
    fx, fy, cx, cy, W, H = c["fx"], c["fy"], c["cx"], c["cy"], c["width"], c["height"]
    dist = np.array(c.get("dist") or [], np.float64)
    if abs(w * H - h * W) > 0.01 * w * H and abs(w * W - h * H) < 0.01 * w * H:
        # the phone is turned 90 degrees from when it was calibrated: same lens, axes swapped
        fx, fy, cx, cy, W, H = fy, fx, H - cy, cx, H, W
        if len(dist) >= 5:
            dist = np.array([dist[0], dist[1], 0.0, 0.0, dist[4]])   # radial terms are unchanged
    s = w / W
    if abs(h - H * s) > 4:
        return None, None            # a different aspect ratio than anything we calibrated
    K = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1.0]])
    return K, dist


def _box(layer, slot):
    a = (-_HALF + _WIDTH * slot, -_HALF + _WIDTH * (slot + 1))
    z = (_HEIGHT * layer, _HEIGHT * (layer + 1))
    if layer % 2 == 0:
        return np.array([-_HALF, a[0], z[0]]), np.array([_HALF, a[1], z[1]])
    return np.array([a[0], -_HALF, z[0]]), np.array([a[1], _HALF, z[1]])


def removable(layers, removed):
    """(layer, slot) -> True if that block may be pulled.  Jenga: only layers below the highest
    complete layer; and the layer must stay standing -- two blocks left, or just the middle one."""
    occ = {(l, s): (l, s) not in removed for l in range(layers) for s in range(3)}
    counts = [sum(occ[(l, s)] for s in range(3)) for l in range(layers)]
    top_complete = max([l for l in range(layers) if counts[l] == 3], default=-1)
    out = {}
    for (l, s), here in occ.items():
        if not here:
            continue
        if l >= top_complete:
            out[(l, s)] = False
            continue
        rest = [u for u in range(3) if u != s and occ[(l, u)]]
        out[(l, s)] = len(rest) >= 2 or rest == [1]
    return out


_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def _ruler(img, K, R, t, cam, layers):
    """Read-off labels so a block can be named: layer numbers up a corner, slot digits along the base.
    yellow digits = slot for layers 1,3,5,..   cyan digits = slot for layers 2,4,6,.."""
    F = cv2.FONT_HERSHEY_SIMPLEX
    YEL, CYA, WHT = (0, 215, 255), (255, 220, 0), (255, 255, 255)

    def label(text, p, col, dx=0):
        c = R @ np.asarray(p, float) + t
        if c[2] <= 0:
            return
        u = K @ c
        xy = (int(u[0] / u[2]) - 6 + dx, int(u[1] / u[2]) + 5)
        cv2.putText(img, text, xy, F, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, xy, F, 0.55, col, 1, cv2.LINE_AA)

    sx = 1.0 if cam[0] >= 0 else -1.0
    sy = 1.0 if cam[1] >= 0 else -1.0
    for s in range(3):
        c = -_HALF + _WIDTH * (s + 0.5)
        label(str(s), (sx * (_HALF + 18), c, 0), YEL)
        label(str(s), (c, sy * (_HALF + 18), 0), CYA)
    # layer numbers sit ON the tower's near vertical edge (so they line up with the right layer from any
    # angle); only the text is nudged sideways on screen, away from the tower
    def sx_of(p):
        c = R @ np.asarray(p, float) + t
        return (K @ c)[0] / (K @ c)[2]
    edge = (sx * _HALF, sy * _HALF, 0.0)
    out = (sx * (_HALF + 25), sy * (_HALF + 25), 0.0)
    side = 14 if sx_of(out) >= sx_of(edge) else -26
    for l in range(layers):
        label(str(l + 1), (edge[0], edge[1], _HEIGHT * (l + 0.5)), WHT, dx=side)
    lines = (("white: layer, counted from the bottom", WHT),
             ("yellow digit: slot of layers 1,3,5,7,9", YEL),
             ("cyan digit: slot of layers 2,4,6,8,10", CYA))
    h = img.shape[0]
    cv2.rectangle(img, (4, h - 4 - 22 * len(lines)), (330, h - 4), (0, 0, 0), -1)
    for i, (text, col) in enumerate(lines):
        cv2.putText(img, text, (10, h - 10 - 22 * (len(lines) - 1 - i)), F, 0.5, col, 1, cv2.LINE_AA)


_last_pose = {}      # kind -> (R, t, time): a phone on a stand does not move in a second


def _draw_overlay(img, K, dist, dets, tower, kind=None):
    """Pose from the markers in this very frame, then every block in red or green."""
    if not HAVE_POSE:
        return None
    sol = None
    if len(dets) >= 3:
        d = {i: c for i, c in dets.items()}
        if dist is not None and len(dist):
            d = {i: cv2.undistortPoints(c.reshape(-1, 1, 2), K, dist, P=K).reshape(4, 2) for i, c in d.items()}
        sol = _tp.solve(d, K, layout=_tp.LAYOUTS["A"])
        if sol and sol[2] > 10.0:
            sol = None
    if sol:
        R, t, _ = sol
        if kind:
            _last_pose[kind] = (R, t, time.time())
    else:                                  # a dropped marker for a moment: keep the last good pose briefly
        held = _last_pose.get(kind)
        if not held or time.time() - held[2] > 1.5:
            return None
        R, t, _ = held
    layers = int(tower["layers"])
    removed = {tuple(r) for r in tower["removed"]}
    ok = removable(layers, removed)
    cam = -R.T @ t
    items = []
    for (l, s), good in ok.items():
        lo, hi = _box(l, s)
        items.append((np.linalg.norm((lo + hi) / 2 - cam), l, s, good, lo, hi))
    items.sort(reverse=True)               # far blocks first, so near ones paint over them
    fill = img.copy()
    for _d, l, s, good, lo, hi in items:
        col = GREEN if good else RED
        C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        uvw = (C @ R.T + t) @ K.T
        if (uvw[:, 2] <= 0).any():
            continue
        uv = (uvw[:, :2] / uvw[:, 2:3]).astype(np.int32)
        centre = (lo + hi) / 2
        for f in _FACES:
            P = C[list(f)]
            fc = P.mean(axis=0)
            n = np.cross(P[1] - P[0], P[3] - P[0])
            n = n / (np.linalg.norm(n) + 1e-9)
            if n @ (fc - centre) < 0:
                n = -n
            ax = int(np.argmax(np.abs(n)))
            # only the tower's outer surface: faces that sit on the footprint edge (and the very top)
            outer = (ax < 2 and abs(fc[ax]) > _HALF - 0.1 and np.sign(n[ax]) == np.sign(fc[ax])) or \
                    (ax == 2 and n[2] > 0 and l == layers - 1)
            if not outer or n @ (cam - fc) <= 0:
                continue
            quad = uv[list(f)]
            cv2.fillConvexPoly(fill, quad, col)
            cv2.polylines(img, [quad.reshape(-1, 1, 2)], True, col, 2, cv2.LINE_AA)
    cv2.addWeighted(fill, 0.38, img, 0.62, 0, img)
    _ruler(img, K, R, t, cam, layers)
    return {"green": int(sum(ok.values())), "red": int(len(ok) - sum(ok.values()))}


def _annotate(jpeg_bytes, kind=None):
    """Draw detected markers on the preview; return (jpeg, [ids])."""
    if not HAVE_CV:
        return jpeg_bytes, []
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes, []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _DETECTOR.detectMarkers(gray)
    found, dets = [], {}
    if ids is not None:
        found = sorted(int(i) for i in ids.flatten() if int(i) < 4)
        cv2.aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
        dets = {int(i): c.reshape(4, 2).astype(np.float64) for c, i in zip(corners, ids.flatten()) if int(i) < 4}
    stats = None
    if kind:
        h, w = img.shape[:2]
        K, dist = _k_iphone(w) if kind == "iphone" else _k_android(w, h)
        if K is not None:
            with _lock:
                tower = dict(_state["tower"])
            try:
                stats = _draw_overlay(img, K, dist, dets, tower, kind)
            except Exception:
                stats = None
        with _lock:
            _state["overlay" if kind == "iphone" else "android_overlay"] = stats
    ok, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return (out.tobytes() if ok else jpeg_bytes), found


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body=b"", ctype="text/plain", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    # --- viewer -------------------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/preview.jpg":
            with _lock:
                jpeg = _state["jpeg"]
            if jpeg is None:
                self._send(404)
            else:
                self._send(200, jpeg, "image/jpeg")
        elif path == "/coverage.txt":
            try:
                body = _coverage_text().encode()
            except Exception as e:
                body = f"error: {type(e).__name__}: {e}".encode()
            self._send(200, body, "text/plain; charset=utf-8")
        elif path == "/cam":
            self._send(200, CAM_PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/preview_raw.jpg":
            with _lock:
                jpeg = _state["jpeg_raw"]
            if jpeg is None:
                self._send(404)
            else:
                self._send(200, jpeg, "image/jpeg")
        elif path in ("/android.jpg", "/android_raw.jpg"):
            with _lock:
                jpeg = _state["android_jpeg" if path == "/android.jpg" else "android_raw"]
            if jpeg is None:
                self._send(404)
            else:
                self._send(200, jpeg, "image/jpeg")
        elif path == "/status.json":
            now = time.time()
            with _lock:
                pt = list(_preview_times)
                ut = [t for t in _upload_times if now - t < 5.0]
                s = dict(_state)
                last_up = _upload_times[-1] if _upload_times else None
            fps = (len(pt) - 1) / (pt[-1] - pt[0]) if len(pt) > 2 and pt[-1] > pt[0] else 0.0
            body = json.dumps({
                "cv": HAVE_CV, "markers": s["markers"], "capturing": s["capturing"],
                "saved": s["saved"], "marks": s["marks"], "want": s["want"], "tracking": s["tracking"], "files": s["files"],
                "preview_age": (now - s["preview_t"]) if s["preview_t"] else None,
                "preview_fps": fps, "upload_rate": len(ut) / 5.0,
                "upload_age": (now - last_up) if last_up else None,
                "android_markers": s["android_markers"], "android_saved": s["android_saved"],
                "android_age": (now - s["android_t"]) if s["android_t"] else None,
                "android_size": s["android_size"], "android_dbg": list(_android_dbg)[-12:], "tower": s["tower"], "tracker_msg": s["tracker_msg"], "tracker_on": s["tracker_on"], "dataset": s["dataset"], "autotower": s["autotower"], "last_label": s["last_label"], "label_stats": s["label_stats"], "marks_pending": sum(1 for m in s["marks_log"] if not m["labelled"]),
                "want_mesh": s["want_mesh"], "mesh_files": s["mesh_files"],
                "overlay": s["overlay"], "android_overlay": s["android_overlay"],
            }).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"Not Found")

    # --- phone --------------------------------------------------------------------------
    def do_POST(self):
        if self.path == "/cmd/tower":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                layers = max(1, min(18, int(body.get("layers", 10))))
                removed = [[int(a), int(b)] for a, b in body.get("removed", [])
                           if 0 <= int(a) < layers and 0 <= int(b) < 3]
                with _lock:
                    _state["tower"] = {"layers": layers, "removed": removed}
                    _state["tower_epoch"] += 1
                self._send(200, b"OK")
            except Exception as e:
                self._send(400, f"bad tower: {e}".encode())
            return
        if self.path.startswith("/cmd/mesh/"):
            with _lock:
                _state["want_mesh"] = self.path.endswith("/on")
            self._send(200, b"OK")
            return
        if self.path == "/cmd/label":
            try:
                import move_infer as mi
                text = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}").get("text", "")
                pos, layer = mi.parse_text(text)
                with _lock:
                    todo = [m for m in _state["marks_log"] if not m["labelled"]]
                    last = _state["last_label"]
                if pos is None and layer is not None and last and last.get("status") == "done":
                    # a bare number corrects the previous answer
                    with _lock:
                        if layer != last.get("layer") and not last.get("corrected"):
                            _state["label_stats"]["wrong"] += 1
                        _state["last_label"] = dict(last, corrected=True, message=last["message"].split(". Wrong?")[0].split(". NOT CONFIDENT")[0]
                                                    + f". Corrected by you to layer {layer}")
                    try:
                        ds = _state["dataset"].get("name")
                        with open(os.path.join(HERE, "dataset", ds, "labels.jsonl"), "a") as f:
                            f.write(json.dumps({"seq": last["seq"], "correction": layer, "time": time.strftime("%H:%M:%S")}) + "\n")
                    except Exception:
                        pass
                    self._send(200, b"OK")
                    return
                if pos is None:
                    with _lock:
                        _state["last_label"] = {"status": "error", "message": "type left, middle or right (optionally plus the layer number)"}
                    self._send(200, b"OK")
                    return
                if not todo:
                    with _lock:
                        _state["last_label"] = {"status": "error", "message": "press MARK first, then type"}
                    self._send(200, b"OK")
                    return
                entry = todo[-1]
                with _lock:
                    entry["labelled"] = True
                threading.Thread(target=_label_job, args=(entry, pos, layer), daemon=True).start()
                self._send(200, b"OK")
            except Exception as e:
                self._send(400, f"bad label: {e}".encode())
            return
        if self.path == "/cmd/redetect":
            with _lock:
                _state["autotower_redetect"] = True
            self._send(200, b"OK")
            return
        if self.path.startswith("/cmd/dataset/"):
            on = self.path.endswith("/on")
            with _lock:
                _state["dataset"]["on"] = on
                _state["want"] = True if on else False        # start / stop the iPhone recording too
            self._send(200, b"OK")
            return
        if self.path.startswith("/cmd/"):
            cmd = self.path.split("/cmd/")[1]
            with _lock:
                if cmd == "start":
                    _state["want"] = True
                elif cmd == "stop":
                    _state["want"] = False
                elif cmd == "mark":
                    _state["mark_seq"] += 1
                    _state["marks_log"].append({"seq": _state["mark_seq"], "t_ms": int(time.time() * 1000), "labelled": False})
                else:
                    self._send(404, b"unknown command")
                    return
            self._send(200, b"OK")
            return
        if self.path == "/android":
            try:
                t_hdr = time.time()
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n)
                t_body = time.time()
                with _lock:
                    _android_dbg.append({"at": t_hdr, "body_s": round(t_body - t_hdr, 3), "kb": n // 1024,
                                         "ip": self.client_address[0], "ua": (self.headers.get("User-Agent") or "")[:40],
                                         "conn": self.headers.get("Connection", ""), "proto": self.request_version,
                                         "draw": self.headers.get("X-Draw"), "enc": self.headers.get("X-Enc"),
                                         "up": self.headers.get("X-Up")})
                jpeg, ids = _annotate(raw, "android")
                now = time.time()
                with _lock:
                    recording = bool(_state["capturing"]) and (now - _state["preview_t"] < 3.0)
                    _state.update(android_jpeg=jpeg, android_raw=raw, android_markers=ids, android_t=now)
                    _android_times.append(now)
                saved = 0
                if recording:
                    ms = int(now * 1000)
                    with open(os.path.join(SAVE_DIR, f"{ms}_android.jpg"), "wb") as f:
                        f.write(raw)
                    with _lock:
                        _state["android_saved"] += 1
                    if HAVE_CV and _state["android_size"] is None:
                        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
                        if im is not None:
                            size = [int(im.shape[1]), int(im.shape[0])]
                            with _lock:
                                _state["android_size"] = size
                            json.dump({"width": size[0], "height": size[1]},
                                      open(os.path.join(SAVE_DIR, "android_info.json"), "w"))
                with _lock:
                    saved = _state["android_saved"]
                self._send(200, json.dumps({"markers": ids, "recording": recording, "saved": saved}).encode(),
                           "application/json")
            except Exception as e:
                self._send(500, f"android error: {e}".encode())
            return
        if self.path == "/preview":
            try:
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n)
                jpeg, ids = _annotate(raw, "iphone")
                with _lock:
                    _state.update(
                        jpeg=jpeg, jpeg_raw=raw, markers=ids, preview_t=time.time(),
                        capturing=self.headers.get("X-Capturing") == "1",
                        saved=int(self.headers.get("X-Saved", 0) or 0),
                        tracking=self.headers.get("X-Tracking", "?"),
                        marks=int(self.headers.get("X-Marks", 0) or 0))
                    _preview_times.append(time.time())
                    if _state["want"] is not None and _state["capturing"] == _state["want"]:
                        _state["want"] = None
                    reply = {"capture": _state["want"], "mark_seq": _state["mark_seq"], "mesh": _state["want_mesh"]}
                self._send(200, json.dumps(reply).encode(), "application/json")
            except Exception as e:
                self._send(500, f"preview error: {e}".encode())
            return

        if self.path == "/upload":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                content_type = self.headers.get("Content-Type", "")
                if "boundary=" not in content_type:
                    self._send(400, b"Bad Request: No boundary")
                    return
                boundary = content_type.split("boundary=")[1].encode()
                body = self.rfile.read(content_length)
                for part in body.split(b"--" + boundary):
                    if b"filename=" not in part:
                        continue
                    try:
                        start = part.find(b'filename="') + len(b'filename="')
                        end = part.find(b'"', start)
                        filename = os.path.basename(part[start:end].decode("utf-8"))
                        data_start = part.find(b"\r\n\r\n") + 4
                        file_data = part[data_start:len(part) - 2]
                        with open(os.path.join(SAVE_DIR, filename), "wb") as f:
                            f.write(file_data)
                        with _lock:
                            _state["files"] += 1
                            if filename.endswith("_mesh.bin"):
                                _state["mesh_files"] += 1
                            _upload_times.append(time.time())
                        print(f"✅ Saved: {filename} ({len(file_data):,} bytes)")
                    except Exception as e:
                        print(f"❌ Error parsing file: {e}")
                self._send(200, b"OK")
            except Exception as e:
                print(f"❌ Server error: {e}")
                self._send(500, f"Server Error: {e}".encode())
            return

        self._send(404, b"Not Found")

    def log_message(self, format, *args):
        pass  # per-request logging is replaced by the "Saved:" lines above


# --- live tower tracker: frames + MARK taps -> which blocks are present ----------------------
def _frame_ready(ts):
    d = SAVE_DIR
    if not (os.path.exists(f"{d}/{ts}_rgb.jpg") and os.path.exists(f"{d}/{ts}_intrinsics.json")):
        return False
    import glob as _g
    dep, conf = _g.glob(f"{d}/{ts}_depth_*.bin"), _g.glob(f"{d}/{ts}_conf_*.bin")
    return bool(dep and conf and os.path.getsize(dep[0]) == 256 * 192 * 4 and os.path.getsize(conf[0]) == 256 * 192)


def _tracker_loop():
    import glob as _g
    try:
        import live_tracker as lt
    except Exception as e:
        with _lock:
            _state["tracker_msg"] = f"tracker unavailable: {e}"
        return
    tr = lt.LiveTracker(layers=10)
    seen, seen_marks, was_cap, epoch = set(), set(), False, 0
    while True:
        time.sleep(0.4)
        try:
            with _lock:
                cap = bool(_state["capturing"]) and (time.time() - _state["preview_t"] < 5)
                tower, tep = dict(_state["tower"]), _state["tower_epoch"]
            if cap and not was_cap:                      # START: fresh reference, ignore old files
                tr.reset(tower["layers"])
                seen = {os.path.basename(f).split("_")[0] for f in _g.glob(f"{SAVE_DIR}/*_intrinsics.json")}
                seen_marks = set(_g.glob(f"{SAVE_DIR}/*_mark.json"))
                epoch = tep
                with _lock:
                    _state["tracker_on"] = True
            was_cap = cap
            if tep != epoch:                              # tower edited by hand in the viewer
                tr.layers = int(tower["layers"])
                tr.present = {(l, s) for l in range(tr.layers) for s in range(3)} - \
                             {tuple(r) for r in tower["removed"]}
                tr.prev = {k: v for k, v in tr.prev.items()}
                epoch = tep
            with _lock:
                ds_on = _state["dataset"]["on"]
            if ds_on:
                with _lock:
                    _state["tracker_msg"] = "collecting training data (tracker paused)"
                continue
            if not cap and not tr.pending_marks:
                with _lock:
                    _state["tracker_on"] = False
                continue
            new = sorted(ts for ts in (os.path.basename(f).split("_")[0] for f in _g.glob(f"{SAVE_DIR}/*_intrinsics.json"))
                         if ts not in seen)
            for ts in new:
                if not _frame_ready(ts):
                    if time.time() - os.path.getmtime(f"{SAVE_DIR}/{ts}_intrinsics.json") < 6:
                        break                              # still arriving; try again next tick
                    seen.add(ts)
                    continue
                seen.add(ts)
                tr.add_frame(int(ts), _tp.load_frame(ts))
            for f in sorted(set(_g.glob(f"{SAVE_DIR}/*_mark.json")) - seen_marks):
                seen_marks.add(f)
                try:
                    tr.mark(json.load(open(f))["t_ms"], time.time())
                except Exception:
                    pass
            results = tr.poll(time.time())
            with _lock:
                _state["tracker_msg"] = tr.msg
                if results and tr.ready:
                    _state["tower"] = tr.tower()
                    epoch = _state["tower_epoch"]          # our own write is not a manual edit
        except Exception as e:
            with _lock:
                _state["tracker_msg"] = f"tracker error: {type(e).__name__}: {e}"


# --- training-data capture: same face-crop extractor, run live while a game is played -------------
def _dataset_totals():
    n = 0
    try:
        for d in os.listdir(os.path.join(HERE, "dataset")):
            f = os.path.join(HERE, "dataset", d, "index.jsonl")
            if os.path.exists(f):
                with open(f) as fh:
                    n += sum(1 for _ in fh)
    except Exception:
        pass
    return n


def _dataset_loop():
    import glob as _g
    try:
        import face_crops as fc
    except Exception as e:
        with _lock:
            _state["dataset"]["error"] = str(e)
        return
    writer, started_marks = None, set()
    while True:
        time.sleep(0.8)
        try:
            with _lock:
                on = _state["dataset"]["on"]
            if on and writer is None:
                name = time.strftime("live_%Y%m%d_%H%M%S")
                writer = fc.DatasetWriter(SAVE_DIR, os.path.join(HERE, "dataset", name), stride=1)   # every frame: more clean reference crops
                writer.skip_existing()
                started_marks = set(_g.glob(f"{SAVE_DIR}/*_mark.json"))
                with _lock:
                    _state["dataset"].update(name=name, frames=0, crops=0)
            if writer is not None:
                writer.step()
                with _lock:
                    _state["dataset"].update(frames=writer.counts["frames"], crops=writer.counts["crops"],
                                             total_crops=_dataset_totals())
                if not on:                               # stopped: finish and record the taps
                    marks = []
                    for f in set(_g.glob(f"{SAVE_DIR}/*_mark.json")) - started_marks:
                        try:
                            marks.append(json.load(open(f))["t_ms"])
                        except Exception:
                            pass
                    json.dump({"t0_ms": writer.t0, "marks_ms": sorted(marks)},
                              open(os.path.join(writer.out, "marks.json"), "w"))
                    writer.close()
                    writer = None
        except Exception as e:
            with _lock:
                _state["dataset"]["error"] = f"{type(e).__name__}: {e}"


# --- automatic tower setup: layer count + which way the blocks run, from the two cameras ----------------
def _autotower_loop():
    """Runs only when the Re-detect button is pressed: a short burst of frames from both cameras, averaged."""
    try:
        import tower_auto as ta
    except Exception as e:
        with _lock:
            _state["autotower"] = {"status": f"unavailable: {e}"}
        return

    def grab():
        with _lock:
            ip, an = _state["jpeg_raw"], _state["android_raw"]
            fresh_i = time.time() - _state["preview_t"] < 4
            fresh_a = _state["android_t"] and time.time() - _state["android_t"] < 4
        dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) if b else None
        return {"iphone": dec(ip) if fresh_i else None, "android": dec(an) if fresh_a else None}

    def say(text, **extra):
        with _lock:
            _state["autotower"] = dict({"status": text}, **extra)

    while True:
        time.sleep(0.4)
        try:
            with _lock:
                go = _state["autotower_redetect"]
                busy = bool(_state["capturing"]) or _state["dataset"]["on"]
            if not go:
                continue
            with _lock:
                _state["autotower_redetect"] = False
            if busy:
                say("not while a game is being recorded")
                continue
            final, t_start = None, time.time()
            for attempt in range(1, 4):
                say(f"measuring (attempt {attempt}/3, keep both phones still)...")
                r = ta.analyse_burst(grab, n=8, interval=0.3)
                if not r["ok"]:
                    say(f"attempt {attempt}/3 failed: {r['reason']}")
                    time.sleep(1.0)
                    continue
                L, axis, ratio = r["layers"], r["axis"], r["ratio"]
                expect_x = ((L - 1) % 2 == 0)
                if (axis == "x") != expect_x:
                    if ratio < 2.0:
                        say(f"{L} layers, but not sure which way the top blocks run (evidence x{ratio:.1f}). "
                            "Is the top layer complete?")
                        continue
                    k0 = _tp.TOWER_ROT
                    _tp.set_tower_rotation(min(((k0 + 1) % 4, (k0 - 1) % 4), key=lambda k: min(k, 4 - k)))
                    _last_pose.clear()
                    time.sleep(1.0)
                    continue                         # the frame changed: measure again to confirm
                final = (L, axis, r)
                break
            if final:
                L, axis, r = final
                with _lock:
                    _state["tower"] = {"layers": L, "removed": []}
                    _state["tower_epoch"] += 1
                    _state["autotower_lock"] = True
                    _state["autotower_last"] = f"{L} layers, top blocks along {axis}"
                say(f"done in {time.time() - t_start:.0f} s: {L} layers, top blocks along {axis}. Locked until you press Re-detect",
                    layers=L, axis=axis, z=r["z"], ncc=r["ncc"], cams=r["cams"])
        except Exception as e:
            say(f"error: {type(e).__name__}: {e}")


# --- "which layer?": the player types left / middle / right after a MARK, the face crops find the layer ----
def _label_job(entry, pos, user_layer):
    """Wait for the 'after' pictures, then find which layer's block in that column changed most."""
    import move_infer as mi
    def say(**kw):
        with _lock:
            _state["last_label"] = dict(seq=entry["seq"], pos=pos, **kw)
    try:
        wait = entry["t_ms"] / 1000.0 + 5.2 - time.time()
        say(status="waiting for the after-pictures...")
        if wait > 0:
            time.sleep(wait)
        with _lock:
            ds = dict(_state["dataset"]); layers = int(_state["tower"]["layers"])
            prev = [m for m in _state["marks_log"] if m["seq"] < entry["seq"]]
        if not ds.get("name"):
            return say(status="error", message="data capture was not on, so there are no face crops to look at")
        run_dir = os.path.join(HERE, "dataset", ds["name"])
        held = _last_pose.get("iphone")
        if not held:
            return say(status="error", message="the iPhone has not found the markers, so I can't map left/middle/right")
        K = _k_iphone(960)[0]
        slots = mi.slots_for_positions(held[0], held[1], K, layers)
        with open(os.path.join(run_dir, "index.jsonl")) as f:
            start_ms = json.loads(f.readline())["ts_ms"]
        gone = mi.already_removed(run_dir, entry["seq"])
        cands = [l for l in range(max(2, layers - 1)) if (pos, l) not in gone]     # nothing is ever put back
        r = mi.infer(run_dir, entry["t_ms"], prev[-1]["t_ms"] if prev else None, start_ms, pos, slots, cands)
        rec = {"seq": entry["seq"], "mark_ms": entry["t_ms"], "pos": pos, "user_layer": user_layer,
               "time": time.strftime("%H:%M:%S"), "result": r}
        if not r["ok"]:
            say(status="error", message=r["reason"])
        else:
            agree = (user_layer == r["layer"]) if user_layer else None
            with _lock:
                _state["label_stats"]["moves"] += 1
                if user_layer and not agree:
                    _state["label_stats"]["wrong"] += 1
            msg = f"{pos} column -> layer {r['layer']} (score {r['z']:.1f}, next best layer {r['runner_up']} at {r['runner_up_z']:.1f})"
            if r.get("suspicious"):
                msg += ". Unusually large: a hand or shadow may be over that face"
            elif not r["confident"]:
                msg += ". NOT CONFIDENT: please check, and type the real layer if it is wrong"
            if user_layer:
                msg += f". You said layer {user_layer}: {'MATCH' if agree else 'different'}"
            else:
                msg += ". Wrong? type the real layer number and press Enter"
            say(status="done", message=msg, layer=r["layer"], confident=r["confident"], top=r["top"])
        with open(os.path.join(run_dir, "labels.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        say(status="error", message=f"{type(e).__name__}: {e}")


def _coverage_text():
    """Which blocks can the two phones, where they stand right now, actually judge?  One cell per block."""
    import face_crops as fc, move_infer as mi
    with _lock:
        layers = int(_state["tower"]["layers"])
    poses = {n: _last_pose.get(n) for n in ("iphone", "android")}
    if not poses["iphone"]:
        return "the iPhone has not found the markers yet, so I can't tell (needs 3+ markers)"
    K = _k_iphone(960)[0]
    slots = mi.slots_for_positions(poses["iphone"][0], poses["iphone"][1], K, layers)
    def cos_of(pose, g):
        corners, c, n = g
        cam = -pose[0].T @ pose[1]
        to = cam - c
        return float(n @ to) / (np.linalg.norm(to) + 1e-9)
    lines = ["          left  middle  right      (best view of that block: # good  + ok  . poor  - blind)"]
    weak = []
    for layer in range(layers - 1, -1, -1):
        cells = []
        for pos in ("left", "middle", "right"):
            slot = slots[layer][pos]
            best, who = 0.0, ""
            for name, pose in poses.items():
                if not pose:
                    continue
                for sign in (1, -1):
                    c = cos_of(pose, fc.face_geometry(layer, slot, sign))
                    if c > best: best, who = c, name[0]
                if slot in (0, 2):
                    g = fc.side_geometry(layer, slot, -1 if slot == 0 else 1)
                    if g is not None:
                        c = cos_of(pose, g)
                        if c > best: best, who = c, name[0]
            sym = "#" if best >= 0.45 else "+" if best >= 0.30 else "." if best >= 0.15 else "-"
            cells.append(f"{sym} {best:.2f}{who}")
            if best < 0.30:
                weak.append(f"L{layer + 1} {pos}")
        lines.append(f"  layer {layer + 1:2d}  " + "   ".join(cells) + ("   <- pulled blocks here are hard to see" if layer < layers - 1 and any(c[0] in ".-" for c in cells) else ""))
    lines.append("")
    lines.append("letters: i = iPhone sees it best, a = Android.  Layer " + str(layers) + " (top) can't be pulled anyway.")
    lines.append(f"{len(weak)} blocks are hard to judge from here: " + (", ".join(weak[:14]) + (" ..." if len(weak) > 14 else "") if weak else "none"))
    return "\n".join(lines)


class TLSThreadingServer(ThreadingHTTPServer):
    """HTTPS where each connection does its own TLS handshake in its own thread.
    Wrapping the listening socket instead does the handshake in the accept loop, so one idle
    connection (phone browsers open speculative ones) freezes every other request."""
    daemon_threads = True

    def __init__(self, addr, handler, ctx):
        super().__init__(addr, handler)
        self._ctx = ctx

    def finish_request(self, request, client_address):
        try:
            request.settimeout(10)
            request = self._ctx.wrap_socket(request, server_side=True)
        except Exception:
            return
        super().finish_request(request, client_address)


def _ensure_cert(ip):
    """Self-signed certificate for the Mac's IP, so phone browsers allow camera access."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs")
    os.makedirs(d, exist_ok=True)
    cert, key, stamp = (os.path.join(d, n) for n in ("cert.pem", "key.pem", "ip.txt"))
    if os.path.exists(cert) and os.path.exists(key) and os.path.exists(stamp) and open(stamp).read() == ip:
        return cert, key
    cfg = os.path.join(d, "openssl.cnf")
    open(cfg, "w").write(
        "[req]\ndistinguished_name=dn\nx509_extensions=v3\nprompt=no\n[dn]\nCN=lidar-capture\n"
        f"[v3]\nsubjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost\n")
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key,
                    "-out", cert, "-days", "825", "-config", cfg],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    open(stamp, "w").write(ip)
    return cert, key


if __name__ == "__main__":
    PORT = int(os.environ.get("LIDAR_PORT", 8000))
    try:
        server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = "?"
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except OSError:
            pass
        finally:
            s.close()
        print(f"🚀 Server running on port {PORT}")
        print(f"📁 Saving files to: {SAVE_DIR}")
        print(f"🌐 Enter this IP on the phone: {ip}")
        print(f"🖥  Live view (open in a browser):  http://localhost:{PORT}/")
        if HAVE_POSE and HAVE_CV:
            threading.Thread(target=_tracker_loop, daemon=True).start()
            threading.Thread(target=_dataset_loop, daemon=True).start()
            threading.Thread(target=_autotower_loop, daemon=True).start()
            print("🧱 Live tower tracker on: START, keep hands out 5 s, then MARK after each move")
        HPORT = int(os.environ.get("LIDAR_HTTPS_PORT", 8443))
        try:
            cert, key = _ensure_cert(ip)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            httpsd = TLSThreadingServer(("0.0.0.0", HPORT), Handler, ctx)
            threading.Thread(target=httpsd.serve_forever, daemon=True).start()
            print(f"📷 Android camera page (open in Chrome on the phone):  https://{ip}:{HPORT}/cam")
            print("   Chrome will warn about the certificate: Advanced -> Proceed.")
        except Exception as e:
            print(f"ℹ️  HTTPS camera page unavailable: {e}")
        if not HAVE_CV:
            print("ℹ️  Marker overlay is off. Start with .venv/bin/python iphone_saved.py to enable it.")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
