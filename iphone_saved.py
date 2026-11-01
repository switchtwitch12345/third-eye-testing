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
import glob
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
_OLD_MAC_DIR = "/Users/mitul/Desktop/lidar-capture/iphone_saved"   # kept so existing Mac setups don't move
SAVE_DIR = os.environ.get("LIDAR_SAVE_DIR") or (
    _OLD_MAC_DIR if os.path.isdir(_OLD_MAC_DIR) else os.path.join(HERE, "iphone_saved"))
sys.path.insert(0, HERE)
try:
    import tower_points as _tp          # marker layout + pose solver
    _tp.SAVE_DIR = SAVE_DIR
    HAVE_POSE = True
except Exception:
    HAVE_POSE = False
os.makedirs(SAVE_DIR, exist_ok=True)

_lock = threading.Lock()
_new_frame = threading.Condition(_lock)   # notified when a camera frame lands, for the viewer's long-poll
_state = {
    "jpeg": None, "markers": [], "preview_t": 0.0, "capturing": False, "saved": 0,
    "tracking": "?", "marks": 0, "want": None, "mark_seq": 0, "files": 0,
    "android_jpeg": None, "android_raw": None, "jpeg_raw": None, "android_markers": [], "android_t": 0.0,
    "android_saved": 0, "android_size": None, "preview_seq": 0, "android_seq": 0,
    "tower": {"layers": 10, "removed": []}, "loose": [], "loose_info": None, "game_over": None, "height_info": None, "tower_fit": None, "overlay": None, "android_overlay": None, "labels": {}, "shapes": {}, "placed_on_top": True, "auto": None, "edge": {"iphone": None, "android": None}, "locked": False, "view_rot": {"iphone": 0},
    "autotower": {"status": "waiting for both cameras"}, "marks_log": [], "last_label": None, "label_stats": {"moves": 0, "wrong": 0}, "autotower_lock": True, "autotower_last": "none yet", "autotower_redetect": False, "autotower_auto": False, "lidar_top": None, "depth_dbg": {"n": 0, "why": None}, "tower_epoch": 0, "tracker_msg": "tracker idle - press START", "tracker_on": False,
    "dataset": {"on": False, "name": None, "frames": 0, "crops": 0, "total_crops": 0, "total_frames": 0}, "started": time.time(), "image_size": None,
}
_preview_times = collections.deque(maxlen=40)
_upload_times = collections.deque(maxlen=400)
_android_times = collections.deque(maxlen=40)
_android_dbg = collections.deque(maxlen=60)
# The app streams much faster than the old ~3 fps; disk saves stay at about the old rate unless asked for more.
ANDROID_SAVE_FPS = float(os.environ.get("ANDROID_SAVE_FPS", 3))
ANDROID_DRAW_MIN_S = 1.0 / float(os.environ.get("ANDROID_DRAW_FPS", 25))   # most the Android picture is annotated per second

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
</style></head><body><div id="nomoves" style="display:none;position:fixed;top:14px;left:50%;transform:translateX(-50%);z-index:50;background:#c0281c;color:#fff;font:800 30px -apple-system,Roboto,sans-serif;padding:14px 34px;border-radius:12px;box-shadow:0 4px 24px #000a">NO MOVES LEFT &mdash; every remaining block would topple the tower</div><div class="wrap">
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
  <button id="bNew" style="background:#3b1d1d;border-color:#8f2f2f;margin-top:10px">NEW GAME (clear everything)</button>
  <div class="hint" style="margin-bottom:10px">Control everything from here. Don't touch the phone.</div>
  <h1 style="margin-top:6px;color:#ffb020">After MARK: which block did you pull?</h1>
  <div class="hint" style="margin-top:0">Press MARK after each move, hand off the tower. The system names the block itself: at 90% sure or more it
  adds it to the tower and the 3-D view and asks Yes / No (keys Y / N). Below that nothing changes: type <b>left</b>, <b>middle</b> or
  <b>right</b> and the layer, e.g. <b>l 5</b>. Needs data capture ON.</div>
  <input id="lbl" placeholder="left / middle / right  [layer]" autocomplete="off"
    style="width:100%;margin-top:8px;background:#22262d;color:#e8eaed;border:1px solid #333944;border-radius:8px;padding:10px;font-size:15px">
  <div id="autocard" style="display:none;margin-top:8px;padding:10px;border:1px solid #2fb36d;border-radius:8px;background:#12261c">
    <div id="automsg" style="font-weight:600"></div>
    <div id="autoyn" style="display:none;gap:10px;margin-top:10px">
      <button id="bYes" style="flex:1;background:#1f8f55;border-color:#2fb36d;color:#fff;font-size:16px;font-weight:700;padding:12px 8px">Yes, that's right (Y)</button>
      <button id="bNo" style="flex:1;background:#8f2f2f;border-color:#d85c5c;color:#fff;font-size:16px;font-weight:700;padding:12px 8px">No (N)</button>
    </div></div>
  <div id="lres" class="chip idle" style="font-size:14px;margin-top:8px">no label yet</div>
  <div class="row"><span>Frames saved</span><span class="big" id="saved">0</span></div>
  <div class="row"><span>Moves marked</span><span class="big" id="marks">0</span></div>
  <h1 style="margin-top:22px">Setup</h1>
  <div id="mk" class="chip idle">markers: –</div>
  <div class="row"><span>Marker IDs</span><span id="ids">–</span></div>
  <div class="row"><span>Phone tracking</span><span id="trk">–</span></div>
  <h1 style="margin-top:22px">Camera 2 (Android browser)</h1>
  <div id="amk" class="chip idle">not connected</div>
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
  <label class="row" style="cursor:pointer"><span>Each pulled block goes back on top</span><input id="tplaced" type="checkbox" checked></label>
  <div id="thand" class="hint" style="color:#ffb000;display:none">hand at the tower - judging paused until it leaves</div>
  <div id="tcount" class="hint">–</div>
  <div id="tauto" class="hint">auto-detect: waiting…</div>
  <button id="bRedetect">Re-detect tower (after rebuilding)</button>
  <button id="bLock">Lock overlay (when it looks perfect)</button>
  <button id="bAuto">Auto-detect: on</button>
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
let LAB={}, SHP={};
const LEG=[['white: layer, counted from the bottom','#fff'],['yellow digit: slot of layers 1,3,5,7,9','#ffd700'],['cyan digit: slot of layers 2,4,6,8,10','#00dcff']];
function drawTo(cv,img,rot,lab,shp){
  const x=cv.getContext('2d');
  const iw=img.naturalWidth||img.width, ih=img.naturalHeight||img.height;
  const sw=rot%180? ih:iw, sh=rot%180? iw:ih;
  if(cv.width!==sw||cv.height!==sh){cv.width=sw;cv.height=sh}
  x.save();x.translate(cv.width/2,cv.height/2);x.rotate(rot*Math.PI/180);
  x.drawImage(img,-iw/2,-ih/2);x.restore();
  const k=iw/960, a=rot*Math.PI/180, co=Math.cos(a), si=Math.sin(a);
  const at=(px,py)=>{const dx=px-iw/2, dy=py-ih/2; return [cv.width/2+dx*co-dy*si, cv.height/2+dx*si+dy*co]};
  if(shp&&shp.w===iw&&shp.h===ih){                      // the overlay arrives as shapes; the browser draws them over the raw picture
    x.lineJoin='round';
    for(const it of shp.items){
      if(it[0]==='q'){
        const p=it[1].map(([px,py])=>at(px,py)); x.beginPath(); p.forEach((q,i)=>i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1])); x.closePath();
        x.globalAlpha=0.2; x.fillStyle=it[2]; x.fill(); x.globalAlpha=1; x.lineWidth=Math.max(1,k); x.strokeStyle=it[2]; x.stroke();
      }else if(it[0]==='m'){
        const p=it[1].map(([px,py])=>at(px,py)); x.beginPath(); p.forEach((q,i)=>i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1])); x.closePath();
        x.lineWidth=Math.max(1,k); x.strokeStyle='#00ff00'; x.stroke(); x.strokeStyle='#ff0000'; x.strokeRect(p[0][0]-3*k,p[0][1]-3*k,6*k,6*k);
      }else{
        x.font='bold '+Math.round(22*k)+'px sans-serif'; x.textAlign='center'; const w=x.measureText(it[1]).width;
        x.fillStyle='#000'; x.fillRect(cv.width/2-w/2-10*k,20*k,w+20*k,34*k); x.fillStyle='#ffb000'; x.fillText(it[1],cv.width/2,45*k); x.textAlign='start';
      }
    }
  }
  if(lab&&lab.w===iw&&lab.h===ih){   // labels are drawn upright, at the turned positions
    for(const [t,px,py,sc,col] of lab.items){
      const dx=px-lab.w/2, dy=py-lab.h/2, X=cv.width/2+dx*co-dy*si, Y=cv.height/2+dx*si+dy*co;
      x.font='bold '+Math.round(sc*34*k)+'px sans-serif'; x.lineWidth=4*k; x.strokeStyle='#000'; x.strokeText(t,X,Y);
      x.fillStyle=col; x.fillText(t,X,Y);
    }
  }
  const lh=Math.round(22*k); x.font=Math.round(15*k)+'px sans-serif';
  x.fillStyle='#000'; x.fillRect(4*k,cv.height-4*k-lh*3-6*k,330*k,lh*3+6*k);
  LEG.forEach(([t,col],i)=>{x.fillStyle=col;x.fillText(t,10*k,cv.height-10*k-lh*(2-i))});
}
// Long-poll: the server holds the request until a frame newer than `after` exists, so the view
// runs at whatever rate the phone sends, without re-fetching the same frame.
const shown={a:[],b:[]};   // when each camera's frames were drawn here, to compare with what the server receives
function shownFps(w){const t=shown[w],now=performance.now();while(t.length&&t[0]<now-2000)t.shift();
  return t.length<2?0:(t.length-1)*1000/(t[t.length-1]-t[0])}
async function feed(cv,url,which){
  let after=0;
  for(;;){
    try{
      const r=await fetch(url+'?after='+after);
      if(r.status===204) continue;
      if(!r.ok) throw 0;
      after=+(r.headers.get('X-Seq')||0);
      const bmp=await createImageBitmap(await r.blob());
      drawTo(cv,bmp,rots[which],which==='a'?LAB.iphone:LAB.android,which==='a'?SHP.iphone:SHP.android); bmp.close(); shown[which].push(performance.now());
    }catch(e){await new Promise(f=>setTimeout(f,600))}
  }
}
setInterval(async()=>{try{SHP=await (await fetch('/shapes.json')).json()}catch(e){}},150);
feed(c,'/preview.jpg','a');
feed(c2,'/android.jpg','b');
setInterval(()=>fetch('/cmd/viewrot',{method:'POST',body:JSON.stringify({iphone:rots.a})}),4000);
document.getElementById('rot').onclick=()=>rotate('a');
document.getElementById('rotb').onclick=()=>rotate('b');
document.getElementById('lbl').addEventListener('keydown',e=>{if(e.key==='Enter'){
  const v=e.target.value.trim(); if(v){fetch('/cmd/label',{method:'POST',body:JSON.stringify({text:v})}); e.target.value='';}}});
addEventListener('keydown',e=>{ if(e.target&&e.target.tagName==='INPUT') return;
  if(e.key==='r'||e.key==='R')rotate('a');
  if(e.key==='t'||e.key==='T')rotate('b');
  if(e.code==='Space'){e.preventDefault();cmd('mark')}
  if(e.key==='y'||e.key==='Y')fetch('/cmd/confirm/yes',{method:'POST'});
  if(e.key==='n'||e.key==='N')fetch('/cmd/confirm/no',{method:'POST'});});
function cmd(n){fetch('/cmd/'+n,{method:'POST'})}
document.getElementById('tapply').onclick=()=>{
  const rem=[...document.getElementById('tr').value.matchAll(/L\s*(\d+)\s*S\s*(\d)/gi)].map(m=>[+m[1]-1,+m[2]]);
  fetch('/cmd/tower',{method:'POST',body:JSON.stringify({layers:+document.getElementById('tl').value,removed:rem})});
};
let dataOn=false;
document.getElementById('bCover').onclick=async()=>{const p=document.getElementById('cover');p.style.display='block';p.textContent='measuring...';p.textContent=await (await fetch('/coverage.txt')).text();};
document.getElementById('bRedetect').onclick=()=>fetch('/cmd/redetect',{method:'POST'});
document.getElementById('bNew').onclick=()=>{ if(confirm('Start a new game? This stops recording and clears the pulled blocks, taps, answers and labels.')){ fetch('/cmd/newgame',{method:'POST'}); document.getElementById('lbl').value=''; document.getElementById('tr').value=''; } };
document.getElementById('bYes').onclick=()=>fetch('/cmd/confirm/yes',{method:'POST'});
document.getElementById('bNo').onclick=()=>fetch('/cmd/confirm/no',{method:'POST'});
document.getElementById('tplaced').onchange=e=>fetch('/cmd/placed/'+(e.target.checked?'on':'off'),{method:'POST'});
let isLocked=false;
document.getElementById('bLock').onclick=()=>fetch('/cmd/lock/'+(isLocked?'off':'on'),{method:'POST'});
let autoOn=true;
document.getElementById('bAuto').onclick=()=>fetch('/cmd/autotower/'+(autoOn?'off':'on'),{method:'POST'});
document.getElementById('bData').onclick=()=>fetch('/cmd/dataset/'+(dataOn?'off':'on'),{method:'POST'});
document.getElementById('bStart').onclick=()=>cmd('start');
document.getElementById('bStop').onclick=()=>cmd('stop');
document.getElementById('bMark').onclick=()=>cmd('mark');
function rotate(w){rots[w]=(rots[w]+90)%360;localStorage.setItem(w==='a'?'rotA':'rotB',rots[w])}
const $=id=>document.getElementById(id);
async function poll(){
  try{
    const s=await (await fetch('/status.json')).json(); LAB=s.labels||{};
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
      a.textContent='markers '+(s.android_markers.join(', ')||'none')+'  ·  '+s.android_fps.toFixed(1)+' fps (shown '+shownFps('b').toFixed(0)+')  ·  saved '+s.android_saved;
    }else{a.className='chip idle';a.textContent='not connected';}
    const tag=(el,ok,ids)=>{ if(!ok){el.className='tag';el.textContent='no signal';return}
      el.className='tag '+(ids.length>=3?'ok':ids.length===2?'warn':'bad');
      el.textContent='markers '+(ids.length?ids.join(', '):'none');};
    tag($('lbA'),live,s.markers);
    tag($('lbB'),s.android_age!==null&&s.android_age<10,s.android_markers);
    $('trk').textContent=s.tracking;
    const ds=s.dataset||{}; dataOn=!!ds.on;
    const db=$('bData'); db.textContent=dataOn?'■ Stop data capture':'Start data capture';
    db.style.background=dataOn?'#5a1a16':'#16324a';
    $('dinfo').textContent=dataOn?('RECORDING DATA · this run: '+ds.frames+' frames, '+ds.crops+' crops · all runs: '+(ds.total_crops||0)+' crops')
      :('off · '+(ds.total_crops||0)+' crops saved so far');
    if(s.tower&&document.activeElement!==$('tr')&&document.activeElement!==$('tl')){
      $('tr').value=s.tower.removed.map(b=>'L'+(b[0]+1)+'S'+b[1]).join(' '); $('tl').value=s.tower.layers; }
    $('nomoves').style.display=(s.no_moves||s.game_over)?'block':'none';
    $('nomoves').textContent=s.game_over?('GAME OVER \u2014 '+s.game_over.reason):'NO MOVES LEFT \u2014 every remaining block would topple the tower';
    const au=s.auto, ac=$('autocard');
    if(au){ ac.style.display='block'; $('automsg').textContent=au.message||au.status;
      const sure=au.status==='sure'&&!au.answered; $('autoyn').style.display=sure?'flex':'none';
      ac.style.borderColor=sure?'#2fb36d':(au.status==='unsure'?'#d89a1b':'#333944'); } else ac.style.display='none';
    const ll=s.last_label, st=s.label_stats||{}, lr=$('lres');
    if(ll){ lr.textContent=(ll.message||ll.status)+(st.moves?('   ['+(st.moves-st.wrong)+'/'+st.moves+' right so far; no correction counts as right]'):'');
      lr.className='chip '+(ll.status==='error'?'bad':ll.status==='done'?(ll.confident?'ok':'warn'):'idle'); }
    else if(s.marks_pending){ lr.textContent='move marked: type left / middle / right'; lr.className='chip warn'; }
    const ov=s.overlay||s.android_overlay;
    document.getElementById('thand').style.display=s.hand?'block':'none'; document.getElementById('tplaced').checked=s.placed_on_top!==false;
    isLocked=!!s.locked; $('bLock').textContent=isLocked?'Overlay LOCKED for this game - click to unlock':'Lock overlay (when it looks perfect)';
    autoOn=!!s.autotower_auto; $('bAuto').textContent='Auto-detect: '+(autoOn?'on':'off');
    const au=s.autotower||{}; $('tauto').textContent='auto-detect: '+(au.status||'–')+(au.note?(' · '+au.note):'');
    const tm=$('tmsg'); tm.textContent=s.tracker_msg;
    tm.className='chip '+(s.tracker_msg.indexOf('could not')>=0||s.tracker_msg.indexOf('error')>=0||s.tracker_msg.indexOf('COLLAPSED')>=0?'bad':s.tracker_msg.indexOf('ready')===0||s.tracker_msg.indexOf('move')===0?'ok':'idle');
    $('tcount').textContent=ov?('green '+ov.green+(ov.amber?'  ·  amber (loose) '+ov.amber:'')+'  ·  red '+ov.red)+(s.height_info?('  ·  LiDAR height ≈ '+s.height_info.lidar_layers+' layers'):''):'overlay needs 2+ markers in view';
    $('pfps').textContent=live?s.preview_fps.toFixed(1)+' fps (shown '+shownFps('a').toFixed(0)+')':'–';
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
GREEN, RED, AMBER = (60, 200, 60), (50, 50, 235), (0, 165, 255)
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


def _effective(layers, removed):
    """(layers, removed) of the tower as it really stands: each pulled block put back on top (real Jenga), unless the game is
    played with blocks only removed.  The state keeps the plain record (start layers + pulls); everything that looks at the
    tower, the overlay, the rules and the pictures, uses this."""
    import jenga_rules
    if not _state.get("placed_on_top", True):
        return int(layers), [tuple(r) for r in removed]
    return jenga_rules.effective_tower(int(layers), [tuple(r) for r in removed], True)


def _eff_state():
    with _lock:
        return _effective(_state["tower"]["layers"], _state["tower"]["removed"])


def removable(layers, removed):
    """(layer, slot) -> True if that block may be pulled; the rules and their reasons are in jenga_rules.py"""
    import jenga_rules
    return jenga_rules.removable(layers, removed)


_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


_sink = threading.local()      # text is not baked into the picture: the viewer draws it upright after turning the view


def _text(text, xy, col, scale=0.55):
    """Queue a label for this frame (x, y in picture pixels, col = BGR); _annotate hands them to the viewer."""
    items = getattr(_sink, "items", None)
    if items is not None:
        items.append([text, int(xy[0]), int(xy[1]), scale, "#%02x%02x%02x" % (col[2], col[1], col[0])])


_distort = _tp.distort_px


def _ruler(img, K, R, t, cam, layers, dist=None):
    """Read-off labels so a block can be named: layer numbers up a corner, slot digits along the base.
    yellow digits = slot for layers 1,3,5,..   cyan digits = slot for layers 2,4,6,.."""
    F = cv2.FONT_HERSHEY_SIMPLEX
    YEL, CYA, WHT = (0, 215, 255), (255, 220, 0), (255, 255, 255)

    def label(text, p, col, dx=0):
        c = R @ np.asarray(p, float) + t
        if c[2] <= 0:
            return
        u = K @ c
        pt = _distort(np.array([[u[0] / u[2], u[1] / u[2]]]), K, dist)[0]
        xy = (int(pt[0]) - 6 + dx, int(pt[1]) + 5)
        _text(text, xy, col)

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


_last_pose = {}      # kind -> (R, t, time): a phone on a stand does not move in a second
_pose_lock = {}      # kind -> (R, t): the last good pose; phones sit on stands, so it stays valid until they are bumped
_pose_keep = {}      # same, but only cleared when the board or the frame changes: lets the LiDAR still check the tower after a fall
_lock_valid = {}     # kind -> time the lock was last confirmed (3+ markers, or visible markers still land where it says)


def _no_moves(tower):
    """True when blocks remain but every one of them is red: nothing can be pulled any more."""
    ly, rm = _effective(tower["layers"], tower["removed"])
    ok = removable(ly, {tuple(r) for r in rm})
    return bool(ok) and not any(ok.values())


MIN_MARKER_PX = 28.0   # mean marker side below this: the pose is too shaky to draw a tall tower from


def _marker_px(dets):
    return float(np.mean([np.linalg.norm(q[k] - q[(k + 1) % 4]) for q in dets.values() for k in range(4)]))


def _hint(img, text):
    """A one-line warning over the picture (drawn by the viewer)."""
    getattr(_sink, "shapes", []).append(["h", text])


def _carries_fit(kind):
    """The LiDAR fit was measured with the iPhone's own pose, so it corrects the iPhone's overlay.  Applied to the Android
    it carried the iPhone's pose error over (the Android overlay sat a whole layer too high), so the Android keeps the plain
    marker pose."""
    return kind != "android"


def _lock_pose(kind):
    R, t = _pose_lock[kind]
    return _tp._corrected(R, t) if _carries_fit(kind) else (R, t)


_delta = {"iphone": None, "android": None}      # tower_edges.refine() result per camera: how the tower frame sits in THAT picture


def _refine_camera(kind):
    """Snap one camera's overlay onto the real tower's outline.  Runs after Re-detect (once the LiDAR has settled) and uses
    three frames, the median of what they agree on, and only if the outline really fits better than the pose it starts from.
    The iPhone starts from its LiDAR-corrected pose, the Android from the plain marker pose."""
    import tower_edges as te
    try:
        layers = _eff_state()[0]
        got = []
        for k in range(3):
            if k:
                time.sleep(0.35)
            with _lock:
                raw = _state["jpeg_raw" if kind == "iphone" else "android_raw"]
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
            if img is None:
                continue
            K, dist = _k_iphone(img.shape[1]) if kind == "iphone" else _k_android(img.shape[1], img.shape[0])
            if K is None:
                continue
            corners, ids, _ = _DETECTOR.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if ids is None:
                continue
            d = {int(i): cv2.undistortPoints(c.reshape(-1, 1, 2).astype(np.float64), K, dist, P=K).reshape(4, 2) if dist is not None and len(dist) else c.reshape(4, 2).astype(np.float64)
                 for c, i in zip(corners, ids.flatten()) if int(i) < 4}
            s = _tp.solve(d, K, layout=_tp.LAYOUTS["A"], corrected=_carries_fit(kind)) if len(d) >= 3 else None
            if not s or s[2] > 6.0:
                continue
            x, before, after = te.refine(img, K, dist, s[0], s[1], layers)
            got.append((x, before, after))
        if not got:
            return f"{kind}: no usable markers or tower in the picture right now"
        good = [g for g in got if g[2] >= 1.15 * g[1] + 0.5]          # the outline must fit clearly better than where the pose put it
        if len(good) < 2:
            _delta[kind] = None
            return f"{kind}: the tower outline was not clear enough to refine; using the marker pose"
        x = np.median(np.array([g[0] for g in good]), axis=0)
        _delta[kind] = x
        with _lock:
            _state["edge"][kind] = {"shift_mm": [round(float(v), 1) for v in x[:3]], "tilt_deg": [round(float(v), 1) for v in x[3:]],
                                    "score_before": round(float(np.mean([g[1] for g in good])), 1),
                                    "score_after": round(float(np.mean([g[2] for g in good])), 1), "at": time.strftime("%H:%M:%S")}
        return f"{kind} overlay snapped to the tower outline (moved {np.linalg.norm(x[:3]):.1f} mm, {np.linalg.norm(x[3:]):.1f} deg)"
    except Exception as e:
        return f"{kind} refinement failed: {type(e).__name__}: {e}"


POSE_EMA = float(os.environ.get("POSE_EMA", 0.2))          # how much of each new marker pose is taken while a phone sits still (the rest is kept): kills the jitter
MOVE_DEG, MOVE_MM = 1.2, 10.0   # a camera is "moved" when its pose jumps by more than this; then it follows the markers exactly
_pose_ema = {}          # kind -> (R, t): the smoothed pose
_moved_at = {}          # kind -> time the camera was last seen to move (waiting for the tower outline to be re-fitted)


def _smooth_pose(kind, R, t):
    """The corners the detector finds move by a pixel or so from frame to frame, and the pose with them.  A phone on a stand
    does not move: blend each new pose into the old one.  If it really moved, take the new pose at once and remember it."""
    s = _pose_ema.get(kind)
    now = time.time()
    if s is None:
        _pose_ema[kind] = (R, t)
        return R, t
    Rs, ts = s
    rv = cv2.Rodrigues(Rs.T @ R)[0].ravel()
    ang = float(np.degrees(np.linalg.norm(rv)))
    dC = float(np.linalg.norm((-R.T @ t) - (-Rs.T @ ts)))
    if ang > MOVE_DEG or dC > MOVE_MM:
        _pose_ema[kind] = (R, t)
        _moved_at[kind] = now
        return R, t
    R2 = Rs @ cv2.Rodrigues(rv * POSE_EMA)[0]
    t2 = ts + POSE_EMA * (t - ts)
    _pose_ema[kind] = (R2, t2)
    return R2, t2


_frozen = {}           # kind -> (R, t): the pose each camera had when the overlay was locked


REUSE_PX = 1.5         # markers within this many pixels of where the last solve put them: same pose, no need to solve again
REUSE_S = 0.5          # ...but a full solve still happens twice a second, so a slow drift is caught
_solve_t = {}          # kind -> time of the last full solve


def _still_where_locked(kind, K, dets):
    """True if the held pose is fresh and every visible marker is within REUSE_PX of where it puts it."""
    held = _pose_lock.get(kind)
    now = time.time()
    if held is None or now - _lock_valid.get(kind, 0.0) > 3.0 or now - _solve_t.get(kind, 0.0) > REUSE_S:
        return False
    Rl, tl = held
    for i, c in dets.items():
        if i not in _tp.LAYOUTS["A"]:
            continue
        x, y = _tp.LAYOUTS["A"][i]
        uvw = K @ (Rl @ np.array([x, y, 0.0]) + tl)
        if uvw[2] <= 0 or float(np.linalg.norm(uvw[:2] / uvw[2] - c.mean(axis=0))) > REUSE_PX:
            return False
    return True


def _lock_overlay():
    """Freeze the overlay: each phone's pose and the LiDAR correction stay exactly as they are now."""
    now = time.time()
    got = {}
    for kind in ("iphone", "android"):
        held = _last_pose.get(kind)
        if held and now - held[2] <= 3.0:
            got[kind] = (held[0].copy(), held[1].copy())
    if "iphone" not in got:
        return False, "the iPhone has not found the markers right now: nothing to lock"
    Rc, tc = _tp.get_correction()
    for kind, (R, t) in got.items():
        Rm = R @ Rc.T if _carries_fit(kind) else R     # the same pose in the raw marker frame, for the depth path
        _pose_lock[kind] = (Rm, t - Rm @ tc) if _carries_fit(kind) else (R, t)
        _pose_keep[kind] = _pose_lock[kind]
        _lock_valid[kind] = now
    with _fit_lock:
        _fit_st["converged"] = True
    _frozen.clear(); _frozen.update(got)
    with _lock:
        _state["locked"] = True
    return True, "locked: " + " and ".join(sorted(got))


def _unlock_overlay():
    _frozen.clear()
    with _lock:
        _state["locked"] = False


def _draw_markers(img, K, dist, dets, R, t, kind):
    """Outline each visible marker where the pose puts it (the detector's corners jump by a pixel from frame to frame; the pose
    is smoothed), with the red mark on corner 0 and its id."""
    Rc, tc = _tp.get_correction()
    for mid in dets:
        c3 = _tp.marker_corners_3d(mid, 0, _tp.LAYOUTS["A"])
        if _carries_fit(kind):
            c3 = (c3 - tc) @ Rc                       # the pose is in the LiDAR-corrected frame: bring the marker there too
        X = c3 @ R.T + t
        if (X[:, 2] <= 10).any():
            continue
        uv = _distort(np.stack([X[:, 0] / X[:, 2] * K[0, 0] + K[0, 2], X[:, 1] / X[:, 2] * K[1, 1] + K[1, 2]], axis=1), K, dist)
        _sink.shapes.append(["m", uv.tolist()])
        _text(f"id={mid}", uv.mean(axis=0) - (18, -4), (255, 0, 0), 0.5)
    _sink.drew = True


def _draw_overlay(img, K, dist, dets, tower, kind=None):
    """Pose from the markers in this very frame, then every block in red or green."""
    if not HAVE_POSE:
        return None
    sol = None
    frozen = _frozen.get(kind) if kind else None
    reused = False
    if frozen is not None:
        R, t = frozen                      # locked: the pose is not recomputed, so nothing can make it wobble
        _lock_valid[kind] = time.time()
        sol = True
    elif kind and len(dets) >= 3 and _still_where_locked(kind, K, dets):
        # the phone is on a stand: if the markers still sit where the last good solve says, reuse it (the solver is the
        # expensive part, and at 30-60 frames a second the server would otherwise spend all its time re-solving)
        R, t = _lock_pose(kind)
        _last_pose[kind] = (R, t, time.time())
        _lock_valid[kind] = time.time()
        sol = True
        reused = True
    elif len(dets) >= 3:
        d = {i: c for i, c in dets.items()}
        if dist is not None and len(dist):
            d = {i: cv2.undistortPoints(c.reshape(-1, 1, 2), K, dist, P=K).reshape(4, 2) for i, c in d.items()}
        sol = _tp.solve(d, K, layout=_tp.LAYOUTS["A"], corrected=_carries_fit(kind))
        if sol and sol[2] > 10.0:
            sol = None
    if frozen is None and dets and _marker_px(dets) < MIN_MARKER_PX:
        # markers this small give a pose that is off by centimetres 250 mm up: say so instead of drawing it
        if kind:
            _last_pose.pop(kind, None); _pose_lock.pop(kind, None)
        _hint(img, "markers too small - move this phone closer to the board")
        return None
    if frozen is not None or reused:
        pass
    elif sol:
        R, t, _ = sol
        if kind:
            R, t = _smooth_pose(kind, R, t)
            _last_pose[kind] = (R, t, time.time())
            Rc, tc = _tp.get_correction()                  # keep the lock in the marker frame, whatever the fit says later
            Rm = R @ Rc.T if _carries_fit(kind) else R
            _pose_lock[kind] = (Rm, t - Rm @ tc) if _carries_fit(kind) else (R, t)
            _pose_keep[kind] = _pose_lock[kind]
            _lock_valid[kind] = time.time()
            _solve_t[kind] = time.time()
            with _lock:
                late = _state["locked"]
            if late:                                       # a phone that joined after the lock: freeze it at its first good pose
                _frozen[kind] = (R, t)
    elif kind and dets and kind in _pose_lock:      # fewer than 3 markers: do the ones we see still sit where the lock says?
        try:
            Rl, tl = _pose_lock[kind]
            errs = []
            for i, c in dets.items():
                if i not in _tp.LAYOUTS["A"]:
                    continue
                x, y = _tp.LAYOUTS["A"][i]
                uvw = K @ (Rl @ np.array([x, y, 0.0]) + tl)
                errs.append(float(np.linalg.norm(uvw[:2] / uvw[2] - c.mean(axis=0))))
            if errs and max(errs) < 12.0:
                _lock_valid[kind] = time.time()
        except Exception:
            pass
    if frozen is None and not reused and not sol:         # a dropped marker for a moment: keep the last good pose briefly
        held = _last_pose.get(kind)
        if held and time.time() - held[2] <= 1.5:
            R, t, _ = held
        elif kind in _pose_lock and time.time() - _lock_valid.get(kind, 0) <= 6.0:
            R, t = _lock_pose(kind)   # markers covered, but the ones still visible say the phone has not moved
        else:
            return None
    if kind and dets:
        _draw_markers(img, K, dist, dets, R, t, kind)       # where the calibrated board says the markers are: steady, no jitter
    if kind in _delta and _delta[kind] is not None:
        import tower_edges as _te
        R, t = _te.apply_delta(R, t, _delta[kind])
    layers = int(tower["layers"])
    removed = {tuple(r) for r in tower["removed"]}
    ok = removable(layers, removed)
    loose = {tuple(b) for b in tower.get("loose", [])}
    cam = -R.T @ t
    items = []
    for (l, s), good in ok.items():
        lo, hi = _box(l, s)
        items.append((np.linalg.norm((lo + hi) / 2 - cam), l, s, good, lo, hi))
    items.sort(reverse=True)               # far blocks first, so near ones paint over them
    for _d, l, s, good, lo, hi in items:
        col = (AMBER if (l, s) in loose else GREEN) if good else RED
        C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        uvw = (C @ R.T + t) @ K.T
        if (uvw[:, 2] <= 0).any():
            continue
        uv = _distort(uvw[:, :2] / uvw[:, 2:3], K, dist).astype(np.int32)
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
            _sink.shapes.append(["q", quad.tolist(), "#%02x%02x%02x" % (col[2], col[1], col[0])])    # the viewer fills and outlines it
    _ruler(img, K, R, t, cam, layers, dist)
    n_amber = sum(1 for k, g in ok.items() if g and k in loose)
    return {"green": int(sum(ok.values())) - n_amber, "amber": n_amber, "red": int(len(ok) - sum(ok.values()))}


_act = {"last": 0.0, "since": 0.0, "events": collections.deque(maxlen=6000)}   # server-clock seconds of "something moved at the tower"
_prev_small = {}
MOTION_FRAC = {"iphone": 0.06, "android": 0.10}    # this share of the tower's part of the picture changed since the last look: a hand.
# (real hands gave 0.07-0.7 on the iPhone; the 0.02-0.05 readings were people and shadows moving in the background)
ACTIVITY_HOLD = 0.8      # ...and for this long after it stopped moving
ACTIVITY_MAX = 15.0      # something that keeps moving longer than this is not a hand at a move (flicker?): stop waiting for it


def _motion_frac(prev, cur, roi):
    """Share of the pixels in roi = (x0, y0, x1, y1) (small-picture coordinates) that changed clearly between two looks."""
    x0, y0, x1, y1 = roi
    d = cv2.absdiff(cur[y0:y1, x0:x1], prev[y0:y1, x0:x1]) > 22
    return float(d.mean()) if d.size else 0.0


def _tower_roi(shapes, w, h, small_w, small_h):
    """The tower's part of the picture (the box around its drawn faces, 40% wider), in small-picture coordinates."""
    pts = [p for s in shapes if s[0] == "q" for p in s[1]]
    if not pts:
        return (int(small_w * 0.25), int(small_h * 0.1), int(small_w * 0.75), int(small_h * 0.9))
    a = np.array(pts, float)
    x0, y0 = a.min(axis=0); x1, y1 = a.max(axis=0)
    mx, my = 0.4 * (x1 - x0), 0.4 * (y1 - y0)
    sx, sy = small_w / w, small_h / h
    return (max(0, int((x0 - mx) * sx)), max(0, int((y0 - my) * sy)), min(small_w, int((x1 + mx) * sx)), min(small_h, int((y1 + my) * sy)))


def _act_event(now=None):
    now = now or time.time()
    if now - _act["last"] > 2.0:
        _act["since"] = now
    _act["last"] = now
    _act["events"].append(now)


def _active_now():
    """True while a hand is (just now was) moving at the tower.  Everything that judges the tower waits while this is true."""
    now = time.time()
    if now - _act["last"] >= ACTIVITY_HOLD:
        return False
    return now - _act["since"] < ACTIVITY_MAX


def _bursts(gap=3.0):
    """The recent activity as bursts [start, end] (server seconds): events closer together than `gap` belong to one burst."""
    out = []
    for e in list(_act["events"]):
        if out and e - out[-1][1] <= gap:
            out[-1][1] = e
        else:
            out.append([e, e])
    return out


def _move_windows(mark_s, now=None, before_need=1.2, before_max=3.2, after_need=1.6):
    """Where to look for a move, from when hands were at the tower: "wait" while it is still going on, None when there is
    nothing to go by, else {"a": window before the move, "b": window after it} in ms.  The move is the burst of activity
    (hands that come and go within 3 s are one move) that began before the tap; "before" is the calm stretch just ahead of it, "after" the first calm stretch behind it, so a
    block being put back on top (another burst) never lands inside either picture."""
    now = now or time.time()
    bs = _bursts()
    cand = [x for x in bs if x[0] <= mark_s]
    if not cand or mark_s - cand[-1][1] > 20.0:
        return None
    mv = cand[-1]
    if now - mv[1] < 0.9:
        return "wait"                                          # the hand is still there
    a_lo = mv[1] + 0.4
    a_hi = a_lo + after_need
    nxt = [x for x in bs if x[0] > mv[1]]
    if nxt and nxt[0][0] < a_hi:
        a_hi = nxt[0][0] - 0.1                                 # the next burst (putting the block back) starts: stop before it
    if a_hi - a_lo < 1.0:
        return None
    if now < a_hi + 0.5:
        return "wait"                                          # let the pictures of that stretch arrive and be cut
    prev = [x for x in bs if x[1] < mv[0]]
    b_hi = mv[0] - 0.3
    b_lo = max((prev[-1][1] + 0.4) if prev else b_hi - before_max, b_hi - before_max)
    if b_hi - b_lo < before_need:
        return None
    return {"a": (int(b_lo * 1000), int(b_hi * 1000)), "b": (int(a_lo * 1000), int(a_hi * 1000))}


def _annotate(jpeg_bytes, kind=None):
    """Find the markers in this picture, solve the pose and work out the overlay.  Nothing is drawn or re-encoded here: the
    overlay leaves as a short list of shapes (/shapes.json) and the viewer draws it over the raw picture, on the GPU.
    Returns (None, [ids])."""
    if not HAVE_CV:
        return None, []
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _DETECTOR.detectMarkers(gray)
    _sink.items = []
    _sink.shapes = []
    found, dets = [], {}
    if ids is not None:
        found = sorted(int(i) for i in ids.flatten() if int(i) < 4)
        dets = {int(i): c.reshape(4, 2).astype(np.float64) for c, i in zip(corners, ids.flatten()) if int(i) < 4}
    _sink.drew = False
    stats = None
    if kind:
        h, w = img.shape[:2]
        K, dist = _k_iphone(w) if kind == "iphone" else _k_android(w, h)
        if K is not None:
            with _lock:
                tower = dict(_state["tower"]); tower["loose"] = list(_state.get("loose", [])); tower["game_over"] = _state.get("game_over")
                tower["layers"], tower["removed"] = _effective(tower["layers"], tower["removed"])
            try:
                stats = _draw_overlay(img, K, dist, dets, tower, kind)
            except Exception:
                stats = None
        with _lock:
            _state["overlay" if kind == "iphone" else "android_overlay"] = stats
    if kind:
        small = cv2.resize(gray, (192, 108), interpolation=cv2.INTER_AREA)
        prev = _prev_small.get(kind)
        _prev_small[kind] = small
        if prev is not None and _motion_frac(prev, small, _tower_roi(_sink.shapes, img.shape[1], img.shape[0], 192, 108)) > MOTION_FRAC[kind]:
            _act_event()
    if ids is not None and not getattr(_sink, "drew", False):       # no pose: show what the detector found, as it found it
        for c in dets.values():
            _sink.shapes.append(["m", c.tolist()])
        for i, c in dets.items():
            _text(f"id={i}", c.mean(axis=0) - (18, -4), (255, 0, 0), 0.5)
    if kind:
        with _lock:
            _state["labels"][kind] = {"w": img.shape[1], "h": img.shape[0], "items": _sink.items}
            _state["shapes"][kind] = {"w": img.shape[1], "h": img.shape[0], "items": _sink.shapes, "at": time.time()}
    _sink.items = None
    _sink.shapes = None
    return None, found


LOOSE_MM = 5.0         # a face this much closer to the iPhone than it was at the start of the game is sticking out
_loose_st = {"epoch": None, "layers": None, "base": {}, "hist": {}}


PTOP_DROP_MM = 30.0     # the highest points over the tower must have come down this much too (two layers) before a fall is declared
DROP_MM = 20.0         # the top of the tower falling by more than a layer (15 mm) means pieces fell: pulls never lower it
_height_st = {"epoch": None, "layers": None, "ref": [], "hist": [], "live": [], "pref": [], "phist": []}


def _depth_note(why):
    with _lock:
        d = _state["depth_dbg"]
        d.update(n=d["n"] + 1, why=why, at=time.strftime("%H:%M:%S"))
        if _state["lidar_top"]:
            d["top_mm"] = round(_state["lidar_top"][0], 1)


MAX_CORR_MM = 60.0         # the fitted correction to the marker frame can never legitimately exceed this
MAX_CORR_DEG = 8.0
FIT_GAIN = 0.6              # how much of each fitted correction to take (smooths out noise, still settles in ~3 readings)
_fit_lock = threading.Lock()
_fit_st = {"epoch": None, "Hs": [], "ok": 0, "converged": False, "reads": 0, "relayer": False, "off": 0}


def _reset_fit(relayer=False):
    """Forget the fitted correction (after Re-detect, or a tower rebuilt somewhere else)."""
    with _fit_lock:
        _tp.reset_correction()
        _pose_ema.clear(); _moved_at.clear()
        _fit_st.update(Hs=[], ok=0, converged=False, reads=0, relayer=relayer, off=0)
        _height_st.update(ref=[], hist=[])
    with _lock:
        _state["tower_fit"] = None


def _fit_tower(P, epoch, ver=None):
    """Fit the real tower (position, tilt, twist, height) to the LiDAR points and fold it into the pose correction
    that every camera's overlay and every face crop uses.  Returns (P in the corrected frame, converged?)."""
    import tower_fit as tf
    with _fit_lock:
        if _fit_st["epoch"] != epoch:
            _fit_st.update(epoch=epoch)
        if ver is not None and ver != _tp.corr_version():
            # these points were put into the tower frame before the last correction landed: a step measured on them
            # would be applied twice (that made the overlay swing back and forth when the server was busy)
            return P, _fit_st["converged"], None
        f = tf.fit_box(P)
        _fit_st["reads"] += 1
        if f is None:
            return P, _fit_st["converged"] or _fit_st["reads"] >= 20, None
        Hs = _fit_st["Hs"]
        ref_H = float(np.median(Hs)) if len(Hs) >= 5 else None
        good = f["inliers"] >= 0.55 and f["rms"] <= 4.0 and (ref_H is None or abs(f["H"] - ref_H) <= 12.0)
        small = float(np.linalg.norm(f["t"])) < 1.2 and f["tilt_deg"] < 0.35 and abs(f["yaw_deg"]) < 0.5
        with _lock:
            locked = _state["locked"]
        if locked:
            good = False                                # the overlay is locked: keep measuring the height, never move the frame
        if good and _fit_st["converged"]:
            # settled: from now on only tiny corrections are believed.  A hand, a pulled block or a block pushed out
            # pulls the fit sideways; the phones and the board do not move during a game.
            if float(np.linalg.norm(f["t"])) < 2.5 and f["tilt_deg"] < 0.6 and abs(f["yaw_deg"]) < 1.0:
                _fit_st["off"] = 0
            else:
                _fit_st["off"] += 1
                good = False
                if _fit_st["off"] >= 20:                 # ~10 s of steady disagreement: something really moved
                    _tp.reset_correction()
                    _fit_st.update(Hs=[], ok=0, converged=False, off=0)
                    return P, False, None
        if good:
            rv = tf.log(f["R"]) * FIT_GAIN                  # a fraction of the step: noise averages out
            Rd, td = tf.rot(rv), f["t"] * FIT_GAIN
            Rc, tc = _tp.get_correction()
            Rn, tn = Rc @ Rd, Rc @ td + tc
            if float(np.linalg.norm(tn)) > MAX_CORR_MM or float(np.degrees(np.linalg.norm(tf.log(Rn)))) > MAX_CORR_DEG:
                # the marker frame is never this far off: a fit that wants this has locked onto something else
                _tp.reset_correction()
                _fit_st.update(Hs=[], ok=0, converged=False)
                return P, False, None
            _tp.set_correction(Rn, tn)
            P = (P - td) @ Rd                              # the same points, now in the corrected frame
            Hs.append(f["H"]); del Hs[:-15]
            _fit_st["ok"] = _fit_st["ok"] + 1 if small else 0
            if _fit_st["ok"] >= 2:
                _fit_st["converged"] = True
        Hm = float(np.median(Hs)) if Hs else f["H"]
        info = {"shift_mm": [round(float(x), 1) for x in f["t"]], "tilt_deg": round(f["tilt_deg"], 2),
                "twist_deg": round(f["yaw_deg"], 2), "height_mm": round(Hm, 1), "layers_float": round((Hm - tf.TOP_OFFSET) / tf.LAYER_MM, 2),
                "layers": tf.layers_from_height(Hm), "inliers": round(f["inliers"], 2), "rms_mm": round(f["rms"], 2),
                "converged": _fit_st["converged"], "at": time.strftime("%H:%M:%S")}
        Rc_, tc_ = _tp.get_correction()                  # the total correction to the marker frame so far
        info["corr_shift_mm"] = [round(float(x), 1) for x in tc_]
        info["corr_tilt_deg"] = round(float(np.degrees(np.linalg.norm(tf.log(Rc_)))), 2)
        done = _fit_st["converged"] or _fit_st["reads"] >= 20
        h_obs = float(f["H"]) if f["inliers"] >= 0.45 else None       # the top, from ALL the points (not a small window)
    with _lock:
        _state["tower_fit"] = info
        if _fit_st["relayer"] and info["converged"] and len(Hs) >= 5:
            frac = abs(info["layers_float"] - round(info["layers_float"]))
            _fit_st["relayer"] = False
            if frac <= 0.4:                                     # clearly n layers, not halfway between two counts
                _state["tower"] = {"layers": info["layers"], "removed": []}
                _state["tower_epoch"] += 1
                _state["autotower"] = {"status": f"LiDAR: tower measured {info['height_mm']:.0f} mm = {info['layers']} layers", "layers": info["layers"]}
            else:
                _state["autotower"] = {"status": f"LiDAR: tower is {info['height_mm']:.0f} mm = {info['layers_float']:.1f} layers: "
                                                 f"type {int(info['layers_float'])} or {int(info['layers_float']) + 1}"}
    return P, done, h_obs


_stand_dbg = {}
_stand_st = {"epoch": None, "layers": None, "ref": [], "ref_h": [], "low": 0, "back": 0, "up": 0}
STAND_FRAC = 0.35       # the tower counts as fallen when the LiDAR sees less than this share of the points it saw at the start
STAND_READS = 4         # ...for this many readings in a row (about 2 s)
BLOCKED_ALL = 0.5       # fewer than this share of ALL the depth points as at the start: the lens is covered, not the tower fallen
BLOCKED_CLOSE = 0.35    # ...or this share of the picture closer than 30 cm: a hand over the phone
SEEN_AGAIN = 0.7        # a "fell" raised by this check is withdrawn when the tower is seen again at this share of its start


def _check_standing(P, epoch, layers, stale=False, close_frac=0.0, fit_h=None):
    """Is the tower still there?  Counts the LiDAR points in the tower's footprint (upper part and whole column) and compares
    them with the first readings of this tower.  Works without the markers and without the fit, which both fail when the
    tower comes down: the height watch needs them, this does not.
    A hand or a body covering the phone empties the whole picture, not just the tower: that is "cannot see", never "fell".
    A tower that the box fit still finds at its height is standing, whatever the counts say (a pulled block, a bump, a block
    put on top all change the counts).  And a "fell" from this check goes away again as soon as the tower is seen standing."""
    st = _stand_st
    if st["epoch"] != epoch:
        st.update(epoch=epoch, layers=layers, ref=[], ref_h=[], ref_p=[], low=0, back=0, up=0)
    st["layers"] = layers
    foot = (np.abs(P[:, 0]) < 50) & (np.abs(P[:, 1]) < 50)
    z = P[foot, 2]
    n_col = int(np.count_nonzero(z > 25.0))
    n_top = int(np.count_nonzero(z > 0.6 * layers * 15.0))
    n_all = len(P)
    zc = P[(np.abs(P[:, 0]) < 40) & (np.abs(P[:, 1]) < 40) & (P[:, 2] > 20), 2]
    _stand_dbg["p95"] = round(float(np.percentile(zc, 95)), 1) if len(zc) >= 150 else None
    _stand_dbg.update(n_col=n_col, n_top=n_top, n_all=n_all, close=round(close_frac, 2))
    p95 = _stand_dbg.get("p95")
    if len(st["ref"]) < 6:
        if not stale and n_col >= 150 and close_frac < 0.9:
            st["ref"].append((n_col, n_top, n_all, close_frac))
            if fit_h is not None:
                st["ref_h"].append(fit_h)
            if p95 is not None:
                st.setdefault("ref_p", []).append(p95)
        return
    ref_col = float(np.median([r[0] for r in st["ref"]]))
    ref_top = float(np.median([r[1] for r in st["ref"]]))
    ref_all = float(np.median([r[2] for r in st["ref"]]))
    ref_h = float(np.median(st["ref_h"])) if st["ref_h"] else None
    ref_close = float(np.median([r[3] for r in st["ref"]]))     # how much of THIS phone's picture is normally close (a phone low on the table sees a lot)
    if n_all < BLOCKED_ALL * ref_all or close_frac > max(BLOCKED_CLOSE, ref_close + 0.25):
        st["low"] = 0                                   # covered: say nothing, wait for the next clear look
        return
    fit_ok = fit_h is not None and ref_h is not None and fit_h >= ref_h - 15.0
    if fit_ok and ref_h + 9.0 <= fit_h <= ref_h + 22.0:   # a block was put on top: the tower is taller now, and that is the new normal
        st["up"] += 1
        if st["up"] >= 5:
            st["ref_h"] = [fit_h] * 6
            if p95 is not None:
                st["ref_p"] = [p95] * 6
            st["ref"] = [(max(n_col, ref_col), max(n_top, ref_top), ref_all, ref_close)] * 6
            st["up"] = 0
    else:
        st["up"] = 0
    frac = STAND_FRAC * (0.7 if stale else 1.0)           # without the markers the frame is less certain: be stricter
    ref_p = float(np.median(st["ref_p"])) if st.get("ref_p") else None
    top_down = p95 is None or ref_p is None or ref_p - p95 >= PTOP_DROP_MM       # the highest points over the tower really came down
    gone = (not fit_ok) and top_down and (n_col < frac * ref_col or (ref_top >= 40 and n_top < frac * ref_top))
    st["low"] = st["low"] + 1 if gone else 0
    if st["low"] >= STAND_READS:
        with _lock:
            if not _state["game_over"]:
                _state["game_over"] = {"reason": f"the LiDAR no longer sees the tower (only {100 * n_col / ref_col:.0f}% of it): it fell",
                                       "at": time.strftime("%H:%M:%S"), "src": "lidar-standing"}
    standing = fit_ok or (n_col >= SEEN_AGAIN * ref_col and (ref_top < 40 or n_top >= SEEN_AGAIN * ref_top))
    st["back"] = st["back"] + 1 if standing else 0
    if st["back"] >= 3:
        with _lock:
            go = _state["game_over"]
            if go and go.get("src") == "lidar-standing":
                _state["game_over"] = None              # it was standing all along: a hand, not a collapse


def _watch_height(P, epoch, layers, ready=True, h_obs=None):
    """Top-of-tower height from the depth points over the tower's core.  A legal turn never changes it (the top
    complete layer cannot be pulled), so a drop of more than a layer for 3 readings in a row = game over."""
    if _height_st["epoch"] != epoch:
        _height_st.update(epoch=epoch, layers=layers, ref=[], hist=[], pref=[], phist=[])
        with _lock:
            _state["game_over"] = None
    _height_st["layers"] = layers
    if os.environ.get("DEPTH_DUMP"):
        np.save(os.environ["DEPTH_DUMP"], P)
    if not ready:
        return _depth_note("waiting for the tower fit")   # only then is the window on the tower
    if h_obs is not None:
        top = h_obs                                           # the fitted top: robust, and independent of the marker pose
    else:                                                     # no usable fit (collapsed? hidden?): the small window as before
        core = (np.abs(P[:, 0]) < 30) & (np.abs(P[:, 1]) < 30) & (P[:, 2] > 20)
        z = P[core, 2]
        if len(z) < 60:
            return _depth_note(f"only {len(z)} depth points over the tower core")
        top = float(np.percentile(z, 95))
    _depth_note("ok")
    _height_st["live"] = (_height_st["live"] + [top])[-7:]
    with _lock:
        _state["lidar_top"] = (float(np.median(_height_st["live"])), time.time())   # for the tower auto-detect
    # a second, independent look at the top: the highest points straight over the tower, read directly from the points.  The
    # box fit can settle on a lower height when blocks near the top have been pulled (it read 100 mm for a 140 mm tower),
    # so a drop only counts when this agrees with it.
    zc = P[(np.abs(P[:, 0]) < 40) & (np.abs(P[:, 1]) < 40) & (P[:, 2] > 20), 2]
    p95 = float(np.percentile(zc, 95)) if len(zc) >= 150 else None
    p95_now = p95 if p95 is not None else 0.0                  # nothing over the tower at all: its top is down at the floor
    if len(_height_st["ref"]) < 8:
        _height_st["ref"].append(top)                # the first readings of a new tower: its normal height
        if p95 is not None:
            _height_st["pref"].append(p95)
        return
    ref = float(np.median(_height_st["ref"]))
    _height_st["hist"] = (_height_st["hist"] + [top])[-5:]
    _height_st["phist"] = (_height_st["phist"] + [p95_now])[-5:]
    cur = float(np.median(_height_st["hist"]))
    h5 = _height_st["hist"]
    if len(h5) >= 5 and min(h5) >= ref + 9.0 and max(h5) - min(h5) <= 6.0 and cur <= ref + 22.0:
        _height_st["ref"] = [cur] * 8                              # five steady readings, one layer taller: a block was put on top
        ref = cur
        if len(_height_st["phist"]) >= 5:
            _height_st["pref"] = [float(np.median(_height_st["phist"]))] * 8
    if _height_st["pref"] and len(_height_st["phist"]) >= 3:
        p_drop = float(np.median(_height_st["pref"]) - np.median(_height_st["phist"]))
    else:
        p_drop = 999.0 if p95 is None else 0.0
    corroborated = p_drop >= PTOP_DROP_MM
    with _lock:
        _state["height_info"] = {"top_mm": round(cur, 1), "start_mm": round(ref, 1),
                                 "lidar_layers": int(round(ref / 15.0)), "at": time.strftime("%H:%M:%S")}
        if len(_height_st["hist"]) >= 3 and cur <= ref - DROP_MM and corroborated and not _state["game_over"]:
            _state["game_over"] = {"reason": f"the top of the tower dropped {ref - cur:.0f} mm "
                                             f"(about {(ref - cur) / 15:.1f} layers): pieces fell",
                                   "at": time.strftime("%H:%M:%S")}


def _process_depth(h, body):
    """One depth frame from the iPhone: how far does each visible block face stick out compared with the start?"""
    import collections
    import depth_protrusion as dp
    w, hh = int(h["X-W"]), int(h["X-H"])
    n = w * hh
    if len(body) < 5 * n:
        return _depth_note("short depth frame")
    now = time.time()
    held = _last_pose.get("iphone")
    live = bool(held and now - held[2] <= 3.0)
    ver = _tp.corr_version()
    stale = False
    if "iphone" in _pose_lock and now - _lock_valid.get("iphone", 0) > 6.0:
        stale = True        # no marker has confirmed the pose for a while (fallen blocks cover them): good enough to ask "is it standing?"
    if "iphone" not in _pose_lock and "iphone" in _pose_keep:
        stale = True        # the lock itself was dropped (markers looked too small after the fall): use the last good pose
        _pose_lock["iphone"] = _pose_keep["iphone"]
    if "iphone" in _pose_lock and not stale:
        R, t = _tp._corrected(*_pose_lock["iphone"])   # markers may be hidden (fallen blocks?), but the phone has not moved
    elif stale:
        R, t = _tp._corrected(*_pose_lock["iphone"])
    else:
        return _depth_note("no iPhone pose")
    if _active_now():
        _stand_st["low"] = 0
        return _depth_note("a hand is at the tower: waiting for it to leave")      # no fit, no fall check, no loose check
    depth = np.frombuffer(body[:4 * n], np.float32).reshape(hh, w)
    conf = np.frombuffer(body[4 * n:5 * n], np.uint8).reshape(hh, w)
    K = np.array([[float(h["X-Fx"]), 0, float(h["X-Cx"])], [0, float(h["X-Fy"]), float(h["X-Cy"])], [0, 0, 1.0]])
    with _lock:
        layers, removed_l = _effective(_state["tower"]["layers"], _state["tower"]["removed"])
        epoch = _state["tower_epoch"]
        removed = {tuple(r) for r in removed_l}
    if (_loose_st["epoch"], _loose_st["layers"]) != (epoch, layers):
        _loose_st.update(epoch=epoch, layers=layers, base={}, hist={}, streak={})       # new tower: learn its faces afresh
    P = dp.depth_points(depth, conf, K, int(h["X-RgbW"]), int(h["X-RgbH"]), R, t)
    ok_px = (conf >= 2) & (depth > 0.0)
    close_frac = float(np.count_nonzero(ok_px & (depth < 0.30))) / max(1, int(np.count_nonzero(ok_px)) or 1) if ok_px.any() else 1.0
    if stale:
        _check_standing(P, epoch, layers, stale=True, close_frac=close_frac)
        return _depth_note("markers hidden: only checking that the tower still stands")
    P, ready, h_obs = _fit_tower(P, epoch, ver)          # line the tower model up with the points; P is now in the fitted frame
    R, t = _tp._corrected(*_pose_lock["iphone"])
    if ready:
        _check_standing(P, epoch, layers, close_frac=close_frac, fit_h=h_obs)
    _watch_height(P, epoch, layers, ready, h_obs)
    if not live or not ready:                 # the faces' baseline is learned in the settled frame only
        return
    offs = dp.face_offsets(P, -R.T @ t, layers)
    loose, worst = set(), 0.0
    for (l, s, sign), (v, _n) in offs.items():
        if (l, s) in removed:
            continue
        k = (l, s, sign)
        b = _loose_st["base"].setdefault(k, [])
        if len(b) < 6:                       # the first readings of a face define "untouched"
            b.append(v)
            continue
        if np.std(b) > 3.0:                  # a face whose first readings jump about (grazing angle, edge of view): untrustworthy
            continue
        hist = _loose_st["hist"].setdefault(k, collections.deque(maxlen=5))
        hist.append(v)
        if len(hist) >= 3:
            prot = float(np.median(hist) - np.median(b))
            if prot > 30.0:                  # nothing a block does moves it that far: a hand or a glitch, not a loose block
                continue
            worst = max(worst, prot)
            if prot >= LOOSE_MM:
                loose.add((l, s))
    streak = _loose_st.setdefault("streak", {})       # a block must look loose twice running: one odd reading is not enough
    for b in list(streak):
        if b not in loose:
            del streak[b]
    for b in loose:
        streak[b] = streak.get(b, 0) + 1
    loose = {b for b in loose if streak[b] >= 2}
    with _lock:
        _state["loose"] = sorted([l, s] for l, s in loose)
        _state["loose_info"] = {"faces_seen": len(offs), "faces_with_baseline": sum(1 for b in _loose_st["base"].values() if len(b) >= 6),
                                "most_out_mm": round(worst, 1), "at": time.strftime("%H:%M:%S")}


ANALYSIS_MIN_S = 1.0 / float(os.environ.get("ANALYSIS_FPS", 10))   # most the overlay is recomputed per second for each phone


class _AnalysisWorker:
    """Finds the markers and works out the overlay of the NEWEST frame of one phone, at most ANALYSIS_FPS times a second.  The
    pictures themselves go straight from the phone to the viewer; phones on stands do not move fast enough to need more."""

    def __init__(self, kind):
        self.kind = kind
        self._cv = threading.Condition()
        self._raw = None
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, raw):
        with self._cv:
            self._raw = raw
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._raw is not None)
                raw, self._raw = self._raw, None
            t_start = time.time()
            try:
                _, ids = _annotate(raw, self.kind)
            except Exception:
                ids = []
            with _lock:
                _state["markers" if self.kind == "iphone" else "android_markers"] = ids
            time.sleep(max(0.0, ANALYSIS_MIN_S - (time.time() - t_start)))


_workers = {"iphone": _AnalysisWorker("iphone"), "android": _AnalysisWorker("android")}


# --- replaying a game afterwards -----------------------------------------------------------------------------------------
import re as _re
import urllib.parse as _up


def _jsonl(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def _first_last_ts(index_path):
    """(first, last) frame timestamp in ms of a run's index.jsonl, reading only its two ends."""
    try:
        with open(index_path, "rb") as f:
            first = json.loads(f.readline())["ts_ms"]
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            last = first
            for line in f.read().splitlines()[::-1]:
                try:
                    last = json.loads(line)["ts_ms"]
                    break
                except ValueError:
                    continue
        return int(first), int(last)
    except (OSError, ValueError, KeyError):
        return None, None


def _run_layers(index_path):
    mx = 0
    try:
        with open(index_path) as f:
            for i, line in enumerate(f):
                if i > 3000:
                    break
                try:
                    mx = max(mx, int(json.loads(line)["layer"]) + 1)
                except (ValueError, KeyError):
                    pass
    except OSError:
        pass
    return mx or 8


def _replay(run):
    """Everything needed to replay one recorded game: its tower setup, every pull and the game over, with times in seconds
    from the start of recording.  Games recorded since the event recorder existed have exact times and blocks (moves.jsonl);
    older ones are put together from the answers in labels.jsonl (the block's slot is then a best guess)."""
    d = os.path.join(HERE, "dataset", run)
    start, end = _first_last_ts(os.path.join(d, "index.jsonl"))
    if start is None:
        return None
    labels, moves = _jsonl(os.path.join(d, "labels.jsonl")), _jsonl(os.path.join(d, "moves.jsonl"))
    try:
        mk = json.load(open(os.path.join(d, "marks.json"))).get("marks_ms", [])
    except (OSError, ValueError):
        mk = []
    tap_t = {}
    for r in labels:
        if "mark_ms" in r:
            tap_t[r["seq"]] = r["mark_ms"]
    for i, t in enumerate(mk, 1):
        tap_t.setdefault(i, t)
    taps = [{"seq": s, "t_s": round((t - start) / 1000, 2)} for s, t in sorted(tap_t.items())]
    out = {"id": run, "start_ms": start, "duration_s": round((end - start) / 1000, 1), "taps": taps,
           "start_clock": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start / 1000))}
    rel = lambda t: round((t - start) / 1000, 2)
    if moves:
        meta = next((m for m in moves if m["ev"] == "meta"), {})
        ev = []
        for m in moves:
            if m["ev"] == "pull":
                ev.append({"t_s": rel(m["t"]), "ev": "pull", "block": m["block"], "how": m.get("how"), "pct": m.get("pct"), "seq": m.get("seq")})
            elif m["ev"] == "unpull":
                ev.append({"t_s": rel(m["t"]), "ev": "unpull", "block": m["block"]})
            elif m["ev"] in ("gameover", "gameover_cleared", "reset", "layers", "placed"):
                ev.append({"t_s": rel(m["t"]), **{k: v for k, v in m.items() if k not in ("t",)}})
        out.update(exact=True, layers=int(meta.get("layers") or _run_layers(os.path.join(d, "index.jsonl"))),
                   placed=bool(meta.get("placed", True)), initial_removed=meta.get("removed", []), events=ev)
        return out
    typed = {r["seq"]: r for r in labels if "mark_ms" in r}
    auto = {r["seq"]: r["auto"] for r in labels if "auto" in r}
    yes = {r["seq"]: r["correction"] for r in labels if r.get("by") == "yes"}
    rej = {r["seq"] for r in labels if "rejected" in r}
    corr = {r["seq"]: r["correction"] for r in labels if "correction" in r and r.get("by") != "yes"}
    ev = []
    for s in sorted(set(typed) | set(auto) | set(tap_t)):
        t0 = tap_t.get(s)
        if t0 is None:
            continue
        a, ty = auto.get(s), typed.get(s)
        res = (ty or {}).get("result", {})
        blk, how, pct = None, None, None
        if s in yes and a:
            blk, how, pct = [yes[s] - 1, a["slot"]], "auto + yes", a["pct"]
        elif ty and (ty.get("user_layer") or s in corr or (res.get("ok") and res.get("confident"))):
            layer = ty.get("user_layer") or corr.get(s) or res.get("layer")
            slot = res.get("slot") if res.get("ok") and res.get("layer") == layer else (a["slot"] if a and a["layer"] == layer else 1)
            blk, how, pct = [layer - 1, slot], "typed", res.get("confidence_pct")
        elif a and a["pct"] >= AUTO_PCT and s not in rej:
            blk, how, pct = [a["layer"] - 1, a["slot"]], "auto", a["pct"]
        if blk:
            ev.append({"t_s": round((t0 - start) / 1000 + 3.0, 2), "ev": "pull", "block": blk, "how": how, "pct": pct, "seq": s})
    out.update(exact=False, layers=_run_layers(os.path.join(d, "index.jsonl")), placed=False, initial_removed=[], events=ev)
    return out


def _games_list():
    ddir = os.path.join(HERE, "dataset")
    out = []
    for run in sorted(os.listdir(ddir), reverse=True):
        if not _re.match(r"^live_\d{8}_\d{6}$", run) or not os.path.exists(os.path.join(ddir, run, "labels.jsonl")):
            continue
        try:
            r = _replay(run)
        except Exception:
            r = None
        if not r or not r["taps"]:
            continue
        pulls = sum(1 for e in r["events"] if e["ev"] == "pull") - sum(1 for e in r["events"] if e["ev"] == "unpull")
        out.append({"id": run, "start_clock": r["start_clock"], "duration_s": r["duration_s"], "taps": len(r["taps"]), "pulls": pulls,
                    "exact": r["exact"], "game_over": any(e["ev"] == "gameover" for e in r["events"]), "layers": r["layers"]})
    return out


_frame_idx = {"at": 0.0, "iphone": [], "android": []}


def _frame_near(cam, t_ms, within=4000):
    now = time.time()
    if now - _frame_idx["at"] > 20.0:
        names = os.listdir(SAVE_DIR)
        _frame_idx["iphone"] = sorted(int(n.split("_")[0]) for n in names if n.endswith("_rgb.jpg"))
        _frame_idx["android"] = sorted(int(n.split("_")[0]) for n in names if n.endswith("_android.jpg"))
        _frame_idx["at"] = now
    arr = _frame_idx[cam]
    if not arr:
        return None
    import bisect
    i = bisect.bisect_left(arr, t_ms)
    best = min(arr[max(0, i - 1):i + 1], key=lambda x: abs(x - t_ms))
    if abs(best - t_ms) > within:
        return None
    path = os.path.join(SAVE_DIR, f"{best}_rgb.jpg" if cam == "iphone" else f"{best}_android.jpg")
    return path if os.path.exists(path) else None


class Handler(BaseHTTPRequestHandler):
    # Keep-alive: the Android app sends every frame over one connection instead of a new TCP handshake per
    # frame (costly over Wi-Fi). Every response carries Content-Length, and every POST body is read.
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True      # no ~40 ms Nagle / delayed-ACK stall per reply

    def _send(self, code, body=b"", ctype="text/plain", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self._headers_buffer += [b"\r\n", body]   # end_headers(), but headers + body go out in one write
        self.flush_headers()

    def _send_frame(self, key, seq_key):
        """Latest frame; with ?after=N, wait (up to 2 s) for a frame newer than N first."""
        m = re.search(r"[?&]after=(\d+)", self.path)
        after = int(m.group(1)) if m else -1
        with _new_frame:
            _new_frame.wait_for(lambda: _state[seq_key] > after and _state[key] is not None, timeout=2.0)
            jpeg, seq = _state[key], _state[seq_key]
        if jpeg is None:
            self._send(404)
        elif seq <= after:
            self._send(204)
        else:
            self._send(200, jpeg, "image/jpeg", {"X-Seq": str(seq)})

    # --- viewer -------------------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/preview.jpg":
            self._send_frame("jpeg_raw", "preview_seq")
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
            self._send_frame("android_raw", "android_seq")
        elif path == "/games.json":
            try:
                body = json.dumps(_games_list()).encode()
            except Exception as e:
                self._send(500, f"games error: {e}".encode()); return
            self._send(200, body, "application/json")
        elif path == "/game_replay.json":
            q = _up.parse_qs(_up.urlsplit(self.path).query)
            run = (q.get("run") or [""])[0]
            if not _re.match(r"^live_\d{8}_\d{6}$", run):
                self._send(400, b"bad run"); return
            r = _replay(run)
            if r is None:
                self._send(404, b"no such game"); return
            self._send(200, json.dumps(r).encode(), "application/json")
        elif path == "/game_frame.jpg":
            q = _up.parse_qs(_up.urlsplit(self.path).query)
            run = (q.get("run") or [""])[0]
            cam = (q.get("cam") or ["iphone"])[0]
            try:
                t = float((q.get("t") or ["0"])[0])
            except ValueError:
                t = 0.0
            if not _re.match(r"^live_\d{8}_\d{6}$", run) or cam not in ("iphone", "android"):
                self._send(400, b"bad request"); return
            start, _ = _first_last_ts(os.path.join(HERE, "dataset", run, "index.jsonl"))
            p = _frame_near(cam, start + t * 1000) if start else None
            if not p:
                self._send(404, b"no picture saved for that moment"); return
            with open(p, "rb") as f:
                self._send(200, f.read(), "image/jpeg")
        elif path == "/shapes.json":
            with _lock:
                body = json.dumps(_state["shapes"]).encode()
            self._send(200, body, "application/json")
        elif path == "/status.json":
            now = time.time()
            with _lock:
                pt = list(_preview_times)
                ut = [t for t in _upload_times if now - t < 5.0]
                s = dict(_state)
                last_up = _upload_times[-1] if _upload_times else None
                at = list(_android_times)
            fps = (len(pt) - 1) / (pt[-1] - pt[0]) if len(pt) > 2 and pt[-1] > pt[0] else 0.0
            afps = (len(at) - 1) / (at[-1] - at[0]) if len(at) > 2 and at[-1] > at[0] and now - at[-1] < 2 else 0.0
            body = json.dumps({
                "cv": HAVE_CV, "markers": s["markers"], "capturing": s["capturing"],
                "saved": _counter(s, "saved"), "marks": _counter(s, "marks"), "want": s["want"], "tracking": s["tracking"], "files": _counter(s, "files"),
                "preview_age": (now - s["preview_t"]) if s["preview_t"] else None,
                "preview_fps": fps, "android_fps": afps, "upload_rate": len(ut) / 5.0,
                "upload_age": (now - last_up) if last_up else None,
                "android_markers": s["android_markers"], "android_saved": _counter(s, "android_saved"),
                "android_age": (now - s["android_t"]) if s["android_t"] else None,
                "android_size": s["android_size"], "android_dbg": list(_android_dbg)[-12:], "tower": s["tower"], "tracker_msg": s["tracker_msg"], "tracker_on": s["tracker_on"], "dataset": s["dataset"], "autotower": s["autotower"], "autotower_auto": s["autotower_auto"], "depth_dbg": dict(s["depth_dbg"]), "last_label": s["last_label"], "label_stats": s["label_stats"], "marks_pending": sum(1 for m in s["marks_log"] if not m["labelled"]),
                "overlay": s["overlay"], "android_overlay": s["android_overlay"], "labels": s["labels"], "edge": s["edge"], "hand": _active_now(), "placed_on_top": s["placed_on_top"], "auto": s["auto"],
                "tower_eff": dict(zip(("layers", "removed"), _effective(s["tower"]["layers"], s["tower"]["removed"]))), "locked": s["locked"], "loose": s["loose"], "loose_info": s["loose_info"], "no_moves": _no_moves(s["tower"]), "tower_epoch": s["tower_epoch"], "game_over": s["game_over"], "height_info": s["height_info"], "tower_fit": s["tower_fit"],
            }).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"Not Found")

    # --- phone --------------------------------------------------------------------------
    def do_POST(self):
        if self.path.startswith("/cmd/") and self.path not in ("/cmd/tower", "/cmd/label", "/cmd/viewrot"):   # those two read their own body
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))   # keep the connection in sync
        if self.path == "/cmd/tower":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                layers = max(1, min(18, int(body.get("layers", 10))))
                removed = [[int(a), int(b)] for a, b in body.get("removed", [])
                           if 0 <= int(a) < layers and 0 <= int(b) < 3]
                with _lock:
                    if layers != _state["tower"]["layers"]:
                        _state["autotower_auto"] = False      # a typed layer count wins over the cameras
                    _state["tower"] = {"layers": layers, "removed": removed}
                    _state["auto"] = None
                    _state["tower_epoch"] += 1
                    _state["game_over"] = None
                self._send(200, b"OK")
            except Exception as e:
                self._send(400, f"bad tower: {e}".encode())
            return
        if self.path == "/cmd/label":
            try:
                import move_infer as mi
                text = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}").get("text", "")
                if re.search(r"\b(fell|fall|fallen|collapse|collapsed|knocked|over)\b", text.lower()):
                    with _lock:
                        _state["game_over"] = {"reason": "you reported that pieces fell", "at": time.strftime("%H:%M:%S")}
                    self._send(200, b"OK")
                    return
                pos, layer = mi.parse_text(text)
                with _lock:
                    a_ = _state["auto"]
                    typed_over = bool(a_ and a_.get("status") == "sure" and not a_.get("answered") and pos is not None)
                if typed_over:
                    _auto_answer(False)                            # typing a move means "that was not it": undo and take the typed one
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
                    if last.get("pos_slots", {}).get(layer - 1) is not None:
                        new_b = [layer - 1, last["pos_slots"][layer - 1]]
                        old_b = list(last["block"]) if last.get("block") is not None else None
                        with _lock:       # swap the wrongly marked block for the real one (or add it, if the finder was not sure)
                            rem = [b for b in _state["tower"]["removed"] if list(b) not in (old_b, new_b)]
                            _state["tower"] = {"layers": _state["tower"]["layers"], "removed": rem + [new_b]}
                            _state["last_label"]["block"] = new_b
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
        if self.path == "/cmd/newgame":
            _new_game()
            self._send(200, b"OK")
            return
        if self.path in ("/cmd/confirm/yes", "/cmd/confirm/no"):
            okk, msg = _auto_answer(self.path.endswith("/yes"))
            self._send(200 if okk else 409, msg.encode())
            return
        if self.path in ("/cmd/placed/on", "/cmd/placed/off"):
            with _lock:
                _state["placed_on_top"] = self.path.endswith("/on")
            self._send(200, b"OK")
            return
        if self.path == "/cmd/viewrot":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                rot = int(body.get("iphone", 0)) % 360
                if rot % 90:
                    raise ValueError("rotation must be a multiple of 90")
                with _lock:
                    _state["view_rot"]["iphone"] = rot
                self._send(200, b"OK")
            except Exception as e:
                self._send(400, f"bad rotation: {e}".encode())
            return
        if self.path in ("/cmd/lock/on", "/cmd/lock/off"):
            if self.path.endswith("/on"):
                okk, msg = _lock_overlay()
            else:
                _unlock_overlay(); okk, msg = True, "unlocked: the overlay follows the markers and the LiDAR again"
            with _lock:
                _state["autotower"] = {"status": msg}
            self._send(200 if okk else 409, msg.encode())
            return
        if self.path == "/cmd/redetect":
            with _lock:
                locked = _state["locked"]
            if locked:
                with _lock:
                    _state["autotower"] = {"status": "overlay is locked: unlock it before Re-detect"}
                self._send(200, b"OK")
                return
            _reset_fit(relayer=True)             # LiDAR: re-measure where the tower stands and how tall it is
            with _lock:
                _state["autotower_redetect"] = True
            self._send(200, b"OK")
            return
        if self.path in ("/cmd/autotower/on", "/cmd/autotower/off"):
            with _lock:
                _state["autotower_auto"] = self.path.endswith("/on")
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
                    now_ms = int(time.time() * 1000)
                    if _state["marks_log"] and now_ms - _state["marks_log"][-1]["t_ms"] < 2000:
                        _new_entry = None                  # a double tap on MARK: the same move, not a second one
                    else:
                        _state["marks_log"].append({"seq": _state["mark_seq"], "t_ms": now_ms, "labelled": False})
                        _new_entry = _state["marks_log"][-1]
                        _state["auto"] = dict(seq=_new_entry["seq"], status="working", message="looking for the pulled block...")
                else:
                    self._send(404, b"unknown command")
                    return
            if cmd == "mark" and _new_entry is not None:
                if _state.get("dataset", {}).get("on"):
                    threading.Thread(target=_auto_job, args=(_new_entry,), daemon=True).start()
                else:
                    with _lock:
                        _state["auto"] = dict(seq=_new_entry["seq"], status="unsure",
                                              message="Data capture is off, so there are no pictures to look at. Turn on 'Start data capture' and press MARK after the next move.")
            self._send(200, b"OK")
            return
        if self.path == "/android":
            try:
                t_hdr = time.time()
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n)
                if len(raw) != n:
                    self.close_connection = True
                    return
                t_body = time.time()
                with _lock:
                    _android_dbg.append({"at": t_hdr, "body_s": round(t_body - t_hdr, 3), "kb": n // 1024,
                                         "ip": self.client_address[0], "ua": (self.headers.get("User-Agent") or "")[:40],
                                         "conn": self.headers.get("Connection", ""), "proto": self.request_version,
                                         "draw": self.headers.get("X-Draw"), "enc": self.headers.get("X-Enc"),
                                         "up": self.headers.get("X-Up")})
                now = time.time()
                seq = int(self.headers.get("X-Frame", 0) or 0)   # the app encodes in parallel, so frames can overtake
                with _lock:
                    recording = bool(_state["capturing"]) and (now - _state["preview_t"] < 3.0)
                    last = _state.get("android_frame", 0)
                    stale = seq > 0 and 0 < last - seq < 100         # a big jump back means the app restarted
                    if not stale:
                        _state["android_frame"] = seq
                if stale:
                    with _lock:
                        ids = _state["android_markers"]
                        saved = _state["android_saved"]
                    self._send(200, json.dumps({"markers": ids, "recording": recording, "saved": saved}).encode(),
                               "application/json")
                    return
                with _lock:
                    _state.update(android_raw=raw, android_t=now)
                    _android_times.append(now)
                    save = recording and now - _state.get("android_saved_t", 0.0) >= 1.0 / max(ANDROID_SAVE_FPS, 0.1)
                    if save:
                        _state["android_saved_t"] = now
                # Markers + overlay run on a worker; the reply doesn't wait for them, so the phone can send the
                # next frame straight away (it gets the markers of the latest frame the worker finished).
                with _lock:
                    _state["android_seq"] += 1
                    _new_frame.notify_all()
                _workers["android"].submit(raw)
                if save:
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
                    ids = _state["android_markers"]
                self._send(200, json.dumps({"markers": ids, "recording": recording, "saved": saved}).encode(),
                           "application/json")
            except Exception as e:
                self._send(500, f"android error: {e}".encode())
            return
        if self.path == "/depth":
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                _process_depth(self.headers, body)
            except Exception as e:
                print("depth frame skipped:", type(e).__name__, e)
            self._send(200, b"OK")
            return
        if self.path == "/preview":
            try:
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n)
                _workers["iphone"].submit(raw)
                with _lock:
                    if not _counter_base.get("_seen"):          # first picture since the server started: the phone still shows the old game's counts
                        _counter_base["_seen"] = 1
                        if self.headers.get("X-Capturing") != "1":
                            _counter_base["saved"] = int(self.headers.get("X-Saved", 0) or 0)
                            _counter_base["marks"] = int(self.headers.get("X-Marks", 0) or 0)
                    _state["preview_seq"] += 1
                    _new_frame.notify_all()
                    _state.update(
                        jpeg_raw=raw, preview_t=time.time(),
                        capturing=self.headers.get("X-Capturing") == "1",
                        saved=int(self.headers.get("X-Saved", 0) or 0),
                        tracking=self.headers.get("X-Tracking", "?"),
                        marks=int(self.headers.get("X-Marks", 0) or 0))
                    _preview_times.append(time.time())
                    if _state["want"] is not None and _state["capturing"] == _state["want"]:
                        _state["want"] = None
                    reply = {"capture": _state["want"], "mark_seq": _state["mark_seq"]}
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
                            _upload_times.append(time.time())
                        print(f"✅ Saved: {filename} ({len(file_data):,} bytes)")
                    except Exception as e:
                        print(f"❌ Error parsing file: {e}")
                self._send(200, b"OK")
            except Exception as e:
                print(f"❌ Server error: {e}")
                self._send(500, f"Server Error: {e}".encode())
            return

        self.close_connection = True        # body not read
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
                writer = fc.DatasetWriter(SAVE_DIR, os.path.join(HERE, "dataset", name), stride=1, workers=3)   # every frame, cut by 3 worker processes
                writer.skip_existing()
                started_marks = set(_g.glob(f"{SAVE_DIR}/*_mark.json"))
                with _lock:
                    _state["dataset"].update(name=name, frames=0, crops=0)
            if writer is not None:
                with _lock:
                    writer.layers = max(1, min(18, _effective(_state["tower"]["layers"], _state["tower"]["removed"])[0]))
                writer.deltas = {k: ([float(v) for v in _delta[k]] if _delta[k] is not None else None) for k in ("iphone", "android")}
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
AUTO_EVERY = 12.0      # seconds between automatic tower checks


def _autotower_loop():
    """Re-detect (button): a full measurement that resets the tower.  Auto (on by default): a check every
    AUTO_EVERY seconds that follows layer-count changes and 90-degree turns, keeping the pulled blocks."""
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

    def rotate_frame():
        k0 = _tp.TOWER_ROT
        _tp.set_tower_rotation(min(((k0 + 1) % 4, (k0 - 1) % 4), key=lambda k: min(k, 4 - k)))
        _last_pose.clear(); _pose_lock.clear(); _lock_valid.clear(); _pose_keep.clear()
        _reset_fit(relayer=True)                  # the fitted correction was measured in the old frame: start it again

    def lidar_layers():
        """Layer count from the iPhone's LiDAR top height, if it is fresh; steadier than the two-camera edge."""
        with _lock:
            top = _state["lidar_top"]
        if not top or time.time() - top[1] > 3.0:
            return None
        return max(1, int(round((top[0] - 2.0) / ta.LAYER_MM)))

    def evidence(ratio):
        """How clearly the top blocks run one way: the ratio is >1 for x and <1 for y."""
        return max(ratio, 1.0 / ratio)

    streak = {"key": None, "n": 0}        # an automatic change must be seen twice in a row before it is applied

    def auto_check():
        """One short burst; changes the layer count or the frame only on a repeated, confident reading.
        Pulled blocks are kept.  Returns seconds until the next check."""
        hint = lidar_layers()
        if hint is None:                  # the two-camera layer count is unreliable on a tall tower: wait for the LiDAR
            say("auto: waiting for the LiDAR tower fit; tower left as it is")
            return AUTO_EVERY
        r = ta.analyse_burst(grab, n=6, interval=0.3, layers_hint=hint)
        if not r["ok"]:
            streak.update(key=None, n=0)
            say(f"auto: no reading ({r['reason'] or 'could not tell which way the top blocks run'}); tower left as it is")
            return AUTO_EVERY
        L, axis, ratio = r["layers"], r["axis"], evidence(r["ratio"])
        with _lock:
            have = _state["tower"]["layers"]
        turned = (axis == "x") != ((L - 1) % 2 == 0)
        if turned and ratio < 2.0:
            streak.update(key=None, n=0)
            say(f"auto: {L} layers, but not sure which way the top blocks run (evidence x{ratio:.1f}); tower left as it is")
            return AUTO_EVERY
        if not turned and L == have:
            streak.update(key=None, n=0)
            say(f"auto: {L} layers ({'LiDAR' if hint else 'cameras'}), top blocks along {axis} - matches", layers=L, axis=axis)
            return AUTO_EVERY
        key = (L, axis, turned)
        streak.update(key=key, n=streak["n"] + 1 if streak["key"] == key else 1)
        if streak["n"] < 2:
            say(f"auto: saw {L} layers{' with the tower turned 90 degrees' if turned else ''}; checking again to confirm")
            return 2.0
        streak.update(key=None, n=0)
        if turned:
            rotate_frame()
            say("auto: the tower is turned 90 degrees - frame rotated, measuring again")
            return 2.0
        with _lock:
            removed = [list(b) for b in _state["tower"]["removed"]]
            top_pulled = max((b[0] for b in removed), default=-1)
            if top_pulled < L:
                _state["tower"] = {"layers": L, "removed": removed}
                _state["tower_epoch"] += 1
                _state["autotower_last"] = f"{L} layers, top blocks along {axis}"
        if top_pulled >= L:
            say(f"auto: saw {L} layers, but a block in layer {top_pulled + 1} is marked pulled; ignored")
        else:
            say(f"auto: tower is now {L} layers (was {have}, height from the {'LiDAR' if hint else 'cameras'}); "
                f"{len(removed)} pulled block(s) kept",
                layers=L, axis=axis, z=r["z"], ncc=r["ncc"], cams=r["cams"])
        return AUTO_EVERY

    next_auto = time.time() + 5.0
    while True:
        time.sleep(0.4)
        try:
            with _lock:
                go = _state["autotower_redetect"]
                auto = _state["autotower_auto"] and not _state["tracker_on"] and not _state["locked"]
                busy = bool(_state["capturing"]) or _state["dataset"]["on"]
            if not go:
                _resnap_moved_cameras(busy)
                if auto and time.time() >= next_auto:
                    if busy:
                        say("auto: paused while a game is being recorded")
                        next_auto = time.time() + AUTO_EVERY
                    else:
                        next_auto = time.time() + auto_check()
                continue
            with _lock:
                _state["autotower_redetect"] = False
            streak.update(key=None, n=0)
            _delta["iphone"] = _delta["android"] = None
            next_auto = time.time() + AUTO_EVERY
            if busy:
                say("not while a game is being recorded")
                continue
            final, t_start = None, time.time()
            try:                                  # the board may sit differently every game: measure where the markers are
                if _tp.FIXED:
                    raise RuntimeError("the board is calibrated and permanent")
                import calibrate_markers as cm
                say("measuring where the markers are (keep both phones still)...")
                mm = cm.measure_map(10, grab_fn=lambda p: grab()["iphone" if "preview" in p else "android"], pause=0.3, log=lambda m: None)
                _tp.save_marker_map(mm, time.strftime("%Y-%m-%d %H:%M:%S"))
                _pose_lock.clear(); _lock_valid.clear(); _pose_keep.clear()
                _reset_fit(relayer=True)
            except Exception as e:
                if not _tp.FIXED:
                    say(f"could not re-measure the markers ({e}); using the last measurement")
                    time.sleep(2.0)
            for attempt in range(1, 4):
                say(f"measuring (attempt {attempt}/3, keep both phones still)...")
                r = ta.analyse_burst(grab, n=8, interval=0.3, layers_hint=lidar_layers())
                if not r["ok"]:
                    say(f"attempt {attempt}/3 failed: {r['reason']}")
                    time.sleep(1.0)
                    continue
                L, axis, ratio = r["layers"], r["axis"], evidence(r["ratio"])
                expect_x = ((L - 1) % 2 == 0)
                if (axis == "x") != expect_x:
                    if ratio < 2.0:
                        say(f"{L} layers, but not sure which way the top blocks run (evidence x{ratio:.1f}). "
                            "Is the top layer complete?")
                        continue
                    rotate_frame()
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
                say(f"done in {time.time() - t_start:.0f} s: {L} layers, top blocks along {axis}",
                    layers=L, axis=axis, z=r["z"], ncc=r["ncc"], cams=r["cams"])
                threading.Thread(target=_refine_after_redetect, daemon=True).start()
        except Exception as e:
            say(f"error: {type(e).__name__}: {e}")


# --- "which layer?": the player types left / middle / right after a MARK, the face crops find the layer ----
AUTO_APPLY_PCT = 75     # the finder changes the tower by itself only when it is at least this sure (82% or better in past games)
AFTER_S = 2.5          # the tower is judged from 0.8 s to this many seconds after the tap (2.5 = 3.0 = 4.5 in accuracy on 57 moves; 2.0 is worse)


def _index_latest(run_dir):
    """Timestamp (ms) of the newest face crop written so far for this run; 0 if none."""
    try:
        with open(os.path.join(run_dir, "index.jsonl"), "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            lines = f.read().splitlines()
        for line in reversed(lines):
            try:
                return int(json.loads(line)["ts_ms"])
            except Exception:
                continue
    except OSError:
        pass
    return 0


def _resnap_moved_cameras(busy):
    """A phone that was moved or turned has a new view of the tower: once it has been still for 1.5 s, fit its outline again.
    Not while a game is being recorded or the overlay is locked: then the overlay must not change by itself."""
    with _lock:
        locked = _state["locked"]
    if busy or locked or _delta_job["on"] or _active_now():
        return
    now = time.time()
    for kind, t0 in list(_moved_at.items()):
        if now - t0 < 1.5:
            continue
        _moved_at.pop(kind, None)
        _delta_job["on"] = True

        def run(k=kind):
            try:
                msg = _refine_camera(k)
                with _lock:
                    _state["autotower"] = {"status": f"{k} view changed: {msg}"}
            finally:
                _delta_job["on"] = False
        threading.Thread(target=run, daemon=True).start()
        return


_delta_job = {"on": False}


def _refine_after_redetect():
    """Once the LiDAR has settled the layer count, snap the Android overlay to the tower's outline."""
    deadline = time.time() + 15.0
    while time.time() < deadline:
        with _fit_lock:
            if _fit_st["converged"] and not _fit_st["relayer"]:
                break
        time.sleep(0.5)
    msg = ". ".join(_refine_camera(k) for k in ("iphone", "android"))
    with _lock:
        cur = dict(_state["autotower"])
        cur["status"] = f"{cur.get('status', '')}. {msg}"
        _state["autotower"] = cur


def _depth_hint(entry, prev_ms, start_ms, pos, slots, layers, run_dir):
    """EXPERIMENT (logged, never acted on): what the iPhone's LiDAR says about which layer lost a block.  For each face of the
    typed column it compares the depth of the face before and after the move; a pulled block leaves a hole, so its face
    moves back or disappears.  Written next to the finder's answer so the next games show whether it is worth trusting."""
    import depth_protrusion as dp
    mk = entry["t_ms"]
    a_lo = (prev_ms + 1000) if prev_ms else (start_ms + 500)
    a_hi = min(a_lo + 3500, mk - 2000)
    b_lo, b_hi = mk + 800, mk + int(AFTER_S * 1000)
    held = _pose_lock.get("iphone") or _pose_keep.get("iphone")
    if held is None or a_hi - a_lo < 1500:
        return None
    R, t = _tp._corrected(*held)
    cam = -R.T @ t
    stamps = sorted(int(os.path.basename(f).split("_")[0]) for f in glob.glob(f"{SAVE_DIR}/*_intrinsics.json"))
    def window(lo, hi):
        acc = collections.defaultdict(list)
        for ts in stamps:
            if not (lo <= ts <= hi):
                continue
            fr = _tp.load_frame(str(ts))
            if fr is None:
                continue
            P = _tp.depth_to_tower(fr, R, t)
            for k, (v, n) in dp.face_offsets(P, cam, layers).items():
                acc[k].append((v, n))
        return {k: (float(np.median([a[0] for a in v])), float(np.median([a[1] for a in v]))) for k, v in acc.items() if len(v) >= 2}
    A, B = window(a_lo, a_hi), window(b_lo, b_hi)
    allmv = [abs(B[k][0] - A[k][0]) for k in A if k in B]
    if len(allmv) < 6:
        return None
    med = float(np.median(allmv)); mad = float(1.4826 * np.median(np.abs(np.array(allmv) - med)) + 1.0)
    out = []
    for layer in range(layers):
        slot = slots.get(layer, {}).get(pos)
        if slot is None:
            continue
        best = 0.0
        for sign in (1, -1):
            k = (layer, slot, sign)
            if k in A and k in B:
                best = max(best, abs(B[k][0] - A[k][0]))
            elif k in A and A[k][1] >= 30:                # a well-seen face is gone after the move: a hole shows nothing there
                best = max(best, 25.0)
        if best > 0:
            out.append({"layer": layer + 1, "mm": round(best, 1), "z": round((best - med) / mad, 1)})
    out.sort(key=lambda d: -d["z"])
    return out[:3]


def _first_crop_ms(ds):
    """Timestamp (ms) of the first face crop of this run (0 if unknown)."""
    try:
        with open(os.path.join(HERE, "dataset", ds["name"], "index.jsonl")) as f:
            return int(json.loads(f.readline())["ts_ms"])
    except Exception:
        return 0


def prev_ms_for_choice(entry):
    """Seconds (server clock) of the tap before this one, or None."""
    with _lock:
        earlier = [m["t_ms"] for m in _state["marks_log"] if m["seq"] < entry["seq"]]
    return (max(earlier) / 1000.0) if earlier else None


def _wait_for_windows(entry, say):
    """Wait for the hand to leave the tower and the pictures of the calm stretch after the move to be ready, and choose which
    stretches to compare.  Returns (windows or None, dataset info, run dir, layers, earlier taps, error text or None)."""
    mark_s = entry["t_ms"] / 1000.0
    say(status="waiting for the hand to leave the tower...")
    wins, give_up = None, time.time() + 25.0
    while time.time() < give_up:
        w = _move_windows(mark_s)
        if w == "wait":
            time.sleep(0.3)
            continue
        wins = w
        break
    with _lock:
        ds0 = dict(_state["dataset"])
    if wins and wins["a"][0] < _first_crop_ms(ds0) + 300:   # the calm "before" would be from before recording began: no pictures of it
        wins = None
    if wins is not None:
        # Which pictures to compare?  The calm stretch just before this move is the truest "before", but on a quiet game the
        # old choice (right after the previous tap) did as well on recorded games, so keep it unless the tower changed in
        # between (a block was put on top) or hands were in either of the old windows.
        prev_s = prev_ms_for_choice(entry)
        with _lock:
            placed_mode = bool(_state["placed_on_top"])
            earlier = any(m["seq"] < entry["seq"] for m in _state["marks_log"])
        old_a = ((prev_s + 1.0), (prev_s + 4.5)) if prev_s else None
        old_b = (mark_s + 0.8, mark_s + AFTER_S)
        def busy(lo, hi):
            return any(lo - 0.1 <= e <= hi + 0.1 for e in list(_act["events"]))
        if not (earlier and placed_mode) and old_a is not None and not busy(*old_a) and not busy(*old_b):
            wins = None
    if wins is None:                                       # nothing recorded about hands: judge by the clock, as before
        wait = mark_s + AFTER_S + 0.5 - time.time()
        if wait > 0:
            time.sleep(wait)
    with _lock:
        ds = dict(_state["dataset"]); layers = _effective(_state["tower"]["layers"], _state["tower"]["removed"])[0]
        prev = [m for m in _state["marks_log"] if m["seq"] < entry["seq"]]
    if not ds.get("name"):
        return wins, ds, None, layers, prev, "data capture was not on, so there are no face crops to look at"
    run_dir = os.path.join(HERE, "dataset", ds["name"])
    # The face crops are cut from the saved frames in the background and can lag behind when the Mac is busy:
    # wait until they have caught up with the "after" window instead of judging a move from half the pictures.
    need_ms = (wins["b"][1] + 300) if wins else entry["t_ms"] + int((AFTER_S + 0.3) * 1000)
    deadline = time.time() + 40.0
    while _index_latest(run_dir) < need_ms and time.time() < deadline:
        say(status="waiting for the face crops to catch up...")
        time.sleep(0.7)
    return wins, ds, run_dir, layers, prev, None


AUTO_PCT = 90          # at or above this the finder names the block by itself and asks only "yes / no"


def _column_name(slots_by_layer, layer, slot):
    for name, sl in (slots_by_layer.get(layer) or {}).items():
        if sl == slot:
            return name
    return None


_counter_base = {}      # counter name -> value it had when "new game" was pressed (the phone keeps counting; the page starts from 0)


def _counter(s, key):
    """A running counter of the phone or the server as the page should show it: counted from the last new game.  If the phone
    started a new capture (its counter fell below the base) the base no longer applies."""
    v = int(s[key] or 0)
    b = _counter_base.get(key, 0)
    if v < b:
        _counter_base[key] = b = 0
    return v - b


def _new_game():
    """Everything about the game that was just played goes; the setup (markers, overlay fit, lock) stays.  Recording stops, the
    pulled blocks, taps, answers, labels, game over and the hand history are cleared, and the tower is the same height, whole."""
    with _lock:
        if _state["capturing"]:
            _state["want"] = False
        _state["dataset"]["on"] = False
        _state["dataset"].update(name=None, frames=0, crops=0)
        _state["tower"] = {"layers": _state["tower"]["layers"], "removed": []}
        _state["tower_epoch"] += 1                     # every watch (height, standing, loose) starts again from this tower
        _state["marks_log"] = []
        _state["last_label"] = None
        _state["auto"] = None
        _state["label_stats"] = {"moves": 0, "wrong": 0}
        _state["game_over"] = None
        _state["loose"] = []
        _state["loose_info"] = None
        _state["height_info"] = None
        _state["autotower"] = {"status": "new game: the tower is whole again (press Re-detect if you rebuilt it)"}
        for k in ("saved", "marks", "files", "android_saved"):
            _counter_base[k] = int(_state[k] or 0)           # "Frames saved" and "Moves marked" count from here
    _act["events"].clear()


def _auto_job(entry):
    """After a tap, with no typing: wait for the hand to leave, find the block that changed most anywhere in the tower, and
    say how sure that is.  At AUTO_PCT or more the block goes into the tower at once (red/green model, 3-D view) and the
    player only answers yes / no; below that nothing changes and the player types left / middle / right and the layer."""
    import move_infer as mi
    import jenga_rules
    def say(**kw):
        with _lock:
            _state["auto"] = dict(seq=entry["seq"], at=time.strftime("%H:%M:%S"), **kw)
    def typed_meanwhile():
        if entry["labelled"]:                                   # the player typed the move meanwhile: that path answers it
            with _lock:
                if _state["auto"] and _state["auto"].get("seq") == entry["seq"] and _state["auto"].get("status") == "working":
                    _state["auto"] = None
            return True
        return False
    try:
        wins, ds, run_dir, layers, prev, err = _wait_for_windows(entry, lambda **kw: say(status="working", message=kw.get("status", "")))
        if typed_meanwhile():
            return
        if err:
            return say(status="unsure", message=err + ". Type left / middle / right and the layer.")
        with _lock:
            base_layers = int(_state["tower"]["layers"]); pulled = [tuple(r) for r in _state["tower"]["removed"]]
        n_eff, removed_eff = _effective(base_layers, pulled)
        present = {(l, s) for l in range(n_eff) for s in range(3)} - {tuple(r) for r in removed_eff}
        full = [l for l in range(n_eff) if all((l, s) in present for s in range(3))]
        top_full = max(full) if full else n_eff - 1
        cands = {b for b in present if b[0] <= top_full - 1}      # a pull comes from below the top complete layer
        with open(os.path.join(run_dir, "index.jsonl")) as f:
            start_ms = json.loads(f.readline())["ts_ms"]
        r = mi.infer_free(run_dir, entry["t_ms"], prev[-1]["t_ms"] if prev else None, start_ms, cands,
                          a_window=wins["a"] if wins else None, b_window=wins["b"] if wins else None, after=AFTER_S)
        if typed_meanwhile():
            return
        if not r["ok"]:
            return say(status="unsure", message=r["reason"] + ". Type left / middle / right and the layer.")
        took = time.time() - entry["t_ms"] / 1000.0
        L, S, pct = r["layer"], r["slot"], int(r["confidence_pct"])
        col = None
        held = _last_pose.get("iphone")
        if held:
            try:
                col = _column_name(mi.slots_for_positions(held[0], held[1], _k_iphone(960)[0], n_eff), L - 1, S)
            except Exception:
                col = None
        where = f"layer {L}" + (f", {col} column" if col else f", slot {S}")
        rec = {"seq": entry["seq"], "auto": {"layer": L, "slot": S, "pct": pct, "z": round(r["z"], 1), "gap": round(r["gap"], 1),
                                              "top": r["top"], "took_s": round(took, 1)}}
        try:
            with open(os.path.join(run_dir, "labels.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        if pct >= AUTO_PCT:
            blk = [L - 1, S]
            with _lock:
                if entry["labelled"]:
                    return
                entry["labelled"] = True                       # answered: only yes / no from here
                rem = [b for b in _state["tower"]["removed"] if list(b) != blk]
                _state["tower"] = {"layers": _state["tower"]["layers"], "removed": rem + [blk]}
                _state["auto"] = dict(seq=entry["seq"], at=time.strftime("%H:%M:%S"), status="sure", layer=L, slot=S, column=col,
                                      confidence=pct, block=blk, applied=True, answered=None, took=round(took, 1),
                                      message=f"{where} - {pct}% sure ({took:.1f} s). Added to the tower and the 3-D view.")
        else:
            say(status="unsure", layer=L, slot=S, column=col, confidence=pct, applied=False, took=round(took, 1),
                message=f"Best guess {where} - only {pct}% sure ({took:.1f} s), so nothing was changed. "
                        f"Type left / middle / right and the layer, e.g. l {L}.")
    except Exception as e:
        say(status="unsure", message=f"{type(e).__name__}: {e}. Type left / middle / right and the layer.")


def _auto_answer(yes):
    """The player's yes / no on the automatic answer.  Yes keeps it (and is logged as the true layer); no takes the block out
    again and gives the tap back, so the move can be typed."""
    with _lock:
        a = _state["auto"]
        if not a or a.get("status") != "sure" or a.get("answered"):
            return False, "nothing to answer"
        entry = next((m for m in _state["marks_log"] if m["seq"] == a["seq"]), None)
        a["answered"] = "yes" if yes else "no"
        run = _state["dataset"].get("name")
        if yes:
            _state["label_stats"]["moves"] += 1
            a["message"] = a["message"].split(" - ")[0] + " - confirmed."
        else:
            blk = a.get("block")
            _state["tower"] = {"layers": _state["tower"]["layers"],
                               "removed": [b for b in _state["tower"]["removed"] if list(b) != blk]}
            _state["label_stats"]["wrong"] += 1
            if entry:
                entry["labelled"] = False                       # typing works again for this tap
            a["status"] = "unsure"; a["applied"] = False
            a["message"] = "Undone. Type left / middle / right and the layer, e.g. l 5."
    if run:
        try:
            rec = ({"seq": a["seq"], "correction": a["layer"], "by": "yes"} if yes else {"seq": a["seq"], "rejected": a["layer"]})
            with open(os.path.join(HERE, "dataset", run, "labels.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
    return True, "ok"


# --- the record of a game: every change to the tower, with its time, so the game can be replayed afterwards ----------------
_ev = {"run": None, "epoch": None, "removed": set(), "layers": None, "placed": None, "over": False}


def _ev_write(run, **rec):
    try:
        with open(os.path.join(HERE, "dataset", run, "moves.jsonl"), "a") as f:
            f.write(json.dumps(dict(t=int(time.time() * 1000), **rec)) + "\n")
    except OSError:
        pass


def _events_loop():
    """Watches the tower state and writes each change (a block taken or put back, game over, the setup being reset) to the
    run's moves.jsonl.  Watching the state, not the places that change it, means no way of changing it is missed."""
    while True:
        time.sleep(0.35)
        try:
            with _lock:
                name = _state["dataset"].get("name")
                tower = {"layers": int(_state["tower"]["layers"]), "removed": {tuple(b) for b in _state["tower"]["removed"]}}
                epoch, placed, over = _state["tower_epoch"], bool(_state["placed_on_top"]), _state["game_over"]
                auto, ll = dict(_state["auto"] or {}), dict(_state["last_label"] or {})
            if not name:
                _ev["run"] = None
                continue
            if _ev["run"] != name or _ev["epoch"] != epoch:
                first = _ev["run"] != name
                _ev.update(run=name, epoch=epoch, removed=set(tower["removed"]), layers=tower["layers"], placed=placed, over=bool(over))
                _ev_write(name, ev="meta" if first else "reset", layers=tower["layers"], placed=placed,
                          removed=sorted([list(b) for b in tower["removed"]]))
                continue
            if placed != _ev["placed"]:
                _ev_write(name, ev="placed", on=placed); _ev["placed"] = placed
            if tower["layers"] != _ev["layers"]:
                _ev_write(name, ev="layers", layers=tower["layers"]); _ev["layers"] = tower["layers"]
            for b in sorted(tower["removed"] - _ev["removed"]):
                how = "auto" if auto.get("block") == list(b) and auto.get("status") == "sure" else ("typed" if ll.get("block") == list(b) else "set")
                seq = auto.get("seq") if how == "auto" else ll.get("seq")
                _ev_write(name, ev="pull", block=list(b), how=how, seq=seq, pct=auto.get("confidence") if how == "auto" else ll.get("confidence_pct"))
            for b in sorted(_ev["removed"] - tower["removed"]):
                _ev_write(name, ev="unpull", block=list(b), seq=auto.get("seq"))
            _ev["removed"] = set(tower["removed"])
            if bool(over) != _ev["over"]:
                _ev_write(name, ev="gameover" if over else "gameover_cleared", reason=(over or {}).get("reason") if over else None)
                _ev["over"] = bool(over)
        except Exception:
            pass


def _label_job(entry, pos, user_layer):
    """Wait for the 'after' pictures, then find which layer's block in that column changed most."""
    import move_infer as mi
    def say(**kw):
        with _lock:
            _state["last_label"] = dict(seq=entry["seq"], pos=pos, **kw)
    try:
        wins, ds, run_dir, layers, prev, err = _wait_for_windows(entry, say)
        if err:
            return say(status="error", message=err)
        held = _last_pose.get("iphone")
        if not held:
            return say(status="error", message="the iPhone has not found the markers, so I can't map left/middle/right")
        K = _k_iphone(960)[0]
        slots = mi.slots_for_positions(held[0], held[1], K, layers)             # left/right in the phone's own picture
        with open(os.path.join(run_dir, "index.jsonl")) as f:
            start_ms = json.loads(f.readline())["ts_ms"]
        gone = mi.already_removed(run_dir, entry["seq"])
        def look(p):
            cands = [l for l in range(max(2, layers - 1)) if (p, l) not in gone]   # nothing is ever put back
            return mi.infer(run_dir, entry["t_ms"], prev[-1]["t_ms"] if prev else None, start_ms, p, slots, cands, after=AFTER_S,
                            a_window=wins["a"] if wins else None, b_window=wins["b"] if wins else None)
        r = look(pos)
        flipped = None
        if pos in ("left", "right"):
            # Which hand is "left" depends on how you stand and how the phone hangs, and that changed between games.  The
            # picture knows: if the other outer column shows clearly more change than the one you typed, it is the other one.
            other = "right" if pos == "left" else "left"
            r2 = look(other)
            zt = r["z"] if r["ok"] else -9.0
            if r2["ok"] and r2["z"] >= 3.0 and r2["z"] > zt + 1.0:
                pos, r, flipped = other, r2, other
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
            took = time.time() - entry["t_ms"] / 1000.0
            conf = int(r.get("confidence_pct", 0))
            trusted = bool(r["confident"]) and conf >= AUTO_APPLY_PCT      # sure enough to change the tower without being asked
            msg = (f"you typed {'left' if pos == 'right' else 'right'} but the change is on the {pos} column. " if flipped else "") \
                + f"{pos} column -> layer {r['layer']} (about {conf}% sure, score {r['z']:.1f}, next best layer {r['runner_up']} at {r['runner_up_z']:.1f}; {took:.1f} s)"
            if r.get("suspicious"):
                msg += ". Unusually large: a hand or shadow may be over that face"
            elif not trusted and not user_layer:
                msg += ". NOT SURE ENOUGH to change the tower: type the real layer, e.g. " + f"{pos[0]} {r['layer']}"
            if user_layer:
                msg += f". You said layer {user_layer}: {'MATCH' if agree else 'different'}"
            elif trusted:
                msg += ". Wrong? type the real layer number and press Enter"
            # the pulled block is gone for good: feed it to the red/green rules.  Your typed layer always counts; the finder's
            # own answer counts only when it is sure enough (otherwise the tower waits for you).
            blk = None
            pull_layer = user_layer or (r["layer"] if trusted else None)
            slot = slots.get(pull_layer - 1, {}).get(pos) if pull_layer and 1 <= pull_layer <= layers else None
            if slot is not None:
                blk = [pull_layer - 1, slot]
                with _lock:
                    rem = [b for b in _state["tower"]["removed"] if list(b) != blk]
                    _state["tower"] = {"layers": _state["tower"]["layers"], "removed": rem + [blk]}
                msg += f". Marked layer {pull_layer} slot {slot} as removed"
            say(status="done", message=msg, layer=r["layer"], confident=r["confident"], top=r["top"],
                block=blk, pos_slots={l: s[pos] for l, s in slots.items()})

            def _hint_later(p=pos, st=start_ms, sl=slots, ly=layers, rd=run_dir, e=entry, pv=(prev[-1]["t_ms"] if prev else None)):
                try:
                    h = _depth_hint(e, pv, st, p, sl, ly, rd)
                except Exception as ex:
                    h = None
                    print("depth hint failed:", type(ex).__name__, ex)
                if h:
                    with _lock:
                        ll = _state["last_label"]
                        if h[0]["z"] >= 4.0 and ll and ll.get("seq") == e["seq"] and ll.get("status") == "done":   # only a clear signal is shown
                            ll["message"] += "  [LiDAR experiment: biggest change at layer " + ", ".join(f"{d['layer']} ({d['mm']} mm)" for d in h[:2]) + "]"
                    try:
                        with open(os.path.join(rd, "labels.jsonl"), "a") as f:
                            f.write(json.dumps({"seq": e["seq"], "depth": h}) + "\n")
                    except OSError:
                        pass
            threading.Thread(target=_hint_later, daemon=True).start()
        with open(os.path.join(run_dir, "labels.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        say(status="error", message=f"{type(e).__name__}: {e}")


def _coverage_text():
    """Which blocks can the two phones, where they stand right now, actually judge?  One cell per block."""
    import face_crops as fc, move_infer as mi
    layers = _eff_state()[0]
    poses = {n: _last_pose.get(n) for n in ("iphone", "android")}
    if not poses["iphone"]:
        return "the iPhone has not found the markers yet, so I can't tell (needs 3+ markers)"
    K = _k_iphone(960)[0]
    with _lock:
        vrot = int(_state["view_rot"]["iphone"])
    slots = mi.slots_for_positions(poses["iphone"][0], poses["iphone"][1], K, layers, vrot)
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
            threading.Thread(target=_events_loop, daemon=True).start()
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
