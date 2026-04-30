import os
import shutil
import datetime


def generate_report(prefix, shorts_file, out_dir,
                    scenes, final_times, args,
                    shorts_thumbs=None, final_thumbs=None):
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, f"{prefix}_report.html")

    # 숏츠 원본을 output 폴더에 복사 (로컬 다운로드 시 함께 사용)
    shorts_basename = os.path.basename(shorts_file)
    shorts_dest = os.path.join(out_dir, shorts_basename)
    if os.path.abspath(shorts_file) != os.path.abspath(shorts_dest):
        shutil.copy2(shorts_file, shorts_dest)

    # HTML과 mp4가 같은 폴더 → 파일명만 사용
    shorts_rel = shorts_basename
    final_rel  = f"{prefix}_final.mp4"

    def fmt(s):
        if s is None:
            return None
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:05.2f}"

    def td_thumb(t, thumb=None):
        if t is None:
            return '<td class="na">—</td>'
        img = (f'<img src="{thumb}" class="thumb">' if thumb else '')
        return f'<td class="ts">{img}{fmt(t)}</td>'

    st_list = shorts_thumbs or [None] * len(scenes)
    ft_list = final_thumbs  or [None] * len(scenes)

    rows = ""
    for i, ((s, e), ft, sth, fth) in enumerate(
            zip(scenes, final_times, st_list, ft_list)):
        rows += f"""
      <tr>
        <td class="num">{i+1}</td>
        {td_thumb(s, sth)}
        {td_thumb(ft, fth)}
      </tr>"""

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{prefix} 리포트</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#111;color:#eee;font-family:'Segoe UI',sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{padding:9px 18px;background:#1a1a1a;border-bottom:1px solid #252525;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}}
header h1{{font-size:13px;color:#fff;font-weight:600}}
header .meta{{font-size:10px;color:#444;text-align:right;line-height:1.7}}
.main{{display:flex;flex:1;overflow:hidden;min-height:0}}
.table-wrap{{flex:2;overflow-y:auto;border-right:1px solid #252525;min-width:0}}
img.thumb{{height:80px;border-radius:4px;vertical-align:middle;margin-right:8px;object-fit:cover}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#1c1c1c;padding:6px 12px;color:#555;font-weight:500;position:sticky;top:0;text-align:left;white-space:nowrap}}
td{{padding:8px 12px;border-bottom:1px solid #1d1d1d;white-space:nowrap}}
td.num{{color:#444;text-align:center;width:36px}}
td.ts{{color:#999;font-variant-numeric:tabular-nums;vertical-align:middle;font-size:12px}}
td.na{{color:#333}}
th.shorts{{color:#a78bfa}}
th.final{{color:#f59e0b}}
tr:hover td{{background:#1a1a1a}}
.players{{flex:1;min-width:260px;max-width:480px;flex-shrink:0;display:flex;flex-direction:column;gap:6px;padding:8px;overflow-y:auto;background:#111}}
.cell{{display:flex;flex-direction:column;background:#1c1c1c;border-radius:8px;overflow:hidden;border:2px solid #2a2a2a;flex-shrink:0}}
.cell.ref{{border-color:#555}}.cell.final{{border-color:#f59e0b}}
.lbl{{padding:5px 8px;font-size:11px;text-align:center;background:#181818;flex-shrink:0;line-height:1.4}}
.cell.ref .lbl{{color:#aaa}}.cell.final .lbl{{color:#f59e0b}}
.lbl small{{display:block;color:#3a3a3a;font-size:10px}}
video{{width:100%;height:260px;background:#000;display:block;object-fit:contain}}
.vc{{padding:5px 6px;background:#161616;display:flex;flex-direction:column;gap:3px}}
.sr{{display:flex;align-items:center;gap:4px}}
.t{{font-size:9px;color:#555;min-width:38px;font-variant-numeric:tabular-nums}}
.t.r{{text-align:right}}
input[type=range]{{flex:1;-webkit-appearance:none;height:3px;border-radius:2px;cursor:pointer;outline:none}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:10px;height:10px;border-radius:50%;background:#fff;cursor:pointer}}
.br{{display:flex;align-items:center;justify-content:center;gap:3px}}
.btn{{background:#2a2a2a;border:none;color:#999;padding:3px 6px;border-radius:5px;cursor:pointer;font-size:10px}}
.btn:hover{{background:#363636;color:#fff}}
.btn.p{{background:#4a9eff;color:#fff;min-width:58px}}
.btn.p:hover{{background:#3a8eef}}
.btn.on{{background:#383838;color:#fff}}
select{{background:#2a2a2a;color:#999;border:none;padding:3px 5px;border-radius:5px;cursor:pointer;font-size:10px}}
</style>
</head>
<body>
<header>
  <h1>📊 {prefix} — 매칭 결과 리포트</h1>
  <div class="meta">생성: {now}<br>숏츠: {os.path.basename(shorts_file)} &nbsp;|&nbsp; threshold: {args.threshold} &nbsp;|&nbsp; buffer: {args.buffer}s</div>
</header>
<div class="main">
<div class="table-wrap">
  <table>
    <thead><tr>
      <th>#</th><th class="shorts">레퍼런스</th><th class="final">Final</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
<div class="players">
  <div class="cell ref">
    <div class="lbl">🎬 레퍼런스 숏츠<small>{os.path.basename(shorts_file)}</small></div>
    <video id="v0" src="{shorts_rel}" preload="metadata"></video>
    <div class="vc">
      <div class="sr">
        <span class="t" id="c0">00:00</span>
        <input type="range" id="s0" min="0" max="100" step="0.05" value="0" oninput="si(0)" onchange="sc(0)" onmousedown="dd(0,true)" onmouseup="dd(0,false)">
        <span class="t r" id="d0">00:00</span>
      </div>
      <div class="br">
        <button class="btn" onclick="sk(0,-5)">◀5s</button>
        <button class="btn p" id="pb0" onclick="tp(0)">▶</button>
        <button class="btn" onclick="sk(0,5)">5s▶</button>
        <button class="btn" id="mb0" onclick="tm(0)">🔊</button>
        <select onchange="sr(0,this.value)"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select>
      </div>
    </div>
  </div>
  <div class="cell final">
    <div class="lbl">🏆 결과 영상<small>{prefix}</small></div>
    <video id="v1" src="{final_rel}" preload="metadata" muted></video>
    <div class="vc">
      <div class="sr">
        <span class="t" id="c1">00:00</span>
        <input type="range" id="s1" min="0" max="100" step="0.05" value="0" oninput="si(1)" onchange="sc(1)" onmousedown="dd(1,true)" onmouseup="dd(1,false)">
        <span class="t r" id="d1">00:00</span>
      </div>
      <div class="br">
        <button class="btn" onclick="sk(1,-5)">◀5s</button>
        <button class="btn p" id="pb1" onclick="tp(1)">▶</button>
        <button class="btn" onclick="sk(1,5)">5s▶</button>
        <button class="btn on" id="mb1" onclick="tm(1)">🔇</button>
        <select onchange="sr(1,this.value)"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select>
      </div>
    </div>
  </div>
</div>
</div>
<script>
const V=[0,1].map(i=>document.getElementById('v'+i));
const drag=[false,false];
const fm=s=>{{if(isNaN(s))return'00:00';const h=~~(s/3600),m=~~((s%3600)/60),sec=~~(s%60);return h?[h,m,sec].map(n=>('0'+n).slice(-2)).join(':'):[m,sec].map(n=>('0'+n).slice(-2)).join(':');}};
const sp=t=>t.style.setProperty('--p',(t._v&&t._v.duration?(+t.value/t._v.duration*100):0).toFixed(2)+'%');
[0,1].forEach(i=>{{
  const v=V[i],s=document.getElementById('s'+i),c=document.getElementById('c'+i),d=document.getElementById('d'+i);
  s._v=v;
  v.addEventListener('loadedmetadata',()=>{{s.max=v.duration;d.textContent=fm(v.duration);}});
  v.addEventListener('timeupdate',()=>{{if(!drag[i]){{s.value=v.currentTime;c.textContent=fm(v.currentTime);sp(s);}}}});
  v.addEventListener('ended',()=>{{document.getElementById('pb'+i).textContent='▶';}});
}});
function dd(i,v){{drag[i]=v;}}
function si(i){{const s=document.getElementById('s'+i);document.getElementById('c'+i).textContent=fm(+s.value);sp(s);}}
function sc(i){{V[i].currentTime=+document.getElementById('s'+i).value;drag[i]=false;}}
function tp(i){{const v=V[i],b=document.getElementById('pb'+i);if(v.paused){{v.play().catch(()=>{{}});b.textContent='⏸';}}else{{v.pause();b.textContent='▶';}}}}
function sk(i,d){{V[i].currentTime=Math.max(0,Math.min((V[i].currentTime||0)+d,V[i].duration||0));}}
function sr(i,v){{V[i].playbackRate=+v;}}
function tm(i){{const v=V[i],b=document.getElementById('mb'+i);v.muted=!v.muted;b.textContent=v.muted?'🔇':'🔊';b.classList.toggle('on',v.muted);}}
</script>
</body>
</html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📊 리포트 저장: {report_path}")
