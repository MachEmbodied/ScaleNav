"""Build a local video gallery and overview from saved navigation rollouts."""
import argparse
import html
import json
import math
from pathlib import Path
import shutil
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BENCHMARK_TITLES = {"habitatgs_ln": "Habitat-GS-LN", "short_horizon_ovon": "Short-Horizon OVON"}


def read_cases(rows, manifest, videos):
    lookup = {(row["scene_id"], row["episode_id"]): row for row in manifest}
    keys = [(row["scene_id"], row["episode_id"]) for row in rows]
    if not rows or len(keys) != len(set(keys)) or len(lookup) != len(manifest):
        raise ValueError("Expected nonempty, uniquely identified episodes")
    cases = []
    for row, key in zip(rows, keys):
        episode = lookup[key]
        if row["instruction"] != episode["instruction"]:
            raise ValueError(f"Instruction differs between summary and manifest: {key}")
        if not row.get("video_path"):
            raise ValueError(f"Missing rollout video: {key}; run evaluation with --visualize or --save_rollout_video")
        path = videos / Path(row["video_path"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        capture = cv2.VideoCapture(str(path))
        first = last = None
        count = 0
        fps = capture.get(cv2.CAP_PROP_FPS)
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                last = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if first is None:
                    first = last
                count += 1
        finally:
            capture.release()
        positions = np.asarray(row["executed_positions_world"], dtype=float)
        if (not count or not math.isfinite(fps) or fps <= 0 or positions.ndim != 2
                or positions.shape[1] != 3 or not np.isfinite(positions).all()):
            raise ValueError(f"Invalid video or trajectory: {key}")
        if len(positions) not in (count, count + 1):
            raise ValueError(f"Video frames and trajectory samples do not align: {key}")
        contract = episode.get("target_contract") or {}
        polygons = []
        if contract.get("semantic_target_footprint_world_xz"):
            polygons.append(contract["semantic_target_footprint_world_xz"])
        polygons.extend(support["footprint_world_xz"] for support in contract.get("supporting_instances", []))
        cases.append({"row": row, "source": path, "first": first, "last": last,
                      "positions": positions[:, [0, 2]], "polygons": polygons,
                      "goal": np.asarray(episode["goal_xyz"])[[0, 2]].tolist() if not polygons else None,
                      "frames": count, "fps": fps, "duration": count / fps})
    return cases


def project(case, width, height):
    shapes = [case["positions"]] + [np.asarray(p) for p in case["polygons"]]
    if case["goal"] is not None:
        shapes.append(np.asarray([case["goal"]]))
    points = np.concatenate(shapes)
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high - low, 0.1)
    scale = min((width - 32) / span[0], (height - 32) / span[1])
    center = (low + high) / 2

    def transform(values):
        result = (np.asarray(values) - center) * [scale, -scale] + [width / 2, height / 2]
        return result.round(2).tolist()

    return transform(case["positions"]), [transform(p) for p in case["polygons"]], (
        transform([case["goal"]])[0] if case["goal"] is not None else None)


def route_svg(case):
    route, polygons, goal = project(case, 240, 220)
    points = lambda values: " ".join(f"{x},{y}" for x, y in values)
    shapes = "".join(f'<polygon points="{points(p)}" fill="#f9e1ae" stroke="#b17b22"/>' for p in polygons)
    if goal:
        shapes += f'<circle cx="{goal[0]}" cy="{goal[1]}" r="7" fill="#b17b22"/>'
    start = route[0]
    svg = (f'<svg viewBox="0 0 240 220" role="img" aria-label="Executed path and target">{shapes}'
           f'<polyline points="{points(route)}" fill="none" stroke="#d3dce1" stroke-width="3"/>'
           f'<polyline class="progress" points="{points(route[:1])}" fill="none" stroke="#087e8b" stroke-width="3"/>'
           f'<circle cx="{start[0]}" cy="{start[1]}" r="5" fill="white" stroke="#087e8b" stroke-width="2"/>'
           f'<circle class="position" cx="{start[0]}" cy="{start[1]}" r="5" fill="#087e8b"/></svg>')
    return svg, route


def overview(cases, output, title, start=0):
    from matplotlib import get_data_path
    font_path = Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    font = lambda size: ImageFont.truetype(str(font_path), size)
    columns, card_width, card_height, gap = 2, 756, 350, 24
    width = 2 * card_width + 3 * gap
    height = 120 + math.ceil(len(cases) / columns) * (card_height + gap) + gap
    canvas = Image.new("RGB", (width, height), "#f3f5f7")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 22), f"ScaleNav / {title}", font=font(34), fill="#172b3a")
    draw.text((gap, 72), f"{len(cases)} recorded rollouts · {len({c['row']['scene_id'] for c in cases})} scenes · start / last observation / executed path",
              font=font(19), fill="#536574")
    for i, case in enumerate(cases):
        row = case["row"]
        x, y = gap + (i % columns) * (card_width + gap), 120 + (i // columns) * (card_height + gap)
        draw.rounded_rectangle((x, y, x + card_width, y + card_height), 14, fill="white")
        draw.text((x + 16, y + 14), f"{start + i + 1:02d}   {row['scene_id']}", font=font(22), fill="#172b3a")
        instruction = "\n".join(textwrap.wrap(row["instruction"], width=75))
        draw.multiline_text((x + 16, y + 52), instruction, font=font(17), fill="#384f61", spacing=5)
        for index, frame in enumerate((case["first"], case["last"])):
            thumb = frame.copy()
            thumb.thumbnail((250, 150), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x + 16 + index * 262, y + 124))
            draw.text((x + 16 + index * 262, y + 104), "Start" if index == 0 else "Last observation", font=font(14), fill="#536574")
        route, polygons, goal = project(case, 190, 150)
        offset = lambda values: [(x + 550 + a, y + 124 + b) for a, b in values]
        for polygon in polygons:
            draw.polygon(offset(polygon), fill="#f9e1ae", outline="#b17b22")
        if goal:
            gx, gy = offset([goal])[0]
            draw.ellipse((gx - 5, gy - 5, gx + 5, gy + 5), fill="#b17b22")
        path = offset(route)
        draw.line(path, fill="#087e8b", width=3)
        for point, color in ((path[0], "white"), (path[-1], "#087e8b")):
            px, py = point
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color, outline="#087e8b", width=2)
        draw.text((x + 550, y + 104), "Path / target", font=font(14), fill="#536574")
        result = "Success" if row["success"] else "Failure"
        stop = "model STOP" if row.get("stopped_via_model") else "no model STOP"
        draw.text((x + 16, y + 292), f"{result} · {stop}    |    {row['path_length']:.2f} m    |    {case['duration']:.2f} s",
                  font=font(17), fill="#087e8b" if row["success"] else "#9f412f")
        draw.text((x + 16, y + 321), "Trajectory: open circle = start; filled circle = final position; amber = target", font=font(13), fill="#647683")
    canvas.save(output)


STYLE = """
:root{font-family:ui-sans-serif,system-ui,sans-serif;color:#243447;background:white}*{box-sizing:border-box}
body{max-width:1360px;margin:auto;padding:28px 24px}header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:24px}
h1{font-size:24px;letter-spacing:-.02em;margin:0}header p{color:#627586;font-size:13px;margin:6px 0 0}
select{font:inherit;font-size:13px;max-width:100%;border:1px solid #d5dfe5;border-radius:6px;padding:8px 10px;background:white;color:#294352;cursor:pointer}
select:focus-visible{outline:3px solid #4aa9bc}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.card{border:1px solid #e0e7ed;border-radius:8px;overflow:hidden;background:white}
.card[hidden]{display:none}.heading{padding:14px 16px}.number{font-size:12px;color:#627586}
.instruction{font-size:16px;line-height:1.5;margin:6px 0 0;min-height:48px}.media{display:grid;grid-template-columns:minmax(0,2.1fr) minmax(130px,1fr);background:#f8fafb}
.player{position:relative;background:#0b141d;align-self:center}video{display:block;width:100%;height:auto}.state{position:absolute;top:9px;left:9px;padding:5px 8px;border-radius:5px;background:#102633d9;color:white;font-size:11px;pointer-events:none}
.map{padding:8px;display:flex;flex-direction:column;justify-content:center}.map svg{width:100%;height:auto}.legend{text-align:center;font-size:10px;color:#627586}.legend b{color:#b17b22;font-weight:500}
.metrics{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;padding:12px 16px;font-size:12px;color:#627586}.outcome{color:#087e8b}.failure{color:#9f412f}
details{font-size:12px;color:#687d8a;padding:0 16px 14px;overflow-wrap:anywhere}summary{cursor:pointer}a{color:#087e8b}footer{color:#687d8a;font-size:12px;margin-top:24px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}@media(max-width:480px){body{padding:20px 12px}.instruction{font-size:15px}.media{grid-template-columns:minmax(0,2fr) minmax(100px,1fr)}}
"""

SCRIPT = """
const cards=[...document.querySelectorAll('.card')];
cards.forEach((card,i)=>{
 const video=card.querySelector('video'), path=card.querySelector('.progress'), marker=card.querySelector('.position'), state=card.querySelector('.state'), record=records[i];
 const update=()=>{
  const finished=video.ended&&video.currentTime>=record.duration-.05;
  const index=finished?record.route.length-1:Math.min(record.route.length-1,Math.floor(video.currentTime*record.fps));
  path.setAttribute('points',record.route.slice(0,index+1).map(p=>p.join(',')).join(' '));
  marker.setAttribute('cx',record.route[index][0]);marker.setAttribute('cy',record.route[index][1]);
  state.textContent=finished?record.outcome:(video.currentTime<.25?'START':`${video.currentTime.toFixed(1)} / ${record.duration.toFixed(1)} s`);
 };
 ['timeupdate','ended','seeked','loadedmetadata'].forEach(event=>video.addEventListener(event,update));update();
});
document.querySelector('#scene').onchange=event=>cards.forEach(card=>{
 card.hidden=event.target.value!==''&&card.dataset.scene!==event.target.value;
 if(card.hidden)card.querySelector('video').pause();
});
"""


def build_gallery(cases, output, title="Evaluation"):
    if not cases:
        raise ValueError("No evaluation cases to visualize")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("videos", "frames"):
        (output / name).mkdir(exist_ok=True)
    cards, records = [], []
    escape = html.escape
    for i, case in enumerate(cases):
        row = case["row"]
        number = f"{i + 1:02d}"
        video = f"videos/{number}.mp4"
        destination = output / video
        if not destination.exists() or not destination.samefile(case["source"]):
            destination.unlink(missing_ok=True)
            try:
                destination.hardlink_to(case["source"])
            except OSError:
                shutil.copyfile(case["source"], destination)
        for name in ("first", "last"):
            case[name].save(output / f"frames/{number}-{name}.jpg", quality=92)
        svg, route = route_svg(case)
        outcome = ("Success" if row["success"] else "Failure") + (
            " · model STOP" if row.get("stopped_via_model") else " · no model STOP")
        records.append({"route": route, "fps": case["fps"], "duration": case["duration"], "outcome": outcome})
        cards.append(f'''<article class="card" data-scene="{escape(row['scene_id'])}">
<div class="heading"><span class="number">Case {number}</span>
<p class="instruction">{escape(row['instruction'])}</p></div>
<div class="media"><div class="player"><video controls muted playsinline preload="none" poster="frames/{number}-first.jpg" src="{video}" aria-label="Case {number} rollout"></video><span class="state">START</span></div>
<div class="map">{svg}<div class="legend">Executed path · <b>target</b></div></div></div>
<div class="metrics"><span class="outcome {'failure' if not row['success'] else ''}">{outcome}</span><span>{row['path_length']:.2f} m · {case['duration']:.2f} s · {row['steps_taken']} actions</span></div>
<details><summary>Details &amp; files</summary><p>{escape(row['scene_id'])} / {escape(row['episode_id'])}<br>{escape(row.get('termination_reason', ''))}</p><a href="frames/{number}-first.jpg">Start</a> · <a href="frames/{number}-last.jpg">Last observation</a> · <a href="{video}">Video</a></details></article>''')
    scenes = sorted({case["row"]["scene_id"] for case in cases})
    filters = '<option value="">All scenes</option>' + "".join(
        f'<option value="{escape(scene)}">{escape(scene)}</option>' for scene in scenes)
    data = json.dumps(records).replace("<", "\\u003c")
    overviews = []
    for start in range(0, len(cases), 20):
        name = "overview.png" if len(cases) <= 20 else f"overview-{start // 20 + 1:03d}.png"
        overview(cases[start:start + 20], output / name, title, start)
        overviews.append(f'<a href="{name}">Cases {start + 1}–{min(start + 20, len(cases))}</a>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScaleNav · {escape(title)}</title><style>{STYLE}</style><body><header><div><h1>{escape(title)}</h1>
<p>{len(cases)} episodes · {len(scenes)} scenes</p></div><select id="scene" aria-label="Filter by scene">{filters}</select></header>
<main class="grid">{''.join(cards)}</main><footer>Overview: {' · '.join(overviews)}</footer>
<script>const records={data};{SCRIPT}</script></body></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--videos", type=Path, help="Defaults to the videos directory beside summary.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    manifest = json.loads(args.episodes.read_text(encoding="utf-8"))
    cases = read_cases(summary["episodes"], manifest["episodes"], args.videos or args.summary.parent / "videos")
    title = BENCHMARK_TITLES.get(summary.get("benchmark", manifest.get("benchmark")), "Evaluation")
    build_gallery(cases, args.output, title=title)
    print(f"Prepared {len(cases)} cases: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
