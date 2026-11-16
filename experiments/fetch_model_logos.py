"""Download the vendor logos for the Terminal-Bench 2.0 and 4.x leaderboard panels and convert them to raw RGBA
for R.

Source (read 2026-11-09): unpkg.com/@lobehub/icons-static-png@2.6.0/light/<name>.png, the npm package of
lobehub/lobe-icons (MIT). The logos are trademarks of their owners. R has no PNG reader installed here, so
each logo is shrunk to fit 96 x 96 and written as raw RGBA bytes (<name>.rgba) with its width and height
(<name>.dim); analysis/45b_fig_tbench.R and 46c_fig_tbench4.R read them with readBin.
DeepSeek, Tencent and Meta added 2026-11-13 for the Terminal-Bench 4.x figures.

Output: data/raw/external/model_logos/{png/<name>.png, <name>.rgba, <name>.dim}
Run from the project root: python3 -I experiments/fetch_model_logos.py (needs Pillow)
"""
import os
import urllib.request

from PIL import Image

NAMES = ["anthropic", "openai", "google-color", "xai", "zai", "alibaba-color", "moonshot",
         "minimax-color", "deepseek-color", "tencent-color", "meta-color"]
URL = "https://unpkg.com/@lobehub/icons-static-png@2.6.0/light/{}.png"
OUT = "data/raw/external/model_logos"

os.makedirs(os.path.join(OUT, "png"), exist_ok=True)
for name in NAMES:
    png = os.path.join(OUT, "png", name + ".png")
    req = urllib.request.Request(URL.format(name), headers={"User-Agent": "research-script"})
    with urllib.request.urlopen(req, timeout=60) as r, open(png, "wb") as f:
        f.write(r.read())
    im = Image.open(png).convert("RGBA")
    im.thumbnail((96, 96))
    with open(os.path.join(OUT, name + ".rgba"), "wb") as f:
        f.write(im.tobytes())
    with open(os.path.join(OUT, name + ".dim"), "w") as f:
        f.write(f"{im.width} {im.height}\n")
    print(name, im.size)
