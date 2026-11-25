"""Phone-friendly layout, added to every page after it is built (like the GitHub link), so pages from
earlier runs can be updated without rebuilding them. Below 760 px wide:

* the header shrinks (and the heat maps help text); a long description is clipped to two lines (tap to read it all) with the page
  links kept visible underneath
* drop-downs take the full width; full-height map pages scroll instead, with the map at 68% of the
  screen and the side panel (map layers) underneath it in two columns
* map legends are smaller and can be tapped to fold them away (tap again to bring them back)
* cartograms, galleries, card grids and wide tables fit the screen
"""
from __future__ import annotations

import re
from pathlib import Path

VIEWPORT = '<meta name="viewport" content="width=device-width, initial-scale=1">'

BLOCK = r"""<!--mobile--><style>
@media (max-width:760px){
 header{padding:10px 14px} header h1{font-size:18px;margin-bottom:2px} header p{font-size:12.5px;line-height:1.4}
 header p.clip{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;cursor:pointer}
 header .hlinks{font-size:12.5px;margin-top:4px}
 main{padding:12px 14px}
 #bar{padding:8px 12px !important;gap:6px 12px !important;font-size:13px}
 #bar label:has(select){flex:1 1 100%;display:flex;flex-direction:column;gap:2px;min-width:0}
 #bar select{width:100%;min-width:0;max-width:none !important;font-size:16px;box-sizing:border-box}
 #bar>*{max-width:100%;min-width:0}
 #bar .grp{flex:1 1 100%;gap:6px 12px !important}
 #help{padding:6px 12px !important;font-size:12px !important}
 #help.clip{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;cursor:pointer;padding-bottom:0 !important;border-bottom:8px solid #fafaf7 !important}
 html:has(#map),html:has(#map) body{height:auto !important}
 #main{flex-direction:column !important;flex:none !important}
 #map{flex:none !important;height:68vh;min-height:300px}
 #side{width:auto !important;border-left:0 !important;border-top:1px solid #ddd;overflow:visible !important}
 #layers{columns:2;column-gap:10px;font-size:12px} #layers label{break-inside:avoid} #layers .sw{width:14px;margin:0 4px 0 2px}
 .leaflet-container .legend{font-size:11px !important;line-height:15px !important;max-width:170px !important;width:auto !important;
   max-height:42vh;overflow:auto;padding:6px 8px !important;cursor:pointer}
 .leaflet-container .legend::before{content:'tap to hide';display:block;float:right;margin-left:6px;color:#999;font-size:10px}
 .leaflet-container .legend.mini{max-height:17px;overflow:hidden}
 .leaflet-container .legend.mini::before{content:'tap for key'}
 #wrap{padding:10px 12px !important;gap:12px !important}
 #svg{min-width:0 !important;width:100%;flex:1 1 100% !important}
 #grid,.grid,.cards{grid-template-columns:1fr !important;padding-left:12px;padding-right:12px}
 main table{display:block;overflow-x:auto} #q{width:100% !important;box-sizing:border-box}
}
@media (min-width:761px) and (max-width:1000px){.cards{grid-template-columns:repeat(2,1fr) !important}}
</style><script>
(function () {
  if (!matchMedia('(max-width:760px)').matches) return;
  const p = document.querySelector('header p');
  if (p && p.textContent.length > 140) {
    // keep the page links visible, clip the long description to two lines (tap to expand)
    const links = [...p.querySelectorAll('a[href="index.html"], a.repo')];
    if (links.length) {
      const d = document.createElement('div'); d.className = 'hlinks';
      links.forEach((a, i) => { if (i) d.append(' | '); d.append(a); });
      p.after(d);
      while (p.lastChild && p.lastChild.nodeType === 3 && /^[\s|]*$/.test(p.lastChild.textContent)) p.lastChild.remove();
    }
    p.classList.add('clip'); p.title = 'tap to read more';
    p.addEventListener('click', () => p.classList.toggle('clip'));
  }
  const help = document.getElementById('help');   // long help text (heat maps): three lines, tap for the rest
  if (help) { help.classList.add('clip'); help.title = 'tap to read more'; help.addEventListener('click', () => help.classList.toggle('clip')); }
  // map legends fold away on tap
  document.addEventListener('click', e => {
    const lg = e.target.closest('.leaflet-container .legend');
    if (lg && !e.target.closest('a,input,select,button')) { lg.classList.toggle('mini'); e.stopPropagation(); }
  }, true);
  // the map's size changed with the layout: tell Leaflet
  setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
})();
</script><!--/mobile-->"""

MARK = re.compile(r"<!--mobile-->.*?<!--/mobile-->", re.S)


def add_mobile(out_dir: Path) -> int:
    """Add (or refresh) the phone layout on every page in out_dir; returns the number of pages changed."""
    n = 0
    for p in Path(out_dir).glob("*.html"):
        s = p.read_text(encoding="utf-8")
        new = MARK.sub("", s)
        if 'name="viewport"' not in new:
            new = new.replace('<meta charset="utf-8">', '<meta charset="utf-8">' + VIEWPORT, 1)
        new = new.replace("</body>", BLOCK + "</body>", 1) if "</body>" in new else new + BLOCK
        if new != s:
            p.write_text(new, encoding="utf-8")
            n += 1
    return n
