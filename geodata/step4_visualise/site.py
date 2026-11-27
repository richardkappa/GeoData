"""Main-website look on every page, added after each build (like the phone layout) so pages from earlier runs can be
restyled without rebuilding them. Settings come from `global.site` in config/4_visualise.yaml:

* the site's black bar with its name (linking to the main site) and menu; on phones the menu folds into a button
* a second bar of map pages, the current one highlighted (scrolls sideways on phones)
* the site's fonts and colours on the page header, cards, links and buttons
* a footer with links and data credits (left off full-height map pages, where the space goes to the map)
* page titles in the form "Page - Site name", and the site's favicon
* ``?embed=1`` hides the bars and footer, for showing a map inside another page
"""
from __future__ import annotations

import html
import re
from pathlib import Path

MARK = re.compile(r"<!--site-->.*?<!--/site-->", re.S)
HEAD_MARK = re.compile(r"<!--sitehead-->.*?<!--/sitehead-->", re.S)
# the older "GeoData home | site | Code on GitHub" links in the page header, now in the bars instead
OLD_NAV = re.compile(r'\s*(?:<br>)?\s*<a href="index\.html">(?:All visualisations|GeoData home)</a>'
                     r'(?: \| <a class="(?:site|repo)"[^>]*>[^<]*</a>)*')

DEFAULTS = {
    "name": "Richard Kapustynskyj",
    "url": "https://richardkappa.co.uk/",
    "favicon": None,
    "colours": {"bar": "#101010", "accent": "#0bb4aa", "link": "#077d76", "background": "#f9fafd",
                "text": "#101010", "muted": "#6c6c77", "border": "#e4e6ee"},
    "fonts": {"body": "Inter", "headings": "Onest"},
    "menu": [],
    "pages": [],
    "footer": [],
    "credits": "",
}

CSS = """
:root{--rk-bar:%(bar)s;--rk-accent:%(accent)s;--rk-link:%(link)s;--rk-bg:%(background)s;--rk-text:%(text)s;
 --rk-muted:%(muted)s;--rk-border:%(border)s}
body{font-family:'%(body)s',system-ui,-apple-system,'Segoe UI',sans-serif !important;background:var(--rk-bg) !important;color:var(--rk-text)}
h1,h2,h3,.rk-name{font-family:'%(head)s','%(body)s',system-ui,sans-serif}
a{color:var(--rk-link)}
#rk-top{flex:none;background:var(--rk-bar);color:#fff;height:52px;display:flex;align-items:center;justify-content:space-between;
 padding:0 26px;box-sizing:border-box;position:relative;z-index:1200}
#rk-top .rk-name{color:#fff;text-decoration:none;font-weight:600;letter-spacing:.14em;text-transform:uppercase;font-size:15px;white-space:nowrap}
#rk-top nav{display:flex;gap:26px;font-size:14px}
#rk-top nav a{color:#fff;text-decoration:none;opacity:.85;padding:4px 0;border-bottom:2px solid transparent}
#rk-top nav a:hover,#rk-top nav a.on{opacity:1;border-bottom-color:var(--rk-accent)}
#rk-menu{display:none;background:none;border:1px solid #fff6;color:#fff;border-radius:6px;font-size:15px;padding:4px 10px;cursor:pointer}
#rk-sub{flex:none;background:#fff;border-bottom:1px solid var(--rk-border);display:flex;gap:4px;padding:0 20px;overflow-x:auto;
 white-space:nowrap;font-size:13.5px;scrollbar-width:thin;position:relative;z-index:1100}
#rk-sub a{color:var(--rk-muted);text-decoration:none;padding:9px 10px 8px;border-bottom:2px solid transparent}
#rk-sub a:hover{color:var(--rk-text)}
#rk-sub a.on{color:var(--rk-text);font-weight:600;border-bottom-color:var(--rk-accent)}
body>header{background:var(--rk-bg) !important;color:var(--rk-text) !important;padding:14px 26px 12px !important;border-bottom:1px solid var(--rk-border)}
body>header h1{font-weight:600;letter-spacing:-.01em}
body>header p{color:var(--rk-muted);opacity:1 !important}
body>header a{color:var(--rk-link)}
main h2{font-weight:600}
.card,.dd{background:#fff !important;border:1px solid var(--rk-border) !important;border-radius:10px !important;box-shadow:none !important;
 transition:border-color .15s,transform .15s}
.card:hover,.dd:hover{border-color:var(--rk-accent) !important;transform:translateY(-1px)}
.card b,.dd b{color:var(--rk-text) !important}
.dd{border-top:3px solid var(--rk-accent) !important}
#rk-foot{flex:none;background:var(--rk-bar);color:#bbb;font-size:13px;padding:22px 26px;margin-top:30px;display:flex;flex-wrap:wrap;
 gap:10px 28px;align-items:center}
#rk-foot a{color:#fff;text-decoration:none} #rk-foot a:hover{color:var(--rk-accent)}
#rk-foot .rk-credits{flex:1 1 100%%;color:#999;font-size:12px;line-height:1.5}
html:has(#map) #rk-foot{display:none}
html.rk-embed #rk-top,html.rk-embed #rk-sub,html.rk-embed #rk-foot{display:none !important}
@media (max-width:760px){
 #rk-top{height:46px;padding:0 14px} #rk-top .rk-name{font-size:13px;letter-spacing:.1em}
 #rk-menu{display:block}
 #rk-top nav{display:none;position:absolute;top:46px;right:0;left:0;background:var(--rk-bar);flex-direction:column;gap:0;padding:4px 14px 10px}
 #rk-top nav a{padding:10px 0;border-bottom:1px solid #ffffff1a}
 #rk-top.open nav{display:flex}
 #rk-sub{padding:0 8px;font-size:13px}
 body>header{padding:10px 14px !important}
 #rk-foot{padding:16px 14px}
}
"""

SCRIPT = """<script>
(function () {
  if (/[?&]embed=1\\b/.test(location.search)) document.documentElement.classList.add('rk-embed');
  const b = document.getElementById('rk-menu'), t = document.getElementById('rk-top');
  if (b) b.addEventListener('click', () => { const o = t.classList.toggle('open'); b.setAttribute('aria-expanded', o); });
  const on = document.querySelector('#rk-sub a.on');        // keep the current page in view on narrow screens
  if (on && on.parentNode.scrollWidth > on.parentNode.clientWidth) on.parentNode.scrollLeft = on.offsetLeft - 40;
})();
</script>"""


def _cfg(site: dict | None) -> dict:
    c = {**DEFAULTS, **(site or {})}
    c["colours"] = {**DEFAULTS["colours"], **((site or {}).get("colours") or {})}
    c["fonts"] = {**DEFAULTS["fonts"], **((site or {}).get("fonts") or {})}
    return c


def _link(item: dict, current: str, repo: str | None) -> str:
    url = item.get("url") or ""
    if url == "repo":
        url = repo or ""
    if not url:
        return ""
    on = ' class="on" aria-current="page"' if url == current else ""
    return f'<a href="{html.escape(url)}"{on}>{html.escape(item["label"])}</a>'


def _head(c: dict) -> str:
    f = c["fonts"]
    fams = "&".join(f"family={html.escape(x.replace(' ', '+'))}:wght@400;500;600;700" for x in dict.fromkeys([f["body"], f["headings"]]))
    out = ['<!--sitehead-->',
           '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
           f'<link rel="stylesheet" href="https://fonts.googleapis.com/css2?{fams}&display=swap">']
    if c.get("favicon"):
        out.append(f'<link rel="icon" href="{html.escape(c["favicon"])}">')
    out.append('<!--/sitehead-->')
    return "".join(out)


def _title(s: str, name: str) -> str:
    """'<h1> - Site name' as the page title (the h1 is the page's own name)."""
    m = re.search(r"<header[^>]*>\s*<h1[^>]*>(.*?)</h1>", s, re.S)
    if not m:
        return s
    page = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    return re.sub(r"<title>.*?</title>", lambda _: f"<title>{page} – {html.escape(name)}</title>", s, count=1, flags=re.S)


def add_site(out_dir: Path, site: dict | None, repo: str | None = None) -> int:
    """Add (or refresh) the main-website bars, styles and footer on every page in out_dir; returns pages changed."""
    c = _cfg(site)
    col = c["colours"]
    css = CSS % {**col, "body": c["fonts"]["body"], "head": c["fonts"]["headings"]}
    head = _head(c)
    n = 0
    for p in Path(out_dir).glob("*.html"):
        s = p.read_text(encoding="utf-8")
        if "<header" not in s:          # redirect stubs (e.g. minimap_explorer.html) stay as they are
            continue
        new = HEAD_MARK.sub("", MARK.sub("", s))
        new = OLD_NAV.sub("", new)
        new = re.sub(r"<p>\s*</p>(?=\s*</header>)", "", new)          # header line that only held the old links
        new = new.replace("</head>", head + "</head>", 1)
        new = _title(new, c["name"])
        cur = p.name
        menu = "".join(_link(i, "index.html", repo) for i in c["menu"])   # every page is in the Maps section
        pages = "".join(_link(i, cur, repo) for i in c["pages"])
        top = (f'<!--site--><style>{css}</style>'
               f'<div id="rk-top"><a class="rk-name" href="{html.escape(c["url"])}">{html.escape(c["name"])}</a>'
               f'<button id="rk-menu" aria-label="Menu" aria-expanded="false">&#9776;</button><nav>{menu}</nav></div>'
               + (f'<nav id="rk-sub" aria-label="Map pages">{pages}</nav>' if pages else "") + '<!--/site-->')
        foot_links = "".join(_link(i, "", repo) for i in c["footer"])
        foot = (f'<!--site--><footer id="rk-foot">{foot_links}'
                + (f'<div class="rk-credits">{c["credits"]}</div>' if c["credits"] else "")
                + f'</footer>{SCRIPT}<!--/site-->')
        new = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + top, new, count=1)
        # footer before the phone-layout block (or </body>) so it sits at the end of the page
        if "<!--mobile-->" in new:
            new = new.replace("<!--mobile-->", foot + "<!--mobile-->", 1)
        else:
            new = new.replace("</body>", foot + "</body>", 1)
        if new != s:
            p.write_text(new, encoding="utf-8")
            n += 1
    return n
