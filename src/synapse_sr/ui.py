"""Live feedback for terminals and notebooks: stage spinner, tile and download bars, run summary.

Rich renders them when installed (it is a dependency, but everything degrades to plain stderr lines without it).
Set ``SYNAPSE_SR_QUIET=1`` to silence all of it.
"""

import os
import sys
import time

try:
    from rich.console import Console
    from rich.progress import (BarColumn, DownloadColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn, TransferSpeedColumn)
    from rich.table import Table
except ImportError:                                    # pragma: no cover
    Console = None

ACCENT = "#22d3ee"


def quiet():
    return os.environ.get("SYNAPSE_SR_QUIET", "0") not in ("", "0")


def in_notebook():
    return Console is not None and Console().is_jupyter


def interactive():
    """True in a terminal or a Jupyter / Colab / Kaggle notebook, False when piped, logged or under test."""
    if quiet():
        return False
    return in_notebook() or (hasattr(sys.stderr, "isatty") and sys.stderr.isatty())


def console():
    return Console(stderr=True, highlight=False) if Console is not None else None


def glyphs():
    """Typographic symbols, or ASCII stand-ins on consoles that cannot encode them (legacy Windows code pages)."""
    enc = (getattr(sys.stderr, "encoding", None) or "ascii").lower().replace("-", "")
    if enc.startswith("utf") or in_notebook():
        return {"ok": "✓", "x": "×", "dot": "·", "le": "≤", "tau": "τ"}
    return {"ok": "ok", "x": "x", "dot": "|", "le": "<=", "tau": " tau"}


class RunProgress:
    """Stage spinner plus a tile bar for one super_resolve call."""

    def __init__(self, title):
        self.title = title
        self.t0 = time.time()
        self.con = console()
        self.task = None
        if self.con is not None:
            self.bar = Progress(SpinnerColumn(style=ACCENT), TextColumn("[bold]{task.description}"),
                                BarColumn(complete_style=ACCENT, finished_style="green"), MofNCompleteColumn(),
                                TextColumn("[dim]{task.fields[unit]}"), TimeElapsedColumn(),
                                TextColumn("[dim]eta"), TimeRemainingColumn(), console=self.con, transient=True)

    def __enter__(self):
        if self.con is not None:
            self.bar.start()
            self.task = self.bar.add_task(self.title, total=None, unit="")
        return self

    def __exit__(self, *exc):
        if self.con is not None:
            self.bar.stop()
        return False

    def stage(self, text):
        if self.con is not None:
            self.bar.update(self.task, description=f"{self.title} {glyphs()['dot']} {text}", total=None, completed=0,
                            unit="")
        else:
            sys.stderr.write(f"synapse-sr: {text}\n")

    def tiles(self, done, total):
        if self.con is not None:
            self.bar.update(self.task, description=f"{self.title} {glyphs()['dot']} super-resolving", total=total,
                            completed=done, unit="tiles")
        else:
            el = time.time() - self.t0
            sys.stderr.write("\rsynapse-sr: tile %d/%d  %.0f s%s" % (done, total, el, "\n" if done >= total else ""))
            sys.stderr.flush()

    def done(self, result, seconds):
        line = summary_line(result, seconds)
        if self.con is not None:
            self.con.print(f"[green]{glyphs()['ok']}[/green] {line}")
        else:
            sys.stderr.write(f"synapse-sr: done - {line}\n")


def summary_line(r, seconds):
    c = r.consistency
    worst = max(c.values()) if c else float("nan")
    high = float((r.support[r.valid] == 2).mean()) if r.valid.any() else 0.0
    h, w = r.image.shape[-2:]
    g = glyphs()
    return (f"{r.metadata.get('model', 'model')}: {w}{g['x']}{h} px at {r.gsd:g} m in {seconds:.1f} s {g['dot']} "
            f"consistency {g['le']} {worst:.2f}{g['tau']} {g['dot']} {100 * high:.0f}% observation-determined")


def summary_table(r):
    """A Rich table describing a Result (used by Result.summary and the CLI)."""
    t = Table(title="synapse-sr result", title_style=f"bold {ACCENT}", show_header=False, box=None, pad_edge=False)
    t.add_column(style="dim", no_wrap=True)
    t.add_column()
    m = r.metadata
    h, w = r.image.shape[-2:]
    g = glyphs()
    t.add_row("model", str(m.get("model")))
    t.add_row("output", f"{w} {g['x']} {h} px {g['dot']} {r.gsd:g} m {g['dot']} bands {' '.join(m.get('bands', []))}")
    t.add_row("backend", f"{m.get('scan_backend')} {g['dot']} {m.get('precision')} {g['dot']} tile {m.get('tile')} "
                         f"halo {m.get('halo')}")
    t.add_row("consistency", "  ".join(f"{b} {v:.2f}{g['tau']}" for b, v in r.consistency.items()))
    if r.valid.any():
        s = r.support[r.valid]
        t.add_row("support", f"HIGH {100 * (s == 2).mean():.0f}%  MEDIUM {100 * (s == 1).mean():.0f}%  "
                             f"LOW {100 * (s == 0).mean():.0f}%")
    t.add_row("invalid input", f"{100 * m.get('invalid_input_fraction', 0):.1f}%")
    if m.get("seconds") is not None:
        t.add_row("time", f"{m['seconds']:.1f} s")
    return t


class DownloadBar:
    def __init__(self, name, total):
        self.con = console() if interactive() else None
        if self.con is not None:
            self.bar = Progress(SpinnerColumn(style=ACCENT), TextColumn("[bold]downloading {task.description}"),
                                BarColumn(complete_style=ACCENT), DownloadColumn(), TransferSpeedColumn(),
                                TimeRemainingColumn(), console=self.con, transient=True)
            self.bar.start()
            self.task = self.bar.add_task(name, total=total)

    def update(self, n):
        if self.con is not None:
            self.bar.advance(self.task, n)

    def close(self):
        if self.con is not None:
            self.bar.stop()
