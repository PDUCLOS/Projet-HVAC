# -*- coding: utf-8 -*-
"""
Interactive CLI Menu — HVAC Market Analysis Pipeline.
======================================================

Professional interactive menu for managing the full data pipeline:
    1. Data Collection (local) with date verification
    2. pCloud Data Management (sync, upload, status)
    3. Data Processing (clean, merge, features, outliers)
    4. ML Training & Notebooks

Usage:
    python -m src.cli_menu
    python -m src.pipeline menu
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from config.settings import config

console = Console()
logger = logging.getLogger("cli_menu")


# =====================================================================
# Utilities
# =====================================================================

def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def pause(message: str = "Press Enter to continue...") -> None:
    """Pause and wait for user input."""
    try:
        input(f"\n  {message}")
    except (EOFError, KeyboardInterrupt):
        pass


def get_choice(prompt: str = "Select", valid: Optional[List[str]] = None) -> str:
    """Get user input with validation.

    Args:
        prompt: Prompt text to display.
        valid: List of valid choices. If None, accept any input.

    Returns:
        User's choice as a stripped lowercase string.
    """
    try:
        choice = input(f"\n  {prompt} > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "0"

    if valid and choice not in valid:
        console.print(f"  [red]Invalid choice: '{choice}'[/red]")
        return ""
    return choice


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _get_file_info(filepath: Path) -> Dict[str, Any]:
    """Get file metadata (size, modified date, row count).

    Args:
        filepath: Path to the file.

    Returns:
        Dictionary with file info.
    """
    info = {
        "exists": filepath.exists(),
        "size": 0,
        "size_str": "N/A",
        "modified": "N/A",
        "rows": 0,
    }
    if filepath.exists():
        stat = filepath.stat()
        info["size"] = stat.st_size
        info["size_str"] = _format_size(stat.st_size)
        info["modified"] = datetime.fromtimestamp(
            stat.st_mtime
        ).strftime("%Y-%m-%d %H:%M")
        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                info["rows"] = sum(1 for _ in f) - 1  # exclude header
        except OSError:
            info["rows"] = 0
    return info


def _detect_date_range(filepath: Path, date_col: str) -> Dict[str, str]:
    """Detect the date range in a CSV file.

    Args:
        filepath: Path to the CSV file.
        date_col: Name of the date column.

    Returns:
        Dictionary with 'min' and 'max' date strings.
    """
    result = {"min": "N/A", "max": "N/A"}
    if not filepath.exists():
        return result

    try:
        import pandas as pd
        # Read only the date column for performance
        df = pd.read_csv(
            filepath,
            usecols=[date_col],
            dtype={date_col: str},
            low_memory=False,
        )
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not dates.empty:
            result["min"] = dates.min().strftime("%Y-%m-%d")
            result["max"] = dates.max().strftime("%Y-%m-%d")
    except Exception:
        pass

    return result


# =====================================================================
# Header
# =====================================================================

def show_header() -> None:
    """Display the main application header."""
    header = Text()
    header.append("HVAC Market Analysis", style="bold cyan")
    header.append(" | ", style="dim")
    header.append("Interactive Pipeline Manager", style="italic")

    console.print()
    console.print(Panel(
        header,
        border_style="cyan",
        box=box.DOUBLE,
        padding=(0, 2),
    ))


# =====================================================================
# MENU 1 — Data Collection (Local)
# =====================================================================

# Source definitions with metadata
COLLECTION_SOURCES = {
    "weather": {
        "label": "Weather (Open-Meteo)",
        "description": "Daily weather data for 96 reference cities",
        "file": "weather/weather_france.csv",
        "date_col": "date",
        "estimate": "~5 min, ~50 MB",
    },
    "insee": {
        "label": "INSEE (Economic Indicators)",
        "description": "Monthly macroeconomic indicators (national)",
        "file": "insee/indicateurs_economiques.csv",
        "date_col": "date",
        "estimate": "~1 min, ~3 KB",
    },
    "eurostat": {
        "label": "Eurostat (HVAC Production Index)",
        "description": "Monthly HVAC production index (IPI C2825)",
        "file": "eurostat/ipi_hvac_france.csv",
        "date_col": "date",
        "estimate": "~1 min, ~7 KB",
    },
    "sitadel": {
        "label": "SITADEL (Building Permits)",
        "description": "Monthly building permits by department",
        "file": "sitadel/permis_construire_france.csv",
        "date_col": "date_autorisation",
        "estimate": "~5 min, ~250 KB",
    },
    "dpe": {
        "label": "DPE (Energy Diagnostics — ADEME)",
        "description": "Individual DPE records (proxy for HVAC installs)",
        "file": "dpe/dpe_france_all.csv",
        "date_col": "date_etablissement_dpe",
        "estimate": "~10-30 min, ~1-2 GB",
    },
}


def _build_collection_table() -> Table:
    """Build a rich table showing all data sources with date info.

    Returns:
        Rich Table with source status, dates, and file info.
    """
    table = Table(
        title="Data Sources — Local Status",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold white",
        border_style="blue",
        padding=(0, 1),
    )

    table.add_column("#", style="bold cyan", justify="center", width=3)
    table.add_column("Source", style="bold white", min_width=15)
    table.add_column("Status", justify="center", min_width=9)
    table.add_column("Rows", justify="right", min_width=10)
    table.add_column("Size", justify="right", min_width=8)
    table.add_column("Date Range", min_width=25)
    table.add_column("Last Modified", min_width=16)

    for i, (key, src) in enumerate(COLLECTION_SOURCES.items(), 1):
        filepath = config.raw_data_dir / src["file"]
        info = _get_file_info(filepath)

        if info["exists"]:
            status = "[bold green]READY[/bold green]"
            rows_str = f"{info['rows']:,}"
            size_str = info["size_str"]
            modified_str = info["modified"]

            # Detect date range
            dates = _detect_date_range(filepath, src["date_col"])
            if dates["min"] != "N/A":
                date_range = f"{dates['min']}  ->  {dates['max']}"
            else:
                date_range = "[dim]Could not parse[/dim]"
        else:
            status = "[bold red]MISSING[/bold red]"
            rows_str = "-"
            size_str = "-"
            modified_str = "-"
            date_range = "[dim]No data[/dim]"

        table.add_row(
            str(i), src["label"], status,
            rows_str, size_str, date_range, modified_str,
        )

    return table


def menu_collection() -> None:
    """Display the Data Collection menu with date verification."""
    while True:
        clear_screen()
        show_header()

        console.print()
        console.print(
            Panel(
                "[bold]1 - DATA COLLECTION (Local)[/bold]\n"
                "[dim]Collect data from external APIs and store locally[/dim]",
                border_style="green",
                box=box.HEAVY,
            )
        )

        # Show the sources table with dates
        console.print()
        with console.status("[cyan]Scanning local data files...[/cyan]"):
            table = _build_collection_table()
        console.print(table)

        # Menu options
        console.print()
        console.print("  [bold cyan]Options:[/bold cyan]")
        console.print("    [bold]a[/bold]  Collect ALL sources (full pipeline)")
        console.print("    [bold]1-5[/bold]  Collect a specific source:")

        for i, (key, src) in enumerate(COLLECTION_SOURCES.items(), 1):
            console.print(f"         [bold]{i}[/bold]  {src['label']}  [dim]({src['estimate']})[/dim]")

        console.print("    [bold]v[/bold]  Verify data dates (detailed report)")
        console.print("    [bold]0[/bold]  Back to main menu")

        choice = get_choice("Select")

        if choice == "0":
            return

        elif choice == "a":
            _run_collect_all()

        elif choice == "v":
            _show_date_verification()

        elif choice in ("1", "2", "3", "4", "5"):
            source_keys = list(COLLECTION_SOURCES.keys())
            idx = int(choice) - 1
            _run_collect_single(source_keys[idx])

        elif choice == "":
            continue


def _run_collect_all() -> None:
    """Run collection for all sources."""
    console.print()
    console.print(
        Panel(
            "[bold yellow]Starting FULL data collection...[/bold yellow]\n"
            "This will collect from all 5 sources.\n"
            "Estimated time: 20-40 minutes (DPE is the longest).",
            border_style="yellow",
        )
    )

    confirm = get_choice("Proceed? [y/N]")
    if confirm not in ("y", "yes"):
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    from src.pipeline import run_collect, setup_logging
    setup_logging("INFO")

    console.print()
    console.rule("[bold green]Collection Started[/bold green]")
    console.print()

    try:
        run_collect()
        console.print()
        console.rule("[bold green]Collection Complete[/bold green]")
    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")

    pause()


def _run_collect_single(source: str) -> None:
    """Run collection for a single source.

    Args:
        source: Source name (weather, insee, eurostat, sitadel, dpe).
    """
    src_info = COLLECTION_SOURCES[source]
    console.print()
    console.print(
        Panel(
            f"[bold yellow]Collecting: {src_info['label']}[/bold yellow]\n"
            f"{src_info['description']}\n"
            f"Estimated: {src_info['estimate']}",
            border_style="yellow",
        )
    )

    confirm = get_choice("Proceed? [y/N]")
    if confirm not in ("y", "yes"):
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    from src.pipeline import run_collect, setup_logging
    setup_logging("INFO")

    console.print()
    console.rule(f"[bold green]Collecting {source.upper()}[/bold green]")
    console.print()

    try:
        run_collect(sources=[source])
        console.print()
        console.rule("[bold green]Collection Complete[/bold green]")
    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")

    pause()


def _show_date_verification() -> None:
    """Show detailed date verification report for all sources."""
    clear_screen()
    show_header()

    console.print()
    console.print(
        Panel(
            "[bold]DATA DATE VERIFICATION REPORT[/bold]\n"
            f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="magenta",
            box=box.DOUBLE,
        )
    )

    for key, src in COLLECTION_SOURCES.items():
        filepath = config.raw_data_dir / src["file"]
        info = _get_file_info(filepath)

        console.print()

        if not info["exists"]:
            console.print(
                Panel(
                    f"[bold red]NOT COLLECTED[/bold red]\n"
                    f"File: {src['file']}\n"
                    f"Run collection to download this source.",
                    title=f"[bold]{src['label']}[/bold]",
                    border_style="red",
                )
            )
            continue

        dates = _detect_date_range(filepath, src["date_col"])

        # Build detail info
        detail_lines = []
        detail_lines.append(f"File:          {src['file']}")
        detail_lines.append(f"Rows:          {info['rows']:,}")
        detail_lines.append(f"Size:          {info['size_str']}")
        detail_lines.append(f"Last Modified: {info['modified']}")
        detail_lines.append("")
        detail_lines.append(f"Date Column:   {src['date_col']}")

        if dates["min"] != "N/A":
            detail_lines.append(f"Date Range:    {dates['min']}  ->  {dates['max']}")

            # Calculate coverage
            try:
                from datetime import datetime as dt
                d_min = dt.strptime(dates["min"], "%Y-%m-%d")
                d_max = dt.strptime(dates["max"], "%Y-%m-%d")
                span_days = (d_max - d_min).days
                span_months = span_days / 30.44
                detail_lines.append(
                    f"Coverage:      {span_days:,} days (~{span_months:.0f} months)"
                )

                # Freshness check
                days_old = (datetime.now() - d_max).days
                if days_old <= 30:
                    freshness = "[bold green]UP TO DATE[/bold green]"
                elif days_old <= 90:
                    freshness = f"[bold yellow]{days_old} days old[/bold yellow]"
                else:
                    freshness = f"[bold red]{days_old} days old — STALE[/bold red]"
                detail_lines.append(f"Freshness:     {freshness}")
            except (ValueError, TypeError):
                pass
        else:
            detail_lines.append("Date Range:    [dim]Could not parse dates[/dim]")

        content = "\n".join(detail_lines)
        status_color = "green" if dates["min"] != "N/A" else "yellow"

        console.print(
            Panel(
                content,
                title=f"[bold]{src['label']}[/bold]",
                border_style=status_color,
            )
        )

    pause()


# =====================================================================
# MENU 2 — pCloud Data Management
# =====================================================================

def menu_pcloud() -> None:
    """Display the pCloud Data Management menu."""
    while True:
        clear_screen()
        show_header()

        console.print()
        console.print(
            Panel(
                "[bold]2 - PCLOUD DATA MANAGEMENT[/bold]\n"
                "[dim]Sync, upload, and monitor data on pCloud[/dim]",
                border_style="blue",
                box=box.HEAVY,
            )
        )

        # Show current sync status
        _show_pcloud_status_summary()

        console.print()
        console.print("  [bold cyan]Options:[/bold cyan]")
        console.print("    [bold]1[/bold]  Sync from pCloud (download new/modified files)")
        console.print("    [bold]2[/bold]  Upload local data to pCloud")
        console.print("    [bold]3[/bold]  Check for updates (compare local vs remote)")
        console.print("    [bold]4[/bold]  View detailed sync status")
        console.print("    [bold]5[/bold]  Force re-download ALL files")
        console.print("    [bold]6[/bold]  List remote files on pCloud")
        console.print("    [bold]0[/bold]  Back to main menu")

        choice = get_choice("Select", ["0", "1", "2", "3", "4", "5", "6", ""])

        if choice == "0":
            return
        elif choice == "1":
            _run_pcloud_sync()
        elif choice == "2":
            _run_pcloud_upload()
        elif choice == "3":
            _run_pcloud_check_updates()
        elif choice == "4":
            _show_pcloud_detailed_status()
        elif choice == "5":
            _run_pcloud_force_sync()
        elif choice == "6":
            _run_pcloud_list_remote()


def _show_pcloud_status_summary() -> None:
    """Show a quick summary of pCloud sync state."""
    import json

    state_file = Path("data") / ".pcloud_sync_state.json"

    if not state_file.exists():
        console.print()
        console.print(
            "  [dim]No sync history found. Run a sync to get started.[/dim]"
        )
        return

    try:
        state = json.loads(state_file.read_text())
    except (json.JSONDecodeError, IOError):
        console.print("  [yellow]Sync state file corrupted.[/yellow]")
        return

    if not state:
        console.print("  [dim]Sync state is empty.[/dim]")
        return

    table = Table(
        title="Last Sync Status",
        box=box.SIMPLE_HEAVY,
        border_style="blue",
        show_lines=False,
    )
    table.add_column("File", style="white", min_width=30)
    table.add_column("Last Sync", style="cyan", min_width=20)

    for filename, info in state.items():
        last_sync = info.get("last_sync", "never")
        if last_sync != "never":
            try:
                dt = datetime.fromisoformat(last_sync)
                last_sync = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                pass
        table.add_row(filename, last_sync)

    console.print()
    console.print(table)


def _run_pcloud_sync() -> None:
    """Run pCloud sync (download new/modified files)."""
    console.print()
    console.print(
        Panel(
            "[bold yellow]Syncing from pCloud...[/bold yellow]\n"
            "Will download new or modified files and trigger the import pipeline.",
            border_style="yellow",
        )
    )

    confirm = get_choice("Proceed? [y/N]")
    if confirm not in ("y", "yes"):
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    from src.pipeline import run_sync_pcloud, setup_logging
    setup_logging("INFO")

    try:
        run_sync_pcloud()
        console.print("\n  [bold green]Sync complete.[/bold green]")
    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")

    pause()


def _run_pcloud_upload() -> None:
    """Upload local data to pCloud."""
    console.print()

    # Check for access token
    token = os.getenv("PCLOUD_ACCESS_TOKEN", "")
    if not token:
        console.print(
            Panel(
                "[bold red]PCLOUD_ACCESS_TOKEN not found[/bold red]\n\n"
                "To upload to pCloud, you need an access token.\n"
                "Set it in your .env file:\n"
                "  PCLOUD_ACCESS_TOKEN=your_token_here",
                border_style="red",
            )
        )
        pause()
        return

    console.print(
        Panel(
            "[bold yellow]Uploading local data to pCloud...[/bold yellow]\n"
            "Will upload all collected CSV files, features datasets,\n"
            "and the database to your pCloud folder.",
            border_style="yellow",
        )
    )

    confirm = get_choice("Proceed? [y/N]")
    if confirm not in ("y", "yes"):
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    from src.pipeline import run_upload_pcloud, setup_logging
    setup_logging("INFO")

    try:
        run_upload_pcloud()
        console.print("\n  [bold green]Upload complete.[/bold green]")
    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")

    pause()


def _run_pcloud_check_updates() -> None:
    """Check for updates on pCloud without downloading."""
    console.print()
    console.print("  [cyan]Checking for updates on pCloud...[/cyan]")

    try:
        from src.collectors.pcloud_sync import PCloudSync
        sync = PCloudSync(config)
        updates = sync.check_for_updates()

        console.print()
        if not updates:
            console.print(
                Panel(
                    "[bold green]Everything is up to date![/bold green]\n"
                    "No new or modified files on pCloud.",
                    border_style="green",
                )
            )
        else:
            table = Table(
                title=f"{len(updates)} Update(s) Available",
                box=box.ROUNDED,
                border_style="yellow",
            )
            table.add_column("File", style="white")
            table.add_column("Reason", style="yellow")
            table.add_column("Size", justify="right")

            for f in updates:
                table.add_row(
                    f["name"],
                    f.get("update_reason", "unknown"),
                    _format_size(f.get("size", 0)),
                )

            console.print(table)
            console.print(
                "\n  [dim]Use option 1 to sync these files.[/dim]"
            )
    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")

    pause()


def _show_pcloud_detailed_status() -> None:
    """Show detailed pCloud sync status."""
    import json

    clear_screen()
    show_header()

    console.print()
    console.print(
        Panel(
            "[bold]PCLOUD SYNC — DETAILED STATUS[/bold]",
            border_style="blue",
            box=box.DOUBLE,
        )
    )

    # Environment variables check
    env_table = Table(
        title="Configuration",
        box=box.SIMPLE,
        border_style="blue",
    )
    env_table.add_column("Variable", style="white")
    env_table.add_column("Status")

    token = os.getenv("PCLOUD_ACCESS_TOKEN", "")
    code = os.getenv("PCLOUD_PUBLIC_CODE", "")

    env_table.add_row(
        "PCLOUD_ACCESS_TOKEN",
        "[green]Set[/green]" if token else "[red]Not set (upload disabled)[/red]",
    )
    env_table.add_row(
        "PCLOUD_PUBLIC_CODE",
        "[green]Set[/green]" if code else "[yellow]Not set (using default)[/yellow]",
    )

    console.print(env_table)

    # Sync state
    state_file = Path("data") / ".pcloud_sync_state.json"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            if state:
                sync_table = Table(
                    title="Tracked Files",
                    box=box.ROUNDED,
                    border_style="cyan",
                    show_lines=True,
                )
                sync_table.add_column("File", style="white", min_width=30)
                sync_table.add_column("Hash", style="dim", min_width=10)
                sync_table.add_column("Size", justify="right")
                sync_table.add_column("Last Sync", style="cyan")

                for name, info in state.items():
                    sync_table.add_row(
                        name,
                        str(info.get("hash", "?"))[:12],
                        _format_size(info.get("size", 0)),
                        info.get("last_sync", "never")[:16],
                    )
                console.print()
                console.print(sync_table)
        except (json.JSONDecodeError, IOError):
            console.print("  [yellow]Sync state file corrupted.[/yellow]")
    else:
        console.print("\n  [dim]No sync history (state file not found).[/dim]")

    # Local file comparison
    console.print()
    local_table = Table(
        title="Local Data Files",
        box=box.ROUNDED,
        border_style="green",
    )
    local_table.add_column("Source", style="white")
    local_table.add_column("File", style="dim")
    local_table.add_column("Size", justify="right")
    local_table.add_column("Modified")

    for key, src in COLLECTION_SOURCES.items():
        filepath = config.raw_data_dir / src["file"]
        info = _get_file_info(filepath)
        local_table.add_row(
            key.upper(),
            src["file"],
            info["size_str"] if info["exists"] else "[red]MISSING[/red]",
            info["modified"] if info["exists"] else "-",
        )

    console.print(local_table)

    pause()


def _run_pcloud_force_sync() -> None:
    """Force re-download all files from pCloud."""
    console.print()
    console.print(
        Panel(
            "[bold red]FORCE RE-DOWNLOAD ALL FILES[/bold red]\n\n"
            "This will re-download ALL files from pCloud,\n"
            "even if they haven't changed.\n"
            "Existing local files will be overwritten.",
            border_style="red",
        )
    )

    confirm = get_choice("Are you sure? Type 'yes' to confirm")
    if confirm != "yes":
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    from src.pipeline import run_sync_pcloud, setup_logging
    setup_logging("INFO")

    try:
        run_sync_pcloud(force=True)
        console.print("\n  [bold green]Force sync complete.[/bold green]")
    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")

    pause()


def _run_pcloud_list_remote() -> None:
    """List files available on the pCloud public folder."""
    console.print()
    console.print("  [cyan]Listing remote pCloud files...[/cyan]")

    try:
        from src.collectors.pcloud_sync import PCloudSync
        sync = PCloudSync(config)
        files = sync.list_public_folder()

        if not files:
            console.print("\n  [yellow]No files found (or connection error).[/yellow]")
            pause()
            return

        table = Table(
            title=f"pCloud Remote Files ({len(files)} files)",
            box=box.ROUNDED,
            border_style="cyan",
        )
        table.add_column("#", style="dim", justify="center", width=3)
        table.add_column("File", style="white", min_width=30)
        table.add_column("Size", justify="right")
        table.add_column("Modified", style="dim")
        table.add_column("Folder", style="cyan")

        for i, f in enumerate(files, 1):
            table.add_row(
                str(i),
                f["name"],
                _format_size(f.get("size", 0)),
                f.get("modified", "")[:16],
                f.get("folder", "-"),
            )

        console.print()
        console.print(table)

    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")

    pause()


# =====================================================================
# MENU 3 — Data Processing
# =====================================================================

def menu_processing() -> None:
    """Display the Data Processing menu."""
    while True:
        clear_screen()
        show_header()

        console.print()
        console.print(
            Panel(
                "[bold]3 - DATA PROCESSING[/bold]\n"
                "[dim]Clean, merge, feature engineering, and outlier detection[/dim]",
                border_style="yellow",
                box=box.HEAVY,
            )
        )

        # Show current processing status
        _show_processing_status()

        console.print()
        console.print("  [bold cyan]Options:[/bold cyan]")
        console.print("    [bold]1[/bold]  Run FULL pipeline  [dim](clean -> merge -> features -> outliers)[/dim]")
        console.print("    [bold]2[/bold]  Run full pipeline INTERACTIVE  [dim](preview before cleaning)[/dim]")
        console.print("    [bold]3[/bold]  Clean data only")
        console.print("    [bold]4[/bold]  Merge datasets only")
        console.print("    [bold]5[/bold]  Feature engineering only")
        console.print("    [bold]6[/bold]  Outlier detection only")
        console.print("    [bold]7[/bold]  Initialize database  [dim](create/reset tables)[/dim]")
        console.print("    [bold]8[/bold]  Import CSVs into database")
        console.print("    [bold]9[/bold]  Import CSVs INTERACTIVE  [dim](choose sources)[/dim]")
        console.print("    [bold]0[/bold]  Back to main menu")

        choice = get_choice(
            "Select",
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ""],
        )

        if choice == "0":
            return
        elif choice == "1":
            _run_processing_stage("process")
        elif choice == "2":
            _run_processing_stage("process_interactive")
        elif choice == "3":
            _run_processing_stage("clean")
        elif choice == "4":
            _run_processing_stage("merge")
        elif choice == "5":
            _run_processing_stage("features")
        elif choice == "6":
            _run_processing_stage("outliers")
        elif choice == "7":
            _run_processing_stage("init_db")
        elif choice == "8":
            _run_processing_stage("import_data")
        elif choice == "9":
            _run_processing_stage("import_interactive")


def _show_processing_status() -> None:
    """Show the current state of processed data files."""
    table = Table(
        title="Processing Status",
        box=box.SIMPLE_HEAVY,
        border_style="yellow",
    )
    table.add_column("Stage", style="white", min_width=20)
    table.add_column("Output File", style="dim", min_width=30)
    table.add_column("Status", justify="center")
    table.add_column("Rows", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="dim")

    stages = [
        ("Database", Path("data/hvac_market.db")),
        ("Cleaned (weather)", config.processed_data_dir / "weather_cleaned.csv"),
        ("Cleaned (DPE)", config.processed_data_dir / "dpe_cleaned.csv"),
        ("ML Dataset", config.features_data_dir / "hvac_ml_dataset.csv"),
        ("Features Dataset", config.features_data_dir / "hvac_features_dataset.csv"),
    ]

    for name, path in stages:
        info = _get_file_info(path)
        if info["exists"]:
            status = "[green]READY[/green]"
            rows = f"{info['rows']:,}" if info["rows"] > 0 else "[dim]binary[/dim]"
        else:
            status = "[red]MISSING[/red]"
            rows = "-"

        table.add_row(
            name,
            str(path),
            status,
            rows,
            info["size_str"] if info["exists"] else "-",
            info["modified"] if info["exists"] else "-",
        )

    console.print()
    console.print(table)


def _run_processing_stage(stage: str) -> None:
    """Run a specific processing stage.

    Args:
        stage: Stage name to run.
    """
    from src.pipeline import setup_logging
    setup_logging("INFO")

    stage_labels = {
        "process": "Full Pipeline (clean -> merge -> features -> outliers)",
        "process_interactive": "Full Pipeline INTERACTIVE",
        "clean": "Data Cleaning",
        "merge": "Dataset Merging",
        "features": "Feature Engineering",
        "outliers": "Outlier Detection",
        "init_db": "Database Initialization",
        "import_data": "CSV Import",
        "import_interactive": "CSV Import (Interactive)",
    }

    console.print()
    console.print(
        Panel(
            f"[bold yellow]Running: {stage_labels.get(stage, stage)}[/bold yellow]",
            border_style="yellow",
        )
    )

    confirm = get_choice("Proceed? [y/N]")
    if confirm not in ("y", "yes"):
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    console.print()
    console.rule(f"[bold green]{stage_labels.get(stage, stage)}[/bold green]")
    console.print()

    try:
        if stage == "process":
            from src.pipeline import run_process
            run_process(interactive=False)
        elif stage == "process_interactive":
            from src.pipeline import run_process
            run_process(interactive=True)
        elif stage == "clean":
            from src.pipeline import run_clean
            run_clean()
        elif stage == "merge":
            from src.pipeline import run_merge
            run_merge()
        elif stage == "features":
            from src.pipeline import run_features
            run_features()
        elif stage == "outliers":
            from src.pipeline import run_outliers
            run_outliers()
        elif stage == "init_db":
            from src.pipeline import run_init_db
            run_init_db()
        elif stage == "import_data":
            from src.pipeline import run_import_data
            run_import_data()
        elif stage == "import_interactive":
            from src.pipeline import run_import_data
            run_import_data(interactive=True)

        console.print()
        console.rule("[bold green]Complete[/bold green]")

    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")
        import traceback
        console.print(f"  [dim]{traceback.format_exc()}[/dim]")

    pause()


# =====================================================================
# MENU 4 — ML Training & Notebooks
# =====================================================================

NOTEBOOKS = {
    "1": {
        "file": "01_data_exploration.ipynb",
        "label": "Data Exploration (EDA)",
        "description": "Data overview, distributions, missing values, department summaries",
    },
    "2": {
        "file": "02_modeling_ml.ipynb",
        "label": "ML Modeling (Ridge, LightGBM, Prophet)",
        "description": "Traditional ML models, cross-validation, hyperparameter tuning",
    },
    "3": {
        "file": "03_deep_learning_lstm.ipynb",
        "label": "Deep Learning (LSTM)",
        "description": "LSTM architecture with attention, PyTorch, temporal sequences",
    },
    "4": {
        "file": "04_results_analysis.ipynb",
        "label": "Results Analysis",
        "description": "Model evaluation, residuals, error distributions, SHAP",
    },
    "5": {
        "file": "05_feature_review.ipynb",
        "label": "Feature Review",
        "description": "Feature importance, correlation studies, column selection",
    },
}


def menu_training() -> None:
    """Display the ML Training & Notebooks menu."""
    while True:
        clear_screen()
        show_header()

        console.print()
        console.print(
            Panel(
                "[bold]4 - ML TRAINING & NOTEBOOKS[/bold]\n"
                "[dim]Train models via CLI or launch Jupyter notebooks[/dim]",
                border_style="magenta",
                box=box.HEAVY,
            )
        )

        # Show model status
        _show_model_status()

        # Notebook list
        console.print()
        console.print("  [bold cyan]Notebooks:[/bold cyan]")
        for key, nb in NOTEBOOKS.items():
            filepath = Path("notebooks") / nb["file"]
            exists = filepath.exists()
            status = "[green]OK[/green]" if exists else "[red]MISSING[/red]"
            console.print(
                f"    [bold]{key}[/bold]  {nb['label']}  {status}"
            )
            console.print(f"       [dim]{nb['description']}[/dim]")

        console.print()
        console.print("  [bold cyan]CLI Training:[/bold cyan]")
        console.print("    [bold]t[/bold]  Train all models (Ridge, LightGBM, Prophet, LSTM)")
        console.print("    [bold]e[/bold]  Evaluate and compare models")
        console.print("    [bold]d[/bold]  Run EDA analysis (generate charts)")
        console.print()
        console.print("  [bold cyan]Jupyter:[/bold cyan]")
        console.print("    [bold]j[/bold]  Launch Jupyter Notebook server")
        console.print("    [bold]l[/bold]  Launch JupyterLab server")
        console.print()
        console.print("    [bold]0[/bold]  Back to main menu")

        choice = get_choice(
            "Select",
            ["0", "1", "2", "3", "4", "5", "t", "e", "d", "j", "l", ""],
        )

        if choice == "0":
            return
        elif choice in ("1", "2", "3", "4", "5"):
            _open_notebook(choice)
        elif choice == "t":
            _run_training()
        elif choice == "e":
            _run_evaluation()
        elif choice == "d":
            _run_eda()
        elif choice == "j":
            _launch_jupyter("notebook")
        elif choice == "l":
            _launch_jupyter("lab")


def _show_model_status() -> None:
    """Show the status of trained models."""
    models_dir = Path("data/models")

    table = Table(
        title="Trained Models",
        box=box.SIMPLE_HEAVY,
        border_style="magenta",
    )
    table.add_column("Model", style="white", min_width=20)
    table.add_column("Status", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="dim")

    model_files = [
        ("Ridge", "ridge_model.joblib"),
        ("LightGBM", "lightgbm_model.joblib"),
        ("Prophet", "prophet_model.joblib"),
        ("LSTM", "lstm_model.pt"),
        ("Training Results", "training_results.csv"),
        ("Evaluation Report", "evaluation_report.txt"),
    ]

    for name, filename in model_files:
        path = models_dir / filename
        info = _get_file_info(path)
        if info["exists"]:
            status = "[green]TRAINED[/green]"
        else:
            status = "[dim]Not trained[/dim]"

        table.add_row(
            name,
            status,
            info["size_str"] if info["exists"] else "-",
            info["modified"] if info["exists"] else "-",
        )

    console.print()
    console.print(table)


def _open_notebook(key: str) -> None:
    """Open a specific notebook with Jupyter.

    Args:
        key: Notebook key from NOTEBOOKS dict.
    """
    nb = NOTEBOOKS[key]
    filepath = Path("notebooks") / nb["file"]

    if not filepath.exists():
        console.print(f"\n  [red]Notebook not found: {filepath}[/red]")
        pause()
        return

    console.print()
    console.print(
        Panel(
            f"[bold]Opening: {nb['label']}[/bold]\n"
            f"File: {filepath}\n\n"
            f"[dim]The notebook server will start in the background.\n"
            f"Press Ctrl+C in the server terminal to stop it.[/dim]",
            border_style="magenta",
        )
    )

    confirm = get_choice("Launch? [y/N]")
    if confirm not in ("y", "yes"):
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    try:
        console.print(f"\n  [cyan]Starting Jupyter for {nb['file']}...[/cyan]")
        subprocess.Popen(
            ["jupyter", "notebook", str(filepath)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        console.print("  [green]Jupyter started. Check your browser.[/green]")
    except FileNotFoundError:
        console.print(
            "\n  [red]Jupyter not found. Install it with:[/red]"
            "\n  [dim]pip install notebook[/dim]"
        )
    except Exception as e:
        console.print(f"\n  [red]Error: {e}[/red]")

    pause()


def _launch_jupyter(mode: str = "notebook") -> None:
    """Launch Jupyter server in the notebooks directory.

    Args:
        mode: Either 'notebook' or 'lab'.
    """
    console.print()
    console.print(
        Panel(
            f"[bold]Launching Jupyter {'Lab' if mode == 'lab' else 'Notebook'}[/bold]\n"
            f"Directory: notebooks/\n\n"
            f"[dim]The server will start in the background.\n"
            f"Press Ctrl+C in the server terminal to stop it.[/dim]",
            border_style="magenta",
        )
    )

    confirm = get_choice("Launch? [y/N]")
    if confirm not in ("y", "yes"):
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    cmd = "lab" if mode == "lab" else "notebook"
    try:
        console.print(f"\n  [cyan]Starting Jupyter {cmd}...[/cyan]")
        subprocess.Popen(
            ["jupyter", cmd, "--notebook-dir=notebooks/"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        console.print("  [green]Jupyter started. Check your browser.[/green]")
    except FileNotFoundError:
        pkg = "jupyterlab" if mode == "lab" else "notebook"
        console.print(
            f"\n  [red]Jupyter {cmd} not found. Install it with:[/red]"
            f"\n  [dim]pip install {pkg}[/dim]"
        )
    except Exception as e:
        console.print(f"\n  [red]Error: {e}[/red]")

    pause()


def _run_training() -> None:
    """Run ML model training via CLI."""
    console.print()
    console.print(
        Panel(
            "[bold yellow]ML MODEL TRAINING[/bold yellow]\n\n"
            "Models to train:\n"
            "  - Ridge Regression (robust baseline)\n"
            "  - LightGBM (gradient boosting)\n"
            "  - Prophet (time series)\n"
            "  - LSTM (deep learning, if PyTorch available)\n\n"
            "[dim]Prerequisite: features dataset must exist.[/dim]",
            border_style="yellow",
        )
    )

    # Target selection
    console.print("  [cyan]Select target variable:[/cyan]")
    console.print("    [bold]1[/bold]  nb_installations_pac  [dim](heat pump installs — default)[/dim]")
    console.print("    [bold]2[/bold]  nb_installations_clim  [dim](air conditioning installs)[/dim]")
    console.print("    [bold]3[/bold]  nb_dpe_total  [dim](total DPE count)[/dim]")

    target_choice = get_choice("Target [1]")
    targets = {
        "1": "nb_installations_pac",
        "2": "nb_installations_clim",
        "3": "nb_dpe_total",
        "": "nb_installations_pac",
    }
    target = targets.get(target_choice, "nb_installations_pac")

    confirm = get_choice(f"Train with target='{target}'? [y/N]")
    if confirm not in ("y", "yes"):
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    from src.pipeline import run_train, setup_logging
    setup_logging("INFO")

    console.print()
    console.rule(f"[bold green]Training — target: {target}[/bold green]")
    console.print()

    try:
        run_train(target=target)
        console.print()
        console.rule("[bold green]Training Complete[/bold green]")
    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")
        import traceback
        console.print(f"  [dim]{traceback.format_exc()}[/dim]")

    pause()


def _run_evaluation() -> None:
    """Run model evaluation and comparison."""
    console.print()
    console.print(
        Panel(
            "[bold yellow]MODEL EVALUATION[/bold yellow]\n\n"
            "Generates:\n"
            "  - Comparative metrics table\n"
            "  - Predictions vs actual charts\n"
            "  - Residual analysis\n"
            "  - Feature importance\n"
            "  - SHAP analysis\n"
            "  - Full evaluation report",
            border_style="yellow",
        )
    )

    confirm = get_choice("Proceed? [y/N]")
    if confirm not in ("y", "yes"):
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    from src.pipeline import run_evaluate, setup_logging
    setup_logging("INFO")

    console.print()
    console.rule("[bold green]Evaluation[/bold green]")
    console.print()

    try:
        run_evaluate()
        console.print()
        console.rule("[bold green]Evaluation Complete[/bold green]")
    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")

    pause()


def _run_eda() -> None:
    """Run Exploratory Data Analysis."""
    console.print()
    console.print(
        Panel(
            "[bold yellow]EXPLORATORY DATA ANALYSIS[/bold yellow]\n\n"
            "Generates charts and reports:\n"
            "  - Distribution plots\n"
            "  - Correlation heatmaps\n"
            "  - Time series visualizations\n"
            "  - Geographic analysis\n\n"
            "Output: data/analysis/figures/",
            border_style="yellow",
        )
    )

    confirm = get_choice("Proceed? [y/N]")
    if confirm not in ("y", "yes"):
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    from src.pipeline import run_eda, setup_logging
    setup_logging("INFO")

    console.print()
    console.rule("[bold green]EDA Analysis[/bold green]")
    console.print()

    try:
        run_eda()
        console.print()
        console.rule("[bold green]EDA Complete[/bold green]")
    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")

    pause()


# =====================================================================
# MAIN MENU
# =====================================================================

def main_menu() -> None:
    """Display the main interactive menu."""
    while True:
        clear_screen()
        show_header()

        # Quick status line
        db_path = Path("data/hvac_market.db")
        features_path = config.features_data_dir / "hvac_features_dataset.csv"
        db_ok = db_path.exists()
        features_ok = features_path.exists()

        status_parts = []
        if db_ok:
            status_parts.append("[green]DB: OK[/green]")
        else:
            status_parts.append("[red]DB: Missing[/red]")
        if features_ok:
            status_parts.append("[green]Features: OK[/green]")
        else:
            status_parts.append("[red]Features: Missing[/red]")

        # Count raw data sources
        n_sources = sum(
            1 for src in COLLECTION_SOURCES.values()
            if (config.raw_data_dir / src["file"]).exists()
        )
        status_parts.append(f"Sources: {n_sources}/5")

        console.print()
        console.print(f"  Status: {' | '.join(status_parts)}")

        console.print()
        console.print(
            Panel(
                "[bold cyan]MAIN MENU[/bold cyan]",
                border_style="cyan",
                box=box.HEAVY,
                padding=(0, 2),
            )
        )

        console.print()
        console.print("    [bold green]1[/bold green]  [bold]Data Collection (Local)[/bold]")
        console.print(
            "       [dim]Collect from APIs, verify dates, download raw data[/dim]"
        )
        console.print()
        console.print("    [bold blue]2[/bold blue]  [bold]pCloud Data Management[/bold]")
        console.print(
            "       [dim]Sync, upload, and monitor data on pCloud[/dim]"
        )
        console.print()
        console.print("    [bold yellow]3[/bold yellow]  [bold]Data Processing[/bold]")
        console.print(
            "       [dim]Clean, merge, feature engineering, outlier detection[/dim]"
        )
        console.print()
        console.print("    [bold magenta]4[/bold magenta]  [bold]ML Training & Notebooks[/bold]")
        console.print(
            "       [dim]Train models, evaluate, launch Jupyter notebooks[/dim]"
        )
        console.print()
        console.print("    [bold red]0[/bold red]  [bold]Exit[/bold]")

        choice = get_choice("Select", ["0", "1", "2", "3", "4", ""])

        if choice == "0":
            console.print("\n  [dim]Goodbye![/dim]\n")
            sys.exit(0)
        elif choice == "1":
            menu_collection()
        elif choice == "2":
            menu_pcloud()
        elif choice == "3":
            menu_processing()
        elif choice == "4":
            menu_training()


# =====================================================================
# Entry point
# =====================================================================

def main() -> None:
    """CLI entry point for the interactive menu."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n\n  [dim]Interrupted. Goodbye![/dim]\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
