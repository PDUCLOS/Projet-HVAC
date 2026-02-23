# -*- coding: utf-8 -*-
"""
Interactive CLI Menu — HVAC Market Analysis Pipeline.
======================================================

Professional interactive menu for managing the full data pipeline:
    1. Data Collection & Data Sources (API + manual copy)
    2. Data Processing (clean, merge, features, outliers)
    3. ML Training & Notebooks
    4. Deep Learning & RL (LSTM, Reinforcement Learning, results exploration)

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
                "[bold]1 - DATA COLLECTION & SOURCES[/bold]\n"
                "[dim]Collect from APIs or manually copy pre-collected data[/dim]",
                border_style="green",
                box=box.HEAVY,
            )
        )

        # Show the sources table with dates
        console.print()
        with console.status("[cyan]Scanning local data files...[/cyan]"):
            table = _build_collection_table()
        console.print(table)

        # Data origin info
        console.print()
        console.print(
            Panel(
                "[bold cyan]Data Origin — Choose your method:[/bold cyan]\n\n"
                "[bold]Option A — Collect from APIs[/bold] [dim](~1h for DPE)[/dim]\n"
                "  Use options [bold]a[/bold] or [bold]1-5[/bold] below to collect "
                "directly from Open Data APIs.\n"
                "  All sources are Open Data — no API key, no GDPR constraints.\n\n"
                "[bold]Option B — Manual copy from pCloud[/bold] [dim](fast, ~5 min)[/dim]\n"
                "  Download pre-collected data from pCloud:\n"
                "  [cyan]https://e.pcloud.link/publink/show?"
                "code=kZbQQ3Zg1slD5WfRgh42fH5rRpDDYWyBEsy[/cyan]\n\n"
                "  Copy files into these directories:\n"
                "    [green]data/raw/weather/[/green]   <- weather_france.csv\n"
                "    [green]data/raw/insee/[/green]     <- indicateurs_economiques.csv,"
                " reference_departements.csv\n"
                "    [green]data/raw/eurostat/[/green]  <- ipi_hvac_france.csv\n"
                "    [green]data/raw/sitadel/[/green]   <- permis_construire_france.csv\n"
                "    [green]data/raw/dpe/[/green]       <- dpe_france_all.csv"
                " (or per-dept files)\n"
                "    [green]data/[/green]               <- hvac_market.db (SQLite)\n"
                "    [green]data/features/[/green]      <- hvac_ml_dataset.csv,"
                " hvac_features_dataset.csv",
                border_style="green",
                box=box.ROUNDED,
            )
        )

        # Menu options
        console.print()
        console.print("  [bold cyan]API Collection:[/bold cyan]")
        console.print("    [bold]a[/bold]  Collect ALL sources (full pipeline)")
        console.print("    [bold]1-5[/bold]  Collect a specific source:")

        for i, (key, src) in enumerate(COLLECTION_SOURCES.items(), 1):
            console.print(f"         [bold]{i}[/bold]  {src['label']}  [dim]({src['estimate']})[/dim]")

        console.print()
        console.print("  [bold cyan]Verification & Analysis:[/bold cyan]")
        console.print("    [bold]v[/bold]  Verify data dates (detailed report)")
        console.print("    [bold]p[/bold]  PAC efficiency map (altitude & climate analysis)")
        console.print()
        console.print("    [bold]0[/bold]  Back to main menu")

        choice = get_choice("Select")

        if choice == "0":
            return

        elif choice == "a":
            _run_collect_all()

        elif choice == "v":
            _show_date_verification()

        elif choice == "p":
            _show_pac_efficiency_map()

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


def _show_pac_efficiency_map() -> None:
    """Show PAC efficiency analysis by department (altitude + climate).

    Displays a comprehensive table with:
    - Prefecture altitude (point estimate for the reference city)
    - Department mean altitude (full territory average from IGN BD ALTI)
    - Mountain zone percentage (loi montagne classification)
    - Population density (hab/km2, proxy for urbanization vs rural mountain)
    - COP estimate and PAC viability rating
    """
    clear_screen()
    show_header()

    console.print()
    console.print(
        Panel(
            "[bold]PAC (HEAT PUMP) EFFICIENCY ANALYSIS[/bold]\n"
            f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n\n"
            "[dim]COP = Coefficient of Performance (higher = more efficient)\n"
            "Below -7°C, air-source heat pump COP drops critically (<2.0)\n"
            "Altitude mean = IGN BD ALTI department average (not just prefecture)\n"
            "Zone montagne = % territory classified mountain (loi montagne)[/dim]",
            border_style="magenta",
            box=box.DOUBLE,
        )
    )

    from config.settings import PREFECTURE_ELEVATIONS, FRANCE_DEPARTMENTS

    # Load altitude distribution data from reference CSV
    import pandas as pd
    ref_path = Path("data/raw/insee/reference_departements.csv")
    ref_data = {}
    if ref_path.exists():
        ref_df = pd.read_csv(ref_path, dtype={"dept": str})
        ref_df["dept"] = ref_df["dept"].astype(str).str.zfill(2)
        for _, row in ref_df.iterrows():
            ref_data[row["dept"]] = {
                "altitude_mean": row.get("altitude_mean", 0),
                "pct_zone_montagne": row.get("pct_zone_montagne", 0),
                "densite_pop": row.get("densite_pop", 0),
            }

    # Build department table sorted by mean altitude (highest first)
    table = Table(
        title="Department Analysis — Altitude Distribution & PAC Viability",
        box=box.ROUNDED,
        show_lines=False,
        border_style="magenta",
    )

    table.add_column("Dept", style="bold", width=5)
    table.add_column("Prefecture", min_width=16)
    table.add_column("Alt. Pref.", justify="right", min_width=8)
    table.add_column("Alt. Moy.", justify="right", min_width=8)
    table.add_column("Zone Mt.", justify="right", min_width=7)
    table.add_column("Densité", justify="right", min_width=8)
    table.add_column("Mountain", justify="center", width=8)
    table.add_column("COP Est.", justify="center", min_width=8)
    table.add_column("PAC", min_width=10)

    # Collect data and sort by mean altitude descending
    rows = []
    for city, info in FRANCE_DEPARTMENTS.items():
        dept = info["dept"]
        alt_pref = PREFECTURE_ELEVATIONS.get(dept, 0)
        ref = ref_data.get(dept, {})
        alt_mean = ref.get("altitude_mean", alt_pref)
        pct_mt = ref.get("pct_zone_montagne", 0)
        densite = ref.get("densite_pop", 0)
        # Use mean altitude for COP (more representative)
        is_mountain = alt_mean > 800 or pct_mt > 50
        cop_base = max(1.0, min(5.0, 4.5 - 0.0005 * alt_mean))
        rows.append((dept, city, alt_pref, alt_mean, pct_mt, densite,
                      is_mountain, cop_base))

    rows.sort(key=lambda x: x[3], reverse=True)

    for dept, city, alt_pref, alt_mean, pct_mt, densite, is_mountain, cop_base in rows:
        # Color coding for altitude
        if alt_mean > 800:
            alt_style = "[bold red]"
            mountain = "[bold red]YES[/bold red]"
        elif alt_mean > 400:
            alt_style = "[yellow]"
            mountain = "[bold yellow]PARTIAL[/bold yellow]" if pct_mt > 30 else "[dim]-[/dim]"
        else:
            alt_style = "[green]"
            mountain = "[dim]-[/dim]"

        # Mountain zone color
        if pct_mt >= 70:
            mt_str = f"[bold red]{pct_mt}%[/bold red]"
        elif pct_mt >= 30:
            mt_str = f"[yellow]{pct_mt}%[/yellow]"
        elif pct_mt > 0:
            mt_str = f"[dim]{pct_mt}%[/dim]"
        else:
            mt_str = "[dim]-[/dim]"

        # COP and viability
        if cop_base < 3.0:
            cop_str = f"[bold red]{cop_base:.1f}[/bold red]"
            viability = "[bold red]LOW[/bold red]"
        elif cop_base < 4.0:
            cop_str = f"[yellow]{cop_base:.1f}[/yellow]"
            viability = "[yellow]MODERATE[/yellow]"
        else:
            cop_str = f"[green]{cop_base:.1f}[/green]"
            viability = "[green]HIGH[/green]"

        # Density formatting
        if densite > 1000:
            dens_str = f"[cyan]{densite:,}[/cyan]"
        elif densite > 100:
            dens_str = f"{densite:,}"
        else:
            dens_str = f"[dim]{densite:,}[/dim]"

        table.add_row(
            dept, city,
            f"{alt_pref}m",
            f"{alt_style}{alt_mean}m",
            mt_str,
            dens_str,
            mountain,
            cop_str,
            viability,
        )

    console.print()
    console.print(table)

    # Summary stats using mean altitude and mountain zone
    n_mountain = sum(1 for r in rows if r[3] > 800 or r[4] > 50)
    n_partial = sum(1 for r in rows if not (r[3] > 800 or r[4] > 50) and r[4] > 15)
    n_low = sum(1 for r in rows if r[3] <= 400 and r[4] == 0)
    n_other = len(rows) - n_mountain - n_partial - n_low

    console.print()
    console.print(
        Panel(
            f"[bold]Summary (based on mean altitude + mountain zone %):[/bold]\n"
            f"  [red]Mountain (alt>800m or >50% zone):[/red]  {n_mountain} departments — PAC viability LOW\n"
            f"  [yellow]Partial mountain (15-50% zone):[/yellow]   {n_partial} departments — PAC viability MODERATE\n"
            f"  [green]Lowland (<400m, no mountain):[/green]     {n_low} departments — PAC viability HIGH\n"
            f"  [white]Other (mixed):[/white]                    {n_other} departments\n\n"
            f"[dim]Sources:\n"
            f"  - Alt. Pref.: Open-Meteo Elevation API (Copernicus DEM GLO-90)\n"
            f"  - Alt. Moy.: IGN BD ALTI (department mean altitude)\n"
            f"  - Zone Mt.: Loi montagne classification (%% territory)\n"
            f"  - Densité: INSEE Recensement (hab/km²)[/dim]",
            border_style="cyan",
        )
    )

    pause()


# =====================================================================
# MENU 2 — Data Processing
# =====================================================================

def menu_processing() -> None:
    """Display the Data Processing menu."""
    while True:
        clear_screen()
        show_header()

        console.print()
        console.print(
            Panel(
                "[bold]2 - DATA PROCESSING[/bold]\n"
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
        console.print("    [bold]g[/bold]  Geographic profile  [dim](altitude, mountain zones, density)[/dim]")
        console.print("    [bold]0[/bold]  Back to main menu")

        choice = get_choice(
            "Select",
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "g", ""],
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
        elif choice == "g":
            _show_geographic_profile()


def _show_geographic_profile() -> None:
    """Show geographic profile of reference departments (altitude distribution).

    Displays altitude_mean, pct_zone_montagne, and densite_pop from
    reference_departements.csv, giving a complete picture of each
    department's geographic characteristics for HVAC analysis.
    """
    clear_screen()
    show_header()

    console.print()
    console.print(
        Panel(
            "[bold]GEOGRAPHIC PROFILE — Altitude & Population Distribution[/bold]\n"
            f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n\n"
            "[dim]Data sources:\n"
            "  - altitude_mean: Department average altitude (IGN BD ALTI)\n"
            "  - pct_zone_montagne: % territory classified mountain (loi montagne)\n"
            "  - densite_pop: Population density in hab/km² (INSEE Recensement)\n"
            "  - pct_maisons: % houses vs apartments (INSEE Recensement)[/dim]",
            border_style="magenta",
            box=box.DOUBLE,
        )
    )

    import pandas as pd
    ref_path = Path("data/raw/insee/reference_departements.csv")

    if not ref_path.exists():
        console.print("\n  [bold red]Reference file not found:[/bold red] "
                       f"{ref_path}")
        pause()
        return

    ref_df = pd.read_csv(ref_path, dtype={"dept": str})
    ref_df["dept"] = ref_df["dept"].astype(str).str.zfill(2)

    # Check required columns
    required = ["altitude_mean", "pct_zone_montagne", "densite_pop"]
    missing = [c for c in required if c not in ref_df.columns]
    if missing:
        console.print(f"\n  [bold red]Missing columns:[/bold red] {missing}")
        console.print("  Run [bold]git pull[/bold] to get the updated reference file.")
        pause()
        return

    # Build table sorted by altitude_mean descending
    table = Table(
        title="Department Geographic Profile (sorted by mean altitude)",
        box=box.ROUNDED,
        show_lines=False,
        border_style="magenta",
    )

    table.add_column("Dept", style="bold", width=5)
    table.add_column("Name", min_width=20)
    table.add_column("Alt. Moy.", justify="right", min_width=8)
    table.add_column("Zone Mt.", justify="right", min_width=7)
    table.add_column("Densité", justify="right", min_width=9)
    table.add_column("% Maisons", justify="right", min_width=8)
    table.add_column("Classification", min_width=14)

    ref_df = ref_df.sort_values("altitude_mean", ascending=False)

    for _, row in ref_df.iterrows():
        alt_mean = int(row.get("altitude_mean", 0))
        pct_mt = int(row.get("pct_zone_montagne", 0))
        densite = int(row.get("densite_pop", 0))
        pct_m = int(row.get("pct_maisons", 0))

        # Classification
        if alt_mean > 800 or pct_mt > 50:
            classif = "[bold red]MOUNTAIN[/bold red]"
            alt_style = "[bold red]"
        elif pct_mt > 15 or alt_mean > 400:
            classif = "[yellow]SEMI-MOUNTAIN[/yellow]"
            alt_style = "[yellow]"
        elif densite > 500:
            classif = "[cyan]URBAN[/cyan]"
            alt_style = "[green]"
        else:
            classif = "[green]LOWLAND[/green]"
            alt_style = "[green]"

        # Mountain zone color
        if pct_mt >= 70:
            mt_str = f"[bold red]{pct_mt}%[/bold red]"
        elif pct_mt >= 30:
            mt_str = f"[yellow]{pct_mt}%[/yellow]"
        elif pct_mt > 0:
            mt_str = f"[dim]{pct_mt}%[/dim]"
        else:
            mt_str = "[dim]-[/dim]"

        # Density color
        if densite > 1000:
            dens_str = f"[cyan]{densite:,}[/cyan]"
        elif densite > 100:
            dens_str = f"{densite:,}"
        else:
            dens_str = f"[dim]{densite:,}[/dim]"

        table.add_row(
            str(row["dept"]),
            str(row.get("dept_name", "")),
            f"{alt_style}{alt_mean}m",
            mt_str,
            dens_str,
            f"{pct_m}%",
            classif,
        )

    console.print()
    console.print(table)

    # Summary
    n_mountain = len(ref_df[(ref_df["altitude_mean"] > 800) | (ref_df["pct_zone_montagne"] > 50)])
    n_semi = len(ref_df[
        ~((ref_df["altitude_mean"] > 800) | (ref_df["pct_zone_montagne"] > 50))
        & ((ref_df["pct_zone_montagne"] > 15) | (ref_df["altitude_mean"] > 400))
    ])
    n_urban = len(ref_df[
        ~((ref_df["altitude_mean"] > 800) | (ref_df["pct_zone_montagne"] > 50))
        & ~((ref_df["pct_zone_montagne"] > 15) | (ref_df["altitude_mean"] > 400))
        & (ref_df["densite_pop"] > 500)
    ])
    n_lowland = len(ref_df) - n_mountain - n_semi - n_urban

    console.print()
    console.print(
        Panel(
            f"[bold]Classification Summary:[/bold]\n"
            f"  [red]Mountain[/red]       {n_mountain:>2} depts  "
            f"[dim](alt>800m or >50% zone montagne)[/dim]\n"
            f"  [yellow]Semi-Mountain[/yellow]  {n_semi:>2} depts  "
            f"[dim](15-50% zone montagne or alt 400-800m)[/dim]\n"
            f"  [cyan]Urban[/cyan]          {n_urban:>2} depts  "
            f"[dim](lowland + density > 500 hab/km²)[/dim]\n"
            f"  [green]Lowland[/green]        {n_lowland:>2} depts  "
            f"[dim](plain, low density)[/dim]\n\n"
            f"[bold]Key Stats:[/bold]\n"
            f"  Mean altitude range: {int(ref_df['altitude_mean'].min())}m "
            f"- {int(ref_df['altitude_mean'].max())}m\n"
            f"  Density range: {int(ref_df['densite_pop'].min())} "
            f"- {int(ref_df['densite_pop'].max()):,} hab/km²\n"
            f"  Departments with mountain zone: "
            f"{len(ref_df[ref_df['pct_zone_montagne'] > 0])}/96",
            border_style="cyan",
        )
    )

    pause()


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
        ("Cleaned (weather)", config.processed_data_dir / "weather" / "weather_france.csv"),
        ("Cleaned (DPE)", config.processed_data_dir / "dpe" / "dpe_france_clean.csv"),
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
# MENU 3 — ML Training & Notebooks
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
                "[bold]3 - ML TRAINING & NOTEBOOKS[/bold]\n"
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
            [sys.executable, "-m", "jupyter", "notebook", str(filepath)],
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
            [sys.executable, "-m", "jupyter", cmd, "--notebook-dir=notebooks/"],
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
# MENU 4 — Deep Learning & RL
# =====================================================================


def _check_pytorch_available() -> bool:
    """Check if PyTorch is installed and importable."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _check_gymnasium_available() -> bool:
    """Check if gymnasium is installed and importable."""
    try:
        import gymnasium  # noqa: F401
        return True
    except ImportError:
        return False


def _show_dl_rl_status() -> None:
    """Show the status of Deep Learning and RL artifacts."""
    table = Table(
        title="DL & RL Artifacts",
        box=box.SIMPLE_HEAVY,
        border_style="bright_red",
    )
    table.add_column("Artifact", style="white", min_width=25)
    table.add_column("Status", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="dim")

    artifacts = [
        ("LSTM Model", Path("data/models/lstm_model.pt"), None),
        ("Training Results", Path("data/models/training_results.csv"), None),
        ("Evaluation JSON", Path("data/models/evaluation_results.json"), None),
        (
            "RL Learning Curve",
            Path("reports/figures/rl_learning_curve.png"),
            Path("reports/figures/rl_courbe_apprentissage.png"),
        ),
        ("RL Q-Values Heatmap", Path("reports/figures/rl_heatmap_q_values.png"), None),
    ]

    for name, filepath, alt_path in artifacts:
        check_path = filepath
        if not filepath.exists() and alt_path and alt_path.exists():
            check_path = alt_path

        info = _get_file_info(check_path)
        if info["exists"]:
            status = "[green]READY[/green]"
        else:
            status = "[dim]Not generated[/dim]"

        table.add_row(
            name,
            status,
            info["size_str"] if info["exists"] else "-",
            info["modified"] if info["exists"] else "-",
        )

    console.print()
    console.print(table)


def _run_lstm_standalone() -> None:
    """Train the LSTM model standalone with configurable hyperparameters."""
    console.print()
    console.print(
        Panel(
            "[bold bright_red]LSTM STANDALONE TRAINING[/bold bright_red]\n\n"
            "Architecture:\n"
            "  - 1-layer LSTM + Dropout(0.3) + Linear\n"
            "  - HuberLoss (robust to outliers)\n"
            "  - Adam optimizer (lr=0.001)\n"
            "  - Early stopping on validation loss\n\n"
            "[dim]Warning: With ~288 training rows, LSTM is pedagogical.\n"
            "Classical models (Ridge, LightGBM) are expected to perform better.[/dim]",
            border_style="bright_red",
        )
    )

    # Check PyTorch
    if not _check_pytorch_available():
        console.print(
            "\n  [bold red]PyTorch not available.[/bold red]"
            "\n  [dim]Install via: pip install -r requirements-dl.txt[/dim]"
        )
        pause()
        return

    # Check features dataset
    features_path = config.features_data_dir / "hvac_features_dataset.csv"
    if not features_path.exists():
        console.print(
            "\n  [bold red]Features dataset not found.[/bold red]"
            "\n  [dim]Run Menu 2 > Processing first.[/dim]"
        )
        pause()
        return

    # Hyperparameter configuration
    console.print("\n  [cyan]Configure hyperparameters (press Enter for defaults):[/cyan]")

    lookback_input = get_choice("  Lookback months [3]")
    lookback = int(lookback_input) if lookback_input.isdigit() else 3

    hidden_input = get_choice("  Hidden size [32]")
    hidden_size = int(hidden_input) if hidden_input.isdigit() else 32

    epochs_input = get_choice("  Max epochs [100]")
    epochs = int(epochs_input) if epochs_input.isdigit() else 100

    console.print(
        f"\n  Configuration: lookback={lookback}, hidden_size={hidden_size}, "
        f"epochs={epochs}"
    )

    confirm = get_choice("  Proceed with LSTM training? [y/N]")
    if confirm not in ("y", "yes"):
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    from src.pipeline import setup_logging
    setup_logging("INFO")

    console.print()
    console.rule("[bold bright_red]LSTM Training[/bold bright_red]")
    console.print()

    try:
        from src.models.train import ModelTrainer
        from src.models.deep_learning import LSTMModel
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        import pandas as pd

        trainer = ModelTrainer(config, target="nb_installations_pac")
        df = trainer.load_dataset()
        df_train, df_val, df_test = trainer.temporal_split(df)

        X_train, y_train = trainer.prepare_features(df_train)
        X_val, y_val = trainer.prepare_features(df_val)
        X_test, y_test = trainer.prepare_features(df_test)

        # Remove NaN targets
        mask_train = y_train.notna()
        X_train, y_train = X_train[mask_train], y_train[mask_train]
        mask_val = y_val.notna()
        X_val, y_val = X_val[mask_val], y_val[mask_val]
        mask_test = y_test.notna()
        X_test, y_test = X_test[mask_test], y_test[mask_test]

        # Drop all-NaN columns, then impute remaining NaN (same as train.py)
        all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
        if all_nan_cols:
            console.print(
                f"  [yellow]Dropping {len(all_nan_cols)} all-NaN columns: "
                f"{', '.join(all_nan_cols)}[/yellow]"
            )
            X_train = X_train.drop(columns=all_nan_cols)
            X_val = X_val.drop(columns=all_nan_cols)
            X_test = X_test.drop(columns=all_nan_cols)

        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns, index=X_train.index,
        )
        X_val_imp = pd.DataFrame(
            imputer.transform(X_val),
            columns=X_val.columns, index=X_val.index,
        )
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns, index=X_test.index,
        )

        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_imp),
            columns=X_train_imp.columns, index=X_train_imp.index,
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val_imp),
            columns=X_val_imp.columns, index=X_val_imp.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_imp),
            columns=X_test_imp.columns, index=X_test_imp.index,
        )

        # Train LSTM with custom hyperparameters
        lstm = LSTMModel(
            config, target="nb_installations_pac",
            lookback=lookback, hidden_size=hidden_size, epochs=epochs,
        )
        result = lstm.train_and_evaluate(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test,
        )

        # Save LSTM model
        import torch
        models_dir = Path(config.data_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "lstm_model.pt"
        torch.save(result["model"].state_dict(), model_path)
        console.print(f"\n  [green]Model saved → {model_path}[/green]")

        # Save imputer and scaler for inference
        import pickle
        with open(models_dir / "lstm_imputer.pkl", "wb") as f:
            pickle.dump(imputer, f)
        with open(models_dir / "lstm_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        console.print(f"  [green]Imputer & scaler saved → {models_dir}[/green]")

        # Display results in a rich table
        console.print()
        results_table = Table(
            title="LSTM Results",
            border_style="bright_red",
            box=box.ROUNDED,
        )
        results_table.add_column("Metric", style="white")
        results_table.add_column("Validation", justify="right", style="cyan")
        results_table.add_column("Test", justify="right", style="green")

        for metric in ["rmse", "mae", "mape", "r2"]:
            val_v = result["metrics_val"].get(metric, float("nan"))
            test_v = result["metrics_test"].get(metric, float("nan"))
            results_table.add_row(metric.upper(), f"{val_v:.4f}", f"{test_v:.4f}")

        results_table.add_row(
            "Best Epoch", str(result.get("best_epoch", "N/A")), "-",
        )
        console.print(results_table)

        console.print()
        console.rule("[bold green]LSTM Training Complete[/bold green]")

    except ImportError as e:
        console.print(f"\n  [bold red]Import error:[/bold red] {e}")
        console.print("  [dim]Install missing dependencies.[/dim]")
    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")
        import traceback
        console.print(f"  [dim]{traceback.format_exc()}[/dim]")

    pause()


def _run_rl_demo() -> None:
    """Run the Reinforcement Learning Q-Learning demo."""
    console.print()
    console.print(
        Panel(
            "[bold bright_red]REINFORCEMENT LEARNING DEMO[/bold bright_red]\n\n"
            "Q-Learning agent for HVAC maintenance optimization:\n"
            "  - Environment: 12-month episodes, 6-dim state, 4 actions\n"
            "  - Training: 1000 episodes, epsilon-greedy exploration\n"
            "  - Comparison: Q-Learning vs random policy baseline\n"
            "  - Output: learning curve + Q-values heatmap\n\n"
            f"[dim]Gymnasium: "
            f"{'installed' if _check_gymnasium_available() else 'not installed (simplified mode)'}"
            f"[/dim]",
            border_style="bright_red",
        )
    )

    confirm = get_choice("  Run RL demo? [y/N]")
    if confirm not in ("y", "yes"):
        console.print("  [dim]Cancelled.[/dim]")
        pause()
        return

    from src.pipeline import setup_logging
    setup_logging("INFO")

    console.print()
    console.rule("[bold bright_red]RL Training[/bold bright_red]")
    console.print()

    try:
        from src.models.reinforcement_learning_demo import main as rl_main
        rl_main()
        console.print()
        console.rule("[bold green]RL Demo Complete[/bold green]")
        console.print("  [dim]Figures saved in reports/figures/[/dim]")
    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")
        import traceback
        console.print(f"  [dim]{traceback.format_exc()}[/dim]")

    pause()


def _view_lstm_results() -> None:
    """View LSTM training results and metrics."""
    console.print()
    console.print(
        Panel(
            "[bold bright_red]LSTM RESULTS[/bold bright_red]",
            border_style="bright_red",
        )
    )

    # Check for evaluation results JSON
    json_path = Path("data/models/evaluation_results.json")
    if json_path.exists():
        import json

        with open(json_path, encoding="utf-8") as f:
            results = json.load(f)

        if "lstm" in results:
            lstm_data = results["lstm"]
            table = Table(
                title="LSTM Metrics (from evaluation)",
                border_style="bright_red",
                box=box.ROUNDED,
            )
            table.add_column("Metric", style="white")
            table.add_column("Value", justify="right", style="cyan")

            for key, value in lstm_data.items():
                if isinstance(value, (int, float)):
                    table.add_row(key, f"{value:.4f}")
                else:
                    table.add_row(key, str(value))

            console.print(table)
        else:
            console.print("  [yellow]LSTM not found in evaluation results.[/yellow]")
            console.print(
                "  [dim]Train with Menu 3 > 't' or Menu 4 > '1' first, "
                "then evaluate with Menu 3 > 'e'.[/dim]"
            )
    else:
        console.print("  [yellow]No evaluation results found.[/yellow]")
        console.print("  [dim]Run evaluation first (Menu 3 > 'e').[/dim]")

    # Show LSTM prediction chart paths
    console.print()
    console.print("  [cyan]LSTM prediction charts:[/cyan]")
    figures_dir = Path("data/models/figures")
    lstm_figs = sorted(figures_dir.glob("*lstm*")) if figures_dir.exists() else []
    if lstm_figs:
        for fig in lstm_figs:
            info = _get_file_info(fig)
            console.print(
                f"    [green]OK[/green]  {fig.name}  "
                f"[dim]({info['size_str']}, {info['modified']})[/dim]"
            )
    else:
        console.print("    [dim]No LSTM charts found.[/dim]")

    pause()


def _view_rl_results() -> None:
    """View RL training results, plots, and policy summary."""
    console.print()
    console.print(
        Panel(
            "[bold bright_red]RL RESULTS[/bold bright_red]",
            border_style="bright_red",
        )
    )

    # List RL figures
    console.print("  [cyan]RL figures:[/cyan]")
    rl_files = [
        ("Learning Curve", "rl_learning_curve.png", "rl_courbe_apprentissage.png"),
        ("Q-Values Heatmap", "rl_heatmap_q_values.png", None),
    ]

    reports_dir = Path("reports/figures")
    found_any = False
    for label, primary, alternate in rl_files:
        path = reports_dir / primary
        if not path.exists() and alternate:
            path = reports_dir / alternate
        if path.exists():
            found_any = True
            info = _get_file_info(path)
            console.print(
                f"    [green]OK[/green]  {label}: {path.name}  "
                f"[dim]({info['size_str']}, {info['modified']})[/dim]"
            )
        else:
            console.print(f"    [dim]--[/dim]  {label}: [dim]Not generated[/dim]")

    if not found_any:
        console.print("\n  [yellow]No RL results found. Run the RL demo first (option 2).[/yellow]")

    pause()


def _open_figure(filepath: Path) -> None:
    """Open a figure with the system's default viewer.

    Args:
        filepath: Path to the PNG file to open.
    """
    if not filepath.exists():
        console.print(f"  [red]File not found: {filepath}[/red]")
        return

    console.print(f"\n  [cyan]Opening: {filepath}[/cyan]")
    try:
        if os.name == "nt":
            os.startfile(str(filepath))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(filepath)])
        else:
            subprocess.Popen(["xdg-open", str(filepath)])
        console.print("  [green]Opened in system viewer.[/green]")
    except Exception as e:
        console.print(f"  [red]Could not open: {e}[/red]")


def _browse_figures() -> None:
    """Browse all generated figures across the project."""
    console.print()
    console.print(
        Panel(
            "[bold bright_red]FIGURE BROWSER[/bold bright_red]\n"
            "[dim]Browse all generated charts and visualizations[/dim]",
            border_style="bright_red",
        )
    )

    figure_dirs = [
        ("ML Model Figures", Path("data/models/figures")),
        ("RL & Reports Figures", Path("reports/figures")),
        ("EDA Figures", Path("data/analysis/figures")),
    ]

    all_figures = []
    for section, dirpath in figure_dirs:
        if dirpath.exists():
            pngs = sorted(dirpath.glob("*.png"))
            if pngs:
                console.print(f"\n  [bold cyan]{section}[/bold cyan] ({dirpath})")
                for png in pngs:
                    idx = len(all_figures) + 1
                    info = _get_file_info(png)
                    console.print(
                        f"    [bold]{idx:>3}[/bold]  {png.name}  "
                        f"[dim]({info['size_str']}, {info['modified']})[/dim]"
                    )
                    all_figures.append(png)

    if not all_figures:
        console.print(
            "\n  [yellow]No figures found. "
            "Run training/evaluation/EDA first.[/yellow]"
        )
        pause()
        return

    console.print(f"\n  [dim]Total: {len(all_figures)} figures[/dim]")
    choice = get_choice("  Enter figure number to open (or 0 to go back)")

    if choice == "0" or not choice:
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(all_figures):
            _open_figure(all_figures[idx])
        else:
            console.print("  [red]Invalid number.[/red]")
    except ValueError:
        console.print("  [red]Invalid input.[/red]")

    pause()


def _view_evaluation_metrics() -> None:
    """Display evaluation metrics from evaluation_results.json in a rich table."""
    console.print()
    json_path = Path("data/models/evaluation_results.json")

    if not json_path.exists():
        console.print(
            "  [yellow]No evaluation results found.[/yellow]\n"
            "  [dim]Run evaluation first (Menu 3 > 'e').[/dim]"
        )
        pause()
        return

    import json

    with open(json_path, encoding="utf-8") as f:
        results = json.load(f)

    table = Table(
        title="Model Evaluation Results",
        box=box.ROUNDED,
        border_style="bright_red",
        show_lines=True,
    )
    table.add_column("Model", style="bold white", min_width=15)
    table.add_column("Val RMSE", justify="right", style="cyan")
    table.add_column("Val MAE", justify="right")
    table.add_column("Val R2", justify="right", style="green")
    table.add_column("Test RMSE", justify="right", style="cyan")
    table.add_column("Test MAE", justify="right")
    table.add_column("Test R2", justify="right", style="green")

    for model_name, metrics in results.items():
        row = [model_name]
        for key in ["val_rmse", "val_mae", "val_r2", "test_rmse", "test_mae", "test_r2"]:
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                row.append(f"{val:.4f}")
            else:
                row.append("N/A")
        table.add_row(*row)

    console.print()
    console.print(table)
    pause()


def _view_text_report() -> None:
    """View text reports inline in the terminal."""
    console.print()
    console.print(
        Panel(
            "[bold bright_red]TEXT REPORTS[/bold bright_red]",
            border_style="bright_red",
        )
    )

    reports = [
        ("Evaluation Report", Path("data/models/evaluation_report.txt")),
        ("Outlier Report", Path("data/analysis/outlier_report.txt")),
    ]

    for i, (label, path) in enumerate(reports, 1):
        exists = path.exists()
        status = "[green]Available[/green]" if exists else "[dim]Not found[/dim]"
        console.print(f"    [bold]{i}[/bold]  {label}  {status}")

    choice = get_choice("  Select report to view (or 0 to go back)")
    if choice == "0" or not choice:
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(reports):
            label, path = reports[idx]
            if path.exists():
                console.print()
                console.rule(f"[bold]{label}[/bold]")
                content = path.read_text(encoding="utf-8")
                console.print(f"\n{content}")
                console.rule()
            else:
                console.print(f"  [red]{label} not found at {path}[/red]")
        else:
            console.print("  [red]Invalid choice.[/red]")
    except ValueError:
        console.print("  [red]Invalid input.[/red]")

    pause()


def _open_notebook_from_dl(key: str) -> None:
    """Open a notebook from the DL & RL menu.

    Args:
        key: Notebook key from NOTEBOOKS dict.
    """
    if key not in NOTEBOOKS:
        console.print(f"  [red]Unknown notebook key: {key}[/red]")
        pause()
        return
    _open_notebook(key)


def menu_deep_learning() -> None:
    """Display the Deep Learning & RL menu."""
    while True:
        clear_screen()
        show_header()

        console.print()
        console.print(
            Panel(
                "[bold]4 - DEEP LEARNING & RL[/bold]\n"
                "[dim]LSTM training, Q-Learning demo, results exploration[/dim]",
                border_style="bright_red",
                box=box.HEAVY,
            )
        )

        # Show DL/RL artifact status
        _show_dl_rl_status()

        # Dependency status
        pytorch_ok = _check_pytorch_available()
        gym_ok = _check_gymnasium_available()
        console.print()
        console.print(
            f"  Dependencies: "
            f"PyTorch {'[green]OK[/green]' if pytorch_ok else '[red]MISSING[/red]'}"
            f" | "
            f"Gymnasium {'[green]OK[/green]' if gym_ok else '[yellow]Optional[/yellow]'}"
        )

        # Menu options
        console.print()
        console.print("  [bold cyan]Training:[/bold cyan]")
        console.print(
            "    [bold bright_red]1[/bold bright_red]  "
            "Train LSTM standalone (configurable hyperparameters)"
        )
        console.print(
            "    [bold bright_red]2[/bold bright_red]  "
            "Run RL demo (Q-Learning training + visualization)"
        )

        console.print()
        console.print("  [bold cyan]View Results:[/bold cyan]")
        console.print(
            "    [bold bright_red]3[/bold bright_red]  "
            "View LSTM results (metrics, predictions)"
        )
        console.print(
            "    [bold bright_red]4[/bold bright_red]  "
            "View RL results (learning curves, Q-values)"
        )

        console.print()
        console.print("  [bold cyan]Notebooks:[/bold cyan]")
        console.print(
            "    [bold bright_red]5[/bold bright_red]  "
            "Open LSTM notebook (03_deep_learning_lstm.ipynb)"
        )
        console.print(
            "    [bold bright_red]6[/bold bright_red]  "
            "Open ML Modeling notebook (02_modeling_ml.ipynb)"
        )
        console.print(
            "    [bold bright_red]7[/bold bright_red]  "
            "Open Results Analysis notebook (04_results_analysis.ipynb)"
        )

        console.print()
        console.print("  [bold cyan]Results Exploration:[/bold cyan]")
        console.print(
            "    [bold bright_red]8[/bold bright_red]  "
            "Browse all figures (ML, RL, EDA)"
        )
        console.print(
            "    [bold bright_red]9[/bold bright_red]  "
            "View evaluation metrics (all models comparison)"
        )
        console.print(
            "    [bold bright_red]r[/bold bright_red]  "
            "View text reports (evaluation, outliers)"
        )

        console.print()
        console.print("    [bold]0[/bold]  Back to main menu")

        choice = get_choice(
            "Select",
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "r", ""],
        )

        if choice == "0":
            return
        elif choice == "1":
            _run_lstm_standalone()
        elif choice == "2":
            _run_rl_demo()
        elif choice == "3":
            _view_lstm_results()
        elif choice == "4":
            _view_rl_results()
        elif choice == "5":
            _open_notebook_from_dl("3")
        elif choice == "6":
            _open_notebook_from_dl("2")
        elif choice == "7":
            _open_notebook_from_dl("4")
        elif choice == "8":
            _browse_figures()
        elif choice == "9":
            _view_evaluation_metrics()
        elif choice == "r":
            _view_text_report()


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
        console.print("    [bold green]1[/bold green]  [bold]Data Collection & Sources[/bold]")
        console.print(
            "       [dim]Collect from APIs, manual data copy, verify dates[/dim]"
        )
        console.print()
        console.print("    [bold yellow]2[/bold yellow]  [bold]Data Processing[/bold]")
        console.print(
            "       [dim]Clean, merge, feature engineering, outlier detection[/dim]"
        )
        console.print()
        console.print("    [bold magenta]3[/bold magenta]  [bold]ML Training & Notebooks[/bold]")
        console.print(
            "       [dim]Train models, evaluate, launch Jupyter notebooks[/dim]"
        )
        console.print()
        console.print("    [bold bright_red]4[/bold bright_red]  [bold]Deep Learning & RL[/bold]")
        console.print(
            "       [dim]LSTM training, Q-Learning demo, results exploration[/dim]"
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
            menu_processing()
        elif choice == "3":
            menu_training()
        elif choice == "4":
            menu_deep_learning()


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
