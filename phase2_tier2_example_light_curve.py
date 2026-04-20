"""Generate a paper-ready example SPCC multi-band light curve."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


TIER2_RESULTS_DIR = "results/phase2_tier2"
DEFAULT_SNID = 319694
BAND_COLORS = {
    "g": "#2ca02c",
    "r": "#d62728",
    "i": "#9467bd",
    "z": "#8c564b",
}


def ensure_results_dir() -> None:
    os.makedirs(TIER2_RESULTS_DIR, exist_ok=True)


def raw_path_for_snid(snid: int) -> Path:
    return Path(f"data/spcc/raw/DES_SN{snid:06d}.DAT")


def parse_spcc_dat(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {"source_file": str(path)}
    observations: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("SNID:"):
                metadata["snid"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("SIM_COMMENT:") and "SN Type =" in line:
                metadata["sim_type"] = line.split("SN Type =", 1)[1].split(",", 1)[0].strip()
            elif line.startswith("SIM_REDSHIFT:"):
                metadata["sim_redshift"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("SIM_PEAKMJD:"):
                metadata["sim_peak_mjd"] = float(line.split(":", 1)[1].split()[0])
            elif line.startswith("NOBS:"):
                metadata["nobs_header"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("OBS:"):
                fields = line.split(":", 1)[1].split()
                observations.append(
                    {
                        "mjd": float(fields[0]),
                        "band": fields[1],
                        "fluxcal": float(fields[3]),
                        "fluxcalerr": float(fields[4]),
                        "snr": float(fields[5]),
                        "mag": float(fields[6]),
                        "magerr": float(fields[7]),
                        "sim_mag": float(fields[8]),
                    }
                )
    if not observations:
        raise ValueError(f"No observations found in {path}.")
    metadata["nobs_parsed"] = len(observations)
    metadata["bands"] = sorted({obs["band"] for obs in observations})
    metadata["mjd_min"] = min(obs["mjd"] for obs in observations)
    metadata["mjd_max"] = max(obs["mjd"] for obs in observations)
    metadata["time_span_days"] = metadata["mjd_max"] - metadata["mjd_min"]
    return metadata, observations


def write_observation_csv(path: str, metadata: dict[str, Any], observations: list[dict[str, Any]]) -> None:
    first_mjd = metadata["mjd_min"]
    fieldnames = ["snid", "phase_days", "mjd", "band", "fluxcal", "fluxcalerr", "snr", "mag", "magerr", "sim_mag"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for obs in observations:
            writer.writerow({"snid": metadata["snid"], "phase_days": obs["mjd"] - first_mjd, **obs})


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def plot_light_curve(
    metadata: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    output_png: str,
    output_pdf: str,
) -> None:
    import matplotlib.pyplot as plt

    first_mjd = metadata["mjd_min"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for band in ["g", "r", "i", "z"]:
        band_obs = [obs for obs in observations if obs["band"] == band]
        if not band_obs:
            continue
        phase = [obs["mjd"] - first_mjd for obs in band_obs]
        flux = [obs["fluxcal"] for obs in band_obs]
        flux_err = [obs["fluxcalerr"] for obs in band_obs]
        ax.errorbar(
            phase,
            flux,
            yerr=flux_err,
            fmt="o",
            markersize=3.6,
            linewidth=1.0,
            capsize=1.8,
            color=BAND_COLORS[band],
            label=f"{band} band",
            alpha=0.88,
        )

    peak_mjd = metadata.get("sim_peak_mjd")
    if peak_mjd is not None:
        ax.axvline(peak_mjd - first_mjd, color="black", linestyle="--", linewidth=1.1, alpha=0.75, label="simulated peak MJD")

    ax.axhline(0.0, color="0.25", linewidth=0.8, alpha=0.55)
    ax.set_xlabel("Days since first observation")
    ax.set_ylabel("Fluxcal")
    ax.set_title(f"Example SPCC Type Ia light curve: SNID {metadata['snid']}")
    subtitle = f"z = {metadata.get('sim_redshift', 'unknown')}, observations = {metadata['nobs_parsed']}"
    ax.text(0.01, 0.98, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snid", type=int, default=DEFAULT_SNID)
    args = parser.parse_args()

    ensure_results_dir()
    raw_path = raw_path_for_snid(args.snid)
    if not raw_path.exists():
        raise FileNotFoundError(f"Could not find raw SPCC file for SNID {args.snid}: {raw_path}")

    metadata, observations = parse_spcc_dat(raw_path)
    stem = f"example_light_curve_sn{args.snid:06d}"
    output_png = f"{TIER2_RESULTS_DIR}/{stem}.png"
    output_pdf = f"{TIER2_RESULTS_DIR}/{stem}.pdf"
    output_csv = f"{TIER2_RESULTS_DIR}/{stem}.csv"
    output_json = f"{TIER2_RESULTS_DIR}/{stem}.json"

    plot_light_curve(metadata, observations, output_png=output_png, output_pdf=output_pdf)
    write_observation_csv(output_csv, metadata, observations)
    write_json(
        output_json,
        {
            "artifact": "phase2_tier2_example_light_curve",
            "metadata": metadata,
            "outputs": {
                "png": output_png,
                "pdf": output_pdf,
                "csv": output_csv,
            },
            "caption_note": "Example simulated SPCC Type Ia multi-band light curve. Points show observed FLUXCAL values with FLUXCALERR uncertainties in g, r, i, and z bands.",
        },
    )
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
