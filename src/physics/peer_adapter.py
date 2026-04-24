"""
src/physics/peer_adapter.py — PEER NGA-West2 .AT2 record adapter
================================================================
Parses ground-motion records in the PEER NGA-West2 format (`.AT2` files),
extracts the acceleration time history, optionally resamples it to a
target frequency (Nyquist-consistent linear interpolation), and provides
a PGA scaler for incremental dynamic analysis.

Typical NGA-West2 file layout:
    Line 1: Organization header (e.g. PEER NGA STRONG MOTION DATABASE RECORD)
    Line 2: Earthquake metadata (date, magnitude, station)
    Line 3: Data type (ACCELERATION) and units (G or cm/s^2)
    Line 4: NPTS=  [N] , DT=  [dt] SEC
    Line 5+: Acceleration values (whitespace-separated, may wrap across lines)

Dependencies: numpy, scipy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# Linear interpolation satisfies Nyquist for band-limited signals whose
# sampling rate already resolves the dominant frequency content.
_INTERP_KIND = "linear"


class PeerAdapter:
    def __init__(self, target_frequency_hz: float = 100.0):
        """
        :param target_frequency_hz: Resampling rate (Hz). Use the acquisition
            rate of the downstream consumer (e.g. Arduino, FEM solver).
        """
        self.target_dt = 1.0 / target_frequency_hz
        self.target_freq = target_frequency_hz

    def read_at2_file(self, filepath: Path) -> dict:
        """Parse an NGA-West2 `.AT2` file and return metadata + acceleration array."""
        if not filepath.exists():
            raise FileNotFoundError(f"[PEER] File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) < 5:
            raise ValueError("[PEER] Invalid or empty .AT2 file format.")

        header_eq = lines[1].strip()

        # Parse NPTS and DT from line 4.
        line4 = lines[3].upper()
        try:
            parts = line4.split(",")
            npts_part = parts[0].split("=")[1].strip()
            dt_part = parts[1].split("=")[1].replace("SEC", "").strip()
            npts = int(npts_part)
            dt = float(dt_part)
        except (ValueError, IndexError, AttributeError) as e:
            raise ValueError(
                f"[PEER] Could not parse NPTS/DT on line 4: {line4} -> {e}"
            )

        # Extract acceleration array (whitespace-separated, may wrap).
        accel_data_g: list[float] = []
        skipped_tokens = 0
        for line in lines[4:]:
            for v in line.strip().split():
                try:
                    accel_data_g.append(float(v))
                except ValueError:
                    skipped_tokens += 1  # trailing text tokens in some files
        if skipped_tokens > 10:
            print(
                f"[PEER] WARNING: {skipped_tokens} non-numeric tokens skipped "
                f"in {filepath.name}",
                file=sys.stderr,
            )

        if len(accel_data_g) == 0:
            raise ValueError("[PEER] No acceleration data extracted.")

        print(
            f"[PEER] Loaded: {header_eq[:50]}... | "
            f"points: {len(accel_data_g)} | dt: {dt}s"
        )

        return {
            "metadata": header_eq,
            "npts_original": len(accel_data_g),
            "dt_original": dt,
            "acceleration_g": np.array(accel_data_g, dtype=np.float64),
        }

    def normalize_and_resample(self, raw_data_dict: dict) -> np.ndarray:
        """Resample the record to `self.target_dt` via linear interpolation.

        The declared NPTS in the header sometimes differs from the actual
        number of values; we truncate to the shorter of the two.
        """
        dt_orig = raw_data_dict["dt_original"]
        npts_orig = raw_data_dict["npts_original"]
        accel_orig = raw_data_dict["acceleration_g"]

        time_orig = np.arange(0, npts_orig * dt_orig, dt_orig)

        min_len = min(len(time_orig), len(accel_orig))
        time_orig = time_orig[:min_len]
        accel_orig = accel_orig[:min_len]

        duration = time_orig[-1]
        time_target = np.arange(0, duration, self.target_dt)

        interpolator = interp1d(
            time_orig, accel_orig, kind=_INTERP_KIND, fill_value="extrapolate"
        )
        accel_resampled = interpolator(time_target)

        print(
            f"[PEER] Resample: {dt_orig}s ({1/dt_orig:.1f}Hz) -> "
            f"{self.target_dt}s ({self.target_freq}Hz)"
        )
        print(
            f"[PEER] Points adjusted: {len(accel_orig)} -> {len(accel_resampled)}"
        )

        return accel_resampled

    def scale_to_pga(
        self, accel_array: np.ndarray, target_pga_g: float
    ) -> np.ndarray:
        """Scale the time history so its PGA equals `target_pga_g` (linear scaling)."""
        current_peak = float(np.max(np.abs(accel_array)))
        if current_peak == 0:
            raise ValueError(
                "[PEER] Cannot scale an empty or flat array (current PGA = 0)."
            )

        scale_factor = target_pga_g / current_peak
        scaled_array = accel_array * scale_factor

        print(
            f"[PEER] PGA scaler: current={current_peak:.3f}g "
            f"target={target_pga_g:.3f}g factor={scale_factor:.3f}"
        )
        return scaled_array


if __name__ == "__main__":
    print("PEER NGA-West2 .AT2 adapter — import `PeerAdapter` to use.")
