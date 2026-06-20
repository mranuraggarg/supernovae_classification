"""Explicit registry for owned native SPCC features.

Each feature is defined here so the project can later accept or reject features based
on demonstrated training importance rather than inherited defaults.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    definition: str
    source: str
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


FEATURE_REGISTRY = [
    FeatureSpec(
        name="observation_count",
        group="coverage",
        definition="Total number of raw observations for the supernova after cleaning.",
        source="native_owned",
        status="candidate",
    ),
    FeatureSpec(
        name="time_span",
        group="coverage",
        definition="Maximum normalized observation time minus minimum normalized observation time.",
        source="native_owned",
        status="candidate",
    ),
    FeatureSpec(
        name="observed_band_count",
        group="coverage",
        definition="Count of distinct passbands observed across g, r, i, z.",
        source="native_owned",
        status="candidate",
    ),
    FeatureSpec(
        name="peak_flux_all",
        group="global_shape",
        definition="Log-compressed global peak flux stored as log10(1 + max(global peak flux, 0)).",
        source="native_owned",
        status="candidate",
    ),
    FeatureSpec(
        name="time_of_peak_all",
        group="global_shape",
        definition="Normalized time at which the global peak flux occurs.",
        source="native_owned",
        status="candidate",
    ),
    FeatureSpec(
        name="amplitude_all",
        group="global_shape",
        definition="Log-compressed global amplitude stored as log10(1 + max(global peak flux - global minimum flux, 0)).",
        source="native_owned",
        status="candidate",
    ),
    FeatureSpec(
        name="mean_flux_all",
        group="global_shape",
        definition="Signed-log mean reconstructed flux stored as sign(value) * log10(1 + abs(value)).",
        source="native_owned",
        status="candidate",
    ),
    FeatureSpec(
        name="std_flux_all",
        group="global_shape",
        definition="Log-compressed standard deviation of reconstructed flux stored as log10(1 + std).",
        source="native_owned",
        status="candidate",
    ),
    FeatureSpec(
        name="total_snr",
        group="quality",
        definition="Log-compressed total signal-to-noise proxy stored as log10(1 + total_snr).",
        source="native_owned",
        status="candidate",
    ),
]


for band in ("g", "r", "i", "z"):
    FEATURE_REGISTRY.extend(
        [
            FeatureSpec(
                name=f"{band}_peak_flux",
                group="band_shape",
                definition=f"Log-compressed maximum reconstructed {band}-band flux stored as log10(1 + peak flux).",
                source="native_owned",
                status="candidate",
            ),
            FeatureSpec(
                name=f"{band}_time_of_peak",
                group="band_shape",
                definition=f"Normalized time at which reconstructed {band}-band flux reaches its maximum.",
                source="native_owned",
                status="candidate",
            ),
            FeatureSpec(
                name=f"{band}_mean_flux",
                group="band_shape",
                definition=f"Signed-log mean reconstructed {band}-band flux stored as sign(value) * log10(1 + abs(value)).",
                source="native_owned",
                status="candidate",
            ),
            FeatureSpec(
                name=f"{band}_std_flux",
                group="band_shape",
                definition=f"Log-compressed standard deviation of reconstructed {band}-band flux stored as log10(1 + std).",
                source="native_owned",
                status="candidate",
            ),
            FeatureSpec(
                name=f"{band}_amplitude",
                group="band_shape",
                definition=f"Log-compressed reconstructed {band}-band amplitude stored as log10(1 + max peak-to-trough amplitude).",
                source="native_owned",
                status="candidate",
            ),
        ]
    )


FEATURE_REGISTRY.extend(
    [
        FeatureSpec(
            name="peak_color_g_minus_r",
            group="color_proxy",
            definition="Magnitude-style color proxy computed as -2.5 * log10(g-band peak flux / r-band peak flux), then clipped to [-5, 5].",
            source="native_owned",
            status="candidate",
        ),
        FeatureSpec(
            name="peak_color_r_minus_i",
            group="color_proxy",
            definition="Magnitude-style color proxy computed as -2.5 * log10(r-band peak flux / i-band peak flux), then clipped to [-5, 5].",
            source="native_owned",
            status="candidate",
        ),
        FeatureSpec(
            name="peak_color_i_minus_z",
            group="color_proxy",
            definition="Magnitude-style color proxy computed as -2.5 * log10(i-band peak flux / z-band peak flux), then clipped to [-5, 5].",
            source="native_owned",
            status="candidate",
        ),
    ]
)
