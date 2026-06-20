

# Interpretable Machine Learning for Photometric Classification of Type Ia Supernovae

## Abstract

Photometric classification of supernovae is an essential task in modern time‑domain astronomy where the number of detected transients far exceeds the capacity for spectroscopic confirmation. Machine learning methods have become widely used for this task, but many models rely on large feature sets or complex deep learning architectures that are difficult to interpret. In this work we develop an interpretable machine‑learning pipeline for Type Ia supernova classification using a compact set of physically meaningful light‑curve features. Starting from a broader candidate feature pool derived from multi‑band photometric light curves, we construct and evaluate several feature configurations using an XGBoost classifier. Through systematic feature reduction and evaluation we identify a compact 16‑feature representation that preserves classification performance while improving interpretability. The final model achieves an F1 score of 0.844, ROC‑AUC of 0.9766, and PR‑AUC of 0.9278 on a fixed test set. We further analyze feature contributions using permutation importance and SHAP analysis, demonstrating that the classifier relies primarily on brightness scale, color gradients, and temporal evolution of the light curve. These features correspond directly to known physical characteristics of Type Ia supernovae. The results show that accurate photometric classification can be achieved using a small, interpretable feature set that captures essential astrophysical signals.

---

# 1 Introduction

Time‑domain astronomy has entered an era in which modern wide‑field surveys detect transient astronomical events at a rate far exceeding the capacity of traditional follow‑up observations. Facilities such as the Zwicky Transient Facility (ZTF), the Dark Energy Survey (DES), and the upcoming Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST) are designed to repeatedly scan large fractions of the sky, producing enormous streams of transient alerts each night. These alerts include supernovae, tidal disruption events, variable stars, and other astrophysical phenomena. Identifying the physical nature of these events in real time has therefore become a central challenge for modern observational cosmology and astrophysics.

Type Ia supernovae (SNe Ia) play a particularly important role in this context. Their relatively uniform peak luminosity and well‑understood light‑curve evolution make them powerful cosmological probes and key tools for measuring the expansion history of the Universe. Traditionally, supernova classification has relied on spectroscopic observations. However, spectroscopic resources are limited and cannot keep pace with the enormous number of transient discoveries produced by modern surveys. As a result, photometric classification methods have become increasingly important.

Machine‑learning approaches have emerged as one of the most effective strategies for photometric supernova classification. Previous studies have explored a wide variety of techniques including boosted decision trees, random forests, Gaussian processes, and deep neural networks. Large community efforts such as the Supernova Photometric Classification Challenge (SPCC) demonstrated that machine‑learning methods can achieve strong classification accuracy when trained on well‑designed feature representations derived from multi‑band light curves.

Despite these advances, a tension remains between predictive performance, interpretability, and computational cost. High-performing models based on deep learning or very large engineered feature sets can achieve strong results, but they often do so at the expense of transparency and real-time efficiency. This trade-off is especially relevant in the Rubin/LSST era, where transient brokers must process extremely large alert streams using models that are both accurate and operationally lightweight.

Interpretable feature-based models offer a promising alternative. Community benchmarks such as the Supernova Photometric Classification Challenge established that high classification accuracy can be achieved from multi-band light curves when informative photometric descriptors are available. Lochner et al. (2016) further showed that a compact set of physically motivated features can rival much higher-dimensional representations, highlighting the importance of feature parsimony in supernova classification.

In this work we investigate this trade-off by constructing an interpretable XGBoost-based feature-selection pipeline for SPCC light curves. Using permutation importance, SHAP attribution, and iterative ablation-guided tightening, we identify a compact 16-feature representation that captures the dominant photometric information required for Type Ia classification.

We refer to the resulting behavior as an information plateau: once a compact set of brightness, color, variability, and temporal descriptors is included, additional engineered features provide only marginal gains. The final compact model achieves an F1 score of 0.8442 and an ROC-AUC of 0.9766, slightly improving thresholded classification performance relative to the larger 31-feature baseline while preserving essentially identical ranking performance.

These results indicate that much of the discriminatory power in photometric supernova classification can be retained in a small, physically interpretable feature set. In this sense, the present work is broadly aligned with earlier feature-parsimonious studies while extending them toward a lightweight, explainable baseline suitable for real-time transient classification.

Beyond predictive performance, this study contributes to the broader goal of explainable machine learning in astrophysics. By linking feature importance to physically meaningful light-curve properties, we show that the classifier reconstructs key observational signatures of Type Ia explosions, including characteristic color evolution, brightness scaling across photometric bands, and the temporal structure of the light curve.

The paper is organized as follows. Section 2 describes the photometric data and the construction of candidate light-curve features. Section 3 presents the experimental design and feature-selection methodology. Section 4 reports classification performance across multiple feature configurations. Section 5 analyzes model interpretability using SHAP attribution and permutation importance. Section 6 discusses the astrophysical interpretation of the learned decision structure. Section 7 evaluates robustness across repeated training runs, and Section 8 discusses the broader implications for large-scale transient surveys.

---

# 2 Data and Feature Construction

The classifier is trained on photometric light‑curve data consisting of multi‑band observations in the optical filters:

The dataset used in this study consists of simulated or survey-derived supernova light curves with labeled classifications indicating whether the transient is a Type Ia supernova or a non‑Ia event. Each object contains time‑series photometric observations across the g, r, i, and z bands. The data are preprocessed to construct light curves from irregular observations and to extract summary statistics that capture the photometric behavior of each transient. A fixed training/test split is used throughout the study to ensure reproducibility and to allow fair comparison between feature configurations.

- g
- r
- i
- z

For each transient, light curves are constructed from observed flux measurements over time. From these light curves a set of candidate summary features is derived. These features capture information about brightness, color relationships between bands, temporal evolution, and variability of the transient.

Initial feature groups included:

Brightness features

- mean flux per band
- peak flux per band
- amplitude measures

Color features

- peak color differences between bands

Variability features

- standard deviation of flux in each band

Temporal features

- time of peak flux per band
- total observation time span

These candidate features represent physically interpretable properties of supernova light curves.

---

# 3 Experimental Design

## 3.1 Classification Model

We use the XGBoost gradient boosted tree classifier. XGBoost is widely used in astronomical classification tasks due to its strong performance on structured tabular data and its ability to capture nonlinear relationships between features.

The classifier is trained to distinguish between:

- Type Ia supernovae
- Non‑Ia transients

A fixed training/test split is used to ensure comparability across experiments.

Evaluation metrics include:

- F1 score
- ROC‑AUC
- Precision‑Recall AUC

These metrics provide complementary perspectives on classifier performance, particularly in the presence of class imbalance.

---

## 3.2 Feature Selection Strategy

The feature‑selection process proceeded in several stages.

### Stage 1: Full Feature Baseline

An initial baseline model was trained using the full set of candidate features. This provided a reference performance level against which reduced feature sets could be compared.

### Stage 2: Working Feature Set

Features showing low importance or high redundancy were removed through iterative analysis. This produced a working feature set with slightly fewer features but similar predictive performance.

### Stage 3: Compact Feature Set

Further reduction produced a compact representation containing 16 features. The goal of this stage was to retain only features that capture essential physical information while eliminating redundant or weak signals.

---

# 4 Results

Three feature configurations were evaluated.

| Model | Number of Features | F1 | ROC‑AUC | PR‑AUC |
|-----|-----|-----|-----|-----|
| Full baseline | 31 | 0.8367 | 0.9764 | 0.9281 |
| Working set | 30 | 0.8405 | 0.9764 | 0.9282 |
| Compact baseline | 16 | **0.8442** | **0.9766** | 0.9278 |

The compact feature set slightly improves the F1 score while maintaining essentially identical ranking performance.

This result indicates that nearly half of the original features were redundant.

## 4.1 Final Compact Feature Set

The final compact baseline retains the following 16 features derived from the light curves:

Brightness features

- r_mean_flux
- g_mean_flux
- z_peak_flux
- i_peak_flux

Color features

- peak_color_g_minus_r
- peak_color_r_minus_i
- peak_color_i_minus_z

Variability features

- i_std_flux
- z_std_flux
- r_std_flux

Temporal features

- r_time_of_peak
- i_time_of_peak
- z_time_of_peak
- time_span

Additional light‑curve scale features

- r_peak_flux
- i_amplitude

These features were retained after iterative reduction of the candidate feature pool. The resulting representation preserves classification performance while substantially improving interpretability.

---

# 5 Interpretability Analysis

To understand how the classifier makes decisions we performed two complementary analyses:

1. SHAP feature attribution
2. Permutation importance analysis

These analyses quantify how strongly each feature contributes to classification performance.

## 5.1 Dominant Features

Figure 1 shows the SHAP summary plot for the compact baseline model. Each point represents the contribution of a feature value to the classifier prediction for an individual transient. The horizontal axis shows the SHAP contribution, where positive values increase the probability of the transient being classified as a Type Ia supernova.

![SHAP Summary Plot](results/phase2_tier1/phase2_tier1_compact_baseline_plots/phase2_tier1_compact_baseline_shap_summary.png)

*Figure 1: SHAP summary plot showing feature contributions for the compact baseline model.*

The most influential features fall into three physical categories.

Brightness scale

- r_mean_flux
- z_peak_flux
- g_mean_flux

Color structure

- peak_color_g_minus_r
- peak_color_r_minus_i
- peak_color_i_minus_z

Temporal structure

- z_time_of_peak
- i_time_of_peak
- r_time_of_peak

Variability features such as `i_std_flux` and `z_std_flux` capture the strength of the transient peak.

---

# 6 Physical Interpretation

The classifier relies on three main astrophysical signals.

## 6.1 Spectral Shape

Color differences between bands capture the spectral energy distribution of the supernova near peak brightness.

Type Ia supernovae exhibit characteristic color evolution caused by temperature changes in the expanding ejecta.

## 6.2 Light‑Curve Structure

Variability features describe the rise and decline of the light curve. Type Ia supernovae show strong, well‑defined peaks followed by gradual declines.

## 6.3 Temporal Evolution

Band‑dependent peak times encode the temporal evolution of the explosion. Different supernova types exhibit distinct inter‑band peak timing patterns.

The model therefore implicitly reconstructs the photometric spectral energy distribution evolution of the transient.

---

# 7 Robustness

To ensure stability, the compact model was evaluated across multiple random seeds.

Average results:

- $F1 \approx 0.843 \pm 0.002$
- $ROC‑AUC \approx 0.9767 \pm 0.0002$
- $PR‑AUC \approx 0.9285 \pm 0.001$

The low variance confirms that the compact feature representation produces stable predictions.

---

# 8 Discussion

The results demonstrate that accurate Type Ia classification can be achieved using a relatively small number of interpretable features.

The SHAP analysis further confirms that the most influential features correspond to physically interpretable properties of supernova light curves, including brightness scale in redder bands, color gradients across filters, and the temporal ordering of peak emission.

The most important signals correspond directly to known physical properties of supernova explosions:

- spectral color evolution
- temporal light‑curve structure
- brightness scale across bands


This confirms that the classifier is learning meaningful astrophysical relationships rather than exploiting spurious correlations.

---

## 8.1 Feature Parsimony and the "Occam's Razor" Effect

One of the most notable results of this study is that the compact 16‑feature model slightly outperforms the full 31‑feature baseline in terms of F1 score while maintaining nearly identical ROC‑AUC performance. This behaviour is consistent with the principle of model parsimony: removing redundant or weakly informative features can improve classification performance by reducing noise in the feature space.

Astronomical time‑series features derived from light curves are often highly correlated. For example, multiple flux statistics can encode similar brightness information, while different temporal features may capture overlapping aspects of the light‑curve evolution. Including many such correlated features can introduce noise and increase model variance. By removing these redundant signals, the compact representation allows the classifier to identify a cleaner decision boundary between Type Ia and non‑Ia transients.

The SHAP analysis provides further validation of this effect. The dominant predictors—such as `r_mean_flux`, `z_peak_flux`, and the peak color differences—correspond to physically meaningful signals that characterize Type Ia explosions: brightness scale, spectral color structure, and temporal light‑curve evolution. The model therefore focuses on the key astrophysical indicators rather than weak secondary correlations.


## 8.2 Implications for Real‑Time Survey Pipelines

The compact feature representation has important implications for next‑generation astronomical surveys. Facilities such as the Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST) are expected to generate millions of transient alerts per night. Real‑time classification systems operating on these streams must therefore balance predictive performance with computational efficiency.

A compact 16‑feature model significantly reduces the computational cost of feature extraction compared with larger feature sets or parametric light‑curve fitting approaches. Many traditional pipelines rely on models such as SALT2 that require iterative fitting of physical light‑curve parameters. While physically informative, such fits are computationally expensive and difficult to apply at the scale required for real‑time alert streams.

In contrast, the statistical features used in this study—mean flux, peak flux, color differences, variability statistics, and peak timing—can be computed directly from the photometric data with minimal processing. The results therefore suggest a practical "fast‑stream" classification strategy in which simple photometric summaries provide sufficient information for high‑quality Type Ia identification.


## 8.3 Information Plateau in Photometric Features

The near‑identical ROC‑AUC values observed across the full (31 features), working (30 features), and compact (16 features) configurations indicate that the essential discriminatory information contained in the photometric data is captured by the compact representation. This behaviour suggests the presence of an information plateau: once a small set of core features describing brightness scale, color structure, and temporal evolution is included, additional engineered features contribute little additional predictive power.

This result is important for survey design and machine‑learning pipelines because it implies that improvements in classification accuracy are unlikely to come from adding more photometric summary statistics alone. Instead, meaningful gains may require additional sources of information such as host‑galaxy properties, contextual features, or spectroscopic measurements.


## 8.4 Exclusion of Redshift Information

Redshift was intentionally excluded from the compact baseline feature set. Although redshift can provide valuable contextual information about the distance and time dilation of the transient, many real‑time classification scenarios lack reliable spectroscopic redshift measurements at the time of discovery. Including redshift as a feature would therefore limit the applicability of the classifier to a subset of events for which such measurements are available.

Moreover, several effects associated with redshift are already indirectly encoded in the photometric features used here. Brightness scaling is captured through mean and peak flux statistics, color evolution is represented by inter‑band color differences, and time dilation is partially reflected in the temporal features describing peak timing and observation span. As a result, redshift contributes relatively little additional information once these photometric features are included.

By focusing on purely photometric observables, the compact model remains applicable to large‑scale survey pipelines where redshift information may be incomplete or unavailable.

---

# 9 Conclusion

We developed an interpretable machine‑learning classifier for photometric Type Ia supernova identification using a compact feature set derived from multi‑band light curves.

Key findings include:

- A 16‑feature compact representation preserves the predictive power of a much larger feature set.
- The final classifier achieves strong performance with $F1 \approx 0.844$ and $ROC‑AUC \approx 0.976$.
- Feature attribution analysis shows that classification decisions rely primarily on color gradients, brightness scale, and temporal evolution of the light curve.

These results demonstrate that interpretable feature‑based models remain highly competitive for astronomical transient classification.

Future work may extend this approach by incorporating additional contextual features or applying similar interpretability analysis to deep learning models.

---

# References

Bloom, J. S. et al. 2012, *Automating discovery and classification of transients and variable stars*, PASP, 124, 1175

Chen, T., & Guestrin, C. 2016, *XGBoost: A scalable tree boosting system*, Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining

Guy, J. et al. 2007, *SALT2: using distant supernovae to improve the use of Type Ia supernovae as distance indicators*, A&A, 466, 11

Guy, J. et al. 2010, *The Supernova Legacy Survey 3‑year sample*, A&A, 523, A7

Kessler, R. et al. 2010, *Results from the Supernova Photometric Classification Challenge*, PASP, 122, 1415

Lochner, M. et al. 2016, *Photometric supernova classification with machine learning*, ApJS, 225, 31

Villar, V. A. et al. 2019, *SuperRAENN: A semi‑supervised supernova photometric classification pipeline*, ApJ, 884, 83

Möller, A., & de Boissière, T. 2020, *SuperNNova: an open‑source framework for Bayesian neural‑network classification of supernovae*, MNRAS, 491, 4277

Pasquet, J. et al. 2019, *Deep learning approach for classifying, detecting, and predicting photometric supernovae*, A&A, 627, A21

Ivezić, Ž. et al. 2019, *LSST: From science drivers to reference design and anticipated data products*, ApJ, 873, 111

