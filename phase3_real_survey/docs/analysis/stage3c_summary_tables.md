# Stage 3C Summary Tables

All ranges and counts summarize the retained strata without pooling their values.
Sign-count columns are `negative / includes zero / positive` for the 88
redshift-by-SBMAG cells within one survey comparison.

## Data Integrity

| Item | Result |
|---|---|
| Analysis SHA-256 | `ea2a709250782f03e82806eae8c520fb2186bc44387d2b7a16b9f96070423094` |
| Complete design records | 176,000 |
| Bootstrap replicates | 5,000 |
| Undefined-statistic diagnostics | 0 |

## Acceptance by Branch

| Branch | Accepted | Rejected | Cell-fraction range |
|---|---:|---:|---:|
| SDSS-II / 82N | 44,000 | 0 | 100.0% to 100.0% |
| SDSS-II / 82S | 44,000 | 0 | 100.0% to 100.0% |
| DES-SN5YR / SHALLOW | 42,375 | 1,625 | 95.6% to 97.0% |
| DES-SN5YR / DEEP | 42,447 | 1,553 | 95.6% to 97.2% |

## Rejection Mechanisms

| Mechanism | Records | Fraction of rejections |
|---|---:|---:|
| missing usable g band | 1,630 | 51.29% |
| missing usable r band | 735 | 23.13% |
| missing usable z band | 634 | 19.95% |
| missing usable i band | 167 | 5.25% |
| nonpositive colour representative g | 6 | 0.19% |
| nonpositive colour representative i | 2 | 0.06% |
| nonpositive colour representative r | 2 | 0.06% |
| nonpositive colour representative z | 2 | 0.06% |

## Feature Results by Registered Comparison

| Feature | Comparison | Median range | q10 range | q90 range | W1 range | Pointwise sign counts | Simultaneous sign counts |
|---|---|---:|---:|---:|---:|---:|---:|
| `peak_color_r_minus_i` | SHALLOW - 82N | -0.1562 to 0.0149 | -0.1683 to 0.0293 | -0.1675 to 0.0307 | 0.0193 to 0.1568 | 80/0/8 | 80/1/7 |
| `peak_color_r_minus_i` | SHALLOW - 82S | -0.1557 to 0.0160 | -0.1751 to 0.0316 | -0.1719 to 0.0334 | 0.0172 to 0.1608 | 80/0/8 | 80/4/4 |
| `peak_color_r_minus_i` | DEEP - 82N | -0.1507 to 0.0204 | -0.2017 to 0.0010 | -0.1164 to 0.1210 | 0.0606 to 0.1811 | 80/0/8 | 80/1/7 |
| `peak_color_r_minus_i` | DEEP - 82S | -0.1527 to 0.0164 | -0.2050 to -0.0008 | -0.1085 to 0.1194 | 0.0615 to 0.1837 | 80/0/8 | 80/0/8 |
| `peak_color_i_minus_z` | SHALLOW - 82N | -0.2156 to 0.0795 | -0.1824 to 0.2376 | -0.2756 to -0.0491 | 0.0572 to 0.2177 | 56/6/26 | 56/11/21 |
| `peak_color_i_minus_z` | SHALLOW - 82S | -0.2248 to 0.0673 | -0.1828 to 0.2122 | -0.2988 to -0.0629 | 0.0634 to 0.2288 | 57/21/10 | 56/24/8 |
| `peak_color_i_minus_z` | DEEP - 82N | -0.2215 to 0.0665 | -0.2062 to 0.1806 | -0.2580 to -0.0326 | 0.0608 to 0.2371 | 56/14/18 | 56/18/14 |
| `peak_color_i_minus_z` | DEEP - 82S | -0.2334 to 0.0534 | -0.2128 to 0.1623 | -0.2755 to -0.0528 | 0.0609 to 0.2472 | 61/19/8 | 59/21/8 |
| `i_peak_flux` | SHALLOW - 82N | -0.0866 to -0.0040 | -0.0713 to 0.0060 | -0.1082 to -0.0020 | 0.0037 to 0.0874 | 88/0/0 | 86/2/0 |
| `i_peak_flux` | SHALLOW - 82S | -0.0875 to -0.0044 | -0.0728 to 0.0065 | -0.1063 to -0.0024 | 0.0052 to 0.0874 | 88/0/0 | 87/1/0 |
| `i_peak_flux` | DEEP - 82N | -0.0887 to -0.0045 | -0.0669 to 0.0037 | -0.1172 to -0.0153 | 0.0111 to 0.0933 | 88/0/0 | 88/0/0 |
| `i_peak_flux` | DEEP - 82S | -0.0895 to -0.0067 | -0.0708 to 0.0033 | -0.1152 to -0.0174 | 0.0118 to 0.0942 | 88/0/0 | 88/0/0 |
| `z_peak_flux` | SHALLOW - 82N | -0.1172 to -0.0621 | -0.0743 to 0.0231 | -0.1809 to -0.1306 | 0.0696 to 0.1200 | 88/0/0 | 88/0/0 |
| `z_peak_flux` | SHALLOW - 82S | -0.1236 to -0.0722 | -0.0772 to 0.0177 | -0.1926 to -0.1410 | 0.0795 to 0.1283 | 88/0/0 | 88/0/0 |
| `z_peak_flux` | DEEP - 82N | -0.1182 to -0.0644 | -0.0745 to 0.0175 | -0.1926 to -0.1474 | 0.0791 to 0.1285 | 88/0/0 | 88/0/0 |
| `z_peak_flux` | DEEP - 82S | -0.1248 to -0.0739 | -0.0764 to 0.0087 | -0.1981 to -0.1562 | 0.0873 to 0.1362 | 88/0/0 | 88/0/0 |
| `r_time_of_peak` | SHALLOW - 82N | 0.2100 to 2.9555 | -6.8518 to 0.8735 | 612.0742 to 934.1562 | 135.8041 to 206.7461 | 0/8/80 | 0/45/43 |
| `r_time_of_peak` | SHALLOW - 82S | 0.0960 to 2.8140 | -6.8638 to 0.8839 | 612.0034 to 934.2204 | 134.9318 to 206.0793 | 0/8/80 | 0/41/47 |
| `r_time_of_peak` | DEEP - 82N | 2.0690 to 4.9780 | 0.0249 to 6.0238 | 443.1702 to 983.3020 | 130.5394 to 226.4601 | 0/0/88 | 0/0/88 |
| `r_time_of_peak` | DEEP - 82S | 2.0505 to 4.9520 | 0.9847 to 6.0048 | 443.0798 to 984.2772 | 130.5293 to 227.1746 | 0/0/88 | 0/0/88 |
| `time_span` | SHALLOW - 82N | 50.7965 to 75.5540 | 3.9590 to 26.1148 | 999.4085 to 1149.4131 | 295.7689 to 412.2780 | 0/0/88 | 0/44/44 |
| `time_span` | SHALLOW - 82S | 52.8215 to 79.9190 | 7.0310 to 24.0271 | 1000.4394 to 1148.4432 | 298.2716 to 414.1988 | 0/0/88 | 0/44/44 |
| `time_span` | DEEP - 82N | 56.9550 to 78.9880 | 17.9210 to 42.2052 | 994.2551 to 1162.9939 | 281.3735 to 420.9979 | 0/0/88 | 0/0/88 |
| `time_span` | DEEP - 82S | 58.9845 to 82.9250 | 20.9640 to 47.2793 | 995.2273 to 1166.1050 | 283.8432 to 424.6526 | 0/0/88 | 0/0/88 |

## Key Redshift Envelopes

Each envelope is the minimum and maximum among the 32 unpooled
comparison-by-SBMAG coordinates at that redshift.

| Feature | z | Median envelope | Pointwise negative / zero / positive |
|---|---:|---:|---:|
| `peak_color_r_minus_i` | 0.050 | -0.1021 to -0.0830 | 32 / 0 / 0 |
| `peak_color_r_minus_i` | 0.075 | -0.1495 to -0.1305 | 32 / 0 / 0 |
| `peak_color_r_minus_i` | 0.100 | -0.1562 to -0.1369 | 32 / 0 / 0 |
| `peak_color_r_minus_i` | 0.125 | -0.0982 to -0.0801 | 32 / 0 / 0 |
| `peak_color_r_minus_i` | 0.150 | -0.0750 to -0.0636 | 32 / 0 / 0 |
| `peak_color_r_minus_i` | 0.175 | -0.1503 to -0.1393 | 32 / 0 / 0 |
| `peak_color_r_minus_i` | 0.200 | -0.1483 to -0.1404 | 32 / 0 / 0 |
| `peak_color_r_minus_i` | 0.225 | -0.1104 to -0.1003 | 32 / 0 / 0 |
| `peak_color_r_minus_i` | 0.250 | -0.0818 to -0.0671 | 32 / 0 / 0 |
| `peak_color_r_minus_i` | 0.275 | -0.0297 to -0.0176 | 32 / 0 / 0 |
| `peak_color_r_minus_i` | 0.300 | 0.0095 to 0.0204 | 0 / 0 / 32 |
| `peak_color_i_minus_z` | 0.050 | -0.0338 to 0.0229 | 6 / 24 / 2 |
| `peak_color_i_minus_z` | 0.075 | -0.0147 to 0.0513 | 0 / 20 / 12 |
| `peak_color_i_minus_z` | 0.100 | 0.0303 to 0.0795 | 0 / 0 / 32 |
| `peak_color_i_minus_z` | 0.125 | -0.0069 to 0.0388 | 0 / 16 / 16 |
| `peak_color_i_minus_z` | 0.150 | -0.0772 to -0.0293 | 32 / 0 / 0 |
| `peak_color_i_minus_z` | 0.175 | -0.0855 to -0.0264 | 32 / 0 / 0 |
| `peak_color_i_minus_z` | 0.200 | -0.0981 to -0.0568 | 32 / 0 / 0 |
| `peak_color_i_minus_z` | 0.225 | -0.1030 to -0.0607 | 32 / 0 / 0 |
| `peak_color_i_minus_z` | 0.250 | -0.1344 to -0.0813 | 32 / 0 / 0 |
| `peak_color_i_minus_z` | 0.275 | -0.2005 to -0.1563 | 32 / 0 / 0 |
| `peak_color_i_minus_z` | 0.300 | -0.2334 to -0.2027 | 32 / 0 / 0 |
| `i_peak_flux` | 0.050 | -0.0724 to -0.0628 | 32 / 0 / 0 |
| `i_peak_flux` | 0.075 | -0.0802 to -0.0690 | 32 / 0 / 0 |
| `i_peak_flux` | 0.100 | -0.0895 to -0.0791 | 32 / 0 / 0 |
| `i_peak_flux` | 0.125 | -0.0699 to -0.0610 | 32 / 0 / 0 |
| `i_peak_flux` | 0.150 | -0.0554 to -0.0482 | 32 / 0 / 0 |
| `i_peak_flux` | 0.175 | -0.0673 to -0.0597 | 32 / 0 / 0 |
| `i_peak_flux` | 0.200 | -0.0605 to -0.0523 | 32 / 0 / 0 |
| `i_peak_flux` | 0.225 | -0.0527 to -0.0442 | 32 / 0 / 0 |
| `i_peak_flux` | 0.250 | -0.0424 to -0.0348 | 32 / 0 / 0 |
| `i_peak_flux` | 0.275 | -0.0227 to -0.0177 | 32 / 0 / 0 |
| `i_peak_flux` | 0.300 | -0.0096 to -0.0040 | 32 / 0 / 0 |
| `z_peak_flux` | 0.050 | -0.1029 to -0.0744 | 32 / 0 / 0 |
| `z_peak_flux` | 0.075 | -0.0929 to -0.0688 | 32 / 0 / 0 |
| `z_peak_flux` | 0.100 | -0.0879 to -0.0621 | 32 / 0 / 0 |
| `z_peak_flux` | 0.125 | -0.0855 to -0.0625 | 32 / 0 / 0 |
| `z_peak_flux` | 0.150 | -0.1006 to -0.0786 | 32 / 0 / 0 |
| `z_peak_flux` | 0.175 | -0.1171 to -0.0913 | 32 / 0 / 0 |
| `z_peak_flux` | 0.200 | -0.1216 to -0.0988 | 32 / 0 / 0 |
| `z_peak_flux` | 0.225 | -0.1139 to -0.0926 | 32 / 0 / 0 |
| `z_peak_flux` | 0.250 | -0.1160 to -0.1009 | 32 / 0 / 0 |
| `z_peak_flux` | 0.275 | -0.1240 to -0.1076 | 32 / 0 / 0 |
| `z_peak_flux` | 0.300 | -0.1248 to -0.1109 | 32 / 0 / 0 |
| `r_time_of_peak` | 0.050 | 0.0960 to 3.9985 | 0 / 3 / 29 |
| `r_time_of_peak` | 0.075 | 0.1130 to 3.2230 | 0 / 3 / 29 |
| `r_time_of_peak` | 0.100 | 0.9080 to 3.1550 | 0 / 2 / 30 |
| `r_time_of_peak` | 0.125 | 0.9865 to 4.1015 | 0 / 0 / 32 |
| `r_time_of_peak` | 0.150 | 0.8830 to 4.9780 | 0 / 1 / 31 |
| `r_time_of_peak` | 0.175 | 0.4710 to 4.0045 | 0 / 4 / 28 |
| `r_time_of_peak` | 0.200 | 0.8770 to 4.0085 | 0 / 1 / 31 |
| `r_time_of_peak` | 0.225 | 1.1050 to 4.0830 | 0 / 0 / 32 |
| `r_time_of_peak` | 0.250 | 0.8485 to 4.9520 | 0 / 1 / 31 |
| `r_time_of_peak` | 0.275 | 0.9420 to 4.0540 | 0 / 0 / 32 |
| `r_time_of_peak` | 0.300 | 0.8500 to 4.2080 | 0 / 1 / 31 |
| `time_span` | 0.050 | 50.7965 to 60.0375 | 0 / 0 / 32 |
| `time_span` | 0.075 | 55.9480 to 64.0405 | 0 / 0 / 32 |
| `time_span` | 0.100 | 56.0130 to 65.7410 | 0 / 0 / 32 |
| `time_span` | 0.125 | 57.4990 to 68.9175 | 0 / 0 / 32 |
| `time_span` | 0.150 | 57.3850 to 68.9450 | 0 / 0 / 32 |
| `time_span` | 0.175 | 61.0030 to 72.8040 | 0 / 0 / 32 |
| `time_span` | 0.200 | 63.9850 to 74.0305 | 0 / 0 / 32 |
| `time_span` | 0.225 | 65.9495 to 76.0510 | 0 / 0 / 32 |
| `time_span` | 0.250 | 69.0690 to 78.9860 | 0 / 0 / 32 |
| `time_span` | 0.275 | 70.9900 to 80.0545 | 0 / 0 / 32 |
| `time_span` | 0.300 | 71.4085 to 82.9250 | 0 / 0 / 32 |

## Physical Timing Comparisons

| Metric | Comparison | Difference range (days) | Pointwise sign counts | Simultaneous sign counts |
|---|---|---:|---:|---:|
| `signed_median_days` | SHALLOW - 82N | -2.9719 to -1.0325 | 88/0/0 | 88/0/0 |
| `signed_median_days` | SHALLOW - 82S | -2.9580 to -1.0432 | 88/0/0 | 88/0/0 |
| `signed_median_days` | DEEP - 82N | -2.5269 to -0.6224 | 88/0/0 | 83/5/0 |
| `signed_median_days` | DEEP - 82S | -2.5279 to -0.7053 | 88/0/0 | 86/2/0 |
| `median_absolute_error_days` | SHALLOW - 82N | -0.4055 to 1.2555 | 0/16/72 | 0/16/72 |
| `median_absolute_error_days` | SHALLOW - 82S | 0.0514 to 1.2512 | 0/11/77 | 0/16/72 |
| `median_absolute_error_days` | DEEP - 82N | -0.3950 to 1.2442 | 0/16/72 | 0/16/72 |
| `median_absolute_error_days` | DEEP - 82S | 0.0604 to 1.2375 | 0/11/77 | 0/16/72 |
| `mad_days` | SHALLOW - 82N | -1.5304 to -1.0069 | 88/0/0 | 88/0/0 |
| `mad_days` | SHALLOW - 82S | -1.4947 to -1.0051 | 88/0/0 | 88/0/0 |
| `mad_days` | DEEP - 82N | -0.8416 to -0.0334 | 58/30/0 | 30/58/0 |
| `mad_days` | DEEP - 82S | -0.7492 to 0.0791 | 60/28/0 | 22/66/0 |
| `iqr_days` | SHALLOW - 82N | -2.0075 to 0.7497 | 42/46/0 | 18/70/0 |
| `iqr_days` | SHALLOW - 82S | -1.9846 to 0.7413 | 35/53/0 | 12/76/0 |
| `iqr_days` | DEEP - 82N | -1.1724 to 1.1197 | 13/66/9 | 4/84/0 |
| `iqr_days` | DEEP - 82S | -0.9690 to 0.8941 | 10/72/6 | 0/88/0 |
| `q10_days` | SHALLOW - 82N | -0.2281 to 1.7435 | 0/20/68 | 0/35/53 |
| `q10_days` | SHALLOW - 82S | -0.2541 to 1.4925 | 1/25/62 | 0/51/37 |
| `q10_days` | DEEP - 82N | -0.9150 to 1.0209 | 14/55/19 | 2/79/7 |
| `q10_days` | DEEP - 82S | -0.9933 to 0.8332 | 16/57/15 | 3/85/0 |
| `q90_days` | SHALLOW - 82N | -2.3600 to 0.7473 | 55/33/0 | 33/55/0 |
| `q90_days` | SHALLOW - 82S | -2.2592 to 0.9973 | 54/29/5 | 30/58/0 |
| `q90_days` | DEEP - 82N | -1.6140 to 1.0252 | 39/47/2 | 14/73/1 |
| `q90_days` | DEEP - 82S | -1.3816 to 1.2753 | 35/46/7 | 15/72/1 |

## Availability

Every registered feature is available for every accepted record in
all 352 branch/redshift/SBMAG cells. Missingness occurs at whole-record
rejection, not as feature-specific missing values among accepted records.
