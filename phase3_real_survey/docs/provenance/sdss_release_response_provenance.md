# SDSS-II Final SMP Release Response Provenance Audit

Audit date: 2026-10-04  
Scope: final 10,258-object SDSS-II SNANA SMP release and its release-specific response/calibration system  
Method: provenance audit only; no feature extraction, distribution analysis, normalization, or classification

## Executive finding

The previously missing release-to-response link is now **VERIFIED**.

The official NERSC archive does not merely contain a nearby SDSS example. Its `README01.DATA` identifies the released SNANA version as `SDSS_allCandidates+BOSS`, states that the relevant calibration files are in `$SNDATA_ROOT/kcor/SDSS_Doi2010`, and explicitly prescribes `kcor_SDSS_Bessell90_BD17.fits` for both fitting and simulations. Release-supplied fit configurations repeat that exact path. `README02.FILTERS-ugrizUGRIZ` explicitly identifies `$SNDATA_ROOT/filters/SDSS_Doi2010` as the location of the average and CCD-dependent transmissions for these data.

The preserved KCOR artifact identifies `kcor_SDSS.input`, contains 16 filters, and embeds ten SDSS `ugrizUGRIZ` response columns. Its generating input uses the Doi2010 CCD-averaged response directory, AB/count calibration, BD+17 for the Bessell rest-frame filters, and the documented SDSS AB-offset table. The release can therefore be associated with a specific response-function and calibration family without substituting an assumed passband.

## 1. Official archive recovery

| Item | Record |
|---|---|
| Release page | `https://portal.nersc.gov/project/dessn/SDSS/dataRelease/` |
| Archive URL | `https://portal.nersc.gov/project/dessn/SDSS/dataRelease/SDSS_dataRelease-snana.tar.gz` |
| HTTP status | `200` |
| Content type | `application/x-gzip` |
| Content length | 49,007,071 bytes |
| Last-Modified | `Mon, 04 Apr 2016 18:45:45 GMT` |
| ETag | `"2ebc9df-52fad22e78840"` |
| SHA-256 | `b0861ce0d8bd5ab138bc2e9f41dd5321912d76cb43c718789a7bc637d56e3dbd` |
| Extraction | Temporary case-sensitive APFS volume; no existing extraction overwritten |

The archive contains a nested `SDSS_allCandidates+BOSS.tar.gz` (SHA-256 `033992473b0dace08233bf9b4ef395a2267148f64ee021cc725a475d70d34d42`). Its released FITS files contain 10,258 header rows and 1,120,566 photometry rows. The newly extracted HEAD and PHOT files are byte-identical after decompression to the locally installed copies:

| File | Uncompressed bytes | SHA-256 | Local match |
|---|---:|---|---|
| `SDSS_allCandidates+BOSS_HEAD.FITS` | 2,378,880 | `488ed81fb93b7e783e34cf48ed66252d39556c56617b4ec32ab06e56ef0ca0ea` | Yes |
| `SDSS_allCandidates+BOSS_PHOT.FITS` | 114,307,200 | `3c88f726182690df24858de35793a5e1c70155a7a4835ec762dd57a301c96152` | Yes |
| `SDSS_allCandidates+BOSS.IGNORE` | 10,313 | `42813453b0333e5fa54ddf9e3dd51a6b75ebaaf5fdb0b6a358200fecd06b43e5` | Yes |
| `SDSS_allCandidates+BOSS.LIST` | 34 | `e1c8ff5181ea67b2f2bf61ca28a0e42401a4d6def82ec61262a2bf5232a3f75f` | Yes |

The installed README is a later expanded metadata version and is not byte-identical to the nested 2016 README; this does not affect HEAD/PHOT identity.

## 2. Exhaustive relevant occurrences

### Direct release documentation

| File and location | Evidence | Interpretation |
|---|---|---|
| `README01.DATA:26` | `VERSION_PHOTOMETRY = 'SDSS_allCandidates+BOSS'` | Names the exact released photometry version. |
| `README01.DATA:39-47` | Names `$SNDATA_ROOT/kcor/SDSS_Doi2010` and recommends `kcor_SDSS_Bessell90_BD17.fits` for fitting and simulations. | Direct release-to-KCOR link. |
| `README02.FILTERS-ugrizUGRIZ:3-12` | Defines ten filters `ugrizUGRIZ`; locates average and CCD-dependent transmissions in `$SNDATA_ROOT/filters/SDSS_Doi2010`. | Direct release-to-response-family link. |
| `README02.FILTERS-ugrizUGRIZ:24-31` | States that data fits use `ugrizUGRIZ`, simulations use `ugriz`, and the same KCOR/calibration file contains both sets. | Explains duplicate-band handling. |
| nested release README, lines 14, 69, 199 | Photometry comes from `Holtz/v6_reCalib/{2005,2006,2007}`. | Release reprocessing/calibration state. |
| nested release README, lines 32, 87, 217 | Conversion commands use `SDSS_get_SMP.exe ... --mag asinh`. | Conversion provenance. |
| nested release README, lines 16-18 and repetitions | `FLUXCAL_ERRTOT` is formed by scaling the input `FLUXERR(uJy)`. | Error-conversion provenance. |

### Release-supplied analysis and simulation configurations

The following literal occurrences all name the Doi2010 artifact:

- `SDSS_snfit4par_SALT2.nml:11`
- `SDSS_snfit4par_mlcs2k2.nml:20`
- `SDSS_snfit5par_SALT2.nml:14`
- `SDSS_snfit5par_SALT2+marg.nml:12`
- `SDSS_SIMGEN_S11DM15.INPUT:16`
- `SDSS_SIMGEN_NONIA.INPUT:17`
- duplicated `SNgrid/` and `SNgrid_OLD/` simulation inputs

`SDSS_snfit4par_repeatK09.nml` has a legacy base setting pointing to `SNCOSM09+fitsFormat`, but its release comparison options at lines 19 and 22 explicitly switch to `SDSS_Doi2010/kcor_SDSS_Bessell90_BD17.fits` for the updated AB offsets. This legacy reproduction option does not displace the release README's nominal prescription.

The archived simulation logs provide an independent execution trace: they report opening `/data/dp62.b/data/SNDATA_ROOT/kcor/SDSS_Doi2010/kcor_SDSS_Bessell90_BD17.fits`, reading the AB primary, and using the SDSS ugriz filters and AB-offset values.

## 3. Released photometric system

### Directly documented

- The final FITS product is `SDSS_allCandidates+BOSS`, generated from three `Holtz/v6_reCalib` seasonal inputs.
- It was converted with `SDSS_get_SMP.exe --mag asinh`.
- The official release prescribes the Doi2010 KCOR/calibration artifact for analysis.
- The FITS primary headers identify `SURVEY=SDSS` and `FILTERS=ugrizUGRIZ`.
- The PHOT table contains `FLUXCAL`, `FLUXCALERR`, `MAG`, `MAGERR`, and per-epoch `ZEROPT`/`ZEROPT_ERR`.

### Calibration representation

- The KCOR generating input sets `MAGSYSTEM: AB` and `FILTSYSTEM: COUNT` for the SDSS filters.
- The response metadata states that the final SDSS AB offsets are from the joint SNLS+SDSS calibration and records:
  - u: -0.06791 mag
  - g: +0.02028 mag
  - r: +0.00493 mag
  - i: +0.01780 mag
  - z: +0.01015 mag
- `ZPOFF.DAT` repeats these values for both lower- and upper-case band tokens.
- The modern KCOR build log states that these offsets are stored for application during analysis, rather than baked into the response transmission samples themselves.

The defensible interpretation is therefore an **AB-referenced SNANA FLUXCAL product analyzed with the named Doi2010 calibration artifact**. The archive does not justify applying the listed offsets directly and independently to released `FLUXCAL`; they belong to the prescribed calibration/KCOR interpretation. Any future physical comparison must keep the released photometry and its named calibration artifact paired.

## 4. KCOR artifact and generating inputs

| Property | Recovered value | Status |
|---|---|---|
| Artifact | `kcor_SDSS_Bessell90_BD17.fits.gz` | Verified |
| Artifact SHA-256 | `7febdd9faa7d6032a3a264446773a1032d8cb71160240d8d8f8c4709a47355eb` | Verified |
| Preserved original artifact SHA-256 | `0656632cf757e2df33ac224f532057c351f07c47d7144d796a688bf90206e48a` | Verified |
| Generating input | `kcor_SDSS.input` | Verified |
| Input SHA-256 | `51a2c555ae1359a9f76442a8776e0e834d9d7dec83953ec90019888c9c0a6d55` | Verified |
| Embedded input name | `kcor_SDSS.input` | Verified from FITS header |
| KCOR internal version | 4 in current artifact; preserved original response columns are numerically identical | Verified |
| Observer response directory | `$SNDATA_ROOT/filters/SDSS/SDSS_Doi2010/CCDAVG` | Verified |
| Observer magnitude/flux system | AB / photon-counting | Verified |
| Observer filters | SDSS `ugrizUGRIZ` | Verified |
| Rest-frame filters | Bessell90 `UBVRI` plus `BX` | Verified |
| Rest-frame reference | BD+17 4708 | Verified |
| SN SED | Hsiao07 | Verified |
| Wavelength grid in embedded responses | 2100-11300 Angstrom, 10 Angstrom spacing, 921 rows | Verified |
| Atmosphere | Doi2010 metadata says included at airmass 1.3 | Strong; model provenance text contains an unresolved `model from ??` note |
| CCD representation | CCD-averaged over Stripe 82 north/south and focal plane for this KCOR; CCD1-CCD6 alternatives also exist | Verified |

The current and `_ORIG` KCOR binaries are not byte-identical, but all 16 embedded `FilterTrans` columns are numerically identical. The difference is therefore not a response-function change.

## 5. Case-collision audit

The official 49 MB release archive contains the light curves, documentation, and analysis inputs, but not the standalone Doi response directory. It points users to `SNDATA_ROOT` for those files. Therefore no `u/U` collision occurred within the newly recovered official release archive.

The prior case-insensitive extraction of `SNDATA_ROOT_2024-07-03` did lose standalone case-distinct names. The generating input requires:

`u.dat g.dat r.dat i.dat z.dat U.dat G.dat R.dat I.dat Z.dat`

The local CCDAVG directory preserves only:

| Preserved file | SHA-256 |
|---|---|
| `U.dat` | `f90864aee587f78db55022acccde3f39ad7d97b2fc0e0e9d3b0bc92f7ad8b7e4` |
| `G.dat` | `10bbd59b1b747c7e6b265a33bed24756b431bf3edd60fcfe41e4398aab3983eb` |
| `r.dat` | `11c78b2cbc316ed121cfb99671ec26c14bbf83dacedf0e3a4a8a8b35e578e1c2` |
| `I.dat` | `d82051349de854853f72a9fe3bc0205c6bdd2858bb753d06fb7e95ead842f523` |
| `z.dat` | `2baf3b5ca9c0dda5187230c7b5ca7a98c8f44f504e9a10dd74b197f32d445633` |

The standalone original-file SHA-256 values for the overwritten counterparts cannot be recovered from this extraction and are not invented here. However:

1. the KCOR binary contains all ten response columns;
2. `SDSS-u` equals `SDSS-U`, `g=G`, `r=R`, `i=I`, and `z=Z` exactly in the embedded table;
3. interpolation of each surviving standalone curve onto the KCOR grid agrees with its embedded column to better than `3.7e-8` absolute transmission;
4. the current and preserved-original KCOR embedded response columns agree exactly.

Thus the case collision prevents a byte-level manifest of every original ASCII filename, but it does **not** remove any response function required for the release-prescribed CCD-averaged KCOR comparison. CCD-specific work should use a fresh case-sensitive SNDATA extraction, but that is not required to establish the release's nominal response family.

## 6. Provenance chains

### SDSS

| Arrow | Classification | Evidence |
|---|---|---|
| Official NERSC archive -> `SDSS_allCandidates+BOSS` released SMP | VERIFIED | Nested archive; exact HEAD/PHOT identity; 10,258 rows. |
| Released SMP -> `Holtz/v6_reCalib` reprocessing | VERIFIED | Embedded release README and conversion commands for 2005-2007. |
| Released SMP -> `SDSS_Doi2010` calibration family | VERIFIED | `README01.DATA:39-47`. |
| Released SMP -> `kcor_SDSS_Bessell90_BD17.fits` | VERIFIED | Direct README prescription and multiple supplied fit inputs. |
| KCOR artifact -> `kcor_SDSS.input` | VERIFIED | FITS header and recovered input. |
| KCOR input -> Doi2010 CCDAVG responses | VERIFIED | Explicit `FILTPATH` and filter filenames. |
| Doi2010 response family -> Doi et al. 2010 system responses | STRONG | Local response documentation explicitly cites Doi 2010 and describes averaging/atmosphere; standalone files and embedded KCOR agree numerically. |

### Alternative response set check

No alternative release-specific nominal set supersedes Doi2010 in the official release. The only different path occurs in the legacy Kessler-2009 reproduction base configuration; the same file explicitly provides Doi2010 fit options, while the release README selects Doi2010 for current analysis.

## 7. Final gate

1. **Is the final SDSS SMP release tied to a specific response-function set?**  
   **YES**

2. **Is that response set the Doi 2010 / audited KCOR response family?**  
   **YES**

3. **Can the SDSS and DES response systems now be compared physically without assuming undocumented passbands?**  
   **YES**

4. **Are both selected photometric products sufficiently version-pinned to prevent double application or omission of calibration/chromatic corrections?**  
   **YES**, provided the pinned photometry and the named calibration artifacts are treated as inseparable product pairs. DES v1.2 states that DCR/chromatic corrections are already in the released photometry; SDSS and DES AB offsets remain encoded in their prescribed analysis calibration artifacts and must not be applied as an additional ad hoc flux correction.

## Final decision

**GO**

The pair is ready for the next scientific phase: pre-registering physical predictions before inspecting cross-survey compact-feature distributions.
