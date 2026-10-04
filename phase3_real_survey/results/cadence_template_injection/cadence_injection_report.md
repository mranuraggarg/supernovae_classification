# Phase 3 Object-Level Cadence Injection

## Scientific boundary

- No class labels were read.
- No object redshifts were read.
- No real compact features were calculated.
- Real flux values were not used as template measurements.
- The same frozen SDSS Doi2010 r response was used for both surveys.
- Cadence was reconstructed from each object's own HEAD/PHOT pointer range.

## Registered-direction result

- DES median absolute timing error larger at **11/11** frozen redshift points.
- DES signed-error IQR larger at **11/11** frozen redshift points.
- Interpretation: **FULLY_CONSISTENT_WITH_REGISTERED_DIRECTION**.

## Important correction

The original field-union cadence implementation was abandoned before scientific use because SDSS drift-scan MJDs are object/location-specific. This implementation uses each transient's actual released observing schedule and therefore does not manufacture a global SDSS cadence.

No arbitrary success threshold was introduced after seeing the result.
