# 09 — Testing

The project includes a complete automated test suite with **33 tests** across two test files. Tests are written with `pytest` and cover both isolated unit logic and the full end-to-end inference pipeline.

---

## Running Tests

```bash
# Run all 33 tests
pytest tests/ -v

# Run only unit tests (fast — no model loading, ~2 seconds)
pytest tests/test_processor.py -v

# Run only integration tests (loads models — ~30–60 seconds)
pytest tests/test_inference.py -v

# Run with test summary only (no verbose output)
pytest tests/ -q
```

**Expected result:** `33 passed`

---

## Test File 1: `tests/test_processor.py`

**Type:** Unit tests — fully isolated, no model loading required.  
**Count:** 18 tests  
**Target module:** `src/processor.py`

Tests use synthetic numpy arrays to simulate MRI scan inputs — no real images or models are needed.

### Fixtures

```python
@pytest.fixture
def synthetic_mri_rgb():
    """256×256 RGB image with a bright tumor-like region at center (100:150, 100:150)."""
    img = np.random.randint(60, 200, (256, 256, 3), dtype=np.uint8)
    img[100:150, 100:150] = 240    # Bright spot simulates tumor
    return img

@pytest.fixture
def synthetic_heatmap():
    """32×32 normalized heatmap with peak activation at center."""
    hm = np.zeros((32, 32), dtype=np.float32)
    hm[14:18, 14:18] = 1.0         # Full activation center
    hm[12:20, 12:20] = np.maximum(hm[12:20, 12:20], 0.6)  # High activation ring
    return hm
```

---

### `TestCreateSegmentation` (7 tests)

| Test | What It Checks |
|---|---|
| `test_returns_three_values` | Output is a tuple of length 3 |
| `test_output_image_same_shape` | Segmented image has same spatial shape as input |
| `test_mask_is_binary` | Mask pixels are only 0 or 255 |
| `test_area_percentage_is_non_negative` | Tumor area is ≥ 0.0 |
| `test_area_percentage_is_below_100` | Tumor area is ≤ 100.0 |
| `test_handles_blank_heatmap_gracefully` | Zero heatmap returns original image + 0.0 area — no crash |
| `test_handles_grayscale_input` | 2D grayscale input does not crash |

---

### `TestEstimateLocation` (4 tests)

| Test | What It Checks |
|---|---|
| `test_returns_string` | Returns a non-empty string |
| `test_top_left_activation_is_superior_left_frontal` | Peak at top-left → "Left Frontal Lobe (Superior)" |
| `test_bottom_right_activation_is_inferior_right_occipital` | Peak at bottom-right → "Right Occipital Lobe (Inferior)" |
| `test_fallback_on_bad_input` | Empty array returns "Unknown Location" string without exception |

---

### `TestEstimateSeverity` (7 tests)

| Test | What It Checks |
|---|---|
| `test_severity_classification[96, 6, High]` | Parametrized: High severity at 96% + 6% area |
| `test_severity_classification[85, 4, Moderate]` | Parametrized: Moderate at 85% + 4% area |
| `test_severity_classification[70, 0, Low]` | Parametrized: Low at 70% with 0% area |
| `test_severity_classification[40, 0, Uncertain]` | Parametrized: Uncertain at 40% |
| `test_returns_tuple` | Returns a 2-element tuple |
| `test_color_is_hex_string` | Second tuple element starts with `#` |
| `test_high_confidence_no_area_is_low_not_high` | 99% confidence + 0.5% area → NOT High (requires area > 5%) |

---

## Test File 2: `tests/test_inference.py`

**Type:** Integration tests — loads the real TensorFlow models.  
**Count:** 15 tests  
**Target module:** `src/inference.py`  
**Warning:** First run takes 30–60 seconds due to model loading. Subsequent tests in the same session are fast.

### Fixtures

```python
@pytest.fixture(scope="module")
def real_mri_scan():
    """Loads test_images/Tr-me_0025.jpg once per test module."""
    path = "test_images/Tr-me_0025.jpg"
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@pytest.fixture(scope="module")
def prediction_result(real_mri_scan):
    """Runs full inference once; all tests in this class share the result."""
    return predict_tumor_logic(real_mri_scan)
```

The `scope="module"` ensures models are loaded only once regardless of how many tests run.

---

### `TestPredictTumorLogic` (11 tests)

| Test | What It Checks |
|---|---|
| `test_valid_flag_is_true` | `is_valid` is `True` for a real scan |
| `test_class_name_is_valid` | `class_name` is one of the 4 valid classes |
| `test_confidence_is_percentage` | `confidence` is between 0.0 and 100.0 |
| `test_is_tumor_is_boolean` | `is_tumor` is a Python `bool` |
| `test_inference_time_is_positive` | `inference_time > 0` |
| `test_avg_pred_sums_to_one` | Softmax probabilities sum to 1.0 (±0.0001) |
| `test_four_class_probabilities` | `avg_pred` has exactly 4 elements |
| `test_gradcam_overlay_is_rgb_image` | Grad-CAM overlay is a 3D array with 3 channels |
| `test_segmentation_image_is_rgb` | Segmentation image is a 3D array with 3 channels |
| `test_severity_is_valid_string` | `severity` is one of: High, Moderate, Low, Uncertain |
| `test_severity_color_is_hex` | `severity_color` starts with `#` |
| `test_class_idx_matches_class_name` | `classes[class_idx] == class_name` |

---

### `TestEdgeCases` (3 tests)

| Test | What It Checks |
|---|---|
| `test_none_input_returns_invalid` | `predict_tumor_logic(None)` returns `{"is_valid": False}` |
| `test_none_input_error_is_string` | The `"error"` field is a non-empty string |
| `test_float_image_is_normalized` | Float32 image (0–255 range) is accepted and processed correctly |

---

## Test Coverage Summary

| Module | Tests | Type | Model Required |
|---|---|---|---|
| `src/processor.py` | 18 | Unit | ❌ No |
| `src/inference.py` | 15 | Integration | ✅ Yes |
| **Total** | **33** | Mixed | — |

---

## Common Issues

**Issue:** `ModuleNotFoundError: No module named 'src'`  
**Fix:** Run pytest from the project root directory:
```bash
cd /path/to/Brain_Tumor_Project
pytest tests/ -v
```

**Issue:** Integration tests timeout or skip  
**Fix:** Ensure `models/densenet_best.h5` and `models/inception_best.h5` exist. If the path is missing, pytest will skip with `pytest.skip()`.

**Issue:** `test_images/Tr-me_0025.jpg` not found  
**Fix:** Ensure the `test_images/` directory is present at the project root. Integration tests will auto-skip if the file is missing.
