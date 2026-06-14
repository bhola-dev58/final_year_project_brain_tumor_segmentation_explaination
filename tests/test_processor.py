"""
Unit tests for src/processor.py

Tests are fully isolated — no model loading required.
Uses synthetic numpy arrays to simulate MRI scan inputs.
"""
import numpy as np
import pytest

from src.processor import create_segmentation, estimate_location, estimate_severity


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def synthetic_mri_rgb():
    """Creates a 256x256 synthetic RGB MRI-like scan (gray matter simulation)."""
    img = np.random.randint(60, 200, (256, 256, 3), dtype=np.uint8)
    # Simulate a bright tumor region in the center
    img[100:150, 100:150] = 240
    return img


@pytest.fixture
def synthetic_heatmap():
    """Creates a normalized heatmap with peak activation at center."""
    hm = np.zeros((32, 32), dtype=np.float32)
    hm[14:18, 14:18] = 1.0
    hm[12:20, 12:20] = np.maximum(hm[12:20, 12:20], 0.6)
    return hm


# ------------------------------------------------------------------ #
# create_segmentation tests
# ------------------------------------------------------------------ #
class TestCreateSegmentation:
    def test_returns_three_values(self, synthetic_heatmap, synthetic_mri_rgb):
        result = create_segmentation(synthetic_heatmap, synthetic_mri_rgb)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_output_image_same_shape(self, synthetic_heatmap, synthetic_mri_rgb):
        seg_img, mask, area_pct = create_segmentation(synthetic_heatmap, synthetic_mri_rgb)
        assert seg_img.shape == synthetic_mri_rgb.shape

    def test_mask_is_binary(self, synthetic_heatmap, synthetic_mri_rgb):
        _, mask, _ = create_segmentation(synthetic_heatmap, synthetic_mri_rgb)
        unique_vals = np.unique(mask)
        assert all(v in [0, 255] for v in unique_vals), f"Unexpected mask values: {unique_vals}"

    def test_area_percentage_is_non_negative(self, synthetic_heatmap, synthetic_mri_rgb):
        _, _, area_pct = create_segmentation(synthetic_heatmap, synthetic_mri_rgb)
        assert area_pct >= 0.0

    def test_area_percentage_is_below_100(self, synthetic_heatmap, synthetic_mri_rgb):
        _, _, area_pct = create_segmentation(synthetic_heatmap, synthetic_mri_rgb)
        assert area_pct <= 100.0

    def test_handles_blank_heatmap_gracefully(self, synthetic_mri_rgb):
        """A zero heatmap should return a copy of the original image without crashing."""
        blank_heatmap = np.zeros((32, 32), dtype=np.float32)
        seg_img, mask, area_pct = create_segmentation(blank_heatmap, synthetic_mri_rgb)
        assert seg_img.shape == synthetic_mri_rgb.shape
        assert area_pct == 0.0

    def test_handles_grayscale_input(self, synthetic_heatmap):
        """Grayscale input (2D array) should be handled without crashing."""
        gray_img = np.random.randint(60, 200, (256, 256), dtype=np.uint8)
        seg_img, mask, area_pct = create_segmentation(synthetic_heatmap, gray_img)
        assert area_pct >= 0.0


# ------------------------------------------------------------------ #
# estimate_location tests
# ------------------------------------------------------------------ #
class TestEstimateLocation:
    def test_returns_string(self, synthetic_heatmap, synthetic_mri_rgb):
        result = estimate_location(synthetic_heatmap, synthetic_mri_rgb.shape)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_top_left_activation_is_superior_left_frontal(self):
        """Peak activation in top-left quadrant → Left Frontal Lobe (Superior)."""
        hm = np.zeros((32, 32), dtype=np.float32)
        hm[2:5, 2:5] = 1.0
        location = estimate_location(hm, (256, 256, 3))
        assert "Left" in location
        assert "Frontal" in location
        assert "Superior" in location

    def test_bottom_right_activation_is_inferior_right_occipital(self):
        """Peak activation in bottom-right → Right Occipital Lobe (Inferior)."""
        hm = np.zeros((32, 32), dtype=np.float32)
        hm[28:32, 28:32] = 1.0
        location = estimate_location(hm, (256, 256, 3))
        assert "Right" in location
        assert "Occipital" in location
        assert "Inferior" in location

    def test_fallback_on_bad_input(self):
        """Should return 'Unknown Location' string without raising an exception."""
        result = estimate_location(np.array([]), (0, 0, 3))
        assert isinstance(result, str)


# ------------------------------------------------------------------ #
# estimate_severity tests
# ------------------------------------------------------------------ #
class TestEstimateSeverity:
    @pytest.mark.parametrize("confidence, area, expected_severity", [
        (96.0, 6.0,  "High"),
        (85.0, 4.0,  "Moderate"),
        (70.0, 0.0,  "Low"),
        (40.0, 0.0,  "Uncertain"),
    ])
    def test_severity_classification(self, confidence, area, expected_severity):
        severity, color = estimate_severity(confidence, area)
        assert severity == expected_severity, (
            f"Expected '{expected_severity}' for confidence={confidence}, area={area}, got '{severity}'"
        )

    def test_returns_tuple(self):
        result = estimate_severity(90.0, 5.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_color_is_hex_string(self):
        _, color = estimate_severity(90.0, 5.0)
        assert color.startswith("#"), f"Expected hex color, got: {color}"

    def test_high_confidence_no_area_is_low_not_high(self):
        """High confidence but tiny area should NOT be classified as High severity."""
        severity, _ = estimate_severity(99.0, 0.5)
        assert severity != "High"
