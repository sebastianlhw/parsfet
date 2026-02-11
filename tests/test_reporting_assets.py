
import pytest
from pathlib import Path
from unittest.mock import patch
from parsfet.reporting.html_generator import validate_assets

def test_validate_assets_all_present():
    """Verifies that validate_assets does not raise an error when all assets exist."""
    with patch.object(Path, 'exists') as mock_exists:
        # Mock exists to always return True
        mock_exists.return_value = True
        
        # Should not raise any exception
        validate_assets()
        
        # Verify it checked for the expected number of assets
        assert mock_exists.call_count >= 3

def test_validate_assets_missing_single():
    """Verifies that validate_assets raises RuntimeError when one asset is missing."""
    with patch.object(Path, 'exists') as mock_exists:
        # Mock exists to return False for the second asset (alpine.min.js)
        # Order: styles.css, alpine.min.js, plotly.min.js
        mock_exists.side_effect = [True, False, True]
        
        with pytest.raises(RuntimeError) as excinfo:
            validate_assets()
            
        assert "Missing required report assets" in str(excinfo.value)
        assert "js/alpine.min.js" in str(excinfo.value)
        assert "npm install && npm run build" in str(excinfo.value)

def test_validate_assets_missing_multiple():
    """Verifies that validate_assets raises RuntimeError when multiple assets are missing."""
    with patch.object(Path, 'exists') as mock_exists:
        # Mock exists to return False for styles.css and plotly.min.js
        # Order: styles.css, alpine.min.js, plotly.min.js
        mock_exists.side_effect = [False, True, False]
        
        with pytest.raises(RuntimeError) as excinfo:
            validate_assets()
            
        assert "css/styles.css" in str(excinfo.value)
        assert "js/plotly.min.js" in str(excinfo.value)
        assert "js/alpine.min.js" not in str(excinfo.value)
