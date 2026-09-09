"""
Tests for Phase 0 thinking fixes and agentic token support.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_preparer.thinking_quality import validate_thinking, filter_low_quality
from training.trainer import TrainingConfig


class TestThinkingQualityThreshold:
    """Tests for strengthened quality validator (Task 0.5)."""

    def test_reject_meta_commentary_below_07(self):
        """Meta-commentary scoring below 0.7 should be rejected."""
        thinking = "El usuario saluda. Debo responder amigable."
        result = validate_thinking(thinking)
        assert result.valid is False, f"Meta-commentary should be rejected, got score={result.score}"

    def test_accept_real_reasoning(self):
        """Genuine reasoning with causal links should pass."""
        thinking = (
            "La entrada 'hello' es un saludo en inglés. "
            "La respuesta apropiada es un saludo recíproco porque establece "
            "el tono de la conversación. Respondo con un saludo amigable."
        )
        result = validate_thinking(thinking)
        assert result.valid is True, f"Real reasoning should pass, got issues={result.issues}"

    def test_filter_low_quality_default_threshold(self):
        """filter_low_quality should use 0.5 as default min_score."""
        import inspect
        sig = inspect.signature(filter_low_quality)
        default = sig.parameters['min_score'].default
        assert default == 0.5, f"Expected default min_score=0.5, got {default}"


class TestThinkingLossWeight:
    """Tests for thinking_loss_weight default (Task 0.6)."""

    def test_training_config_default_is_10(self):
        """TrainingConfig thinking_loss_weight should default to 1.0."""
        config = TrainingConfig()
        assert config.thinking_loss_weight == 1.0, \
            f"Expected thinking_loss_weight=1.0, got {config.thinking_loss_weight}"


class TestDoubleForwardPass:
    """Tests for eliminated double forward pass (Task 0.7 - already DONE)."""

    def test_compute_thinking_metrics_accepts_logits(self):
        """_compute_thinking_metrics should accept logits parameter."""
        import inspect
        from training.trainer import Trainer
        sig = inspect.signature(Trainer._compute_thinking_metrics)
        assert 'logits' in sig.parameters, \
            "_compute_thinking_metrics should accept 'logits' parameter"
