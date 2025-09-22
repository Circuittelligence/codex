"""Engagement band calculation logic for OCEAN personality scores.

This module implements the engagement band calculation based on OCEAN scores
and configuration parameters from the protocol.
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class OceanScores:
    """Container for OCEAN personality scores."""
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    def validate(self) -> bool:
        """Validate that all scores are within 0-1 range."""
        scores = [self.openness, self.conscientiousness, self.extraversion,
                 self.agreeableness, self.neuroticism]
        return all(0.0 <= score <= 1.0 for score in scores)


@dataclass
class EngagementConfig:
    """Configuration parameters for engagement band calculation."""
    # Weight factors for each OCEAN dimension
    openness_weight: float = 0.2
    conscientiousness_weight: float = 0.25
    extraversion_weight: float = 0.3
    agreeableness_weight: float = 0.15
    neuroticism_weight: float = 0.1  # Inverse weight (high neuroticism lowers engagement)
    
    # Engagement band thresholds
    low_threshold: float = 0.3
    medium_threshold: float = 0.6
    high_threshold: float = 0.8
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        # Check that weights sum to 1.0 (within tolerance)
        total_weight = (self.openness_weight + self.conscientiousness_weight + 
                       self.extraversion_weight + self.agreeableness_weight + 
                       self.neuroticism_weight)
        
        weight_valid = abs(total_weight - 1.0) < 0.001
        
        # Check that thresholds are in ascending order and within range
        threshold_valid = (0.0 <= self.low_threshold <= self.medium_threshold <= 
                          self.high_threshold <= 1.0)
        
        return weight_valid and threshold_valid


class EngagementBandCalculator:
    """Calculator for determining engagement bands from OCEAN scores."""
    
    def __init__(self, config: Optional[EngagementConfig] = None):
        """Initialize calculator with optional custom configuration."""
        self.config = config or EngagementConfig()
        if not self.config.validate():
            raise ValueError("Invalid engagement configuration parameters")
    
    def calculate_engagement_score(self, ocean_scores: OceanScores) -> float:
        """Calculate raw engagement score from OCEAN personality scores.
        
        Args:
            ocean_scores: OCEAN personality scores (0-1 range)
            
        Returns:
            Engagement score (0-1 range)
            
        Raises:
            ValueError: If OCEAN scores are invalid
        """
        if not ocean_scores.validate():
            raise ValueError("OCEAN scores must be in 0-1 range")
        
        # Calculate weighted engagement score
        # Note: Neuroticism is inversely weighted (high neuroticism reduces engagement)
        engagement_score = (
            ocean_scores.openness * self.config.openness_weight +
            ocean_scores.conscientiousness * self.config.conscientiousness_weight +
            ocean_scores.extraversion * self.config.extraversion_weight +
            ocean_scores.agreeableness * self.config.agreeableness_weight +
            (1.0 - ocean_scores.neuroticism) * self.config.neuroticism_weight
        )
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, engagement_score))
    
    def determine_engagement_band(self, ocean_scores: OceanScores) -> str:
        """Determine engagement band from OCEAN scores.
        
        Args:
            ocean_scores: OCEAN personality scores
            
        Returns:
            Engagement band: 'low', 'medium', 'high', or 'very_high'
        """
        score = self.calculate_engagement_score(ocean_scores)
        
        if score < self.config.low_threshold:
            return 'low'
        elif score < self.config.medium_threshold:
            return 'medium'
        elif score < self.config.high_threshold:
            return 'high'
        else:
            return 'very_high'
    
    def get_engagement_analysis(self, ocean_scores: OceanScores) -> Dict[str, Any]:
        """Get comprehensive engagement analysis.
        
        Args:
            ocean_scores: OCEAN personality scores
            
        Returns:
            Dictionary containing engagement score, band, and detailed analysis
        """
        score = self.calculate_engagement_score(ocean_scores)
        band = self.determine_engagement_band(ocean_scores)
        
        # Calculate individual dimension contributions
        contributions = {
            'openness': ocean_scores.openness * self.config.openness_weight,
            'conscientiousness': ocean_scores.conscientiousness * self.config.conscientiousness_weight,
            'extraversion': ocean_scores.extraversion * self.config.extraversion_weight,
            'agreeableness': ocean_scores.agreeableness * self.config.agreeableness_weight,
            'neuroticism_inverse': (1.0 - ocean_scores.neuroticism) * self.config.neuroticism_weight
        }
        
        return {
            'engagement_score': score,
            'engagement_band': band,
            'ocean_scores': {
                'openness': ocean_scores.openness,
                'conscientiousness': ocean_scores.conscientiousness,
                'extraversion': ocean_scores.extraversion,
                'agreeableness': ocean_scores.agreeableness,
                'neuroticism': ocean_scores.neuroticism
            },
            'dimension_contributions': contributions,
            'config': {
                'weights': {
                    'openness': self.config.openness_weight,
                    'conscientiousness': self.config.conscientiousness_weight,
                    'extraversion': self.config.extraversion_weight,
                    'agreeableness': self.config.agreeableness_weight,
                    'neuroticism': self.config.neuroticism_weight
                },
                'thresholds': {
                    'low': self.config.low_threshold,
                    'medium': self.config.medium_threshold,
                    'high': self.config.high_threshold
                }
            }
        }


def calculate_engagement_band(ocean_data: Dict[str, float], 
                            config_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function for calculating engagement band from dictionary data.
    
    Args:
        ocean_data: Dictionary with OCEAN scores
        config_data: Optional configuration dictionary
        
    Returns:
        Engagement analysis dictionary
    """
    # Parse OCEAN scores
    ocean_scores = OceanScores(
        openness=ocean_data.get('openness', 0.5),
        conscientiousness=ocean_data.get('conscientiousness', 0.5),
        extraversion=ocean_data.get('extraversion', 0.5),
        agreeableness=ocean_data.get('agreeableness', 0.5),
        neuroticism=ocean_data.get('neuroticism', 0.5)
    )
    
    # Parse configuration if provided
    config = None
    if config_data:
        config = EngagementConfig(
            openness_weight=config_data.get('openness_weight', 0.2),
            conscientiousness_weight=config_data.get('conscientiousness_weight', 0.25),
            extraversion_weight=config_data.get('extraversion_weight', 0.3),
            agreeableness_weight=config_data.get('agreeableness_weight', 0.15),
            neuroticism_weight=config_data.get('neuroticism_weight', 0.1),
            low_threshold=config_data.get('low_threshold', 0.3),
            medium_threshold=config_data.get('medium_threshold', 0.6),
            high_threshold=config_data.get('high_threshold', 0.8)
        )
    
    # Calculate engagement band
    calculator = EngagementBandCalculator(config)
    return calculator.get_engagement_analysis(ocean_scores)


if __name__ == "__main__":
    # Example usage
    sample_ocean = {
        'openness': 0.7,
        'conscientiousness': 0.8,
        'extraversion': 0.6,
        'agreeableness': 0.7,
        'neuroticism': 0.3
    }
    
    result = calculate_engagement_band(sample_ocean)
    print(json.dumps(result, indent=2))
