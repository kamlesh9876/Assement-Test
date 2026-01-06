"""
Emotional Intelligence and Sentiment Analysis for Phase 2
Advanced emotion detection and sentiment analysis using facial expressions and behavioral patterns
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import time
import math

from computer_vision import VisualAnalysisResult, FacialExpressionAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionType(Enum):
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    CONFUSION = "confusion"
    FRUSTRATION = "frustration"
    CONFIDENCE = "confidence"

class SentimentPolarity(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class CognitiveLoad(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    OVERLOAD = "overload"

@dataclass
class EmotionScore:
    """Individual emotion score with confidence"""
    emotion: EmotionType
    intensity: float  # 0-1 scale
    confidence: float  # 0-1 scale
    duration: float  # seconds detected

@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    polarity: SentimentPolarity
    positivity_score: float  # 0-1 scale
    negativity_score: float  # 0-1 scale
    neutrality_score: float  # 0-1 scale
    arousal: float  # 0-1 scale (calm to excited)
    valence: float  # -1 to 1 (negative to positive)

@dataclass
class CognitiveState:
    """Cognitive state analysis"""
    cognitive_load: CognitiveLoad
    focus_level: float  # 0-1 scale
    mental_fatigue: float  # 0-1 scale
    engagement_depth: float  # 0-1 scale
    processing_speed: float  # 0-1 scale

@dataclass
class EmotionalIntelligenceResult:
    """Complete emotional intelligence analysis"""
    timestamp: float
    primary_emotion: EmotionType
    emotion_scores: List[EmotionScore]
    sentiment: SentimentAnalysis
    cognitive_state: CognitiveState
    emotional_stability: float  # 0-1 scale
    stress_indicators: List[str]
    confidence_indicators: List[str]

class EmotionalIntelligenceEngine:
    """Advanced emotional intelligence analysis engine"""
    
    def __init__(self):
        # Emotion detection models (simplified rule-based for Phase 2)
        self.emotion_patterns = self._initialize_emotion_patterns()
        
        # Historical data for trend analysis
        self.emotion_history = deque(maxlen=120)  # Last 2 minutes
        self.sentiment_history = deque(maxlen=180)  # Last 3 minutes
        self.cognitive_history = deque(maxlen=90)   # Last 1.5 minutes
        
        # Baseline measurements
        self.baseline_emotions = {}
        self.baseline_sentiment = SentimentPolarity.NEUTRAL
        self.baseline_cognitive_load = CognitiveLoad.LOW
        
        # Configuration
        self.emotion_thresholds = {
            'intensity_low': 0.3,
            'intensity_medium': 0.6,
            'confidence_min': 0.4,
            'stability_window': 30  # seconds
        }
        
        logger.info("Emotional Intelligence Engine initialized")
    
    def _initialize_emotion_patterns(self) -> Dict[EmotionType, Dict[str, Any]]:
        """Initialize emotion detection patterns"""
        return {
            EmotionType.HAPPINESS: {
                'facial_cues': ['smile', 'raised_cheeks', 'eye_crinkles'],
                'behavioral_cues': ['upright_posture', 'steady_gaze', 'moderate_movement'],
                'sentiment_bias': 0.8,
                'arousal_level': 0.6
            },
            EmotionType.SADNESS: {
                'facial_cues': ['frown', 'drooped_mouth', 'downturned_eyes'],
                'behavioral_cues': ['slumped_posture', 'slow_movement', 'avoidant_gaze'],
                'sentiment_bias': -0.7,
                'arousal_level': 0.3
            },
            EmotionType.ANGER: {
                'facial_cues': ['furrowed_brows', 'tight_lips', 'intense_stare'],
                'behavioral_cues': ['tense_posture', 'abrupt_movements', 'direct_gaze'],
                'sentiment_bias': -0.8,
                'arousal_level': 0.9
            },
            EmotionType.FEAR: {
                'facial_cues': ['wide_eyes', 'open_mouth', 'raised_eyebrows'],
                'behavioral_cues': ['rigid_posture', 'trembling', 'scattered_gaze'],
                'sentiment_bias': -0.6,
                'arousal_level': 0.8
            },
            EmotionType.SURPRISE: {
                'facial_cues': ['raised_eyebrows', 'wide_eyes', 'open_mouth'],
                'behavioral_cues': ['alert_posture', 'quick_movements', 'focused_gaze'],
                'sentiment_bias': 0.1,
                'arousal_level': 0.7
            },
            EmotionType.CONFUSION: {
                'facial_cues': ['furrowed_brows', 'tilted_head', 'squinted_eyes'],
                'behavioral_cues': ['hesitant_movements', 'scattered_gaze', 'fidgeting'],
                'sentiment_bias': -0.2,
                'arousal_level': 0.5
            },
            EmotionType.FRUSTRATION: {
                'facial_cues': ['tense_jaw', 'pressed_lips', 'furrowed_brows'],
                'behavioral_cues': ['restless_movements', 'avoidant_gaze', 'tense_posture'],
                'sentiment_bias': -0.6,
                'arousal_level': 0.7
            },
            EmotionType.CONFIDENCE: {
                'facial_cues': ['direct_gaze', 'relaced_face', 'slight_smile'],
                'behavioral_cues': ['upright_posture', 'steady_movements', 'focused_attention'],
                'sentiment_bias': 0.7,
                'arousal_level': 0.4
            },
            EmotionType.NEUTRAL: {
                'facial_cues': ['relaxed_face', 'neutral_expression'],
                'behavioral_cues': ['balanced_posture', 'natural_movements'],
                'sentiment_bias': 0.0,
                'arousal_level': 0.3
            }
        }
    
    def analyze_emotional_state(self, visual_result: VisualAnalysisResult, response_time: Optional[float] = None) -> EmotionalIntelligenceResult:
        """Perform comprehensive emotional intelligence analysis"""
        try:
            timestamp = time.time()
            
            # Detect emotions from facial expressions
            emotion_scores = self._detect_emotions(visual_result.expression)
            
            # Determine primary emotion
            primary_emotion = self._determine_primary_emotion(emotion_scores)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(emotion_scores, visual_result)
            
            # Assess cognitive state
            cognitive_state = self._assess_cognitive_state(visual_result, response_time)
            
            # Calculate emotional stability
            emotional_stability = self._calculate_emotional_stability(emotion_scores)
            
            # Identify stress and confidence indicators
            stress_indicators = self._identify_stress_indicators(visual_result, emotion_scores)
            confidence_indicators = self._identify_confidence_indicators(visual_result, emotion_scores)
            
            result = EmotionalIntelligenceResult(
                timestamp=timestamp,
                primary_emotion=primary_emotion,
                emotion_scores=emotion_scores,
                sentiment=sentiment,
                cognitive_state=cognitive_state,
                emotional_stability=emotional_stability,
                stress_indicators=stress_indicators,
                confidence_indicators=confidence_indicators
            )
            
            # Store in history
            self.emotion_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Emotional analysis error: {e}")
            return self._create_default_result()
    
    def _detect_emotions(self, expression: FacialExpressionAnalysis) -> List[EmotionScore]:
        """Detect emotions from facial expression analysis"""
        try:
            emotion_scores = []
            
            # Base emotion detection from expression
            if expression.expression == "happy":
                emotion_scores.append(EmotionScore(
                    emotion=EmotionType.HAPPINESS,
                    intensity=expression.emotion_confidence,
                    confidence=expression.emotion_confidence,
                    duration=0.0
                ))
            
            elif expression.expression == "sad":
                emotion_scores.append(EmotionScore(
                    emotion=EmotionType.SADNESS,
                    intensity=expression.emotion_confidence,
                    confidence=expression.emotion_confidence,
                    duration=0.0
                ))
            
            elif expression.expression == "surprised":
                emotion_scores.append(EmotionScore(
                    emotion=EmotionType.SURPRISE,
                    intensity=expression.emotion_confidence,
                    confidence=expression.emotion_confidence,
                    duration=0.0
                ))
            
            # Detect secondary emotions based on facial features
            if expression.frown_detected:
                emotion_scores.append(EmotionScore(
                    emotion=EmotionType.FRUSTRATION,
                    intensity=0.6,
                    confidence=0.5,
                    duration=0.0
                ))
            
            if expression.eyebrow_raised:
                emotion_scores.append(EmotionScore(
                    emotion=EmotionType.CONFUSION,
                    intensity=0.4,
                    confidence=0.4,
                    duration=0.0
                ))
            
            # Add neutral emotion as baseline
            emotion_scores.append(EmotionScore(
                emotion=EmotionType.NEUTRAL,
                intensity=0.3,
                confidence=0.8,
                duration=0.0
            ))
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            return [EmotionScore(EmotionType.NEUTRAL, 0.3, 0.8, 0.0)]
    
    def _determine_primary_emotion(self, emotion_scores: List[EmotionScore]) -> EmotionType:
        """Determine the primary emotion from scores"""
        try:
            if not emotion_scores:
                return EmotionType.NEUTRAL
            
            # Sort by intensity and confidence
            sorted_scores = sorted(
                emotion_scores, 
                key=lambda x: x.intensity * x.confidence, 
                reverse=True
            )
            
            # Return the highest scoring emotion above threshold
            for score in sorted_scores:
                if (score.intensity >= self.emotion_thresholds['intensity_low'] and 
                    score.confidence >= self.emotion_thresholds['confidence_min']):
                    return score.emotion
            
            return EmotionType.NEUTRAL
            
        except Exception as e:
            logger.error(f"Primary emotion determination error: {e}")
            return EmotionType.NEUTRAL
    
    def _analyze_sentiment(self, emotion_scores: List[EmotionScore], visual_result: VisualAnalysisResult) -> SentimentAnalysis:
        """Analyze sentiment from emotions and visual cues"""
        try:
            positivity_score = 0.0
            negativity_score = 0.0
            arousal = 0.0
            
            # Calculate sentiment from emotions
            total_weight = 0.0
            
            for score in emotion_scores:
                weight = score.intensity * score.confidence
                total_weight += weight
                
                pattern = self.emotion_patterns.get(score.emotion, {})
                sentiment_bias = pattern.get('sentiment_bias', 0.0)
                arousal_level = pattern.get('arousal_level', 0.3)
                
                if sentiment_bias > 0:
                    positivity_score += weight * sentiment_bias
                elif sentiment_bias < 0:
                    negativity_score += weight * abs(sentiment_bias)
                
                arousal += weight * arousal_level
            
            # Normalize scores
            if total_weight > 0:
                positivity_score /= total_weight
                negativity_score /= total_weight
                arousal /= total_weight
            
            # Calculate neutrality
            neutrality_score = 1.0 - (positivity_score + negativity_score)
            neutrality_score = max(0.0, neutrality_score)
            
            # Calculate valence (-1 to 1)
            valence = (positivity_score - negativity_score)
            
            # Determine polarity
            if valence > 0.2:
                polarity = SentimentPolarity.POSITIVE
            elif valence < -0.2:
                polarity = SentimentPolarity.NEGATIVE
            else:
                polarity = SentimentPolarity.NEUTRAL
            
            return SentimentAnalysis(
                polarity=polarity,
                positivity_score=min(1.0, positivity_score),
                negativity_score=min(1.0, negativity_score),
                neutrality_score=min(1.0, neutrality_score),
                arousal=min(1.0, arousal),
                valence=np.clip(valence, -1.0, 1.0)
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return SentimentAnalysis(SentimentPolarity.NEUTRAL, 0.3, 0.3, 0.4, 0.3, 0.0)
    
    def _assess_cognitive_state(self, visual_result: VisualAnalysisResult, response_time: Optional[float]) -> CognitiveState:
        """Assess cognitive state from visual and performance data"""
        try:
            # Cognitive load based on attention and engagement
            attention_score = 0.0
            if visual_result.behavior.attention_level.value == "high":
                attention_score = 0.2
            elif visual_result.behavior.attention_level.value == "medium":
                attention_score = 0.5
            else:
                attention_score = 0.8
            
            # Mental fatigue based on blink rate and expression
            fatigue_score = 0.0
            if visual_result.eye_analysis.blink_rate > 25:
                fatigue_score += 0.3
            if visual_result.expression.expression in ["sad", "confused"]:
                fatigue_score += 0.2
            
            # Engagement depth based on eye contact and posture
            engagement_depth = (visual_result.eye_analysis.eye_contact + visual_result.behavior.posture_score) / 2
            
            # Processing speed based on response time (if available)
            processing_speed = 0.5  # Default
            if response_time is not None:
                if response_time < 15:
                    processing_speed = 0.8
                elif response_time < 30:
                    processing_speed = 0.6
                elif response_time < 45:
                    processing_speed = 0.4
                else:
                    processing_speed = 0.2
            
            # Focus level
            focus_level = (visual_result.eye_analysis.eye_contact + 
                          (1.0 - visual_result.behavior.movement_intensity)) / 2
            
            # Determine cognitive load
            load_score = attention_score + fatigue_score + (1.0 - engagement_depth)
            load_score = load_score / 3.0
            
            if load_score < 0.3:
                cognitive_load = CognitiveLoad.LOW
            elif load_score < 0.6:
                cognitive_load = CognitiveLoad.MEDIUM
            elif load_score < 0.8:
                cognitive_load = CognitiveLoad.HIGH
            else:
                cognitive_load = CognitiveLoad.OVERLOAD
            
            return CognitiveState(
                cognitive_load=cognitive_load,
                focus_level=focus_level,
                mental_fatigue=min(1.0, fatigue_score),
                engagement_depth=engagement_depth,
                processing_speed=processing_speed
            )
            
        except Exception as e:
            logger.error(f"Cognitive state assessment error: {e}")
            return CognitiveState(CognitiveLoad.MEDIUM, 0.5, 0.3, 0.5, 0.5)
    
    def _calculate_emotional_stability(self, emotion_scores: List[EmotionScore]) -> float:
        """Calculate emotional stability based on emotion consistency"""
        try:
            if len(self.emotion_history) < 2:
                return 0.7  # Default stability
            
            # Get recent emotions
            recent_emotions = list(self.emotion_history)[-10:]  # Last 10 analyses
            
            # Calculate emotion variance
            primary_emotions = [e.primary_emotion for e in recent_emotions]
            emotion_counts = {}
            
            for emotion in primary_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Stability based on consistency of primary emotion
            most_common_count = max(emotion_counts.values()) if emotion_counts else 1
            consistency = most_common_count / len(recent_emotions)
            
            # Consider intensity variance
            intensities = []
            for analysis in recent_emotions:
                primary_score = next((s for s in analysis.emotion_scores if s.emotion == analysis.primary_emotion), None)
                if primary_score:
                    intensities.append(primary_score.intensity)
            
            intensity_variance = np.var(intensities) if intensities else 0.0
            intensity_stability = 1.0 - min(1.0, intensity_variance)
            
            # Combined stability score
            stability = (consistency * 0.7 + intensity_stability * 0.3)
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            logger.error(f"Emotional stability calculation error: {e}")
            return 0.5
    
    def _identify_stress_indicators(self, visual_result: VisualAnalysisResult, emotion_scores: List[EmotionScore]) -> List[str]:
        """Identify stress indicators from visual and emotional data"""
        try:
            indicators = []
            
            # High blink rate
            if visual_result.eye_analysis.blink_rate > 30:
                indicators.append("high_blink_rate")
            
            # Low eye contact
            if visual_result.eye_analysis.eye_contact < 0.3:
                indicators.append("poor_eye_contact")
            
            # Negative emotions
            negative_emotions = [e for e in emotion_scores if e.emotion in [EmotionType.FEAR, EmotionType.ANGER, EmotionType.FRUSTRATION]]
            if negative_emotions:
                avg_intensity = np.mean([e.intensity for e in negative_emotions])
                if avg_intensity > 0.5:
                    indicators.append("negative_emotions")
            
            # Excessive movement
            if visual_result.behavior.movement_intensity > 0.5:
                indicators.append("restless_behavior")
            
            # Poor posture
            if visual_result.behavior.posture_score < 0.4:
                indicators.append("poor_posture")
            
            # High cognitive load
            if len(self.cognitive_history) > 0:
                recent_load = list(self.cognitive_history)[-5:]
                high_load_count = sum(1 for c in recent_load if c.cognitive_load in [CognitiveLoad.HIGH, CognitiveLoad.OVERLOAD])
                if high_load_count >= 3:
                    indicators.append("high_cognitive_load")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Stress indicators identification error: {e}")
            return []
    
    def _identify_confidence_indicators(self, visual_result: VisualAnalysisResult, emotion_scores: List[EmotionScore]) -> List[str]:
        """Identify confidence indicators from visual and emotional data"""
        try:
            indicators = []
            
            # Good eye contact
            if visual_result.eye_analysis.eye_contact > 0.7:
                indicators.append("strong_eye_contact")
            
            # Positive emotions
            positive_emotions = [e for e in emotion_scores if e.emotion in [EmotionType.HAPPINESS, EmotionType.CONFIDENCE]]
            if positive_emotions:
                avg_intensity = np.mean([e.intensity for e in positive_emotions])
                if avg_intensity > 0.4:
                    indicators.append("positive_emotions")
            
            # Steady gaze
            if len(visual_result.eye_analysis.gaze_direction) == 2:
                gaze_x, gaze_y = visual_result.eye_analysis.gaze_direction
                if abs(gaze_x) < 0.3 and abs(gaze_y) < 0.3:
                    indicators.append("steady_gaze")
            
            # Good posture
            if visual_result.behavior.posture_score > 0.7:
                indicators.append("good_posture")
            
            # Low movement (calm)
            if visual_result.behavior.movement_intensity < 0.2:
                indicators.append("calm_demeanor")
            
            # Low cognitive load
            if len(self.cognitive_history) > 0:
                recent_load = list(self.cognitive_history)[-5:]
                low_load_count = sum(1 for c in recent_load if c.cognitive_load == CognitiveLoad.LOW)
                if low_load_count >= 3:
                    indicators.append("low_cognitive_load")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Confidence indicators identification error: {e}")
            return []
    
    def get_emotional_trends(self, minutes: int = 5) -> Dict[str, Any]:
        """Get emotional trends over specified time period"""
        try:
            cutoff_time = time.time() - (minutes * 60)
            recent_emotions = [e for e in self.emotion_history if e.timestamp > cutoff_time]
            
            if not recent_emotions:
                return {}
            
            # Emotion frequency
            emotion_counts = {}
            for analysis in recent_emotions:
                emotion = analysis.primary_emotion
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Sentiment trends
            sentiments = [e.sentiment.polarity for e in recent_emotions]
            sentiment_distribution = {
                'positive': sentiments.count(SentimentPolarity.POSITIVE),
                'negative': sentiments.count(SentimentPolarity.NEGATIVE),
                'neutral': sentiments.count(SentimentPolarity.NEUTRAL)
            }
            
            # Average metrics
            avg_stability = np.mean([e.emotional_stability for e in recent_emotions])
            avg_positivity = np.mean([e.sentiment.positivity_score for e in recent_emotions])
            avg_arousal = np.mean([e.sentiment.arousal for e in recent_emotions])
            
            # Cognitive load distribution
            cognitive_loads = [e.cognitive_state.cognitive_load for e in recent_emotions]
            load_distribution = {
                'low': cognitive_loads.count(CognitiveLoad.LOW),
                'medium': cognitive_loads.count(CognitiveLoad.MEDIUM),
                'high': cognitive_loads.count(CognitiveLoad.HIGH),
                'overload': cognitive_loads.count(CognitiveLoad.OVERLOAD)
            }
            
            return {
                'emotion_frequency': {e.value: count for e, count in emotion_counts.items()},
                'sentiment_distribution': {k.value: v for k, v in sentiment_distribution.items()},
                'cognitive_load_distribution': {k.value: v for k, v in load_distribution.items()},
                'average_stability': avg_stability,
                'average_positivity': avg_positivity,
                'average_arousal': avg_arousal,
                'sample_count': len(recent_emotions)
            }
            
        except Exception as e:
            logger.error(f"Emotional trends analysis error: {e}")
            return {}
    
    def get_emotional_insights(self) -> Dict[str, Any]:
        """Get actionable emotional insights"""
        try:
            if not self.emotion_history:
                return {"insights": [], "recommendations": []}
            
            recent_analysis = list(self.emotion_history)[-20:]  # Last 20 analyses
            
            insights = []
            recommendations = []
            
            # Analyze stress patterns
            stress_events = []
            for analysis in recent_analysis:
                if analysis.stress_indicators:
                    stress_events.append(len(analysis.stress_indicators))
            
            if stress_events:
                avg_stress = np.mean(stress_events)
                if avg_stress > 2:
                    insights.append("High stress levels detected during assessment")
                    recommendations.append("Consider taking short breaks to reduce stress")
            
            # Analyze confidence patterns
            confidence_events = []
            for analysis in recent_analysis:
                if analysis.confidence_indicators:
                    confidence_events.append(len(analysis.confidence_indicators))
            
            if confidence_events:
                avg_confidence = np.mean(confidence_events)
                if avg_confidence < 2:
                    insights.append("Low confidence indicators detected")
                    recommendations.append("Review material to build confidence")
            
            # Analyze emotional stability
            stability_scores = [a.emotional_stability for a in recent_analysis]
            avg_stability = np.mean(stability_scores)
            
            if avg_stability < 0.5:
                insights.append("Emotional state appears unstable")
                recommendations.append("Focus on calming techniques before assessment")
            
            # Analyze cognitive load
            cognitive_loads = [a.cognitive_state.cognitive_load for a in recent_analysis]
            high_load_count = sum(1 for load in cognitive_loads if load in [CognitiveLoad.HIGH, CognitiveLoad.OVERLOAD])
            
            if high_load_count > len(cognitive_loads) * 0.6:
                insights.append("High cognitive load sustained")
                recommendations.append("Break down complex problems into smaller steps")
            
            return {
                "insights": insights,
                "recommendations": recommendations,
                "emotional_health_score": avg_stability,
                "stress_level": avg_stress if stress_events else 0.0,
                "confidence_level": avg_confidence if confidence_events else 0.5
            }
            
        except Exception as e:
            logger.error(f"Emotional insights error: {e}")
            return {"insights": [], "recommendations": []}
    
    def _create_default_result(self) -> EmotionalIntelligenceResult:
        """Create default result for error cases"""
        return EmotionalIntelligenceResult(
            timestamp=time.time(),
            primary_emotion=EmotionType.NEUTRAL,
            emotion_scores=[EmotionScore(EmotionType.NEUTRAL, 0.3, 0.8, 0.0)],
            sentiment=SentimentAnalysis(SentimentPolarity.NEUTRAL, 0.3, 0.3, 0.4, 0.3, 0.0),
            cognitive_state=CognitiveState(CognitiveLoad.MEDIUM, 0.5, 0.3, 0.5, 0.5),
            emotional_stability=0.7,
            stress_indicators=[],
            confidence_indicators=[]
        )
    
    def cleanup(self):
        """Clean up emotional intelligence resources"""
        try:
            self.emotion_history.clear()
            self.sentiment_history.clear()
            self.cognitive_history.clear()
            logger.info("Emotional Intelligence Engine cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Factory function
def create_emotional_intelligence_engine() -> EmotionalIntelligenceEngine:
    """Create and return emotional intelligence engine instance"""
    return EmotionalIntelligenceEngine()

# Test function
if __name__ == "__main__":
    # Test the emotional intelligence engine
    engine = create_emotional_intelligence_engine()
    
    # Create mock visual result
    from computer_vision import VisualAnalysisResult, FaceAnalysis, EyeAnalysis, HeadPoseAnalysis, FacialExpressionAnalysis, BehavioralMetrics, AttentionLevel, EngagementState
    
    mock_result = VisualAnalysisResult(
        timestamp=time.time(),
        face_analysis=FaceAnalysis(True, 1, 0.8, (100, 100, 200, 200), 0.15),
        eye_analysis=EyeAnalysis(True, True, (0.1, 0.1), 0.8, 15.0, 0.7),
        head_pose=HeadPoseAnalysis(5.0, 3.0, 2.0, 0.8),
        expression=FacialExpressionAnalysis("happy", 0.7, True, False, False),
        behavior=BehavioralMetrics(0.2, 0.8, AttentionLevel.HIGH, EngagementState.ENGAGED, 0, []),
        frame_quality=0.9
    )
    
    # Analyze emotional state
    result = engine.analyze_emotional_state(mock_result, 25.0)
    
    print("Emotional Intelligence Engine Test:")
    print(f"Primary emotion: {result.primary_emotion.value}")
    print(f"Sentiment polarity: {result.sentiment.polarity.value}")
    print(f"Emotional stability: {result.emotional_stability:.2f}")
    print(f"Cognitive load: {result.cognitive_state.cognitive_load.value}")
    print(f"Stress indicators: {len(result.stress_indicators)}")
    print(f"Confidence indicators: {len(result.confidence_indicators)}")
    
    # Get insights
    insights = engine.get_emotional_insights()
    print(f"Insights: {len(insights['insights'])}")
    print(f"Recommendations: {len(insights['recommendations'])}")
    
    # Cleanup
    engine.cleanup()
