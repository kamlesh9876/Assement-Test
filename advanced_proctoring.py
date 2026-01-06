"""
Advanced AI Proctoring System for Phase 2
Intelligent monitoring and behavioral analysis using real computer vision
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import numpy as np
from datetime import datetime, timedelta

from computer_vision import (
    ComputerVisionEngine, 
    VisualAnalysisResult, 
    AttentionLevel, 
    EngagementState,
    create_computer_vision_engine
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProctoringSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    MULTIPLE_FACES = "multiple_faces"
    NO_FACE = "no_face"
    LOOKING_AWAY = "looking_away"
    EXCESSIVE_MOVEMENT = "excessive_movement"
    UNUSUAL_POSTURE = "unusual_posture"
    HIGH_STRESS = "high_stress"
    DISTRACTION = "distraction"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"

@dataclass
class ProctoringAlert:
    """Individual proctoring alert"""
    alert_type: AlertType
    severity: ProctoringSeverity
    timestamp: float
    description: str
    confidence: float  # 0-1 scale
    evidence: Dict[str, Any]
    resolved: bool = False

@dataclass
class ProctoringMetrics:
    """Overall proctoring metrics"""
    total_frames: int
    attention_score: float  # 0-1 scale
    engagement_score: float  # 0-1 scale
    integrity_score: float  # 0-1 scale
    stress_level: float  # 0-1 scale
    posture_score: float  # 0-1 scale
    movement_score: float  # 0-1 scale
    alert_count: int
    critical_alerts: int

@dataclass
class BehavioralPattern:
    """Detected behavioral pattern"""
    pattern_name: str
    frequency: float  # occurrences per minute
    duration: float  # average duration in seconds
    significance: float  # 0-1 scale
    description: str

class AdvancedProctoringSystem:
    """Advanced AI proctoring system with real-time analysis"""
    
    def __init__(self, sensitivity_level: float = 0.7):
        self.vision_engine = create_computer_vision_engine()
        self.sensitivity_level = sensitivity_level
        
        # Analysis history
        self.analysis_history = deque(maxlen=300)  # Last 5 minutes at 1fps
        self.alerts = deque(maxlen=100)  # Last 100 alerts
        self.behavioral_patterns = []
        
        # Thresholds for detection
        self.thresholds = {
            'attention_low': 0.3,
            'attention_medium': 0.6,
            'movement_high': 0.4,
            'posture_low': 0.4,
            'stress_high': 0.7,
            'gaze_offset': 0.6,
            'face_size_min': 0.05,
            'face_size_max': 0.4
        }
        
        # Pattern detection windows
        self.attention_window = deque(maxlen=60)  # 1 minute
        self.movement_window = deque(maxlen=30)   # 30 seconds
        self.posture_window = deque(maxlen=90)    # 1.5 minutes
        
        # Statistics
        self.start_time = time.time()
        self.last_analysis_time = 0
        self.frame_count = 0
        
        logger.info("Advanced Proctoring System initialized")
    
    def analyze_frame(self, frame: np.ndarray) -> Optional[VisualAnalysisResult]:
        """Analyze a single frame and update proctoring state"""
        try:
            # Perform computer vision analysis
            result = self.vision_engine.analyze_frame(frame)
            
            # Store in history
            self.analysis_history.append(result)
            self.frame_count += 1
            self.last_analysis_time = time.time()
            
            # Update behavioral windows
            self._update_behavioral_windows(result)
            
            # Detect alerts
            self._detect_alerts(result)
            
            # Update behavioral patterns
            self._update_behavioral_patterns()
            
            return result
            
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return None
    
    def _update_behavioral_windows(self, result: VisualAnalysisResult):
        """Update sliding windows for pattern detection"""
        try:
            # Attention window
            attention_score = result.behavior.attention_level
            if attention_score == AttentionLevel.HIGH:
                self.attention_window.append(1.0)
            elif attention_level == AttentionLevel.MEDIUM:
                self.attention_window.append(0.5)
            else:
                self.attention_window.append(0.0)
            
            # Movement window
            self.movement_window.append(result.behavior.movement_intensity)
            
            # Posture window
            self.posture_window.append(result.behavior.posture_score)
            
        except Exception as e:
            logger.error(f"Behavioral window update error: {e}")
    
    def _detect_alerts(self, result: VisualAnalysisResult):
        """Detect proctoring alerts from analysis results"""
        try:
            current_time = time.time()
            
            # Multiple faces detection
            if result.face_analysis.face_count > 1:
                self._create_alert(
                    AlertType.MULTIPLE_FACES,
                    ProctoringSeverity.HIGH,
                    f"Multiple faces detected: {result.face_analysis.face_count}",
                    result.face_analysis.face_confidence,
                    {"face_count": result.face_analysis.face_count}
                )
            
            # No face detection
            elif not result.face_analysis.face_detected:
                self._create_alert(
                    AlertType.NO_FACE,
                    ProctoringSeverity.CRITICAL,
                    "No face detected in frame",
                    1.0 - result.frame_quality,
                    {"frame_quality": result.frame_quality}
                )
            
            # Looking away detection
            if len(self.vision_engine.gaze_history) > 10:
                recent_gaze = list(self.vision_engine.gaze_history)[-10:]
                avg_gaze_x = np.mean([g[0] for g in recent_gaze])
                avg_gaze_y = np.mean([g[1] for g in recent_gaze])
                
                if abs(avg_gaze_x) > self.thresholds['gaze_offset'] or abs(avg_gaze_y) > self.thresholds['gaze_offset']:
                    self._create_alert(
                        AlertType.LOOKING_AWAY,
                        ProctoringSeverity.MEDIUM,
                        f"Consistently looking away: gaze ({avg_gaze_x:.2f}, {avg_gaze_y:.2f})",
                        min(1.0, (abs(avg_gaze_x) + abs(avg_gaze_y)) / 2),
                        {"gaze_x": avg_gaze_x, "gaze_y": avg_gaze_y}
                    )
            
            # Excessive movement detection
            if result.behavior.movement_intensity > self.thresholds['movement_high']:
                self._create_alert(
                    AlertType.EXCESSIVE_MOVEMENT,
                    ProctoringSeverity.MEDIUM,
                    f"Excessive movement detected: {result.behavior.movement_intensity:.2f}",
                    result.behavior.movement_intensity,
                    {"movement_intensity": result.behavior.movement_intensity}
                )
            
            # Unusual posture detection
            if result.behavior.posture_score < self.thresholds['posture_low']:
                self._create_alert(
                    AlertType.UNUSUAL_POSTURE,
                    ProctoringSeverity.LOW,
                    f"Unusual posture detected: score {result.behavior.posture_score:.2f}",
                    1.0 - result.behavior.posture_score,
                    {"posture_score": result.behavior.posture_score}
                )
            
            # High stress detection
            stress_level = self._calculate_stress_level(result)
            if stress_level > self.thresholds['stress_high']:
                self._create_alert(
                    AlertType.HIGH_STRESS,
                    ProctoringSeverity.MEDIUM,
                    f"High stress level detected: {stress_level:.2f}",
                    stress_level,
                    {"stress_level": stress_level}
                )
            
            # Distraction detection
            if result.behavior.engagement_state == EngagementState.DISTRACTED:
                self._create_alert(
                    AlertType.DISTRACTION,
                    ProctoringSeverity.LOW,
                    "User appears distracted",
                    0.6,
                    {"engagement_state": result.behavior.engagement_state.value}
                )
            
            # Suspicious behavior detection
            if result.behavior.suspicious_activities:
                for activity in result.behavior.suspicious_activities:
                    self._create_alert(
                        AlertType.SUSPICIOUS_BEHAVIOR,
                        ProctoringSeverity.MEDIUM,
                        f"Suspicious activity: {activity}",
                        0.7,
                        {"activity": activity}
                    )
            
        except Exception as e:
            logger.error(f"Alert detection error: {e}")
    
    def _create_alert(
        self, 
        alert_type: AlertType, 
        severity: ProctoringSeverity, 
        description: str, 
        confidence: float, 
        evidence: Dict[str, Any]
    ):
        """Create a new proctoring alert"""
        try:
            # Check if similar alert already exists recently
            recent_time = time.time() - 30  # Last 30 seconds
            recent_alerts = [a for a in self.alerts if a.timestamp > recent_time and a.alert_type == alert_type]
            
            if recent_alerts:
                # Update existing alert instead of creating duplicate
                recent_alerts[0].confidence = max(recent_alerts[0].confidence, confidence)
                recent_alerts[0].evidence.update(evidence)
                return
            
            # Create new alert
            alert = ProctoringAlert(
                alert_type=alert_type,
                severity=severity,
                timestamp=time.time(),
                description=description,
                confidence=confidence,
                evidence=evidence
            )
            
            self.alerts.append(alert)
            logger.info(f"Alert created: {alert_type.value} - {description}")
            
        except Exception as e:
            logger.error(f"Alert creation error: {e}")
    
    def _calculate_stress_level(self, result: VisualAnalysisResult) -> float:
        """Calculate stress level from various indicators"""
        try:
            stress_indicators = []
            
            # High blink rate
            if result.eye_analysis.blink_rate > 25:
                stress_indicators.append(min(1.0, result.eye_analysis.blink_rate / 40))
            
            # Low eye contact
            if result.eye_analysis.eye_contact < 0.3:
                stress_indicators.append(1.0 - result.eye_analysis.eye_contact)
            
            # Excessive movement
            if result.behavior.movement_intensity > 0.3:
                stress_indicators.append(result.behavior.movement_intensity)
            
            # Unusual head pose
            pose_deviation = (abs(result.head_pose.pitch) + abs(result.head_pose.yaw) + abs(result.head_pose.roll)) / 90
            if pose_deviation > 0.3:
                stress_indicators.append(pose_deviation)
            
            # Negative expression
            if result.expression.expression in ["sad", "angry", "confused"]:
                stress_indicators.append(0.7)
            
            # Average stress level
            if stress_indicators:
                return np.mean(stress_indicators)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Stress calculation error: {e}")
            return 0.0
    
    def _update_behavioral_patterns(self):
        """Update detected behavioral patterns"""
        try:
            current_time = time.time()
            
            # Analyze attention patterns
            if len(self.attention_window) >= 30:
                attention_trend = np.mean(list(self.attention_window))
                if attention_trend < 0.4:
                    self._add_pattern(
                        "low_attention",
                        frequency=self._calculate_frequency("low_attention"),
                        duration=60.0,
                        significance=1.0 - attention_trend,
                        description="Consistently low attention during assessment"
                    )
            
            # Analyze movement patterns
            if len(self.movement_window) >= 20:
                movement_avg = np.mean(list(self.movement_window))
                if movement_avg > 0.3:
                    self._add_pattern(
                        "restless_behavior",
                        frequency=self._calculate_frequency("restless_behavior"),
                        duration=30.0,
                        significance=movement_avg,
                        description="Frequent movement indicating restlessness"
                    )
            
            # Analyze posture patterns
            if len(self.posture_window) >= 45:
                posture_avg = np.mean(list(self.posture_window))
                if posture_avg < 0.5:
                    self._add_pattern(
                        "poor_posture",
                        frequency=self._calculate_frequency("poor_posture"),
                        duration=90.0,
                        significance=1.0 - posture_avg,
                        description="Consistently poor posture during assessment"
                    )
            
        except Exception as e:
            logger.error(f"Behavioral pattern update error: {e}")
    
    def _calculate_frequency(self, pattern_name: str) -> float:
        """Calculate frequency of a specific pattern"""
        try:
            # Simplified frequency calculation based on recent occurrences
            time_window = 300  # 5 minutes
            current_time = time.time()
            
            # Count recent occurrences (simplified)
            recent_alerts = [a for a in self.alerts if current_time - a.timestamp < time_window]
            
            if pattern_name == "low_attention":
                count = sum(1 for a in recent_alerts if a.alert_type in [AlertType.LOOKING_AWAY, AlertType.DISTRACTION])
            elif pattern_name == "restless_behavior":
                count = sum(1 for a in recent_alerts if a.alert_type == AlertType.EXCESSIVE_MOVEMENT)
            elif pattern_name == "poor_posture":
                count = sum(1 for a in recent_alerts if a.alert_type == AlertType.UNUSUAL_POSTURE)
            else:
                count = 0
            
            frequency = count / (time_window / 60)  # per minute
            return frequency
            
        except Exception as e:
            logger.error(f"Frequency calculation error: {e}")
            return 0.0
    
    def _add_pattern(self, pattern_name: str, frequency: float, duration: float, significance: float, description: str):
        """Add or update a behavioral pattern"""
        try:
            # Check if pattern already exists
            existing_pattern = next((p for p in self.behavioral_patterns if p.pattern_name == pattern_name), None)
            
            if existing_pattern:
                # Update existing pattern
                existing_pattern.frequency = (existing_pattern.frequency + frequency) / 2
                existing_pattern.significance = max(existing_pattern.significance, significance)
            else:
                # Add new pattern
                pattern = BehavioralPattern(
                    pattern_name=pattern_name,
                    frequency=frequency,
                    duration=duration,
                    significance=significance,
                    description=description
                )
                self.behavioral_patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Pattern addition error: {e}")
    
    def get_proctoring_metrics(self) -> ProctoringMetrics:
        """Get comprehensive proctoring metrics"""
        try:
            if not self.analysis_history:
                return ProctoringMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
            
            # Calculate average metrics
            recent_analyses = list(self.analysis_history)[-60:]  # Last 60 frames
            
            # Attention score
            attention_scores = []
            for analysis in recent_analyses:
                if analysis.behavior.attention_level == AttentionLevel.HIGH:
                    attention_scores.append(1.0)
                elif analysis.behavior.attention_level == AttentionLevel.MEDIUM:
                    attention_scores.append(0.5)
                else:
                    attention_scores.append(0.0)
            avg_attention = np.mean(attention_scores) if attention_scores else 0.0
            
            # Engagement score
            engagement_scores = []
            for analysis in recent_analyses:
                if analysis.behavior.engagement_state == EngagementState.ENGAGED:
                    engagement_scores.append(1.0)
                elif analysis.behavior.engagement_state == EngagementState.DISTRACTED:
                    engagement_scores.append(0.5)
                else:
                    engagement_scores.append(0.0)
            avg_engagement = np.mean(engagement_scores) if engagement_scores else 0.0
            
            # Integrity score (inverse of alerts)
            recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 300]  # Last 5 minutes
            alert_penalty = min(0.5, len(recent_alerts) * 0.1)
            integrity_score = max(0.0, 1.0 - alert_penalty)
            
            # Stress level
            stress_levels = [self._calculate_stress_level(a) for a in recent_analyses]
            avg_stress = np.mean(stress_levels) if stress_levels else 0.0
            
            # Posture score
            posture_scores = [a.behavior.posture_score for a in recent_analyses]
            avg_posture = np.mean(posture_scores) if posture_scores else 0.0
            
            # Movement score (inverse - lower movement is better)
            movement_scores = [1.0 - a.behavior.movement_intensity for a in recent_analyses]
            avg_movement = np.mean(movement_scores) if movement_scores else 0.0
            
            # Alert counts
            critical_alerts = sum(1 for a in self.alerts if a.severity == ProctoringSeverity.CRITICAL)
            
            return ProctoringMetrics(
                total_frames=self.frame_count,
                attention_score=avg_attention,
                engagement_score=avg_engagement,
                integrity_score=integrity_score,
                stress_level=avg_stress,
                posture_score=avg_posture,
                movement_score=avg_movement,
                alert_count=len(self.alerts),
                critical_alerts=critical_alerts
            )
            
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return ProctoringMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    
    def get_recent_alerts(self, minutes: int = 5) -> List[ProctoringAlert]:
        """Get recent alerts within specified time window"""
        try:
            cutoff_time = time.time() - (minutes * 60)
            recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
            
            # Sort by severity and timestamp
            severity_order = {ProctoringSeverity.CRITICAL: 4, ProctoringSeverity.HIGH: 3, 
                            ProctoringSeverity.MEDIUM: 2, ProctoringSeverity.LOW: 1}
            
            recent_alerts.sort(key=lambda a: (severity_order.get(a.severity, 0), a.timestamp), reverse=True)
            
            return recent_alerts
            
        except Exception as e:
            logger.error(f"Recent alerts retrieval error: {e}")
            return []
    
    def get_behavioral_summary(self) -> Dict[str, Any]:
        """Get summary of behavioral patterns and insights"""
        try:
            summary = {
                "session_duration": time.time() - self.start_time,
                "frames_analyzed": self.frame_count,
                "avg_fps": self.frame_count / max(1, time.time() - self.start_time),
                "behavioral_patterns": [],
                "recommendations": [],
                "risk_assessment": "low"
            }
            
            # Add behavioral patterns
            for pattern in self.behavioral_patterns:
                if pattern.significance > 0.3:  # Only include significant patterns
                    summary["behavioral_patterns"].append({
                        "name": pattern.pattern_name,
                        "frequency": pattern.frequency,
                        "significance": pattern.significance,
                        "description": pattern.description
                    })
            
            # Generate recommendations
            metrics = self.get_proctoring_metrics()
            
            if metrics.attention_score < 0.4:
                summary["recommendations"].append("Consider taking breaks to improve focus")
            
            if metrics.stress_level > 0.6:
                summary["recommendations"].append("Stress level is high - consider relaxation techniques")
            
            if metrics.posture_score < 0.5:
                summary["recommendations"].append("Adjust posture for better comfort and focus")
            
            if metrics.integrity_score < 0.7:
                summary["recommendations"].append("Ensure assessment environment is free from distractions")
            
            # Risk assessment
            if metrics.critical_alerts > 0:
                summary["risk_assessment"] = "high"
            elif metrics.alert_count > 10:
                summary["risk_assessment"] = "medium"
            else:
                summary["risk_assessment"] = "low"
            
            return summary
            
        except Exception as e:
            logger.error(f"Behavioral summary error: {e}")
            return {}
    
    def export_proctoring_data(self) -> Dict[str, Any]:
        """Export all proctoring data for analysis"""
        try:
            export_data = {
                "session_info": {
                    "start_time": self.start_time,
                    "duration": time.time() - self.start_time,
                    "sensitivity_level": self.sensitivity_level,
                    "total_frames": self.frame_count
                },
                "metrics": asdict(self.get_proctoring_metrics()),
                "alerts": [asdict(alert) for alert in self.alerts],
                "behavioral_patterns": [asdict(pattern) for pattern in self.behavioral_patterns],
                "vision_summary": self.vision_engine.get_summary_metrics()
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Data export error: {e}")
            return {}
    
    def resolve_alert(self, alert_index: int):
        """Mark an alert as resolved"""
        try:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].resolved = True
                logger.info(f"Alert resolved: {self.alerts[alert_index].alert_type.value}")
            
        except Exception as e:
            logger.error(f"Alert resolution error: {e}")
    
    def adjust_sensitivity(self, new_sensitivity: float):
        """Adjust proctoring sensitivity level"""
        try:
            self.sensitivity_level = max(0.1, min(1.0, new_sensitivity))
            
            # Adjust thresholds based on sensitivity
            base_thresholds = {
                'attention_low': 0.3,
                'attention_medium': 0.6,
                'movement_high': 0.4,
                'posture_low': 0.4,
                'stress_high': 0.7,
                'gaze_offset': 0.6,
                'face_size_min': 0.05,
                'face_size_max': 0.4
            }
            
            # Higher sensitivity = lower thresholds
            sensitivity_factor = 2.0 - self.sensitivity_level
            
            for key, base_value in base_thresholds.items():
                if key in ['attention_low', 'posture_low', 'face_size_min']:
                    self.thresholds[key] = base_value * sensitivity_factor
                elif key in ['attention_medium', 'movement_high', 'stress_high', 'gaze_offset', 'face_size_max']:
                    self.thresholds[key] = base_value * (2.0 - sensitivity_factor)
            
            logger.info(f"Sensitivity adjusted to {self.sensitivity_level:.2f}")
            
        except Exception as e:
            logger.error(f"Sensitivity adjustment error: {e}")
    
    def cleanup(self):
        """Clean up proctoring resources"""
        try:
            self.vision_engine.cleanup()
            self.analysis_history.clear()
            self.alerts.clear()
            self.behavioral_patterns.clear()
            logger.info("Advanced Proctoring System cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Factory function
def create_proctoring_system(sensitivity_level: float = 0.7) -> AdvancedProctoringSystem:
    """Create and return advanced proctoring system instance"""
    return AdvancedProctoringSystem(sensitivity_level)

# Test function
if __name__ == "__main__":
    # Test the proctoring system
    proctor = create_proctoring_system()
    
    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Analyze test frame
    result = proctor.analyze_frame(test_frame)
    
    print("Advanced Proctoring System Test:")
    print(f"Frames analyzed: {proctor.frame_count}")
    
    # Get metrics
    metrics = proctor.get_proctoring_metrics()
    print(f"Attention score: {metrics.attention_score:.2f}")
    print(f"Engagement score: {metrics.engagement_score:.2f}")
    print(f"Integrity score: {metrics.integrity_score:.2f}")
    
    # Get behavioral summary
    summary = proctor.get_behavioral_summary()
    print(f"Risk assessment: {summary.get('risk_assessment', 'unknown')}")
    print(f"Recommendations: {len(summary.get('recommendations', []))}")
    
    # Cleanup
    proctor.cleanup()
