"""
Computer Vision Engine for Phase 2
Real-time visual analysis using MediaPipe for advanced proctoring
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class EngagementState(Enum):
    ENGAGED = "engaged"
    DISTRACTED = "distracted"
    ABSENT = "absent"

@dataclass
class FaceAnalysis:
    """Face detection and analysis results"""
    face_detected: bool
    face_count: int
    face_confidence: float
    face_rect: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    face_size_ratio: float  # face size relative to frame

@dataclass
class EyeAnalysis:
    """Eye tracking and gaze analysis"""
    left_eye_detected: bool
    right_eye_detected: bool
    gaze_direction: Tuple[float, float]  # horizontal, vertical (-1 to 1)
    eye_contact: float  # 0-1 scale
    blink_rate: float  # blinks per minute
    eye_openness: float  # 0-1 scale

@dataclass
class HeadPoseAnalysis:
    """Head pose estimation"""
    pitch: float  # nodding (-30 to 30 degrees)
    yaw: float  # turning (-45 to 45 degrees)
    roll: float  # tilting (-30 to 30 degrees)
    pose_confidence: float  # 0-1 scale

@dataclass
class FacialExpressionAnalysis:
    """Facial expression and emotion detection"""
    expression: str  # neutral, happy, sad, angry, surprised, confused
    emotion_confidence: float  # 0-1 scale
    smile_detected: bool
    frown_detected: bool
    eyebrow_raised: bool

@dataclass
class BehavioralMetrics:
    """Behavioral analysis metrics"""
    movement_intensity: float  # 0-1 scale
    posture_score: float  # 0-1 scale
    attention_level: AttentionLevel
    engagement_state: EngagementState
    distraction_events: int
    suspicious_activities: List[str]

@dataclass
class VisualAnalysisResult:
    """Complete visual analysis result"""
    timestamp: float
    face_analysis: FaceAnalysis
    eye_analysis: EyeAnalysis
    head_pose: HeadPoseAnalysis
    expression: FacialExpressionAnalysis
    behavior: BehavioralMetrics
    frame_quality: float  # 0-1 scale

class ComputerVisionEngine:
    """Advanced computer vision engine using MediaPipe"""
    
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_holistic = mp.solutions.holistic
        
        # Face detection model
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Face mesh for detailed analysis
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Holistic for full body analysis
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Historical data for trend analysis
        self.gaze_history = deque(maxlen=30)  # Last 30 frames
        self.pose_history = deque(maxlen=60)  # Last 60 frames
        self.expression_history = deque(maxlen=20)  # Last 20 frames
        self.movement_history = deque(maxlen=15)  # Last 15 frames
        
        # Blink detection
        self.blink_timestamps = deque(maxlen=10)
        self.last_eye_openness = 0.5
        
        # Configuration
        self.frame_count = 0
        self.start_time = time.time()
        
        logger.info("Computer Vision Engine initialized with MediaPipe")
    
    def analyze_frame(self, frame: np.ndarray) -> VisualAnalysisResult:
        """Perform comprehensive visual analysis on a single frame"""
        try:
            timestamp = time.time()
            self.frame_count += 1
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Analyze face
            face_analysis = self._analyze_face(rgb_frame, frame.shape)
            
            # Analyze eyes and gaze
            eye_analysis = self._analyze_eyes(rgb_frame, face_analysis)
            
            # Analyze head pose
            head_pose = self._analyze_head_pose(rgb_frame, face_analysis)
            
            # Analyze facial expressions
            expression = self._analyze_expressions(rgb_frame, face_analysis)
            
            # Analyze behavior
            behavior = self._analyze_behavior(frame, face_analysis, eye_analysis, head_pose)
            
            # Assess frame quality
            frame_quality = self._assess_frame_quality(frame)
            
            return VisualAnalysisResult(
                timestamp=timestamp,
                face_analysis=face_analysis,
                eye_analysis=eye_analysis,
                head_pose=head_pose,
                expression=expression,
                behavior=behavior,
                frame_quality=frame_quality
            )
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return self._create_default_result()
    
    def _analyze_face(self, rgb_frame: np.ndarray, frame_shape: Tuple[int, int, int]) -> FaceAnalysis:
        """Detect and analyze faces"""
        try:
            # Face detection
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]
                
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w = frame_shape[:2]
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Calculate face size ratio
                face_area = width * height
                frame_area = w * h
                face_size_ratio = face_area / frame_area
                
                return FaceAnalysis(
                    face_detected=True,
                    face_count=len(results.detections),
                    face_confidence=detection.score[0],
                    face_rect=(x, y, width, height),
                    face_size_ratio=face_size_ratio
                )
            else:
                return FaceAnalysis(
                    face_detected=False,
                    face_count=0,
                    face_confidence=0.0,
                    face_rect=None,
                    face_size_ratio=0.0
                )
                
        except Exception as e:
            logger.error(f"Face analysis error: {e}")
            return FaceAnalysis(False, 0, 0.0, None, 0.0)
    
    def _analyze_eyes(self, rgb_frame: np.ndarray, face_analysis: FaceAnalysis) -> EyeAnalysis:
        """Analyze eyes and gaze direction"""
        try:
            if not face_analysis.face_detected:
                return EyeAnalysis(False, False, (0.0, 0.0), 0.0, 0.0, 0.0)
            
            # Face mesh for detailed eye analysis
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Eye landmarks indices
                left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
                
                # Calculate eye openness
                left_eye_openness = self._calculate_eye_openness(landmarks, left_eye_indices)
                right_eye_openness = self._calculate_eye_openness(landmarks, right_eye_indices)
                avg_openness = (left_eye_openness + right_eye_openness) / 2
                
                # Detect blinks
                current_blink = self._detect_blink(avg_openness)
                
                # Calculate gaze direction
                gaze_x, gaze_y = self._calculate_gaze_direction(landmarks, left_eye_indices, right_eye_indices)
                
                # Calculate eye contact (simplified - based on gaze centering)
                eye_contact = 1.0 - (abs(gaze_x) + abs(gaze_y)) / 2
                
                # Calculate blink rate
                blink_rate = len(self.blink_timestamps) / max(1, (time.time() - self.start_time) / 60)
                
                return EyeAnalysis(
                    left_eye_detected=left_eye_openness > 0.1,
                    right_eye_detected=right_eye_openness > 0.1,
                    gaze_direction=(gaze_x, gaze_y),
                    eye_contact=max(0.0, eye_contact),
                    blink_rate=blink_rate,
                    eye_openness=avg_openness
                )
            else:
                return EyeAnalysis(False, False, (0.0, 0.0), 0.0, 0.0, 0.0)
                
        except Exception as e:
            logger.error(f"Eye analysis error: {e}")
            return EyeAnalysis(False, False, (0.0, 0.0), 0.0, 0.0, 0.0)
    
    def _calculate_eye_openness(self, landmarks: Any, eye_indices: List[int]) -> float:
        """Calculate eye openness based on landmark positions"""
        try:
            # Get eye landmarks
            eye_points = []
            for idx in eye_indices:
                if idx < len(landmarks.landmark):
                    eye_points.append([landmarks.landmark[idx].x, landmarks.landmark[idx].y])
            
            if len(eye_points) < 6:
                return 0.0
            
            # Calculate vertical eye opening (simplified)
            eye_points = np.array(eye_points)
            
            # Top and bottom eye points
            top_points = eye_points[[1, 2, 5, 6]]  # Upper eye landmarks
            bottom_points = eye_points[[0, 3, 4, 7]]  # Lower eye landmarks
            
            if len(top_points) > 0 and len(bottom_points) > 0:
                avg_y_top = np.mean(top_points[:, 1])
                avg_y_bottom = np.mean(bottom_points[:, 1])
                eye_height = abs(avg_y_bottom - avg_y_top)
                
                # Normalize (typical eye height is around 0.05-0.15 in normalized coordinates)
                normalized_height = min(1.0, eye_height / 0.1)
                return max(0.0, normalized_height)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Eye openness calculation error: {e}")
            return 0.0
    
    def _detect_blink(self, current_openness: float) -> bool:
        """Detect blink based on eye openness change"""
        try:
            # Blink detection threshold
            blink_threshold = 0.2
            
            # Check for rapid closure and opening
            if self.last_eye_openness > blink_threshold and current_openness < blink_threshold:
                self.blink_timestamps.append(time.time())
                return True
            
            self.last_eye_openness = current_openness
            return False
            
        except Exception as e:
            logger.error(f"Blink detection error: {e}")
            return False
    
    def _calculate_gaze_direction(self, landmarks: Any, left_indices: List[int], right_indices: List[int]) -> Tuple[float, float]:
        """Calculate gaze direction from eye landmarks"""
        try:
            # Get iris center (simplified - using eye center)
            left_eye_center = self._get_eye_center(landmarks, left_indices)
            right_eye_center = self._get_eye_center(landmarks, right_indices)
            
            if left_eye_center is None or right_eye_center is None:
                return (0.0, 0.0)
            
            # Average eye center
            avg_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
            avg_center_y = (left_eye_center[1] + right_eye_center[1]) / 2
            
            # Calculate gaze offset from center (0.5, 0.5 is center)
            gaze_x = (avg_center_x - 0.5) * 2  # -1 to 1 scale
            gaze_y = (avg_center_y - 0.5) * 2  # -1 to 1 scale
            
            # Store in history
            self.gaze_history.append((gaze_x, gaze_y))
            
            return (np.clip(gaze_x, -1, 1), np.clip(gaze_y, -1, 1))
            
        except Exception as e:
            logger.error(f"Gaze calculation error: {e}")
            return (0.0, 0.0)
    
    def _get_eye_center(self, landmarks: Any, eye_indices: List[int]) -> Optional[Tuple[float, float]]:
        """Get eye center from landmarks"""
        try:
            eye_points = []
            for idx in eye_indices:
                if idx < len(landmarks.landmark):
                    eye_points.append([landmarks.landmark[idx].x, landmarks.landmark[idx].y])
            
            if len(eye_points) > 0:
                eye_points = np.array(eye_points)
                return (np.mean(eye_points[:, 0]), np.mean(eye_points[:, 1]))
            
            return None
            
        except Exception as e:
            logger.error(f"Eye center calculation error: {e}")
            return None
    
    def _analyze_head_pose(self, rgb_frame: np.ndarray, face_analysis: FaceAnalysis) -> HeadPoseAnalysis:
        """Analyze head pose estimation"""
        try:
            if not face_analysis.face_detected:
                return HeadPoseAnalysis(0.0, 0.0, 0.0, 0.0)
            
            # Use face mesh for pose estimation
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Key facial landmarks for pose estimation
                nose_tip = landmarks.landmark[1]
                chin = landmarks.landmark[175]
                left_eye_corner = landmarks.landmark[33]
                right_eye_corner = landmarks.landmark[263]
                left_mouth_corner = landmarks.landmark[61]
                right_mouth_corner = landmarks.landmark[291]
                
                # Calculate pose angles (simplified 2D approximation)
                # Yaw (left-right rotation)
                eye_center_x = (left_eye_corner.x + right_eye_corner.x) / 2
                mouth_center_x = (left_mouth_corner.x + right_mouth_corner.x) / 2
                face_center_x = (eye_center_x + mouth_center_x) / 2
                
                yaw = (face_center_x - 0.5) * 90  # Convert to degrees
                
                # Pitch (up-down rotation)
                eye_y = (left_eye_corner.y + right_eye_corner.y) / 2
                mouth_y = (left_mouth_corner.y + right_mouth_corner.y) / 2
                vertical_center = (eye_y + mouth_y) / 2
                
                pitch = (vertical_center - 0.5) * 60  # Convert to degrees
                
                # Roll (tilt) - using eye line angle
                eye_line_angle = math.atan2(
                    right_eye_corner.y - left_eye_corner.y,
                    right_eye_corner.x - left_eye_corner.x
                )
                roll = math.degrees(eye_line_angle)
                
                # Calculate confidence based on face size and detection quality
                confidence = min(1.0, face_analysis.face_confidence * face_analysis.face_size_ratio * 10)
                
                # Store in history
                self.pose_history.append((pitch, yaw, roll))
                
                return HeadPoseAnalysis(
                    pitch=np.clip(pitch, -30, 30),
                    yaw=np.clip(yaw, -45, 45),
                    roll=np.clip(roll, -30, 30),
                    confidence=confidence
                )
            else:
                return HeadPoseAnalysis(0.0, 0.0, 0.0, 0.0)
                
        except Exception as e:
            logger.error(f"Head pose analysis error: {e}")
            return HeadPoseAnalysis(0.0, 0.0, 0.0, 0.0)
    
    def _analyze_expressions(self, rgb_frame: np.ndarray, face_analysis: FaceAnalysis) -> FacialExpressionAnalysis:
        """Analyze facial expressions and emotions"""
        try:
            if not face_analysis.face_detected:
                return FacialExpressionAnalysis("neutral", 0.0, False, False, False)
            
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Key facial points for expression analysis
                left_mouth_corner = landmarks.landmark[61]
                right_mouth_corner = landmarks.landmark[291]
                mouth_top = landmarks.landmark[13]
                mouth_bottom = landmarks.landmark[14]
                left_eyebrow = landmarks.landmark[70]
                right_eyebrow = landmarks.landmark[300]
                nose_tip = landmarks.landmark[1]
                
                # Smile detection (mouth corner curvature)
                mouth_width = abs(right_mouth_corner.x - left_mouth_corner.x)
                mouth_height = abs(mouth_bottom.y - mouth_top.y)
                mouth_curvature = mouth_height / max(mouth_width, 0.001)
                
                smile_detected = mouth_curvature > 0.15
                
                # Frown detection (opposite of smile)
                frown_detected = mouth_curvature < 0.05 and mouth_height > 0.02
                
                # Eyebrow raise detection
                eyebrow_height = (left_eyebrow.y + right_eyebrow.y) / 2
                eyebrow_raised = eyebrow_height < 0.3
                
                # Simple expression classification
                if smile_detected and not eyebrow_raised:
                    expression = "happy"
                    confidence = 0.7
                elif frown_detected:
                    expression = "sad"
                    confidence = 0.6
                elif eyebrow_raised and not smile_detected:
                    expression = "surprised"
                    confidence = 0.5
                else:
                    expression = "neutral"
                    confidence = 0.8
                
                # Store in history
                self.expression_history.append(expression)
                
                return FacialExpressionAnalysis(
                    expression=expression,
                    emotion_confidence=confidence,
                    smile_detected=smile_detected,
                    frown_detected=frown_detected,
                    eyebrow_raised=eyebrow_raised
                )
            else:
                return FacialExpressionAnalysis("neutral", 0.0, False, False, False)
                
        except Exception as e:
            logger.error(f"Expression analysis error: {e}")
            return FacialExpressionAnalysis("neutral", 0.0, False, False, False)
    
    def _analyze_behavior(
        self, 
        frame: np.ndarray, 
        face_analysis: FaceAnalysis, 
        eye_analysis: EyeAnalysis, 
        head_pose: HeadPoseAnalysis
    ) -> BehavioralMetrics:
        """Analyze behavioral patterns and detect suspicious activities"""
        try:
            # Movement intensity analysis
            movement_intensity = self._calculate_movement_intensity(frame)
            
            # Posture scoring
            posture_score = self._calculate_posture_score(head_pose, face_analysis)
            
            # Attention level based on eye contact and head pose
            attention_score = (eye_analysis.eye_contact + (1 - abs(head_pose.yaw) / 45)) / 2
            
            if attention_score > 0.7:
                attention_level = AttentionLevel.HIGH
            elif attention_score > 0.4:
                attention_level = AttentionLevel.MEDIUM
            else:
                attention_level = AttentionLevel.LOW
            
            # Engagement state
            if face_analysis.face_detected and attention_score > 0.5:
                engagement_state = EngagementState.ENGAGED
            elif face_analysis.face_detected:
                engagement_state = EngagementState.DISTRACTED
            else:
                engagement_state = EngagementState.ABSENT
            
            # Detect suspicious activities
            suspicious_activities = self._detect_suspicious_activities(
                face_analysis, eye_analysis, head_pose, movement_intensity
            )
            
            return BehavioralMetrics(
                movement_intensity=movement_intensity,
                posture_score=posture_score,
                attention_level=attention_level,
                engagement_state=engagement_state,
                distraction_events=len(suspicious_activities),
                suspicious_activities=suspicious_activities
            )
            
        except Exception as e:
            logger.error(f"Behavior analysis error: {e}")
            return BehavioralMetrics(0.0, 0.0, AttentionLevel.LOW, EngagementState.ABSENT, 0, [])
    
    def _calculate_movement_intensity(self, frame: np.ndarray) -> float:
        """Calculate movement intensity between frames"""
        try:
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Store current frame for next comparison
            if hasattr(self, 'prev_gray_frame'):
                # Calculate optical flow (simplified frame difference)
                diff = cv2.absdiff(self.prev_gray_frame, gray)
                movement = np.mean(diff) / 255.0
                
                # Store in history
                self.movement_history.append(movement)
                
                # Return normalized movement intensity
                return min(1.0, movement * 2)
            else:
                movement = 0.0
            
            self.prev_gray_frame = gray
            return movement
            
        except Exception as e:
            logger.error(f"Movement calculation error: {e}")
            return 0.0
    
    def _calculate_posture_score(self, head_pose: HeadPoseAnalysis, face_analysis: FaceAnalysis) -> float:
        """Calculate posture score based on head pose and face position"""
        try:
            if not face_analysis.face_detected:
                return 0.0
            
            # Score based on head pose (ideal is centered)
            pose_score = 1.0 - (abs(head_pose.pitch) / 30 + abs(head_pose.yaw) / 45 + abs(head_pose.roll) / 30) / 3
            
            # Score based on face size (not too close, not too far)
            ideal_face_ratio = 0.15  # Ideal face occupies 15% of frame
            size_score = 1.0 - abs(face_analysis.face_size_ratio - ideal_face_ratio) / ideal_face_ratio
            
            # Combined posture score
            posture_score = (pose_score * 0.7 + size_score * 0.3)
            
            return max(0.0, min(1.0, posture_score))
            
        except Exception as e:
            logger.error(f"Posture calculation error: {e}")
            return 0.0
    
    def _detect_suspicious_activities(
        self, 
        face_analysis: FaceAnalysis, 
        eye_analysis: EyeAnalysis, 
        head_pose: HeadPoseAnalysis, 
        movement_intensity: float
    ) -> List[str]:
        """Detect suspicious activities for proctoring"""
        suspicious = []
        
        # Multiple faces detected
        if face_analysis.face_count > 1:
            suspicious.append("multiple_faces_detected")
        
        # No face detected
        if not face_analysis.face_detected:
            suspicious.append("no_face_detected")
        
        # Looking away consistently
        if len(self.gaze_history) > 10:
            recent_gaze = list(self.gaze_history)[-10:]
            avg_gaze_x = np.mean([g[0] for g in recent_gaze])
            avg_gaze_y = np.mean([g[1] for g in recent_gaze])
            
            if abs(avg_gaze_x) > 0.7 or abs(avg_gaze_y) > 0.7:
                suspicious.append("looking_away")
        
        # Excessive movement
        if movement_intensity > 0.5:
            suspicious.append("excessive_movement")
        
        # Unusual head pose
        if abs(head_pose.yaw) > 35 or abs(head_pose.pitch) > 25:
            suspicious.append("unusual_head_pose")
        
        # High blink rate (possible stress or distraction)
        if eye_analysis.blink_rate > 30:  # More than 30 blinks per minute
            suspicious.append("high_blink_rate")
        
        return suspicious
    
    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess overall frame quality"""
        try:
            # Calculate sharpness (Laplacian variance)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray) / 255.0
            
            # Calculate contrast
            contrast = np.std(gray) / 255.0
            
            # Combined quality score
            sharpness_score = min(1.0, sharpness / 100)  # Normalize sharpness
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Ideal brightness is 0.5
            contrast_score = min(1.0, contrast * 4)  # Normalize contrast
            
            quality_score = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Frame quality assessment error: {e}")
            return 0.5  # Default medium quality
    
    def _create_default_result(self) -> VisualAnalysisResult:
        """Create default result for error cases"""
        return VisualAnalysisResult(
            timestamp=time.time(),
            face_analysis=FaceAnalysis(False, 0, 0.0, None, 0.0),
            eye_analysis=EyeAnalysis(False, False, (0.0, 0.0), 0.0, 0.0, 0.0),
            head_pose=HeadPoseAnalysis(0.0, 0.0, 0.0, 0.0),
            expression=FacialExpressionAnalysis("neutral", 0.0, False, False, False),
            behavior=BehavioralMetrics(0.0, 0.0, AttentionLevel.LOW, EngagementState.ABSENT, 0, []),
            frame_quality=0.0
        )
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary metrics over time"""
        try:
            if not self.gaze_history and not self.pose_history:
                return {}
            
            summary = {
                "frames_processed": self.frame_count,
                "processing_time": time.time() - self.start_time,
                "avg_fps": self.frame_count / max(1, time.time() - self.start_time)
            }
            
            # Gaze statistics
            if self.gaze_history:
                gaze_x = [g[0] for g in self.gaze_history]
                gaze_y = [g[1] for g in self.gaze_history]
                summary["gaze_stats"] = {
                    "avg_x": np.mean(gaze_x),
                    "avg_y": np.mean(gaze_y),
                    "std_x": np.std(gaze_x),
                    "std_y": np.std(gaze_y)
                }
            
            # Pose statistics
            if self.pose_history:
                pitches = [p[0] for p in self.pose_history]
                yaws = [p[1] for p in self.pose_history]
                rolls = [p[2] for p in self.pose_history]
                summary["pose_stats"] = {
                    "avg_pitch": np.mean(pitches),
                    "avg_yaw": np.mean(yaws),
                    "avg_roll": np.mean(rolls),
                    "pose_stability": 1.0 - (np.std(pitches) + np.std(yaws) + np.std(rolls)) / 90
                }
            
            # Expression distribution
            if self.expression_history:
                expressions = list(self.expression_history)
                expression_counts = {}
                for expr in expressions:
                    expression_counts[expr] = expression_counts.get(expr, 0) + 1
                summary["expression_distribution"] = expression_counts
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary metrics error: {e}")
            return {}
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.face_detection.close()
            self.face_mesh.close()
            self.holistic.close()
            logger.info("Computer Vision Engine cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Factory function
def create_computer_vision_engine() -> ComputerVisionEngine:
    """Create and return computer vision engine instance"""
    return ComputerVisionEngine()

# Test function
if __name__ == "__main__":
    # Test the computer vision engine
    engine = create_computer_vision_engine()
    
    # Create a test frame (black image)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Analyze the test frame
    result = engine.analyze_frame(test_frame)
    
    print("Computer Vision Engine Test Results:")
    print(f"Face detected: {result.face_analysis.face_detected}")
    print(f"Eye contact: {result.eye_analysis.eye_contact:.2f}")
    print(f"Attention level: {result.behavior.attention_level.value}")
    print(f"Engagement state: {result.behavior.engagement_state.value}")
    
    # Get summary metrics
    summary = engine.get_summary_metrics()
    print(f"Summary metrics: {summary}")
    
    # Cleanup
    engine.cleanup()
