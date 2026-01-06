"""
Integrated Assessment Component
Combines adaptive question generation and basic assessment features
"""

import streamlit as st
import time
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from adaptive_agent import (
    AdaptiveQuestionAgent, 
    UserContext, 
    VisualCues, 
    Question, 
    create_adaptive_question_agent
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedAssessmentSystem:
    """Main integrated assessment system with Phase 2 advanced features"""
    
    def __init__(self, llm_providers: Dict[str, Any]):
        self.llm_providers = llm_providers
        self.adaptive_agent = create_adaptive_question_agent(llm_providers)
        
        # Phase 2: Advanced AI components
        self.vision_engine = create_computer_vision_engine()
        self.proctoring_system = create_proctoring_system()
        self.emotional_engine = create_emotional_intelligence_engine()
        
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize integrated assessment session state"""
        if 'assessment_started' not in st.session_state:
            st.session_state.assessment_started = False
        if 'current_question' not in st.session_state:
            st.session_state.current_question = None
        if 'question_start_time' not in st.session_state:
            st.session_state.question_start_time = None
        if 'user_context' not in st.session_state:
            st.session_state.user_context = None
        if 'assessment_results' not in st.session_state:
            st.session_state.assessment_results = []
        if 'visual_cues_history' not in st.session_state:
            st.session_state.visual_cues_history = []
        
        # Phase 2: Advanced session state
        if 'proctoring_alerts' not in st.session_state:
            st.session_state.proctoring_alerts = []
        if 'emotional_history' not in st.session_state:
            st.session_state.emotional_history = []
        if 'last_frame_analysis' not in st.session_state:
            st.session_state.last_frame_analysis = None
    
    def start_assessment(self, user_data: Dict[str, Any]):
        """Start integrated assessment with camera and adaptive questions"""
        
        # Get assessment mode
        assessment_mode = st.session_state.get('assessment_mode', '🚀 Integrated AI Assessment (Camera + Adaptive)')
        
        # Initialize user context
        user_context = UserContext(
            user_id=user_data.get('email', 'anonymous'),
            current_score=0.0,
            questions_attempted=0,
            correct_answers=0,
            average_response_time=0.0,
            current_streak=0,
            weak_topics=[],
            strong_topics=[],
            confidence_level=0.5,
            engagement_level=0.7,
            stress_indicators=0.3
        )
        
        st.session_state.user_context = user_context
        st.session_state.assessment_started = True
        st.session_state.assessment_results = []
        st.session_state.visual_cues_history = []
        st.session_state.assessment_mode = assessment_mode
        
        # Initialize mode-specific components
        self._initialize_mode_specific_features(assessment_mode)
        
        # Generate first question
        self._generate_next_question()
        
        logger.info(f"Assessment started for user: {user_context.user_id} with mode: {assessment_mode}")
    
    def _initialize_mode_specific_features(self, assessment_mode: str):
        """Initialize features based on assessment mode"""
        try:
            if assessment_mode == "🚀 Integrated AI Assessment (Camera + Adaptive)":
                # Standard camera + adaptive features
                st.session_state.camera_active = True
                self.vision_engine.start_analysis()
                
            elif assessment_mode == "🤖 Agentic Assessment (AI-driven)":
                # Enhanced agentic features
                st.session_state.camera_active = True
                self.vision_engine.start_analysis()
                self.proctoring_system.start_proctoring()
                # Enable advanced AI agent behaviors
                self.adaptive_agent.enable_advanced_agents = True
                
            elif assessment_mode == "🎯 Multi-Modal Assessment (Voice + Gesture)":
                # Multi-modal features
                st.session_state.camera_active = True
                self.vision_engine.start_analysis()
                self.proctoring_system.start_proctoring()
                # Enable voice and gesture analysis
                if hasattr(self, 'voice_analysis_engine'):
                    self.voice_analysis_engine.start_voice_analysis()
                if hasattr(self, 'gesture_recognition_engine'):
                    self.gesture_recognition_engine.start_gesture_tracking()
                
            logger.info(f"Initialized features for mode: {assessment_mode}")
            
        except Exception as e:
            logger.error(f"Mode-specific initialization error: {e}")
            # Fallback to basic features
            st.session_state.camera_active = True
    
    def _generate_next_question(self):
        """Generate next adaptive question"""
        user_context = st.session_state.user_context
        
        # Get latest visual cues (mock for now, will be real in Phase 2)
        visual_cues = self._get_current_visual_cues()
        
        # Generate question
        question = self.adaptive_agent.generate_adaptive_question(
            user_context=user_context,
            visual_cues=visual_cues
        )
        
        st.session_state.current_question = question
        st.session_state.question_start_time = time.time()
        
        logger.info(f"Generated question: {question.id} - {question.topic} ({question.difficulty.value})")
    
    def _get_current_visual_cues(self) -> Optional[VisualCues]:
        """Get current visual cues from real computer vision analysis"""
        try:
            # Get the latest frame analysis from proctoring system
            if hasattr(self.proctoring_system, 'analysis_history') and self.proctoring_system.analysis_history:
                latest_analysis = list(self.proctoring_system.analysis_history)[-1]
                
                # Convert visual analysis to visual cues
                return VisualCues(
                    eye_contact=latest_analysis.eye_analysis.eye_contact,
                    attention_level=1.0 if latest_analysis.behavior.attention_level.value == "high" 
                                   else 0.5 if latest_analysis.behavior.attention_level.value == "medium"
                                   else 0.0,
                    stress_indicators=self._calculate_stress_from_analysis(latest_analysis),
                    confidence_level=self._calculate_confidence_from_analysis(latest_analysis),
                    distraction_count=len(latest_analysis.behavior.suspicious_activities),
                    posture_score=latest_analysis.behavior.posture_score
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Visual cues extraction error: {e}")
            return None
    
    def _calculate_stress_from_analysis(self, analysis) -> float:
        """Calculate stress level from visual analysis"""
        try:
            stress_indicators = []
            
            # High blink rate
            if analysis.eye_analysis.blink_rate > 25:
                stress_indicators.append(0.3)
            
            # Low eye contact
            if analysis.eye_analysis.eye_contact < 0.3:
                stress_indicators.append(0.2)
            
            # Excessive movement
            if analysis.behavior.movement_intensity > 0.4:
                stress_indicators.append(0.2)
            
            # Unusual head pose
            if abs(analysis.head_pose.yaw) > 30 or abs(analysis.head_pose.pitch) > 20:
                stress_indicators.append(0.3)
            
            return min(1.0, sum(stress_indicators))
            
        except Exception as e:
            logger.error(f"Stress calculation error: {e}")
            return 0.0
    
    def _calculate_confidence_from_analysis(self, analysis) -> float:
        """Calculate confidence level from visual analysis"""
        try:
            confidence_indicators = []
            
            # Good eye contact
            if analysis.eye_analysis.eye_contact > 0.7:
                confidence_indicators.append(0.3)
            
            # Good posture
            if analysis.behavior.posture_score > 0.7:
                confidence_indicators.append(0.2)
            
            # Low movement (calm)
            if analysis.behavior.movement_intensity < 0.2:
                confidence_indicators.append(0.2)
            
            # Positive expression
            if analysis.expression.expression in ["happy", "neutral"]:
                confidence_indicators.append(0.3)
            
            return min(1.0, sum(confidence_indicators))
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0.5
    
    def show_simplified_assessment(self):
        """Show simplified assessment interface for faster loading"""
        
        # Get current question
        current_question = st.session_state.current_question
        user_context = st.session_state.user_context
        
        if not current_question:
            st.error("No question available. Please restart the assessment.")
            return
        
        # Simple progress bar
        progress = len(st.session_state.assessment_results) / 20
        st.progress(progress)
        st.write(f"Question {len(st.session_state.assessment_results) + 1} of 20")
        
        # Simple question display
        st.subheader(f"📝 {current_question.topic}")
        st.write(current_question.question_text)
        
        # Simple answer options
        if current_question.options:
            selected_answer = st.radio(
                "Choose your answer:",
                current_question.options,
                key=f"q_{current_question.id}"
            )
            
            # Submit button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Answer", type="primary"):
                    self._submit_answer(selected_answer)
            with col2:
                if st.button("Skip Question"):
                    self._submit_answer(None)
        
        # Simple metrics sidebar
        with st.sidebar:
            st.subheader("📊 Your Progress")
            st.metric("Score", f"{user_context.current_score*100:.1f}%")
            st.metric("Streak", user_context.current_streak)
            st.metric("Avg Time", f"{user_context.average_response_time:.1f}s")
            
            # Simple camera status
            if st.session_state.get('camera_active', False):
                st.success("📷 Camera Active")
            else:
                st.info("📷 Camera Off")
        
        # Simple controls
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏸️ Pause"):
                st.info("Assessment paused")
        with col2:
            if st.button("🏁 Finish"):
                self._finish_assessment()
    
    def show_integrated_assessment(self):
        """Display integrated assessment interface"""
        
        if not st.session_state.assessment_started:
            st.error("Assessment not started. Please register first.")
            return
        
        # Main layout with camera and assessment
        col1, col2 = st.columns([7, 3])
        
        with col1:
            self._show_assessment_content()
        
        with col2:
            self._show_camera_panel()
        
        # Show assessment controls
        self._show_assessment_controls()
    
    def _show_assessment_content(self):
        """Display assessment questions and content"""
        user_context = st.session_state.user_context
        current_question = st.session_state.current_question
        
        if not current_question:
            st.error("No question available.")
            return
        
        # Progress header
        self._show_progress_header()
        
        # Question display
        st.markdown("### 📝 Question")
        st.markdown(f"**Topic:** {current_question.topic.title()}")
        st.markdown(f"**Difficulty:** {current_question.difficulty.value.title()}")
        st.markdown(f"**Points:** {current_question.points}")
        
        st.markdown("---")
        
        # Question text
        st.markdown(f"#### {current_question.question_text}")
        
        # Answer options
        if current_question.type.value == "multiple_choice" and current_question.options:
            self._show_multiple_choice(current_question)
        
        # Timer
        self._show_question_timer(current_question)
    
    def _show_multiple_choice(self, question: Question):
        """Display multiple choice options"""
        
        selected_option = st.radio(
            "Select your answer:",
            options=question.options,
            key=f"q_{question.id}"
        )
        
        # Submit button
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Submit Answer", key=f"submit_{question.id}", type="primary"):
                self._submit_answer(selected_option)
        
        with col2:
            if st.button("Skip Question", key=f"skip_{question.id}"):
                self._skip_question()
    
    def _show_question_timer(self, question: Question):
        """Display question timer"""
        
        if st.session_state.question_start_time:
            elapsed = time.time() - st.session_state.question_start_time
            remaining = max(0, question.time_limit - elapsed)
            
            # Progress bar for timer
            progress = remaining / question.time_limit
            
            if remaining > 10:
                st.progress(progress, f"Time remaining: {int(remaining)}s")
            elif remaining > 0:
                st.progress(progress, f"⚠️ Time remaining: {int(remaining)}s")
            else:
                st.progress(0, "⏰ Time's up!")
                # Auto-submit if time expires
                self._submit_answer(None)
    
    def _show_progress_header(self):
        """Show assessment progress header"""
        user_context = st.session_state.user_context
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Questions", f"{user_context.questions_attempted}/20")
        
        with col2:
            st.metric("Score", f"{user_context.current_score:.1%}")
        
        with col3:
            st.metric("Streak", f"{user_context.current_streak}")
        
        with col4:
            st.metric("Avg Time", f"{user_context.average_response_time:.1f}s")
    
    def _show_camera_panel(self):
        """Show assessment status panel"""
        st.markdown("### 📹 Assessment Status")
        
        # Show assessment status
        if st.session_state.get("assessment_active", False):
            st.success("🟢 Assessment Active")
        else:
            st.info("⚪ Assessment Inactive")
            
            # Phase 2: Advanced Proctoring Metrics
            st.markdown("---")
            st.markdown("**🤖 AI Proctoring Metrics:**")
            
            # Get proctoring metrics
            proctoring_metrics = self.proctoring_system.get_proctoring_metrics()
            
            # Display key metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Attention", f"{proctoring_metrics.attention_score:.1%}")
                st.metric("Integrity", f"{proctoring_metrics.integrity_score:.1%}")
            with col2:
                st.metric("Engagement", f"{proctoring_metrics.engagement_score:.1%}")
                st.metric("Posture", f"{proctoring_metrics.posture_score:.1%}")
            
            # Stress level
            stress_color = "🔴" if proctoring_metrics.stress_level > 0.7 else "🟡" if proctoring_metrics.stress_level > 0.4 else "🟢"
            st.markdown(f"{stress_color} **Stress Level:** {proctoring_metrics.stress_level:.1%}")
            
            # Recent alerts
            recent_alerts = self.proctoring_system.get_recent_alerts(minutes=2)
            if recent_alerts:
                st.markdown("**⚠️ Recent Alerts:**")
                for alert in recent_alerts[:3]:  # Show top 3
                    severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "⚡", "low": "ℹ️"}
                    emoji = severity_emoji.get(alert.severity.value, "📢")
                    st.markdown(f"{emoji} {alert.description}")
            
            # Phase 2: Emotional Intelligence
            st.markdown("---")
            st.markdown("**💭 Emotional Intelligence:**")
            
            # Get latest emotional analysis
            if hasattr(self.emotional_engine, 'emotion_history') and self.emotional_engine.emotion_history:
                latest_emotional = list(self.emotional_engine.emotion_history)[-1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Primary Emotion:** {latest_emotional.primary_emotion.value.title()}")
                    st.markdown(f"**Sentiment:** {latest_emotional.sentiment.polarity.value.title()}")
                with col2:
                    st.markdown(f"**Stability:** {latest_emotional.emotional_stability:.1%}")
                    st.markdown(f"**Cognitive Load:** {latest_emotional.cognitive_state.cognitive_load.value.title()}")
                
                # Confidence and stress indicators
                if latest_emotional.confidence_indicators:
                    st.markdown("**✅ Confidence Indicators:**")
                    st.markdown(f", ".join(latest_emotional.confidence_indicators[:3]))
                
                if latest_emotional.stress_indicators:
                    st.markdown("**⚠️ Stress Indicators:**")
                    st.markdown(f", ".join(latest_emotional.stress_indicators[:3]))
            
            # Behavioral insights
            insights = self.emotional_engine.get_emotional_insights()
            if insights.get('recommendations'):
                st.markdown("**💡 Recommendations:**")
                for rec in insights['recommendations'][:2]:  # Show top 2
                    st.markdown(f"• {rec}")
            else:
                st.info("Advanced assessment features not available.")
    
    def _show_assessment_controls(self):
        """Show assessment control buttons"""
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📊 View Progress", key="view_progress"):
                self._show_detailed_progress()
        
        with col2:
            if st.button("⏸️ Pause Assessment", key="pause_assessment"):
                self._pause_assessment()
        
        with col3:
            if st.button("🏁 Finish Assessment", key="finish_assessment", type="secondary"):
                self._finish_assessment()
    
    def _submit_answer(self, selected_answer: Optional[str]):
        """Submit answer and update context"""
        current_question = st.session_state.current_question
        user_context = st.session_state.user_context
        
        # Calculate response time
        response_time = time.time() - st.session_state.question_start_time
        
        # Check if answer is correct
        is_correct = selected_answer == current_question.correct_answer
        
        # Create result entry
        result = {
            "question_id": current_question.id,
            "question_text": current_question.question_text,
            "selected_answer": selected_answer,
            "correct_answer": current_question.correct_answer,
            "is_correct": is_correct,
            "response_time": response_time,
            "points_earned": current_question.points if is_correct else 0,
            "difficulty": current_question.difficulty.value,
            "topic": current_question.topic,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store result
        st.session_state.assessment_results.append(result)
        
        # Update user context
        question_result = {
            "correct": is_correct,
            "response_time": response_time
        }
        
        updated_context = self.adaptive_agent.update_user_context(user_context, question_result)
        st.session_state.user_context = updated_context
        
        # Generate next question or finish
        if updated_context.questions_attempted >= 20:
            self._finish_assessment()
        else:
            self._generate_next_question()
        
        st.rerun()
    
    def _skip_question(self):
        """Skip current question"""
        self._submit_answer(None)
    
    def _pause_assessment(self):
        """Pause the assessment"""
        st.session_state.assessment_started = False
        st.info("Assessment paused. You can resume later.")
        st.rerun()
    
    def _finish_assessment(self):
        """Finish the assessment and show results"""
        st.session_state.assessment_started = False
        
        # Calculate final metrics
        results = st.session_state.assessment_results
        user_context = st.session_state.user_context
        
        final_metrics = {
            "total_questions": len(results),
            "correct_answers": user_context.correct_answers,
            "percentage": user_context.current_score * 100,
            "average_response_time": user_context.average_response_time,
            "topics_performance": self._calculate_topic_performance(results),
            "difficulty_performance": self._calculate_difficulty_performance(results),
            "assessment_time": sum(r["response_time"] for r in results)
        }
        
        # Store final result
        st.session_state.result = {
            "user_id": user_context.user_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": final_metrics,
            "detailed_results": results,
            "visual_cues_history": st.session_state.visual_cues_history
        }
        
        # Navigate to agentic results page for integrated assessments
        st.session_state.page = "agentic_results"
        st.rerun()
    
    def _calculate_topic_performance(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate performance by topic"""
        topic_stats = {}
        
        for result in results:
            topic = result["topic"]
            if topic not in topic_stats:
                topic_stats[topic] = {"correct": 0, "total": 0}
            
            topic_stats[topic]["total"] += 1
            if result["is_correct"]:
                topic_stats[topic]["correct"] += 1
        
        # Calculate percentages
        for topic, stats in topic_stats.items():
            stats["percentage"] = (stats["correct"] / stats["total"]) * 100
        
        return topic_stats
    
    def _calculate_difficulty_performance(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate performance by difficulty"""
        difficulty_stats = {}
        
        for result in results:
            difficulty = result["difficulty"]
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {"correct": 0, "total": 0}
            
            difficulty_stats[difficulty]["total"] += 1
            if result["is_correct"]:
                difficulty_stats[difficulty]["correct"] += 1
        
        # Calculate percentages
        for difficulty, stats in difficulty_stats.items():
            stats["percentage"] = (stats["correct"] / stats["total"]) * 100
        
        return difficulty_stats
    
    def process_frame_for_analysis(self, frame):
        """Process camera frame for real-time analysis (Phase 2)"""
        try:
            if not st.session_state.get('assessment_started', False):
                return
            
            # Analyze frame with computer vision
            visual_result = self.vision_engine.analyze_frame(frame)
            
            if visual_result:
                # Update proctoring system
                self.proctoring_system.analyze_frame(frame)
                
                # Analyze emotional state
                response_time = None
                if st.session_state.question_start_time:
                    response_time = time.time() - st.session_state.question_start_time
                
                emotional_result = self.emotional_engine.analyze_emotional_state(visual_result, response_time)
                
                # Store latest analysis
                st.session_state.last_frame_analysis = {
                    'visual': visual_result,
                    'emotional': emotional_result,
                    'timestamp': time.time()
                }
                
                # Update visual cues history
                visual_cues = self._get_current_visual_cues()
                if visual_cues:
                    st.session_state.visual_cues_history.append({
                        'cues': visual_cues,
                        'timestamp': time.time()
                    })
                
                # Store emotional history
                st.session_state.emotional_history.append({
                    'analysis': emotional_result,
                    'timestamp': time.time()
                })
                
                # Store proctoring alerts
                recent_alerts = self.proctoring_system.get_recent_alerts(minutes=1)
                st.session_state.proctoring_alerts = recent_alerts
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
    
    def get_comprehensive_assessment_report(self) -> Dict[str, Any]:
        """Get comprehensive assessment report with Phase 2 data"""
        try:
            # Basic assessment data
            user_context = st.session_state.user_context
            assessment_results = st.session_state.assessment_results
            
            # Phase 2: Advanced analytics
            proctoring_metrics = self.proctoring_system.get_proctoring_metrics()
            emotional_insights = self.emotional_engine.get_emotional_insights()
            behavioral_summary = self.proctoring_system.get_behavioral_summary()
            
            # Visual trends
            visual_trends = {}
            if st.session_state.visual_cues_history:
                recent_cues = list(st.session_state.visual_cues_history)[-20:]  # Last 20 entries
                
                avg_attention = np.mean([c['cues'].attention_level for c in recent_cues])
                avg_confidence = np.mean([c['cues'].confidence_level for c in recent_cues])
                avg_stress = np.mean([c['cues'].stress_indicators for c in recent_cues])
                
                visual_trends = {
                    'avg_attention': avg_attention,
                    'avg_confidence': avg_confidence,
                    'avg_stress': avg_stress,
                    'total_analyses': len(st.session_state.visual_cues_history)
                }
            
            # Emotional trends
            emotional_trends = self.emotional_engine.get_emotional_trends(minutes=10)
            
            # Proctoring alerts summary
            alerts_summary = {
                'total_alerts': len(st.session_state.proctoring_alerts),
                'critical_alerts': len([a for a in st.session_state.proctoring_alerts if a.severity.value == 'critical']),
                'high_alerts': len([a for a in st.session_state.proctoring_alerts if a.severity.value == 'high']),
                'alert_types': list(set([a.alert_type.value for a in st.session_state.proctoring_alerts]))
            }
            
            return {
                'basic_metrics': {
                    'user_id': user_context.user_id,
                    'total_questions': user_context.questions_attempted,
                    'correct_answers': user_context.correct_answers,
                    'score_percentage': user_context.current_score * 100,
                    'average_response_time': user_context.average_response_time
                },
                'proctoring_metrics': {
                    'attention_score': proctoring_metrics.attention_score,
                    'engagement_score': proctoring_metrics.engagement_score,
                    'integrity_score': proctoring_metrics.integrity_score,
                    'stress_level': proctoring_metrics.stress_level,
                    'posture_score': proctoring_metrics.posture_score,
                    'total_alerts': proctoring_metrics.alert_count,
                    'critical_alerts': proctoring_metrics.critical_alerts
                },
                'emotional_intelligence': {
                    'emotional_health_score': emotional_insights.get('emotional_health_score', 0.5),
                    'stress_level': emotional_insights.get('stress_level', 0.0),
                    'confidence_level': emotional_insights.get('confidence_level', 0.5),
                    'insights': emotional_insights.get('insights', []),
                    'recommendations': emotional_insights.get('recommendations', [])
                },
                'behavioral_analysis': {
                    'risk_assessment': behavioral_summary.get('risk_assessment', 'low'),
                    'behavioral_patterns': behavioral_summary.get('behavioral_patterns', []),
                    'session_duration': behavioral_summary.get('session_duration', 0)
                },
                'visual_trends': visual_trends,
                'emotional_trends': emotional_trends,
                'alerts_summary': alerts_summary,
                'assessment_quality': {
                    'frame_quality': np.mean([r['visual'].frame_quality for r in assessment_results if 'visual' in r]) if assessment_results else 0.0,
                    'analysis_completeness': len(st.session_state.visual_cues_history) / max(1, user_context.questions_attempted)
                }
            }
            
        except Exception as e:
            logger.error(f"Comprehensive report generation error: {e}")
            return {}
    
    def cleanup_assessment(self):
        """Clean up assessment resources"""
        try:
            # Cleanup Phase 2 components
            self.vision_engine.cleanup()
            self.proctoring_system.cleanup()
            self.emotional_engine.cleanup()
            
            # Clear session state
            st.session_state.proctoring_alerts = []
            st.session_state.emotional_history = []
            st.session_state.last_frame_analysis = None
            
            logger.info("Assessment cleanup completed")
            
        except Exception as e:
            logger.error(f"Assessment cleanup error: {e}")

# Streamlit page function
def show_integrated_assessment_page(llm_providers: Dict[str, Any]):
    """Simplified integrated assessment page"""
    
    # Initialize the system
    if 'integrated_system' not in st.session_state:
        st.session_state.integrated_system = IntegratedAssessmentSystem(llm_providers)
    
    system = st.session_state.integrated_system
    
    # Simple title
    st.title("🤖 AI-Powered Assessment")
    st.write("Adaptive questions with real-time monitoring")
    
    # Check if assessment should start
    if not st.session_state.assessment_started:
        # Show start screen
        user_data = st.session_state.get('registration_data', {})
        
        if user_data:
            st.write(f"Welcome, {user_data.get('name', 'User')}!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Start Assessment", type="primary"):
                    system.start_assessment(user_data)
                    st.rerun()
            
            with col2:
                if st.button("⚙️ Settings"):
                    st.info("Assessment Settings:\n- 20 Questions\n- Adaptive Difficulty\n- Camera Monitoring")
        else:
            st.error("Please complete registration first.")
            if st.button("Go to Registration"):
                st.session_state.current_page = "registration"
                st.rerun()
    
    else:
        # Show active assessment - simplified
        system.show_simplified_assessment()

# Test function
if __name__ == "__main__":
    st.set_page_config(page_title="Integrated Assessment Test", layout="wide")
    
    # Mock LLM providers for testing
    mock_providers = {
        "groq": None,  # Would be actual client
        "gemini": None
    }
    
    show_integrated_assessment_page(mock_providers)
