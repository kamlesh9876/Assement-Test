"""
Advanced Conversational AI Assessment for Phase 3
Interactive assessment with natural language conversation and adaptive dialogue
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import re
import uuid

from adaptive_agent import Question, DifficultyLevel, QuestionType, create_adaptive_question_agent
from computer_vision import VisualAnalysisResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DialogueState(Enum):
    GREETING = "greeting"
    QUESTION_PRESENTATION = "question_presentation"
    USER_THINKING = "user_thinking"
    ANSWER_COLLECTION = "answer_collection"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    ENCOURAGEMENT = "encouragement"
    FEEDBACK = "feedback"
    TRANSITION = "transition"

class InteractionMode(Enum):
    VOICE = "voice"
    TEXT = "text"
    GESTURE = "gesture"
    MULTIMODAL = "multimodal"

class ResponseType(Enum):
    QUESTION = "question"
    CLARIFICATION = "clarification"
    ENCOURAGEMENT = "encouragement"
    FEEDBACK = "feedback"
    FOLLOW_UP = "follow_up"
    TRANSITION = "transition"

@dataclass
class DialogueContext:
    """Context for conversational AI assessment"""
    session_id: str
    user_id: str
    current_topic: str
    difficulty_level: DifficultyLevel
    question_count: int
    correct_answers: int
    user_confidence: float
    engagement_level: float
    interaction_history: List[Dict[str, Any]]
    current_state: DialogueState
    last_interaction_time: float
    multimodal_inputs: Dict[str, Any]

@dataclass
class ConversationalResponse:
    """AI-generated conversational response"""
    response_id: str
    response_type: ResponseType
    content: str
    confidence: float
    follow_up_questions: List[str]
    emotional_tone: str
    interaction_suggestions: List[str]
    multimodal_cues: Dict[str, Any]

@dataclass
class UserInput:
    """User input from multiple modalities"""
    input_id: str
    timestamp: float
    modality: InteractionMode
    content: str
    confidence: float
    emotional_state: Optional[str]
    visual_cues: Optional[Dict[str, Any]]
    gesture_data: Optional[Dict[str, Any]]
    processing_time: float

class ConversationalAIAssessment:
    """Advanced conversational AI assessment system"""
    
    def __init__(self, llm_providers: Dict[str, Any]):
        self.llm_providers = llm_providers
        self.adaptive_agent = create_adaptive_question_agent(llm_providers)
        
        # Dialogue management
        self.dialogue_history = deque(maxlen=100)
        self.response_templates = self._initialize_response_templates()
        self.state_transitions = self._initialize_state_transitions()
        
        # Conversation state
        self.current_context = None
        self.active_session = None
        
        # Multimodal integration
        self.multimodal_buffer = deque(maxlen=50)
        self.input_queue = deque(maxlen=20)
        
        # Performance tracking
        self.response_times = deque(maxlen=50)
        self.interaction_counts = {}
        
        logger.info("Conversational AI Assessment initialized")
    
    def _initialize_response_templates(self) -> Dict[DialogueState, List[str]]:
        """Initialize response templates for different dialogue states"""
        return {
            DialogueState.GREETING: [
                "Hello! I'm your AI assessment assistant. Let's begin with a few questions to understand your knowledge level.",
                "Welcome! I'll be conducting an adaptive assessment. Let's start with a warm-up question.",
                "Hi there! Ready to begin your personalized assessment? I'll adapt the questions based on your responses."
            ],
            DialogueState.QUESTION_PRESENTATION: [
                "Here's your next question: {question}",
                "Let's move on to: {question}",
                "Now for this question: {question}",
                "Time for the next question: {question}"
            ],
            DialogueState.USER_THINKING: [
                "Take your time to think through this question.",
                "No rush - carefully consider your answer.",
                "Feel free to think about this one.",
                "Take a moment to formulate your response."
            ],
            DialogueState.ANSWER_COLLECTION: [
                "I'm listening to your answer.",
                "Go ahead and share your thoughts.",
                "I'm ready to hear your response.",
                "Please provide your answer when ready."
            ],
            DialogueState.FOLLOW_UP: [
                "That's interesting! Let me ask a follow-up: {question}",
                "Based on your answer, let's explore: {question}",
                "Great point! Now consider this: {question}"
            ],
            DialogueState.CLARIFICATION: [
                "Could you clarify what you mean by that?",
                "I'd like to understand your answer better. Could you elaborate?",
                "Can you provide more detail about your response?",
                "Let me make sure I understand correctly..."
            ],
            DialogueState.ENCOURAGEMENT: [
                "You're doing great! Keep going.",
                "Excellent thinking! You're on the right track.",
                "Good effort! Let's continue.",
                "Nice work! You're making progress."
            ],
            DialogueState.FEEDBACK: [
                "{feedback} Your answer was {correctness}.",
                "{feedback} The correct approach would be {explanation}.",
                "{feedback} {explanation}",
                "{feedback} Let's move to the next topic."
            ],
            DialogueState.TRANSITION: [
                "Let's move on to the next topic.",
                "Time to switch gears to a new area.",
                "Now let's explore a different concept.",
                "Let's continue with the next section."
            ]
        }
    
    def _initialize_state_transitions(self) -> Dict[DialogueState, List[DialogueState]]:
        """Initialize valid state transitions"""
        return {
            DialogueState.GREETING: [DialogueState.QUESTION_PRESENTATION],
            DialogueState.QUESTION_PRESENTATION: [DialogueState.USER_THINKING, DialogueState.ANSWER_COLLECTION],
            DialogueState.USER_THINKING: [DialogueState.ANSWER_COLLECTION, DialogueState.ENCOURAGEMENT],
            DialogueState.ANSWER_COLLECTION: [DialogueState.FOLLOW_UP, DialogueState.FEEDBACK, DialogueState.CLARIFICATION],
            DialogueState.FOLLOW_UP: [DialogueState.USER_THINKING, DialogueState.ANSWER_COLLECTION],
            DialogueState.CLARIFICATION: [DialogueState.ANSWER_COLLECTION],
            DialogueState.ENCOURAGEMENT: [DialogueState.USER_THINKING, DialogueState.ANSWER_COLLECTION],
            DialogueState.FEEDBACK: [DialogueState.TRANSITION, DialogueState.QUESTION_PRESENTATION],
            DialogueState.TRANSITION: [DialogueState.QUESTION_PRESENTATION]
        }
    
    def start_conversational_assessment(self, user_data: Dict[str, Any]) -> str:
        """Start a new conversational assessment session"""
        try:
            session_id = str(uuid.uuid4())
            
            # Create dialogue context
            context = DialogueContext(
                session_id=session_id,
                user_id=user_data.get('user_id', 'anonymous'),
                current_topic="general",
                difficulty_level=DifficultyLevel.EASY,
                question_count=0,
                correct_answers=0,
                user_confidence=0.5,
                engagement_level=0.7,
                interaction_history=[],
                current_state=DialogueState.GREETING,
                last_interaction_time=time.time(),
                multimodal_inputs={}
            )
            
            self.current_context = context
            self.active_session = session_id
            
            # Generate greeting response
            response = self._generate_response(context)
            
            # Store in history
            self.dialogue_history.append({
                'session_id': session_id,
                'timestamp': time.time(),
                'state': context.current_state.value,
                'response': asdict(response),
                'context': asdict(context)
            })
            
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to start conversational assessment: {e}")
            return "I'm having trouble starting the assessment. Please try again."
    
    def process_user_input(self, user_input: UserInput) -> ConversationalResponse:
        """Process user input and generate appropriate response"""
        try:
            if not self.current_context:
                return self._create_error_response("No active session")
            
            # Update context
            self.current_context.last_interaction_time = user_input.timestamp
            self.current_context.interaction_history.append({
                'timestamp': user_input.timestamp,
                'modality': user_input.modality.value,
                'content': user_input.content,
                'confidence': user_input.confidence,
                'processing_time': user_input.processing_time
            })
            
            # Store multimodal inputs
            if user_input.emotional_state:
                self.current_context.multimodal_inputs['emotional_state'] = user_input.emotional_state.value
            
            if user_input.visual_cues:
                self.current_context.multimodal_inputs['visual_cues'] = user_input.visual_cues
            
            if user_input.gesture_data:
                self.current_context.multimodal_inputs['gesture_data'] = user_input.gesture_data
            
            # Determine next state based on input
            next_state = self._determine_next_state(user_input)
            
            # Generate response
            response = self._generate_contextual_response(user_input, next_state)
            
            # Update state
            self.current_context.current_state = next_state
            
            # Store interaction
            self.dialogue_history.append({
                'session_id': self.active_session,
                'timestamp': time.time(),
                'state': next_state.value,
                'user_input': asdict(user_input),
                'response': asdict(response),
                'context': asdict(self.current_context)
            })
            
            # Track response time
            response_time = time.time() - user_input.timestamp
            self.response_times.append(response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process user input: {e}")
            return self._create_error_response("I had trouble processing your response. Please try again.")
    
    def _determine_next_state(self, user_input: UserInput) -> DialogueState:
        """Determine next dialogue state based on user input"""
        try:
            current_state = self.current_context.current_state
            
            # Analyze user input content
            content = user_input.content.lower().strip()
            
            # Check for clarification needs
            if current_state == DialogueState.ANSWER_COLLECTION:
                if len(content) < 10 or "i don't know" in content or "not sure" in content:
                    return DialogueState.CLARIFICATION
                
                # Check if answer seems complete
                if content.endswith('?') or "what about" in content or "can you" in content:
                    return DialogueState.CLARIFICATION
            
            # Check for confusion or uncertainty
            if "confused" in content or "unclear" in content or "help" in content:
                return DialogueState.CLARIFICATION
            
            # Check for readiness to continue
            if "ready" in content or "next" in content or "continue" in content:
                if current_state in [DialogueState.FEEDBACK, DialogueState.ENCOURAGEMENT]:
                    return DialogueState.TRANSITION
            
            # State-based transitions
            valid_transitions = self.state_transitions.get(current_state, [])
            
            # Default transition based on current state
            if current_state == DialogueState.GREETING:
                return DialogueState.QUESTION_PRESENTATION
            elif current_state == DialogueState.QUESTION_PRESENTATION:
                return DialogueState.USER_THINKING
            elif current_state == DialogueState.USER_THINKING:
                return DialogueState.ANSWER_COLLECTION
            elif current_state == DialogueState.ANSWER_COLLECTION:
                # Evaluate answer quality to determine next state
                if self._evaluate_answer_quality(user_input) > 0.7:
                    return DialogueState.FEEDBACK
                else:
                    return DialogueState.CLARIFICATION
            elif current_state == DialogueState.FEEDBACK:
                return DialogueState.ENCOURAGEMENT
            elif current_state == DialogueState.ENCOURAGEMENT:
                return DialogueState.QUESTION_PRESENTATION
            else:
                return DialogueState.QUESTION_PRESENTATION
                
        except Exception as e:
            logger.error(f"State determination error: {e}")
            return DialogueState.QUESTION_PRESENTATION
    
    def _evaluate_answer_quality(self, user_input: UserInput) -> float:
        """Evaluate the quality of user's answer"""
        try:
            content = user_input.content.strip()
            
            # Basic quality indicators
            quality_score = 0.0
            
            # Length indicator (not too short, not too long)
            if 10 <= len(content) <= 200:
                quality_score += 0.3
            elif len(content) > 200:
                quality_score += 0.1  # Penalize very long answers
            
            # Content indicators
            if any(keyword in content.lower() for keyword in ["because", "since", "due to", "as a result"]):
                quality_score += 0.2  # Shows reasoning
            
            if any(keyword in content.lower() for keyword in ["first", "second", "finally", "in conclusion"]):
                quality_score += 0.1  # Structured answer
            
            # Confidence indicators
            if user_input.confidence > 0.7:
                quality_score += 0.2
            elif user_input.confidence < 0.3:
                quality_score -= 0.1
            
            # Emotional state consideration
            if user_input.emotional_state:
                if user_input.emotional_state.value in ["confidence", "happiness"]:
                    quality_score += 0.1
                elif user_input.emotional_state.value in ["confusion", "frustration"]:
                    quality_score -= 0.1
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"Answer quality evaluation error: {e}")
            return 0.5
    
    def _generate_response(self, context: DialogueContext) -> ConversationalResponse:
        """Generate response based on current context and state"""
        try:
            state = context.current_state
            templates = self.response_templates.get(state, [])
            
            if not templates:
                template = "Let's continue with the assessment."
            else:
                template = templates[0]  # Use first template
            
            # Generate response content
            if state == DialogueState.QUESTION_PRESENTATION:
                question = self._generate_adaptive_question(context)
                content = template.format(question=question.question_text)
            else:
                content = template
            
            return ConversationalResponse(
                response_id=str(uuid.uuid4()),
                response_type=self._map_state_to_response_type(state),
                content=content,
                confidence=0.8,
                follow_up_questions=[],
                emotional_tone="neutral",
                interaction_suggestions=[],
                multimodal_cues={}
            )
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._create_error_response("I'm having trouble generating a response.")
    
    def _generate_contextual_response(self, user_input: UserInput, state: DialogueState) -> ConversationalResponse:
        """Generate contextual response based on user input and state"""
        try:
            templates = self.response_templates.get(state, [])
            
            if not templates:
                template = "Let's continue."
            else:
                template = templates[0]
            
            content = template
            
            # Add contextual elements
            if state == DialogueState.QUESTION_PRESENTATION:
                question = self._generate_adaptive_question(self.current_context)
                content = template.format(question=question.question_text)
            
            elif state == DialogueState.FEEDBACK:
                feedback = self._generate_feedback(user_input)
                content = template.format(feedback=feedback['text'], correctness=feedback['correctness'], explanation=feedback['explanation'])
            
            elif state == DialogueState.FOLLOW_UP:
                follow_up = self._generate_follow_up_question(user_input)
                content = template.format(question=follow_up)
            
            elif state == DialogueState.CLARIFICATION:
                clarification = self._generate_clarification(user_input)
                content = clarification
            
            # Add emotional tone based on user state
            emotional_tone = self._determine_emotional_tone(user_input, state)
            
            # Generate interaction suggestions
            suggestions = self._generate_interaction_suggestions(user_input, state)
            
            return ConversationalResponse(
                response_id=str(uuid.uuid4()),
                response_type=self._map_state_to_response_type(state),
                content=content,
                confidence=0.8,
                follow_up_questions=[],
                emotional_tone=emotional_tone,
                interaction_suggestions=suggestions,
                multimodal_cues=self._generate_multimodal_cues(user_input)
            )
            
        except Exception as e:
            logger.error(f"Contextual response generation error: {e}")
            return self._create_error_response("I'm having trouble generating a response.")
    
    def _generate_adaptive_question(self, context: DialogueContext) -> Question:
        """Generate adaptive question based on context"""
        try:
            # Use the adaptive agent to generate question
            question = self.adaptive_agent.generate_adaptive_question(
                user_context=self._convert_to_user_context(context),
                visual_cues=None  # Could integrate with visual analysis
            )
            
            # Update context
            context.question_count += 1
            
            return question
            
        except Exception as e:
            logger.error(f"Adaptive question generation error: {e}")
            # Fallback question
            return Question(
                id=f"fallback_{int(time.time())}",
                type=QuestionType.MULTIPLE_CHOICE,
                topic="general",
                difficulty=DifficultyLevel.EASY,
                question_text="What is the capital of France?",
                options=["London", "Berlin", "Paris", "Madrid"],
                correct_answer="Paris",
                explanation="Paris is the capital of France.",
                time_limit=30,
                points=5
            )
    
    def _convert_to_user_context(self, context: DialogueContext):
        """Convert DialogueContext to UserContext"""
        from adaptive_agent import UserContext, VisualCues
        
        return UserContext(
            user_id=context.user_id,
            current_score=context.correct_answers / max(1, context.question_count),
            questions_attempted=context.question_count,
            correct_answers=context.correct_answers,
            average_response_time=30.0,  # Would calculate from actual data
            current_streak=0,  # Would calculate from history
            weak_topics=[],
            strong_topics=[],
            confidence_level=context.user_confidence,
            engagement_level=context.engagement_level,
            stress_indicators=0.3  # Would calculate from multimodal data
        )
    
    def _generate_feedback(self, user_input: UserInput) -> Dict[str, str]:
        """Generate feedback for user's answer"""
        try:
            content = user_input.content.strip()
            
            # Simple feedback logic (would be more sophisticated with actual question context)
            if len(content) > 20 and not any(negative in content.lower() for negative in ["don't", "can't", "won't", "not"]):
                return {
                    'text': "That's a thoughtful response!",
                    'correctness': "correct",
                    'explanation': "Your reasoning shows good understanding of the concept."
                }
            else:
                return {
                    'text': "Let me help you think through this better.",
                    'correctness': "partially correct",
                    'explanation': "Consider the key concepts and how they relate to each other."
                }
                
        except Exception as e:
            logger.error(f"Feedback generation error: {e}")
            return {
                'text': "Thank you for your response.",
                'correctness': "received",
                'explanation': "Let's continue with the next question."
            }
    
    def _generate_follow_up_question(self, user_input: UserInput) -> str:
        """Generate follow-up question based on user input"""
        try:
            content = user_input.content.lower()
            
            # Simple follow-up logic
            if "because" in content:
                return "Can you elaborate on the reasoning behind your answer?"
            elif "example" in content:
                return "Can you provide another example to support your point?"
            elif "first" in content or "second" in content:
                return "What about the next point in your explanation?"
            else:
                return "Could you provide more detail about your answer?"
                
        except Exception as e:
            logger.error(f"Follow-up question generation error: {e}")
            return "Could you explain your answer further?"
    
    def _generate_clarification(self, user_input: UserInput) -> str:
        """Generate clarification request"""
        try:
            content = user_input.content.strip()
            
            if len(content) < 10:
                return "Could you provide more detail in your answer?"
            elif "i don't know" in content.lower():
                return "That's okay! Let me rephrase the question to help you think about it differently."
            elif "confused" in content.lower():
                return "I understand this might be confusing. Let me break it down into simpler terms."
            else:
                return "I want to make sure I understand your answer correctly. Could you clarify what you mean?"
                
        except Exception as e:
            logger.error(f"Clarification generation error: {e}")
            return "Could you please clarify your response?"
    
    def _determine_emotional_tone(self, user_input: UserInput, state: DialogueState) -> str:
        """Determine emotional tone for response"""
        try:
            # Base tone by state
            tone_by_state = {
                DialogueState.GREETING: "welcoming",
                DialogueState.QUESTION_PRESENTATION: "neutral",
                DialogueState.USER_THINKING: "patient",
                DialogueState.ANSWER_COLLECTION: "attentive",
                DialogueState.FOLLOW_UP: "curious",
                DialogueState.CLARIFICATION: "helpful",
                DialogueState.ENCOURAGEMENT: "supportive",
                DialogueState.FEEDBACK: "constructive",
                DialogueState.TRANSITION: "neutral"
            }
            
            base_tone = tone_by_state.get(state, "neutral")
            
            # Adjust based on user emotional state
            if user_input.emotional_state:
                if user_input.emotional_state.value in ["frustration", "anger"]:
                    return "calming"
                elif user_input.emotional_state.value in ["sadness", "fear"]:
                    return "reassuring"
                elif user_input.emotional_state.value in ["confidence", "happiness"]:
                    return "enthusiastic"
            
            # Adjust based on user confidence
            if user_input.confidence < 0.3:
                return "encouraging"
            elif user_input.confidence > 0.8:
                return "challenging"
            
            return base_tone
            
        except Exception as e:
            logger.error(f"Emotional tone determination error: {e}")
            return "neutral"
    
    def _generate_interaction_suggestions(self, user_input: UserInput, state: DialogueState) -> List[str]:
        """Generate interaction suggestions for user"""
        try:
            suggestions = []
            
            if state == DialogueState.USER_THINKING:
                suggestions.append("Take your time to think through the question")
                suggestions.append("Feel free to use any resources you need")
            
            elif state == DialogueState.ANSWER_COLLECTION:
                suggestions.append("Speak clearly and at a comfortable pace")
                suggestions.append("Provide as much detail as you can")
            
            elif state == DialogueState.CLARIFICATION:
                suggestions.append("Try to be specific in your explanation")
                suggestions.append("Use examples to support your points")
            
            elif state == DialogueState.FOLLOW_UP:
                suggestions.append("Consider the implications of your answer")
                suggestions.append("Think about real-world applications")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Interaction suggestions generation error: {e}")
            return []
    
    def _generate_multimodal_cues(self, user_input: UserInput) -> Dict[str, Any]:
        """Generate multimodal interaction cues"""
        try:
            cues = {}
            
            # Voice cues
            if user_input.modality == InteractionMode.VOICE:
                cues['voice'] = {
                    'suggestion': "Speak clearly and at a moderate pace",
                    'indicators': ['volume', 'clarity', 'pace']
                }
            
            # Gesture cues
            if user_input.modality == InteractionMode.GESTURE:
                cues['gesture'] = {
                    'suggestion': "Use clear, deliberate gestures",
                    'indicators': ['hand_position', 'movement', 'clarity']
                }
            
            # Multimodal integration
            if user_input.modality == InteractionMode.MULTIMODAL:
                cues['multimodal'] = {
                    'suggestion': "Use a combination of voice, gestures, and expressions",
                    'indicators': ['coordination', 'consistency', 'engagement']
                }
            
            return cues
            
        except Exception as e:
            logger.error(f"Multimodal cues generation error: {e}")
            return {}
    
    def _map_state_to_response_type(self, state: DialogueState) -> ResponseType:
        """Map dialogue state to response type"""
        mapping = {
            DialogueState.GREETING: ResponseType.QUESTION,
            DialogueState.QUESTION_PRESENTATION: ResponseType.QUESTION,
            DialogueState.USER_THINKING: ResponseType.ENCOURAGEMENT,
            DialogueState.ANSWER_COLLECTION: ResponseType.FOLLOW_UP,
            DialogueState.FOLLOW_UP: ResponseType.FOLLOW_UP,
            DialogueState.CLARIFICATION: ResponseType.CLARIFICATION,
            DialogueState.ENCOURAGEMENT: ResponseType.ENCOURAGEMENT,
            DialogueState.FEEDBACK: ResponseType.FEEDBACK,
            DialogueState.TRANSITION: ResponseType.TRANSITION
        }
        
        return mapping.get(state, ResponseType.QUESTION)
    
    def _create_error_response(self, message: str) -> ConversationalResponse:
        """Create error response"""
        return ConversationalResponse(
            response_id=str(uuid.uuid4()),
            response_type=ResponseType.CLARIFICATION,
            content=message,
            confidence=0.3,
            follow_up_questions=[],
            emotional_tone="neutral",
            interaction_suggestions=["Please try again"],
            multimodal_cues={}
        )
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversational assessment"""
        try:
            if not self.current_context:
                return {"status": "no_active_session"}
            
            context = self.current_context
            
            # Calculate metrics
            avg_response_time = np.mean(self.response_times) if self.response_times else 0.0
            total_interactions = len(context.interaction_history)
            
            # Interaction distribution
            modality_counts = {}
            for interaction in context.interaction_history:
                modality = interaction.get('modality', 'unknown')
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
            
            # State distribution
            state_counts = {}
            for entry in self.dialogue_history:
                if entry['session_id'] == self.active_session:
                    state = entry.get('state', 'unknown')
                    state_counts[state] = state_counts.get(state, 0) + 1
            
            return {
                "session_id": context.session_id,
                "user_id": context.user_id,
                "duration": time.time() - (context.last_interaction_time - avg_response_time * total_interactions),
                "questions_asked": context.question_count,
                "correct_answers": context.correct_answers,
                "accuracy": context.correct_answers / max(1, context.question_count),
                "user_confidence": context.user_confidence,
                "engagement_level": context.engagement_level,
                "total_interactions": total_interactions,
                "average_response_time": avg_response_time,
                "modality_distribution": modality_counts,
                "state_distribution": state_counts,
                "multimodal_inputs": context.multimodal_inputs
            }
            
        except Exception as e:
            logger.error(f"Conversation summary error: {e}")
            return {"status": "error"}
    
    def end_conversational_assessment(self) -> Dict[str, Any]:
        """End the conversational assessment and return results"""
        try:
            if not self.current_context:
                return {"status": "no_active_session"}
            
            summary = self.get_conversation_summary()
            
            # Generate final assessment report
            report = {
                "session_id": self.current_context.session_id,
                "completion_time": time.time(),
                "final_metrics": summary,
                "performance_evaluation": self._evaluate_performance(),
                "recommendations": self._generate_final_recommendations(),
                "next_steps": self._suggest_next_steps()
            }
            
            # Clean up
            self.current_context = None
            self.active_session = None
            
            return report
            
        except Exception as e:
            logger.error(f"End assessment error: {e}")
            return {"status": "error"}
    
    def _evaluate_performance(self) -> Dict[str, Any]:
        """Evaluate overall performance"""
        try:
            if not self.current_context:
                return {}
            
            context = self.current_context
            
            performance_score = context.correct_answers / max(1, context.question_count)
            
            evaluation = {
                "overall_score": performance_score,
                "grade": self._calculate_grade(performance_score),
                "strengths": [],
                "areas_for_improvement": [],
                "engagement_rating": context.engagement_level,
                "confidence_rating": context.user_confidence
            }
            
            # Add strengths and improvements
            if performance_score > 0.8:
                evaluation["strengths"].append("High accuracy")
            if context.engagement_level > 0.7:
                evaluation["strengths"].append("High engagement")
            if context.user_confidence > 0.7:
                evaluation["strengths"].append("Strong confidence")
            
            if performance_score < 0.6:
                evaluation["areas_for_improvement"].append("Accuracy")
            if context.engagement_level < 0.5:
                evaluation["areas_for_improvement"].append("Engagement")
            if context.user_confidence < 0.5:
                evaluation["areas_for_improvement"].append("Confidence")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Performance evaluation error: {e}")
            return {}
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _generate_final_recommendations(self) -> List[str]:
        """Generate final recommendations"""
        try:
            recommendations = []
            
            if not self.current_context:
                return recommendations
            
            context = self.current_context
            
            # Performance-based recommendations
            if context.correct_answers / max(1, context.question_count) < 0.7:
                recommendations.append("Focus on studying the fundamental concepts")
                recommendations.append("Practice with more sample questions")
            
            if context.user_confidence < 0.6:
                recommendations.append("Build confidence through practice")
                recommendations.append("Start with easier questions and progress gradually")
            
            if context.engagement_level < 0.6:
                recommendations.append("Try to stay more engaged during assessments")
                recommendations.append("Take breaks when needed but maintain focus")
            
            # General recommendations
            recommendations.append("Continue practicing to improve your skills")
            recommendations.append("Review incorrect answers to learn from mistakes")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation error: {e}")
            return []
    
    def _suggest_next_steps(self) -> List[str]:
        """Suggest next steps for improvement"""
        try:
            steps = [
                "Review your assessment results",
                "Focus on areas where you struggled",
                "Practice similar questions",
                "Seek additional learning resources",
                "Take another assessment after practice"
            ]
            
            return steps
            
        except Exception as e:
            logger.error(f"Next steps suggestion error: {e}")
            return []
    
    def cleanup(self):
        """Clean up conversational AI resources"""
        try:
            self.dialogue_history.clear()
            self.multimodal_buffer.clear()
            self.input_queue.clear()
            self.response_times.clear()
            self.interaction_counts.clear()
            
            self.current_context = None
            self.active_session = None
            
            logger.info("Conversational AI Assessment cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Factory function
def create_conversational_ai_assessment(llm_providers: Dict[str, Any]) -> ConversationalAIAssessment:
    """Create and return conversational AI assessment instance"""
    return ConversationalAIAssessment(llm_providers)

# Test function
if __name__ == "__main__":
    # Test the conversational AI assessment
    mock_providers = {}
    
    try:
        # Import numpy for testing
        import numpy as np
        np.set_printoptions(suppress=True)
        
        conversational_ai = create_conversational_ai_assessment(mock_providers)
        
        # Start assessment
        greeting = conversational_ai.start_conversational_assessment({
            'user_id': 'test_user',
            'name': 'Test User'
        })
        
        print("Conversational AI Assessment Test:")
        print(f"Greeting: {greeting}")
        
        # Simulate user input
        user_input = UserInput(
            modality=InteractionMode.TEXT,
            content="I think the answer is Paris because it's the capital of France",
            confidence=0.8,
            emotional_state=None,
            visual_cues=None,
            gesture_data=None,
            processing_time=2.5
        )
        
        # Process input
        response = conversational_ai.process_user_input(user_input)
        
        print(f"Response: {response.content}")
        print(f"Response type: {response.response_type.value}")
        print(f"Emotional tone: {response.emotional_tone}")
        
        # Get summary
        summary = conversational_ai.get_conversation_summary()
        print(f"Summary keys: {list(summary.keys())}")
        
        # End assessment
        final_report = conversational_ai.end_conversational_assessment()
        print(f"Final report status: {final_report.get('status', 'unknown')}")
        
        # Cleanup
        conversational_ai.cleanup()
        
    except Exception as e:
        print(f"Test error: {e}")
        print("Make sure all dependencies are installed")
