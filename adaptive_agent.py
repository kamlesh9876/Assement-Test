"""
Adaptive Question Generation Agent
Generates dynamic questions based on user performance and visual cues
"""

import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time
import google.generativeai as genai
from groq import Groq
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    CODING = "coding"
    SCENARIO = "scenario"

@dataclass
class UserContext:
    """User performance and context data"""
    user_id: str
    current_score: float
    questions_attempted: int
    correct_answers: int
    average_response_time: float
    current_streak: int
    weak_topics: List[str]
    strong_topics: List[str]
    confidence_level: float
    engagement_level: float
    stress_indicators: float

@dataclass
class VisualCues:
    """Visual analysis data from camera"""
    eye_contact: float  # 0-1 scale
    attention_level: float  # 0-1 scale
    stress_indicators: float  # 0-1 scale
    confidence_level: float  # 0-1 scale
    distraction_count: int
    posture_score: float  # 0-1 scale

@dataclass
class Question:
    """Question structure"""
    id: str
    type: QuestionType
    topic: str
    difficulty: DifficultyLevel
    question_text: str
    options: Optional[List[str]]  # For MCQ
    correct_answer: str
    explanation: str
    time_limit: int  # seconds
    points: int

class AdaptiveQuestionAgent:
    """Adaptive question generation agent"""
    
    def __init__(self, llm_providers: Dict[str, Any]):
        self.llm_providers = llm_providers
        self.question_templates = self._load_question_templates()
        self.topic_difficulty_map = self._load_topic_difficulty()
        self.current_provider_index = 0
        
    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load question templates for different types"""
        return {
            "programming": [
                "What is the time complexity of {concept}?",
                "Which data structure is best suited for {scenario}?",
                "Write a function that {task}.",
                "Debug the following code: {code_snippet}",
                "What will be the output of {code}?"
            ],
            "algorithms": [
                "Explain the {algorithm} algorithm.",
                "When would you use {algorithm} over {alternative}?",
                "What is the space complexity of {algorithm}?",
                "Implement {algorithm} in Python.",
                "Compare {algorithm1} and {algorithm2}."
            ],
            "databases": [
                "Write a SQL query to {task}.",
                "What is the difference between {concept1} and {concept2}?",
                "Normalize the following database schema.",
                "What index would you use for {scenario}?",
                "Explain {database_concept}."
            ],
            "general": [
                "What is {concept}?",
                "Explain {topic} in simple terms.",
                "Compare {concept1} and {concept2}.",
                "When would you use {concept}?",
                "What are the advantages of {concept}?"
            ]
        }
    
    def _load_topic_difficulty(self) -> Dict[str, Dict[str, float]]:
        """Load topic difficulty progression"""
        return {
            "basics": {"easy": 0.7, "medium": 0.25, "hard": 0.05},
            "data_structures": {"easy": 0.4, "medium": 0.4, "hard": 0.2},
            "algorithms": {"easy": 0.3, "medium": 0.4, "hard": 0.3},
            "databases": {"easy": 0.5, "medium": 0.35, "hard": 0.15},
            "oop": {"easy": 0.4, "medium": 0.4, "hard": 0.2},
            "system_design": {"easy": 0.2, "medium": 0.5, "hard": 0.3}
        }
    
    def generate_adaptive_question(
        self, 
        user_context: UserContext, 
        visual_cues: Optional[VisualCues] = None,
        preferred_topics: Optional[List[str]] = None
    ) -> Question:
        """Generate an adaptive question based on user context and visual cues"""
        
        # Determine difficulty based on performance and visual cues
        difficulty = self._determine_difficulty(user_context, visual_cues)
        
        # Select topic based on user performance and preferences
        topic = self._select_topic(user_context, preferred_topics)
        
        # Choose question type
        question_type = self._select_question_type(user_context, visual_cues)
        
        # Generate question using LLM
        question = self._generate_question_with_llm(
            topic=topic,
            difficulty=difficulty,
            question_type=question_type,
            user_context=user_context,
            visual_cues=visual_cues
        )
        
        return question
    
    def _determine_difficulty(
        self, 
        user_context: UserContext, 
        visual_cues: Optional[VisualCues]
    ) -> DifficultyLevel:
        """Determine question difficulty based on performance and visual cues"""
        
        # Base difficulty on performance
        if user_context.current_score > 0.8:
            base_difficulty = DifficultyLevel.HARD
        elif user_context.current_score > 0.6:
            base_difficulty = DifficultyLevel.MEDIUM
        else:
            base_difficulty = DifficultyLevel.EASY
        
        # Adjust based on visual cues
        if visual_cues:
            if visual_cues.stress_indicators > 0.7:
                # User is stressed, reduce difficulty
                if base_difficulty == DifficultyLevel.HARD:
                    base_difficulty = DifficultyLevel.MEDIUM
                elif base_difficulty == DifficultyLevel.MEDIUM:
                    base_difficulty = DifficultyLevel.EASY
            
            elif visual_cues.confidence_level > 0.8 and visual_cues.attention_level > 0.8:
                # User is confident and attentive, can handle more challenge
                if base_difficulty == DifficultyLevel.EASY:
                    base_difficulty = DifficultyLevel.MEDIUM
                elif base_difficulty == DifficultyLevel.MEDIUM:
                    base_difficulty = DifficultyLevel.HARD
        
        return base_difficulty
    
    def _select_topic(
        self, 
        user_context: UserContext, 
        preferred_topics: Optional[List[str]]
    ) -> str:
        """Select topic based on user performance and preferences"""
        
        available_topics = ["basics", "data_structures", "algorithms", "databases", "oop", "system_design"]
        
        # Prioritize weak topics for improvement
        if user_context.weak_topics:
            weak_topic = random.choice(user_context.weak_topics)
            if weak_topic in available_topics:
                return weak_topic
        
        # Use preferred topics if provided
        if preferred_topics:
            for topic in preferred_topics:
                if topic in available_topics:
                    return topic
        
        # Random selection with bias towards medium difficulty topics
        medium_topics = [t for t in available_topics if self.topic_difficulty_map.get(t, {}).get("medium", 0) > 0.3]
        if medium_topics:
            return random.choice(medium_topics)
        
        return random.choice(available_topics)
    
    def _select_question_type(
        self, 
        user_context: UserContext, 
        visual_cues: Optional[VisualCues]
    ) -> QuestionType:
        """Select question type based on user context and visual cues"""
        
        # Base type selection on performance
        if user_context.average_response_time < 30:
            # Fast responder, can handle complex questions
            base_types = [QuestionType.MULTIPLE_CHOICE, QuestionType.SCENARIO, QuestionType.CODING]
        else:
            # Slower responder, stick to simpler types
            base_types = [QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE, QuestionType.SHORT_ANSWER]
        
        # Adjust based on visual cues
        if visual_cues:
            if visual_cues.attention_level < 0.5:
                # Low attention, use engaging types
                base_types = [QuestionType.MULTIPLE_CHOICE, QuestionType.SCENARIO]
            elif visual_cues.confidence_level > 0.8:
                # High confidence, challenge with coding
                base_types.append(QuestionType.CODING)
        
        return random.choice(base_types)
    
    def _generate_question_with_llm(
        self,
        topic: str,
        difficulty: DifficultyLevel,
        question_type: QuestionType,
        user_context: UserContext,
        visual_cues: Optional[VisualCues]
    ) -> Question:
        """Generate question using LLM with fallback"""
        
        prompt = self._create_generation_prompt(
            topic=topic,
            difficulty=difficulty,
            question_type=question_type,
            user_context=user_context,
            visual_cues=visual_cues
        )
        
        # Try different LLM providers
        for provider_name in ["groq", "openrouter", "perplexity", "gemini"]:
            if provider_name in self.llm_providers:
                try:
                    question_data = self._call_llm_provider(provider_name, prompt)
                    return self._parse_question_response(question_data, topic, difficulty, question_type)
                except Exception as e:
                    logger.warning(f"Failed to generate question with {provider_name}: {e}")
                    continue
        
        # Fallback to template-based generation
        return self._generate_fallback_question(topic, difficulty, question_type)
    
    def _create_generation_prompt(
        self,
        topic: str,
        difficulty: DifficultyLevel,
        question_type: QuestionType,
        user_context: UserContext,
        visual_cues: Optional[VisualCues]
    ) -> str:
        """Create prompt for LLM question generation"""
        
        prompt = f"""
Generate a {difficulty.value} {question_type.value} question about {topic}.

User Context:
- Current Score: {user_context.current_score:.2f}
- Questions Attempted: {user_context.questions_attempted}
- Average Response Time: {user_context.average_response_time:.1f}s
- Confidence Level: {user_context.confidence_level:.2f}
"""
        
        if visual_cues:
            prompt += f"""
Visual Cues:
- Attention Level: {visual_cues.attention_level:.2f}
- Stress Indicators: {visual_cues.stress_indicators:.2f}
- Confidence Level: {visual_cues.confidence_level:.2f}
"""
        
        prompt += f"""
Requirements:
1. Generate a challenging but fair question
2. Provide 4 options for multiple choice questions
3. Include clear explanation
4. Set appropriate time limit (15-60 seconds based on difficulty)
5. Assign points based on difficulty (easy: 5, medium: 10, hard: 15)

Return JSON format:
{{
    "question_text": "...",
    "options": ["...", "...", "...", "..."],
    "correct_answer": "...",
    "explanation": "...",
    "time_limit": 30,
    "points": 10
}}
"""
        
        return prompt
    
    def _call_llm_provider(self, provider_name: str, prompt: str) -> Dict[str, Any]:
        """Call specific LLM provider"""
        
        if provider_name == "groq":
            client = self.llm_providers[provider_name]
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return json.loads(response.choices[0].message.content)
        
        elif provider_name == "openrouter":
            client = self.llm_providers[provider_name]
            response = client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return json.loads(response.choices[0].message.content)
        
        elif provider_name == "perplexity":
            client = self.llm_providers[provider_name]
            response = client.chat.completions.create(
                model="llama-3.1-sonar-small-128k-online",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return json.loads(response.choices[0].message.content)
        
        elif provider_name == "gemini":
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return json.loads(response.text)
        
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    def _parse_question_response(
        self, 
        question_data: Dict[str, Any], 
        topic: str, 
        difficulty: DifficultyLevel, 
        question_type: QuestionType
    ) -> Question:
        """Parse LLM response into Question object"""
        
        return Question(
            id=f"q_{int(time.time())}_{random.randint(1000, 9999)}",
            type=question_type,
            topic=topic,
            difficulty=difficulty,
            question_text=question_data.get("question_text", ""),
            options=question_data.get("options"),
            correct_answer=question_data.get("correct_answer", ""),
            explanation=question_data.get("explanation", ""),
            time_limit=question_data.get("time_limit", 30),
            points=question_data.get("points", 10)
        )
    
    def _generate_fallback_question(
        self, 
        topic: str, 
        difficulty: DifficultyLevel, 
        question_type: QuestionType
    ) -> Question:
        """Generate fallback question from templates"""
        
        # Simple fallback logic
        fallback_questions = {
            "basics": {
                "easy": {
                    "question": "What is a variable in programming?",
                    "options": ["Storage location", "Function", "Class", "Loop"],
                    "answer": "Storage location",
                    "explanation": "A variable is a named storage location in memory."
                },
                "medium": {
                    "question": "What is the difference between let and const?",
                    "options": ["Scope", "Type", "Value", "Syntax"],
                    "answer": "Value",
                    "explanation": "let allows reassignment, const does not."
                }
            }
        }
        
        # Get fallback data or use default
        fallback_data = fallback_questions.get(topic, {}).get(difficulty.value, fallback_questions["basics"]["easy"])
        
        return Question(
            id=f"fallback_{int(time.time())}",
            type=question_type,
            topic=topic,
            difficulty=difficulty,
            question_text=fallback_data["question"],
            options=fallback_data["options"],
            correct_answer=fallback_data["answer"],
            explanation=fallback_data["explanation"],
            time_limit=30,
            points=10 if difficulty == DifficultyLevel.MEDIUM else 5
        )
    
    def update_user_context(self, user_context: UserContext, question_result: Dict[str, Any]) -> UserContext:
        """Update user context based on question result"""
        
        # Update basic metrics
        user_context.questions_attempted += 1
        if question_result.get("correct", False):
            user_context.correct_answers += 1
            user_context.current_streak += 1
        else:
            user_context.current_streak = 0
        
        # Update score
        user_context.current_score = user_context.correct_answers / user_context.questions_attempted
        
        # Update response time
        response_time = question_result.get("response_time", 30)
        if user_context.average_response_time == 0:
            user_context.average_response_time = response_time
        else:
            user_context.average_response_time = (
                user_context.average_response_time * 0.8 + response_time * 0.2
            )
        
        # Update confidence based on performance
        if question_result.get("correct", False) and response_time < user_context.average_response_time:
            user_context.confidence_level = min(1.0, user_context.confidence_level + 0.1)
        elif not question_result.get("correct", False):
            user_context.confidence_level = max(0.0, user_context.confidence_level - 0.05)
        
        return user_context

# Factory function to create agent
def create_adaptive_question_agent(llm_providers: Dict[str, Any]) -> AdaptiveQuestionAgent:
    """Create and return adaptive question agent instance"""
    return AdaptiveQuestionAgent(llm_providers)

# Test function
if __name__ == "__main__":
    # Test the agent
    mock_providers = {
        "groq": Groq(api_key="test"),
        "gemini": None
    }
    
    agent = create_adaptive_question_agent(mock_providers)
    
    # Test user context
    user_context = UserContext(
        user_id="test_user",
        current_score=0.7,
        questions_attempted=10,
        correct_answers=7,
        average_response_time=25.0,
        current_streak=3,
        weak_topics=["algorithms"],
        strong_topics=["basics"],
        confidence_level=0.8,
        engagement_level=0.7,
        stress_indicators=0.3
    )
    
    # Test visual cues
    visual_cues = VisualCues(
        eye_contact=0.8,
        attention_level=0.7,
        stress_indicators=0.3,
        confidence_level=0.8,
        distraction_count=2,
        posture_score=0.9
    )
    
    # Generate question
    question = agent.generate_adaptive_question(user_context, visual_cues)
    print(f"Generated Question: {question.question_text}")
    print(f"Difficulty: {question.difficulty.value}")
    print(f"Topic: {question.topic}")
