import json
import logging
import time
import uuid
import hashlib
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import google.generativeai as genai

# Import WebRTC for real camera functionality
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
    import streamlit.components.v1 as components
    import cv2
    import numpy as np
    from typing import List, Tuple
    WEBRTC_AVAILABLE = True
except ImportError as e:
    WEBRTC_AVAILABLE = False
    webrtc_streamer = None
    VideoProcessorBase = None
    WebRtcMode = None
    RTCConfiguration = None
    components = None
    logging.warning(f"WebRTC not available: {e}")

# Import integrated assessment components
try:
    from integrated_assessment import show_integrated_assessment_page
    INTEGRATED_ASSESSMENT_AVAILABLE = True
except ImportError as e:
    INTEGRATED_ASSESSMENT_AVAILABLE = False
    show_integrated_assessment_page = None
    logging.warning(f"Integrated assessment not available: {e}")

# Import adaptive agent for dynamic questions
try:
    from adaptive_agent import create_adaptive_question_agent, UserContext
    ADAPTIVE_AGENT_AVAILABLE = True
except ImportError as e:
    ADAPTIVE_AGENT_AVAILABLE = False
    create_adaptive_question_agent = None
    UserContext = None
    logging.warning(f"Adaptive agent not available: {e}")

# TODO: Future integration of advanced agents (see TODO.txt)
# - advanced_proctoring.py: AI-powered monitoring
# - analytics_dashboard.py: Performance visualization  
# - computer_vision.py: Visual monitoring and analysis
# - conversational_ai.py: Natural language interaction
# - emotional_intelligence.py: Behavioral analysis

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Try to import OpenAI (for OpenRouter and Perplexity)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# ✅ PROFESSIONAL CAMERA PATTERN - Re-attach on every rerun
def render_camera():
    """Render camera that re-attaches stream on every rerun"""
    # Only show camera on test page after assessment has started
    if (st.session_state.get("test_submitted", False) or 
        st.session_state.get("page") != "test" or
        not st.session_state.get("assessment_started", False)):
        return  # Stop camera after submit or if not on test page

    if components:
        components.html(
            """
            <style>
            #cam-box {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 300px;
                height: 220px;
                z-index: 9999;
                border-radius: 12px;
                overflow: hidden;
                background: black;
                border: 2px solid #3b82f6;
                box-shadow: 0 8px 20px rgba(0,0,0,0.4);
            }
            video {
                width: 100%;
                height: 100%;
                object-fit: cover;
                transform: scaleX(-1);
            }
            </style>

            <div id="cam-box">
                <video id="cam" autoplay muted playsinline></video>
            </div>

            <script>
            (function () {
                console.log('🎥 Camera render - checking stream...');
                const video = document.getElementById("cam");

                // If stream already exists, reuse it
                if (window._cameraStream) {
                    console.log('🔄 Reusing existing camera stream');
                    video.srcObject = window._cameraStream;
                    return;
                }

                // Otherwise request camera
                console.log('🎥 Requesting new camera stream...');
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(stream => {
                        console.log('✅ Camera stream obtained');
                        window._cameraStream = stream;
                        video.srcObject = stream;
                    })
                    .catch(err => console.error("❌ Camera error:", err));
            })();
            </script>
            """,
            height=240
        )

# Initialize multiple AI providers
def _initialize_providers():
    """Initialize all available AI providers"""
    providers = {}
    
    # Groq
    if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
        try:
            providers['groq'] = Groq(api_key=os.getenv("GROQ_API_KEY"))
            logging.info("Groq client initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Groq: {e}")
    
    # OpenRouter
    if OPENAI_AVAILABLE and os.getenv("OPENROUTER_API_KEY"):
        try:
            providers['openrouter'] = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
            logging.info("OpenRouter client initialized")
        except Exception as e:
            logging.error(f"Failed to initialize OpenRouter: {e}")
    
    # Perplexity
    if OPENAI_AVAILABLE and os.getenv("PERPLEXITY_API_KEY"):
        try:
            providers['perplexity'] = OpenAI(
                api_key=os.getenv("PERPLEXITY_API_KEY"),
                base_url="https://api.perplexity.ai"
            )
            logging.info("Perplexity client initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Perplexity: {e}")
    
    return providers

# Initialize providers globally
AI_PROVIDERS = _initialize_providers()

# Set page config as the first Streamlit command
st.set_page_config(page_title="Online Assessment (MVP)", layout="wide", initial_sidebar_state="collapsed")

# Initialize API key rotation
if 'current_key_index' not in st.session_state:
    st.session_state['current_key_index'] = 0

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ No GEMINI_API_KEY found in environment variables!")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)


@dataclass
class CandidateProfile:
    name: str
    branch: str
    passing_year: int
    university: str
    programming_language: str
    difficulty: str
    assessment_type: str


@dataclass
class TestConfig:
    total_questions: int
    duration_seconds: int


def _init_state() -> None:
    defaults = {
        "page": "registration",
        "profile": None,
        "accepted_rules": False,
        "rules_confirmed": False,
        "system_check_passed": False,
        "attempt_id": None,
        "test_config": None,
        "questions": None,
        "current_q": 0,
        "answers": {},
        "test_started_at": None,
        "test_submitted_at": None,
        "result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _set_page(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def _get_default_questions(programming_language: str, difficulty: str):
    """Generate diverse default questions to avoid repetition"""
    
    # Create question pools by topic to ensure variety
    python_questions = [
        {
            "id": "q1",
            "topic": "basics",
            "difficulty": "easy",
            "question": "What is the purpose of the 'self' parameter in Python methods?",
            "options": ["Refers to the instance calling the method", "Creates a static method", "Returns the method name as string", "Defines method visibility"],
            "answer_index": 0,
        },
        {
            "id": "q2",
            "topic": "data_structures",
            "difficulty": "easy",
            "question": "Which Python data structure is immutable?",
            "options": ["List", "Dictionary", "Tuple", "Set"],
            "answer_index": 2,
        },
        {
            "id": "q3",
            "topic": "control_flow",
            "difficulty": "easy",
            "question": "What does the 'pass' statement do in Python?",
            "options": ["Skip the current iteration", "Terminate the loop", "Return a value from function", "Define a function"],
            "answer_index": 0,
        },
        {
            "id": "q4",
            "topic": "functions",
            "difficulty": "medium",
            "question": "How do you define a default parameter value in Python functions?",
            "options": ["param = value", "param: type = value", "param = value:", "param(default=value)"],
            "answer_index": 2,
        },
        {
            "id": "q5",
            "topic": "oop",
            "difficulty": "medium",
            "question": "What is the purpose of the '__init__' method in Python classes?",
            "options": ["Initialize instance attributes", "Delete the instance", "Create a new instance", "Compare two instances"],
            "answer_index": 0,
        }
    ]
    
    java_questions = [
        {
            "id": "q1",
            "topic": "basics",
            "difficulty": "easy",
            "question": "What is the entry point of a Java application?",
            "options": ["main method", "constructor", "init method", "start method"],
            "answer_index": 0,
        },
        {
            "id": "q2",
            "topic": "data_types",
            "difficulty": "easy",
            "question": "Which Java data type has a fixed size?",
            "options": ["ArrayList", "String", "LinkedList", "HashMap"],
            "answer_index": 1,
        },
        {
            "id": "q3",
            "topic": "oop",
            "difficulty": "medium",
            "question": "What is polymorphism in Java?",
            "options": ["Method overriding", "Multiple inheritance", "Constructor chaining", "Interface implementation"],
            "answer_index": 0,
        },
        {
            "id": "q4",
            "topic": "collections",
            "difficulty": "medium",
            "question": "Which collection type maintains insertion order?",
            "options": ["HashSet", "TreeSet", "LinkedHashSet", "ArrayList"],
            "answer_index": 2,
        },
        {
            "id": "q5",
            "topic": "exceptions",
            "difficulty": "medium",
            "question": "What is the parent class of all exceptions in Java?",
            "options": ["RuntimeException", "Exception", "Error", "Throwable"],
            "answer_index": 3,
        }
    ]
    
    javascript_questions = [
        {
            "id": "q1",
            "topic": "basics",
            "difficulty": "easy",
            "question": "How do you declare a constant in JavaScript?",
            "options": ["const x = value;", "constant x = value;", "var x = value;", "final x = value;"],
            "answer_index": 0,
        },
        {
            "id": "q2",
            "topic": "data_types",
            "difficulty": "easy",
            "question": "What is the typeof null in JavaScript?",
            "options": ["object", "null", "undefined", "boolean"],
            "answer_index": 0,
        },
        {
            "id": "q3",
            "topic": "functions",
            "difficulty": "medium",
            "question": "What is a closure in JavaScript?",
            "options": ["Function with access to outer scope", "Function without return value", "Anonymous function", "Async function"],
            "answer_index": 0,
        },
        {
            "id": "q4",
            "topic": "arrays",
            "difficulty": "medium",
            "question": "Which array method does not modify the original array?",
            "options": ["map()", "sort()", "push()", "pop()"],
            "answer_index": 0,
        },
        {
            "id": "q5",
            "topic": "es6",
            "difficulty": "medium",
            "question": "How do you create a template literal in ES6?",
            "options": ["`template string`", "'template string'", "\"template string\"", "template string()"],
            "answer_index": 0,
        }
    ]
    
    cpp_questions = [
        {
            "id": "q1",
            "topic": "basics",
            "difficulty": "easy",
            "question": "Which header file is used for input/output operations in C++?",
            "options": ["<iostream>", "<stdio.h>", "<conio.h>", "<stdlib.h>"],
            "answer_index": 0,
        },
        {
            "id": "q2",
            "topic": "pointers",
            "difficulty": "easy",
            "question": "What operator is used to access the value pointed to by a pointer?",
            "options": ["*", "&", "->", "."],
            "answer_index": 0,
        },
        {
            "id": "q3",
            "topic": "memory",
            "difficulty": "medium",
            "question": "Which operator is used to allocate memory dynamically in C++?",
            "options": ["new", "malloc", "alloc", "create"],
            "answer_index": 0,
        },
        {
            "id": "q4",
            "topic": "oop",
            "difficulty": "medium",
            "question": "What is the purpose of a destructor in C++?",
            "options": ["Clean up resources", "Initialize objects", "Create objects", "Copy objects"],
            "answer_index": 0,
        },
        {
            "id": "q5",
            "topic": "templates",
            "difficulty": "hard",
            "question": "Which keyword is used to define a template in C++?",
            "options": ["template", "typename", "class", "interface"],
            "answer_index": 0,
        }
    ]
    
    sql_questions = [
        {
            "id": "q1",
            "topic": "basics",
            "difficulty": "easy",
            "question": "What SQL clause is used to filter results?",
            "options": ["WHERE", "FILTER", "HAVING", "GROUP BY"],
            "answer_index": 0,
        },
        {
            "id": "q2",
            "topic": "joins",
            "difficulty": "medium",
            "question": "Which JOIN returns all records from the left table?",
            "options": ["LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "FULL JOIN"],
            "answer_index": 0,
        },
        {
            "id": "q3",
            "topic": "aggregation",
            "difficulty": "medium",
            "question": "Which function counts the number of rows in a result set?",
            "options": ["COUNT()", "SUM()", "AVG()", "MAX()"],
            "answer_index": 0,
        },
        {
            "id": "q4",
            "topic": "subqueries",
            "difficulty": "hard",
            "question": "Which clause is used to filter results of an aggregation?",
            "options": ["HAVING", "WHERE", "GROUP BY", "ORDER BY"],
            "answer_index": 0,
        },
        {
            "id": "q5",
            "topic": "indexes",
            "difficulty": "hard",
            "question": "Which type of index is automatically created for primary keys?",
            "options": ["Clustered", "Unique", "Non-clustered", "Bitmap"],
            "answer_index": 2,
        }
    ]
    
    # Select questions based on programming language
    question_pools = {
        "Python": python_questions,
        "Java": java_questions,
        "JavaScript": javascript_questions,
        "C++": cpp_questions,
        "SQL": sql_questions
    }
    
    base_questions = question_pools.get(programming_language, python_questions)
    
    # Add more questions for a complete test
    additional_questions = [
        {
            "id": "q6",
            "topic": "debugging",
            "difficulty": "medium",
            "question": f"What is the purpose of a debugger in {programming_language}?",
            "options": ["Find and fix errors", "Optimize performance", "Write documentation", "Test code"],
            "answer_index": 0,
        },
        {
            "id": "q7",
            "topic": "best_practices",
            "difficulty": "medium",
            "question": "What is code review primarily used for?",
            "options": ["Quality assurance", "Performance optimization", "Documentation", "Testing"],
            "answer_index": 0,
        },
        {
            "id": "q8",
            "topic": "version_control",
            "difficulty": "easy",
            "question": "What does 'git commit' do?",
            "options": ["Save changes to repository", "Create new repository", "Delete repository", "Clone repository"],
            "answer_index": 0,
        },
        {
            "id": "q9",
            "topic": "testing",
            "difficulty": "medium",
            "question": "What is unit testing?",
            "options": ["Testing individual components", "Testing entire system", "Testing user interface", "Testing performance"],
            "answer_index": 0,
        },
        {
            "id": "q10",
            "topic": "algorithms",
            "difficulty": "hard",
            "question": "What is the time complexity of binary search on a sorted array?",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
            "answer_index": 1,
        },
        {
            "id": "q11",
            "topic": "complexity",
            "difficulty": "hard",
            "question": "Which sorting algorithm has the best average case time complexity?",
            "options": ["Quick Sort", "Merge Sort", "Heap Sort", "Bubble Sort"],
            "answer_index": 1,
        },
        {
            "id": "q12",
            "topic": "design_patterns",
            "difficulty": "hard",
            "question": "What is the Singleton design pattern used for?",
            "options": ["Ensure only one instance", "Create multiple instances", "Connect different components", "Define interfaces"],
            "answer_index": 0,
        },
        {
            "id": "q13",
            "topic": "security",
            "difficulty": "medium",
            "question": "What is SQL injection primarily used to prevent?",
            "options": ["Database attacks", "Network attacks", "File system attacks", "Memory attacks"],
            "answer_index": 0,
        },
        {
            "id": "q14",
            "topic": "performance",
            "difficulty": "medium",
            "question": "What is the main benefit of caching?",
            "options": ["Faster data access", "Data security", "Data compression", "Data validation"],
            "answer_index": 0,
        },
        {
            "id": "q15",
            "topic": "architecture",
            "difficulty": "hard",
            "question": "What is microservices architecture?",
            "options": ["Small independent services", "Monolithic application", "Layered architecture", "Event-driven architecture"],
            "answer_index": 0,
        },
        {
            "id": "q16",
            "topic": "scalability",
            "difficulty": "medium",
            "question": "What is horizontal scaling?",
            "options": ["Adding more machines", "Increasing server power", "Upgrading hardware", "Optimizing code"],
            "answer_index": 0,
        },
        {
            "id": "q17",
            "topic": "databases",
            "difficulty": "medium",
            "question": "What is a primary key in a database table?",
            "options": ["Unique identifier", "Foreign key reference", "Index key", "Sort key"],
            "answer_index": 0,
        },
        {
            "id": "q18",
            "topic": "web",
            "difficulty": "easy",
            "question": "What does HTTP stand for?",
            "options": ["HyperText Transfer Protocol", "High-Speed Transfer Protocol", "Home Page Text Protocol", "Hyperlink Text Transfer Protocol"],
            "answer_index": 0,
        },
        {
            "id": "q19",
            "topic": "mobile",
            "difficulty": "medium",
            "question": "What is responsive design?",
            "options": ["Adapts to screen sizes", "Fixed layout design", "Mobile-first design", "Desktop-only design"],
            "answer_index": 0,
        },
        {
            "id": "q20",
            "topic": "cloud",
            "difficulty": "medium",
            "question": "What is a primary benefit of cloud computing?",
            "options": ["Scalability and flexibility", "Local data storage", "Single server deployment", "Hardware ownership"],
            "answer_index": 0,
        }
    ]
    
    all_questions = base_questions + additional_questions
    
    # Shuffle questions to prevent repetition
    import random
    random.shuffle(all_questions)
    
    # Reassign IDs after shuffling
    for i, q in enumerate(all_questions):
        q["id"] = f"q{i+1}"
    
    return all_questions


def _render_registration() -> None:
    st.title("👤 Candidate Registration")
    st.write("Please provide your details to begin the assessment")
    
    with st.form("registration_form"):
        # Personal Information
        st.subheader("Personal Information")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", placeholder="John Doe")
            branch = st.text_input("Branch", placeholder="Computer Science")
        with col2:
            passing_year = st.number_input("Passing Year", min_value=1980, max_value=2100, value=2025, step=1)
            university = st.text_input("University", placeholder="MIT University")

        # Assessment Preferences
        st.subheader("Assessment Preferences")
        
        col1, col2 = st.columns(2)
        with col1:
            programming_language = st.selectbox("Programming Language", ["Python", "Java", "JavaScript", "C++", "SQL"])
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        with col2:
            assessment_type = st.selectbox("Question Type", ["MCQ only", "MCQ + Coding", "MCQ + Debugging", "Conceptual / Aptitude"])

        # Submit button
        submitted = st.form_submit_button("Start Assessment", type="primary")
        
        if submitted:
            if not name.strip() or not branch.strip() or not university.strip():
                st.error("Please fill all required fields")
                return
            
            # Store complete profile with all original fields
            st.session_state.profile = CandidateProfile(
                name=name.strip(),
                branch=branch.strip(),
                passing_year=int(passing_year),
                university=university.strip(),
                programming_language=programming_language,
                difficulty=difficulty,
                assessment_type=assessment_type
            )
            
            # Store registration data
            st.session_state.registration_data = {
                "name": name.strip(),
                "email": f"{name.strip().replace(' ', '.').lower()}@example.com",
                "branch": branch.strip(),
                "university": university.strip(),
                "programming_language": programming_language,
                "difficulty": difficulty,
                "assessment_type": assessment_type
            }
            
            # Initialize test configuration
            st.session_state.test_config = TestConfig(
                total_questions=20,
                duration_seconds=20 * 60,
            )
            st.session_state.attempt_id = str(uuid.uuid4())
            
            # Initialize agentic features
            st.session_state.assessment_started = False
            st.session_state.assessment_results = []
            st.session_state.current_question = None
            st.session_state.user_context = None
            
            _set_page("rules")


def _render_rules() -> None:
    st.title("📋 Assessment Rules")
    
    st.subheader("🎯 Test Format")
    st.write("• **20 Questions** - Multiple choice")
    st.write("• **20 Minutes** - Timed assessment")
    st.write("• **No Negative Marking** - Safe to attempt all")
    
    st.subheader("📚 Important Guidelines")
    st.write("• **No External Resources** - Use your knowledge only")
    st.write("• **Answer Independently** - Complete assessment on your own")
    st.write("• **Time Management** - Average 1 minute per question")
    st.write("• **Read Carefully** - Understand each question before answering")
    
    st.subheader("⚖️ Results")
    st.write("• **Instant Results** - Available immediately")
    st.write("• **Performance Analytics** - Detailed breakdown")
    
    st.divider()
    
    # Agreement
    agreed = st.checkbox("I have read and agree to follow these rules")
    
    if st.button("🚀 Start Assessment", type="primary", disabled=not agreed, use_container_width=True):
        st.session_state.accepted_rules = True
        # Reset test state
        st.session_state.current_q = 0
        st.session_state.answers = {}
        st.session_state.assessment_results = []
        st.session_state.assessment_started = False
        st.session_state.current_question = None
        st.session_state.system_check_passed = True
        st.session_state.test_started_at = time.time()
        _set_page("test")
        st.rerun()


def _inject_floating_camera_css():
    """Inject CSS for floating camera - called once"""
    if components is None:
        return
    components.html(
        """
        <style>
        .floating-camera {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 320px;
            height: 240px;
            z-index: 9999;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(0,0,0,0.4);
            background: black;
        }
        </style>
        """,
        height=0,
    )


def _calculate_response_time() -> float:
    """Calculate response time consistently"""
    try:
        q_start_time = st.session_state.get('q_start_time', time.time())
        return time.time() - q_start_time
    except (TypeError, AttributeError):
        return time.time()


def _render_test() -> None:
    """Render test page with auto-start camera and performance"""
    if not st.session_state.profile or not st.session_state.accepted_rules:
        st.error("Please register first")
        if st.button("Go to Registration"):
            _set_page("registration")
        return

    # Start assessment if not already started
    if not st.session_state.assessment_started:
        _start_agentic_assessment()
        st.session_state.assessment_started = True
        st.session_state.q_start_time = time.time()  # Track question start time
        st.rerun()  # Rerun #1: Questions generated
    
    # Show motivational message when assessment starts
    if st.session_state.assessment_started and 'motivation_shown' not in st.session_state:
        st.session_state.motivation_shown = True
        
        # Check if questions were generated by API or default
        if 'questions_generated_by_api' not in st.session_state:
            st.session_state.questions_generated_by_api = False
            
        # Show motivational message based on question source
        if st.session_state.questions_generated_by_api:
            st.success(" Best of luck! Your questions have been generated by AI specifically for you!")
        else:
            st.info(" All the best! You're using our curated question set.")

    # Check if time is up
    if _remaining_seconds() <= 0:
        _finalize_attempt("time_up")
        _set_page("result")
        return

    # Get current question
    questions = st.session_state.questions
    idx = st.session_state.current_q
    qid = questions[idx]["id"]
    q = questions[idx]

    # Layout with questions (camera already floating)
    col1, col2 = st.columns([3, 1])

    with col1:
        # Progress
        st.write(f"Question {idx + 1} of {len(questions)}")
        st.progress((idx + 1) / len(questions))
        
        # Real-time Timer Display with JavaScript
        remaining_seconds = _remaining_seconds()
        initial_minutes = remaining_seconds // 60
        initial_seconds = remaining_seconds % 60
        
        # Inject JavaScript for real-time timer
        components.html(f"""
        <div id="timer-container" style="
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, green, #ff6b6b);
            border-radius: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        ">
            <div id="timer-display" style="font-size: 48px; font-weight: bold; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                ⏱️ {initial_minutes:02d}:{initial_seconds:02d}
            </div>
            <div style="font-size: 18px; color: white; margin-top: 5px;">
                Time Remaining
            </div>
        </div>
        
        <script>
        let remainingSeconds = {remaining_seconds};
        const timerContainer = document.getElementById('timer-container');
        const timerDisplay = document.getElementById('timer-display');
        
        function updateTimer() {{
            if (remainingSeconds <= 0) {{
                timerDisplay.innerHTML = '🚨 00:00';
                timerContainer.style.background = 'linear-gradient(135deg, red, #ff0000)';
                // Auto-submit when time is up
                if (typeof window !== 'undefined') {{
                    setTimeout(() => {{
                        const submitBtn = document.querySelector('button[kind="primary"]');
                        if (submitBtn && submitBtn.textContent.includes('Finish Test')) {{
                            submitBtn.click();
                        }}
                    }}, 1000);
                }}
                return;
            }}
            
            const minutes = Math.floor(remainingSeconds / 60);
            const seconds = remainingSeconds % 60;
            
            // Update color based on remaining time
            if (remainingSeconds <= 60) {{
                timerContainer.style.background = 'linear-gradient(135deg, red, #ff6b6b)';
                timerDisplay.innerHTML = '🚨 ' + String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0');
            }} else if (remainingSeconds <= 300) {{
                timerContainer.style.background = 'linear-gradient(135deg, orange, #ffa500)';
                timerDisplay.innerHTML = '⏰ ' + String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0');
            }} else {{
                timerContainer.style.background = 'linear-gradient(135deg, green, #4caf50)';
                timerDisplay.innerHTML = '⏱️ ' + String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0');
            }}
            
            remainingSeconds--;
        }}
        
        // Update immediately and then every second
        updateTimer();
        setInterval(updateTimer, 1000);
        </script>
        """, height=120)

        # Question
        st.subheader(f"📝 {q.get('topic', 'General')}")
        st.write(q["question"])
        
        # Options - Use consistent key to prevent reruns
        existing_answer = st.session_state.answers.get(qid)
        selected_index = existing_answer if existing_answer is not None else None
        
        # Update radio button state when question changes
        if "last_qid" not in st.session_state or st.session_state.last_qid != qid:
            st.session_state.last_qid = qid
            # Clear radio selection for new question
            if "answer_radio" in st.session_state:
                del st.session_state.answer_radio
        
        selected = st.radio("Choose answer:", q["options"], index=selected_index, key="answer_radio")

        # Navigation - Reduced reruns
        col1_nav, col2_nav, col3_nav = st.columns(3)
        with col1_nav:
            if st.button("Previous", key=f"prev_{qid}") and idx > 0:
                st.session_state.current_q = idx - 1
                # No rerun - let state update naturally
        
        with col2_nav:
            if st.button("Next", key=f"next_{qid}", use_container_width=False):
                # Check if an answer is selected
                if selected is None:
                    st.error("Please select an answer before proceeding.")
                    return
                
                # Save answer
                st.session_state.answers[qid] = q["options"].index(selected)
                
                # Update user context
                if st.session_state.user_context:
                    user_context = st.session_state.user_context
                    is_correct = q["options"].index(selected) == q["answer_index"]
                    
                    user_context.questions_attempted += 1
                    if is_correct:
                        user_context.correct_answers += 1
                        user_context.current_streak += 1
                        user_context.confidence_level = min(1.0, user_context.confidence_level + 0.05)
                    else:
                        user_context.current_streak = 0
                        user_context.confidence_level = max(0.1, user_context.confidence_level - 0.02)
                    
                    user_context.current_score = user_context.correct_answers / user_context.questions_attempted
                    user_context.average_response_time = (user_context.average_response_time * (user_context.questions_attempted - 1) + _calculate_response_time()) / user_context.questions_attempted
                    
                    st.session_state.user_context = user_context
                
                # Store assessment result (only once)
                if 'assessment_results' not in st.session_state:
                    st.session_state.assessment_results = []
                
                st.session_state.assessment_results.append({
                    "question_id": qid,
                    "question_text": q["question"],
                    "topic": q.get("topic", "General"),
                    "difficulty": q.get("difficulty", "medium"),
                    "user_answer": selected,
                    "correct_answer": q["options"][q["answer_index"]],
                    "is_correct": q["options"].index(selected) == q["answer_index"],
                    "response_time": _calculate_response_time(),
                    "confidence": 0.8 if q["options"].index(selected) == q["answer_index"] else 0.3
                })
                
                # Reset question start time for next question
                st.session_state.q_start_time = time.time()
                
                # Navigate to next question or finish
                if idx < len(questions) - 1:
                    st.session_state.current_q = idx + 1
                    # No rerun - let state update naturally
                else:
                    _finalize_attempt("completed")
                    _set_page("result")  # Rerun #2: Test completion
                    
        with col3_nav:
            if st.button("🏁 Finish Test", key=f"finish_{qid}", type="secondary"):
                # Check if an answer is selected
                if selected is None:
                    st.error("Please select an answer before finishing.")
                    # No rerun - just show error
                    return
                
                # Calculate response time
                try:
                    response_time = time.time() - st.session_state.get('q_start_time', time.time())
                except (TypeError, AttributeError):
                    response_time = time.time()
                
                # Save current answer before finishing
                st.session_state.answers[qid] = q["options"].index(selected)
                
                # Update user context
                if st.session_state.user_context:
                    user_context = st.session_state.user_context
                    is_correct = q["options"].index(selected) == q["answer_index"]
                    
                    user_context.questions_attempted += 1
                    if is_correct:
                        user_context.correct_answers += 1
                        user_context.current_streak += 1
                        user_context.confidence_level = min(1.0, user_context.confidence_level + 0.05)
                    else:
                        user_context.current_streak = 0
                        user_context.confidence_level = max(0.1, user_context.confidence_level - 0.02)
                    
                    user_context.current_score = user_context.correct_answers / user_context.questions_attempted
                    user_context.average_response_time = (user_context.average_response_time * (user_context.questions_attempted - 1) + response_time) / user_context.questions_attempted
                    
                    st.session_state.user_context = user_context
                
                # Store final assessment result
                if 'assessment_results' not in st.session_state:
                    st.session_state.assessment_results = []
                
                st.session_state.assessment_results.append({
                    "question_id": qid,
                    "question_text": q["question"],
                    "topic": q.get("topic", "General"),
                    "difficulty": q.get("difficulty", "medium"),
                    "user_answer": selected,
                    "correct_answer": q["options"][q["answer_index"]],
                    "is_correct": q["options"].index(selected) == q["answer_index"],
                    "response_time": response_time,
                    "confidence": 0.8 if q["options"].index(selected) == q["answer_index"] else 0.3
                })
                
                # Finalize test
                _finalize_attempt("user_finished")
                _set_page("result")  # Rerun #2: Test completion

    with col2:
        # Simple progress indicator
        if st.session_state.answers:
            answered = len(st.session_state.answers)
            total = len(st.session_state.questions)
            progress = answered / total if total > 0 else 0
            st.write(f"**Progress:** {answered}/{total} questions answered")
            st.progress(progress)
        
        # Simple navigation hint
        st.write("**Navigation Tips:**")
        st.write("• Use Previous/Next buttons to navigate")
        st.write("• Select an answer before proceeding")
        st.write("• Finish Test when complete")


def _start_agentic_assessment():
    """Initialize agentic assessment with all features - optimized"""
    try:
        # Initialize user context only if not already done
        if 'user_context' not in st.session_state:
            if ADAPTIVE_AGENT_AVAILABLE:
                user_data = st.session_state.registration_data
                st.session_state.user_context = create_adaptive_question_agent(AI_PROVIDERS)(
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
                logging.info("Adaptive agent initialized successfully")
            else:
                # Fallback UserContext if adaptive agent not available
                from dataclasses import dataclass
                from typing import List
                
                @dataclass
                class UserContext:
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
                
                user_data = st.session_state.registration_data
                st.session_state.user_context = UserContext(
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
                logging.warning("Using fallback UserContext - adaptive agent not available")
        
        # Initialize camera by default (always on)
        if INTEGRATED_ASSESSMENT_AVAILABLE:
            pass  # Camera disabled - using simplified version
        
        # Generate questions only if not already done
        if not st.session_state.questions:
            _ensure_test_seeded()
        
        logging.info("Agentic assessment started successfully")
        
    except Exception as e:
        logging.error(f"Failed to start agentic assessment: {e}")
        # Fallback to standard assessment
        st.session_state.user_context = None


def _render_result() -> None:
    if not st.session_state.profile:
        st.error("No profile found")
        if st.button("Go to Registration"):
            _set_page("registration")
        return

    result = st.session_state.result
    if not result:
        st.error("No result found")
        return

    st.title("🎉 Assessment Complete")
    
    # Basic metrics
    metrics = result.get('metrics', {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("✅ Correct", metrics.get('correct', 0))
    with col2:
        st.metric("❌ Incorrect", metrics.get('incorrect', 0))
    with col3:
        st.metric("📊 Total", metrics.get('total_questions', 0))
    with col4:
        st.metric("⏱️ Time", f"{metrics.get('time_taken_seconds', 0)}s")

    # Agentic Insights
    if st.session_state.user_context:
        st.subheader("🤖 AI Insights")
        user_context = st.session_state.user_context
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Final Score", f"{user_context.current_score*100:.1f}%")
            st.metric("🔥 Best Streak", user_context.current_streak)
            st.metric("⚡ Confidence", f"{user_context.confidence_level*100:.1f}%")
        with col2:
            st.metric("📈 Engagement", f"{user_context.engagement_level*100:.1f}%")
            st.metric("🧠 Stress Level", f"{user_context.stress_indicators*100:.1f}%")
            st.metric("⏱️ Avg Response", f"{user_context.average_response_time:.1f}s")

    # Performance Analysis
    if st.session_state.assessment_results:
        st.subheader("📈 Performance Analysis")
        
        results = st.session_state.assessment_results
        correct_by_topic = {}
        total_by_topic = {}
        
        for result in results:
            topic = result.get('topic', 'General')
            if topic not in total_by_topic:
                total_by_topic[topic] = 0
                correct_by_topic[topic] = 0
            total_by_topic[topic] += 1
            if result.get('is_correct'):
                correct_by_topic[topic] += 1
        
        # Show topic performance
        for topic in total_by_topic:
            accuracy = (correct_by_topic[topic] / total_by_topic[topic] * 100) if total_by_topic[topic] > 0 else 0
            st.write(f"**{topic}:** {correct_by_topic[topic]}/{total_by_topic[topic]} ({accuracy:.1f}%)")

    # Recent Questions
    if st.session_state.assessment_results:
        st.subheader("📝 Recent Questions")
        for i, result in enumerate(st.session_state.assessment_results[-3:], 1):
            with st.expander(f"Question {len(st.session_state.assessment_results) - 3 + i}: {result.get('topic', 'Unknown')}"):
                st.write(f"**Question:** {result.get('question_text', 'N/A')}")
                st.write(f"**Result:** {'✅ Correct' if result.get('is_correct') else '❌ Incorrect'}")
                if result.get('response_time'):
                    st.write(f"**Time:** {result['response_time']:.1f}s")
                st.write(f"**Confidence:** {result.get('confidence', 0)*100:.1f}%")

    # AI Recommendations
    if st.session_state.user_context:
        st.subheader("🚀 AI Recommendations")
        user_context = st.session_state.user_context
        
        if user_context.current_score > 0.8:
            st.success("🎯 Excellent performance! Consider advanced topics next.")
        elif user_context.current_score > 0.6:
            st.info("📚 Good performance! Focus on weak areas for improvement.")
        else:
            st.warning("💪 Keep practicing! Review fundamentals and try again.")
        
        if user_context.weak_topics:
            st.write(f"**Areas to improve:** {', '.join(user_context.weak_topics)}")
        if user_context.strong_topics:
            st.write(f"**Strength areas:** {', '.join(user_context.strong_topics)}")

    st.write(f"Attempt ID: {result.get('attempt_id')}")
    st.write(f"Finalized: {result.get('finalized_reason')}")
    
    if st.button("🔄 Start New Attempt"):
        _init_state()
        _set_page("registration")


def _finalize_attempt(reason: str) -> None:
    """Finalize the assessment attempt - simplified to avoid duplicates"""
    if not st.session_state.profile or not st.session_state.questions:
        return
    
    # Calculate time taken
    time_taken = 0
    if st.session_state.test_started_at:
        time_taken = int(time.time() - st.session_state.test_started_at)
    
    # Calculate final metrics
    total_questions = len(st.session_state.questions)
    correct = sum(
        1
        for q in st.session_state.questions
        if st.session_state.answers.get(q["id"]) == q["answer_index"]
    )
    
    # Create result object
    result = {
        'attempt_id': st.session_state.get('attempt_id', 'unknown'),
        'finalized_reason': reason,
        'metrics': {
            'total_questions': total_questions,
            'correct': correct,
            'incorrect': total_questions - correct,
            'time_taken_seconds': time_taken
        }
    }
    
    # Store result
    st.session_state.result = result
    
    # Stop camera
    st.session_state.test_submitted = True
    if components:
        components.html(
            """
            <script>
            if (window._cameraStream) {
                console.log('🛑 Stopping camera stream...');
                window._cameraStream.getTracks().forEach(t => t.stop());
                window._cameraStream = null;
            }
            </script>
            """,
            height=0
        )
    
    # Navigate to result page
    _set_page("result")
    answers = st.session_state.answers
    profile = st.session_state.profile
    
    # Calculate metrics
    correct = 0
    incorrect = 0
    
    for q in questions:
        user_answer = answers.get(q["id"])
        is_correct = user_answer is not None and user_answer == q["answer_index"]
        
        if is_correct:
            correct += 1
        else:
            incorrect += 1
    
    # Store result
    st.session_state.result = {
        "attempt_id": st.session_state.attempt_id,
        "finalized_reason": reason,
        "metrics": {
            "correct": correct,
            "incorrect": incorrect,
            "total_questions": len(questions),
            "time_taken_seconds": time_taken,
            "accuracy": (correct / len(questions) * 100) if questions else 0
        },
        "profile": asdict(profile),
        "answers": answers,
        "submitted_at": datetime.now().isoformat()
    }
    
    # Update user context if available
    if st.session_state.user_context:
        user_context = st.session_state.user_context
        user_context.questions_attempted = len(questions)
        user_context.correct_answers = correct
        user_context.current_score = correct / len(questions) if questions else 0
        user_context.average_response_time = time_taken / len(questions) if questions else 0


def _remaining_seconds() -> int:
    """Calculate remaining time for the test"""
    if not st.session_state.test_started_at or not st.session_state.test_config:
        return 0
    
    elapsed = int(time.time() - st.session_state.test_started_at)
    remaining = st.session_state.test_config.duration_seconds - elapsed
    return max(0, remaining)


def _ensure_test_seeded() -> None:
    """Generate questions from knowledge base using APIs with no repetition"""
    if st.session_state.questions:
        return

    profile = st.session_state.profile
    test_config = st.session_state.test_config
    
    # Try AI providers first with knowledge base context
    questions = _generate_questions_from_knowledge_base(profile, test_config)
    
    if questions:
        st.session_state.questions = questions
        st.session_state.questions_generated_by_api = True
        logging.info(f"Generated {len(questions)} questions using AI from knowledge base")
    else:
        # Fallback to default questions
        st.session_state.questions = _get_default_questions(
            profile.programming_language,
            profile.difficulty
        )
        st.session_state.questions_generated_by_api = False
        logging.info("Using default questions due to API failure")


def _generate_questions_from_knowledge_base(profile: CandidateProfile, test_config: TestConfig) -> list:
    """Generate questions using AI providers with knowledge base context"""
    try:
        # Load knowledge base content
        knowledge_content = _load_knowledge_base(profile.programming_language)
        
        if not knowledge_content:
            logging.warning(f"No knowledge base found for {profile.programming_language}")
            return None
        
        # Load used question hashes to avoid repetition
        used_hashes = _load_used_question_hashes()
        
        # Try AI providers with knowledge base context
        providers = [
            ('groq', _generate_questions_with_groq_kb),
            ('openrouter', _generate_questions_with_openrouter_kb),
            ('perplexity', _generate_questions_with_perplexity_kb),
            ('gemini', lambda p, t: _generate_mcqs_with_gemini_kb(p, t, knowledge_content, used_hashes, recursion_depth=0))
        ]
        
        for provider_name, provider_func in providers:
            try:
                logging.info(f"Trying {provider_name} for question generation with knowledge base...")
                questions = provider_func(profile, test_config, knowledge_content, used_hashes)
                
                if questions and len(questions) > 0:
                    # Save used question hashes
                    _save_used_question_hashes(questions)
                    logging.info(f"Successfully generated {len(questions)} questions using {provider_name}")
                    return questions
                else:
                    logging.warning(f"{provider_name} returned no questions")
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "limit" in error_msg or "exceeded" in error_msg or "429" in error_msg:
                    logging.warning(f"{provider_name} quota exceeded, trying next provider")
                    continue
                else:
                    logging.error(f"Error with {provider_name}: {e}")
                    continue
        
        return None
        
    except Exception as e:
        logging.error(f"Failed to generate questions from knowledge base: {e}")
        return None


def _load_knowledge_base(programming_language: str) -> str:
    """Load knowledge base content for the given programming language"""
    import os
    
    knowledge_file = f"data/knowledge_base/{programming_language.lower()}_syllabus.txt"
    
    if os.path.exists(knowledge_file):
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        logging.warning(f"Knowledge base file not found: {knowledge_file}")
        return None


def _load_used_question_hashes() -> set:
    """Load used question hashes to avoid repetition"""
    import os
    import json
    
    hash_file = "data/knowledge_base/used_question_hashes.json"
    
    if os.path.exists(hash_file):
        try:
            with open(hash_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('hashes', []))
        except Exception as e:
            logging.error(f"Error loading question hashes: {e}")
            return set()
    
    return set()


def _save_used_question_hashes(questions: list) -> None:
    """Save question hashes to avoid repetition"""
    import os
    import json
    import hashlib
    
    hash_file = "data/knowledge_base/used_question_hashes.json"
    
    # Generate hashes for new questions
    new_hashes = []
    for q in questions:
        # Create hash from question text
        question_text = q.get('question', '')
        if question_text:
            hash_obj = hashlib.md5(question_text.encode('utf-8'))
            new_hashes.append(hash_obj.hexdigest())
    
    # Load existing hashes
    existing_data = {}
    if os.path.exists(hash_file):
        try:
            with open(hash_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except Exception as e:
            logging.error(f"Error loading existing hashes: {e}")
    
    # Merge hashes
    all_hashes = existing_data.get('hashes', [])
    all_hashes.extend(new_hashes)
    
    # Keep only last 1000 hashes to prevent file from growing too large
    if len(all_hashes) > 1000:
        all_hashes = all_hashes[-1000:]
    
    # Save updated hashes
    updated_data = {
        'hashes': list(set(all_hashes)),  # Remove duplicates
        'last_updated': time.time()
    }
    
    try:
        with open(hash_file, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=2)
        logging.info(f"Saved {len(new_hashes)} new question hashes")
    except Exception as e:
        logging.error(f"Error saving question hashes: {e}")


def _generate_questions_with_groq_kb(profile: CandidateProfile, test_config: TestConfig, knowledge_content: str, used_hashes: set) -> list:
    """Generate questions using Groq with knowledge base context"""
    if not GROQ_AVAILABLE or not Groq:
        return None
    
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        prompt = f"""Generate exactly {test_config.total_questions} multiple-choice questions for {profile.programming_language} at {profile.difficulty.lower()} difficulty level.

Knowledge Base Context:
{knowledge_content}

Requirements:
- Generate questions based ONLY on the knowledge base provided above
- Questions must be technical and relevant to the topics covered
- Include a mix of topics from the knowledge base
- Questions should NOT be repeated (avoid these hashes: {list(used_hashes)[:10]})
- Each question must have 4 options with exactly one correct answer
- Clear, unambiguous correct answers

Format as JSON array:
[
  {{
    "id": "q1",
    "topic": "topic_name",
    "difficulty": "easy|medium|hard",
    "question": "Your question here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer_index": 0
  }}
]"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        raw = response.choices[0].message.content.strip()
        questions = json.loads(raw)
        
        if not isinstance(questions, list) or len(questions) == 0:
            return None
            
        return questions
        
    except Exception as e:
        logging.error(f"Groq knowledge base generation error: {e}")
        return None


def _generate_questions_with_openrouter_kb(profile: CandidateProfile, test_config: TestConfig, knowledge_content: str, used_hashes: set) -> list:
    """Generate questions using OpenRouter with knowledge base context"""
    if not OPENAI_AVAILABLE or not OpenAI:
        return None
    
    try:
        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        
        prompt = f"""Generate exactly {test_config.total_questions} multiple-choice questions for {profile.programming_language} at {profile.difficulty.lower()} difficulty level.

Knowledge Base Context:
{knowledge_content}

Requirements:
- Generate questions based ONLY on the knowledge base provided above
- Questions must be technical and relevant to the topics covered
- Include a mix of topics from the knowledge base
- Questions should NOT be repeated (avoid these hashes: {list(used_hashes)[:10]})
- Each question must have 4 options with exactly one correct answer
- Clear, unambiguous correct answers

Format as JSON array:
[
  {{
    "id": "q1",
    "topic": "topic_name",
    "difficulty": "easy|medium|hard",
    "question": "Your question here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer_index": 0
  }}
]"""

        response = client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        raw = response.choices[0].message.content.strip()
        questions = json.loads(raw)
        
        if not isinstance(questions, list) or len(questions) == 0:
            return None
            
        return questions
        
    except Exception as e:
        logging.error(f"OpenRouter knowledge base generation error: {e}")
        return None


def _generate_questions_with_perplexity_kb(profile: CandidateProfile, test_config: TestConfig, knowledge_content: str, used_hashes: set) -> list:
    """Generate questions using Perplexity with knowledge base context"""
    if not OPENAI_AVAILABLE or not OpenAI:
        return None
    
    try:
        client = OpenAI(api_key=os.getenv("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai/")
        
        prompt = f"""Generate exactly {test_config.total_questions} multiple-choice questions for {profile.programming_language} at {profile.difficulty.lower()} difficulty level.

Knowledge Base Context:
{knowledge_content}

Requirements:
- Generate questions based ONLY on the knowledge base provided above
- Questions must be technical and relevant to the topics covered
- Include a mix of topics from the knowledge base
- Questions should NOT be repeated (avoid these hashes: {list(used_hashes)[:10]})
- Each question must have 4 options with exactly one correct answer
- Clear, unambiguous correct answers

Format as JSON array:
[
  {{
    "id": "q1",
    "topic": "topic_name",
    "difficulty": "easy|medium|hard",
    "question": "Your question here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer_index": 0
  }}
]"""

        response = client.chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        raw = response.choices[0].message.content.strip()
        questions = json.loads(raw)
        
        if not isinstance(questions, list) or len(questions) == 0:
            return None
            
        return questions
        
    except Exception as e:
        logging.error(f"Perplexity knowledge base generation error: {e}")
        return None


def _generate_mcqs_with_gemini_kb(profile: CandidateProfile, test_config: TestConfig, knowledge_content: str, used_hashes: set, recursion_depth: int = 0) -> list:
    """Generate questions using Gemini with knowledge base context"""
    # Prevent infinite recursion
    if recursion_depth > 2:
        logging.error("Max recursion depth reached")
        return None
    
    model_name = _get_available_gemini_model()
    if not model_name:
        st.error("❌ No compatible Gemini model found.")
        return []

    prompt = f"""Generate exactly {test_config.total_questions} multiple-choice questions for {profile.programming_language} at {profile.difficulty.lower()} difficulty level.

Knowledge Base Context:
{knowledge_content}

Requirements:
- Generate questions based ONLY on the knowledge base provided above
- Questions must be technical and relevant to the topics covered
- Include a mix of topics from the knowledge base
- Questions should NOT be repeated (avoid these hashes: {list(used_hashes)[:10]})
- Each question must have 4 options with exactly one correct answer
- Clear, unambiguous correct answers

Format as JSON array:
[
  {{
    "id": "q1",
    "topic": "topic_name",
    "difficulty": "easy|medium|hard",
    "question": "Your question here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer_index": 0
  }}
]"""

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        raw = response.text.strip()
        
        if not raw:
            logging.warning("Empty response from Gemini API")
            return None
        
        questions = json.loads(raw)
        if not isinstance(questions, list):
            raise ValueError("Response is not a list")
        
        if len(questions) == 0:
            logging.warning("Empty questions list from Gemini")
            return None
        
        return questions
        
    except Exception as e:
        logging.error(f"Gemini knowledge base generation error: {e}")
        return None


def _generate_questions_with_any_provider(profile: CandidateProfile, test_config: TestConfig) -> list:
    """Try all available AI providers with quota management"""
    
    # Provider priority order
    providers = [
        ('groq', _generate_questions_with_groq_kb),
        ('openrouter', _generate_questions_with_openrouter_kb),
        ('perplexity', _generate_questions_with_perplexity_kb),
        ('gemini', lambda p, t: _generate_mcqs_with_gemini(p, t, recursion_depth=0))
    ]
    
    for provider_name, provider_func in providers:
        try:
            logging.info(f"Trying {provider_name} for question generation...")
            questions = provider_func(profile, test_config)
            
            if questions and len(questions) > 0:
                logging.info(f"Successfully generated {len(questions)} questions using {provider_name}")
                return questions
            else:
                logging.warning(f"{provider_name} returned no questions")
                
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "limit" in error_msg or "exceeded" in error_msg or "429" in error_msg:
                logging.warning(f"{provider_name} quota exceeded, trying next provider...")
                # Try rotating API key if it's Gemini
                if provider_name == 'gemini':
                    # Simple key rotation disabled - using single key
                    logging.warning("Gemini quota exceeded, trying next provider...")
                    continue
                elif "json" in error_msg or "parse" in error_msg:
                    logging.warning(f"{provider_name} returned invalid JSON, trying next provider...")
                    continue
                else:
                    logging.error(f"{provider_name} failed: {e}")
                    continue
            else:
                logging.error(f"{provider_name} failed: {e}")
                continue
    
    # If all providers fail, return None to use default questions
    logging.warning("All AI providers failed, using default questions")
    return None


def _generate_mcqs_with_gemini(profile: CandidateProfile, test_config: TestConfig, recursion_depth: int = 0) -> list:
    """Generate questions using Gemini API"""
    # Prevent infinite recursion
    if recursion_depth > 2:
        logging.error("Max recursion depth reached")
        return None
    
    model_name = _get_available_gemini_model()
    if not model_name:
        st.error("❌ No compatible Gemini model found.")
        return []

    prompt = f"""Generate exactly {test_config.total_questions} multiple-choice questions for {profile.programming_language} at {profile.difficulty.lower()} difficulty level.

Format as JSON array:
[
  {{
    "id": "q1",
    "topic": "basics",
    "difficulty": "easy",
    "question": "Your question here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer_index": 0
  }}
]

Requirements:
- Questions must be technical and relevant
- Include topics: data structures, algorithms, syntax, best practices
- Mix of difficulty levels
- Clear, unambiguous correct answers"""

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        raw = response.text.strip()
        
        if not raw:
            logging.warning("Empty response from Gemini API")
            return None
        
        questions = json.loads(raw)
        if not isinstance(questions, list):
            raise ValueError("Response is not a list")
        
        if len(questions) == 0:
            logging.warning("Empty questions list from Gemini")
            return None
        
        return questions
        
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "limit" in error_msg or "exceeded" in error_msg:
            st.warning("⚠️ API quota exceeded. Using default questions.")
            logging.warning(f"API quota exceeded: {e}")
        else:
            logging.error(f"Question generation failed: {e}")
        return None


def _get_available_gemini_model():
    """Get available Gemini model"""
    try:
        models = list(genai.list_models())
        preferred_models = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-pro',
            'gemini-pro-latest',
            'gemini-flash-latest'
        ]
        
        for preferred in preferred_models:
            for m in models:
                if preferred in m.name and "generateContent" in m.supported_generation_methods:
                    logging.info(f"Selected model: {preferred}")
                    return m.name.split("/")[-1]
        
        # Fallback to any available model
        for m in models:
            if "generateContent" in m.supported_generation_methods:
                logging.warning(f"Using fallback model: {m.name}")
                return m.name.split("/")[-1]
        return None
    except Exception as e:
        logging.error(f"Failed to list models: {e}")
        return None


def _generate_questions_with_groq(profile: CandidateProfile, test_config: TestConfig) -> list:
    """Generate questions using Groq API"""
    if 'groq' not in AI_PROVIDERS:
        return None
    
    try:
        prompt = f"""Generate exactly {test_config.total_questions} multiple-choice questions for {profile.programming_language} at {profile.difficulty.lower()} difficulty level.

Format as JSON array:
[
  {{
    "id": "q1",
    "topic": "basics",
    "difficulty": "easy",
    "question": "Your question here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer_index": 0
  }}
]"""

        response = AI_PROVIDERS['groq'].chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        raw = response.choices[0].message.content.strip()
        
        if not raw:
            logging.warning("Groq returned empty response")
            return None
        
        # Try to extract JSON from response
        try:
            questions = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
            else:
                logging.error(f"Groq response parsing failed: {raw[:200]}...")
                return None
        
        if isinstance(questions, list) and len(questions) > 0:
            logging.info(f"Generated {len(questions)} questions with Groq")
            return questions
        else:
            return None
            
    except Exception as e:
        logging.error(f"Groq generation failed: {e}")
        return None


def _generate_questions_with_openrouter(profile: CandidateProfile, test_config: TestConfig) -> list:
    """Generate questions using OpenRouter API"""
    if 'openrouter' not in AI_PROVIDERS:
        return None
    
    try:
        prompt = f"""Generate exactly {test_config.total_questions} multiple-choice questions for {profile.programming_language} at {profile.difficulty.lower()} difficulty level.

Format as JSON array:
[
  {{
    "id": "q1",
    "topic": "basics",
    "difficulty": "easy",
    "question": "Your question here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer_index": 0
  }}
]"""

        response = AI_PROVIDERS['openrouter'].chat.completions.create(
            model="meta-llama/llama-3.2-3b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        raw = response.choices[0].message.content.strip()
        
        if not raw:
            logging.warning("OpenRouter returned empty response")
            return None
        
        # Try to extract JSON from response
        try:
            questions = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
            else:
                logging.error(f"OpenRouter response parsing failed: {raw[:200]}...")
                return None
        
        if isinstance(questions, list) and len(questions) > 0:
            logging.info(f"Generated {len(questions)} questions with OpenRouter")
            return questions
        else:
            return None
            
    except Exception as e:
        logging.error(f"OpenRouter generation failed: {e}")
        return None


def _generate_questions_with_perplexity(profile: CandidateProfile, test_config: TestConfig) -> list:
    """Generate questions using Perplexity API"""
    if 'perplexity' not in AI_PROVIDERS:
        return None
    
    try:
        prompt = f"""Generate exactly {test_config.total_questions} multiple-choice questions for {profile.programming_language} at {profile.difficulty.lower()} difficulty level.

Format as JSON array:
[
  {{
    "id": "q1",
    "topic": "basics",
    "difficulty": "easy",
    "question": "Your question here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer_index": 0
  }}
]"""

        response = AI_PROVIDERS['perplexity'].chat.completions.create(
            model="llama-3.1-sonar-small-128k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        raw = response.choices[0].message.content.strip()
        
        if not raw:
            logging.warning("Perplexity returned empty response")
            return None
        
        # Try to extract JSON from response
        try:
            questions = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
            else:
                logging.error(f"Perplexity response parsing failed: {raw[:200]}...")
                return None
        
        if isinstance(questions, list) and len(questions) > 0:
            logging.info(f"Generated {len(questions)} questions with Perplexity")
            return questions
        else:
            return None
            
    except Exception as e:
        logging.error(f"Perplexity generation failed: {e}")
        return None


def _render_agentic_results() -> None:
    """Simplified agentic assessment results"""
    if 'assessment_results' not in st.session_state or not st.session_state.assessment_results:
        st.warning("No assessment results available.")
        return
    
    results = st.session_state.assessment_results
    user_context = st.session_state.get('user_context')
    
    st.title("🎯 Assessment Results")
    
    # Simple metrics
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r.get('is_correct', False))
    accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Questions", total_questions)
    with col2:
        st.metric("✅ Correct", correct_answers)
    with col3:
        st.metric("🎯 Score", f"{accuracy:.1f}%")
    with col4:
        if user_context:
            st.metric("⚡ Confidence", f"{user_context.confidence_level*100:.1f}%")
    
    # Simple performance analysis
    st.subheader("📈 Performance Analysis")
    if user_context:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Average Response Time:** {user_context.average_response_time:.1f}s")
            st.write(f"**Current Streak:** {user_context.current_streak}")
        with col2:
            st.write(f"**Engagement Level:** {user_context.engagement_level*100:.1f}%")
            st.write(f"**Stress Level:** {user_context.stress_indicators*100:.1f}%")
    
    # Question breakdown
    st.subheader("📝 Question Review")
    for i, result in enumerate(results[:5], 1):  # Show only first 5
        with st.expander(f"Question {i}: {result.get('topic', 'Unknown')}"):
            st.write(f"**Question:** {result.get('question_text', 'N/A')}")
            st.write(f"**Result:** {'✅ Correct' if result.get('is_correct') else '❌ Incorrect'}")
            if result.get('response_time'):
                st.write(f"**Time:** {result['response_time']:.1f}s")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Retake Test", type="primary"):
            for key in ['assessment_started', 'assessment_results', 'user_context', 'current_question']:
                if key in st.session_state:
                    del st.session_state[key]
            _set_page("registration")
    with col2:
        if st.button("📊 Detailed Analytics"):
            st.info("Detailed analytics coming soon!")
    with col3:
        if st.button("🏠 Home"):
            _init_state()
            _set_page("registration")


def main() -> None:
    _init_state()
    
    # ✅ ALWAYS call camera (runs every rerun)
    render_camera()   # 👈 runs every rerun
    
    page = st.session_state.page
    if page == "registration":
        _render_registration()
    elif page == "rules":
        _render_rules()
    elif page == "test":
        _render_test()
    elif page == "result":
        _render_result()
    else:
        _render_registration()


if __name__ == "__main__":
    main()
