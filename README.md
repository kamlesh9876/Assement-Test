# 🚀 Nelumbus AI Assessment System

Enterprise-grade AI-powered assessment platform with advanced agentic features, multi-LLM support, and intelligent quota management.

## 🌟 Key Features

### 🤖 **Agentic Assessment System**
- **📊 Real-time User Context Tracking** - Score, confidence, engagement, stress levels
- **📷 Corner Camera Integration** - Visual monitoring with user control
- **🧠 AI-Powered Hints** - Intelligent assistance with caching
- **📈 Performance Analytics** - Topic-wise analysis and recommendations
- **🔄 Adaptive Question Generation** - Dynamic question creation
- **🎯 Smart Recommendations** - Personalized learning guidance

### 🔧 **Multi-LLM Support**
- **🤖 Google Gemini** - Primary AI provider with key rotation
- **🦾 Groq** - Fast responses with llama models
- **🌐 OpenRouter** - Multiple model access
- **🔍 Perplexity** - Advanced reasoning capabilities
- **🔄 Automatic Fallback** - Seamless API switching on quota issues

### 📊 **Assessment Features**
- **20 Questions** - Multiple choice format
- **20 Minutes** - Fixed duration with timer
- **No Negative Marking** - Safe assessment environment
- **Instant Results** - Real-time performance metrics
- **Detailed Analytics** - Comprehensive performance breakdown

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Web Framework)
- **Backend**: Python 3.8+
- **AI Models**: Gemini, Groq, OpenRouter, Perplexity
- **Data Storage**: JSON files with structured results
- **Camera**: Streamlit WebRTC integration
- **Environment**: Cross-platform compatible

## 📦 Installation

### 1. **Clone the Repository**
```bash
git clone https://github.com/kamlesh9876/DOC-Analyser-Using-LLM.git
cd "Nelumbus Technologies/LLM-learning/Nelumbus"
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Set Up Environment Variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. **Run the Application**
```bash
streamlit run app.py
```

## 🔧 Configuration

### API Keys Required
Add the following to your `.env` file:

```env
# Primary Gemini API Key
GEMINI_API_KEY=your_gemini_api_key

# Backup Gemini Keys (optional but recommended)
GEMINI_API_KEY_1=your_second_gemini_key
GEMINI_API_KEY_2=your_third_gemini_key
GEMINI_API_KEY_3=your_fourth_gemini_key
GEMINI_API_KEY_4=your_fifth_gemini_key

# Other AI Providers
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
```

### Free API Keys Setup
- **Gemini**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Groq**: Get from [Groq Console](https://console.groq.com/keys)
- **OpenRouter**: Get from [OpenRouter](https://openrouter.ai/keys)
- **Perplexity**: Get from [Perplexity](https://www.perplexity.ai/settings/api)

## 🎯 Usage

### Assessment Flow
1. **👤 Registration** - Fill in candidate details (name, university, programming language, difficulty)
2. **📋 Rules** - Review and accept assessment guidelines
3. **🧠 Assessment** - Complete 20 questions with AI assistance
   - Real-time metrics sidebar
   - Corner camera monitoring
   - AI hints on demand
   - Performance tracking
4. **📊 Results** - View comprehensive analysis with AI insights

### Agentic Features During Assessment
- **📊 Live Metrics** - Score, streak, confidence, engagement
- **📷 Camera Toggle** - Enable/disable visual monitoring
- **💡 AI Hints** - Get contextual assistance
- **🎯 Performance Tracking** - Real-time behavior analysis

## 🔄 API Management System

### Automatic Quota Handling
The system automatically manages API quotas:

1. **Priority Order**: Groq → OpenRouter → Perplexity → Gemini
2. **Auto-Rotation**: Switches APIs when quota exceeded
3. **Key Rotation**: Multiple Gemini keys for redundancy
4. **Graceful Fallback**: Uses default questions if all APIs fail

### Quota Monitoring
```bash
# Check current API status
python check_quota_status.py

# Full API testing
python check_api_quota.py
```

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .env.example             # Environment variables template
├── README.md                # This documentation
├── check_api_quota.py       # API testing tool
├── check_quota_status.py    # Quick quota checker
├── data/                    # Data storage
│   ├── results/             # Assessment results
│   └── knowledge_base/     # Reference materials
├── integrated_assessment.py # Agentic assessment system
├── adaptive_agent.py        # AI question generation
├── camera_interface.py      # Camera integration
└── emotional_intelligence.py # Behavioral analysis
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
```bash
# Using Streamlit Cloud
streamlit run app.py --server.headless=true

# Using Docker
docker build -t nelumbus-assessment .
docker run -p 8501:8501 nelumbus-assessment
```

## 🔒 Security Features

- **Environment Variables**: Secure API key storage
- **Session Isolation**: User data privacy protection
- **Input Validation**: Prevents malicious input
- **Camera Control**: User consent and privacy controls
- **Error Handling**: No sensitive data exposure

## 📈 Performance Features

- **⚡ Fast Loading**: Optimized asset delivery
- **💾 Smart Caching**: Reduces API calls with hint caching
- **🔄 API Rotation**: Automatic load balancing
- **📱 Responsive Design**: Works on all devices
- **🛡️ Error Recovery**: Graceful degradation

## 🤖 Agentic Features Deep Dive

### User Context Tracking
```python
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
```

### Performance Metrics
- **📊 Real-time Score** - Live percentage calculation
- **🔥 Streak Counter** - Consecutive correct answers
- **⚡ Confidence Level** - AI-estimated user confidence
- **📈 Engagement Level** - User engagement metrics
- **🧠 Stress Indicators** - Behavioral stress detection
- **⏱️ Response Time** - Average time per question

### AI Recommendations
- **🎯 Performance-Based Feedback** - Tailored to score
- **📚 Weak Area Focus** - Topics needing improvement
- **💪 Strength Recognition** - Acknowledges strong areas
- **🎓 Next Steps** - Guidance for advanced topics

## 🔧 Troubleshooting

### Common Issues

#### "API quota exceeded" Message
**Solution**: 
1. Wait for quota reset (usually 1 hour for free tiers)
2. Add backup API keys to `.env` file
3. Use multiple API providers for redundancy

#### Camera Not Working
**Solution**:
1. Check if `streamlit-webrtc` is installed
2. Ensure browser permissions are granted
3. Use the toggle button to enable/disable

#### Questions Not Generating
**Solution**:
1. Run `python check_quota_status.py` to verify API status
2. Check if API keys are correctly configured
3. Verify internet connection

### API Status Commands
```bash
# Quick status check
python check_quota_status.py

# Full API testing
python check_api_quota.py

# Check specific API
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
     https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🆘 Support

For issues and questions:
- 🐛 Create an issue on GitHub
- 📧 Check the troubleshooting section
- 🔍 Run `python check_quota_status.py` first

## 🔄 Version History

- **v2.0.0**: Complete agentic assessment system
  - Added AI-powered user context tracking
  - Integrated corner camera monitoring
  - Implemented smart API quota management
  - Enhanced performance analytics
  - Added AI recommendations system

- **v1.3.0**: Improved error handling and fallback system
- **v1.2.0**: Enhanced UI/UX design
- **v1.1.0**: Added multi-LLM support
- **v1.0.0**: Initial release with basic assessment functionality

---

**Built with ❤️ for modern AI-powered assessment needs**

🌟 **Current Status**: Production Ready with Full Agentic Features
🔗 **Access**: http://localhost:8501
