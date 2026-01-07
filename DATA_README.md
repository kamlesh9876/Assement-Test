# 📊 Nelumbus Assessment Data Structure

This document explains how assessment data is organized and combined in the Nelumbus AI Assessment System.

## 📁 Data Organization

### Raw Data Folders
```
data/results/
├── candidates/     # Candidate registration and profile data
├── questions/      # Questions served in the assessment
├── responses/      # User responses and assessment results
├── scores/         # Scores, analytics, and performance metrics
├── combined/       # Combined data from all sources
└── proctoring/     # Proctoring and monitoring data
```

### Combined Data Folder
```
data/results/combined/
├── {attempt_id}.json           # Individual combined assessment data
├── processing_summary.json     # Summary of combination process
└── master_report.json          # Master report of all assessments
```

## 📋 Data Schema

### 1. Candidate Data (`candidates/{attempt_id}.json`)
```json
{
  "attempt_id": "uuid-string",
  "timestamp": "2024-01-07T12:00:00",
  "profile": {
    "name": "John Doe",
    "branch": "Computer Science",
    "passing_year": 2025,
    "university": "MIT University",
    "programming_language": "Python",
    "difficulty": "Medium",
    "assessment_type": "MCQ only"
  },
  "registration_data": {...},
  "test_config": {
    "total_questions": 20,
    "duration_seconds": 1200
  },
  "finalized_reason": "completed",
  "test_started_at": "timestamp",
  "test_completed_at": "timestamp"
}
```

### 2. Questions Data (`questions/{attempt_id}.json`)
```json
{
  "attempt_id": "uuid-string",
  "timestamp": "2024-01-07T12:00:00",
  "questions": [
    {
      "id": "q1",
      "question": "What is Python?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer_index": 0,
      "topic": "basics",
      "difficulty": "easy"
    }
  ],
  "total_questions": 20,
  "question_topics": ["basics", "algorithms", ...],
  "difficulty_distribution": {
    "easy": 8,
    "medium": 8,
    "hard": 4
  }
}
```

### 3. Responses Data (`responses/{attempt_id}.json`)
```json
{
  "attempt_id": "uuid-string",
  "timestamp": "2024-01-07T12:00:00",
  "responses": [
    {
      "question_id": "q1",
      "question_text": "What is Python?",
      "user_answer": "Option A",
      "correct_answer": "Option A",
      "is_correct": true,
      "response_time": 45.2,
      "confidence": 0.8
    }
  ],
  "answers": {"q1": 0, "q2": 2, ...},
  "total_answered": 20,
  "unanswered": []
}
```

### 4. Scores Data (`scores/{attempt_id}.json`)
```json
{
  "attempt_id": "uuid-string",
  "timestamp": "2024-01-07T12:00:00",
  "scores": {
    "total_questions": 20,
    "correct_answers": 15,
    "incorrect_answers": 5,
    "total_score": 15,
    "max_score": 20,
    "percentage": 75.0
  },
  "analytics": {
    "performance_level": "Good",
    "finalized_reason": "completed",
    "time_taken_seconds": 900,
    "average_time_per_question": 45.0
  },
  "performance_metrics": {
    "total_time_seconds": 900,
    "average_response_time": 45.0,
    "completion_rate": 100.0
  },
  "user_context": {
    "current_score": 0.75,
    "confidence_level": 0.8,
    "engagement_level": 0.9
  }
}
```

### 5. Combined Data (`combined/{attempt_id}.json`)
```json
{
  "metadata": {
    "combined_at": "2024-01-07T12:00:00",
    "version": "2.1.0",
    "attempt_id": "uuid-string",
    "data_sources": ["candidates", "questions", "responses", "scores"]
  },
  "candidate_info": {...},
  "assessment_config": {...},
  "questions": [...],
  "responses": [...],
  "scores": {...},
  "analytics": {...},
  "performance_metrics": {...},
  "summary": {
    "total_questions": 20,
    "correct_answers": 15,
    "percentage": 75.0,
    "performance_level": "Good",
    "total_time": 900
  }
}
```

## 🔄 Data Combination Process

### Manual Combination
```bash
# Run the combination script
python combine_data.py
```

### Automatic Combination
```bash
# Start auto-combination watcher
python auto_combine.py
```

### What the Scripts Do:

1. **combine_data.py**:
   - Scans all data folders for JSON files
   - Extracts attempt IDs from filenames
   - Combines data from all sources for each attempt
   - Creates unified JSON files in `combined/` folder
   - Generates master report with statistics

2. **auto_combine.py**:
   - Watches data folders for new files
   - Automatically runs combination when new data appears
   - Runs every 5 seconds continuously

## 📊 Master Report

The `master_report.json` contains aggregated data from all assessments:

```json
{
  "report_generated_at": "2024-01-07T12:00:00",
  "total_assessments": 50,
  "candidates": [
    {
      "attempt_id": "uuid",
      "name": "John Doe",
      "university": "MIT",
      "score": 15,
      "percentage": 75.0,
      "performance_level": "Good"
    }
  ],
  "performance_summary": {
    "excellent": 10,
    "good": 20,
    "average": 15,
    "below_average": 4,
    "poor": 1
  },
  "analytics": {
    "average_score": 72.5,
    "average_time": 850,
    "pass_rate": 80.0
  }
}
```

## 🔍 Usage Examples

### Access Combined Data
```python
import json
from pathlib import Path

# Load combined data for a specific attempt
with open("data/results/combined/{attempt_id}.json", "r") as f:
    data = json.load(f)

# Access candidate info
candidate_name = data["candidate_info"]["name"]
score = data["summary"]["percentage"]

# Load master report
with open("data/results/combined/master_report.json", "r") as f:
    master = json.load(f)

total_assessments = master["total_assessments"]
average_score = master["analytics"]["average_score"]
```

### Filter by Performance Level
```python
# Get all "Excellent" performers
excellent_candidates = [
    c for c in master["candidates"] 
    if c["performance_level"] == "Excellent"
]
```

## 🛠️ Data Maintenance

### Clean Old Data
```bash
# Remove data older than 30 days
find data/results -name "*.json" -mtime +30 -delete
```

### Backup Data
```bash
# Create backup
tar -czf backup_$(date +%Y%m%d).tar.gz data/results/
```

### Validate Data Integrity
```python
# Check for missing files
from combine_data import DataCombiner
combiner = DataCombiner()
combiner.process_all_attempts()  # Will report any issues
```

## 📈 Analytics Opportunities

With this structured data, you can analyze:
- Performance trends over time
- Difficulty level effectiveness
- Topic-wise performance
- Time management patterns
- University comparison
- Programming language preferences

## 🔒 Privacy Considerations

- All data is stored locally in JSON format
- No personal data is transmitted externally
- Attempt IDs are UUID-based for anonymity
- Camera data is handled separately in `proctoring/` folder
- Follow GDPR guidelines for data retention
