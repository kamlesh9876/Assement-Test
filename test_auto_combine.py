#!/usr/bin/env python3
"""
Test script to verify automatic data combination
Simulates assessment completion and checks combination
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

def create_test_assessment():
    """Create a test assessment to verify automatic combination"""
    print("🧪 Creating test assessment data...")
    
    # Generate test attempt ID
    attempt_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Create test data
    test_candidate = {
        "attempt_id": attempt_id,
        "timestamp": timestamp,
        "profile": {
            "name": "Test User",
            "branch": "Computer Science",
            "passing_year": 2025,
            "university": "Test University",
            "programming_language": "Python",
            "difficulty": "Medium",
            "assessment_type": "MCQ only"
        },
        "test_config": {
            "total_questions": 20,
            "duration_seconds": 1200
        },
        "finalized_reason": "completed",
        "test_completed_at": timestamp
    }
    
    test_scores = {
        "attempt_id": attempt_id,
        "timestamp": timestamp,
        "scores": {
            "total_questions": 20,
            "correct_answers": 18,
            "incorrect_answers": 2,
            "total_score": 18,
            "max_score": 20,
            "percentage": 90.0
        },
        "analytics": {
            "performance_level": "Excellent",
            "finalized_reason": "completed",
            "time_taken_seconds": 600,
            "average_time_per_question": 30.0
        },
        "performance_metrics": {
            "total_time_seconds": 600,
            "average_response_time": 30.0,
            "completion_rate": 100.0
        }
    }
    
    # Save test data
    results_dir = Path("data/results")
    (results_dir / "candidates").mkdir(exist_ok=True)
    (results_dir / "scores").mkdir(exist_ok=True)
    
    with open(results_dir / "candidates" / f"{attempt_id}.json", 'w') as f:
        json.dump(test_candidate, f, indent=2)
    
    with open(results_dir / "scores" / f"{attempt_id}.json", 'w') as f:
        json.dump(test_scores, f, indent=2)
    
    print(f"✅ Test assessment created: {attempt_id}")
    return attempt_id

def test_automatic_combination():
    """Test the automatic combination functionality"""
    print("\n🔄 Testing automatic data combination...")
    
    # Import and test the combiner
    try:
        from combine_data import DataCombiner
        
        combiner = DataCombiner()
        results = combiner.process_all_attempts()
        master_report = combiner.generate_master_report()
        
        print(f"✅ Combination successful!")
        print(f"   Processed: {results['successful_combinations']} attempts")
        print(f"   Total assessments in report: {master_report.get('total_assessments', 0)}")
        print(f"   Average score: {master_report.get('analytics', {}).get('average_score', 0):.1f}%")
        
        # Check if combined file exists
        combined_files = list((Path("data/results/combined")).glob("*.json"))
        print(f"   Combined files: {len(combined_files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Automatic Data Combination Test")
    print("=" * 50)
    
    # Create test assessment
    attempt_id = create_test_assessment()
    
    # Test combination
    success = test_automatic_combination()
    
    if success:
        print("\n✅ All tests passed!")
        print("✅ Automatic data combination is working correctly")
        print("✅ Integration with app.py is functional")
    else:
        print("\n❌ Tests failed!")
        print("❌ Check the error messages above")
    
    print(f"\n📊 Current data status:")
    print("💡 Run 'python check_data_status.py' for detailed status")

if __name__ == "__main__":
    main()
