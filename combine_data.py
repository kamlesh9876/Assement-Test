#!/usr/bin/env python3
"""
Data Combination Script for Nelumbus Assessment System
Combines candidate data and scores into unified JSON files
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class DataCombiner:
    """Combines assessment data from multiple sources into unified JSON files"""
    
    def __init__(self, base_path: str = "data/results"):
        self.base_path = Path(base_path)
        self.candidates_path = self.base_path / "candidates"
        self.scores_path = self.base_path / "scores"
        self.questions_path = self.base_path / "questions"
        self.combined_path = self.base_path / "combined"
        
        # Ensure directories exist
        self.combined_path.mkdir(exist_ok=True)
        
    def load_json_file(self, file_path: Path) -> Optional[Dict]:
        """Load JSON file safely"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        return None
    
    def save_json_file(self, data: Dict, file_path: Path) -> bool:
        """Save JSON file safely"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
            return False
    
    def get_all_files(self, directory: Path, extension: str = ".json") -> List[Path]:
        """Get all files with specified extension from directory"""
        if not directory.exists():
            return []
        return list(directory.glob(f"*{extension}"))
    
    def extract_attempt_id(self, filename: str) -> Optional[str]:
        """Extract attempt ID from filename"""
        # Handle various filename formats
        if "-" in filename:
            parts = filename.replace(".json", "").split("-")
            if len(parts) >= 5:
                return "-".join(parts[-5:])  # Get last 5 parts as UUID
        return filename.replace(".json", "")
    
    def combine_candidate_data(self) -> Dict[str, Any]:
        """Combine all assessment data for a single attempt"""
        combined_data = {
            "metadata": {
                "combined_at": datetime.now().isoformat(),
                "version": "2.1.0",
                "data_sources": []
            },
            "candidate_info": {},
            "assessment_config": {},
            "questions": [],
            "responses": [],
            "scores": {},
            "analytics": {},
            "performance_metrics": {}
        }
        
        return combined_data
    
    def process_single_attempt(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        """Process a single assessment attempt and combine all data"""
        print(f"Processing attempt: {attempt_id}")
        
        combined_data = self.combine_candidate_data()
        
        # Load candidate data
        candidate_file = self.candidates_path / f"{attempt_id}.json"
        candidate_data = self.load_json_file(candidate_file)
        if candidate_data:
            combined_data["candidate_info"] = candidate_data.get("profile", {})
            combined_data["assessment_config"] = candidate_data.get("test_config", {})
            combined_data["metadata"]["data_sources"].append("candidates")
        else:
            print(f"  No candidate data found for {attempt_id}")
        
        # Load questions data
        questions_file = self.questions_path / f"{attempt_id}.json"
        questions_data = self.load_json_file(questions_file)
        if questions_data:
            combined_data["questions"] = questions_data.get("questions", [])
            combined_data["metadata"]["data_sources"].append("questions")
        else:
            print(f"  No questions data found for {attempt_id}")
        
        # Load scores data
        scores_file = self.scores_path / f"{attempt_id}.json"
        scores_data = self.load_json_file(scores_file)
        if scores_data:
            combined_data["scores"] = scores_data.get("scores", {})
            combined_data["analytics"] = scores_data.get("analytics", {})
            combined_data["performance_metrics"] = scores_data.get("performance_metrics", {})
            combined_data["metadata"]["data_sources"].append("scores")
        else:
            print(f"  No scores data found for {attempt_id}")
        
        # Add attempt ID to metadata
        combined_data["metadata"]["attempt_id"] = attempt_id
        
        # Calculate summary statistics if data is available
        if combined_data["questions"] and combined_data["scores"]:
            combined_data["summary"] = self.calculate_summary(combined_data)
        
        return combined_data if combined_data["metadata"]["data_sources"] else None
    
    def calculate_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from combined data"""
        questions = data.get("questions", [])
        scores = data.get("scores", {})
        
        total_questions = len(questions)
        correct_answers = scores.get("correct_answers", 0)
        total_score = scores.get("total_score", 0)
        max_score = scores.get("max_score", total_questions)
        
        summary = {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "total_score": total_score,
            "max_score": max_score,
            "percentage": (total_score / max_score * 100) if max_score > 0 else 0,
            "performance_level": self.get_performance_level(total_score, max_score)
        }
        
        # Add time metrics if available
        if "performance_metrics" in data:
            metrics = data["performance_metrics"]
            summary.update({
                "total_time": metrics.get("total_time_seconds", 0),
                "average_response_time": metrics.get("average_response_time", 0),
                "time_per_question": metrics.get("time_per_question", 0)
            })
        
        return summary
    
    def get_performance_level(self, score: float, max_score: float) -> str:
        """Determine performance level based on score"""
        percentage = (score / max_score * 100) if max_score > 0 else 0
        
        if percentage >= 90:
            return "Excellent"
        elif percentage >= 80:
            return "Good"
        elif percentage >= 70:
            return "Average"
        elif percentage >= 60:
            return "Below Average"
        else:
            return "Poor"
    
    def find_all_attempt_ids(self) -> List[str]:
        """Find all unique attempt IDs from all data sources"""
        attempt_ids = set()
        
        # Check candidates folder
        for file_path in self.get_all_files(self.candidates_path):
            attempt_id = self.extract_attempt_id(file_path.name)
            attempt_ids.add(attempt_id)
        
        # Check scores folder
        for file_path in self.get_all_files(self.scores_path):
            attempt_id = self.extract_attempt_id(file_path.name)
            attempt_ids.add(attempt_id)
        
        # Check questions folder
        for file_path in self.get_all_files(self.questions_path):
            attempt_id = self.extract_attempt_id(file_path.name)
            attempt_ids.add(attempt_id)
        
        return sorted(list(attempt_ids))
    
    def process_all_attempts(self) -> Dict[str, Any]:
        """Process all assessment attempts and combine data"""
        print("Starting data combination process...")
        
        attempt_ids = self.find_all_attempt_ids()
        print(f"Found {len(attempt_ids)} unique attempts to process")
        
        results = {
            "processed_at": datetime.now().isoformat(),
            "total_attempts": len(attempt_ids),
            "successful_combinations": 0,
            "failed_combinations": 0,
            "attempts": {}
        }
        
        for attempt_id in attempt_ids:
            combined_data = self.process_single_attempt(attempt_id)
            
            if combined_data:
                # Save combined data
                combined_file = self.combined_path / f"{attempt_id}.json"
                if self.save_json_file(combined_data, combined_file):
                    results["attempts"][attempt_id] = {
                        "status": "success",
                        "file": str(combined_file),
                        "data_sources": combined_data["metadata"]["data_sources"],
                        "summary": combined_data.get("summary", {})
                    }
                    results["successful_combinations"] += 1
                    print(f"  ✓ Successfully combined {attempt_id}")
                else:
                    results["attempts"][attempt_id] = {
                        "status": "save_error",
                        "error": "Failed to save combined file"
                    }
                    results["failed_combinations"] += 1
                    print(f"  ✗ Failed to save {attempt_id}")
            else:
                results["attempts"][attempt_id] = {
                    "status": "no_data",
                    "error": "No data found for this attempt"
                }
                results["failed_combinations"] += 1
                print(f"  ✗ No data found for {attempt_id}")
        
        # Save processing summary
        summary_file = self.combined_path / "processing_summary.json"
        self.save_json_file(results, summary_file)
        
        print(f"\nProcessing complete!")
        print(f"Successful: {results['successful_combinations']}")
        print(f"Failed: {results['failed_combinations']}")
        print(f"Combined files saved to: {self.combined_path}")
        
        return results
    
    def generate_master_report(self) -> Dict[str, Any]:
        """Generate a master report of all combined data"""
        print("Generating master report...")
        
        combined_files = self.get_all_files(self.combined_path)
        master_data = {
            "report_generated_at": datetime.now().isoformat(),
            "total_assessments": 0,
            "candidates": [],
            "performance_summary": {
                "excellent": 0,
                "good": 0,
                "average": 0,
                "below_average": 0,
                "poor": 0
            },
            "analytics": {
                "average_score": 0,
                "average_time": 0,
                "pass_rate": 0
            }
        }
        
        all_scores = []
        all_times = []
        
        for file_path in combined_files:
            if file_path.name == "processing_summary.json":
                continue
                
            data = self.load_json_file(file_path)
            if not data:
                continue
            
            master_data["total_assessments"] += 1
            
            # Extract candidate info
            candidate_info = data.get("candidate_info", {})
            summary = data.get("summary", {})
            
            candidate_entry = {
                "attempt_id": data.get("metadata", {}).get("attempt_id", "unknown"),
                "name": candidate_info.get("name", "Unknown"),
                "university": candidate_info.get("university", "Unknown"),
                "programming_language": candidate_info.get("programming_language", "Unknown"),
                "difficulty": candidate_info.get("difficulty", "Unknown"),
                "score": summary.get("total_score", 0),
                "percentage": summary.get("percentage", 0),
                "performance_level": summary.get("performance_level", "Unknown"),
                "total_time": summary.get("total_time", 0),
                "completed_at": data.get("metadata", {}).get("combined_at", "Unknown")
            }
            
            master_data["candidates"].append(candidate_entry)
            
            # Update performance counts
            level = summary.get("performance_level", "unknown").lower()
            if level in master_data["performance_summary"]:
                master_data["performance_summary"][level] += 1
            
            # Collect scores and times for analytics
            if summary.get("percentage", 0) > 0:
                all_scores.append(summary["percentage"])
            if summary.get("total_time", 0) > 0:
                all_times.append(summary["total_time"])
        
        # Calculate analytics
        if all_scores:
            master_data["analytics"]["average_score"] = sum(all_scores) / len(all_scores)
            master_data["analytics"]["pass_rate"] = len([s for s in all_scores if s >= 60]) / len(all_scores) * 100
        
        if all_times:
            master_data["analytics"]["average_time"] = sum(all_times) / len(all_times)
        
        # Save master report
        master_file = self.combined_path / "master_report.json"
        self.save_json_file(master_data, master_file)
        
        print(f"Master report saved to: {master_file}")
        return master_data


def main():
    """Main function to run the data combination process"""
    print("=" * 60)
    print("Nelumbus Assessment System - Data Combination Script")
    print("=" * 60)
    
    # Initialize the data combiner
    combiner = DataCombiner()
    
    # Process all attempts
    results = combiner.process_all_attempts()
    
    # Generate master report
    master_report = combiner.generate_master_report()
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total attempts processed: {results['total_attempts']}")
    print(f"Successful combinations: {results['successful_combinations']}")
    print(f"Failed combinations: {results['failed_combinations']}")
    print(f"Combined data location: {combiner.combined_path}")
    print(f"Master report generated: {len(master_report['candidates'])} candidates")
    
    if results['failed_combinations'] > 0:
        print(f"\n⚠️  {results['failed_combinations']} attempts failed to combine.")
        print("Check processing_summary.json for details.")


if __name__ == "__main__":
    main()
