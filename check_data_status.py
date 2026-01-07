#!/usr/bin/env python3
"""
Data Status Checker for Nelumbus Assessment System
Shows current status of all assessment data
"""

import json
from pathlib import Path

def check_data_status():
    """Check status of all assessment data"""
    print("📊 Nelumbus Assessment Data Status")
    print("=" * 50)
    
    results_dir = Path("data/results")
    
    # Check raw data folders
    folders = ['candidates', 'questions', 'responses', 'scores', 'combined']
    
    print("\n📁 Raw Data Folders:")
    for folder in folders:
        folder_path = results_dir / folder
        if folder_path.exists():
            files = list(folder_path.glob("*.json"))
            print(f"  {folder}/: {len(files)} files")
        else:
            print(f"  {folder}/: Not created")
    
    # Check combined data
    combined_path = results_dir / "combined"
    if combined_path.exists():
        combined_files = list(combined_path.glob("*.json"))
        
        print(f"\n📋 Combined Data: {len(combined_files)} files")
        
        # Check for master report
        master_file = combined_path / "master_report.json"
        if master_file.exists():
            with open(master_file, 'r') as f:
                master_data = json.load(f)
            
            print(f"\n📈 Master Report Summary:")
            print(f"  Total Assessments: {master_data.get('total_assessments', 0)}")
            print(f"  Average Score: {master_data.get('analytics', {}).get('average_score', 0):.1f}%")
            print(f"  Pass Rate: {master_data.get('analytics', {}).get('pass_rate', 0):.1f}%")
            
            # Performance distribution
            perf_summary = master_data.get('performance_summary', {})
            if perf_summary:
                print(f"\n🎯 Performance Distribution:")
                for level, count in perf_summary.items():
                    if count > 0:
                        print(f"  {level.title()}: {count}")
        
        # Show latest combined files
        print(f"\n📄 Latest Combined Files:")
        for file in sorted(combined_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            if file.name != 'master_report.json':
                print(f"  {file.name}")
    else:
        print(f"\n📋 Combined Data: Not created yet")
    
    print(f"\n" + "=" * 50)
    print("💡 Use 'python combine_data.py' to combine data")
    print("💡 Use 'python auto_combine.py' for automatic combination")

if __name__ == "__main__":
    check_data_status()
