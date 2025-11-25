import time
from typing import Dict, List, Optional
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class TimingManager:
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize the detailed timing manager.
        
        Args:
            log_file: Optional path to save timing logs
        """
        self.timings: Dict[str, float] = {}
        self.stage_details: Dict[str, Dict[str, float]] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.log_file = log_file
        
    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.stage_details.clear()
        self.start_time = None
        self.end_time = None
    
    @contextmanager
    def time_stage(self, stage_name: str):
        """
        Context manager to time a stage.
        
        Args:
            stage_name: Name of the stage being timed
        """
        start_time = time.time()
        print(f"⏱️  Starting stage: {stage_name}")
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.timings[stage_name] = duration
            print(f"✅ Stage '{stage_name}' completed in {duration:.3f} seconds")
    
    @contextmanager
    def time_substage(self, stage_name: str, substage_name: str):
        """
        Context manager to time a substage within a main stage.
        
        Args:
            stage_name: Name of the main stage
            substage_name: Name of the substage
        """
        start_time = time.time()
        print(f"  🔄 Starting substage: {substage_name}")
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if stage_name not in self.stage_details:
                self.stage_details[stage_name] = {}
            self.stage_details[stage_name][substage_name] = duration
            print(f"  ✅ Substage '{substage_name}' completed in {duration:.3f} seconds")
    
    def start_total_timing(self):
        """Start timing the total pipeline execution."""
        self.start_time = time.time()
        print("🚀 Pipeline started!")
    
    def end_total_timing(self):
        """End timing the total pipeline execution."""
        self.end_time = time.time()
        if self.start_time:
            total_time = self.end_time - self.start_time
            self.timings['total_pipeline'] = total_time
            print(f"🏁 Pipeline completed in {total_time:.3f} seconds!")
    
    def get_total_time(self) -> float:
        """Get the total execution time."""
        return self.timings.get('total_pipeline', 0.0)
    
    def get_stage_time(self, stage_name: str) -> float:
        """Get the execution time for a specific stage."""
        return self.timings.get(stage_name, 0.0)
    
    def get_substage_time(self, stage_name: str, substage_name: str) -> float:
        """Get the execution time for a specific substage."""
        return self.stage_details.get(stage_name, {}).get(substage_name, 0.0)
    
    def get_detailed_summary(self) -> Dict[str, any]:
        """
        Get a detailed summary of all timings.
        
        Returns:
            Dict containing comprehensive timing information
        """
        total_time = self.get_total_time()
        
        summary = {
            'total_time': total_time,
            'stages': {},
            'stage_details': self.stage_details,
            'percentages': {}
        }
        
        # Calculate stage percentages
        for stage, duration in self.timings.items():
            if stage != 'total_pipeline' and total_time > 0:
                percentage = (duration / total_time) * 100
                summary['stages'][stage] = duration
                summary['percentages'][stage] = percentage
        
        return summary
    
    def print_detailed_summary(self):
        """Print a detailed summary of all timing measurements."""
        summary = self.get_detailed_summary()
        total_time = summary['total_time']
        
        print("\n" + "="*60)
        print("📊 DETAILED PIPELINE TIMING SUMMARY")
        print("="*60)
        
        # Main stages
        for stage, duration in summary['stages'].items():
            percentage = summary['percentages'][stage]
            print(f"  {stage:<20} : {duration:>8.3f}s ({percentage:>5.1f}%)")
            
            # Substages if available
            if stage in self.stage_details:
                for substage, sub_duration in self.stage_details[stage].items():
                    sub_percentage = (sub_duration / total_time) * 100 if total_time > 0 else 0
                    print(f"    └─ {substage:<16} : {sub_duration:>8.3f}s ({sub_percentage:>5.1f}%)")
        
        print("-" * 60)
        print(f"  {'TOTAL TIME':<20} : {total_time:>8.3f}s (100.0%)")
        print("="*60)
    
    def save_timing_log(self, additional_info: Optional[Dict] = None):
        """
        Save timing information to log file.
        
        Args:
            additional_info: Optional additional information to include
        """
        if not self.log_file:
            return
            
        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.get_detailed_summary()
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("DETAILED PIPELINE TIMING LOG\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            # Additional info
            if additional_info:
                f.write("📋 CONFIGURATION:\n")
                for key, value in additional_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Main stages
            f.write("📊 TIMING BREAKDOWN:\n")
            for stage, duration in summary['stages'].items():
                percentage = summary['percentages'][stage]
                f.write(f"  {stage:<20} : {duration:>8.3f}s ({percentage:>5.1f}%)\n")
                
                # Substages
                if stage in self.stage_details:
                    for substage, sub_duration in self.stage_details[stage].items():
                        sub_percentage = (sub_duration / summary['total_time']) * 100 if summary['total_time'] > 0 else 0
                        f.write(f"    └─ {substage:<16} : {sub_duration:>8.3f}s ({sub_percentage:>5.1f}%)\n")
            
            f.write("-" * 50 + "\n")
            f.write(f"  {'TOTAL TIME':<20} : {summary['total_time']:>8.3f}s (100.0%)\n")
            f.write("=" * 60 + "\n")
            
        print(f"📄 Detailed timing log saved to: {self.log_file}")

    def _log_message(self, message: str):
        """
        Log a message to both console and log file if available.
        
        Args:
            message: Message to log
        """
        print(message)
        
        if self.log_file:
            # Ensure directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {message}\n")


__ALL__ = ["TimingManager"]