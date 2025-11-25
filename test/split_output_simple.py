#!/usr/bin/env python3
"""
Simple file splitter - split output file into two equal parts by line count.
Each record has ~12.5 lines on average, so we split by line count for simplicity.

Usage: python split_output_simple.py
"""

import os
from pathlib import Path

def split_output_file_simple():
    """Split the output file into two parts based on Values (TF-IDF weights) lines."""
    
    # Paths
    input_file = Path("data/results/lab17_pipeline_output.txt")
    output_part1 = Path("data/results/lab17_pipeline_output_part1.txt")
    output_part2 = Path("data/results/lab17_pipeline_output_part2.txt")
    
    # Check if input file exists
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"📂 Splitting file: {input_file}")
    print(f"📄 Part 1: {output_part1}")
    print(f"📄 Part 2: {output_part2}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        total_lines = len(lines)
        print(f"📊 Total lines in file: {total_lines}")
        
        # Find all "Values (TF-IDF weights)" lines - these mark the end of each record
        values_lines = []
        for i, line in enumerate(lines):
            if "Values (TF-IDF weights)" in line:
                values_lines.append(i)
        
        total_records = len(values_lines)
        print(f"📋 Total records found: {total_records}")
        
        if total_records < 15000:
            print(f"⚠️  Warning: Only {total_records} records found, less than 15000")
            split_at_record = total_records // 2
        else:
            split_at_record = 15000
        
        print(f"🔄 Splitting at record: {split_at_record}")
        
        # Find the line after the 15000th "Values" line (including its content and empty line)
        if split_at_record < len(values_lines):
            values_line_index = values_lines[split_at_record - 1]  # 15000th record's Values line
            # Find the end of this record (next empty line after the values)
            split_line_index = values_line_index + 1
            while split_line_index < len(lines) and lines[split_line_index].strip():
                split_line_index += 1
            # Skip the empty line to start Part 2 cleanly
            while split_line_index < len(lines) and not lines[split_line_index].strip():
                split_line_index += 1
        else:
            split_line_index = len(lines)
        
        print(f"📍 Split at line: {split_line_index}")
        
        # Write Part 1
        with open(output_part1, 'w', encoding='utf-8') as part1_file:
            part1_file.writelines(lines[:split_line_index])
        
        # Write Part 2  
        with open(output_part2, 'w', encoding='utf-8') as part2_file:
            part2_file.writelines(lines[split_line_index:])
        
        # Get file sizes
        part1_size = output_part1.stat().st_size / (1024 * 1024)  # MB
        part2_size = output_part2.stat().st_size / (1024 * 1024)  # MB
        
        print("\n✅ Split completed successfully!")
        print(f"📄 Part 1: Records 1-{split_at_record} ({split_at_record} records), {part1_size:.2f} MB")
        print(f"📄 Part 2: Records {split_at_record + 1}-{total_records} ({total_records - split_at_record} records), {part2_size:.2f} MB")
        
        # Verify split
        verify_split_simple(output_part1, output_part2)
        
    except Exception as e:
        print(f"❌ Error splitting file: {e}")

def verify_split_simple(part1_file, part2_file):
    """Verify the split by checking first lines and counting records."""
    try:
        # Check Part 1
        with open(part1_file, 'r', encoding='utf-8') as f:
            first_line_part1 = f.readline().strip()
            
        # Check Part 2
        with open(part2_file, 'r', encoding='utf-8') as f:
            first_line_part2 = f.readline().strip()
        
        print("\n🔍 Verification:")
        print(f"Part 1 starts with: {first_line_part1[:50]}...")
        print(f"Part 2 starts with: {first_line_part2[:50]}...")
        
        # Count records in each part
        count_part1 = count_records_simple(part1_file)
        count_part2 = count_records_simple(part2_file)
        
        print(f"📊 Part 1 records: {count_part1}")
        print(f"📊 Part 2 records: {count_part2}")
        print(f"📊 Total records: {count_part1 + count_part2}")
            
    except Exception as e:
        print(f"⚠️  Verification warning: {e}")

def count_records_simple(file_path):
    """Count records by counting 'Values (TF-IDF weights)' lines."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                if "Values (TF-IDF weights)" in line:
                    count += 1
            return count
    except:
        return 0

if __name__ == "__main__":
    print("🔧 Simple Pipeline Output File Splitter")
    print("=" * 45)
    split_output_file_simple()
    print("\n🎉 Done!")