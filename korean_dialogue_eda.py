#!/usr/bin/env python3
"""
Comprehensive EDA for Korean Dialogue Summarization Dataset
Analyzes dialogue structure, Korean text patterns, and preprocessing needs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from collections import Counter
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class KoreanDialogueEDA:
    """Comprehensive EDA for Korean dialogue data."""
    
    def __init__(self, data_path: str = "/home/wb2x/workspace/dialogue-summarizer/data"):
        self.data_path = Path(data_path)
        self.train_df = None
        self.dev_df = None
        self.test_df = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load all dataset splits."""
        print("🔄 Loading data files...")
        
        # Load train data
        train_file = self.data_path / "train.csv"
        if train_file.exists():
            self.train_df = pd.read_csv(train_file)
            print(f"✅ Train: {len(self.train_df)} samples")
        
        # Load dev data
        dev_file = self.data_path / "dev.csv"
        if dev_file.exists():
            self.dev_df = pd.read_csv(dev_file)
            print(f"✅ Dev: {len(self.dev_df)} samples")
            
        # Load test data
        test_file = self.data_path / "test.csv"
        if test_file.exists():
            self.test_df = pd.read_csv(test_file)
            print(f"✅ Test: {len(self.test_df)} samples")
    
    def analyze_basic_stats(self):
        """Analyze basic dataset statistics."""
        print("\n" + "="*60)
        print("📊 BASIC DATASET STATISTICS")
        print("="*60)
        
        stats = {}
        
        for name, df in [("train", self.train_df), ("dev", self.dev_df), ("test", self.test_df)]:
            if df is not None:
                print(f"\n{name.upper()} SET:")
                print(f"  Samples: {len(df):,}")
                print(f"  Columns: {list(df.columns)}")
                
                # Check for missing values
                missing = df.isnull().sum()
                if missing.any():
                    print(f"  Missing values: {missing[missing > 0].to_dict()}")
                
                stats[name] = {
                    'size': len(df),
                    'columns': list(df.columns),
                    'missing': missing.to_dict()
                }
        
        self.analysis_results['basic_stats'] = stats
    
    def analyze_dialogue_structure(self):
        """Analyze Korean dialogue structure and speaker patterns."""
        print("\n" + "="*60)
        print("🗣️ DIALOGUE STRUCTURE ANALYSIS")
        print("="*60)
        
        if self.train_df is None:
            return
            
        dialogues = self.train_df['dialogue'].dropna()
        
        # 1. Speaker Pattern Analysis
        print("\n1. SPEAKER PATTERNS:")
        speaker_patterns = []
        speaker_counts = Counter()
        
        for dialogue in dialogues:
            # Find all speaker markers
            speakers = re.findall(r'#Person\d+#', dialogue)
            unique_speakers = list(set(speakers))
            speaker_patterns.append(len(unique_speakers))
            speaker_counts.update(speakers)
        
        print(f"  Average speakers per dialogue: {np.mean(speaker_patterns):.2f}")
        print(f"  Speaker distribution: {Counter(speaker_patterns)}")
        print(f"  Most common speakers: {speaker_counts.most_common(10)}")
        
        # 2. Dialogue Length Analysis
        print("\n2. DIALOGUE LENGTH:")
        dialogue_chars = dialogues.str.len()
        dialogue_words = dialogues.str.split().str.len()
        
        print(f"  Characters - Mean: {dialogue_chars.mean():.0f}, Std: {dialogue_chars.std():.0f}")
        print(f"  Characters - Min: {dialogue_chars.min()}, Max: {dialogue_chars.max()}")
        print(f"  Words - Mean: {dialogue_words.mean():.0f}, Std: {dialogue_words.std():.0f}")
        print(f"  Words - Min: {dialogue_words.min()}, Max: {dialogue_words.max()}")
        
        # 3. Turn Analysis
        print("\n3. DIALOGUE TURNS:")
        turn_counts = []
        for dialogue in dialogues:
            # Count turns (speaker changes)
            turns = len(re.findall(r'#Person\d+#', dialogue))
            turn_counts.append(turns)
        
        print(f"  Average turns per dialogue: {np.mean(turn_counts):.2f}")
        print(f"  Turn distribution: {Counter(turn_counts)}")
        
        # Store results
        self.analysis_results['dialogue_structure'] = {
            'speaker_patterns': Counter(speaker_patterns),
            'speaker_counts': dict(speaker_counts.most_common(20)),
            'dialogue_stats': {
                'chars_mean': dialogue_chars.mean(),
                'chars_std': dialogue_chars.std(),
                'words_mean': dialogue_words.mean(),
                'words_std': dialogue_words.std()
            },
            'turn_stats': {
                'mean': np.mean(turn_counts),
                'distribution': dict(Counter(turn_counts))
            }
        }
    
    def analyze_summary_quality(self):
        """Analyze summary characteristics and quality."""
        print("\n" + "="*60)
        print("📝 SUMMARY ANALYSIS")
        print("="*60)
        
        if self.train_df is None or 'summary' not in self.train_df.columns:
            return
            
        summaries = self.train_df['summary'].dropna()
        
        # 1. Summary Length Analysis
        print("\n1. SUMMARY LENGTHS:")
        summary_chars = summaries.str.len()
        summary_words = summaries.str.split().str.len()
        
        print(f"  Characters - Mean: {summary_chars.mean():.0f}, Std: {summary_chars.std():.0f}")
        print(f"  Words - Mean: {summary_words.mean():.0f}, Std: {summary_words.std():.0f}")
        
        # 2. Compression Ratio
        dialogues = self.train_df['dialogue'].dropna()
        compression_ratios = summary_chars / dialogues.str.len()
        
        print(f"\n2. COMPRESSION RATIO:")
        print(f"  Mean compression: {compression_ratios.mean():.3f}")
        print(f"  Std compression: {compression_ratios.std():.3f}")
        
        # 3. Summary Content Analysis
        print("\n3. SUMMARY CONTENT:")
        
        # Check for Korean sentence endings
        korean_endings = ['다.', '요.', '습니다.', '니다.', '네요.', '어요.', '아요.']
        ending_counts = Counter()
        
        for summary in summaries:
            for ending in korean_endings:
                if summary.endswith(ending):
                    ending_counts[ending] += 1
                    break
        
        print(f"  Korean endings distribution: {dict(ending_counts)}")
        
        # Store results
        self.analysis_results['summary_analysis'] = {
            'length_stats': {
                'chars_mean': summary_chars.mean(),
                'chars_std': summary_chars.std(),
                'words_mean': summary_words.mean(),
                'words_std': summary_words.std()
            },
            'compression_ratio': {
                'mean': compression_ratios.mean(),
                'std': compression_ratios.std()
            },
            'korean_endings': dict(ending_counts)
        }
    
    def analyze_korean_text_patterns(self):
        """Analyze Korean-specific text patterns and preprocessing needs."""
        print("\n" + "="*60)
        print("🇰🇷 KOREAN TEXT PATTERN ANALYSIS")
        print("="*60)
        
        if self.train_df is None:
            return
            
        dialogues = self.train_df['dialogue'].dropna()
        
        # 1. Special Characters and Markers
        print("\n1. SPECIAL PATTERNS:")
        
        # Find all unique special markers
        all_markers = set()
        for dialogue in dialogues:
            markers = re.findall(r'#[^#]+#', dialogue)
            all_markers.update(markers)
        
        print(f"  Unique special markers found: {len(all_markers)}")
        marker_counts = Counter()
        for dialogue in dialogues:
            for marker in all_markers:
                if marker in dialogue:
                    marker_counts[marker] += 1
        
        print("  Top 10 markers:")
        for marker, count in marker_counts.most_common(10):
            print(f"    {marker}: {count}")
        
        # 2. Text Quality Issues
        print("\n2. TEXT QUALITY ISSUES:")
        
        issues = {
            'literal_newlines': 0,  # \\n instead of actual newlines
            'html_tags': 0,         # <br>, etc.
            'mixed_encoding': 0,    # encoding issues
            'extra_spaces': 0,      # multiple spaces
            'empty_lines': 0        # empty dialogues
        }
        
        for dialogue in dialogues:
            if '\\n' in dialogue:
                issues['literal_newlines'] += 1
            if re.search(r'<[^>]+>', dialogue):
                issues['html_tags'] += 1
            if re.search(r'  +', dialogue):  # multiple spaces
                issues['extra_spaces'] += 1
            if len(dialogue.strip()) == 0:
                issues['empty_lines'] += 1
        
        for issue, count in issues.items():
            if count > 0:
                print(f"  {issue}: {count} samples ({count/len(dialogues)*100:.1f}%)")
        
        # 3. Korean Language Patterns
        print("\n3. KOREAN LANGUAGE PATTERNS:")
        
        # Check for common Korean patterns
        korean_patterns = {
            'formal_endings': r'습니다|니다',
            'informal_endings': r'요\.|아요|어요',
            'question_markers': r'\?|까요',
            'korean_particles': r'는|은|을|를|이|가|에서|에게',
        }
        
        pattern_counts = {}
        for pattern_name, pattern in korean_patterns.items():
            count = sum(1 for dialogue in dialogues if re.search(pattern, dialogue))
            pattern_counts[pattern_name] = count
            print(f"  {pattern_name}: {count} samples ({count/len(dialogues)*100:.1f}%)")
        
        # Store results
        self.analysis_results['korean_patterns'] = {
            'special_markers': dict(marker_counts.most_common(20)),
            'text_issues': issues,
            'language_patterns': pattern_counts
        }
    
    def analyze_topic_distribution(self):
        """Analyze topic distribution if available."""
        print("\n" + "="*60)
        print("🏷️ TOPIC ANALYSIS")
        print("="*60)
        
        if self.train_df is None or 'topic' not in self.train_df.columns:
            print("No topic column found.")
            return
            
        topics = self.train_df['topic'].dropna()
        topic_counts = topics.value_counts()
        
        print(f"Total unique topics: {len(topic_counts)}")
        print("\nTop 10 topics:")
        for topic, count in topic_counts.head(10).items():
            print(f"  {topic}: {count} ({count/len(topics)*100:.1f}%)")
        
        # Analyze topic balance
        print(f"\nTopic distribution:")
        print(f"  Most common topic: {topic_counts.iloc[0]} samples")
        print(f"  Least common topic: {topic_counts.iloc[-1]} samples")
        print(f"  Balance ratio: {topic_counts.iloc[0]/topic_counts.iloc[-1]:.1f}:1")
        
        self.analysis_results['topic_analysis'] = {
            'total_topics': len(topic_counts),
            'topic_distribution': topic_counts.head(20).to_dict(),
            'balance_ratio': topic_counts.iloc[0]/topic_counts.iloc[-1]
        }
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n" + "="*60)
        print("📊 CREATING VISUALIZATIONS")
        print("="*60)
        
        if self.train_df is None:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Korean Dialogue Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Dialogue length distribution
        dialogue_words = self.train_df['dialogue'].str.split().str.len()
        axes[0, 0].hist(dialogue_words, bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Dialogue Length Distribution (Words)')
        axes[0, 0].set_xlabel('Number of Words')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Summary length distribution
        if 'summary' in self.train_df.columns:
            summary_words = self.train_df['summary'].str.split().str.len()
            axes[0, 1].hist(summary_words, bins=30, alpha=0.7, color='lightcoral')
            axes[0, 1].set_title('Summary Length Distribution (Words)')
            axes[0, 1].set_xlabel('Number of Words')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Compression ratio
        if 'summary' in self.train_df.columns:
            dialogue_chars = self.train_df['dialogue'].str.len()
            summary_chars = self.train_df['summary'].str.len()
            compression = summary_chars / dialogue_chars
            axes[0, 2].hist(compression, bins=30, alpha=0.7, color='lightgreen')
            axes[0, 2].set_title('Compression Ratio Distribution')
            axes[0, 2].set_xlabel('Summary/Dialogue Ratio')
            axes[0, 2].set_ylabel('Frequency')
        
        # 4. Speaker count distribution
        speaker_counts = []
        for dialogue in self.train_df['dialogue']:
            speakers = re.findall(r'#Person\d+#', dialogue)
            speaker_counts.append(len(set(speakers)))
        
        speaker_dist = Counter(speaker_counts)
        axes[1, 0].bar(speaker_dist.keys(), speaker_dist.values(), alpha=0.7, color='orange')
        axes[1, 0].set_title('Number of Speakers per Dialogue')
        axes[1, 0].set_xlabel('Number of Speakers')
        axes[1, 0].set_ylabel('Frequency')
        
        # 5. Topic distribution (if available)
        if 'topic' in self.train_df.columns:
            topic_counts = self.train_df['topic'].value_counts().head(10)
            axes[1, 1].barh(range(len(topic_counts)), topic_counts.values, alpha=0.7, color='purple')
            axes[1, 1].set_yticks(range(len(topic_counts)))
            axes[1, 1].set_yticklabels(topic_counts.index, fontsize=8)
            axes[1, 1].set_title('Top 10 Topics')
            axes[1, 1].set_xlabel('Frequency')
        
        # 6. Length correlation (dialogue vs summary)
        if 'summary' in self.train_df.columns:
            dialogue_len = self.train_df['dialogue'].str.len()
            summary_len = self.train_df['summary'].str.len()
            axes[1, 2].scatter(dialogue_len, summary_len, alpha=0.5, s=1)
            axes[1, 2].set_title('Dialogue vs Summary Length')
            axes[1, 2].set_xlabel('Dialogue Length (chars)')
            axes[1, 2].set_ylabel('Summary Length (chars)')
        
        plt.tight_layout()
        plt.savefig('korean_dialogue_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_sample_analysis(self, n_samples=5):
        """Analyze specific samples in detail."""
        print("\n" + "="*60)
        print("🔍 DETAILED SAMPLE ANALYSIS")
        print("="*60)
        
        if self.train_df is None:
            return
            
        # Select diverse samples
        samples = self.train_df.sample(n_samples)
        
        for i, (_, row) in enumerate(samples.iterrows()):
            print(f"\n--- SAMPLE {i+1} ---")
            print(f"ID: {row['fname']}")
            
            if 'topic' in row:
                print(f"Topic: {row['topic']}")
            
            dialogue = row['dialogue']
            print(f"Dialogue ({len(dialogue)} chars, {len(dialogue.split())} words):")
            print(f"  {dialogue[:200]}...")
            
            # Extract speakers
            speakers = re.findall(r'#Person\d+#', dialogue)
            print(f"  Speakers: {list(set(speakers))}")
            
            if 'summary' in row:
                summary = row['summary']
                print(f"Summary ({len(summary)} chars, {len(summary.split())} words):")
                print(f"  {summary}")
                
                # Compression ratio
                compression = len(summary) / len(dialogue)
                print(f"  Compression ratio: {compression:.3f}")
    
    def identify_preprocessing_needs(self):
        """Identify specific preprocessing needs based on analysis."""
        print("\n" + "="*60)
        print("🔧 PREPROCESSING RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        # Based on analysis results
        if 'korean_patterns' in self.analysis_results:
            issues = self.analysis_results['korean_patterns']['text_issues']
            
            if issues['literal_newlines'] > 0:
                recommendations.append(f"🔄 Fix {issues['literal_newlines']} samples with literal \\n characters")
            
            if issues['html_tags'] > 0:
                recommendations.append(f"🔄 Remove HTML tags from {issues['html_tags']} samples")
            
            if issues['extra_spaces'] > 0:
                recommendations.append(f"🔄 Normalize whitespace in {issues['extra_spaces']} samples")
        
        # Length-based recommendations
        if 'dialogue_structure' in self.analysis_results:
            stats = self.analysis_results['dialogue_structure']['dialogue_stats']
            if stats['words_mean'] > 400:
                recommendations.append("✂️ Consider truncation strategy for long dialogues")
            
        # Korean-specific recommendations
        recommendations.extend([
            "🇰🇷 Preserve Korean sentence endings and particles",
            "🏷️ Keep speaker markers (#Person1#, etc.) for dialogue structure",
            "📝 Normalize Korean punctuation and spacing",
            "🔄 Handle informal speech patterns appropriately"
        ])
        
        print("KEY RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  {rec}")
        
        return recommendations
    
    def save_analysis_report(self, output_path="eda_analysis_report.json"):
        """Save complete analysis to JSON file."""
        print(f"\n💾 Saving analysis report to {output_path}")
        
        # Add recommendations to results
        self.analysis_results['preprocessing_recommendations'] = self.identify_preprocessing_needs()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        print("✅ Analysis report saved!")
    
    def run_complete_analysis(self):
        """Run all analysis steps."""
        print("🚀 Starting Comprehensive Korean Dialogue EDA")
        print("="*60)
        
        self.load_data()
        self.analyze_basic_stats()
        self.analyze_dialogue_structure()
        self.analyze_summary_quality()
        self.analyze_korean_text_patterns()
        self.analyze_topic_distribution()
        self.generate_sample_analysis()
        self.create_visualizations()
        self.save_analysis_report()
        
        print("\n🎉 EDA COMPLETE!")
        print("="*60)
        print("📊 Visualization saved as: korean_dialogue_eda.png")
        print("📄 Full report saved as: eda_analysis_report.json")

if __name__ == "__main__":
    # Run the analysis
    eda = KoreanDialogueEDA()
    eda.run_complete_analysis()