# InfoNCE
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AHV-IV Triplet Generator for final_testData.csv
Data format: questions, content, language, answer
Contains German(30), French(65), Trilingual(5) - total 100 records
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple, Union
from sentence_transformers import InputExample
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds
random.seed(42)
np.random.seed(42)

class AHVIVTripletGenerator:
    """Specialized triplet generator for AHV-IV data"""
    
    def __init__(self, csv_path: str = "final_testData.csv"):
        """Initialize generator"""
        self.csv_path = csv_path
        self.df = self.load_data()
        self.data_analysis()
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV"""
        print(f"Loading data: {self.csv_path}")
        
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            print(f"Data loaded successfully")
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Loading failed: {e}")
            raise
    
    def data_analysis(self):
        """Analyze data quality and distribution"""
        print("\n=== Data Analysis ===")
        
        # Language distribution
        language_counts = self.df['language'].value_counts()
        print("Language distribution:")
        for lang, count in language_counts.items():
            print(f"  {lang}: {count} records ({count/len(self.df)*100:.1f}%)")
        
        # Data quality check
        print("\nData quality:")
        for col in ['questions', 'content', 'answer']:
            valid_count = self.df[col].notna().sum()
            print(f"  {col}: {valid_count}/{len(self.df)} ({valid_count/len(self.df)*100:.1f}%)")
        
        # Length statistics
        print("\nText length statistics:")
        for col in ['questions', 'content', 'answer']:
            lengths = self.df[col].dropna().str.len()
            if len(lengths) > 0:
                print(f"  {col}: avg={lengths.mean():.0f}, min={lengths.min()}, max={lengths.max()}")
    
    def create_language_specific_negatives(self, target_lang: str, num_negatives: int = 3) -> Dict:
        """Create random negatives for specific language"""
        print(f"Creating random negatives for {target_lang}...")
        
        # Get all content for the same language
        same_lang_data = self.df[self.df['language'] == target_lang].copy()
        if len(same_lang_data) == 0:
            print(f"No data found for {target_lang}")
            return {}
        
        contents = same_lang_data['content'].tolist()
        hard_negatives = {}
        
        # Randomly select incorrect content from same language
        if len(contents) > 1:
            for idx, row in same_lang_data.iterrows():
                query_key = f"query_{idx}"
                
                # Get all other contents from same language (excluding current one)
                other_contents = [c for c in contents if c != row['content']]
                
                # Randomly select negatives
                if len(other_contents) >= num_negatives:
                    selected_negatives = random.sample(other_contents, num_negatives)
                else:
                    selected_negatives = other_contents
                
                hard_negatives[query_key] = selected_negatives
        
        print(f"Generated random negatives for {len(hard_negatives)} queries")
        return hard_negatives
    
    def create_cross_language_negatives(self, num_negatives: int = 2) -> Dict:
        """Create cross-language negatives for increased difficulty"""
        print("Creating cross-language negatives...")
        
        cross_lang_negatives = {}
        all_contents = self.df['content'].tolist()
        
        for idx, row in self.df.iterrows():
            query_key = f"query_{idx}"
            current_lang = row['language']
            current_content = row['content']
            
            # Select content from other languages as negatives
            other_lang_contents = [
                content for i, content in enumerate(all_contents) 
                if i != idx and self.df.iloc[i]['language'] != current_lang
            ]
            
            if other_lang_contents:
                selected = random.sample(
                    other_lang_contents, 
                    min(num_negatives, len(other_lang_contents))
                )
                cross_lang_negatives[query_key] = selected
            else:
                cross_lang_negatives[query_key] = []
        
        return cross_lang_negatives
    
    def generate_triplets(self, 
                         same_lang_negatives: int = 3,
                         cross_lang_negatives: int = 1,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15) -> Tuple[List, List, List]:
        """Generate complete triplet dataset"""
        print("\n=== Generating Triplet Dataset ===")
        
        all_examples = []
        stats = {'positive': 0, 'same_lang_negative': 0, 'cross_lang_negative': 0}
        
        # Generate same-language negatives for each language
        all_same_lang_negatives = {}
        for lang in self.df['language'].unique():
            lang_negatives = self.create_language_specific_negatives(lang, same_lang_negatives)
            all_same_lang_negatives.update(lang_negatives)
        
        # Generate cross-language negatives
        cross_negatives = self.create_cross_language_negatives(cross_lang_negatives)
        
        # Create all examples
        print("Generating training examples...")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Creating examples"):
            query = row['questions']
            content = row['content']
            language = row['language']
            
            if pd.isna(query) or pd.isna(content):
                continue
            
            query_key = f"query_{idx}"
            
            # 1. Positive examples
            positive_example = InputExample(
                texts=[query, content],
                label=1.0
            )
            all_examples.append(positive_example)
            stats['positive'] += 1
            
            # 2. Same-language negatives
            if query_key in all_same_lang_negatives:
                for neg_content in all_same_lang_negatives[query_key]:
                    negative_example = InputExample(
                        texts=[query, neg_content],
                        label=0.0
                    )
                    all_examples.append(negative_example)
                    stats['same_lang_negative'] += 1
            
            # 3. Cross-language negatives
            if query_key in cross_negatives:
                for neg_content in cross_negatives[query_key]:
                    negative_example = InputExample(
                        texts=[query, neg_content],
                        label=0.0
                    )
                    all_examples.append(negative_example)
                    stats['cross_lang_negative'] += 1
        
        # Statistics
        total_examples = len(all_examples)
        print(f"\nGeneration statistics:")
        print(f"  Positive samples: {stats['positive']}")
        print(f"  Same-language negatives: {stats['same_lang_negative']}")
        print(f"  Cross-language negatives: {stats['cross_lang_negative']}")
        print(f"  Total samples: {total_examples}")
        print(f"  Positive:Negative ratio: 1:{(stats['same_lang_negative'] + stats['cross_lang_negative'])/stats['positive']:.1f}")
        
        # Shuffle and split data
        random.shuffle(all_examples)
        
        train_size = int(total_examples * train_ratio)
        val_size = int(total_examples * val_ratio)
        
        train_examples = all_examples[:train_size]
        val_examples = all_examples[train_size:train_size + val_size]
        test_examples = all_examples[train_size + val_size:]
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_examples)} ({len(train_examples)/total_examples:.1%})")
        print(f"  Validation: {len(val_examples)} ({len(val_examples)/total_examples:.1%})")
        print(f"  Test: {len(test_examples)} ({len(test_examples)/total_examples:.1%})")
        
        return train_examples, val_examples, test_examples
    
    def create_evaluation_corpus(self) -> Tuple[Dict, Dict, Dict]:
        """Create evaluation corpus for information retrieval"""
        print("\n=== Creating Evaluation Corpus ===")
        
        corpus = {}
        queries = {}
        qrels = {}
        
        for idx, row in self.df.iterrows():
            if pd.isna(row['questions']) or pd.isna(row['content']):
                continue
            
            doc_id = f"doc_{idx}"
            query_id = f"query_{idx}"
            
            corpus[doc_id] = row['content']
            queries[query_id] = row['questions']
            qrels[query_id] = {doc_id: 1}
        
        print(f"Evaluation corpus statistics:")
        print(f"  Documents: {len(corpus)}")
        print(f"  Queries: {len(queries)}")
        print(f"  Relevance annotations: {len(qrels)}")
        
        # Language distribution statistics
        lang_stats = {}
        for idx, row in self.df.iterrows():
            lang = row['language']
            if lang not in lang_stats:
                lang_stats[lang] = 0
            lang_stats[lang] += 1
        
        print(f"  Language distribution: {lang_stats}")
        
        return corpus, queries, qrels
    
    def save_data(self, 
                  train_examples: List, 
                  val_examples: List, 
                  test_examples: List,
                  corpus: Dict, 
                  queries: Dict, 
                  qrels: Dict,
                  output_dir: str = "./ahv_iv_triplets"):
        """Save all data to files"""
        print(f"\n=== Saving Data to {output_dir} ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        def save_examples(examples: List, filename: str):
            """Save example data"""
            data = []
            for example in examples:
                data.append({
                    'query': example.texts[0],
                    'document': example.texts[1],
                    'label': example.label
                })
            
            # Save as JSON
            json_path = os.path.join(output_dir, f"{filename}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Save as CSV
            csv_path = os.path.join(output_dir, f"{filename}.csv")
            pd.DataFrame(data).to_csv(csv_path, index=False, encoding='utf-8')
            
            print(f"  {filename}: {len(examples)} samples")
            return len(examples)
        
        def save_infonce_format(examples: List, filename: str):
            """Save data in InfoNCE triplet format"""
            # Group by query
            query_groups = {}
            for example in examples:
                query = example.texts[0]
                if query not in query_groups:
                    query_groups[query] = {'positive': None, 'negatives': []}
                
                if example.label == 1.0:
                    query_groups[query]['positive'] = example.texts[1]
                else:
                    query_groups[query]['negatives'].append(example.texts[1])
            
            # Convert to InfoNCE format
            infonce_data = []
            for query, group in query_groups.items():
                if group['positive'] is not None and len(group['negatives']) > 0:
                    infonce_data.append({
                        'query': query,
                        'positive': group['positive'],
                        'negatives': group['negatives']
                    })
            
            # Save InfoNCE format
            infonce_path = os.path.join(output_dir, f"{filename}_infonce.json")
            with open(infonce_path, 'w', encoding='utf-8') as f:
                json.dump(infonce_data, f, ensure_ascii=False, indent=2)
            
            print(f"  {filename}_infonce: {len(infonce_data)} triplets")
            return len(infonce_data)
        
        # Save standard format
        save_examples(train_examples, "train")
        save_examples(val_examples, "val")
        save_examples(test_examples, "test")
        
        # Save InfoNCE format
        save_infonce_format(train_examples, "train")
        save_infonce_format(val_examples, "val")
        save_infonce_format(test_examples, "test")
        
        # Save evaluation corpus
        eval_data = {
            'corpus': corpus,
            'queries': queries,
            'qrels': qrels
        }
        
        eval_path = os.path.join(output_dir, "evaluation_corpus.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        print(f"  Evaluation corpus: {len(corpus)} documents, {len(queries)} queries")
        
        # Save configuration
        config = {
            'source_file': self.csv_path,
            'total_records': len(self.df),
            'language_distribution': self.df['language'].value_counts().to_dict(),
            'train_size': len(train_examples),
            'val_size': len(val_examples),
            'test_size': len(test_examples),
            'corpus_size': len(corpus)
        }
        
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"  Config file: config.json")
        
        print(f"\nAll data saved to {output_dir}")
        print("Available formats:")
        print("  - Standard format: train.json, val.json, test.json")
        print("  - InfoNCE format: train_infonce.json, val_infonce.json, test_infonce.json")
        return output_dir
    
    def visualize_data(self, output_dir: str = "./ahv_iv_triplets"):
        """Generate data visualization"""
        print("\n=== Generating Data Visualization ===")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Language distribution
        lang_counts = self.df['language'].value_counts()
        ax1.pie(lang_counts.values, labels=lang_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Language Distribution')
        
        # 2. Question length distribution
        question_lengths = self.df['questions'].dropna().str.len()
        ax2.hist(question_lengths, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Question Length Distribution')
        ax2.set_xlabel('Characters')
        ax2.set_ylabel('Frequency')
        
        # 3. Content length distribution
        content_lengths = self.df['content'].dropna().str.len()
        ax3.hist(content_lengths, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_title('Content Length Distribution')
        ax3.set_xlabel('Characters')
        ax3.set_ylabel('Frequency')
        
        # 4. Answer length distribution
        answer_lengths = self.df['answer'].dropna().str.len()
        ax4.hist(answer_lengths, bins=20, alpha=0.7, edgecolor='black')
        ax4.set_title('Answer Length Distribution')
        ax4.set_xlabel('Characters')
        ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "data_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved: {plot_path}")

def main():
    """Main function: run complete triplet generation workflow"""
    print("AHV-IV Triplet Generator")
    print("=" * 50)
    
    try:
        # 1. Initialize generator
        generator = AHVIVTripletGenerator("final_testData.csv")
        
        # 2. Generate triplets
        train_examples, val_examples, test_examples = generator.generate_triplets(
            same_lang_negatives=3,  # 3 same-language negatives per query
            cross_lang_negatives=1, # 1 cross-language negative per query
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        # 3. Create evaluation corpus
        corpus, queries, qrels = generator.create_evaluation_corpus()
        
        # 4. Save all data
        output_dir = generator.save_data(
            train_examples, val_examples, test_examples,
            corpus, queries, qrels
        )
        
        # 5. Generate visualization
        generator.visualize_data(output_dir)
        
        # 6. Show examples
        print("\n=== Sample Data ===")
        print("Training examples (first 3):")
        for i, example in enumerate(train_examples[:3]):
            print(f"\nExample {i+1}:")
            print(f"  Query: {example.texts[0][:60]}...")
            print(f"  Document: {example.texts[1][:60]}...")
            print(f"  Label: {'Positive' if example.label == 1.0 else 'Negative'}")
        
        print(f"\nCompleted! Data ready for training")
        print(f"Output directory: {output_dir}")
        print("\nNext step: Use this data to train Qwen3-Embedding-0.6B model")
        
        return generator, train_examples, val_examples, test_examples, corpus, queries, qrels
        
    except FileNotFoundError:
        print("Error: final_testData.csv file not found")
        print("Please ensure the file is in the current directory")
        return None
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_test():
    """Quick test function to check data loading"""
    print("Testing data loading...")
    
    try:
        df = pd.read_csv("final_testData.csv")
        print(f"File loaded successfully: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Language distribution: {df['language'].value_counts().to_dict()}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run quick test first
    if quick_test():
        print("\n" + "="*50)
        # Run main program
        result = main()
        
        if result:
            print("\n" + "="*50)
            print("AHV-IV Triplet Generation Completed!")
            print("Now you can use the generated data for model fine-tuning.")
    else:
        print("Please check if final_testData.csv exists and has correct format")
