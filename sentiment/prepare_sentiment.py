"""
Prepare Sentiment Analysis Dataset
Download and prepare Allocine (FR) + Italian sentiment data
Binary sentiment: positive (1) vs negative (0)
"""
import os
import pandas as pd
from datasets import load_dataset
from fr_ita.utils import save_embeddings, preprocess_text
from fr_ita.config import DATA_DIR


SENTIMENT_DIR = os.path.join(DATA_DIR, 'sentiment')


def download_allocine():
    """
    Download Allocine French movie reviews dataset.
    """
    print("Downloading Allocine FR dataset...")
    dataset = load_dataset("allocine", split='train')
    
    texts = []
    labels = []
    
    # DEBUG: Check what the first example looks like
    first_example = dataset[0]
    print(f"  DEBUG - First example keys: {first_example.keys()}")
    print(f"  DEBUG - Label value: {first_example['label']}, type: {type(first_example['label'])}")
    
    for ex in dataset:
        texts.append(ex['review'])
        label_val = ex['label']
        
        # Allocine uses 0=negative, 1=positive (already binary!)
        labels.append(int(label_val))
    
    print(f"  Loaded {len(texts)} FR reviews")
    print(f"    Positive: {sum(labels)}, Negative: {len(labels)-sum(labels)}")
    
    return texts, labels


def download_italian_sentiment():
    """
    Download Italian sentiment dataset (theoracle with parsing).
    """
    print("Downloading Italian sentiment dataset...")
    
    try:
        dataset = load_dataset("theoracle/Italian.sentiment.analysis", split='train')
        
        texts = []
        labels = []
        
        for ex in dataset:
            formatted = ex['formatted_text']
            
            # Extract text between brackets: [ ... ]
            import re
            text_match = re.search(r'\[(.*?)\]', formatted)
            if not text_match:
                continue
            text = text_match.group(1).strip()
            
            # Extract label at the end: = positive/negative/neutral
            label_match = re.search(r'= (positive|negative|neutral)', formatted)
            if not label_match:
                continue
            label_str = label_match.group(1)
            
            # Binary: positive=1, negative=0, skip neutral
            if label_str == 'positive':
                texts.append(text)
                labels.append(1)
            elif label_str == 'negative':
                texts.append(text)
                labels.append(0)
            # Skip neutral
        
        print(f"  Loaded {len(texts)} IT samples from theoracle")
        print(f"    Positive: {sum(labels)}, Negative: {len(labels)-sum(labels)}")
        
        if len(texts) > 0:
            return texts, labels
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Fallback: synthetic data
    print("  Creating synthetic Italian data...")
    
    import random
    random.seed(42)
    
    positive = [
        "Questo prodotto è fantastico! Sono molto soddisfatto dell'acquisto.",
        "Esperienza meravigliosa, lo consiglio vivamente a tutti.",
        "Qualità eccellente, vale ogni centesimo speso.",
        "Servizio impeccabile, personale molto gentile e disponibile.",
        "Bellissimo film, una delle migliori produzioni italiane.",
        "Ottima qualità del cibo, ristorante consigliatissimo!",
        "Prodotto arrivato in perfette condizioni, molto contento.",
        "Straordinario! Supera tutte le mie aspettative.",
        "Perfetto in ogni dettaglio, complimenti davvero.",
        "Magnifico acquisto, lo ricomprerei subito senza dubbio.",
    ] * 250
    
    negative = [
        "Molto deluso da questo acquisto, qualità pessima.",
        "Esperienza terribile, non lo consiglio assolutamente.",
        "Prodotto scadente, ha smesso di funzionare dopo pochi giorni.",
        "Servizio orribile, personale scortese e incompetente.",
        "Film noioso e mal fatto, uno spreco di tempo.",
        "Cibo di pessima qualità, non tornerò mai più in questo ristorante.",
        "Prodotto danneggiato all'arrivo, molto insoddisfatto.",
        "Deludente sotto ogni punto di vista, evitate di comprarlo.",
        "Pessimo rapporto qualità-prezzo, soldi buttati.",
        "Completamente inutile, non vale nemmeno la metà del prezzo.",
    ] * 250
    
    texts = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)
    
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    print(f"  Created {len(texts)} IT samples (synthetic)")
    print(f"    Positive: {sum(labels)}, Negative: {len(labels)-sum(labels)}")
    
    return list(texts), list(labels)


def prepare_sentiment_data(max_samples=5000):
    """
    Download and prepare FR + IT sentiment datasets.
    
    Args:
        max_samples (int): Max samples per language
        
    Returns:
        dict: Prepared data
    """
    os.makedirs(SENTIMENT_DIR, exist_ok=True)
    
    # Download
    fr_texts, fr_labels = download_allocine()
    it_texts, it_labels = download_italian_sentiment()
    
    # Balance and limit
    def balance_dataset(texts, labels, max_samples):
        df = pd.DataFrame({'text': texts, 'label': labels})
        n_pos = min(max_samples//2, len(df[df['label']==1]))
        n_neg = min(max_samples//2, len(df[df['label']==0]))
        
        pos = df[df['label'] == 1].sample(n=n_pos, random_state=42)
        neg = df[df['label'] == 0].sample(n=n_neg, random_state=42)
        balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42)
        return balanced['text'].tolist(), balanced['label'].tolist()
    
    fr_texts, fr_labels = balance_dataset(fr_texts, fr_labels, max_samples)
    it_texts, it_labels = balance_dataset(it_texts, it_labels, max_samples)
    
    # Preprocess
    print("\nPreprocessing...")
    fr_processed = preprocess_text(fr_texts, language='french')
    it_processed = preprocess_text(it_texts, language='italian')
    
    data = {
        'french': {
            'texts': fr_texts,
            'tokens': fr_processed,
            'labels': fr_labels,
        },
        'italian': {
            'texts': it_texts,
            'tokens': it_processed,
            'labels': it_labels,
        }
    }
    
    # Save
    save_path = os.path.join(SENTIMENT_DIR, 'sentiment_data.pkl')
    save_embeddings(data, save_path)
    
    print(f"\nSentiment data saved to: {save_path}")
    print(f"  French:  {len(fr_texts)} samples ({sum(fr_labels)} pos, {len(fr_labels)-sum(fr_labels)} neg)")
    print(f"  Italian: {len(it_texts)} samples ({sum(it_labels)} pos, {len(it_labels)-sum(it_labels)} neg)")
    
    return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare sentiment dataset')
    parser.add_argument('--max-samples', type=int, default=5000,
                        help='Max samples per language (default: 5000)')
    args = parser.parse_args()
    
    print("SENTIMENT DATA PREPARATION")
    print("="*60)
    
    data = prepare_sentiment_data(max_samples=args.max_samples)
    
    print("\nReady for sentiment analysis!")