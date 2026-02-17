"""
Train Sentiment Analysis Classifier
Binary sentiment (positive/negative) on French and Italian reviews
Includes cross-lingual transfer experiments
"""
import os
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from fr_ita.utils import load_embeddings, load_word_vectors, save_embeddings
from fr_ita.config import DATA_DIR
from fr_ita.classifiers.features import sentence_to_vector


SENTIMENT_DIR = os.path.join(DATA_DIR, 'sentiment')


def build_sentiment_features(tokens, labels, word_vectors, strategy='mean'):
    """
    Build feature matrix from tokenized reviews.
    
    Args:
        tokens (list): List of tokenized reviews
        labels (list): Sentiment labels (0=neg, 1=pos)
        word_vectors (dict): word -> vector mapping
        strategy (str): 'mean' or 'max'
        
    Returns:
        tuple: (X, y)
    """
    X, y = [], []
    skipped = 0
    
    for sent_tokens, label in zip(tokens, labels):
        vec = sentence_to_vector(sent_tokens, word_vectors, strategy)
        if vec is not None:
            X.append(vec)
            y.append(label)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"  Skipped {skipped} reviews (no vocabulary overlap)")
    
    return np.array(X), np.array(y)


def train_and_evaluate_sentiment(X, y, language, embedding_type, test_size=0.2):
    """
    Train sentiment classifier and evaluate.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels (0=neg, 1=pos)
        language (str): Language or experiment name
        embedding_type (str): embedding name
        test_size (float): Test split ratio
        
    Returns:
        dict: Results
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"  Train: {len(X_train)} samples  Test: {len(X_test)} samples")
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    report = classification_report(
        y_test, y_pred,
        target_names=['Negative', 'Positive'],
        output_dict=True
    )
    
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix:")
    print(f"              Pred NEG  Pred POS")
    print(f"  True NEG      {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"  True POS      {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    return {
        'language':      language,
        'embedding':     embedding_type,
        'accuracy':      report['accuracy'],
        'precision_neg': report['Negative']['precision'],
        'recall_neg':    report['Negative']['recall'],
        'f1_neg':        report['Negative']['f1-score'],
        'precision_pos': report['Positive']['precision'],
        'recall_pos':    report['Positive']['recall'],
        'f1_pos':        report['Positive']['f1-score'],
        'macro_f1':      report['macro avg']['f1-score'],
    }


def cross_lingual_experiment(fr_tokens, fr_labels, it_tokens, it_labels, 
                             embedding_type, use_aligned=False):
    """
    Cross-lingual transfer: train on FR, test on IT
    
    Args:
        fr_tokens: French tokenized reviews
        fr_labels: French labels
        it_tokens: Italian tokenized reviews  
        it_labels: Italian labels
        embedding_type: embedding name
        use_aligned: whether to use aligned embeddings
        
    Returns:
        dict: Results
    """
    print(f"\n{'='*60}")
    if use_aligned:
        print(f"CROSS-LINGUAL WITH ALIGNMENT (train FR → test IT)")
        try:
            # Load aligned French embeddings
            fr_aligned_path = os.path.join('data', 'embeddings', embedding_type, 
                                           f'french_{embedding_type}_aligned.pkl')
            fr_aligned_data = load_embeddings(fr_aligned_path)
            fr_vectors = fr_aligned_data['word_vectors']
            print("  Using Procrustes-aligned FR embeddings")
        except Exception as e:
            print(f"  Error: Aligned embeddings not found")
            print(f"  Run: python3 -m fr_ita.alignment.procrustes_alignment --embedding {embedding_type}")
            return None
    else:
        print(f"CROSS-LINGUAL WITHOUT ALIGNMENT (train FR → test IT)")
        fr_vectors, _ = load_word_vectors(embedding_type)
        print("  Using raw (non-aligned) FR embeddings")
    
    print(f"{'='*60}")
    
    # Load Italian embeddings (always non-aligned for test)
    _, it_vectors = load_word_vectors(embedding_type)
    
    # Build features
    X_fr, y_fr = build_sentiment_features(fr_tokens, fr_labels, fr_vectors, 'mean')
    X_it, y_it = build_sentiment_features(it_tokens, it_labels, it_vectors, 'mean')
    
    print(f"  FR features: {X_fr.shape}")
    print(f"  IT features: {X_it.shape}")
    
    # Train on ALL French data
    print("\n  Training on French...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_fr, y_fr)
    
    # Test on Italian
    print("  Testing on Italian...")
    y_pred = clf.predict(X_it)
    
    report = classification_report(
        y_it, y_pred,
        target_names=['Negative', 'Positive'],
        output_dict=True
    )
    
    print(classification_report(y_it, y_pred, target_names=['Negative', 'Positive']))
    
    cm = confusion_matrix(y_it, y_pred)
    print(f"  Confusion matrix:")
    print(f"              Pred NEG  Pred POS")
    print(f"  True NEG      {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"  True POS      {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    experiment_name = 'FR→IT (aligned)' if use_aligned else 'FR→IT (raw)'
    
    return {
        'language':      experiment_name,
        'embedding':     embedding_type,
        'accuracy':      report['accuracy'],
        'precision_neg': report['Negative']['precision'],
        'recall_neg':    report['Negative']['recall'],
        'f1_neg':        report['Negative']['f1-score'],
        'precision_pos': report['Positive']['precision'],
        'recall_pos':    report['Positive']['recall'],
        'f1_pos':        report['Positive']['f1-score'],
        'macro_f1':      report['macro avg']['f1-score'],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sentiment classifier')
    parser.add_argument('--embedding', type=str, default='word2vec',
                        choices=['word2vec', 'fasttext', 'glove'],
                        help='Embedding type to use')
    parser.add_argument('--all', action='store_true',
                        help='Run on all embedding types')
    parser.add_argument('--strategy', type=str, default='mean',
                        choices=['mean', 'max'],
                        help='Sentence aggregation (default: mean)')
    parser.add_argument('--cross-lingual', action='store_true',
                        help='Run cross-lingual experiments')
    args = parser.parse_args()
    
    print("SENTIMENT ANALYSIS CLASSIFIER  (Positive vs Negative)")
    print("="*60)
    
    # Load sentiment data
    sentiment_path = os.path.join(SENTIMENT_DIR, 'sentiment_data.pkl')
    
    if not os.path.exists(sentiment_path):
        print(f"Sentiment data not found: {sentiment_path}")
        print("Run: python3 -m fr_ita.sentiment.prepare_sentiment")
        exit(1)
    
    data = load_embeddings(sentiment_path)
    
    fr_tokens = data['french']['tokens']
    fr_labels = data['french']['labels']
    it_tokens = data['italian']['tokens']
    it_labels = data['italian']['labels']
    
    print(f"Loaded sentiment data:")
    print(f"  French:  {len(fr_tokens)} reviews ({sum(fr_labels)} pos, {len(fr_labels)-sum(fr_labels)} neg)")
    print(f"  Italian: {len(it_tokens)} reviews ({sum(it_labels)} pos, {len(it_labels)-sum(it_labels)} neg)")
    
    embedding_types = ['word2vec', 'fasttext', 'glove'] if args.all else [args.embedding]
    
    all_results = []
    
    for embedding_type in embedding_types:
        print(f"\n{'='*60}")
        print(f"EMBEDDING: {embedding_type.upper()}")
        print(f"{'='*60}")
        
        try:
            fr_vectors, it_vectors = load_word_vectors(embedding_type)
            
            # EXPERIMENT 1: Monolingual French
            print(f"\n--- EXPERIMENT 1: Monolingual French ---")
            X_fr, y_fr = build_sentiment_features(fr_tokens, fr_labels, fr_vectors, args.strategy)
            print(f"  Feature matrix: {X_fr.shape}")
            results_fr = train_and_evaluate_sentiment(X_fr, y_fr, 'french', embedding_type)
            all_results.append(results_fr)
            
            # EXPERIMENT 2: Monolingual Italian
            print(f"\n--- EXPERIMENT 2: Monolingual Italian ---")
            X_it, y_it = build_sentiment_features(it_tokens, it_labels, it_vectors, args.strategy)
            print(f"  Feature matrix: {X_it.shape}")
            results_it = train_and_evaluate_sentiment(X_it, y_it, 'italian', embedding_type)
            all_results.append(results_it)
            
            # EXPERIMENT 3 & 4: Cross-lingual (if requested)
            if args.cross_lingual:
                # Without alignment
                results_cross_raw = cross_lingual_experiment(
                    fr_tokens, fr_labels, it_tokens, it_labels,
                    embedding_type, use_aligned=False
                )
                if results_cross_raw:
                    all_results.append(results_cross_raw)
                
                # With alignment
                results_cross_aligned = cross_lingual_experiment(
                    fr_tokens, fr_labels, it_tokens, it_labels,
                    embedding_type, use_aligned=True
                )
                if results_cross_aligned:
                    all_results.append(results_cross_aligned)
            
        except FileNotFoundError as e:
            print(f"  Embeddings not found: {e}")
            continue
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY — ALL EXPERIMENTS")
        print(f"{'='*60}")
        print(f"{'Embedding':<12} {'Experiment':<20} {'Accuracy':>9} {'F1':>7} {'P(POS)':>7} {'R(POS)':>7}")
        print("-"*70)
        for r in all_results:
            print(f"{r['embedding']:<12} {r['language']:<20} {r['accuracy']:>9.4f} {r['macro_f1']:>7.4f} "
                  f"{r['precision_pos']:>7.4f} {r['recall_pos']:>7.4f}")
    
    # Save results
    results_path = os.path.join(SENTIMENT_DIR, 'sentiment_results.pkl')
    save_embeddings(all_results, results_path)
    print(f"\nResults saved to: {results_path}")