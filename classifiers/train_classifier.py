"""
Train Language Identification Classifier (FR vs IT)
Uses word embeddings as features - evaluates with precision, recall, F1
"""
import os
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from fr_ita.utils import (load_tatoeba_tsv, preprocess_text,
                           load_word_vectors, save_embeddings)
from fr_ita.config import DATA_DIR
from fr_ita.classifiers.features import build_features


def reduce_vectors(fr_vectors, it_vectors, max_dim=100):
    """
    Reduce high-dimensional vectors (e.g. TF-IDF) using TruncatedSVD.
    Fitted separately per language to avoid OOM.
    """
    fr_words = list(fr_vectors.keys())
    it_words = list(it_vectors.keys())

    fr_arr = np.array([fr_vectors[w] for w in fr_words], dtype=np.float32)
    svd_fr = TruncatedSVD(n_components=max_dim, random_state=42)
    fr_reduced = svd_fr.fit_transform(fr_arr)
    del fr_arr

    it_arr = np.array([it_vectors[w] for w in it_words], dtype=np.float32)
    svd_it = TruncatedSVD(n_components=max_dim, random_state=42)
    it_reduced = svd_it.fit_transform(it_arr)
    del it_arr

    return (
        {w: fr_reduced[i] for i, w in enumerate(fr_words)},
        {w: it_reduced[i] for i, w in enumerate(it_words)},
    )


def train_and_evaluate(X, y, classifier='logreg', test_size=0.2, random_state=42):
    """
    Train classifier and evaluate with precision, recall, F1.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Train: {len(X_train)} samples  Test: {len(X_test)} samples")

    if classifier == 'svm':
        clf = LinearSVC(max_iter=2000, random_state=random_state)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=random_state)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=['French', 'Italian'],
        output_dict=True
    )

    print(classification_report(y_test, y_pred, target_names=['French', 'Italian']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix:")
    print(f"              Pred FR  Pred IT")
    print(f"  True FR       {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"  True IT       {cm[1,0]:4d}     {cm[1,1]:4d}")

    return {
        'accuracy':     report['accuracy'],
        'precision_fr': report['French']['precision'],
        'recall_fr':    report['French']['recall'],
        'f1_fr':        report['French']['f1-score'],
        'precision_it': report['Italian']['precision'],
        'recall_it':    report['Italian']['recall'],
        'f1_it':        report['Italian']['f1-score'],
        'macro_f1':     report['macro avg']['f1-score'],
        'classifier':   classifier,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train language ID classifier')
    parser.add_argument('--embedding', type=str, default='word2vec',
                        choices=['tfidf', 'word2vec', 'fasttext', 'glove'],
                        help='Embedding type to use as features')
    parser.add_argument('--all', action='store_true',
                        help='Run on all embedding types')
    parser.add_argument('--classifier', type=str, default='logreg',
                        choices=['logreg', 'svm'],
                        help='Classifier type (default: logreg)')
    parser.add_argument('--strategy', type=str, default='mean',
                        choices=['mean', 'max'],
                        help='Sentence aggregation strategy (default: mean)')
    args = parser.parse_args()

    print("LANGUAGE IDENTIFICATION CLASSIFIER  (FR vs IT)")
    print("=" * 60)

    french_corpus, italian_corpus, _ = load_tatoeba_tsv(max_sentences=10000)
    french_processed  = preprocess_text(french_corpus,  language='french')
    italian_processed = preprocess_text(italian_corpus, language='italian')

    print(f"French sentences : {len(french_processed)}")
    print(f"Italian sentences: {len(italian_processed)}")

    embedding_types = ['word2vec', 'fasttext', 'glove'] if args.all else [args.embedding]  # tfidf excluded (OOM)

    all_results = {}

    for embedding_type in embedding_types:
        print(f"\n{'='*60}")
        print(f"EMBEDDING: {embedding_type.upper()}")
        print(f"{'='*60}")

        try:
            fr_vectors, it_vectors = load_word_vectors(embedding_type)

            sample_dim = next(iter(fr_vectors.values())).shape[0]
            if sample_dim > 300:
                print(f"  Reducing {sample_dim}d -> 100d...")
                fr_vectors, it_vectors = reduce_vectors(fr_vectors, it_vectors, max_dim=100)

            combined_vectors = {**fr_vectors, **it_vectors}
            print(f"  Combined vocabulary: {len(combined_vectors)} words")

            X, y = build_features(
                french_processed, italian_processed,
                combined_vectors, strategy=args.strategy
            )
            print(f"  Feature matrix: {X.shape}")

            results = train_and_evaluate(X, y, classifier=args.classifier)
            results['embedding'] = embedding_type
            all_results[embedding_type] = results

        except FileNotFoundError as e:
            print(f"  Embeddings not found: {e}")
            continue
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY — ALL EMBEDDINGS")
        print(f"{'='*60}")
        print(f"{'Embedding':<12} {'Accuracy':>9} {'Macro F1':>9} {'P(FR)':>7} {'R(FR)':>7} {'P(IT)':>7} {'R(IT)':>7}")
        print("-" * 65)
        for emb, r in all_results.items():
            print(f"{emb:<12} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f} "
                  f"{r['precision_fr']:>7.4f} {r['recall_fr']:>7.4f} "
                  f"{r['precision_it']:>7.4f} {r['recall_it']:>7.4f}")

    results_path = os.path.join(DATA_DIR, 'classifier_results.pkl')
    save_embeddings(all_results, results_path)
    print(f"\nResults saved to: {results_path}")