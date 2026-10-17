"""Generate publication figures and descriptive paired analysis from saved predictions."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from traffic import CORRUPTIONS, corrupt, save_json

MODEL_NAMES = {'hog_svm': 'HOG + SVM', 'cnn_clean': 'Clean CNN', 'cnn_augmented': 'Augmented CNN'}
COLORS = {'hog_svm': '#617080', 'cnn_clean': '#3078bb', 'cnn_augmented': '#19835c'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    root = args.artifacts
    results, figures = root / 'results', root / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((results / 'metrics.json').read_text())
    predictions = np.load(results / 'predictions.npz')
    truth = predictions['truth']
    labels = json.loads((root / 'labels.json').read_text())
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 250, 'pdf.fonttype': 42})

    def save(fig, name):
        fig.savefig(figures / f'{name}.pdf', bbox_inches='tight')
        fig.savefig(figures / f'{name}.png', bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), sharey=True)
    for axis, kind in zip(axes, CORRUPTIONS):
        for name in MODEL_NAMES:
            values = [next(m['accuracy'] for m in metrics if m['model'] == name and m['corruption'] == 'clean')]
            values += [next(m['accuracy'] for m in metrics if m['model'] == name and m['corruption'] == kind and m['severity'] == s) for s in (1, 2, 3)]
            axis.plot(range(4), np.array(values) * 100, 'o-', markersize=3,
                      color=COLORS[name], label=MODEL_NAMES[name])
        axis.set(title=kind.title(), xlabel='Severity (0 = clean)', xticks=range(4), ylim=(0, 103))
        axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Test accuracy (%)')
    axes[1].legend(loc='lower left', fontsize=6.5, frameon=False)
    fig.tight_layout(pad=.6)
    save(fig, 'robustness')

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.1))
    for name in ['cnn_clean', 'cnn_augmented']:
        history = json.loads((results / f'{name}_history.json').read_text())
        epochs = [r['epoch'] for r in history]
        axes[0].plot(epochs, [r['train_loss'] for r in history], color=COLORS[name], label=MODEL_NAMES[name])
        axes[1].plot(epochs, [r['val_macro_f1'] * 100 for r in history], color=COLORS[name], label=MODEL_NAMES[name])
    axes[0].set(xlabel='Epoch', ylabel='Training cross-entropy')
    axes[1].set(xlabel='Epoch', ylabel='Clean validation macro F1 (%)')
    axes[0].legend(fontsize=7, frameon=False)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout(pad=.6)
    save(fig, 'training')

    fig, axis = plt.subplots(figsize=(6.6, 5.2))
    matrix = np.array(json.loads((results / 'cnn_augmented_confusion.json').read_text()))
    normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    image = axis.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    axis.set(xticks=range(43), yticks=range(43), xlabel='Predicted class ID', ylabel='True class ID')
    axis.tick_params(labelsize=5)
    fig.colorbar(image, ax=axis, label='Fraction within true class', shrink=.8)
    fig.tight_layout()
    save(fig, 'confusion')

    fig, axis = plt.subplots(figsize=(7.1, 2.25))
    class_rows = []
    for name in MODEL_NAMES:
        predicted = predictions[f'{name}__clean__0']
        accuracy = [float((predicted[truth == c] == c).mean()) for c in range(43)]
        axis.plot(range(43), np.array(accuracy) * 100, '.-', color=COLORS[name], linewidth=1,
                  markersize=3, label=MODEL_NAMES[name])
        class_rows += [{'model': name, 'class_id': c, 'label': labels[c],
                        'n': int((truth == c).sum()), 'accuracy': accuracy[c]} for c in range(43)]
    axis.set(xlabel='Class ID', ylabel='Clean test accuracy (%)', ylim=(0, 103), xticks=range(0, 43, 3))
    axis.legend(fontsize=7, frameon=False, loc='lower left')
    axis.grid(alpha=.2)
    fig.tight_layout(pad=.6)
    save(fig, 'per_class')
    with (results / 'per_class.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    paired = []
    for kind in CORRUPTIONS:
        base = predictions[f'cnn_clean__{kind}__3'] == truth
        aug = predictions[f'cnn_augmented__{kind}__3'] == truth
        paired.append({'corruption': kind, 'severity': 3, 'both_correct': int((base & aug).sum()),
                       'aug_only_correct': int((~base & aug).sum()),
                       'clean_only_correct': int((base & ~aug).sum()), 'both_wrong': int((~base & ~aug).sum())})
    summary = {'models': {}, 'paired_severe': paired}
    for name in MODEL_NAMES:
        rows = [m for m in metrics if m['model'] == name]
        clean = next(m for m in rows if m['corruption'] == 'clean')
        changed = [m for m in rows if m['corruption'] != 'clean']
        summary['models'][name] = {'clean_accuracy': clean['accuracy'], 'clean_macro_f1': clean['macro_f1'],
                                  'mean_corrupted_accuracy': float(np.mean([m['accuracy'] for m in changed])),
                                  'mean_corrupted_macro_f1': float(np.mean([m['macro_f1'] for m in changed]))}
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off_diagonal.ravel())[-5:][::-1], matrix.shape))[0]
    summary['largest_confusions'] = [{'true': int(a), 'predicted': int(b), 'count': int(matrix[a, b]),
                                      'true_name': labels[a], 'predicted_name': labels[b]} for a, b in pairs]
    save_json(results / 'summary.json', summary)

    cache_path = root / 'cache' / 'images.npz'
    if cache_path.exists():
        images = np.load(cache_path)['test']
        cases = []
        for kind in CORRUPTIONS:
            base = predictions[f'cnn_clean__{kind}__3']
            aug = predictions[f'cnn_augmented__{kind}__3']
            recovered = np.flatnonzero((base != truth) & (aug == truth))
            failed = np.flatnonzero(aug != truth)
            for pool in [recovered, failed]:
                if len(pool):
                    i = int(pool[0])
                    cases.append((kind, i, int(base[i]), int(aug[i])))
        fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.8))
        for axis, (kind, i, base, aug) in zip(axes.flat, cases):
            original = images[i]
            changed = corrupt(original, kind, 3, 42 + i)
            axis.imshow(np.concatenate([original, changed], axis=1), interpolation='nearest')
            axis.set_title(f'{kind.title()}, true ID {truth[i]}', fontsize=8)
            axis.set_xlabel(f'Clean CNN: {base} | Augmented: {aug}', fontsize=7)
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
        fig.tight_layout(pad=1.1)
        save(fig, 'examples')
        save_json(results / 'example_cases.json', [{'corruption': k, 'test_index': i,
                                                   'true_class': int(truth[i]), 'clean_cnn': b,
                                                   'augmented_cnn': a} for k, i, b, a in cases])

    fig, axis = plt.subplots(figsize=(7.1, 1.1))
    axis.set(xlim=(0, 10), ylim=(0, 1))
    axis.axis('off')
    texts = ['GTSRB\n32 × 32 RGB', 'Track-based\ntrain / validation', 'HOG + SVM\nCNN / Aug. CNN', 'Clean + nine\ncorruptions', 'Accuracy\nMacro F1']
    for i, text in enumerate(texts):
        x = i * 2.03
        axis.add_patch(FancyBboxPatch((x, .18), 1.65, .62, boxstyle='round,pad=0.04',
                                     facecolor='#edf3f6', edgecolor='#547080', linewidth=.8))
        axis.text(x + .825, .49, text, ha='center', va='center', fontsize=8)
        if i < 4:
            axis.add_patch(FancyArrowPatch((x + 1.72, .49), (x + 1.97, .49), arrowstyle='->',
                                           mutation_scale=9, color='#547080'))
    save(fig, 'pipeline')
    print(f'Figures and summary saved to {figures}')


if __name__ == '__main__':
    main()
