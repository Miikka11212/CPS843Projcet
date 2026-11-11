"""Fill the IEEE report with actual experiment values; optionally compile with Tectonic."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def escape(text):
    return str(text).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tectonic', help='Path to the Tectonic executable; omit to generate .tex only')
    args = parser.parse_args()
    results = HERE / 'artifacts' / 'results'
    manifest = json.loads((results / 'manifest.json').read_text())
    summary = json.loads((results / 'summary.json').read_text())
    training = json.loads((results / 'training.json').read_text())
    metrics = json.loads((results / 'metrics.json').read_text())
    replacements = {}
    for prefix, name in [('SVM', 'hog_svm'), ('BASE', 'cnn_clean'), ('AUG', 'cnn_augmented')]:
        stats = summary['models'][name]
        for field, key in [('ACC', 'clean_accuracy'), ('F1', 'clean_macro_f1'),
                           ('CORR', 'mean_corrupted_accuracy'), ('CORRF1', 'mean_corrupted_macro_f1')]:
            replacements[f'{prefix}_{field}'] = f'{100 * stats[key]:.2f}'
        replacements[f'{prefix}_TIME'] = f"{training[name]['training_seconds']:.1f}"
        for condition, suffix in [('blur', 'BLUR'), ('darkness', 'DARK'), ('noise', 'NOISE')]:
            row = next(m for m in metrics if m['model'] == name and m['corruption'] == condition and m['severity'] == 3)
            replacements[f'{prefix}_{suffix}'] = f"{100 * row['accuracy']:.2f}"
    delta = summary['models']['cnn_augmented']['mean_corrupted_accuracy'] - summary['models']['cnn_clean']['mean_corrupted_accuracy']
    replacements['GAIN'] = f'{100 * delta:+.2f}'
    replacements['CLEAN_DELTA'] = f"{100 * (summary['models']['cnn_augmented']['clean_accuracy'] - summary['models']['cnn_clean']['clean_accuracy']):+.2f}"
    for key, field in [('TRAIN_N', 'training_images'), ('VAL_N', 'validation_images'),
                       ('TRAIN_TRACKS', 'training_tracks'), ('VAL_TRACKS', 'validation_tracks'),
                       ('EPOCHS', 'epochs')]:
        replacements[key] = f'{manifest[field]:,}'
    replacements['PARAMS'] = f"{training['cnn_clean']['parameters']:,}"
    replacements['PYTHON'] = escape(manifest['python'])
    replacements['TORCH'] = escape(manifest['torch'])
    replacements['SKLEARN'] = escape(manifest['libraries']['scikit-learn'])
    replacements['PAIRED_ROWS'] = '\n'.join(
        f"{r['corruption'].title()} & {r['both_correct']:,} & {r['aug_only_correct']:,} & {r['clean_only_correct']:,} & {r['both_wrong']:,} " + r'\\'
        for r in summary['paired_severe'])
    confusion = summary['largest_confusions'][0]
    replacements['CONFUSION'] = (f"true class {confusion['true']} ({escape(confusion['true_name'])}) predicted as "
                                  f"class {confusion['predicted']} ({escape(confusion['predicted_name'])}), "
                                  f"with {confusion['count']} images")
    paragraphs = []
    for condition in ['blur', 'darkness', 'noise']:
        values = {name: next(m['accuracy'] for m in metrics if m['model'] == name and m['corruption'] == condition and m['severity'] == 3)
                  for name in ['hog_svm', 'cnn_clean', 'cnn_augmented']}
        difference = 100 * (values['cnn_augmented'] - values['cnn_clean'])
        paragraphs.append(f"At severity 3 for {condition}, the clean CNN obtains {values['cnn_clean'] * 100:.2f}\\% "
                          f"accuracy and the augmented CNN obtains {values['cnn_augmented'] * 100:.2f}\\%, "
                          f"a change of {difference:+.2f} percentage points. "
                          f"The HOG--SVM baseline reaches {values['hog_svm'] * 100:.2f}\\% under the same transformation.")
    replacements['CONDITION_ANALYSIS'] = '\n\n'.join(paragraphs)
    report = HERE / 'report'
    template = (report / 'template.tex').read_text(encoding='utf-8')
    for key, value in replacements.items():
        template = template.replace(f'@@{key}@@', str(value))
    if '@@' in template:
        raise ValueError('Unfilled report variable')
    (report / 'report.tex').write_text(template, encoding='utf-8')
    lines = ['# Measured results', '', 'Official GTSRB test set: 12,630 images per condition. Seed: 42.', '',
             '| Model | Clean accuracy | Clean macro F1 | Mean corrupted accuracy | Mean corrupted macro F1 |',
             '|---|---:|---:|---:|---:|']
    for name, label in [('hog_svm', 'HOG + SVM'), ('cnn_clean', 'Clean CNN'), ('cnn_augmented', 'Augmented CNN')]:
        row = summary['models'][name]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{100 * row[key]:.2f}%' for key in
                     ['clean_accuracy', 'clean_macro_f1', 'mean_corrupted_accuracy', 'mean_corrupted_macro_f1']) + ' |')
    lines += ['', f'Augmentation changes mean corrupted accuracy by **{100 * delta:+.2f} percentage points**.', '',
              'The corrupted mean weights all three severities of blur, darkness, and noise equally. '
              'These are synthetic transformations from the same families used in augmentation. '
              'This is a single-seed experiment, with no statistical significance claim.', '',
              'Full per-condition measurements: `artifacts/results/metrics.csv`. '
              'Raw predictions: `artifacts/results/predictions.npz`. '
              'The IEEE report includes methods, figures, paired outcomes, and limitations.']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if (HERE / 'IEEEtran.cls').exists():
        shutil.copy2(HERE / 'IEEEtran.cls', report / 'IEEEtran.cls')
    if args.tectonic:
        subprocess.run([str(Path(args.tectonic).resolve()), '--keep-logs', 'report.tex'], cwd=report, check=True)
    print(f'Report source: {report / "report.tex"}')


if __name__ == '__main__':
    main()
