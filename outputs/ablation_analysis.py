import pandas as pd
import numpy as np

BASE = 'src/outputs'

baseline = pd.read_csv(f'{BASE}/test_eval_baseline_predictions.csv')
fairness = pd.read_csv(f'{BASE}/test_eval_fairness_predictions.csv')
mc = pd.read_csv(f'{BASE}/mc_dropout_results_test.csv')

variants = {
    'Baseline (CE loss, single pass)': baseline,
    'Fairness (focal+reweight, single pass)': fairness,
    'Fairness + MC-Dropout (30-pass avg)': mc,
}

print("=" * 90)
print("ABLATION: baseline vs +fairness loss vs +MC-Dropout  (held-out TEST set, n=3735)")
print("=" * 90)

rows = []
for name, df in variants.items():
    correct = df['true_label'] == df['predicted_label']
    acc = correct.mean() * 100
    mel = df[df['true_label'] == 'MEL']
    mel_recall = (mel['predicted_label'] == 'MEL').mean() * 100

    group_accs = {}
    for g in ['Light (I-II)', 'Medium (III-IV)', 'Dark (V-VI)']:
        sub = df[df['fitzpatrick_group'] == g]
        group_accs[g] = (sub['true_label'] == sub['predicted_label']).mean() * 100
    gap = max(group_accs.values()) - min(group_accs.values())

    ent_correct = df.loc[correct, 'entropy'].mean()
    ent_incorrect = df.loc[~correct, 'entropy'].mean()
    ent_sep = ent_incorrect - ent_correct

    rows.append({
        'Variant': name,
        'Overall Acc %': round(acc, 2),
        'MEL Recall %': round(mel_recall, 2),
        'Light %': round(group_accs['Light (I-II)'], 2),
        'Medium %': round(group_accs['Medium (III-IV)'], 2),
        'Dark %': round(group_accs['Dark (V-VI)'], 2),
        'Gap (pts)': round(gap, 2),
        'Entropy(correct)': round(ent_correct, 4),
        'Entropy(incorrect)': round(ent_incorrect, 4),
        'Entropy separation': round(ent_sep, 4),
    })

summary = pd.DataFrame(rows)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
print(summary.to_string(index=False))

print("\n--- Calibration detail: entropy quartiles by correctness ---")
for name, df in variants.items():
    correct = df['true_label'] == df['predicted_label']
    print(f"\n{name}:")
    print(f"  Correct   (n={correct.sum():4d}): entropy mean={df.loc[correct,'entropy'].mean():.4f} "
          f"median={df.loc[correct,'entropy'].median():.4f} std={df.loc[correct,'entropy'].std():.4f}")
    print(f"  Incorrect (n={(~correct).sum():4d}): entropy mean={df.loc[~correct,'entropy'].mean():.4f} "
          f"median={df.loc[~correct,'entropy'].median():.4f} std={df.loc[~correct,'entropy'].std():.4f}")

print("\n--- High-confidence error rate: how often is the model >90% confident AND wrong? ---")
for name, df in variants.items():
    if 'confidence' in df.columns:
        conf_col = 'confidence'
    else:
        # mc_dropout script doesn't save max-prob confidence directly, derive from low entropy as proxy
        conf_col = None
    if conf_col:
        high_conf = df[df[conf_col] > 0.9]
        wrong_high_conf = (high_conf['true_label'] != high_conf['predicted_label']).mean() * 100 if len(high_conf) else float('nan')
        print(f"{name}: {len(high_conf)}/{len(df)} predictions >90% confident, {wrong_high_conf:.2f}% of those wrong")
    else:
        low_var = df[df['variance'] < df['variance'].quantile(0.25)]
        wrong_low_var = (low_var['true_label'] != low_var['predicted_label']).mean() * 100
        print(f"{name}: lowest-variance quartile (n={len(low_var)}), {wrong_low_var:.2f}% wrong (variance used as MC confidence proxy)")

print("\n--- Latency ---")
print("Baseline / Fairness single-pass: not explicitly benchmarked here, but architecturally identical single fwd pass (~ms-scale)")
print(f"Fairness + MC-Dropout (30 passes): ~199 ms/image (from mc_dropout_infer.py run)")
