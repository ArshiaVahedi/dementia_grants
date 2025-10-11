# classify_and_export_final.py
# Place this file in the same folder as SearchResult_Export_30Sep2025_091317.xlsx
# Run: python classify_and_export_final.py
# Outputs (folder classification_outputs):
# - classified_strict.csv, classified_balanced.csv, classified_loose.csv
# - combined_summary_modes.csv, wide_summary_modes.csv
# - full_classified_balanced.xlsx
# - balanced_summary_table.csv
# - balanced_counts.png, balanced_funding_billion.png, balanced_funding_share.png
# - balanced_top10_by_funding.csv, balanced_median_by_category.csv
# - ambiguous_balanced.csv (tie/ambiguous cases logged)
# - readme.txt

import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ---------- CONFIG ----------
INPUT_XLSX = "SearchResult_Export_30Sep2025_091317.xlsx"
MIN_COST = 1000
CO_OCCUR_WINDOW_CHARS = 120
OUTPUT_DIR = Path("classification_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Ambiguous token rules: only count token if anchor appears near it
AMBIGUOUS_RULES = {
    'protein': ['amyloid','tau','amyloid-beta','tauopathy','biomarker','csf','oligomer'],
    'motor': ['gait','walk','stride','movement','ambulat','posture','balance','tug','gait speed'],
    'behavior': ['cognit','memory','neuropsych','mci','behavioral','executive']
}

# Comprehensive keyword libraries (adapt/extend as needed)
KEYWORD_LIBRARIES = {
    'proprioception': [
        'propriocept','proprioception','proprioceptive','kinesthetic','kinesthesia',
        'body schema','body awareness','joint position sense','vestibular','vestibul',
        'postural','posture','postur','spatial orientation','spatial navigation','spatial awareness',
        'gait','walking','walk','stride','step','cadence','ambulation','locomot','movement','mobility',
        'treadmill','timed up and go','tug','sit to stand','gait speed','balance','fall','falls',
        'fall risk','posturography','force plate','force platform','postural sway','sensorimotor',
        'somatosensory','tactile','haptic','vibration','proprioceptive acuity',
        'pressure sensation','touch','muscle spindle','spindle','accelerometer','gyroscope','inertial measurement unit','imu',
        'wearable','force sensor','accelerom','gyroscop',
        'ataxia','dysmetria','coordination','motor control','motor planning','vertigo','dizziness',
        'peripheral neuropathy','sensory integration','sensory processing','multisensory'
    ],
    'molecular': [
        'amyloid','amyloid-beta','amyloid beta','tau','apoe','apolipoprotein','alpha-synuclein','synuclein',
        'biomarker','biomarkers','protein','proteomic','proteomics','genom','genetic','genetics',
        'transcript','transcriptomics','rna','mrna','microglia','cytokine','inflamm','inflammation',
        'metabolom','metabolomics','lipid','lipidomics','csf','tauopathy','patholog','neurodegener',
        'oxidative stress','protease','phosphorylation','aggregation','oligomer','synapse','synaptic','phospho'
    ],
    'cognitive': [
        'cognit','cognition','memory','neuropsych','moca','mmse','assessment','diagnos','screen',
        'screening','behavior','behavioral','mci','mild cognitive','subjective cognitive','neuroimaging',
        'pet','fdg','fmri','imaging','hippocamp','executive','attention','language','verbal fluency',
        'olfact','olfactory','smell','odor','odor identification','upsit','olfactory bulb','olfactory testing'
    ]
}

# ---------- HELPERS ----------
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_patterns(keyword_list, mode):
    patterns = []
    for kw in keyword_list:
        kw_clean = kw.strip()
        if mode == 'loose':
            pat = re.compile(re.escape(kw_clean), flags=re.IGNORECASE)
        else:
            if ' ' in kw_clean:
                pat = re.compile(r'\b' + re.escape(kw_clean) + r'\b', flags=re.IGNORECASE)
            else:
                pat = re.compile(r'\b' + re.escape(kw_clean) + r'\w*\b', flags=re.IGNORECASE)
        patterns.append((kw_clean, pat))
    return patterns

def near_anchor(full_text, token, anchors, window_chars=CO_OCCUR_WINDOW_CHARS):
    # find token occurrences; return True if any anchor appears within window_chars around token
    for m in re.finditer(re.escape(token), full_text, flags=re.IGNORECASE):
        start = max(0, m.start() - window_chars)
        end = min(len(full_text), m.end() + window_chars)
        window = full_text[start:end]
        for a in anchors:
            if re.search(re.escape(a), window, flags=re.IGNORECASE):
                return True
    return False

def classify_text(title, terms, phr, patterns_by_cat, mode='balanced'):
    title_n = normalize_text(title)
    terms_n = normalize_text(terms)
    phr_n = normalize_text(phr)
    full = f"{title_n} {terms_n} {phr_n}"

    # Rule 1: Title-first for proprioception
    for kw, pat in patterns_by_cat['proprioception']:
        if pat.search(title_n):
            return 'Proprioception/Sensorimotor', {'proprioception':1,'cognitive':0,'molecular':0}

    # Count unique keyword hits per category
    found = {cat:set() for cat in patterns_by_cat}
    for cat, patterns in patterns_by_cat.items():
        for kw, pat in patterns:
            if pat.search(full):
                # ambiguous token handling
                if mode in ('balanced','strict') and kw in AMBIGUOUS_RULES:
                    if not near_anchor(full, kw, AMBIGUOUS_RULES[kw]):
                        continue
                if mode == 'strict' and kw in AMBIGUOUS_RULES:
                    continue
                found[cat].add(kw)

    counts = {cat: len(found[cat]) for cat in found}
    if sum(counts.values()) == 0:
        return 'Other/Unclassified', counts

    maxc = max(counts.values())
    winners = [cat for cat, c in counts.items() if c == maxc]
    if len(winners) == 1:
        chosen = winners[0]
    else:
        for p in ['proprioception','cognitive','molecular']:
            if p in winners:
                chosen = p
                break
    mapping = {'proprioception':'Proprioception/Sensorimotor','molecular':'Molecular/Basic Science','cognitive':'Cognitive/Diagnostic'}
    return mapping.get(chosen,'Other/Unclassified'), counts

# ---------- MAIN ----------
def run_and_export(input_xlsx=INPUT_XLSX, modes=('strict','balanced','loose')):
    xls = pd.ExcelFile(input_xlsx)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    # Clean dataset
    df_clean = df[(df['Total Cost'].notna()) & (df['Total Cost'] >= MIN_COST)].copy()
    df_clean.reset_index(drop=True, inplace=True)

    results = {}
    # classify for each mode
    for mode in modes:
        patterns = {cat: build_patterns(KEYWORD_LIBRARIES[cat], mode) for cat in KEYWORD_LIBRARIES}
        assigned = []
        counts_all = []
        ambiguous_log = []
        for idx, row in df_clean.iterrows():
            title = row.get('Project Title','')
            terms = row.get('Project Terms','')
            phr = row.get('Public Health Relevance','')
            assigned_cat, counts = classify_text(title, terms, phr, patterns, mode=mode)
            assigned.append(assigned_cat)
            counts_all.append(counts)
            # log ambiguous/tie: if more than one category share top count or low specificity
            if isinstance(counts, dict):
                top = max(counts.values())
                winners = [k for k,v in counts.items() if v==top] if top>0 else []
                if len(winners) > 1:
                    ambiguous_log.append({
                        'Project Number': row.get('Project Number'),
                        'Title': title,
                        'Assigned': assigned_cat,
                        'Counts': counts,
                        'Total Cost': row.get('Total Cost')
                    })
        df_clean[f'Assigned_{mode}'] = assigned
        df_clean[f'Counts_{mode}'] = counts_all
        # save classified CSV
        out_cols = ['Project Number','Project Title','Project Terms','Public Health Relevance','Total Cost', f'Assigned_{mode}', f'Counts_{mode}']
        df_clean[out_cols].to_csv(OUTPUT_DIR / f"classified_{mode}.csv", index=False)
        # summary
        summary = df_clean.groupby(f'Assigned_{mode}')['Total Cost'].agg(['count','sum']).reset_index().rename(columns={'count':'n_grants','sum':'total_funding'})
        summary.to_csv(OUTPUT_DIR / f"summary_{mode}.csv", index=False)
        results[mode] = {'df': df_clean.copy(), 'summary': summary, 'ambiguous': ambiguous_log}
        # save ambiguous for balanced only to help reviewers
        if mode == 'balanced':
            amb_df = pd.DataFrame(ambiguous_log)
            if not amb_df.empty:
                amb_df.to_csv(OUTPUT_DIR / "ambiguous_balanced.csv", index=False)
            else:
                # write empty file for transparency
                pd.DataFrame(columns=['Project Number','Title','Assigned','Counts','Total Cost']).to_csv(OUTPUT_DIR / "ambiguous_balanced.csv", index=False)

    # Combined summary across modes
    combined = []
    for mode in modes:
        s = results[mode]['summary'].copy()
        s['mode'] = mode
        combined.append(s)
    combined_summary = pd.concat(combined, ignore_index=True)
    combined_summary.to_csv(OUTPUT_DIR / "combined_summary_modes.csv", index=False)

    # Wide summary for manuscript table (Category rows, mode columns)
    cats = sorted(combined_summary['Assigned_'+m].unique() if False else combined_summary['Assigned_Category'].unique()) if False else None
    # simpler: use union of categories present across modes
    cat_set = set()
    for mode in modes:
        cat_set.update(results[mode]['summary']['Assigned_'+mode].tolist() if 'Assigned_'+mode in results[mode]['summary'].columns else results[mode]['summary']['AssignedCategory'].tolist() if 'AssignedCategory' in results[mode]['summary'].columns else results[mode]['summary']['Assigned_'+mode].tolist())
    # safer: derive categories from concatenated summaries
    cat_list = sorted(set(combined_summary['AssignedCategory'].tolist()))
    wide = pd.DataFrame({'Category': cat_list})
    for mode in modes:
        temp = combined_summary[combined_summary['mode']==mode].set_index('AssignedCategory')
        wide[f'{mode}_n_grants'] = [int(temp.loc[c,'n_grants']) if c in temp.index else 0 for c in wide['Category']]
        wide[f'{mode}_funding'] = [float(temp.loc[c,'total_funding']) if c in temp.index else 0.0 for c in wide['Category']]
    wide.to_csv(OUTPUT_DIR / "wide_summary_modes.csv", index=False)

    # Balanced-mode detailed outputs for manuscript
    balanced = results['balanced']
    balanced_df = balanced['df']
    # Balanced summary table CSV (counts and funding)
    bal_summary = balanced['summary'].copy()
    bal_summary.to_csv(OUTPUT_DIR / "balanced_summary_table.csv", index=False)
    # Save full classified balanced in Excel for reviewers
    cols_for_full = ['Project Number','Project Title','Project Terms','Public Health Relevance','Total Cost','Assigned_balanced','Counts_balanced']
    balanced_df[cols_for_full].to_excel(OUTPUT_DIR / "full_classified_balanced.xlsx", index=False)

    # Figures: counts bar, funding bar (USD billions), funding share pie
    # Prepare plotting data
    cats = bal_summary['Assigned_balanced'].tolist() if 'Assigned_balanced' in bal_summary.columns else bal_summary['AssignedCategory'].tolist()
    n_vals = bal_summary['n_grants'].tolist()
    f_vals = bal_summary['total_funding'].tolist()
    # bar: counts
    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.bar(cats, n_vals)
    ax1.set_ylabel('Number of grants')
    ax1.set_title('Number of grants by Category (Balanced mode)')
    plt.xticks(rotation=25, ha='right')
    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / "balanced_counts.png", dpi=300)
    plt.close(fig1)
    # bar: funding in billions
    funding_b = [v/1e9 for v in f_vals]
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.bar(cats, funding_b)
    ax2.set_ylabel('Total funding (USD billions)')
    ax2.set_title('Total funding by Category (Balanced mode)')
    plt.xticks(rotation=25, ha='right')
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "balanced_funding_billion.png", dpi=300)
    plt.close(fig2)
    # pie: funding share
    fig3, ax3 = plt.subplots(figsize=(6,6))
    ax3.pie(funding_b, labels=cats, autopct=lambda p: f'{p:.1f}%')
    ax3.set_title('Funding share by Category (Balanced mode)')
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "balanced_funding_share.png", dpi=300)
    plt.close(fig3)

    # Additional balanced analyses useful for reviewers
    # 1) Top 10 grants by funding per assigned category
    top_list = []
    for cat in bal_summary['Assigned_balanced'].tolist() if 'Assigned_balanced' in bal_summary.columns else bal_summary['AssignedCategory'].tolist():
        temp = balanced_df[balanced_df['Assigned_balanced']==cat] if 'Assigned_balanced' in balanced_df.columns else balanced_df[balanced_df['AssignedCategory']==cat]
        temp_sorted = temp.sort_values('Total Cost', ascending=False).head(10)
        if not temp_sorted.empty:
            temp_sorted_small = temp_sorted[['Project Number','Project Title','Total Cost']].copy()
            temp_sorted_small['Category'] = cat
            top_list.append(temp_sorted_small)
    if top_list:
        top10_df = pd.concat(top_list, ignore_index=True)
        top10_df.to_csv(OUTPUT_DIR / "balanced_top10_by_funding.csv", index=False)
    else:
        pd.DataFrame(columns=['Project Number','Project Title','Total Cost','Category']).to_csv(OUTPUT_DIR / "balanced_top10_by_funding.csv", index=False)

    # 2) Median funding per grant by category + bootstrap 95% CI (balanced)
    med_rows = []
    for idx, row in bal_summary.iterrows():
        cat = row['Assigned_balanced'] if 'Assigned_balanced' in row.index else row['AssignedCategory']
        subset = balanced_df[balanced_df['Assigned_balanced']==cat] if 'Assigned_balanced' in balanced_df.columns else balanced_df[balanced_df['AssignedCategory']==cat]
        costs = subset['Total Cost'].dropna().values
        if len(costs) == 0:
            median = np.nan
            ci_low = np.nan
            ci_high = np.nan
        else:
            median = float(np.median(costs))
            # bootstrap for 95% CI
            boot_meds = []
            for i in range(1000):
                samp = np.random.choice(costs, size=len(costs), replace=True)
                boot_meds.append(np.median(samp))
            ci_low = float(np.percentile(boot_meds, 2.5))
            ci_high = float(np.percentile(boot_meds, 97.5))
        med_rows.append({'Category': cat, 'median_funding': median, 'ci_2.5%': ci_low, 'ci_97.5%': ci_high, 'n_grants': int(row['n_grants'])})
    med_df = pd.DataFrame(med_rows)
    med_df.to_csv(OUTPUT_DIR / "balanced_median_by_category.csv", index=False)

    # 3) Table ready for manuscript: category, n_grants, total_funding (USD), percent_funding, median (with CI)
    total_funding_all = bal_summary['total_funding'].sum()
    manuscript_rows = []
    for idx, row in bal_summary.iterrows():
        cat = row['Assigned_balanced'] if 'Assigned_balanced' in row.index else row['AssignedCategory']
        n = int(row['n_grants'])
        tot = float(row['total_funding'])
        pct = 100.0 * tot / total_funding_all if total_funding_all>0 else 0.0
        median_row = med_df[med_df['Category']==cat].iloc[0]
        manuscript_rows.append({
            'Category': cat,
            'n_grants': n,
            'total_funding_USD': tot,
            'percent_of_total_funding': pct,
            'median_funding': median_row['median_funding'],
            'ci_2.5%': median_row['ci_2.5%'],
            'ci_97.5%': median_row['ci_97.5%']
        })
    manuscript_table = pd.DataFrame(manuscript_rows)
    manuscript_table.to_csv(OUTPUT_DIR / "balanced_manuscript_table.csv", index=False)

    # README
    (OUTPUT_DIR / "readme.txt").write_text(
        "classification_outputs folder contents:\n"
        "- classified_strict.csv, classified_balanced.csv, classified_loose.csv\n"
        "- summary_strict.csv, summary_balanced.csv, summary_loose.csv\n"
        "- combined_summary_modes.csv, wide_summary_modes.csv\n"
        "- full_classified_balanced.xlsx (for reviewers)\n"
        "- ambiguous_balanced.csv (ambiguous/tie cases logged)\n"
        "- balanced_summary_table.csv (simple summary)\n"
        "- balanced_counts.png, balanced_funding_billion.png, balanced_funding_share.png\n"
        "- balanced_top10_by_funding.csv, balanced_median_by_category.csv, balanced_manuscript_table.csv\n"
    )

    print("All exports complete. Check folder:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run three-mode classification and export balanced-mode results and figures.")
    parser.add_argument("--input", type=str, default=INPUT_XLSX, help="Input Excel export file")
    args = parser.parse_args()
    run_and_export(input_xlsx=args.input)
