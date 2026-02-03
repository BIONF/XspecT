#!/usr/bin/env nextflow

// --------------------- PARAMETERS ---------------------
params.classifications = "results/benchmark/read_classifications.tsv"
params.publishDir      = "results/score-svm"
params.test_size       = 0.2
params.random_state    = 42
params.sample_size     = 100000
params.mismatch_fraction = 0.2

// --------------------- WORKFLOW -----------------------
workflow {
  classifications_file = file(params.classifications)

  DOWNSAMPLE(classifications_file)
  GET_REMAINING_SAMPLES(classifications_file, DOWNSAMPLE.out.downsampled)
  UNSORTED_SVM(DOWNSAMPLE.out.downsampled)
  SPLIT_REMAINING(GET_REMAINING_SAMPLES.out.remaining)
  APPLY_SVM_TO_REMAINING(SPLIT_REMAINING.out.split_files.flatten(), UNSORTED_SVM.out.model)
  MERGE_RESULTS(APPLY_SVM_TO_REMAINING.out.rejected.collect())
  calculateStats(MERGE_RESULTS.out.merged)
}

// --------------------- PROCESSES ----------------------

process GET_REMAINING_SAMPLES {
  conda "conda-forge::pandas==2.3.3"
  cpus 16
  memory '256 GB'
  publishDir params.publishDir, mode: 'copy'

  input:
    path classifications
    path downsampled

  output:
    path "remaining_samples.tsv", emit: remaining

  script:
  """
  #!/usr/bin/env python
  import pandas as pd

  df_full = pd.read_csv("${classifications}", sep="\\t")

  df_downsampled = pd.read_csv("${downsampled}", sep="\\t")

  df_full['composite_key'] = df_full['Assembly Accession'].astype(str) + '||' + df_full['Read'].astype(str)
  df_downsampled['composite_key'] = df_downsampled['Assembly Accession'].astype(str) + '||' + df_downsampled['Read'].astype(str)

  downsampled_keys = set(df_downsampled['composite_key'].values)
  mask = ~df_full['composite_key'].isin(downsampled_keys)
  
  df_remaining = df_full[mask].copy()
  
  df_remaining = df_remaining.drop(columns=['composite_key'])

  print(f"Total samples in classifications: {len(df_full)}")
  print(f"Samples in downsampled: {len(df_downsampled)}")
  print(f"Remaining samples: {len(df_remaining)}")

  df_remaining.to_csv("remaining_samples.tsv", sep="\\t", index=False)
  """
}

process DOWNSAMPLE {
  conda "conda-forge::pandas==2.3.3"
  cpus 16
  memory '256 GB'
  publishDir params.publishDir, mode: 'copy'

  input:
    path classifications

  output:
    path "downsampled.tsv", emit: downsampled

  script:
  """
  #!/usr/bin/env python
  import pandas as pd

  df = pd.read_csv("${classifications}", sep="\\t")

  # apply to_numeric to reduce memory usage
  score_cols = [col for col in df.columns if col.isdigit()]
  for col in score_cols:
      df[col] = pd.to_numeric(df[col], downcast='float')

  # Same rows may already be rejected due to ambiguous classifications - filter them out
  df = df[df["Rejected"] == False].copy()
      
  df["label"] = (df["Species ID"].astype(str) == df["Prediction"].astype(str)).astype(int)
  
  sample_size = ${params.sample_size}
  mismatch_fraction = ${params.mismatch_fraction}
  random_state = ${params.random_state}
  
  # Separate mismatches (label=0) and correct predictions (label=1)
  mismatches = df[df["label"] == 0].copy()
  correct = df[df["label"] == 1].copy()
  
  # Calculate target sizes for each class
  n_mismatches = int(sample_size * mismatch_fraction)
  n_correct = sample_size - n_mismatches
  
  print(f"Target: {n_mismatches} mismatches, {n_correct} correct predictions")
  print(f"Available: {len(mismatches)} mismatches, {len(correct)} correct predictions")
  
  # Function to sample with even species distribution using groupby
  def stratified_sample_by_species(data, n_samples, random_state):
      if len(data) <= n_samples:
          return data
      
      n_species = data["Species ID"].nunique()
      samples_per_species = n_samples // n_species
      
      # Sample evenly from each species
      sampled = data.groupby("Species ID", group_keys=False).apply(
          lambda x: x.sample(n=min(samples_per_species, len(x)), random_state=random_state)
      )
      
      # If we need more samples, sample remaining from the full dataset
      if len(sampled) < n_samples:
          remaining_needed = n_samples - len(sampled)
          additional = data.drop(sampled.index).sample(n=min(remaining_needed, len(data) - len(sampled)), random_state=random_state)
          sampled = pd.concat([sampled, additional])
      
      return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
  
  # Sample each class with stratification by species
  sampled_mismatches = stratified_sample_by_species(mismatches, n_mismatches, random_state)
  sampled_correct = stratified_sample_by_species(correct, n_correct, random_state + 1000)
  
  # Combine and shuffle
  df_sampled = pd.concat([sampled_mismatches, sampled_correct], ignore_index=True)
  df_sampled = df_sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
  
  print(f"Final dataset: {len(df_sampled)} samples")
  print(f"  Mismatches: {(df_sampled['label'] == 0).sum()} ({(df_sampled['label'] == 0).sum() / len(df_sampled):.2%})")
  print(f"  Correct: {(df_sampled['label'] == 1).sum()} ({(df_sampled['label'] == 1).sum() / len(df_sampled):.2%})")
  print(f"  Unique species in mismatches: {df_sampled[df_sampled['label'] == 0]['Species ID'].nunique()}")
  print(f"  Unique species in correct: {df_sampled[df_sampled['label'] == 1]['Species ID'].nunique()}")
  
  df_sampled.to_csv("downsampled.tsv", sep="\\t", index=False)
  """
}

process UNSORTED_SVM {
  publishDir params.publishDir, mode: 'copy'
  conda "conda-forge::python=3.12 conda-forge::pandas conda-forge::scikit-learn conda-forge::numpy"
  cpus 16
  memory '256 GB'

  input:
    path classifications

  output:
    path "classification_report.txt", emit: classification_report
    path "svm_model.pkl", emit: model

  script:
  """
  #!/usr/bin/env python
  import pandas as pd
  import pickle
  from sklearn.svm import SVC
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import classification_report

  df = pd.read_csv("${classifications}", sep="\\t")

  # apply use_to_numeric to reduce memory usage
  score_cols = [col for col in df.columns if col.isdigit()]
  for col in score_cols:
      df[col] = pd.to_numeric(df[col], downcast='float')
      
  df["label"] = (df["Species ID"].astype(str) == df["Prediction"].astype(str)).astype(int)
  
  feature_cols = [c for c in df.columns if c.isdigit()]
  X = df[feature_cols].values
  y = df["label"].values
  
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=${params.test_size}, random_state=${params.random_state}, stratify=y
  )

  svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
  svm.fit(X_train, y_train)
  
  with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)
  
  y_pred = svm.predict(X_test)
  
  with open('classification_report.txt', 'w') as f:
    f.write(f"SVM Classification Report\\n")
    f.write(f"========================\\n\\n")
    f.write(f"Total samples used: {len(df)}\\n")
    f.write(f"Training samples: {len(X_train)}\\n")
    f.write(f"Test samples: {len(X_test)}\\n\\n")
    f.write(classification_report(y_test, y_pred, target_names=["Incorrect", "Correct"]))
  """
}

process SPLIT_REMAINING {
  cpus 8
  memory '64 GB'

  input:
    path classifications

  output:
    path "split_*.tsv", emit: split_files

  script:
  """
  #!/bin/bash
  
  input_file="${classifications}"
  n_files=1000
  
  total_lines=\$(tail -n +2 "\$input_file" | wc -l)
  header=\$(head -1 "\$input_file")
  
  echo "Total rows: \$total_lines"
  echo "Target files: \$n_files"
  
  rows_per_file=\$(( (total_lines + n_files - 1) / n_files ))
  echo "Rows per file: \$rows_per_file"
  
  tail -n +2 "\$input_file" > temp_data.tsv
  
  split -l \$rows_per_file temp_data.tsv temp_split_
  
  i=0
  for temp_file in temp_split_*; do
    filename=\$(printf "split_%05d.tsv" \$i)
    # Combine header with data
    { echo "\$header"; cat "\$temp_file"; } > "\$filename"
    num_rows=\$(wc -l < "\$temp_file")
    echo "Created \$filename with \$num_rows rows"
    rm "\$temp_file"
    ((i++))
  done
  
  rm temp_data.tsv
  """
}

process APPLY_SVM_TO_REMAINING {
  conda "conda-forge::python=3.12 conda-forge::pandas conda-forge::scikit-learn conda-forge::numpy"
  cpus 4
  memory '32 GB'

  input:
    path split_file
    path model

  output:
    path "*_with_svm_rejection.tsv", emit: rejected

  script:
  """
  #!/usr/bin/env python
  import pandas as pd
  import pickle

  df = pd.read_csv("${split_file}", sep="\\t")
  with open('${model}', 'rb') as f:
    svm = pickle.load(f)

  # reduce memory usage
  score_cols = [col for col in df.columns if col.isdigit()]
  for col in score_cols:
      df[col] = pd.to_numeric(df[col], downcast='float')

  feature_cols = [c for c in df.columns if c.isdigit()]
  X = df[feature_cols].values
  
  y_pred = svm.predict(X)
  
  initially_rejected = df["Rejected"].sum()
  
  mask_not_rejected = ~df["Rejected"]
  df.loc[mask_not_rejected & (y_pred == 0), "Rejected"] = True
  
  newly_rejected = df["Rejected"].sum() - initially_rejected
  
  print(f"File: ${split_file}")
  print(f"Total samples: {len(df)}")
  print(f"Initially rejected: {initially_rejected}")
  print(f"SVM newly rejected: {newly_rejected}")
  print(f"Total rejected: {df['Rejected'].sum()}")
  
  output_filename = "${split_file}_with_svm_rejection.tsv"
  df.to_csv(output_filename, sep="\\t", index=False)
  """
}

process MERGE_RESULTS {
  publishDir params.publishDir, mode: 'copy'
  conda "conda-forge::pandas==2.3.3"
  cpus 8
  memory '256 GB'

  input:
    path split_files

  output:
    path "remaining_with_svm_rejection.tsv", emit: merged

  script:
  """
  #!/usr/bin/env python
  import pandas as pd
  import glob

  split_files = sorted(glob.glob("split_*.tsv"))
    
  dfs = []
  for split_file in split_files:
    df = pd.read_csv(split_file, sep="\\t")
    dfs.append(df)
  
  df_merged = pd.concat(dfs, ignore_index=True)
  
  total_rejected = df_merged["Rejected"].sum()
  
  print(f"Total samples in merged file: {len(df_merged)}")
  print(f"Total rejected: {total_rejected} ({total_rejected / len(df_merged):.2%})")
  print(f"Not rejected: {(~df_merged['Rejected']).sum()} ({(~df_merged['Rejected']).sum() / len(df_merged):.2%})")
  
  df_merged.to_csv("remaining_with_svm_rejection.tsv", sep="\\t", index=False)
  """
}

process calculateStats {
  conda "conda-forge::pandas conda-forge::scikit-learn"
  cpus 8
  memory '256 GB'
  publishDir params.publishDir, mode: 'copy'

  input:
    path read_classifications

  output:
    path "stats.txt"

  script:
  """
  #!/usr/bin/env python
  import pandas as pd
  from sklearn.metrics import classification_report

  # --- Reads ---
  df_read = pd.read_csv('${read_classifications}', sep='\\t')
  df_read['Species ID'] = df_read['Species ID'].astype(str)
  df_read['Prediction'] = df_read['Prediction'].astype(str)

  y_true_read = df_read['Species ID']
  y_pred_read = df_read['Prediction']

  read_matches = (y_true_read == y_pred_read).sum()
  read_total = len(df_read)

  read_labels = sorted(set(y_true_read.unique()).union(set(y_pred_read.unique())))
  read_report = classification_report(
      y_true_read,
      y_pred_read,
      labels=read_labels,
      zero_division=0
  )

  # --- Abstaining Metrics (Reads only) ---
  # Determine actual misclassification (prediction != ground truth)
  df_read['Actually_Misclassified'] = df_read['Species ID'] != df_read['Prediction']
  
  # Get rejection status from Rejected column
  rejected = df_read['Rejected']
  not_rejected = ~rejected
  
  # Coverage: proportion of samples that are NOT rejected
  coverage = not_rejected.sum() / read_total
  
  # Selective Accuracy: accuracy on non-rejected samples only
  if not_rejected.sum() > 0:
      selective_correct = ((df_read['Species ID'] == df_read['Prediction']) & not_rejected).sum()
      selective_accuracy = selective_correct / not_rejected.sum()
      selective_risk = 1 - selective_accuracy
  else:
      selective_accuracy = 0.0
      selective_risk = 1.0
  
  # Rejection Precision: of all rejected samples, how many were actually misclassified
  if rejected.sum() > 0:
      rejection_precision = (rejected & df_read['Actually_Misclassified']).sum() / rejected.sum()
  else:
      rejection_precision = 0.0
  
  # Rejection Recall: of all misclassified samples, how many were rejected
  if df_read['Actually_Misclassified'].sum() > 0:
      rejection_recall = (rejected & df_read['Actually_Misclassified']).sum() / df_read['Actually_Misclassified'].sum()
  else:
      rejection_recall = 0.0

  # --- Output ---
  with open('stats.txt', 'w') as f:
      f.write("=== Reads ===\\n")
      f.write(f"Total: {read_total}\\n")
      f.write(f"Matches: {read_matches}\\n")
      f.write(f"Mismatches: {read_total - read_matches}\\n")
      f.write(f"Match Rate: {read_matches / read_total * 100:.2f}%\\n")
      f.write(f"Mismatch Rate: {(read_total - read_matches) / read_total * 100:.2f}%\\n\\n")
      f.write("Classification report (per class):\\n")
      f.write(read_report + "\\n")
      
      f.write("\\n=== Abstaining Metrics (Reads) ===\\n")
      f.write(f"Total Reads: {read_total}\\n")
      f.write(f"Rejected Reads: {rejected.sum()}\\n")
      f.write(f"Accepted Reads: {not_rejected.sum()}\\n")
      f.write(f"Coverage: {coverage * 100:.2f}% (proportion of non-rejected samples)\\n")
      f.write(f"Selective Accuracy: {selective_accuracy * 100:.2f}% (accuracy on non-rejected samples)\\n")
      f.write(f"Selective Risk: {selective_risk * 100:.2f}% (error rate on non-rejected samples)\\n")
      f.write(f"Rejection Precision: {rejection_precision * 100:.2f}% (of rejected, how many were truly misclassified)\\n")
      f.write(f"Rejection Recall: {rejection_recall * 100:.2f}% (of misclassified, how many were rejected)\\n")
  """
}

