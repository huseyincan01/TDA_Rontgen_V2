import json

with open("artifacts/runs/zone_svm_training_report.json", "r", encoding="utf-8") as f:
    svm_rep = json.load(f)
with open("artifacts/runs/zone_knn_training_report.json", "r", encoding="utf-8") as f:
    knn_rep = json.load(f)
with open("artifacts/runs/zone_cnn_training_report.json", "r", encoding="utf-8") as f:
    cnn_rep = json.load(f)

print("=" * 100)
print("HASTA DÜZEYİ TEST PERFORMANSI KARŞILAŞTIRMASI (v2.1 TEMİZ VERİ HATTI)")
print("=" * 100)
print(f"{'Model':<12} | {'Th':<6} | {'Val ROC':<8} | {'Test ROC':<8} | {'PR-AUC':<8} | {'Sens':<8} | {'Spec':<8} | {'F1':<8} | {'Prec':<8} | {'Acc':<8} | {'TP':<4} {'FP':<4} {'TN':<4} {'FN':<4}")
print("-" * 100)
for name, rep in [("RBF-SVM", svm_rep), ("k-NN", knn_rep), ("Küçük CNN", cnn_rep)]:
    m = rep["patient_test_metrics"]
    vm = rep["patient_val_metrics"]
    th = rep["patient_threshold"]
    print(f"{name:<12} | {th:<6.4f} | {vm['roc_auc']:<8.4f} | {m['roc_auc']:<8.4f} | {m['pr_auc']:<8.4f} | {m['sensitivity']:<8.4f} | {m['specificity']:<8.4f} | {m['f1']:<8.4f} | {m['precision']:<8.4f} | {m['accuracy']:<8.4f} | {m['tp']:<4} {m['fp']:<4} {m['tn']:<4} {m['fn']:<4}")

print("\n" + "=" * 100)
print("ZON BAZINDA TEST VE VALIDATION METRİKLERİ")
print("=" * 100)
for zid in range(1, 6):
    zkey = f"zone_{zid}"
    zname = svm_rep["zone_results"][zkey]["zone_name"]
    print(f"\n[ZON {zid}: {zname}]")
    print(f"{'Model':<12} | {'En İyi Param':<30} | {'Eşik':<6} | {'Val ROC':<8} | {'Test ROC':<8} | {'PR-AUC':<8} | {'Sens':<8} | {'Spec':<8}")
    print("-" * 95)
    for name, rep in [("RBF-SVM", svm_rep), ("k-NN", knn_rep), ("Küçük CNN", cnn_rep)]:
        zres = rep["zone_results"][zkey]
        vm = zres["val_metrics"]
        tm = zres["test_metrics"]
        bp = str(zres.get("best_params", f"Epoch {zres.get('best_epoch')}"))
        th = zres["optimal_threshold"]
        print(f"{name:<12} | {bp:<30} | {th:<6.4f} | {vm['roc_auc']:<8.4f} | {tm['roc_auc']:<8.4f} | {tm['pr_auc']:<8.4f} | {tm['sensitivity']:<8.4f} | {tm['specificity']:<8.4f}")
