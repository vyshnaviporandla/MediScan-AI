# Implementation and Evidence Plan

## Project Workflow

1. Obtain the dataset and verify its current terms and provenance.
2. Preserve the existing training, validation, and test partitions.
3. Load and resize RGB images to 224×224; augment training data only.
4. Train the MobileNetV2 transfer-learning classifier on the training split.
5. Select the best checkpoint using validation AUC and early stopping.
6. Evaluate the saved checkpoint on the held-out test split.
7. Review metrics, plots, sample predictions, and errors; document limitations.
8. Inspect split counts/readability and save a grayscale preview for dataset familiarization.
9. Run single-image inference only as an educational demonstration.

## Module Map

- `src/data.py`: loads a directory split, enforces class order, resizes images, batches, and prefetches.
- `src/model.py`: builds augmentation, frozen pretrained MobileNetV2, pooling, dropout, and the sigmoid output.
- `src/train.py`: trains against train/validation data, saves the selected checkpoint, and generates a CSV log and training plot.
- `src/evaluate.py`: evaluates the held-out test data and writes metrics, a classification report, a confusion matrix, and example predictions.
- `src/inspect_data.py`: validates expected split/class folders, reports readable/unreadable image counts, and saves a grayscale training preview.
- `src/plotting.py`: produces consistently styled training curves and confusion-matrix visualizations for the app and reports.
- `src/predict.py`: runs one-image inference and prints the non-diagnostic-use notice.
- `main.py`: dispatches the `inspect`, `train`, `evaluate`, and `predict` commands.
- `app.py`: provides a local Streamlit interface for image inference, displaying the saved evaluation snapshot, and explaining project limitations.
- `tests/`: checks dataset inspection, image loading, image shape, and stable class indices.

## Phased Work and Verification

| Phase | Work | Evidence to capture | Guardrail |
|---|---|---|---|
| Environment | Create a virtual environment and install dependencies | Python/TensorFlow versions and successful imports | Check TensorFlow compatibility with the selected Python/OS |
| Dataset | Arrange the supplied splits and record source, version, terms, and counts | Folder layout and class counts | Do not commit images or mix splits |
| Dataset inspection | Run `python main.py inspect` to verify image readability and preview training examples | Counts by split/class, corrupt-file list, `outputs/dataset_preview.png` | The grayscale figure is for display; model inputs remain RGB |
| Input pipeline | Load labels in stable order and resize to 224×224 RGB | Batch dimensions and label mapping | Do not augment validation or test data |
| Model | Build the transfer-learning classifier | Model summary and output shape | Treat scores as model outputs, not diagnoses |
| Training | Train on `train`, choose a checkpoint on `val` | Training log, curves, checkpoint | Never train or select checkpoints using `test` |
| Evaluation | Evaluate on the held-out `test` split | Metrics JSON, report, confusion matrix, error examples | Do not tune or repeatedly optimize against test results |
| Inference | Predict one image | Command and output with educational-use context | Use only data that is authorized for this purpose |
| Web interface | Launch `python -m streamlit run app.py` | Local upload demo, model metrics, and training plots | Do not expose it publicly or upload identifiable patient images |
| Reporting | Record reproducibility details and limitations | Completed notes below and README assets if permitted | Never fabricate results or imply clinical validation |

## Reproducibility Record

Complete this record only after an actual run:

- Dataset source, version, access date, and terms reviewed:
- Split provenance and train/validation/test counts by class:
- Python and TensorFlow versions:
- Machine and accelerator:
- Training command, epoch limit, and selected epoch:
- Checkpoint path:
- Test threshold:
- Test accuracy, pneumonia sensitivity, normal specificity, and ROC AUC:
- Confusion matrix and class-wise results:
- Error examples, duplicate/leakage checks, and remaining limitations:

Do not report values that were not produced or verified by the run. The result of this project is not evidence of clinical performance.

Run project checks with `python -m unittest discover -s tests -v` and `python -m compileall main.py src tests`.

## Evidence Assets

Generated artifacts belong in `outputs/` and are ignored by Git. Optional README-ready images may go in `images/` only if the dataset's terms allow redistribution and the assets contain no patient-identifying information. Do not commit raw X-rays, trained weights, or local run outputs unless there is a specific, authorized reason to do so.
