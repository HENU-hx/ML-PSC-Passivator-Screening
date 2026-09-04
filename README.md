ML-assisted passivator screening



This data and code package supports reproduction of the machine-learning-assisted molecular passivator screening workflow reported in the manuscript.



1. Files

code/
01\_data\_preparation/               Descriptor calculation and data preparation
02\_feature\_engineering/          Final all-data feature selection and library preprocessing
03\_model\_construction/          Leakage-controlled outer evaluation and paper figures
04\_model\_interpretation/        SHAP and permutation-importance analysis
05\_virtual\_screening/               Final RF training, AD assessment, and diversity selection
06\_auxiliary\_xTB\_analysis/        Optional xTB electronic-structure analysis

data/
smiles\_training.csv                                 Training-set SMILES
smiles\_library.csv                                   Virtual-screening library SMILES
training\_14descriptors.csv                     Final 14-descriptor model-training dataset

training\_25descriptors\_labeled.csv        Core input for the 50 repeated outer evaluations.
processed/                                           Intermediate processed datasets

outputs/                                               Generated models, results, and figures



2. Requirements

Python >= 3.9

conda create -n passivator\_ml python=3.9
conda activate passivator\_ml
pip install -r requirements.txt

Key dependencies include numpy, pandas, scikit-learn, matplotlib, seaborn, rdkit, shap, and joblib. The optional xTB analysis additionally requires a local xtb installation.



3. Reproduction
Run commands from the project root directory. Scripts use project-root-relative paths and accept explicit input/output arguments.

3.1 Training-data preparation

python code/01\_data\_preparation/01\_calculate\_molecular\_descriptors.py --input data/smiles\_training.csv --output data/processed/training\_25descriptors.csv

python code/01\_data\_preparation/02\_delta\_PCE.py --input data/training\_25descriptors\_labeled.csv --output-dir outputs/delta\_PCE\_distribution

python code/02\_feature\_engineering/01\_preprocess\_training\_descriptors.py --input data/processed/training\_25descriptors.csv --output data/processed/training\_25descriptors\_clean.csv

python code/02\_feature\_engineering/02\_pearson\_filter\_descriptors.py --input data/training\_25descriptors\_labeled.csv --output-dir outputs/final\_pearson\_selection

python code/02\_feature\_engineering/03\_L1\_feature\_selection.py --input outputs/final\_pearson\_selection/training\_all203\_pearson\_selected.csv --output-dir outputs/final\_l1\_selection

python code/02\_feature\_engineering/04\_preprocess\_library\_descriptors.py --input data/processed/library\_25descriptors.csv --output data/processed/library\_25descriptors\_clean.csv --model-output data/processed/library\_14descriptors.csv

The final model-ready dataset (data/training\_14descriptors.csv) is provided directly in this package and was used for final RF construction, model interpretation, applicability-domain assessment, and virtual screening.

3.2 Model construction and interpretation

python code/03\_model\_construction/00\_outer\_repeated\_stratified\_split.py --input data/training\_25descriptors\_labeled.csv --output-dir outputs/outer\_repeated\_splits

python code/03\_model\_construction/01\_outer\_train\_pearson\_selection.py --input data/training\_25descriptors\_labeled.csv --split-assignments outputs/outer\_repeated\_splits/outer\_split\_assignments.csv --output-dir outputs/outer\_train\_pearson\_selection

python code/03\_model\_construction/02\_outer\_train\_l1\_feature\_selection.py --input data/training\_25descriptors\_labeled.csv --split-assignments outputs/outer\_repeated\_splits/outer\_split\_assignments.csv --pearson-features outputs/outer\_train\_pearson\_selection/pearson\_selected\_features\_by\_iteration.csv --output-dir outputs/outer\_train\_l1\_selection

python code/03\_model\_construction/03\_outer\_train\_eight\_model\_comparison.py --input data/training\_25descriptors\_labeled.csv --split-assignments outputs/outer\_repeated\_splits/outer\_split\_assignments.csv --l1-features outputs/outer\_train\_l1\_selection/l1\_selected\_features\_by\_iteration.csv --output-dir outputs/outer\_eight\_model\_comparison

python code/03\_model\_construction/04\_plot\_rf\_repeated\_evaluation.py --input-dir outputs/outer\_eight\_model\_comparison --output-dir outputs/rf\_paper\_figures

python code/03\_model\_construction/05\_plot\_feature\_selection\_stability.py --training-data data/training\_25descriptors\_labeled.csv --pearson-selected outputs/outer\_train\_pearson\_selection/pearson\_selected\_features\_by\_iteration.csv --l1-selected outputs/outer\_train\_l1\_selection/l1\_selected\_features\_by\_iteration.csv --output-dir outputs/feature\_selection\_stability

python code/05\_virtual\_screening/02\_train\_final\_rf\_model.py --input data/training\_14descriptors.csv --outer-best-parameters outputs/outer\_eight\_model\_comparison/eight\_models\_best\_parameters\_per\_iteration.csv --output-dir outputs/final\_rf\_model

python code/04\_model\_interpretation/01\_shap\_analysis.py --input data/training\_14descriptors.csv --model-dir outputs/final\_rf\_model --output-dir outputs/final\_shap\_analysis

python code/04\_model\_interpretation/02\_permutation\_importance.py --input data/training\_25descriptors\_labeled.csv --split-assignments outputs/outer\_repeated\_splits/outer\_split\_assignments.csv --l1-features outputs/outer\_train\_l1\_selection/l1\_selected\_features\_by\_iteration.csv --rf-best-parameters outputs/outer\_eight\_model\_comparison/eight\_models\_best\_parameters\_per\_iteration.csv --final-features outputs/final\_rf\_model/rf\_final\_feature\_order.csv --output-dir outputs/outer\_rf\_permutation\_importance

The eight-model comparison is exploratory. Final RF performance is evaluated using repeated nested hold-out evaluation.

3.3 Virtual screening

The provided trained RF model (rf\_final\_model.pkl) corresponds to the final model used for virtual screening and candidate ranking reported in the manuscript.

python code/01\_data\_preparation/01\_calculate\_molecular\_descriptors.py --input data/smiles\_library.csv --output data/processed/library\_25descriptors.csv

python code/02\_feature\_engineering/04\_preprocess\_library\_descriptors.py --input data/processed/library\_25descriptors.csv --output data/processed/library\_processed.csv --model-output data/processed/library\_14descriptors.csv

python code/05\_virtual\_screening/01\_preliminary\_screening.py --input data/processed/library\_14descriptors.csv --output data/processed/library\_14descriptors\_prescreened.csv

python code/05\_virtual\_screening/02\_train\_final\_rf\_model.py --input data/training\_14descriptors.csv --outer-best-parameters outputs/outer\_eight\_model\_comparison/eight\_models\_best\_parameters\_per\_iteration.csv --output-dir outputs/final\_rf\_model

python code/05\_virtual\_screening/03\_applicable\_domain.py --model outputs/final\_rf\_model/rf\_final\_model.pkl --feature-names outputs/final\_rf\_model/feature\_names.pkl --training data/training\_14descriptors.csv --library data/processed/library\_14descriptors\_prescreened.csv --output outputs/final\_rf\_screening/ad\_scored\_library.csv

python code/05\_virtual\_screening/04\_select\_diverse\_top\_candidates.py --input outputs/final\_rf\_screening/ad\_scored\_library.csv --top20-output outputs/final\_rf\_screening/top20\_inside\_candidates.csv --diverse-output outputs/final\_rf\_screening/final\_5\_diverse\_candidates.csv

The screening sequence is: Descriptor calculation → preprocessing → preliminary screening
→ AD-aware RF scoring → top-20 candidate pool → diversity selection



4. Auxiliary xTB Analysis

The auxiliary xTB analysis uses the standardized training-set SMILES as input. The file data/xtb/xtb\_electronic\_descriptors16.csv is a supplied integrated dataset combining xTB-derived descriptors, RDKit molecular descriptors, and ΔPCE labels, and is used directly as the input for the eight-model xTB comparison.

powershell

python code/06\_auxiliary\_xTB\_analysis/01\_generate\_xyz\_from\_smiles.py --input data/smiles\_training.csv --output-dir data/xtb/xyz\_initial

python code/06\_auxiliary\_xTB\_analysis/02\_run\_xtb\_single\_point\_out.py --input-dir data/xtb/xyz\_initial --output-dir data/xtb/xtb\_sp

python code/06\_auxiliary\_xTB\_analysis/03\_electronic\_descriptors.py --input-dir data/xtb/xtb\_sp --output data/xtb/xtb\_electronic\_descriptors.csv

python code/06\_auxiliary\_xTB\_analysis/04\_eight\_model\_tuned\_rankplots\_xTB.py --input data/xtb/xtb\_electronic\_descriptors16.csv --output-dir outputs/model\_comparison/xtb

The generated XYZ geometries, xTB single-point calculation outputs, and extracted electronic descriptors are saved in data/xtb/. A local installation of the xtbexecutable is required for the second command.



Notes
Random seeds are specified in the model scripts. Minor numerical differences may occur across operating systems or package versions. Intermediate datasets and manuscript-related outputs are retained in data/processed/ and outputs/.

