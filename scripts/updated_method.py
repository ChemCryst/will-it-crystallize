from rdkit import Chem 
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn import preprocessing, svm, metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np



# Create list of descriptor names from rdkit list of names (one per line).
# This descriptor list method only works for rdkit version 2012.09 and earlier,
# as newer versions contain extra descriptors which give non-numerical values
# for some molecules. These are:
# MinPartialCharge, MaxPartialCharge, MinAbsPartialCharge and MaxAbsPartialCharge.
# These descriptors must be removed from the descriptor list manually for later
# RDKit versions

names = [x[0] for x in Descriptors._descList]
names.remove('MinPartialCharge')
names.remove('MaxPartialCharge')
names.remove('MinAbsPartialCharge')
names.remove('MaxAbsPartialCharge')
calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)

train_desc_unscaled = []
train_labels = []
test_desc_unscaled = []
test_labels = []

def add_molecules (filename, iscryst, unscaled_descriptors, labels):
    print filename
    molecule_supplier = Chem.SmilesMolSupplier(filename)
    for molecule in molecule_supplier:
        if molecule is not None:
            descriptors = calc.CalcDescriptors(molecule)
            unscaled_descriptors.append(descriptors)
            labels.append(iscryst) 

# Generate descriptors and labels for all data (training and test, cryst and non-cryst)
add_molecules( 'non_crystalline_train_file.smi', 0, train_desc_unscaled, train_labels )
add_molecules( 'crystalline_train_file.smi', 1, train_desc_unscaled, train_labels )
add_molecules( 'non_crystalline_test_file.smi', 0, test_desc_unscaled, test_labels)
add_molecules( 'crystalline_test_file.smi', 1, test_desc_unscaled, test_labels )
  
# Scale descriptors for use with SVM
train_desc_unscaled = np.array(train_desc_unscaled)
train_labels = np.array(train_labels)
scaler = preprocessing.StandardScaler().fit(train_desc_unscaled)
train_desc = scaler.transform(train_desc_unscaled)

# Train a Support Vector Machine predictor
SVM_classifier = svm.SVC(gamma=0.001, C=100., probability = True)
SVM_classifier = SVM_classifier.fit(train_desc,train_labels)

#Train a Random Forest Classifier (on unscaled descriptors)
RF_classifier = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0,n_jobs=1)
RF_classifier = RF_classifier.fit(train_desc_unscaled,train_labels)

# Scale test descriptors
test_desc_unscaled = np.array(test_desc_unscaled)
test_labels = np.array(test_labels)
test_desc = scaler.transform(test_desc_unscaled)

# Output confusion matrix and percentage accuracy on test sets
print 'SVM'
SVM_predictions = SVM_classifier.predict(test_desc)
print metrics.confusion_matrix(test_labels,SVM_predictions)
SVM_accuracy = SVM_classifier.score(test_desc,test_labels)
print SVM_accuracy

print "Random Forest"
RF_predictions = RF_classifier.predict(test_desc_unscaled)
print metrics.confusion_matrix(test_labels,RF_predictions)
RF_accuracy = RF_classifier.score(test_desc_unscaled,test_labels)
print RF_accuracy

#Calculate probability of belonging to either class for each test molecule
SVM_probabilities = SVM_classifier.predict_proba(test_desc)
