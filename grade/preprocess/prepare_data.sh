# load raw dailydialog
python load_data.py

# lexical negative sampling
javac IndexCreate.java
java IndexCreate
javac IndexSearch.java
java IndexSearch

# generate standard training data
python prepare_data.py

# generate pkl
python prepare_pkl.py