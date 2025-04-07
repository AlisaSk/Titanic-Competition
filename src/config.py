class Config:
    FEATURES_TO_DROP = ['Name', 'Ticket', 'SibSp', 'Parch', 'Cabin_nan']

    NEW_FEATURES = ['AgeGroup', 'FamilySize']

    AGE_BINS = [0.0, 7.5, 30, 50, 80.0]
    AGE_LABELS = [4, 1, 3, 2]
    
    FEATURES_TO_ENCODE = ['Sex', 'Embarked', 'Cabin']