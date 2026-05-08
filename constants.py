LABEL_TO_ID = {
    "normal": 0,
    "bacterial": 1,
    "viral": 2,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
CLASS_NAMES = [ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL)]