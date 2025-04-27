import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Label Encoding Function
def label_encode_columns(df, columns):
    """
    Label encode specified columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list): List of columns to label encode.

    Returns:
        df (pd.DataFrame): Dataframe with encoded columns.
        label_encoders (dict): Dict of fitted LabelEncoders for inverse transformation.
    """
    label_encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# One-Hot Encoding Function
def one_hot_encode_columns(df, one_hot_cols):
    """
    One-hot encode specified columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        one_hot_cols (list): List of columns to one-hot encode.

    Returns:
        encoded_df (pd.DataFrame): Fully encoded dataframe.
        encoder (ColumnTransformer): Fitted encoder (for future transformations).
    """
    encoder = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False), one_hot_cols)
        ],
        remainder='passthrough'
    )

    encoded_data = encoder.fit_transform(df)
    encoded_feature_names = encoder.get_feature_names_out()

    encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=df.index)

    # Clean column names
    encoded_df.columns = encoded_df.columns.str.replace('remainder__', '')
    
    return encoded_df, encoder
