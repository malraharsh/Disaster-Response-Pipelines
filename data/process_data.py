import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories datasets from the specified filepaths
    
    Args:
        messages_filepath: Filepath to the messages dataset
        categories_filepath: Filepath to the categories dataset
        
    Returns:
        (DataFrame) df: Merged Pandas dataframe
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """
    Cleans merged dataset
    
    Args:
        df: Merged pandas dataframe
    Returns:
        (DataFrame) df: Cleaned dataframe
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split('-').str[0]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    
    return df
    

def save_data(df, database_filename):
    """
    Save the clean data into an sqlite database
    
    Args:
        df:  Clean dataframe
        database_filename: name of the database file
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('labeled_messages', engine, index=False, if_exists='replace')
    engine.dispose()

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()