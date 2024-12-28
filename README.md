# Assignment with Azure ML

## Pre-process text
Using the "Execute Python Script" component to perform text preprocessing such as stemming, stopword removal, and lemmatization.
```
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'\W', ' ', text.lower())
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and apply stemming/lemmatization
    tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def azureml_main(dataframe1 = None, dataframe2 = None):
    # Apply preprocessing to the dataset
    dataframe1['processed_review'] = dataframe1['review'].apply(preprocess_text)
    return dataframe1,

```

## Feature Extraction
Use the TF-IDF Vectorization technique to convert text data into numerical features.

```
from sklearn.feature_extraction.text import TfidfVectorizer

def azureml_main(dataframe1 = None, dataframe2 = None):
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000)
    
    # Fit and transform the processed reviews
    X = tfidf.fit_transform(dataframe1['processed_review']).toarray()
    
    # Create a DataFrame from the TF-IDF features
    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(X, columns=feature_names)
    
    # Concatenate the original DataFrame with the TF-IDF DataFrame
    dataframe1.reset_index(drop=True, inplace=True)
    tfidf_df.reset_index(drop=True, inplace=True)
    result_df = pd.concat([dataframe1, tfidf_df], axis=1)
    
    return result_df,
```

## Model Building
Build a PyTorch-based deep learning model for sentiment analysis.

```
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the SentimentClassifier model
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

def azureml_main(dataframe1 = None, dataframe2 = None):
    # Prepare data
    X = dataframe1.iloc[:, 2:].values  # TF-IDF features
    y = pd.get_dummies(dataframe1['sentiment'], drop_first=True).values.ravel()

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Initialize model, loss function, and optimizer
    input_dim = X_train.shape[1]
    model = SentimentClassifier(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Predict on the test set
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).round().numpy()

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    return pd.DataFrame({'Accuracy': [accuracy]}),
```

## Visualization
Include visualizations such as loss curves, accuracy trends, and sample predictions.

```
# Loss curve visualization (add this to your training loop)
losses = []

# Inside the training loop:
# ...
losses.append(loss.item())

# Plotting Loss Curve
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()

# Sample predictions visualization
sample_indices = np.random.choice(len(y_test), 10, replace=False)
for i in sample_indices:
    print(f'Review: {dataframe1.iloc[i]["review"][:100]}...')
    print(f'Actual Sentiment: {y_test[i]}, Predicted Sentiment: {predictions[i][0]}')
```
