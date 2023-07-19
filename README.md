# Emotion_Detection_Text

1. Importing the necessary libraries:
```python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
```
The code begins by importing the required libraries: `nltk` for natural language processing tasks, `SentimentIntensityAnalyzer` from `nltk.sentiment` for sentiment analysis, `sent_tokenize` from `nltk.tokenize` for sentence tokenization, and `matplotlib.pyplot` for data visualization.

2. Defining the function `detect_emotion`:
```python
def detect_emotion(text, return_probabilities=False, check_bad_words=False):
    # Function code goes here
```
This function takes three parameters:
- `text`: The input text on which emotion detection will be performed.
- `return_probabilities`: A boolean parameter indicating whether to return emotion probabilities or not.
- `check_bad_words`: A boolean parameter indicating whether to check for the presence of bad words or not.

3. Input validation:
```python
if text is None or text.strip() == "":
    return None
```
This code checks if the input text is None or contains only whitespace characters. If so, it returns None, indicating that the emotion detection cannot be performed on an empty input.

4. SentimentIntensityAnalyzer initialization and sentence tokenization:
```python
analyzer = SentimentIntensityAnalyzer()
sentences = sent_tokenize(text)
```
An instance of the `SentimentIntensityAnalyzer` class is created to perform sentiment analysis. The input text is then tokenized into individual sentences using the `sent_tokenize` function from NLTK.

5. Emotion and bad word tracking:
```python
emotions = {
    'Happy': 0,
    'Sad': 0,
    'Neutral': 0
}
bad_words = ['bad', 'terrible', 'awful']  # Add more bad words as needed
```
The `emotions` dictionary is initialized with three emotion categories: 'Happy', 'Sad', and 'Neutral'. The `bad_words` list contains some example bad words. You can customize this list by adding more bad words as needed.

6. Emotion detection and bad word checking:
```python
for sentence in sentences:
    sentiment_scores = analyzer.polarity_scores(sentence)
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.1:
        emotions['Happy'] += 1
    elif compound_score <= -0.1:
        emotions['Sad'] += 1
    else:
        emotions['Neutral'] += 1
    
    if check_bad_words:
        for word in bad_words:
            if word in sentence.lower():
                print("Warning: Bad word detected!")
```
The code iterates over each sentence in the input text. For each sentence, sentiment analysis is performed using the `polarity_scores` method of the `SentimentIntensityAnalyzer`. The compound score is extracted, and based on its value, the corresponding emotion category count is incremented in the `emotions` dictionary.

If the `check_bad_words` parameter is set to `True`, the code checks if any of the bad words appear in the sentence. If a bad word is detected, a warning message is printed. You can customize this part of the code to perform other actions as needed.

7. Total sentence count and emotion probabilities:
```python
total_sentences = len(sentences)

if return_probabilities:
    probabilities = {emotion: count / total_sentences for emotion, count in emotions.items()}
    return probabilities
```
If the `return_probabilities` parameter is `True`, the code calculates the total number of sentences in the input text. It then computes the probabilities of each emotion category by dividing the count of each emotion by the total number of sentences. Finally, the probabilities dictionary is returned.

8. Dominant emotion determination:
```python
dominant_emotion = max(emotions, key=emotions.get)
return dominant_emotion
```
If the `return_probabilities` parameter is `False`, the code determines the dominant emotion category by finding the emotion with the highest count in the `emotions` dictionary. The dominant emotion category is returned.

9. Plotting function `plot_emotion_probabilities`:
```python
def plot_emotion_probabilities(emotion_probabilities):
    emotions = list(emotion_probabilities.keys())
    probabilities = list(emotion_probabilities.values())

    plt.bar(emotions, probabilities)
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.title('Emotion Probabilities')
    plt.show()
```
This function takes the `emotion_probabilities` dictionary as input. It extracts the emotions and probabilities from the dictionary, then creates a bar chart using `plt.bar` from `matplotlib.pyplot`. The `plt.xlabel`, `plt.ylabel`, and `plt.title` functions are used to set the labels and title of the chart. Finally, `plt.show()` is called to display the chart.

10. Example usage:
```python
text_data = "I'm feeling excited and happy! This day couldn't get any better. However, the movie was terrible."
emotion_probabilities = detect_emotion(text_data, return_probabilities=True, check_bad_words=True)
print("Emotion Probabilities:")
for emotion, probability in emotion_probabilities.items():
    print(f"{emotion}: {probability}")

plot_emotion_probabilities(emotion_probabilities)
```
This example code demonstrates the usage of the functions. It defines an example text, performs emotion detection with the `detect_emotion` function, and passes the returned emotion probabilities to the `plot_emotion_probabilities` function for visualization. The emotion probabilities are also printed for reference.
