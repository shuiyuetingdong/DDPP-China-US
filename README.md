# DDPP-China-US



## Introduction

This repository is for the paper **Digital Diplomacy and Public Perception: A Social Media Study of China-U.S. Relations and Regional Variations**. It includes code for topic modeling and sentiment analysis, as well as 10% random sample with comments, topic labels, and sentiment labels.

The sentiment labels of the sample were labeled by GPT4o-mini, and the corresponding technical details are in our paper



## Getting Started

Run the following command to install the required environment.

``` shell
pip install -r requirements.txt
```



## Experimental Results

| Method                      | Country | Negative | Neutral | Positive | Censored | Accuracy |
|-----------------------------|---------|----------|---------|----------|----------|----------|
| 2* Manual labels             | China   | 40.1     | 48.7    | 11.2     | /        | /        |
|                             | U.S.    | 10.3     | 52.4    | 37.3     | /        | /        |
| 2* Qwen-Plus                 | China   | 34.7     | 53.2    | 11.6     | 1.5      | 70.3%    |
|                             | U.S.    | 15.9     | 62.6    | 18.9     | 2.6      | 63.9%    |
| 2* GPT-4o mini               | China   | 45.5     | 39.3    | 35.0     | 0       | 79.0%    |
|                             | U.S.    | 17.4     | 63.3    | 35.0     | 0       | 72.6%    |
| 2* Qwen-Plus (PO)            | China   | 50.1     | 38.9    | 11.0     | 0        | 71.5%    |
|                             | U.S.    | 9.3      | 69.3    | 21.2     | 0.2      | 70.7%    |
| 2* GPT-4o mini (PO)          | China   | 45.6     | 43.5    | 10.9     | 0       | **88.4%**    |
|                             | U.S.    | 11.7     | 47.6    | 40.7     | 0       | **85.9%**    |
| 2* GPT-4o mini (FT)          | China   | 40.3     | 53.8    | 5.9      | 0       | 76.3%    |
|                             | U.S.    | 9.1      | 67.3    | 23.6     | 0       | 75.0%    |
| 2* GPT-4o mini (FT + PO)     | China   | 42.5     | 51.5    | 6.0      | 0       | 78.3%    |
|                             | U.S.    | 9.2      | 66.4    | 24.4     | 0        | 73.9%    |
| 2* GPT-4o mini (dspy)     | China   | 32.0     | 58.7    | 9.3      | 0       | 62.3%    |
|                             | U.S.    | 19.1      | 58.3    | 22.6     | 0        | 61.0%    |